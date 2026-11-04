from contextvars import ContextVar


_BUILDING_GLM5 = ContextVar("hcu_building_glm5", default=False)


def use_upstream_method(instance):
    return _BUILDING_GLM5.get() or getattr(instance, "_hcu_glm5_upstream", False)


def around_model_init(original, model, *args, **kwargs):
    model._hcu_glm5_upstream = True
    token = _BUILDING_GLM5.set(True)
    try:
        return original(model, *args, **kwargs)
    finally:
        _BUILDING_GLM5.reset(token)


def around_mlp_init(original, layer, *args, **kwargs):
    if _BUILDING_GLM5.get():
        layer._hcu_glm5_upstream = True
    return original(layer, *args, **kwargs)


def around_mlp_forward(
    original, layer, x, forward_batch=None,
    gemm_output_zero_allocator=None, gateup_pre_quant=None,
):
    from hcu_sglang.layers import row_fusion
    from hcu_sglang.utils.detection import is_hcu

    selected = (
        is_hcu()
        and row_fusion._use_fused_silu_mul_quant
        and getattr(layer, "_hcu_glm5_upstream", False)
        and layer.is_replicated
        and layer.swiglu_limit is not None
        and type(layer.down_proj.quant_method).__name__ in {
            "W8A8Int8LinearMethod", "SlimQuantW4A8Int8LinearMethod",
        }
    )
    if not selected:
        return original(
            layer, x, forward_batch=forward_batch,
            gemm_output_zero_allocator=gemm_output_zero_allocator,
            gateup_pre_quant=gateup_pre_quant,
        )
    if not layer.down_proj.supports_fused_silu_mul_quant_input():
        raise RuntimeError("GLM5 LightOp clamp quantization interface is unavailable")
    if gateup_pre_quant is not None:
        raise ValueError("GLM5 W8A8 fused activation does not accept FP8 gate-up input")
    if x.shape[0] == 0:
        return x
    gate_up, _ = layer.gate_up_proj(x)
    output, _ = layer.down_proj(
        gate_up, use_fused_silu_mul_quant=True,
        swiglu_limit=float(layer.swiglu_limit),
    )
    return output


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target in (
        "sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration.__init__",
        "sglang.srt.models.glm5_next_nextn.Glm5NextForConditionalGenerationNextN.__init__",
    ):
        HookRegistry.register(target, around_model_init, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.models.deepseek_v2.DeepseekV2MLP.__init__",
        around_mlp_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.deepseek_v2.DeepseekV2MLP.forward",
        around_mlp_forward,
        HookType.AROUND,
    )
