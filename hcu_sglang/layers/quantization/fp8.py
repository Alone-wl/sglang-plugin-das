"""DCU FP8 quantization adaptations."""

from __future__ import annotations

import torch


def around_fallback_scaled_mm(
    original,
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    from hcu_sglang.utils.detection import is_dcu

    if not is_dcu() or qinput.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        return original(
            qinput,
            weight,
            x_scale,
            weight_scale,
            input_2d_shape,
            output_shape,
            bias,
            input_dtype,
        )

    try:
        from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm
    except ImportError:
        return original(
            qinput,
            weight,
            x_scale,
            weight_scale,
            input_2d_shape,
            output_shape,
            bias,
            input_dtype,
        )

    _, output = hipblaslt_w8a8_channelwise_gemm(
        qinput.unsqueeze(0).contiguous(),
        weight.T.unsqueeze(0).contiguous(),
        x_scale.reshape(1, -1, 1).contiguous(),
        weight_scale.reshape(1, -1, 1).contiguous(),
        qinput.shape[0],
        weight.shape[1],
        weight.shape[0],
        "NT",
        input_dtype,
        bias,
    )
    if output.dim() == 3:
        output = output.squeeze(0)
    return output.narrow(0, 0, input_2d_shape[0]).view(*output_shape)


def around_rms_norm_forward_hip(
    original, self, x, residual=None, post_residual_addition=None, quant_linear=None
):
    from hcu_sglang.utils.detection import is_dcu

    if is_dcu():
        from hcu_sglang.layers.norm_current import rms_forward_hip
        return rms_forward_hip(self, x, residual, post_residual_addition, quant_linear)
    return original(self, x, residual, post_residual_addition, quant_linear)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils._apply_fallback_scaled_mm",
        around_fallback_scaled_mm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.layernorm.RMSNorm.forward_hip",
        around_rms_norm_forward_hip,
        HookType.AROUND,
    )

    from hcu_sglang.utils.detection import is_dcu
    if is_dcu():
        from hcu_sglang.layers.norm_current import register_hooks as register_norm_hooks
        register_norm_hooks()
