from __future__ import annotations

import torch

from hcu_sglang.layers.moe.deepgemm_int8 import (
    DeepGemmInt8Runner,
    prepare_weights,
    requested,
)


def _dispatch_requested():
    if not requested():
        return False
    from sglang.srt.runtime_context import get_flags

    return get_flags().moe.quantization in ("w8a8_int8", "slimquant_w4a8")


def _dispatch_topk_ids(topk_ids, topk_weights):
    return torch.where(topk_weights == 0, -1, topk_ids).to(torch.int64)


def around_create_runner(original, method, layer, config):
    if not requested():
        return original(method, layer, config)
    method.moe_runner_config = config
    method.runner = DeepGemmInt8Runner(config)
    method.fuse_routed_scaling_factor_in_topk = True


def around_prepare_weights(original, method, layer):
    if not isinstance(getattr(method, "runner", None), DeepGemmInt8Runner):
        return original(method, layer)
    prepare_weights(layer, 8)


def around_apply(original, method, layer, dispatch_output):
    if not isinstance(getattr(method, "runner", None), DeepGemmInt8Runner):
        return original(method, layer, dispatch_output)
    return method.runner.run(dispatch_output, layer)


def around_run_moe_core(original, layer, dispatch_output):
    if not isinstance(getattr(layer.quant_method, "runner", None), DeepGemmInt8Runner):
        return original(layer, dispatch_output)
    return layer.quant_method.apply(layer, dispatch_output)


def around_set_dtype(original, dispatcher):
    if not _dispatch_requested():
        return original(dispatcher)
    from sglang.srt.layers.moe.utils import DispatcherOutputDtype
    from sglang.srt.runtime_context import get_exec

    if get_exec().moe.deepep_dispatcher_output_dtype != "int8":
        raise ValueError("HCU INT8 DeepGEMM requires --deepep-dispatcher-output-dtype int8")
    dispatcher.deepep_output_dtype = DispatcherOutputDtype.INT8
    dispatcher.use_fp8 = dispatcher.use_mxfp4 = dispatcher.use_mxfp8 = False
    dispatcher.use_nvfp4 = False


def around_normal_dispatch_a(original, dispatcher, hidden_states, topk_output):
    if not _dispatch_requested():
        return original(dispatcher, hidden_states, topk_output)
    from lightop.quant import per_token_quant_int8
    from sglang.srt.layers.moe.token_dispatcher.deepep import Buffer

    quantized = per_token_quant_int8(hidden_states)
    ids = _dispatch_topk_ids(topk_output.topk_ids, topk_output.topk_weights)
    event = Buffer.capture() if dispatcher.async_finish else None
    return quantized, ids, topk_output.topk_weights, event


def around_normal_dispatch_core(
    original, dispatcher, x, topk_ids, topk_weights, previous_event
):
    if not _dispatch_requested():
        return original(dispatcher, x, topk_ids, topk_weights, previous_event)
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPConfig,
        _deepep_precompile_tp_barrier,
        get_global_expert_distribution_recorder,
    )

    buffer = dispatcher._get_buffer()
    ranks, rdma_ranks, experts, token_in_rank, previous_event = buffer.get_dispatch_layout(
        topk_ids, dispatcher.num_experts, previous_event=previous_event,
        async_finish=dispatcher.async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )
    _deepep_precompile_tp_barrier()
    recv_x, recv_ids, recv_weights, counts, dispatcher.handle, event = buffer.dispatch(
        x, topk_idx=topk_ids, topk_weights=topk_weights,
        num_tokens_per_rank=ranks, num_tokens_per_rdma_rank=rdma_ranks,
        is_token_in_rank=token_in_rank, num_tokens_per_expert=experts,
        previous_event=previous_event, async_finish=dispatcher.async_finish,
        allocate_on_comm_stream=previous_event is not None and dispatcher.async_finish,
        expert_alignment=256,
        config=DeepEPConfig.get_instance().normal_dispatch_config,
    )
    get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
        counts, num_tokens_per_rank=ranks, num_tokens_per_rdma_rank=rdma_ranks,
        num_tokens_per_expert=experts,
    )
    return recv_x, recv_ids, recv_weights, counts, event


def around_low_latency_dispatch_a(original, dispatcher, hidden_states, topk_output):
    if not _dispatch_requested():
        return original(dispatcher, hidden_states, topk_output)
    topk_output = topk_output._replace(
        topk_ids=_dispatch_topk_ids(topk_output.topk_ids, topk_output.topk_weights)
    )
    return original(dispatcher, hidden_states, topk_output)


def around_low_latency_dispatch_core(
    original, dispatcher, hidden_states, topk_ids, topk_weights
):
    if not _dispatch_requested():
        return original(dispatcher, hidden_states, topk_ids, topk_weights)
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        _deepep_precompile_tp_barrier,
    )

    buffer = dispatcher._get_buffer()
    _deepep_precompile_tp_barrier()
    received, dispatcher.packed_recv_count, dispatcher.handle, event, hook = (
        buffer.low_latency_dispatch(
            hidden_states, topk_ids, topk_weights,
            dispatcher.num_max_dispatch_tokens_per_rank, dispatcher.num_experts,
            quant_type=1, fp8_round_scale=False,
            async_finish=not dispatcher.return_recv_hook,
            return_recv_hook=dispatcher.return_recv_hook,
        )
    )
    return received, dispatcher.packed_recv_count, event, hook


def around_normal_combine_a(original, dispatcher, hidden_states, topk_ids, topk_weights):
    if not _dispatch_requested():
        return original(dispatcher, hidden_states, topk_ids, topk_weights)
    from sglang.srt.layers.moe.token_dispatcher.deepep import Buffer

    event = Buffer.capture() if dispatcher.async_finish else None
    return hidden_states, event


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    quant = "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod"
    dispatch = "sglang.srt.layers.moe.token_dispatcher.deepep"
    hooks = (
        (f"{quant}.create_moe_runner", around_create_runner),
        (f"{quant}.process_weights_after_loading", around_prepare_weights),
        (f"{quant}.apply", around_apply),
        (
            "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core",
            around_run_moe_core,
        ),
        (
            f"{dispatch}._DeepEPDispatcherImplBase.set_deepep_dispatcher_dtype",
            around_set_dtype,
        ),
        (f"{dispatch}._DeepEPDispatcherImplNormal.dispatch_a", around_normal_dispatch_a),
        (
            f"{dispatch}._DeepEPDispatcherImplNormal._dispatch_core",
            around_normal_dispatch_core,
        ),
        (f"{dispatch}._DeepEPDispatcherImplNormal.combine_a", around_normal_combine_a),
        (
            f"{dispatch}._DeepEPDispatcherImplLowLatency.dispatch_a",
            around_low_latency_dispatch_a,
        ),
        (
            f"{dispatch}._DeepEPDispatcherImplLowLatency._dispatch_core",
            around_low_latency_dispatch_core,
        ),
    )
    for target, hook in hooks:
        HookRegistry.register(target, hook, HookType.AROUND)
