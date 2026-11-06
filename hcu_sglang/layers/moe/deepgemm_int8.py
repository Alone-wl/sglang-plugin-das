from __future__ import annotations

import torch
from torch.nn import Parameter

from hcu_sglang.utils.detection import is_hcu


def requested():
    if not is_hcu():
        return False
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend

    return (
        get_moe_runner_backend().is_deep_gemm()
        and get_moe_a2a_backend().is_deepep()
    )


def prepare_weights(layer, bits):
    import deepgemm

    if hasattr(layer, "_hcu_deepgemm_bits"):
        raise RuntimeError("HCU DeepGEMM weights have already been packed")
    if bits not in (4, 8):
        raise ValueError("HCU DeepGEMM requires INT4 or INT8 weights")
    pack = (
        deepgemm.pack_w4a8_moe_hipc_weight
        if bits == 4
        else deepgemm.pack_i8_masked_weight
    )
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale")
        if weight.dtype != torch.int8 or weight.ndim != 3:
            raise ValueError("HCU DeepGEMM requires INT8 storage in [E,N,K] order")
        if scale.shape != (*weight.shape[:2], 1):
            raise ValueError("HCU DeepGEMM requires per-channel [E,N,1] scales")
        setattr(
            layer, f"{prefix}_weight",
            Parameter(pack(weight.data), requires_grad=False),
        )
        setattr(
            layer, f"{prefix}_weight_scale",
            Parameter(scale.data.float().contiguous(), requires_grad=False),
        )
    layer._hcu_deepgemm_bits = bits


def _activate(x, config, masked_m=None, expected_m=-1):
    from lightop import fuse_silu_mul_clamp_quant
    from lightop.activation import fuse_silu_mul_quant, fuse_silu_mul_quant_ep

    if config.swiglu_limit is not None:
        return fuse_silu_mul_clamp_quant(
            x, float(config.swiglu_limit), mask_m=masked_m, expect_m=expected_m
        )
    if masked_m is not None:
        return fuse_silu_mul_quant_ep(x, masked_m)
    return fuse_silu_mul_quant(x)


class DeepGemmInt8Runner:
    def __init__(self, config):
        from sglang.srt.layers.moe import MoeRunnerBackend

        if config.params_dtype != torch.bfloat16:
            raise ValueError("HCU INT8 DeepGEMM requires BF16 model activations")
        if config.activation != "silu" or config.apply_router_weight_on_input:
            raise ValueError("HCU INT8 DeepGEMM requires SiLU and output router weights")
        self.config = config
        self.runner_backend = MoeRunnerBackend.DEEP_GEMM
        self.down_gemm_overlap_args = None
        self.meta_overlap_args = None

    def set_overlap_args(self, down_gemm_overlap_args, meta_overlap_args):
        if down_gemm_overlap_args is not None or meta_overlap_args not in (None, {}):
            raise NotImplementedError("HCU INT8 DeepGEMM overlap is not supported")

    def clear_overlap_args(self):
        self.down_gemm_overlap_args = self.meta_overlap_args = None

    def run(self, dispatch_output, layer):
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPLLDispatchOutput,
            DeepEPNormalCombineInput,
            DeepEPNormalDispatchOutput,
        )

        if dispatch_output.hidden_states.dtype != torch.int8:
            raise ValueError("HCU INT8 DeepGEMM requires INT8 DeepEP dispatch")
        if dispatch_output.hidden_states_scale is None:
            raise ValueError("INT8 DeepEP dispatch must include per-token scales")
        if isinstance(dispatch_output, DeepEPNormalDispatchOutput):
            output = self._normal(dispatch_output, layer)
            wrapper = DeepEPNormalCombineInput
        elif isinstance(dispatch_output, DeepEPLLDispatchOutput):
            output = self._masked(dispatch_output, layer)
            wrapper = DeepEPLLCombineInput
        else:
            raise ValueError("HCU INT8 DeepGEMM requires a DeepEP dispatch result")
        return wrapper(output, dispatch_output.topk_ids, dispatch_output.topk_weights)

    def _masked(self, dispatch, layer):
        import deepgemm

        x, scale = dispatch.hidden_states, dispatch.hidden_states_scale
        e, capacity, hidden = x.shape
        expected = min(capacity, dispatch.expected_m)
        op = (
            deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc
            if layer._hcu_deepgemm_bits == 4
            else deepgemm.m_grouped_i8_gemm_nt_masked
        )
        gate_up = torch.empty(
            (e, capacity, layer.w13_weight_scale.shape[1]),
            device=x.device, dtype=torch.bfloat16,
        )
        op((x, scale), (layer.w13_weight, layer.w13_weight_scale),
           gate_up, dispatch.masked_m, expected)
        activated = _activate(gate_up, self.config, dispatch.masked_m, expected)
        del gate_up
        output = torch.empty(
            (e, capacity, hidden), device=x.device, dtype=torch.bfloat16
        )
        op(activated, (layer.w2_weight, layer.w2_weight_scale),
           output, dispatch.masked_m, expected)
        return output

    def _normal(self, dispatch, layer):
        import deepgemm
        from lightop import moe as lightop_moe
        from sglang.srt.utils.offloader import get_offloader

        x, scale = dispatch.hidden_states, dispatch.hidden_states_scale
        counts = dispatch.num_recv_tokens_per_expert
        if counts is None or any(count % 256 for count in counts):
            raise ValueError("HCU DeepGEMM normal dispatch requires 256-aligned counts")
        total = sum(counts)
        output = torch.zeros(x.shape, device=x.device, dtype=torch.bfloat16)
        if total == 0:
            return output
        if scale.dtype != torch.float32 or scale.numel() != x.shape[0]:
            raise ValueError("HCU INT8 scatter requires per-token FP32 scales")
        ids = dispatch.topk_ids.contiguous()
        if ids.dtype != torch.int64 or ids.shape[1] > 16:
            raise ValueError("HCU INT8 scatter requires INT64 IDs and top-k <= 16")
        if get_offloader().forbid_copy_engine_usage:
            count_gpu = torch.tensor(counts, dtype=torch.int32, device=x.device)
        else:
            count_gpu = torch.tensor(
                counts, dtype=torch.int32, pin_memory=True, device="cpu"
            ).to(device=x.device, non_blocking=True)
        grouped_x = torch.zeros((total, x.shape[1]), device=x.device, dtype=x.dtype)
        grouped_scale = torch.zeros((total, 1), device=x.device, dtype=torch.float32)
        indices = torch.full((total,), -1, dtype=torch.int32, device=x.device)
        output_index = torch.full(ids.shape, -1, dtype=torch.int32, device=x.device)
        lightop_moe.ep_scatter(
            x, scale, ids, None, count_gpu, grouped_x, grouped_scale,
            indices, output_index, len(counts), 256,
        )
        op = (
            deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc
            if layer._hcu_deepgemm_bits == 4
            else deepgemm.m_grouped_i8_gemm_nt_contiguous
        )
        gate_up = torch.zeros(
            (total, layer.w13_weight_scale.shape[1]), device=x.device, dtype=torch.bfloat16
        )
        op((grouped_x, grouped_scale), (layer.w13_weight, layer.w13_weight_scale),
           gate_up, indices)
        activated = _activate(gate_up, self.config)
        del gate_up, grouped_x, grouped_scale
        down = torch.zeros((total, x.shape[1]), device=x.device, dtype=torch.bfloat16)
        op(activated, (layer.w2_weight, layer.w2_weight_scale), down, indices)
        lightop_moe.ep_gather(
            down, ids, dispatch.topk_weights.contiguous(), output_index, None, output
        )
        return output
