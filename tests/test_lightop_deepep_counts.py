from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class DeviceTensor:
    def __init__(self, shape, dtype="int8", device="cuda:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def cpu(self):
        raise AssertionError("expert counts must not synchronize back to CPU")

    def numel(self):
        import math

        return math.prod(self.shape)

    def bfloat16(self):
        return DeviceTensor(self.shape, "bfloat16", self.device)


class LightopDeepEPCountsTests(unittest.TestCase):
    def run_normal(self, counts, fp8=False):
        torch = SimpleNamespace(
            int32="int32", float32="float32", bfloat16="bfloat16",
            tensor=Mock(side_effect=lambda values, **kw: DeviceTensor((len(values),), **kw)),
            empty=Mock(side_effect=lambda shape, **kw: DeviceTensor(shape, **kw)),
            full=Mock(side_effect=lambda shape, value, **kw: DeviceTensor(shape, **kw)),
            zeros_like=Mock(side_effect=lambda x, **kw: DeviceTensor(x.shape, **kw)),
        )
        moe = SimpleNamespace(ep_scatter=Mock(), ep_gather=Mock())
        activation = SimpleNamespace(
            fuse_silu_mul_quant=Mock(return_value=(DeviceTensor((512, 8)), DeviceTensor((512, 1)))),
            fuse_silu_mul_fp8_quant=Mock(return_value=(DeviceTensor((512, 8)), DeviceTensor((512, 1)))),
        )
        lightop = SimpleNamespace(moe=moe, m_grouped_w8a8_gemm_nt_contig_asm=Mock())
        deepgemm = SimpleNamespace(m_grouped_fp8_gemm_nt_contiguous=Mock())
        combine = SimpleNamespace(DeepEPNormalCombineInput=SimpleNamespace, DeepEPLLCombineInput=SimpleNamespace)
        modules = {
            "torch": torch, "lightop": lightop, "lightop.activation": activation,
            "deepgemm": deepgemm,
            "hcu_sglang.compat.aiter": SimpleNamespace(install_aiter_compat=lambda: None),
            "sglang.srt.layers.moe.token_dispatcher.deepep": combine,
        }
        dispatch = SimpleNamespace(
            format=SimpleNamespace(value="deepep_normal"),
            num_recv_tokens_per_expert=counts,
            hidden_states=DeviceTensor((4, 8)), hidden_states_scale=DeviceTensor((4, 1)),
            topk_ids=DeviceTensor((4, 2)), topk_weights=DeviceTensor((4, 2)),
        )
        quant = SimpleNamespace(
            deepep_w13_weight=DeviceTensor((2, 16, 8)),
            deepep_w2_weight=DeviceTensor((2, 8, 8)),
            w13_scale=DeviceTensor((2, 16)), w2_scale=DeviceTensor((2, 8)),
            use_fp8_w8a8=fp8, use_int8_w8a8=not fp8,
        )
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "counts_runtime", ROOT / "hcu_sglang/layers/moe/lightop_deepep.py"
            )
            runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)
            try:
                result = runtime.fused_deepep_lightop(dispatch, quant, None)
            except RuntimeError:
                torch.tensor.assert_not_called()
                moe.ep_scatter.assert_not_called()
                raise
        return result, dispatch, torch, moe, lightop, deepgemm

    def test_normal_quantized_paths_preserve_native_counts_without_readback(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                result, dispatch, torch, moe, lightop, deepgemm = self.run_normal([0, 512], fp8)
                torch.tensor.assert_called_once_with([0, 512], dtype="int32", device="cuda:0")
                self.assertIs(result.topk_ids, dispatch.topk_ids)
                self.assertIs(result.topk_weights, dispatch.topk_weights)
                self.assertEqual(result.hidden_states.shape, (4, 8))
                self.assertEqual(moe.ep_scatter.call_args.args[-2:], (2, 256))
                self.assertEqual(moe.ep_scatter.call_args.args[4].shape, (2,))
                self.assertIs(moe.ep_gather.call_args.args[-1], result.hidden_states)
                gemm = deepgemm.m_grouped_fp8_gemm_nt_contiguous if fp8 else lightop.m_grouped_w8a8_gemm_nt_contig_asm
                self.assertEqual(gemm.call_count, 2)

    def test_unaligned_counts_fail_before_device_copy(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8), self.assertRaisesRegex(RuntimeError, "256-aligned"):
                self.run_normal([1, 511], fp8)

    def test_empty_dispatch_still_bypasses_expert_kernels(self):
        for counts in (None, [], [0, 0]):
            with self.subTest(counts=counts):
                result, _, torch, moe, _, _ = self.run_normal(counts)
                torch.tensor.assert_not_called()
                moe.ep_scatter.assert_not_called()
                self.assertEqual(result.hidden_states.dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
