from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, shape, dtype, device="cuda:3"):
        self.shape, self.dtype, self.device = shape, dtype, device

    def contiguous(self):
        return self

    def numel(self):
        return math.prod(self.shape)


class NormalDispatch:
    pass


class CombineInput:
    def __init__(self, output, ids, weights):
        self.output, self.ids, self.weights = output, ids, weights


class DeepGemmNormalCountsTests(unittest.TestCase):
    def test_public_runner_preserves_counts_device_and_legacy_offloader_branch(self):
        for forbid_copy_engine in (False, True):
            with self.subTest(forbid_copy_engine=forbid_copy_engine):
                gpu_counts = object()
                cpu_counts = SimpleNamespace(to=Mock(return_value=gpu_counts))
                tensor = Mock(return_value=gpu_counts if forbid_copy_engine else cpu_counts)
                torch = SimpleNamespace(
                    int8="int8", int32="int32", int64="int64",
                    float32="float32", bfloat16="bfloat16", tensor=tensor,
                    zeros=lambda shape, **kw: Tensor(shape, kw["dtype"], kw["device"]),
                    full=lambda shape, value, **kw: Tensor(shape, kw["dtype"], kw["device"]),
                )
                scatter, gather, gemm = Mock(), Mock(), Mock()
                modules = {
                    "torch": torch,
                    "torch.nn": SimpleNamespace(Parameter=object),
                    "hcu_sglang.utils.detection": SimpleNamespace(is_hcu=lambda: True),
                    "deepgemm": SimpleNamespace(m_grouped_i8_gemm_nt_contiguous=gemm),
                    "lightop": SimpleNamespace(moe=SimpleNamespace(ep_scatter=scatter, ep_gather=gather)),
                    "sglang.srt.layers.moe": SimpleNamespace(
                        MoeRunnerBackend=SimpleNamespace(DEEP_GEMM="deep_gemm")),
                    "sglang.srt.utils.offloader": SimpleNamespace(
                        get_offloader=lambda: SimpleNamespace(forbid_copy_engine_usage=forbid_copy_engine)),
                    "sglang.srt.layers.moe.token_dispatcher.deepep": SimpleNamespace(
                        DeepEPNormalDispatchOutput=NormalDispatch,
                        DeepEPLLDispatchOutput=type("LLDispatch", (), {}),
                        DeepEPNormalCombineInput=CombineInput,
                        DeepEPLLCombineInput=CombineInput),
                }
                with patch.dict(sys.modules, modules):
                    spec = importlib.util.spec_from_file_location(
                        "counts_runner_under_test", ROOT / "hcu_sglang/layers/moe/deepgemm_int8.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module._activate = Mock(return_value=(object(), object()))
                    config = SimpleNamespace(params_dtype="bfloat16", activation="silu",
                                             apply_router_weight_on_input=False)
                    runner = module.DeepGemmInt8Runner(config)
                    dispatch = NormalDispatch()
                    dispatch.hidden_states = Tensor((16, 4096), "int8")
                    dispatch.hidden_states_scale = Tensor((16, 1), "float32")
                    dispatch.num_recv_tokens_per_expert = [256, 0, 512, 256]
                    dispatch.topk_ids = Tensor((16, 2), "int64")
                    dispatch.topk_weights = Tensor((16, 2), "float32")
                    layer = SimpleNamespace(
                        _hcu_deepgemm_bits=8, w13_weight=object(), w2_weight=object(),
                        w13_weight_scale=Tensor((4, 4096, 1), "float32"),
                        w2_weight_scale=Tensor((4, 4096, 1), "float32"))
                    result = runner.run(dispatch, layer)
                self.assertIs(scatter.call_args.args[4], gpu_counts)
                self.assertEqual(gemm.call_count, 2)
                self.assertIs(gather.call_args.args[-1], result.output)
                self.assertIs(result.ids, dispatch.topk_ids)
                self.assertIs(result.weights, dispatch.topk_weights)
                if forbid_copy_engine:
                    tensor.assert_called_once_with(
                        dispatch.num_recv_tokens_per_expert, dtype="int32", device="cuda:3")
                    cpu_counts.to.assert_not_called()
                else:
                    tensor.assert_called_once_with(
                        dispatch.num_recv_tokens_per_expert, dtype="int32",
                        pin_memory=True, device="cpu")
                    cpu_counts.to.assert_called_once_with(device="cuda:3", non_blocking=True)


if __name__ == "__main__":
    unittest.main()
