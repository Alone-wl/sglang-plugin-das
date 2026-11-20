import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.kernels.ops.layernorm.mhc import hc_pre as early_hc_pre
from sglang.kernels.ops.layernorm import mhc as native
from hcu_sglang import kernel_compat
from hcu_sglang.environ import register_envs
from hcu_sglang.kernels import mhc
from hcu_sglang.layers.communicator_glm5_next_mhc import MHCLayerCommunicator
from sglang.srt.plugins.hook_registry import HookRegistry

register_envs()
HookRegistry.reset()
kernel_compat.register_hooks()
HookRegistry.apply_hooks()


class MHCOutputNormTests(unittest.TestCase):
    def test_registered_dispatch_and_communicator_normalize_once(self):
        torch.manual_seed(81)
        for tokens in (1, 6, 32):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 16384, device='cuda', dtype=torch.bfloat16) * .01
                fn = torch.randn(24, 16384, device='cuda') * .005
                scale = torch.ones(3, device='cuda')
                base = torch.zeros(24, device='cuda')
                weight = torch.linspace(.05, .15, 4096, device='cuda', dtype=torch.bfloat16)
                norm = torch.nn.RMSNorm(4096, eps=1e-6, device='cuda', dtype=torch.bfloat16)
                norm.weight.data.copy_(weight)
                norm.variance_epsilon = 1e-6
                owner = object.__new__(MHCLayerCommunicator)
                owner.is_first_layer = False
                owner.input_layernorm = norm
                owner.qkv_latent_func = None
                owner.hc_attn_pre = lambda value, norm_weight, norm_eps: early_hc_pre(
                    value, fn, scale, base, 4, 1e-6, 1e-6, 20,
                    out_norm_weight=norm_weight, out_norm_eps=norm_eps,
                )
                def run():
                    return owner.prepare_attn(x, None, SimpleNamespace())[0]
                def check(actual):
                    _, _, raw = native._mhc_pre_torch(x.view(tokens, 4, 4096), fn, scale, base, 1e-6, 1e-6, 1e-6, 2., 20)
                    expected = F.rms_norm(raw.float(), (4096,), weight.float(), 1e-6).to(torch.bfloat16)
                    print('tokens', tokens, 'reference_norm', expected.float().norm().item(), 'actual_norm', actual.float().norm().item(), flush=True)
                    torch.testing.assert_close(actual, expected, rtol=.03, atol=.003)
                with patch.object(mhc, 'get_parallel', return_value=SimpleNamespace(tp_group=None)), patch.object(mhc, 'is_allocation_symmetric', return_value=False):
                    check(run())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = run()
                    for multiplier in (.5, 1.7):
                        x.mul_(multiplier)
                        graph.replay()
                        check(result)


if __name__ == '__main__':
    unittest.main()
