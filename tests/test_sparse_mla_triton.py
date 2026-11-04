import unittest


class SparseMLATritonTest(unittest.TestCase):
    def test_kpool_tail_and_sink_in_graph(self):
        try:
            import torch
            import triton
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.cuda.is_available() or not torch.version.hip:
            self.skipTest("Requires a HIP device")

        from hcu_sglang.layers.attention.current_dsa.sparse_mla_triton import sparse_mla

        torch.manual_seed(42)
        for heads, query_dim, packed in ((8, 576, False), (32, 576, False), (64, 576, False), (64, 512, False), (64, 512, True)):
            with self.subTest(heads=heads, query_dim=query_dim, packed=packed):
                q = torch.randn(3, heads, query_dim, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(53, 1, query_dim, device="cuda", dtype=torch.bfloat16)
                cache = kv
                if packed:
                    fp8 = kv.to(torch.float8_e4m3fn)
                    scales = torch.rand(53, 1, 4, device="cuda") + 0.5
                    cache = torch.cat((fp8.view(torch.uint8), scales.view(torch.uint8)), dim=-1)
                    kv = (fp8.float() * scales.repeat_interleave(128, dim=-1)).bfloat16()
                indices = torch.randint(53, (3, 35), device="cuda", dtype=torch.int32)
                indices[0, 4:32] = -1
                indices[1] = -1
                sink = torch.randn(heads, device="cuda")
                scale = query_dim**-0.5

                def check(result):
                    selected = kv[indices.clamp_min(0).long(), 0].float()
                    logits = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
                    logits.masked_fill_(indices[:, None] < 0, -torch.inf)
                    logits = torch.cat((logits, sink.expand(3, heads)[..., None]), -1)
                    probs = torch.softmax(logits, -1)[..., :-1]
                    ref = torch.einsum("thk,tkd->thd", probs, selected[..., :512])
                    torch.testing.assert_close(result.float(), ref, atol=0.012, rtol=0.035)

                check(sparse_mla(q, cache, indices, scale, 512, None, sink, packed=packed))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = sparse_mla(q, cache, indices, scale, 512, None, sink, packed=packed)
                indices[0, -1] = (indices[0, -1] + 1) % kv.shape[0]
                graph.replay()
                check(result)
