from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.mem_cache import cp_hicache_write_ack as lifecycle


class CpHiCacheWriteAckTests(unittest.TestCase):
    def fixture(self):
        pool = SimpleNamespace(_hcu_glm5_cache=True, layer_shard_enabled=True,
                               layer_shard_size=8)
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(write_policy="write_through", ack_write_queue=object()),
            pp_size=1, enable_storage=False, host_memory_mode="cache",
            attn_cp_group="cp", attn_tp_group="tp",
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
            _count_ready_acks=Mock(return_value=7))
        return cache, pool

    def invoke(self, original, cache, **kwargs):
        dist = SimpleNamespace(get_world_size=lambda group: {"cp": 8, "tp2": 2}.get(group, 1))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(distributed=dist),
                                      "torch.distributed": dist}):
            return lifecycle.around_writing_check(original, cache, **kwargs)

    def test_completed_local_writes_are_reaped_despite_zero_global_count(self):
        cache, _ = self.fixture()
        original = Mock(return_value="original-result")
        self.assertEqual(self.invoke(original, cache, finish_count=0), "original-result")
        original.assert_called_once_with(cache, write_back=False, finish_count=7)
        cache._count_ready_acks.assert_called_once_with(cache.cache_controller.ack_write_queue)

    def test_empty_queue_and_incomplete_events_are_not_fabricated(self):
        cache, _ = self.fixture()
        cache._count_ready_acks.return_value = 0
        original = Mock()
        self.invoke(original, cache, finish_count=7)
        original.assert_called_once_with(cache, write_back=False, finish_count=0)

    def test_other_cache_policies_and_topologies_keep_existing_path(self):
        for field, value in [("pp_size", 2), ("enable_storage", True),
                             ("host_memory_mode", "buffer_only"), ("attn_cp_group", None),
                             ("attn_tp_group", "tp2")]:
            cache, _ = self.fixture()
            setattr(cache, field, value)
            original = Mock()
            self.invoke(original, cache, finish_count=3)
            original.assert_called_once_with(cache, write_back=False, finish_count=3)
            cache._count_ready_acks.assert_not_called()
        for field, value in [("_hcu_glm5_cache", False), ("layer_shard_enabled", False),
                             ("layer_shard_size", 4)]:
            cache, pool = self.fixture()
            setattr(pool, field, value)
            original = Mock()
            self.invoke(original, cache, finish_count=3)
            original.assert_called_once_with(cache, write_back=False, finish_count=3)
            cache._count_ready_acks.assert_not_called()
        cache, _ = self.fixture()
        cache.cache_controller.write_policy = "write_back"
        original = Mock()
        self.invoke(original, cache, write_back=True, finish_count=None)
        original.assert_called_once_with(cache, write_back=True, finish_count=None)
        cache._count_ready_acks.assert_not_called()

    @unittest.skipUnless(
        os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1"
        and int(os.environ.get("WORLD_SIZE", "1")) > 1,
        "requires an isolated torchrun with HCU_PLUGIN_RUN_GPU_TESTS=1",
    )
    def test_upstream_write_locks_with_unequal_cp_ack_queues(self):
        import importlib.util
        from array import array
        from datetime import timedelta
        from pathlib import Path

        import sglang
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo", timeout=timedelta(seconds=90))
        try:
            rank, size = dist.get_rank(), dist.get_world_size()
            fixture = (Path(sglang.__file__).resolve().parents[2]
                       / "test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py")
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(fixture.parent))
                spec = importlib.util.spec_from_file_location("cp_write_upstream_fixture", fixture)
                upstream = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = upstream
                spec.loader.exec_module(upstream)
            finally:
                sys.path[:] = old_path

            from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
            from sglang.srt.plugins.hook_registry import HookRegistry

            cfg = upstream.CacheConfig(
                page_size=256, kv_size=2048, max_context_len=2048,
                components=(ComponentType.FULL, ComponentType.MAMBA),
                num_layers=2, full_attention_layer_ids=(0,), mamba_cache_size=4,
                enable_mamba_extra_buffer=True)
            cache, allocator, pool = upstream.build_fixture(cfg)
            snapshots, acks = [], []
            writer_count = min(3, size - 1)
            for index in range(4):
                tokens = array("q", range(index * 1000, index * 1000 + 256))
                slot = pool.mamba_allocator.alloc(1)
                pool.mamba_pool.mamba_cache.temporal[:, slot] = index + 1
                cache.insert(InsertParams(key=RadixKey(tokens), value=allocator.alloc(256),
                                          mamba_value=slot))
                node = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens))).best_match_node
                if rank < writer_count:
                    receipt = cache.inc_lock_ref(node).to_dec_params()
                    cache._track_write_through_node(node, receipt, publish_node_ids=[node])
                    src = pool.mamba_pool.mamba_cache.temporal[:, slot]
                    host = torch.empty_like(src, device="cpu", pin_memory=True)
                    host.copy_(src, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record()
                    event.synchronize()
                    snapshots.append((host, index + 1))
                    acks.append(SimpleNamespace(finish_event=event, node_ids=[node]))
            cache.cache_controller = SimpleNamespace(
                ack_write_queue=acks, ack_load_queue=[], write_policy="write_through")
            cache.attn_cp_group, cache.attn_tp_group = dist.group.WORLD, None
            finished = cache._sync_hicache_ready_counts()[0]
            self.assertEqual(finished, 0)
            cache.writing_check(finish_count=finished)
            failed = False
            try:
                cache.components[ComponentType.MAMBA]._alloc_mamba_slot()
            except AssertionError as error:
                self.assertEqual(str(error), "Can not alloc mamba cache")
                failed = True
            verdicts = [None] * size
            dist.all_gather_object(verdicts, failed)
            self.assertEqual(verdicts, [True] * writer_count + [False] * (size - writer_count))

            device_pool = allocator.get_kvcache()
            device_pool._hcu_glm5_cache = True
            device_pool.layer_shard_enabled = True
            device_pool.layer_shard_size = size
            lifecycle.register_hooks()
            HookRegistry.apply_hooks()
            cache.writing_check(finish_count=finished)
            self.assertFalse(cache.ongoing_write_through)
            self.assertFalse(cache.cache_controller.ack_write_queue)
            for host, value in snapshots:
                torch.testing.assert_close(host, torch.full_like(host, value), rtol=0, atol=0)
            self.assertIsNotNone(cache.components[ComponentType.MAMBA]._alloc_mamba_slot())
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
