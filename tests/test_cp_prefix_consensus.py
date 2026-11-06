from dataclasses import dataclass
from types import SimpleNamespace
import sys
import os
import unittest
from unittest.mock import Mock, patch
from typing import NamedTuple

from hcu_sglang.mem_cache import cp_prefix_consensus as consensus


@dataclass
class Key:
    size: int

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return Key(key.stop)

    def maybe_to_bigram_view(self, eagle):
        return self, None


@dataclass
class Params:
    key: Key
    cow_mamba: bool = True


class Result(NamedTuple):
    device_prefix_len: int
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    swa_branching_seqlen: object = None
    mamba_branching_seqlen: object = None
    cache_protected_len: object = None
    full_kv_hit_length: int = 0
    cache_actions: object = ()


class CpuTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class PrefixConsensusTests(unittest.TestCase):
    def modules(self):
        dist = SimpleNamespace(get_world_size=lambda **kw: 2, get_rank=lambda **kw: 0,
                               ReduceOp=SimpleNamespace(MIN="min"))
        torch = SimpleNamespace(distributed=dist, int64="int64",
                                tensor=lambda values, **kw: CpuTensor(values))
        return {"torch": torch, "torch.distributed": dist}

    def test_real_match_scope_retries_before_actions_and_cow_finalization(self):
        core = SimpleNamespace(is_eagle=False)
        events = []

        def reduce(tensor, op):
            self.assertEqual(op, "min")
            if tensor.values[0] == 32:
                tensor.values[0] = 0

        cache = SimpleNamespace(tree_core=core, attn_cp_group=object(), supports_mamba=lambda: True,
                                _all_reduce_attn_groups=reduce,
                                _apply_cache_actions=lambda actions: events.append(("actions", actions)))

        def core_match(actual_core, params):
            self.assertIs(actual_core, core)
            events.append(("lookup", len(params.key)))
            length = len(params.key)
            return Result(length, full_kv_hit_length=length,
                          cache_actions=("split",) if length else ("final",))

        def cache_match(actual_cache, params):
            result = consensus.around_core_match(core_match, core, params)
            actual_cache._apply_cache_actions(result.cache_actions)
            events.append(("cow", result.device_prefix_len))
            return result

        with patch.dict(sys.modules, self.modules()):
            result = consensus.around_match_tree(cache_match, cache, Params(Key(32)))
        self.assertEqual(result.device_prefix_len, 0)
        self.assertEqual(events, [("lookup", 32), ("actions", ("split",)),
                                  ("lookup", 0), ("actions", ("final",)), ("cow", 0)])
        self.assertIsNone(consensus._MATCH_CACHE.get())

    def test_uniform_prefix_remains_a_hit(self):
        core = SimpleNamespace()
        cache = SimpleNamespace(tree_core=core, attn_cp_group=object(), supports_mamba=lambda: True,
                                _all_reduce_attn_groups=lambda tensor, op: None)
        original = Mock(return_value=Result(32, full_kv_hit_length=32))
        with patch.dict(sys.modules, self.modules()):
            result = consensus.around_match_tree(
                lambda c, p: consensus.around_core_match(original, core, p), cache, Params(Key(32)))
        self.assertEqual(result.device_prefix_len, 32)
        original.assert_called_once()

    def test_scope_resets_on_failure_and_unrelated_core_is_unchanged(self):
        cache = SimpleNamespace(tree_core=object(), attn_cp_group=object(), supports_mamba=lambda: True)
        with patch.dict(sys.modules, self.modules()):
            with self.assertRaisesRegex(ValueError, "original failure"):
                consensus.around_match_tree(Mock(side_effect=ValueError("original failure")),
                                            cache, Params(Key(32)))
        self.assertIsNone(consensus._MATCH_CACHE.get())
        original = Mock(return_value="unmodified")
        self.assertEqual(consensus.around_core_match(original, object(), Params(Key(32))), "unmodified")

    def test_non_cow_and_non_cp_lookups_keep_original(self):
        for cow, group, mamba in [(False, object(), True), (True, None, True),
                                  (True, object(), False)]:
            cache = SimpleNamespace(attn_cp_group=group, supports_mamba=lambda: mamba)
            original = Mock(return_value="unmodified")
            self.assertEqual(consensus.around_match_tree(original, cache, Params(Key(32), cow)),
                             "unmodified")
            original.assert_called_once()

    @unittest.skipUnless(
        os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1"
        and int(os.environ.get("WORLD_SIZE", "1")) > 1,
        "requires an isolated torchrun with HCU_PLUGIN_RUN_GPU_TESTS=1",
    )
    def test_upstream_cache_gpu_pools_and_distributed_cow(self):
        import importlib.util
        from array import array
        from datetime import timedelta
        from pathlib import Path

        import sglang
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
        try:
            rank, size = dist.get_rank(), dist.get_world_size()
            fixture = (Path(sglang.__file__).resolve().parents[2]
                       / "test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py")
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(fixture.parent))
                spec = importlib.util.spec_from_file_location("cp_upstream_cache_fixture", fixture)
                upstream = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = upstream
                spec.loader.exec_module(upstream)
            finally:
                sys.path[:] = old_path

            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
            from sglang.srt.plugins.hook_registry import HookRegistry
            from sglang.srt.sampling.sampling_params import SamplingParams
            from hcu_sglang.mem_cache import mamba_checkpoint_locks

            consensus.register_hooks()
            mamba_checkpoint_locks.register_hooks()
            HookRegistry.apply_hooks()
            cfg = upstream.CacheConfig(
                page_size=256, kv_size=2048, max_context_len=2048,
                components=(ComponentType.FULL, ComponentType.MAMBA),
                num_layers=2, full_attention_layer_ids=(0,), mamba_cache_size=20,
                enable_mamba_extra_buffer=True)
            tokens = list(range(512))

            def request(name):
                return Req(rid=name, origin_input_text="", origin_input_ids=array("q", tokens),
                           sampling_params=SamplingParams(temperature=0, max_new_tokens=1))

            for divergent in (True, False):
                cache, allocator, pool = upstream.build_fixture(cfg)
                owner = request("insert")
                pool.alloc([owner])
                cache.insert(InsertParams(key=RadixKey(array("q", tokens)),
                                          value=allocator.alloc(512),
                                          mamba_value=owner.kv.mamba_pool_idx.unsqueeze(0)))
                if divergent and rank == 0:
                    cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                raw = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
                raw_hits = [None] * size
                dist.all_gather_object(raw_hits, raw.device_prefix_len)
                self.assertEqual(raw_hits, ([0] + [512] * (size - 1)) if divergent else [512] * size)
                cache.attn_cp_group = dist.group.WORLD
                cache.attn_tp_group = None
                query = request("query")
                before = pool.mamba_allocator.free_slots.numel()
                result = cache.match_prefix(MatchPrefixParams(
                    key=RadixKey(array("q", tokens)), cow_mamba=True, req=query))
                hits = [None] * size
                dist.all_gather_object(hits, result.device_prefix_len)
                self.assertEqual(hits, [0 if divergent else 512] * size)
                self.assertEqual(pool.mamba_allocator.free_slots.numel(), before - int(not divergent))
                if divergent:
                    self.assertIsNone(query.kv.mamba_cow_src_index)
                else:
                    self.assertEqual(int(query.kv.mamba_cow_src_index.item()),
                                     int(owner.kv.mamba_pool_idx.item()))

            from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            query.lock = cache.lock(result.best_match_node)
            second = request("second-cow")
            second_hit = cache.match_prefix(MatchPrefixParams(
                key=RadixKey(array("q", tokens)), cow_mamba=True, req=second))
            second.lock = cache.lock(second_hit.best_match_node)
            pool.mamba_pool._hcu_glm5_cache = True
            state = pool.mamba_pool.mamba_cache
            source = owner.kv.mamba_pool_idx.view(1)
            state.temporal[:, source] = 1
            for conv in state.conv:
                conv[:, source] = 2
            for req in (query, second):
                pool.mamba_pool.copy_from(
                    pool.translate_mamba_indices(req.kv.mamba_cow_src_index.view(1)),
                    pool.translate_mamba_indices(req.kv.mamba_pool_idx.view(1)))
            pool.mamba_allocator.alloc(pool.mamba_allocator.free_slots.numel())
            fence = torch.cuda.Event()
            fence.record()
            fence.synchronize()
            node = query.lock.node
            self.assertEqual(upstream._device_lock_ref(cache, node, ComponentType.MAMBA), 2)
            SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts(
                SimpleNamespace(tree_cache=cache, reqs=[query, second], forward_mode=ForwardMode.EXTEND),
                SimpleNamespace(copy_done=fence, auxiliary_host_output=None))
            slot = cache.components[ComponentType.MAMBA]._alloc_mamba_slot()
            self.assertEqual(upstream._device_lock_ref(cache, node, ComponentType.FULL), 2)
            self.assertEqual(upstream._device_lock_ref(cache, node, ComponentType.MAMBA), 0)
            state.temporal[:, slot] = 99
            for conv in state.conv:
                conv[:, slot] = 99
            for req in (query, second):
                index = req.kv.mamba_pool_idx
                torch.testing.assert_close(state.temporal[:, index], torch.ones_like(state.temporal[:, index]),
                                           rtol=0, atol=0)
                for conv in state.conv:
                    torch.testing.assert_close(conv[:, index], torch.full_like(conv[:, index], 2),
                                               rtol=0, atol=0)
                cache.unlock(req.lock)

            from dataclasses import replace
            from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
            from sglang.srt.mem_cache.common import release_kv_cache
            from sglang.srt.managers.scheduler import Scheduler

            self.assertIs(Scheduler.checkpoint_disagg_prefill,
                          SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill)
            cache, allocator, pool = upstream.build_fixture(replace(cfg, mamba_cache_size=11))
            pool.mamba_pool._hcu_glm5_cache = True
            scheduler = SimpleNamespace(tree_cache=cache)
            inflight = []
            for index in range(3):
                tokens = list(range(index * 1000, index * 1000 + 256))
                req = request("inflight-" + str(index))
                pool.alloc([req])
                inflight.append(req)
                req.full_untruncated_fill_ids = array("q", tokens)
                req.extend_end = 256
                req.kv.kv_committed_len = 256
                req.kv.kv_allocated_len = 256
                req.last_node = cache.root_node_handle()
                req.lock = None
                req.kv.cache_protected_len = 0
                req.kv.mamba_last_track_seqlen = 256
                pool.write((req.kv.req_pool_idx, slice(0, 256)), allocator.alloc(256))
                pool.mamba_pool.mamba_cache.temporal[:, req.kv.mamba_pool_idx] = index + 1
                fence = torch.cuda.Event()
                fence.record()
                fence.synchronize()
                mamba_checkpoint_locks.around_prefill_result(
                    lambda actual_scheduler, batch, result:
                        SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(actual_scheduler, req),
                    scheduler, SimpleNamespace(tree_cache=cache, reqs=[req]),
                    SimpleNamespace(copy_done=fence))
                self.assertEqual(upstream._device_lock_ref(cache, req.lock.node, ComponentType.FULL), 1)
                self.assertEqual(upstream._device_lock_ref(cache, req.lock.node, ComponentType.MAMBA), 0)
            for index, req in enumerate(inflight):
                state = pool.mamba_pool.mamba_cache.temporal[:, req.kv.mamba_pool_idx]
                torch.testing.assert_close(state, torch.full_like(state, index + 1), rtol=0, atol=0)
                release_kv_cache(req, cache, checkpoint=False)
            cache.sanity_check()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
