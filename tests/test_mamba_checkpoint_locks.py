from dataclasses import dataclass
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.mem_cache import mamba_checkpoint_locks as lifecycle


@dataclass
class Receipt:
    skipped_lock_components: tuple = ()


class MambaCheckpointLockTests(unittest.TestCase):
    def fixture(self, *, ready=True, extend=True, tagged=True):
        cache = SimpleNamespace(
            enable_mamba_extra_buffer=True,
            req_to_token_pool=SimpleNamespace(mamba_pool=SimpleNamespace(_hcu_glm5_cache=tagged)),
            tree_components=(0, 2), dec_lock_ref=Mock())
        requests = [SimpleNamespace(session=None, lock=SimpleNamespace(
            node=7, receipt=Receipt(), swa_released=False)) for _ in range(2)]
        batch = SimpleNamespace(tree_cache=cache, reqs=requests,
                                forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: extend))
        result = SimpleNamespace(copy_done=SimpleNamespace(query=lambda: ready))
        return cache, batch, result

    def invoke(self, batch, result, snapshot=None):
        with patch.dict(sys.modules, {
            "sglang.srt.mem_cache.unified_cache.component_type":
                SimpleNamespace(ComponentType=SimpleNamespace(MAMBA=2)),
        }):
            return lifecycle.after_auxiliary_snapshot(snapshot, object, batch, result)

    def test_fenced_batch_releases_shared_source_once_and_keeps_kv_receipts(self):
        cache, batch, result = self.fixture()
        snapshot = [1, 2]
        self.assertIs(self.invoke(batch, result, snapshot), snapshot)
        self.assertEqual(cache.dec_lock_ref.call_count, 2)
        for call in cache.dec_lock_ref.call_args_list:
            self.assertEqual(call.args[0], 7)
            self.assertEqual(call.args[1].skipped_lock_components, (0,))
        for req in batch.reqs:
            self.assertEqual(req.lock.receipt.skipped_lock_components, (2,))
        self.invoke(batch, result)
        self.assertEqual(cache.dec_lock_ref.call_count, 2)

    def test_unfenced_decode_and_other_models_do_not_release(self):
        for options in ({"ready": False}, {"extend": False}, {"tagged": False}):
            cache, batch, result = self.fixture(**options)
            self.invoke(batch, result)
            cache.dec_lock_ref.assert_not_called()
        cache, batch, result = self.fixture()
        result.copy_done = None
        self.invoke(batch, result)
        cache.dec_lock_ref.assert_not_called()

    def test_session_and_already_released_locks_keep_their_owner(self):
        cache, batch, result = self.fixture()
        batch.reqs[0].session = object()
        batch.reqs[1].lock.swa_released = True
        self.invoke(batch, result)
        cache.dec_lock_ref.assert_not_called()

    def test_completed_prefill_releases_the_new_checkpoint_lock(self):
        cache, batch, result = self.fixture()
        req = batch.reqs[0]
        req.inflight_middle_chunks = 0
        req.extend_end = 8
        req.full_untruncated_fill_ids = list(range(8))
        scheduler = SimpleNamespace(tree_cache=cache)

        def process(actual_scheduler, actual_batch, actual_result):
            self.assertIs(actual_scheduler, scheduler)
            self.assertIs(actual_batch, batch)
            req.lock = SimpleNamespace(node=9, receipt=Receipt(), swa_released=False)
            return lifecycle.after_prefill_checkpoint("checkpoint", scheduler, req)

        with patch.dict(sys.modules, {
            "sglang.srt.mem_cache.unified_cache.component_type":
                SimpleNamespace(ComponentType=SimpleNamespace(MAMBA=2)),
        }):
            self.assertEqual(lifecycle.around_prefill_result(process, scheduler, batch, result),
                             "checkpoint")
        self.assertEqual(cache.dec_lock_ref.call_count, 1)
        self.assertEqual(cache.dec_lock_ref.call_args.args[0], 9)
        self.assertEqual(req.lock.receipt.skipped_lock_components, (2,))
        self.assertIsNone(lifecycle._PREFILL_RESULT.get())

    def test_prefill_checkpoint_requires_same_completed_request_and_fence(self):
        for change in ("pending", "chunk", "unfenced", "other_scheduler", "other_request"):
            cache, batch, result = self.fixture()
            req = batch.reqs[0]
            req.inflight_middle_chunks = int(change == "pending")
            req.extend_end = 7 if change == "chunk" else 8
            req.full_untruncated_fill_ids = list(range(8))
            scheduler = SimpleNamespace(tree_cache=cache)
            if change == "unfenced":
                result.copy_done = None
            actual_scheduler = SimpleNamespace(tree_cache=cache) if change == "other_scheduler" else scheduler
            actual_req = SimpleNamespace() if change == "other_request" else req
            lifecycle.around_prefill_result(
                lambda *args: lifecycle.after_prefill_checkpoint(None, actual_scheduler, actual_req),
                scheduler, batch, result)
            cache.dec_lock_ref.assert_not_called()

    def test_prefill_result_scope_resets_on_exception(self):
        with self.assertRaisesRegex(ValueError, "forward failure"):
            lifecycle.around_prefill_result(Mock(side_effect=ValueError("forward failure")),
                                            object(), object(), object())
        self.assertIsNone(lifecycle._PREFILL_RESULT.get())


if __name__ == "__main__":
    unittest.main()
