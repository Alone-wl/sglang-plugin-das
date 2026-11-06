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


if __name__ == "__main__":
    unittest.main()
