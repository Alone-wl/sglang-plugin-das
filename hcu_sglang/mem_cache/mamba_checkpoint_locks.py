from dataclasses import replace
from contextvars import ContextVar


_PREFILL_RESULT = ContextVar("hcu_completed_prefill_result", default=None)


def _release_mamba_lock(cache, req):
    if getattr(req, "session", None) is not None:
        return
    lock = req.lock
    if lock is None or lock.swa_released:
        return

    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    component = ComponentType.MAMBA
    if component in lock.receipt.skipped_lock_components:
        return
    cache.dec_lock_ref(
        lock.node,
        replace(
            lock.receipt,
            skipped_lock_components=tuple(
                item for item in cache.tree_components if item != component
            ),
        ),
    )
    lock.receipt.skipped_lock_components += (component,)


def _supports_lifecycle(cache):
    return (
        cache is not None
        and getattr(cache, "enable_mamba_extra_buffer", False)
        and getattr(cache.req_to_token_pool.mamba_pool, "_hcu_glm5_cache", False)
    )


def after_auxiliary_snapshot(snapshot, cls, batch, result):
    if not batch.forward_mode.is_context_parallel_extend():
        return snapshot
    cache = batch.tree_cache
    if not _supports_lifecycle(cache):
        return snapshot
    fence = result.copy_done
    if fence is None or not fence.query():
        return snapshot

    for req in batch.reqs:
        _release_mamba_lock(cache, req)
    return snapshot


def around_prefill_result(original, scheduler, batch, result):
    token = _PREFILL_RESULT.set((scheduler, batch, result))
    try:
        return original(scheduler, batch, result)
    finally:
        _PREFILL_RESULT.reset(token)


def after_prefill_checkpoint(output, scheduler, req):
    context = _PREFILL_RESULT.get()
    if context is None or context[0] is not scheduler:
        return output
    _, batch, result = context
    if not any(item is req for item in batch.reqs):
        return output
    cache = scheduler.tree_cache
    if not _supports_lifecycle(cache) or batch.tree_cache is not cache:
        return output
    if req.inflight_middle_chunks > 0 or req.extend_end < len(req.full_untruncated_fill_ids):
        return output
    fence = result.copy_done
    if fence is not None and fence.query():
        _release_mamba_lock(cache, req)
    return output


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    HookRegistry.register(
        "sglang.srt.managers.scheduler_components.batch_result_processor."
        "SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts",
        after_auxiliary_snapshot, HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
        "process_batch_result_disagg_prefill",
        around_prefill_result, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
        "checkpoint_disagg_prefill",
        after_prefill_checkpoint, HookType.AFTER,
    )
