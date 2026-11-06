from dataclasses import replace


def after_auxiliary_snapshot(snapshot, cls, batch, result):
    if not batch.forward_mode.is_context_parallel_extend():
        return snapshot
    cache = batch.tree_cache
    if cache is None or not getattr(cache, "enable_mamba_extra_buffer", False):
        return snapshot
    pool = cache.req_to_token_pool.mamba_pool
    if not getattr(pool, "_hcu_glm5_cache", False):
        return snapshot
    fence = result.copy_done
    if fence is None or not fence.query():
        return snapshot

    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    component = ComponentType.MAMBA
    for req in batch.reqs:
        if getattr(req, "session", None) is not None:
            continue
        lock = req.lock
        if (
            lock is None or lock.swa_released
            or component in lock.receipt.skipped_lock_components
        ):
            continue
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
    return snapshot


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
