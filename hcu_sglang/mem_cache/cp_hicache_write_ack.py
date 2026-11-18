import logging


logger = logging.getLogger(__name__)


def around_writing_check(original, cache, write_back=False, finish_count=None):
    controller = cache.cache_controller
    if (
        write_back
        or controller is None
        or controller.write_policy != "write_through"
        or cache.pp_size != 1
        or cache.enable_storage
        or cache.host_memory_mode != "cache"
        or cache.attn_cp_group is None
    ):
        return original(cache, write_back=write_back, finish_count=finish_count)

    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    if not (
        getattr(pool, "_hcu_glm5_cache", False)
        and getattr(pool, "layer_shard_enabled", False)
    ):
        return original(cache, write_back=write_back, finish_count=finish_count)

    import torch.distributed as dist

    if (
        dist.get_world_size(group=cache.attn_cp_group) != pool.layer_shard_size
        or pool.layer_shard_size <= 1
        or (cache.attn_tp_group is not None
            and dist.get_world_size(group=cache.attn_tp_group) != 1)
    ):
        return original(cache, write_back=write_back, finish_count=finish_count)

    local_count = cache._count_ready_acks(controller.ack_write_queue)
    output = original(cache, write_back=False, finish_count=local_count)
    if local_count and local_count != finish_count and not getattr(cache, "_hcu_cp_ack_reported", False):
        logger.info("CP LayerSplit HiCache reaped %d local write ACKs; synchronized count was %s",
                    local_count, finish_count)
        cache._hcu_cp_ack_reported = True
    return output


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    HookRegistry.register(
        "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.writing_check",
        around_writing_check, HookType.AROUND,
    )
