from contextvars import ContextVar
from dataclasses import replace
import logging


_MATCH_CACHE = ContextVar("hcu_cp_match_cache", default=None)
_MATCH_FIELDS = (
    "device_prefix_len", "host_hit_length", "swa_host_hit_length",
    "mamba_host_hit_length", "swa_branching_seqlen", "mamba_branching_seqlen",
    "cache_protected_len", "full_kv_hit_length",
)
logger = logging.getLogger(__name__)


def around_match_tree(original, cache, params):
    group = getattr(cache, "attn_cp_group", None)
    if not params.cow_mamba or group is None or not cache.supports_mamba():
        return original(cache, params)

    import torch.distributed as dist

    if dist.get_world_size(group=group) <= 1:
        return original(cache, params)
    token = _MATCH_CACHE.set(cache)
    try:
        return original(cache, params)
    finally:
        _MATCH_CACHE.reset(token)


def around_core_match(original, core, params):
    cache = _MATCH_CACHE.get()
    if cache is None or cache.tree_core is not core:
        return original(core, params)

    import torch
    import torch.distributed as dist

    while True:
        result = original(core, params)
        values = [getattr(result, field) for field in _MATCH_FIELDS]
        values = [-1 if value is None else int(value) for value in values]
        bounds = torch.tensor(
            [item for value in values for item in (value, -value)],
            dtype=torch.int64, device="cpu",
        )
        cache._all_reduce_attn_groups(bounds, dist.ReduceOp.MIN)
        bounds = bounds.tolist()
        if all(bounds[i] == -bounds[i + 1] for i in range(0, len(bounds), 2)):
            return result
        cache._apply_cache_actions(result.cache_actions)
        limit = bounds[0]
        key, _ = params.key.maybe_to_bigram_view(core.is_eagle)
        if not 0 <= limit < len(key):
            raise RuntimeError("CP prefix consensus did not reduce the cache lookup limit")
        if dist.get_rank(group=cache.attn_cp_group) == 0:
            logger.warning(
                "CP cache match differs across ranks; retrying at common device prefix %d",
                limit,
            )
        params = replace(params, key=key[:limit])


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    HookRegistry.register(
        "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache._match_tree",
        around_match_tree, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore.match_prefix",
        around_core_match, HookType.AROUND,
    )
