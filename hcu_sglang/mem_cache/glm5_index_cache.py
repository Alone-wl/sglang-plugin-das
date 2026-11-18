from contextvars import ContextVar

from hcu_sglang.utils.detection import is_hcu


_FP8_INDEX_CACHE = ContextVar("hcu_glm53_flash_fp8_index_cache", default=False)


def uses_glm53_flash_fp8(config):
    text = getattr(config, "text_config", config)
    return (
        is_hcu()
        and bool({
            "Glm5NextForConditionalGeneration",
            "Glm5NextForConditionalGenerationNextN",
        }.intersection(getattr(config, "architectures", None) or ()))
        and getattr(text, "index_kpool", None) == 4
        and getattr(text, "index_kpool_compress", False)
        and getattr(text, "index_kpool_always_select_tail", False)
        and getattr(text, "index_head_dim", None) == 128
    )


def fp8_index_cache_active():
    return _FP8_INDEX_CACHE.get()


def around_build_pool(original, configurator, **kwargs):
    token = _FP8_INDEX_CACHE.set(uses_glm53_flash_fp8(configurator.model_config.hf_config))
    try:
        return original(configurator, **kwargs)
    finally:
        _FP8_INDEX_CACHE.reset(token)


def _index_pool_and_layer(pool, layer_id):
    if hasattr(pool, "full_kv_pool"):
        pool._wait_for_layer(layer_id)
        return pool.full_kv_pool, pool._transfer_full_attention_id(layer_id)
    return pool, layer_id


def get_kpool_write_buffer(pool, layer_id):
    index_pool, local_layer = _index_pool_and_layer(pool, layer_id)
    return index_pool.get_index_k_with_scale_write_buffer(local_layer)


def commit_kpool_write_buffer(pool, layer_id, locations):
    index_pool, local_layer = _index_pool_and_layer(pool, layer_id)
    commit = getattr(index_pool, "commit_index_k_with_scale_write_buffer", None)
    if commit is not None:
        commit(local_layer, locations)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for method in ("_build_hybrid_linear_kv_pool", "_build_dsa_kv_pool"):
        HookRegistry.register(
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator."
            + method,
            around_build_pool,
            HookType.AROUND,
        )
