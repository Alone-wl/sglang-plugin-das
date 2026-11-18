from functools import lru_cache

from hcu_sglang.utils.detection import is_hcu


def owned_global_layers(pool):
    inner = pool.full_kv_pool
    return [
        global_id
        for global_id, local_id in pool.full_attention_layer_id_mapping.items()
        if not inner.layer_shard_enabled
        or inner._is_layer_owned(inner.start_layer + local_id)
    ]


@lru_cache(maxsize=1)
def layer_split_pool_class():
    from sglang.srt.mem_cache.dsa_cache_layer_split import (
        LayerSplitDSATokenToKVPool as UpstreamLayerSplitDSATokenToKVPool,
    )

    class LayerSplitDSATokenToKVPool(UpstreamLayerSplitDSATokenToKVPool):
        def __init__(self, *args, **kwargs):
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            super().__init__(
                *args,
                layer_shard_rank=parallel.attn_cp_rank,
                layer_shard_size=parallel.attn_cp_size,
                **kwargs,
            )

        def _broadcast_tensor_from_owner(self, tensor, layer_id, src_tensor=None, use_layer_broadcast_comm=False):
            from hcu_sglang.mem_cache.glm5_layer_split_pages import broadcast_pages

            return broadcast_pages(
                self, super()._broadcast_tensor_from_owner, tensor, layer_id,
                src_tensor, use_layer_broadcast_comm,
            )

        def get_index_k_with_scale_buffer(self, layer_id):
            if not self._is_layer_owned(layer_id):
                return self.index_key_cache.remote_buffer
            return super().get_index_k_with_scale_buffer(layer_id)

        def host_pool_decls(self):
            import msgspec

            owned = tuple(
                i for i in range(self.layer_num)
                if self._is_layer_owned(self.start_layer + i)
            )
            return tuple(
                msgspec.structs.replace(
                    decl,
                    owned_device_layers=tuple(
                        i for i in owned
                        if decl.owned_device_layers is None or i in decl.owned_device_layers
                    ),
                )
                for decl in super().host_pool_decls()
            )

    return LayerSplitDSATokenToKVPool


def around_full_pool_class(original, configurator, **kwargs):
    from sglang.srt.runtime_context import get_parallel

    if not (
        is_hcu()
        and get_parallel().enable_dsa_cache_layer_split
        and not configurator.is_draft_worker
        and configurator.model_config.hf_config.architectures[0]
        == "Glm5NextForConditionalGeneration"
    ):
        return original(configurator, **kwargs)
    if not configurator.use_mla_backend or get_parallel().attn_cp_size <= 1:
        raise ValueError("GLM LayerSplit requires MLA and prefill CP")
    return layer_split_pool_class()


def after_hybrid_pool(result, configurator, **kwargs):
    if is_hcu() and configurator.model_config.hf_config.architectures[0] in (
        "Glm5NextForConditionalGeneration", "Glm5NextForConditionalGenerationNextN",
    ):
        result._hcu_glm5_cache = True
        result.mamba_pool._hcu_glm5_cache = True
        inner = result.full_kv_pool
        if isinstance(inner, layer_split_pool_class()):
            result.layer_shard_enabled = True
            result.layer_shard_rank = inner.layer_shard_rank
            result.layer_shard_size = inner.layer_shard_size
            owned = owned_global_layers(result)
            result.layer_shard_start = owned[0] if owned else result.start_layer
    return result


def around_kv_layer_ids(original, pool):
    if is_hcu() and isinstance(pool.full_kv_pool, layer_split_pool_class()):
        return owned_global_layers(pool)
    return original(pool)


def configure_indexer_layer_ids(kv_args, pool):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from hcu_sglang.disagg.protocol import StateType

    if not (
        is_hcu()
        and isinstance(pool, HybridLinearKVPool)
        and getattr(pool, "_hcu_glm5_cache", False)
        and pool.use_dsa
    ):
        return
    if StateType.DSA not in kv_args.state_types:
        return
    component = kv_args.state_types.index(StateType.DSA)
    layer_ids = owned_global_layers(pool)
    if len(layer_ids) != len(kv_args.state_data_ptrs[component]):
        raise ValueError("GLM indexer buffer count does not match global layer IDs")
    kv_args.state_layer_ids[component] = layer_ids


def around_mamba_host_pool(original, mamba_pool, *, host_size, layout, mtp_draft_device_pools=()):
    if is_hcu() and getattr(mamba_pool, "_hcu_glm5_cache", False) and layout == "layer_first":
        layout = "page_first_direct"
    return original(
        mamba_pool, host_size=host_size, layout=layout,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )


def draft_transfer_view(host_pool, layer_id, io_backend, is_draft):
    from copy import copy

    if not (
        is_hcu()
        and is_draft
        and io_backend in ("kernel", "direct")
        and isinstance(host_pool.device_pool, layer_split_pool_class())
    ):
        return host_pool, layer_id
    depth = layer_id - host_pool.device_pool.layer_num
    if not 0 <= depth < len(host_pool.mtp_draft_device_pools):
        raise ValueError("GLM HiCache draft layer is outside the packed draft range")
    host_layer_id = host_pool.target_layer_num + depth
    view = copy(host_pool)
    buffers = list(host_pool.device_kv_buffers)
    buffers[host_layer_id] = buffers[layer_id]
    view.device_kv_buffers = tuple(buffers)
    return view, host_layer_id


def around_mla_host_backup(original, host_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False):
    view, mapped_layer = draft_transfer_view(host_pool, layer_id, io_backend, is_draft)
    return original(
        view, host_indices, device_indices, mapped_layer, io_backend,
        is_draft=is_draft,
    )


def around_mla_host_load(original, host_pool, device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False):
    view, mapped_layer = draft_transfer_view(host_pool, layer_id, io_backend, is_draft)
    return original(
        view, device_pool, host_indices, device_indices, mapped_layer, io_backend,
        is_draft=is_draft,
    )


def around_dsa_metadata(original, backend, forward_batch):
    pool = getattr(backend.token_to_kv_pool, "full_kv_pool", None)
    if is_hcu() and isinstance(pool, layer_split_pool_class()):
        from hcu_sglang.mem_cache.glm5_layer_split_pages import configure_pages

        configure_pages(pool, backend.req_to_token, forward_batch)
    return original(backend, forward_batch)


def register_hooks():
    from hcu_sglang.mem_cache import glm5_index_cache

    glm5_index_cache.register_hooks()

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata",
        around_dsa_metadata,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._hybrid_full_attention_pool_class",
        around_full_pool_class,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_linear_kv_pool",
        after_hybrid_pool,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.get_kv_layer_ids",
        around_kv_layer_ids,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.build_mamba_host_pool",
        around_mamba_host_pool,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost._backup_from_device_per_layer",
        around_mla_host_backup,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost.load_to_device_per_layer",
        around_mla_host_load,
        HookType.AROUND,
    )
