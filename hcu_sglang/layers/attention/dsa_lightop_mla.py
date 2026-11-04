from hcu_sglang.utils.detection import is_hcu


def packed_decode(q, cache, indices, scale):
    import torch
    from hcu_sglang.layers.attention.current_dsa.sparse_mla_triton import sparse_mla
    from lightop import op

    rows, heads, dim = q.shape
    if rows == 0:
        return q.new_empty((0, 1, heads, dim))
    padding = (-indices.shape[-1]) % 64
    if padding:
        indices = torch.nn.functional.pad(indices, (0, padding), value=-1)
    cache = cache.view(torch.uint8).view(-1, 528)
    valid = (indices >= 0) & (indices < cache.shape[0])
    active = valid.any(-1)
    safe = torch.where(valid, indices, -1).contiguous()
    safe[:, 0] = torch.where(active, safe[:, 0], 0)
    lengths = torch.full((rows,), cache.shape[0], dtype=torch.int32, device=q.device)
    gathered = torch.empty((rows, indices.shape[-1], dim), dtype=q.dtype, device=q.device)
    compact = torch.empty_like(safe)
    op.decode_gather_and_up_convert_with_indices(cache, safe, gathered, lengths, compact)
    output = sparse_mla(q, gathered, compact, scale, dim, None, None)
    return torch.where(active[:, None, None], output, 0).unsqueeze(1)


def around_forward_kv(original, backend, q_all, kv_cache, v_head_dim, sm_scale, layer, metadata, page_table_1, *args, **kwargs):
    import torch
    from sglang.srt.environ import envs
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import get_flashmla_kv_valid_rows

    forward_batch = kwargs.get("forward_batch", args[0] if args else None)
    attn_sink = kwargs.get("attn_sink", args[1] if len(args) > 1 else None)
    if not (
        is_hcu()
        and attn_sink is None
        and len(args) <= 2
        and not (kwargs.keys() - {"forward_batch", "attn_sink"})
        and envs.SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER.get()
        and backend.dsa_kv_cache_store_fp8
        and backend.flashmla_kv_format == "V32_NO_ROPE"
        and backend.qk_rope_head_dim == 0
        and layer.head_dim == v_head_dim == 512
        and 0 < layer.tp_q_head_num <= 64
        and q_all.dtype == torch.bfloat16
        and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and kv_cache.shape[-1] == 528
        and kv_cache.is_contiguous()
        and kv_cache.numel() > 0
        and page_table_1.ndim == 2
        and 2048 <= page_table_1.shape[1] <= 2112
        and page_table_1.dtype == torch.int32
    ):
        return original(backend, q_all, kv_cache, v_head_dim, sm_scale, layer, metadata, page_table_1, *args, **kwargs)
    query = q_all.reshape(-1, layer.tp_q_head_num, 512)
    if query.shape[0] != page_table_1.shape[0]:
        raise ValueError("LightOp MLA query and sparse-index rows disagree")
    num_valid = (
        get_flashmla_kv_valid_rows(forward_batch, query.shape[0])
        if forward_batch is not None
        else None
    )
    if num_valid is None:
        return packed_decode(query, kv_cache, page_table_1, sm_scale)
    output = packed_decode(query[:num_valid], kv_cache, page_table_1[:num_valid], sm_scale)
    full_output = output.new_zeros((query.shape[0], *output.shape[1:]))
    full_output[:num_valid] = output
    return full_output


def supports_kpool(backend, topk_indices):
    from sglang.srt.environ import envs

    return (
        is_hcu()
        and envs.SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER.get()
        and backend.dsa_kv_cache_store_fp8
        and backend.flashmla_kv_format == "V32_NO_ROPE"
        and backend.qk_rope_head_dim == 0
        and backend.dsa_index_topk == 2048
        and backend.dsa_index_kpool == 4
        and topk_indices is not None
        and topk_indices.ndim == 2
        and topk_indices.shape[-1] == 2051
    )


def around_check_kpool(original, backend, topk_indices, dsa_impl, phase):
    if supports_kpool(backend, topk_indices) and dsa_impl in ("flashmla_kv", "flashmla_sparse"):
        return None
    return original(backend, topk_indices, dsa_impl, phase)


def around_resolve_kpool(original, backend, topk_indices, dsa_impl):
    if supports_kpool(backend, topk_indices) and dsa_impl == "flashmla_sparse":
        return dsa_impl
    return original(backend, topk_indices, dsa_impl)


def around_forward_sparse(original, backend, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, topk_length=None, attn_sink=None, **kwargs):
    if supports_kpool(backend, page_table_1):
        import torch.nn.functional as F

        padding = (-page_table_1.shape[-1]) % 64
        if padding:
            page_table_1 = F.pad(page_table_1, (0, padding), value=-1)
        topk_length = None
    return original(
        backend, q_all, kv_cache, v_head_dim, page_table_1, sm_scale,
        topk_length=topk_length, attn_sink=attn_sink, **kwargs,
    )


def around_forward_extend(original, backend, q, k, v, layer, forward_batch, *args, **kwargs):
    from dataclasses import replace
    import torch
    from sglang.srt.layers.attention.dsa_backend import TopkTransformMethod

    metadata = backend.forward_metadata
    indices = kwargs.get("topk_indices", args[3] if len(args) > 3 else None)
    offsets = metadata.topk_indices_offset
    rows = q.numel() // (layer.tp_q_head_num * layer.head_dim)
    if not (
        supports_kpool(backend, indices)
        and backend.get_topk_transform_method(forward_batch.forward_mode) == TopkTransformMethod.RAGGED
        and offsets is not None
        and offsets.shape[0] < rows
        and indices.shape[0] in (offsets.shape[0], rows)
    ):
        return original(backend, q, k, v, layer, forward_batch, *args, **kwargs)
    padded_offsets = torch.cat((offsets, offsets.new_zeros((rows - offsets.shape[0], *offsets.shape[1:]))))
    backend.forward_metadata = replace(metadata, topk_indices_offset=padded_offsets)
    try:
        return original(backend, q, k, v, layer, forward_batch, *args, **kwargs)
    finally:
        backend.forward_metadata = metadata


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_flashmla_kv",
        around_forward_kv,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_backend_kpool.DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend",
        around_check_kpool,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_backend_kpool.DeepseekSparseAttnBackendKPoolMixin._resolve_kpool_tail_backend",
        around_resolve_kpool,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_flashmla_sparse",
        around_forward_sparse,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend",
        around_forward_extend,
        HookType.AROUND,
    )
