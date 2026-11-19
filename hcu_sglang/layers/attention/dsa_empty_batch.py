from hcu_sglang.utils.detection import is_hcu


def around_init_forward_metadata(original, backend, batch):
    from importlib import import_module

    import torch

    if not is_hcu() or batch.batch_size != 0 or not batch.forward_mode.is_target_verify():
        return original(backend, batch)
    if batch.seq_lens.numel() or batch.input_ids.numel():
        raise ValueError("Empty DSA verification batch contains tokens or lengths")
    empty = batch.seq_lens.to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=empty.device)
    table = backend.req_to_token_pool.req_to_token[:0, :0]
    metadata_type = getattr(import_module(backend.__class__.__module__), "DSAMetadata")
    page_size = getattr(backend, "physical_page_size", None)
    if page_size is None:
        page_size = backend.real_page_size
    backend.forward_metadata = metadata_type(
        page_size=page_size,
        cache_seqlens_int32=empty,
        max_seq_len_q=1,
        max_seq_len_k=0,
        cu_seqlens_q=zero,
        cu_seqlens_k=zero,
        page_table_1=table,
        real_page_table=table,
        dsa_cache_seqlens_int32=empty,
        dsa_cu_seqlens_q=zero,
        dsa_cu_seqlens_k=zero,
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=empty,
        seq_lens_sum=0,
        indexer_seq_lens_cpu=batch.seq_lens_cpu,
        indexer_seq_lens=batch.seq_lens,
    )


def around_kda_target_verify(original, backend, layer, forward_batch, mixed_qkv, a, b):
    if (
        not is_hcu()
        or forward_batch.batch_size != 0
        or not forward_batch.forward_mode.is_target_verify()
    ):
        return original(backend, layer, forward_batch, mixed_qkv, a, b)
    if mixed_qkv.shape[0] or forward_batch.input_ids.numel():
        raise ValueError("Empty KDA verification batch contains tokens")
    return mixed_qkv.new_empty((1, 0, layer.num_v_heads, layer.head_v_dim))


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata",
        around_init_forward_metadata,
        HookType.AROUND,
    )
    HookRegistry.register(
        "hcu_sglang.layers.attention.glm5_next.dsa_backend.NativeSparseAttnBackend.init_forward_metadata",
        around_init_forward_metadata,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend._forward_target_verify",
        around_kda_target_verify,
        HookType.AROUND,
    )
