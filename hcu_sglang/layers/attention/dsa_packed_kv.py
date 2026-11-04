from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu


def around_forward_fa3(
    original,
    backend,
    q_rope,
    kv_cache,
    v_head_dim,
    q_nope,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    sm_scale,
    logit_cap,
    page_size,
):
    if not (
        is_hcu()
        and backend.dsa_kv_cache_store_fp8
        and backend.qk_rope_head_dim == 0
        and v_head_dim == 512
        and kv_cache.shape[-1] == 528
        and max_seqlen_q == 1
        and page_size == 1
        and not logit_cap
    ):
        return original(
            backend, q_rope, kv_cache, v_head_dim, q_nope, page_table,
            cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            sm_scale, logit_cap, page_size,
        )

    import torch
    from sgl_kernel.flash_mla import flash_mla_with_kvcache

    num_queries = q_nope.shape[0]
    if page_table.shape[0] != num_queries or cache_seqlens.shape[0] < num_queries:
        raise ValueError("Packed NoPE DSA query and page metadata disagree")
    lengths = cache_seqlens[:num_queries]
    output, _ = flash_mla_with_kvcache(
        q=q_nope.unsqueeze(1),
        k_cache=kv_cache.unsqueeze(1),
        block_table=torch.empty((num_queries, 0), dtype=torch.int32, device=q_nope.device),
        cache_seqlens=lengths,
        head_dim_v=v_head_dim,
        softmax_scale=sm_scale,
        indices=page_table.unsqueeze(1),
        topk_length=lengths,
        is_fp8_kvcache=True,
        kv_format="V32_NO_ROPE",
    )
    return output.squeeze(1)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_fa3",
        around_forward_fa3,
        HookType.AROUND,
    )
