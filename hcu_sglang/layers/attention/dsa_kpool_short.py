"""Preserve native HCU short-sequence KPool index ordering and mapping."""


def ordered_short_sequence_indices(self, metadata, device):
    import torch
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    # Short sequences do not need a real topk: selecting arange covers the
    # whole sequence. Build the transformed indices directly so HCU/GLM5
    # topk=128 does not depend on fused RAGGED kernels specialized for 2048.
    seq_lens = metadata.get_seqlens_expanded().to(torch.int32)
    rows = seq_lens.shape[0]
    cols = torch.arange(self.index_topk, device=device, dtype=torch.int64)
    token_ids = cols.unsqueeze(0).expand(rows, -1)
    valid = cols.unsqueeze(0) < seq_lens.to(torch.int64).unsqueeze(1)

    use_fused_mapping = envs.SGLANG_DSA_FUSE_TOPK.get() and not getattr(
        metadata, "force_unfused_topk", False
    )
    if not use_fused_mapping:
        # MTP index sharing uses logical token positions as its canonical
        # cross-worker representation. The attention backend localizes a
        # temporary copy against the current worker's allocator.
        topk_full = token_ids.to(torch.int32)
    elif metadata.topk_transform_method == TopkTransformMethod.PAGED:
        page_table = metadata.attn_metadata.page_table_1
        if page_table.shape[0] != rows:
            token_to_batch_idx = metadata.get_token_to_batch_idx()
            if token_to_batch_idx is not None:
                page_table = page_table.index_select(
                    0, token_to_batch_idx.to(page_table.device).to(torch.long)
                )
            elif page_table.shape[0] == 1:
                page_table = page_table.expand(rows, -1)
            else:
                raise RuntimeError(
                    "Cannot align paged page_table rows with short-sequence "
                    f"topk rows: page_table={tuple(page_table.shape)}, rows={rows}"
                )
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        topk_full = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif metadata.topk_transform_method == TopkTransformMethod.RAGGED:
        offsets = metadata.attn_metadata.topk_indices_offset
        if offsets is not None:
            if offsets.ndim == 2:
                offsets = offsets.squeeze(1)
            topk_full = (token_ids + offsets.to(torch.int64).unsqueeze(1)).to(
                torch.int32
            )
        else:
            topk_full = token_ids.to(torch.int32)
    else:
        topk_full = token_ids.to(torch.int32)

    topk_full = torch.where(valid, topk_full, torch.full_like(topk_full, -1))
    # Pad to kpool path width [qo_len, index_topk + kpool - 1] with -1 sentinels
    # so consumers like the MTP index-share buffer see a consistent shape.
    pad = torch.full(
        (topk_full.shape[0], self.index_kpool - 1),
        -1,
        dtype=topk_full.dtype,
        device=topk_full.device,
    )
    return torch.cat([topk_full, pad], dim=1)


def around_short_sequence(original, indexer, metadata, device):
    return ordered_short_sequence_indices(indexer, metadata, device)


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.IndexerKPool._full_topk_for_short_sequence",
        around_short_sequence,
        HookType.AROUND,
    )
