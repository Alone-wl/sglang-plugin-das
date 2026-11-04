def configure_pages(pool, req_to_token, batch):
    import torch

    pool._finalize_pending_kv_broadcast(set_remote_layer_id=False)
    pool.remote_kv_layer_id = None
    pool.index_key_cache.remote_layer_id = None
    pool._hcu_broadcast_pages = None
    if batch.seq_lens_cpu is None or batch.seq_lens_cpu.device.type != "cpu":
        return
    lengths = batch.seq_lens_cpu.tolist()
    if len(lengths) != batch.req_pool_indices.numel():
        return
    parts = [torch.zeros(1, device=req_to_token.device, dtype=torch.int64)]
    for row, length in enumerate(lengths):
        positions = torch.arange(0, length, pool.page_size, device=req_to_token.device)
        slots = req_to_token[batch.req_pool_indices[row], positions]
        parts.append(torch.div(slots, pool.page_size, rounding_mode="floor").to(torch.int64))
    pool._hcu_broadcast_pages = torch.cat(parts)


def broadcast_pages(pool, original, tensor, layer_id, src_tensor, use_layer_broadcast_comm):
    import torch

    page_ids = getattr(pool, "_hcu_broadcast_pages", None)
    if page_ids is None:
        return original(tensor, layer_id, src_tensor, use_layer_broadcast_comm)
    if tensor is pool.remote_kv_buffer:
        pages = tensor.view(-1, pool.page_size, *tensor.shape[1:])
        source_pages = None if src_tensor is None else src_tensor.view_as(pages)
    elif tensor is pool.index_key_cache.remote_buffer:
        pages = tensor
        source_pages = src_tensor
    else:
        return original(tensor, layer_id, src_tensor, use_layer_broadcast_comm)
    if page_ids.numel() >= pages.shape[0]:
        return original(tensor, layer_id, src_tensor, use_layer_broadcast_comm)
    owner = pool._is_layer_owned(layer_id)
    with torch.profiler.record_function("hcu_layersplit_pack_pages"):
        payload = (
            torch.index_select(source_pages, 0, page_ids)
            if owner else pages.new_empty((page_ids.numel(), *pages.shape[1:]))
        )
    with torch.profiler.record_function("hcu_layersplit_broadcast_pages"):
        original(payload, layer_id, payload if owner else None, use_layer_broadcast_comm)
    with torch.profiler.record_function("hcu_layersplit_scatter_pages"):
        pages.index_copy_(0, page_ids, payload)
    return tensor
