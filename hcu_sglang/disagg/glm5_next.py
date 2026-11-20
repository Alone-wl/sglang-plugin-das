"""HCU GLM state registration using main's per-component transfer metadata."""

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


def append_hcu_dsa_state(kv_args, token_pool, draft_pool, num_hidden_layers):
    from sglang.srt.disaggregation.utils import append_state_component

    def unwrap(pool):
        return pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool

    target = unwrap(token_pool)
    draft = unwrap(draft_pool)
    pools = [(target, 0)]
    if draft is not None:
        if not all(
            hasattr(draft, name)
            for name in ("get_state_layer_ids", "get_state_buf_infos", "start_layer")
        ):
            raise ValueError("HCU GLM draft pool is missing the Index-K state contract")
        if num_hidden_layers is None:
            raise ValueError("HCU GLM draft state requires total_kv_layers")
        # Draft owns one attention layer. IDs stay stable when a CP rank has
        # no draft buffer; both PD peers reserve the same band above target.
        pools.append((draft, num_hidden_layers - draft.start_layer))

    index_ptrs, index_lens, index_items, index_ids = [], [], [], []
    tail_ptrs, tail_lens, tail_items, tail_ids = [], [], [], []
    has_tail = False
    for pool, offset in pools:
        ids = [offset + i for i in pool.get_state_layer_ids()]
        ptrs, lens, items = pool.get_state_buf_infos()
        if not len(ptrs) == len(lens) == len(items) == len(ids):
            raise ValueError("HCU indexer buffer IDs do not match its owned layers")
        index_ptrs.extend(ptrs)
        index_lens.extend(lens)
        index_items.extend(items)
        index_ids.extend(ids)
        has_tail |= pool.kpool_use_compress
        if not pool.kpool_use_compress:
            continue
        ptrs, lens, items = pool.get_compress_tail_buf_infos()
        ids = list(range(offset + pool.start_layer, offset + pool.start_layer + pool.layer_num)) * 2
        if not len(ptrs) == len(lens) == len(items) == len(ids):
            raise ValueError("HCU GLM tail buffers do not match the dense layer range")
        tail_ptrs.extend(ptrs)
        tail_lens.extend(lens)
        tail_items.extend(items)
        tail_ids.extend(ids)
    append_state_component(
        kv_args,
        StateType.DSA,
        index_ptrs,
        index_lens,
        index_items,
        layer_ids=index_ids,
    )
    if has_tail:
        # Preserve component order even on a rank that owns no tail buffers.
        append_state_component(
            kv_args,
            StateType.DSA_TAIL,
            tail_ptrs,
            tail_lens,
            tail_items,
            layer_ids=tail_ids,
        )
