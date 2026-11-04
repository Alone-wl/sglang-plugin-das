# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    o_ptr,
    sink_ptr,
    KV_ROWS: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    sm_scale,
    topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX64: tl.constexpr,
):
    t = tl.program_id(0)
    # Token offsets in int64: at h=128 the q offset t*H*D_QK passes 2^31 from
    # t=29128 and the output offset from t=32768, wrapping to another token.
    t64 = t.to(tl.int64)
    D_TAIL: tl.constexpr = D_QK - D_V

    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + t64 * H * D_QK
    q_main = tl.load(
        qb + h[:, None] * D_QK + dv[None, :], mask=hmask[:, None], other=0.0
    )
    q_tail = tl.load(
        qb + h[:, None] * D_QK + (D_V + dt)[None, :], mask=hmask[:, None], other=0.0
    )

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = topk
    if HAS_LENGTHS:
        k_len = tl.minimum(topk, tl.load(len_ptr + t))
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t64 * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        valid = (idx >= 0) & (idx < KV_ROWS)
        if IDX64:
            row = tl.where(valid, idx, 0).to(tl.int64)
        else:
            row = tl.where(valid, idx, 0)
        kb = kv_ptr + row[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    if HAS_SINK:
        sink = tl.load(sink_ptr + h, mask=hmask, other=-float("inf"))
        m_final = tl.maximum(m_i, sink)
        m_final = tl.where(m_final == -float("inf"), 0.0, m_final)
        alpha = tl.exp(m_i - m_final)
        acc *= alpha[:, None]
        l_i = l_i * alpha + tl.exp(sink - m_final)
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (1.0 / l_safe[:, None])
    tl.store(
        o_ptr + t64 * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


def sparse_mla(q, kv, indices, scale, value_dim, lengths, sink):
    tokens, heads, query_dim = q.shape
    output = q.new_empty((tokens, heads, value_dim))
    if tokens == 0:
        return output
    kv = kv.reshape(-1, query_dim).contiguous()
    _sparse_mla_prefill_kernel[(tokens, triton.cdiv(heads, 16))](
        q.contiguous(), kv, indices.contiguous(),
        lengths.contiguous() if lengths is not None else indices,
        output, sink.contiguous() if sink is not None else q,
        kv.shape[0], lengths is not None, sink is not None,
        scale, indices.shape[-1], heads, 16, query_dim, value_dim, 16, True,
        num_warps=4, num_stages=1,
    )
    return output
