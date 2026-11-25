# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-query block-sparse decode producer for MiniMax-M3 short-sequence verify steps.

In short-sequence mode every request attends at most ``topk`` KV blocks and block selection is the identity, so
all ``decode_query_len`` query tokens of a request read the same blocks. The per-token producer
(``_gqa_sparse_decode_kernel``) then dequantizes every block once per query token. This producer runs one program
per (request, block) for all of the request's query tokens: rows are (token, head) pairs, each block is
dequantized once, and the causal limit is a per-row mask.

It writes the same per-block partials (output scaled by 1/l and the V global scale, log2 LSE) as the per-token
producer with one block per chunk, so ``_merge_topk_attn_out_kernel`` merges them unchanged.
"""

import triton
import triton.language as tl

from .triton_sparse_decode import _dequant_nvfp4_rows


@triton.jit(
    do_not_specialize=["num_reqs"], do_not_specialize_on_alignment=["block_table_ptr", "seq_lens"]
)
def _gqa_sparse_decode_mq_kernel(
    q_ptr,  # [num_reqs * QLEN, num_heads, head_dim], FP8 or FP16/BF16
    k_ptr,  # packed NVFP4 K: [num_pages, 1, page_size, head_dim / 2] uint8
    v_ptr,  # packed NVFP4 V: same layout
    k_sc_ptr,  # K block scales [num_pages, 1, page_size, head_dim / 16], token-linear
    v_sc_ptr,  # V block scales, 4x4 token-quad order
    k_gs_ptr,
    v_gs_ptr,
    t_ptr,  # topk table [1, total_q, topk]; the request's last token's row is read
    o_ptr,  # partials [topk, total_q, num_heads, head_dim] fp32
    lse_ptr,  # partials [topk, total_q, num_heads] fp32 (log2)
    block_table_ptr,
    seq_lens,
    num_reqs,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_k_blk,
    stride_k_pos,
    stride_v_blk,
    stride_v_pos,
    stride_ksc_blk,
    stride_vsc_blk,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    QLEN: tl.constexpr,
    GROUP: tl.constexpr,  # query heads per KV head (16)
    BLOCK_R: tl.constexpr,  # next power of two of QLEN * GROUP
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)  # top-k position == block index under identity selection
    sm_scale_log2e = sm_scale * 1.4426950409 * tl.load(k_gs_ptr)
    v_global_scale = tl.load(v_gs_ptr)

    off_r = tl.arange(0, BLOCK_R)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, HEAD_DIM)
    off_dp = tl.arange(0, HEAD_DIM // 2)
    off_s = tl.arange(0, SCALE_COLS)
    row_t = off_r // GROUP
    row_h = off_r % GROUP
    r_mask = off_r < QLEN * GROUP
    tok = pid_r * QLEN + row_t

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    # Token t of the request sits at seq_len - QLEN + t and attends positions [0, kv_len_t).
    kv_len = tl.maximum(seq_len - QLEN + row_t + 1, 0)
    blk = tl.load(t_ptr + (pid_r * QLEN + QLEN - 1) * stride_tn + pid_c * stride_tk)
    blk_ok = blk >= 0
    blk = tl.where(blk_ok, blk, 0)
    page = tl.load(block_table_ptr + pid_r * stride_bt_b + blk, mask=blk_ok, other=0).to(tl.int64)
    # Rows past the last token's position are never read: their cache bytes may be stale (0 * NaN in the PV dot).
    n_valid = tl.where(
        blk_ok, tl.minimum(tl.maximum(seq_len - blk * BLOCK_SIZE_K, 0), BLOCK_SIZE_K), 0
    )
    n_mask = off_n < n_valid

    q = tl.load(
        q_ptr + tok[:, None] * stride_qn + row_h[:, None] * stride_qh + off_d[None, :],
        mask=r_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    k_deq = _dequant_nvfp4_rows(
        tl.load(
            k_ptr + page * stride_k_blk + off_n[:, None] * stride_k_pos + off_dp[None, :],
            mask=n_mask[:, None],
            other=0,
        ),
        tl.load(
            k_sc_ptr + page * stride_ksc_blk + off_n[:, None] * SCALE_COLS + off_s[None, :],
            mask=n_mask[:, None],
            other=0,
        ),
        BLOCK_N=BLOCK_SIZE_K,
        D=HEAD_DIM,
        SCALE_COLS=SCALE_COLS,
        SF_VEC=HEAD_DIM // SCALE_COLS,
    )
    valid = (
        (blk * BLOCK_SIZE_K + off_n[None, :] < kv_len[:, None]) & n_mask[None, :] & r_mask[:, None]
    )
    qk = tl.dot(q, tl.trans(k_deq).to(tl.float16)) * sm_scale_log2e
    qk = tl.where(valid, qk, float("-inf"))
    m = tl.max(qk, axis=1)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    p = tl.exp2(qk - m_safe[:, None])
    p_sum = tl.sum(p, axis=1)
    v = _dequant_nvfp4_rows(
        tl.load(
            v_ptr + page * stride_v_blk + off_n[:, None] * stride_v_pos + off_dp[None, :],
            mask=n_mask[:, None],
            other=0,
        ),
        tl.load(
            v_sc_ptr
            + page * stride_vsc_blk
            + (off_n[:, None] // 4) * (4 * SCALE_COLS)
            + 4 * off_s[None, :]
            + (off_n[:, None] % 4),
            mask=n_mask[:, None],
            other=0,
        ),
        BLOCK_N=BLOCK_SIZE_K,
        D=HEAD_DIM,
        SCALE_COLS=SCALE_COLS,
        SF_VEC=HEAD_DIM // SCALE_COLS,
    ).to(tl.float16)
    acc = tl.dot(p.to(tl.float16), v)
    # An empty partial stores zero with LSE -inf, so the merge gives it weight 0.
    scale = tl.where(p_sum > 0.0, 1.0 / p_sum, 0.0) * v_global_scale
    lse = tl.where(p_sum > 0.0, m + tl.log2(p_sum), float("-inf"))
    acc = acc * scale[:, None]
    tl.store(
        o_ptr
        + pid_c * stride_o_c
        + tok[:, None] * stride_o_b
        + row_h[:, None] * stride_o_h
        + off_d[None, :],
        acc,
        mask=r_mask[:, None],
    )
    tl.store(lse_ptr + pid_c * stride_l_c + tok * stride_l_b + row_h * stride_l_h, lse, mask=r_mask)
    if USE_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


def launch_mq_producer(
    q,
    k_paged,
    v_paged,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    topk_idx,
    block_table,
    seq_lens,
    o_arg,
    lse_arg,
    *,
    sm_scale,
    decode_query_len,
    use_pdl,
    early_trigger,
    num_warps=8,
    num_stages=1,
):
    """Launch the multi-query producer; arguments as minimax_m3_sparse_attn_decode after its NVFP4 views."""
    total_q, num_heads, head_dim = q.shape
    num_reqs = total_q // decode_query_len
    max_topk = int(topk_idx.shape[-1])
    rows = decode_query_len * num_heads
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    _gqa_sparse_decode_mq_kernel[(num_reqs, max_topk)](
        q,
        k_paged,
        v_paged,
        k_block_scale,
        v_block_scale,
        k_global_scale,
        v_global_scale,
        topk_idx,
        o_arg,
        lse_arg,
        block_table,
        seq_lens,
        num_reqs,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_paged.stride(0),
        k_paged.stride(2),
        v_paged.stride(0),
        v_paged.stride(2),
        k_block_scale.stride(0),
        v_block_scale.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_arg.stride(0),
        o_arg.stride(1),
        o_arg.stride(2),
        lse_arg.stride(0),
        lse_arg.stride(1),
        lse_arg.stride(2),
        block_table.stride(0),
        QLEN=decode_query_len,
        GROUP=num_heads,
        BLOCK_R=triton.next_power_of_2(rows),
        HEAD_DIM=head_dim,
        BLOCK_SIZE_K=int(k_paged.shape[2]),
        SCALE_COLS=head_dim // 16,
        USE_PDL=use_pdl,
        EARLY_TRIGGER=early_trigger,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_launch,
    )
