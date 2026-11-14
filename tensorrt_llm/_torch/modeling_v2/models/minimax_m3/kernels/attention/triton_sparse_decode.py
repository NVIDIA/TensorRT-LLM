# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton
import triton.language as tl


@triton.jit
def _e2m1x2_to_f16x2(packed):
    """Convert one packed E2M1 byte into its low/high FP16 values."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b8 src;
        .reg .b32 pair;
        mov.b32 {src, _, _, _}, $2;
        cvt.rn.f16x2.e2m1x2 pair, src;
        mov.b32 {$0, $1}, pair;
        }""",
        constraints="=h,=h,r",
        args=(packed,),
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dequant_nvfp4_rows(
    packed,  # [BLOCK_N, D // 2] uint8, two E2M1 nibbles per byte
    scale_bytes,  # [BLOCK_N, SCALE_COLS] uint8, one E4M3 scale per 16 elements
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SF_VEC: tl.constexpr,
):
    """Dequantize one packed NVFP4 [BLOCK_N, D] tile to fp16.

    The caller gathers the block scales in whichever layout its role stores
    them, so this handles only the value math and the scale broadcast. The
    per-tensor scale is applied by the caller.
    """
    # Blackwell converts both E2M1 nibbles with one instruction. Joining the
    # low/high results along a new innermost axis and flattening it rebuilds
    # the original element order without scalar exponent/sign reconstruction.
    value_lo, value_hi = _e2m1x2_to_f16x2(packed)
    values = tl.reshape(tl.join(value_lo, value_hi), (BLOCK_N, D))

    scales = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    scales = tl.reshape(
        tl.broadcast_to(scales[:, :, None], (BLOCK_N, SCALE_COLS, SF_VEC)), (BLOCK_N, D)
    )
    return values * scales


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
# Mixed ctx+gen steps launch on [num_ctx:] slices (unaligned seq_lens / block_table) with arbitrary
# total_q; specializing on those compiled a new variant mid-serving (~650 ms TTFT stall at c8).
@triton.jit(
    do_not_specialize=["decode_query_len", "total_q"],
    do_not_specialize_on_alignment=["block_table_ptr", "seq_lens"],
)
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # paged K: [num_pages, num_kv_heads, page_size, head_dim (or /2 packed)]
    v_ptr,  # paged V: same layout as K
    kv_scale_ptr,  # scalar dequant scale, or a dummy when USE_SCALE is False
    k_sc_ptr,  # paged K block scales: [num_pages, num_kv_heads, page_size, D/16]
    v_sc_ptr,  # paged V block scales: same shape, 4x4 token-quad order
    k_gs_ptr,  # NVFP4 per-tensor K dequant scale, one fp32
    v_gs_ptr,  # NVFP4 per-tensor V dequant scale, one fp32
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out, or final [total_q, num_heads, head_dim] when DIRECT_OUTPUT
    lse_ptr,  # partial lse (log2), unused when DIRECT_OUTPUT
    out_ptr,  # final [total_q, num_heads, head_dim] for FUSED_MERGE, else unused
    counter_ptr,  # int32 [total_q, num_kv_heads] arrival counters for FUSED_MERGE, zero at rest
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_blk,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_ksc_blk,
    stride_ksc_h,
    stride_vsc_blk,
    stride_vsc_h,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    num_bt_cols,  # block-table width: a top-k id at or past it is not a block of the request
    num_pages,  # KV pool pages: a page id outside [0, num_pages) is not read
    stride_out_n,
    stride_out_h,
    stride_out_d,
    num_kv_heads,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_SCALE: tl.constexpr,  # apply the scalar dequant scale to an fp8 cache
    WIDEN_Q: tl.constexpr,  # q arrives FP8 and is widened in-register
    COMPUTE_DTYPE: tl.constexpr,  # dtype q is widened to; drives the QK/PV math
    USE_PDL: tl.constexpr,
    KV_NVFP4: tl.constexpr,  # cache holds packed E2M1 plus E4M3 block scales
    USE_LINEAR_SOFTMAX: tl.constexpr,  # measured NVFP4 online-softmax recurrence
    SCALE_COLS: tl.constexpr,  # head_dim // NVFP4_SF_VEC_SIZE, NVFP4 only
    DIRECT_OUTPUT: tl.constexpr,  # one chunk needs neither FP32 scratch nor a merge
    FUSED_MERGE: tl.constexpr,  # the last chunk to finish merges the partials in place
    EARLY_TRIGGER: tl.constexpr,  # release PDL dependents (weight prefetch) at entry
    IDENTITY: tl.constexpr = False,  # identity selection: slot i is block i (t_ptr is not read)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    if KV_NVFP4:
        # Both per-tensor dequant scales factor out of the inner loop: QK is
        # linear in K, and V's scale commutes with the softmax weights. Reading
        # them from device memory keeps the launch free of a host sync, so
        # CUDA-graph capture stays unconstrained.
        sm_scale_log2e = sm_scale_log2e * tl.load(k_gs_ptr)
        v_global_scale = tl.load(v_gs_ptr)
    # Split-K over the topk dimension: pid(0) folds (query token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start = pid_c * chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            # Dependents still wait for this grid to complete; triggering here only
            # lets them start their input-independent prologue (e.g. weight loads).
            tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Bound the walk by the token's own valid block count rather than a
    # sentinel: the selector emits ascending ids and only pads with -1 past
    # min(topk, cdiv(kv_len, blk)) entries, so those are never dereferenced.
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end = tl.minimum(chunk_start + chunk_size_topk, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    if KV_NVFP4:
        # Packed byte column and block-scale column. The NVFP4 path runs only at
        # head_dim == BLOCK_SIZE_D (128), so neither needs a tail mask.
        off_dp = tl.arange(0, BLOCK_SIZE_D // 2)
        off_s = tl.arange(0, SCALE_COLS)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size
    hd_mask = h_mask[:, None] & d_mask[None, :]
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    if USE_LINEAR_SOFTMAX:
        # Keep the linear softmax denominator and reuse the producer's alpha.
        # This removes one exp2 plus one log2 from every visited KV page; only
        # the final LSE written for split-K merging still needs a logarithm.
        l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    else:
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q = tl.load(
        q_ptr
        + pid_b * stride_qn
        + (pid_h + off_h[:, None]) * stride_qh
        + off_d[None, :] * stride_qd,
        mask=hd_mask,
        other=0.0,
    )
    # Widen before anything reads q.dtype: K, V and p are all cast to it below,
    # so leaving q narrow would silently run the whole attention in FP8 and
    # quantize the softmax probabilities. E4M3 -> BF16 is exact, so the result
    # matches a caller that widened q itself before the call.
    if WIDEN_Q:
        q = q.to(COMPUTE_DTYPE)
    kv_scale = tl.load(kv_scale_ptr) if USE_SCALE else 1.0

    cur_idx_ptr = t_ptr + pid_kh * stride_th + pid_b * stride_tn + chunk_start * stride_tk
    for it in tl.range(chunk_start, chunk_end):
        if IDENTITY:
            # Slot `it` of the identity selection, bounded by this row's own kv extent above.
            blk = it
        else:
            blk = tl.load(cur_idx_ptr)
            cur_idx_ptr = cur_idx_ptr + stride_tk
        # int64 page offsets: a large cache overflows int32 well before the
        # per-page block offsets do.
        # Fail closed: a top-k id outside the request's block-table row, or the page it names outside the pool,
        # contributes nothing instead of a read through a wrong page (valid inputs are unaffected).
        blk_ok = (blk >= 0) & (blk < num_bt_cols)
        page = tl.load(bt_row + blk, mask=blk_ok, other=0).to(tl.int64)
        blk_ok = blk_ok & (page >= 0) & (page < num_pages)
        page = tl.where(blk_ok, page, 0)
        pos_mask = blk * BLOCK_SIZE_K + off_n < kv_len
        # A slot that failed closed is skipped whole: no score, no rescale (an all-masked first block
        # would make exp2(-inf - -inf) = NaN).
        if blk_ok:
            if KV_NVFP4:
                # K block scales are token-major linear within the page-head region;
                # see _fused_nvfp4_paged_scatter_kernel in msa_scatter.
                k_deq = _dequant_nvfp4_rows(
                    tl.load(
                        k_ptr
                        + page * stride_k_blk
                        + pid_kh * stride_k_h
                        + off_n[:, None] * stride_k_pos
                        + off_dp[None, :] * stride_k_d,
                        mask=pos_mask[:, None],
                        other=0,
                    ),
                    tl.load(
                        k_sc_ptr
                        + page * stride_ksc_blk
                        + pid_kh * stride_ksc_h
                        + off_n[:, None] * SCALE_COLS
                        + off_s[None, :],
                        mask=pos_mask[:, None],
                        other=0,
                    ),
                    BLOCK_N=BLOCK_SIZE_K,
                    D=BLOCK_SIZE_D,
                    SCALE_COLS=SCALE_COLS,
                    SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
                )
                k = tl.trans(k_deq).to(q.dtype)
            else:
                k = tl.load(
                    k_ptr
                    + page * stride_k_blk
                    + pid_kh * stride_k_h
                    + off_n[None, :] * stride_k_pos
                    + off_d[:, None] * stride_k_d,
                    mask=d_mask[:, None] & pos_mask[None, :],
                    other=0.0,
                ).to(q.dtype)
                if USE_SCALE:
                    k = (k * kv_scale).to(q.dtype)
            qk = tl.where(pos_mask[None, :], 0.0, float("-inf"))
            qk += tl.dot(q, k) * sm_scale_log2e
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            if USE_LINEAR_SOFTMAX:
                alpha = tl.exp2(m_i - m_ij)
                acc_o = acc_o * alpha[:, None]
            else:
                acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            if KV_NVFP4:
                # V block scales use vLLM's 4x4 token-quad order rather than K's
                # linear one: [token // 4, scale_col, token % 4].
                v = _dequant_nvfp4_rows(
                    tl.load(
                        v_ptr
                        + page * stride_v_blk
                        + pid_kh * stride_v_h
                        + off_n[:, None] * stride_v_pos
                        + off_dp[None, :] * stride_v_d,
                        mask=pos_mask[:, None],
                        other=0,
                    ),
                    tl.load(
                        v_sc_ptr
                        + page * stride_vsc_blk
                        + pid_kh * stride_vsc_h
                        + (off_n[:, None] // 4) * (4 * SCALE_COLS)
                        + 4 * off_s[None, :]
                        + (off_n[:, None] % 4),
                        mask=pos_mask[:, None],
                        other=0,
                    ),
                    BLOCK_N=BLOCK_SIZE_K,
                    D=BLOCK_SIZE_D,
                    SCALE_COLS=SCALE_COLS,
                    SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
                ).to(q.dtype)
            else:
                v = tl.load(
                    v_ptr
                    + page * stride_v_blk
                    + pid_kh * stride_v_h
                    + off_n[:, None] * stride_v_pos
                    + off_d[None, :] * stride_v_d,
                    mask=pos_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(q.dtype)
                if USE_SCALE:
                    v = (v * kv_scale).to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            if USE_LINEAR_SOFTMAX:
                l_i = l_i * alpha + l_ij
            else:
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
            m_i = m_ij

    # An empty chunk of an active row must store zero, or the merge hits 0 * NaN.
    if USE_LINEAR_SOFTMAX:
        scale = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
        lse_out = tl.where(l_i > 0.0, m_i + tl.log2(l_i), float("-inf"))
    else:
        scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), 0.0)
        lse_out = lse_i
    if KV_NVFP4:
        # Applying V's scale per partial is exact: the merge is a weighted
        # average whose weights sum to one, so a constant survives it.
        scale = scale * v_global_scale
    acc_o = acc_o * scale[:, None]
    o_base = o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + off_h[:, None] * stride_o_h + off_d[None, :] * stride_o_d,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=hd_mask,
    )
    if not DIRECT_OUTPUT:
        lse_base = lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
        tl.store(lse_base + off_h * stride_l_h, lse_out, mask=h_mask)

    if FUSED_MERGE:
        # Split-K serial reduction: every chunk publishes its partial, and the
        # chunk that arrives last for this (token, KV head) merges all of them
        # exactly as _merge_topk_attn_out_kernel does, then re-arms the counter.
        tl.debug_barrier()
        counter = counter_ptr + pid_b * num_kv_heads + pid_kh
        arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
        if arrived == NUM_TOPK_CHUNKS - 1:
            tl.debug_barrier()
            o_rows = (
                o_ptr
                + pid_b * stride_o_b
                + (pid_h + off_h[:, None]) * stride_o_h
                + off_d[None, :] * stride_o_d
            )
            l_rows = lse_ptr + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
            lse_max = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            for c in tl.static_range(NUM_TOPK_CHUNKS):
                lse_c = tl.load(
                    l_rows + c * stride_l_c, mask=h_mask, other=float("-inf"), cache_modifier=".cg"
                )
                lse_max = tl.maximum(lse_max, lse_c)
            # All chunks empty (a CUDA-graph padding row): zero, never NaN.
            lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
            denom = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
            for c in tl.static_range(NUM_TOPK_CHUNKS):
                lse_c = tl.load(
                    l_rows + c * stride_l_c, mask=h_mask, other=float("-inf"), cache_modifier=".cg"
                )
                denom += tl.exp2(lse_c - lse_max)
            denom = tl.where(denom > 0, denom, 1.0)
            merged = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
            for c in tl.static_range(NUM_TOPK_CHUNKS):
                lse_c = tl.load(
                    l_rows + c * stride_l_c, mask=h_mask, other=float("-inf"), cache_modifier=".cg"
                )
                w_c = tl.exp2(lse_c - lse_max) / denom
                o_c = tl.load(
                    o_rows + c * stride_o_c, mask=hd_mask, other=0.0, cache_modifier=".cg"
                ).to(tl.float32)
                merged += o_c * w_c[:, None]
            tl.store(
                out_ptr
                + pid_b * stride_out_n
                + (pid_h + off_h[:, None]) * stride_out_h
                + off_d[None, :] * stride_out_d,
                merged.to(out_ptr.dtype.element_ty),
                mask=hd_mask,
            )
            tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")

    # After the stores, never before: this releases either the merge grid or,
    # for direct output, the next PDL-enabled consumer of the final tensor.
    if USE_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16,
            triton.next_power_of_2(args["gqa_group_size"] * args["QUERY_GROUP_SIZE"]),
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(
    do_not_specialize=["decode_query_len", "total_q"],
    do_not_specialize_on_alignment=["block_table_ptr", "seq_lens"],
)
def _gqa_sparse_decode_pair_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # paged K: [num_pages, num_kv_heads, page_size, head_dim (or /2 packed)]
    v_ptr,  # paged V: same layout as K
    kv_scale_ptr,  # scalar dequant scale, or a dummy when USE_SCALE is False
    k_sc_ptr,  # paged K block scales: [num_pages, num_kv_heads, page_size, D/16]
    v_sc_ptr,  # paged V block scales: same shape, 4x4 token-quad order
    k_gs_ptr,  # NVFP4 per-tensor K dequant scale, one fp32
    v_gs_ptr,  # NVFP4 per-tensor V dequant scale, one fp32
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out, or final [total_q, num_heads, head_dim] when DIRECT_OUTPUT
    lse_ptr,  # partial lse (log2), unused when DIRECT_OUTPUT
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_blk,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_ksc_blk,
    stride_ksc_h,
    stride_vsc_blk,
    stride_vsc_h,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    num_bt_cols,  # block-table width: a top-k id at or past it is not a block of the request
    num_pages,  # KV pool pages: a page id outside [0, num_pages) is not read
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_SCALE: tl.constexpr,  # apply the scalar dequant scale to an fp8 cache
    WIDEN_Q: tl.constexpr,  # q arrives FP8 and is widened in-register
    COMPUTE_DTYPE: tl.constexpr,  # dtype q is widened to; drives the QK/PV math
    USE_PDL: tl.constexpr,
    KV_NVFP4: tl.constexpr,  # cache holds packed E2M1 plus E4M3 block scales
    USE_LINEAR_SOFTMAX: tl.constexpr,  # measured NVFP4 online-softmax recurrence
    QUERY_GROUP_SIZE: tl.constexpr,  # consecutive Eagle queries sharing one block set
    MAX_TOPK: tl.constexpr,
    SCALE_COLS: tl.constexpr,  # head_dim // NVFP4_SF_VEC_SIZE, NVFP4 only
    DIRECT_OUTPUT: tl.constexpr,  # one chunk needs neither FP32 scratch nor a merge
    IDENTITY: tl.constexpr = False,  # identity selection: slot i is block i (t_ptr is not read)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    if KV_NVFP4:
        # Both per-tensor dequant scales factor out of the inner loop: QK is
        # linear in K, and V's scale commutes with the softmax weights. Reading
        # them from device memory keeps the launch free of a host sync, so
        # CUDA-graph capture stays unconstrained.
        sm_scale_log2e = sm_scale_log2e * tl.load(k_gs_ptr)
        v_global_scale = tl.load(v_gs_ptr)
    # Split-K over the topk dimension: pid(0) folds (query group, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    num_query_groups = total_q // QUERY_GROUP_SIZE
    pid_g = pid_bc % num_query_groups
    pid_c = pid_bc // num_query_groups
    groups_per_req = decode_query_len // QUERY_GROUP_SIZE
    req_id = pid_g // groups_per_req
    q_group_offset = (pid_g - req_id * groups_per_req) * QUERY_GROUP_SIZE
    pid_b = req_id * decode_query_len + q_group_offset
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start = pid_c * chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    # Query rows are flattened as [query-in-group, GQA head].
    off_h = tl.arange(0, BLOCK_SIZE_H)
    q_lane = off_h // gqa_group_size
    h_lane = off_h - q_lane * gqa_group_size
    h_mask = off_h < gqa_group_size * QUERY_GROUP_SIZE
    q_token = pid_b + q_lane
    query_pos = seq_len - decode_query_len + q_group_offset + q_lane
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    kv_len_max = tl.max(tl.where(h_mask, kv_len, 0), axis=0)

    # Each query can have a different valid-block count at a page boundary.
    # The selector puts those valid entries first and pads only the tail.
    kv_len_0 = tl.maximum(seq_len - decode_query_len + q_group_offset + 1, 0)
    num_blocks_0 = (kv_len_0 + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk_0 = tl.minimum(max_topk, num_blocks_0)
    chunk_end_0 = tl.minimum(chunk_start + chunk_size_topk, real_topk_0)
    if QUERY_GROUP_SIZE == 2:
        kv_len_1 = tl.maximum(seq_len - decode_query_len + q_group_offset + 2, 0)
        num_blocks_1 = (kv_len_1 + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        real_topk_1 = tl.minimum(max_topk, num_blocks_1)
        chunk_end_1 = tl.minimum(chunk_start + chunk_size_topk, real_topk_1)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    if KV_NVFP4:
        # Packed byte column and block-scale column. The NVFP4 path runs only at
        # head_dim == BLOCK_SIZE_D (128), so neither needs a tail mask.
        off_dp = tl.arange(0, BLOCK_SIZE_D // 2)
        off_s = tl.arange(0, SCALE_COLS)
    d_mask = off_d < head_dim
    hd_mask = h_mask[:, None] & d_mask[None, :]
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    if USE_LINEAR_SOFTMAX:
        # Keep the linear softmax denominator and reuse the producer's alpha.
        # This removes one exp2 plus one log2 from every visited KV page; only
        # the final LSE written for split-K merging still needs a logarithm.
        l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    else:
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q = tl.load(
        q_ptr
        + q_token[:, None] * stride_qn
        + (pid_h + h_lane[:, None]) * stride_qh
        + off_d[None, :] * stride_qd,
        mask=hd_mask,
        other=0.0,
    )
    # Widen before anything reads q.dtype: K, V and p are all cast to it below,
    # so leaving q narrow would silently run the whole attention in FP8 and
    # quantize the softmax probabilities. E4M3 -> BF16 is exact, so this
    # reproduces the values the caller's standalone widening kernel produced.
    if WIDEN_Q:
        q = q.to(COMPUTE_DTYPE)
    kv_scale = tl.load(kv_scale_ptr) if USE_SCALE else 1.0

    idx_ptr_0 = t_ptr + pid_kh * stride_th + pid_b * stride_tn + chunk_start * stride_tk
    if QUERY_GROUP_SIZE == 2:
        idx_ptr_1 = idx_ptr_0 + stride_tn
        idx_0 = chunk_start
        idx_1 = chunk_start
        count_0 = chunk_end_0 - chunk_start
        count_1 = chunk_end_1 - chunk_start
        chunk_capacity: tl.constexpr = (MAX_TOPK + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
        chunk_offsets = tl.arange(0, chunk_capacity)
        sentinel = 2147483647
        if IDENTITY:
            blocks_0 = tl.where(chunk_offsets < count_0, chunk_start + chunk_offsets, sentinel)
            blocks_1 = tl.where(chunk_offsets < count_1, chunk_start + chunk_offsets, sentinel)
        else:
            blocks_0 = tl.load(
                idx_ptr_0 + chunk_offsets * stride_tk,
                mask=chunk_offsets < count_0,
                other=sentinel,
            )
            blocks_1 = tl.load(
                idx_ptr_1 + chunk_offsets * stride_tk,
                mask=chunk_offsets < count_1,
                other=sentinel,
            )
        chunk_equal = tl.sum(blocks_0 != blocks_1, axis=0) == 0
        # Equal chunks take the measured shared-page loop. A divergent chunk
        # walks the sorted union, bounded by the sum of both row lengths.
        num_candidates = tl.where(chunk_equal, count_0, count_0 + count_1)
    else:
        num_candidates = chunk_end_0 - chunk_start

    for cand in tl.range(0, num_candidates):
        if QUERY_GROUP_SIZE == 2:
            if chunk_equal:
                has_block = True
                if IDENTITY:
                    blk = chunk_start + cand
                else:
                    blk = tl.load(idx_ptr_0)
                idx_ptr_0 += stride_tk
                query_active = h_mask
            else:
                valid_0 = idx_0 < chunk_end_0
                valid_1 = idx_1 < chunk_end_1
                if IDENTITY:
                    blk_0 = tl.where(valid_0, idx_0, sentinel)
                    blk_1 = tl.where(valid_1, idx_1, sentinel)
                else:
                    blk_0 = tl.load(idx_ptr_0, mask=valid_0, other=sentinel)
                    blk_1 = tl.load(idx_ptr_1, mask=valid_1, other=sentinel)
                use_0 = valid_0 & (blk_0 <= blk_1)
                use_1 = valid_1 & (blk_1 <= blk_0)
                has_block = use_0 | use_1
                blk = tl.minimum(blk_0, blk_1)
                idx_0 += use_0.to(tl.int32)
                idx_1 += use_1.to(tl.int32)
                idx_ptr_0 += use_0.to(tl.int64) * stride_tk
                idx_ptr_1 += use_1.to(tl.int64) * stride_tk
                query_active = h_mask & (((q_lane == 0) & use_0) | ((q_lane == 1) & use_1))
        else:
            has_block = True
            if IDENTITY:
                blk = chunk_start + cand
            else:
                blk = tl.load(idx_ptr_0)
            idx_ptr_0 += stride_tk
            query_active = h_mask

        if has_block:
            # int64 page offsets: a large cache overflows int32 well before the
            # per-page block offsets do.
            # Fail closed as in _gqa_sparse_decode_kernel: an id outside the row, or a page outside the
            # pool, reads nothing.
            blk_ok = (blk >= 0) & (blk < num_bt_cols)
            page = tl.load(bt_row + blk, mask=blk_ok, other=0).to(tl.int64)
            blk_ok = blk_ok & (page >= 0) & (page < num_pages)
            page = tl.where(blk_ok, page, 0)
            pos_mask = blk * BLOCK_SIZE_K + off_n < kv_len_max
            q_pos_mask = blk * BLOCK_SIZE_K + off_n[None, :] < kv_len[:, None]
            # A slot that failed closed is skipped whole: no score, no rescale (an all-masked first block
            # would make exp2(-inf - -inf) = NaN).
            if blk_ok:
                if KV_NVFP4:
                    # K scales are token-major linear within the page-head region;
                    # see _fused_nvfp4_paged_scatter_kernel in msa_scatter.
                    k_deq = _dequant_nvfp4_rows(
                        tl.load(
                            k_ptr
                            + page * stride_k_blk
                            + pid_kh * stride_k_h
                            + off_n[:, None] * stride_k_pos
                            + off_dp[None, :] * stride_k_d,
                            mask=pos_mask[:, None],
                            other=0,
                        ),
                        tl.load(
                            k_sc_ptr
                            + page * stride_ksc_blk
                            + pid_kh * stride_ksc_h
                            + off_n[:, None] * SCALE_COLS
                            + off_s[None, :],
                            mask=pos_mask[:, None],
                            other=0,
                        ),
                        BLOCK_N=BLOCK_SIZE_K,
                        D=BLOCK_SIZE_D,
                        SCALE_COLS=SCALE_COLS,
                        SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
                    )
                    k = tl.trans(k_deq).to(q.dtype)
                else:
                    k = tl.load(
                        k_ptr
                        + page * stride_k_blk
                        + pid_kh * stride_k_h
                        + off_n[None, :] * stride_k_pos
                        + off_d[:, None] * stride_k_d,
                        mask=d_mask[:, None] & pos_mask[None, :],
                        other=0.0,
                    ).to(q.dtype)
                    if USE_SCALE:
                        k = (k * kv_scale).to(q.dtype)
                active_pos = query_active[:, None] & q_pos_mask
                qk = tl.where(active_pos, 0.0, float("-inf"))
                qk += tl.dot(q, k) * sm_scale_log2e
                m_ij = tl.where(query_active, tl.maximum(m_i, tl.max(qk, axis=1)), m_i)
                p = tl.where(active_pos, tl.exp2(qk - m_ij[:, None]), 0.0)
                l_ij = tl.sum(p, axis=1)
                alpha = tl.where(query_active, tl.exp2(m_i - m_ij), 1.0)
                acc_o = acc_o * alpha[:, None]
                if KV_NVFP4:
                    # V scales use vLLM's 4x4 token-quad order rather than K's
                    # linear one: [token // 4, scale_col, token % 4].
                    v = _dequant_nvfp4_rows(
                        tl.load(
                            v_ptr
                            + page * stride_v_blk
                            + pid_kh * stride_v_h
                            + off_n[:, None] * stride_v_pos
                            + off_dp[None, :] * stride_v_d,
                            mask=pos_mask[:, None],
                            other=0,
                        ),
                        tl.load(
                            v_sc_ptr
                            + page * stride_vsc_blk
                            + pid_kh * stride_vsc_h
                            + (off_n[:, None] // 4) * (4 * SCALE_COLS)
                            + 4 * off_s[None, :]
                            + (off_n[:, None] % 4),
                            mask=pos_mask[:, None],
                            other=0,
                        ),
                        BLOCK_N=BLOCK_SIZE_K,
                        D=BLOCK_SIZE_D,
                        SCALE_COLS=SCALE_COLS,
                        SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
                    ).to(q.dtype)
                else:
                    v = tl.load(
                        v_ptr
                        + page * stride_v_blk
                        + pid_kh * stride_v_h
                        + off_n[:, None] * stride_v_pos
                        + off_d[None, :] * stride_v_d,
                        mask=pos_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    ).to(q.dtype)
                    if USE_SCALE:
                        v = (v * kv_scale).to(q.dtype)
                acc_o += tl.dot(p.to(v.dtype), v)
                if USE_LINEAR_SOFTMAX:
                    l_i = tl.where(query_active, l_i * alpha + l_ij, l_i)
                else:
                    lse_ij = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
                    lse_i = tl.where(query_active, lse_ij, lse_i)
                m_i = m_ij

    # An empty chunk of an active row must store zero, or the merge hits 0 * NaN.
    if USE_LINEAR_SOFTMAX:
        scale = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
        lse_out = tl.where(l_i > 0.0, m_i + tl.log2(l_i), float("-inf"))
    else:
        scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), 0.0)
        lse_out = lse_i
    if KV_NVFP4:
        # Applying V's scale per partial is exact: the merge is a weighted
        # average whose weights sum to one, so a constant survives it.
        scale = scale * v_global_scale
    acc_o = acc_o * scale[:, None]
    tl.store(
        o_ptr
        + pid_c * stride_o_c
        + q_token[:, None] * stride_o_b
        + (pid_h + h_lane[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=hd_mask,
    )
    if not DIRECT_OUTPUT:
        tl.store(
            lse_ptr + pid_c * stride_l_c + q_token * stride_l_b + (pid_h + h_lane) * stride_l_h,
            lse_out,
            mask=h_mask,
        )

    # After the stores, never before: this releases either the merge grid or,
    # for direct output, the next PDL-enabled consumer of the final tensor.
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _store_mxfp8_head(x, q8_ptr, sf_ptr, row, head, num_sf_k_tiles, HEAD_DIM: tl.constexpr):
    """Quantize one row's [head * HEAD_DIM, (head + 1) * HEAD_DIM) slice to MXFP8 as
    torch.ops.trtllm.mxfp8_quantize(x, True) does: UE8M0 scale = round-up(amax * rcp(448)), E4M3 RN-satfinite,
    scales in the 128x4-swizzled layout."""
    NUM_SF: tl.constexpr = HEAD_DIM // 32
    xr = tl.reshape(x.to(tl.float32), (NUM_SF, 32))
    amax = tl.max(tl.abs(xr), axis=1)
    sfv = amax * _rcp_approx_ftz(tl.full((NUM_SF,), 448.0, tl.float32))
    bits = sfv.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.minimum(e, 254)
    scale = ((254 - e) << 23).to(tl.float32, bitcast=True)
    scale = tl.where(amax != 0, scale, 0.0)
    q = (xr * scale[:, None]).to(tl.float8e4nv)
    off_q = tl.reshape(tl.arange(0, HEAD_DIM), (NUM_SF, 32))
    tl.store(q8_ptr + row * (num_sf_k_tiles * 128) + head * HEAD_DIM + off_q, q)
    k_sf = head * NUM_SF + tl.arange(0, NUM_SF)
    off_sf = (
        (row // 128) * num_sf_k_tiles * 512
        + (k_sf // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + k_sf % 4
    )
    tl.store(sf_ptr + off_sf, e.to(tl.uint8))


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr = False,
    q8_ptr=None,  # MXFP8 copy of out: [total_q, num_heads * head_dim] E4M3
    sf_ptr=None,  # its UE8M0 block scales, 128x4-swizzled
    num_sf_k_tiles=0,  # ceil(num_heads * head_dim / 128)
    MXFP8_OUT: tl.constexpr = False,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    if USE_PDL:
        if EARLY_TRIGGER:
            # Release the dependent (the o_proj GEMM) before waiting on the attention producer, so its weight
            # prefetch overlaps the attention kernel as well as this merge; its griddepcontrol.wait still waits
            # for this grid (and therefore the producer) to complete. Because this trigger precedes this grid's own
            # wait, it says nothing about what upstream has finished: a dependent may read only step-invariant data
            # (weights) before its griddepcontrol.wait, never this step's attention output, q, top-k table or lengths.
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + off_c[:, None] * stride_o_c
        + off_d[None, :] * stride_o_d,
        mask=off_d[None, :] < head_dim,
        other=0.0,
    ).to(tl.float32)
    # Empty chunks contribute -inf, hence weight 0.
    lse = tl.load(lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c)
    lse_max = tl.max(lse, axis=0)
    # A row whose every chunk is empty (a CUDA-graph padding row) would give
    # -inf - -inf here. Zero it instead: the padded output is discarded, but a
    # NaN would survive into the residual stream and the all-reduce.
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    weights = tl.exp2(lse - lse_max)
    denom = tl.sum(weights, axis=0)
    weights = weights / tl.where(denom > 0, denom, 1.0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)
    if MXFP8_OUT:
        _store_mxfp8_head(
            o_merged.to(out_ptr.dtype.element_ty),
            q8_ptr,
            sf_ptr,
            pid_b,
            pid_h,
            num_sf_k_tiles,
            BLOCK_SIZE_D,
        )

    if USE_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
