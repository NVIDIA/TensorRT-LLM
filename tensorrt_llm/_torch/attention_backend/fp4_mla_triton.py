# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton FP4 MLA decode-path kernels.

This file owns the no-dequant ``triton`` attention backend selected via
``TRTLLM_FLASHINFER_FP4_MLA_ATTENTION_BACKEND=triton``. It is separate from
``fp4_mla_kernels.py``, which holds shared KV-cache scatter/dequant and HP-pool
helper kernels.

Optimizations -- self-contained, public-Triton compatible (no ``tl.ext.*``
or any private-Triton extension):

* ``USE_TMA_DATA_LOAD`` path through ``_fp4_mla_qk_scores_tile`` -- builds
  device-side TMA descriptors via ``tl.make_tensor_descriptor`` for the
  full-K window and the Q residual tail, with the residual-Q permute/split
  idiom that maps the Q tail's interleaved groups onto the K tail.
* ``ASSUME_FULL_HEADS`` / ``ASSUME_FULL_PAGES`` / ``ASSUME_VALID_PAGES``
  constexpr branches that drop ``tl.where`` masks on the hot decode path.
* ``PACK_PROBS`` fused page-stats kernel that quantizes P to FP4 in registers
  before the page-max correction, eliminating the ``p_probs`` HBM round-trip.
* ``_fp4_mla_swizzled_sf_offset_row_block`` faster offset helper for the
  perfect (NUM_HEADS=128, BLOCK_H=128) case used by page-stats and PV.
* ``tl.assume()`` stride hints in front of every ``make_tensor_descriptor``
  call -- helps Triton vectorize TMA loads.
* Pipelined PV loop via ``tl.range(..., num_stages=PV_LOOP_STAGES)``.
"""

import triton
import triton.language as tl

_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fp4_mla_swizzled_sf_offset(
    row_idx,
    col_idx,
    SF_PER_TOKEN: tl.constexpr,
):
    padded_cols = ((SF_PER_TOKEN + 3) // 4) * 4
    col_in_group = col_idx % 4
    col_group = col_idx // 4
    row_in_group0 = row_idx % 32
    row_in_group1 = (row_idx % 128) // 32
    row_group = row_idx // 128
    return (
        col_in_group
        + col_group * (4 * 128)
        + row_in_group0 * 16
        + row_in_group1 * 4
        + row_group * (128 * padded_cols)
    )


@triton.jit
def _fp4_mla_swizzled_sf_offset_row_block(
    row_group, row_offsets, col_idx, SF_PER_TOKEN: tl.constexpr
):
    """Faster offset variant when the row group is a known constant.

    ``row_offsets`` ranges over [0, 128) within the row group; ``row_group``
    is the constant block index. Skips the divmod by 128 that the generic
    helper has to do.
    """
    padded_cols = ((SF_PER_TOKEN + 3) // 4) * 4
    col_part = (col_idx % 4) + (col_idx // 4) * (4 * 128)
    row_part = (row_offsets % 32) * 16 + ((row_offsets % 128) // 32) * 4
    return col_part + row_part + row_group * (128 * padded_cols)


@triton.jit
def _fp4_e2m1_quantize(x):
    abs_x = tl.abs(x)
    magnitude = tl.where(
        abs_x < 0.25,
        0,
        tl.where(
            abs_x < 0.75,
            1,
            tl.where(
                abs_x < 1.25,
                2,
                tl.where(
                    abs_x < 1.75,
                    3,
                    tl.where(abs_x < 2.5, 4, tl.where(abs_x < 3.5, 5, tl.where(abs_x < 5.0, 6, 7))),
                ),
            ),
        ),
    )
    sign = tl.where(x < 0.0, 8, 0)
    return (magnitude | sign).to(tl.uint8)


@triton.jit
def _fp4_pack_low_nibbles(even_packed, odd_packed):
    """PTX helper: pack the low nibbles of two bytes into one byte (low + high<<4)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 lo;
            .reg .b32 hi;
            and.b32 lo, $1, 15;
            and.b32 hi, $2, 15;
            shl.b32 hi, hi, 4;
            or.b32 $0, lo, hi;
        }
        """,
        constraints="=r,r,r",
        args=[even_packed, odd_packed],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_pack_high_nibbles(even_packed, odd_packed):
    """PTX helper: pack the high nibbles of two bytes into one byte (low + high<<4)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 lo;
            .reg .b32 hi;
            shr.u32 lo, $1, 4;
            and.b32 lo, lo, 15;
            and.b32 hi, $2, 240;
            or.b32 $0, lo, hi;
        }
        """,
        constraints="=r,r,r",
        args=[even_packed, odd_packed],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_e2m1_quantize_packed(even, odd):
    """Quantize two FP32 values into a single packed E2M1x2 byte via PTX."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 r;
            cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
            mov.b32 $0, {r, r, r, r};
        }
        """,
        constraints="=r,f,f",
        args=[odd.to(tl.float32), even.to(tl.float32)],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_mla_qk_scores_tile(
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    src_page_ids_ptr,
    compact_page,
    q_row_base,
    head_start,
    head_offsets,
    token_offsets,
    q_num_rows,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    page_ids_len,
    num_pages,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
):
    if ASSUME_VALID_PAGES:
        safe_compact_page = compact_page
        physical_page = tl.load(src_page_ids_ptr + safe_compact_page).to(tl.int64)
        safe_physical_page = physical_page
    else:
        valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
        safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page, mask=valid_compact_page, other=-1
        ).to(tl.int64)
        valid_physical_page = (
            valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
        )
        safe_physical_page = tl.where(valid_physical_page, physical_page, 0)
    q_rows = q_row_base + head_offsets
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_q_rows = q_rows
    else:
        mask_h = head_offsets < NUM_HEADS
        safe_q_rows = tl.where(mask_h, q_rows, q_row_base)
    scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

    packed_k_offsets = tl.arange(0, BLOCK_K // 2)
    scale_offsets = tl.arange(0, BLOCK_K // FP4_BLOCK)
    residual_groups = Q_RESIDUAL_D // FP4_BLOCK
    non_residual_groups = K_HEAD_D // FP4_BLOCK - residual_groups
    if USE_TMA_DATA_LOAD and FULL_BLOCK_END > 0:
        tl.assume(q_fp4_s0 % 8 == 0)
        tl.assume(q_fp4_s1 == 1)
        tl.assume(kv_s0 % 8 == 0)
        tl.assume(kv_s2 % 8 == 0)
        tl.assume(kv_s4 == 1)
        q_desc = tl.make_tensor_descriptor(
            q_fp4_ptr,
            shape=[q_num_rows, Q_HEAD_D // 2],
            strides=[q_fp4_s0, q_fp4_s1],
            block_shape=[BLOCK_H, BLOCK_K // 2],
        )
        k_desc = tl.make_tensor_descriptor(
            kv_cache_ptr,
            shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            block_shape=[1, BLOCK_T, BLOCK_K // 2],
        )
    if USE_TMA_DATA_LOAD and Q_RESIDUAL_D == 64 and TAIL_BLOCK_K == 128:
        q_tail_desc = tl.make_tensor_descriptor(
            q_fp4_ptr,
            shape=[q_num_rows, Q_HEAD_D // 2],
            strides=[q_fp4_s0, q_fp4_s1],
            block_shape=[BLOCK_H, 64],
        )
        k_tail_desc = tl.make_tensor_descriptor(
            kv_cache_ptr,
            shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            block_shape=[1, BLOCK_T, 32],
        )
    # Static Python loop (not tl.range) — Triton unrolls it; this also keeps
    # the Triton 3.6 AutomaticWarpSpecialization / NVWSInsertTmemAref passes
    # from picking up the loop and ICEing on sm_100.
    for q_start in range(0, FULL_BLOCK_END, BLOCK_K):
        q_elem_offsets = q_start + packed_k_offsets * 2
        q_group_offsets = q_elem_offsets // FP4_BLOCK
        k_group_offsets = tl.where(
            q_group_offsets < non_residual_groups,
            q_group_offsets,
            non_residual_groups + (q_group_offsets - non_residual_groups) // 2,
        )
        byte_offsets_in_group = (q_elem_offsets % FP4_BLOCK) // 2
        packed_q_cols = q_start // 2 + packed_k_offsets
        packed_k_cols = k_group_offsets * (FP4_BLOCK // 2) + byte_offsets_in_group
        mask_k = q_elem_offsets < Q_HEAD_D
        safe_packed_q_cols = tl.where(mask_k, packed_q_cols, 0)
        safe_packed_k_cols = tl.where(mask_k, packed_k_cols, 0)
        if (
            USE_TMA_DATA_LOAD
            and FULL_BLOCK_END > 0
            and q_start + BLOCK_K <= non_residual_groups * FP4_BLOCK
        ):
            q_vals = q_desc.load([(q_row_base + head_start).to(tl.int32), q_start // 2])
            k_vals = k_desc.load([safe_physical_page.to(tl.int32), 0, q_start // 2])
            k_vals = tl.reshape(k_vals, (BLOCK_T, BLOCK_K // 2))
            if not ASSUME_VALID_PAGES:
                k_vals = tl.where(valid_physical_page, k_vals, 0)
        else:
            q_vals = tl.load(
                q_fp4_ptr
                + safe_q_rows[:, None] * q_fp4_s0
                + safe_packed_q_cols[None, :] * q_fp4_s1,
                mask=mask_k[None, :] if ASSUME_FULL_HEADS else mask_h[:, None] & mask_k[None, :],
                other=0,
            )
            k_vals = tl.load(
                kv_cache_ptr
                + safe_physical_page * kv_s0
                + token_offsets[:, None].to(tl.int64) * kv_s2
                + safe_packed_k_cols[None, :] * kv_s4,
                mask=mask_k[None, :]
                if ASSUME_VALID_PAGES
                else valid_physical_page & mask_k[None, :],
                other=0,
            )

        q_sf_cols = q_start // FP4_BLOCK + scale_offsets
        k_sf_cols = tl.where(
            q_sf_cols < non_residual_groups,
            q_sf_cols,
            non_residual_groups + (q_sf_cols - non_residual_groups) // 2,
        )
        mask_sf = q_sf_cols < Q_SF_PER_TOKEN
        safe_q_sf_cols = tl.where(mask_sf, q_sf_cols, 0)
        safe_k_sf_cols = tl.where(mask_sf, k_sf_cols, 0)
        q_sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_q_rows[:, None], safe_q_sf_cols[None, :], Q_SF_PER_TOKEN
        )
        k_sf_offsets = _fp4_mla_swizzled_sf_offset(
            token_offsets[:, None], safe_k_sf_cols[None, :], K_SF_PER_TOKEN
        )
        q_scales = tl.load(q_sf_ptr + q_sf_offsets)
        k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets)
        scores = tl.dot_scaled(
            q_vals,
            q_scales,
            "e2m1",
            k_vals.T,
            k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )

    if FULL_BLOCK_END < Q_HEAD_D:
        q_start = FULL_BLOCK_END
        # Residual Q fast path disabled: it issues two chained dot_scaled
        # calls into the same accumulator (one for even Q lane, one for odd),
        # which lowers to a TMEM alloc with multiple uses and trips
        # NVWSInsertTmemAref::hasOneUse() on Triton 3.6.0 / sm_100.
        if False and Q_RESIDUAL_D == 64 and TAIL_BLOCK_K == 128:
            residual_packed_offsets = tl.arange(0, 32)
            residual_scale_offsets = tl.arange(0, 4)
            packed_k_cols = non_residual_groups * (FP4_BLOCK // 2) + residual_packed_offsets
            if USE_TMA_DATA_LOAD:
                k_vals = k_tail_desc.load(
                    [
                        safe_physical_page.to(tl.int32),
                        0,
                        (non_residual_groups * (FP4_BLOCK // 2)).to(tl.int32),
                    ]
                )
                k_vals = tl.reshape(k_vals, (BLOCK_T, 32))
                if not ASSUME_VALID_PAGES:
                    k_vals = tl.where(valid_physical_page, k_vals, 0)
            elif ASSUME_VALID_PAGES:
                k_vals = tl.load(
                    kv_cache_ptr
                    + safe_physical_page * kv_s0
                    + token_offsets[:, None].to(tl.int64) * kv_s2
                    + packed_k_cols[None, :] * kv_s4,
                )
            else:
                k_vals = tl.load(
                    kv_cache_ptr
                    + safe_physical_page * kv_s0
                    + token_offsets[:, None].to(tl.int64) * kv_s2
                    + packed_k_cols[None, :] * kv_s4,
                    mask=valid_physical_page,
                    other=0,
                )
            k_sf_cols = non_residual_groups + residual_scale_offsets
            k_sf_offsets = _fp4_mla_swizzled_sf_offset(
                token_offsets[:, None], k_sf_cols[None, :], K_SF_PER_TOKEN
            )
            k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets)

            q_tail_cols = q_start // 2 + tl.arange(0, 64)
            if USE_TMA_DATA_LOAD and ASSUME_FULL_HEADS:
                q_tail_vals = q_tail_desc.load(
                    [(q_row_base + head_start).to(tl.int32), q_start // 2]
                )
            elif ASSUME_FULL_HEADS:
                q_tail_vals = tl.load(
                    q_fp4_ptr + safe_q_rows[:, None] * q_fp4_s0 + q_tail_cols[None, :] * q_fp4_s1
                )
            else:
                q_tail_vals = tl.load(
                    q_fp4_ptr + safe_q_rows[:, None] * q_fp4_s0 + q_tail_cols[None, :] * q_fp4_s1,
                    mask=mask_h[:, None],
                    other=0,
                )
            # Map Q tail groups [0, 1, ..., 7] onto K tail groups [0, 0, 1, 1, ..., 3, 3].
            q_tail_vals = q_tail_vals.reshape([BLOCK_H, 4, 2, 8]).trans(0, 1, 3, 2)
            q_even_vals, q_odd_vals = tl.split(q_tail_vals)
            q_even_vals = q_even_vals.reshape([BLOCK_H, 32])
            q_odd_vals = q_odd_vals.reshape([BLOCK_H, 32])

            q_tail_sf_cols = q_start // FP4_BLOCK + tl.arange(0, 8)
            q_tail_sf_offsets = _fp4_mla_swizzled_sf_offset(
                safe_q_rows[:, None], q_tail_sf_cols[None, :], Q_SF_PER_TOKEN
            )
            q_tail_scales = tl.load(q_sf_ptr + q_tail_sf_offsets)
            q_tail_scales = q_tail_scales.reshape([BLOCK_H, 4, 2])
            q_even_scales, q_odd_scales = tl.split(q_tail_scales)
            # Compute even/odd partial dots into fresh accumulators, then add
            # back. Routing through `acc=scores` for both calls gives a single
            # TMEM alloc with multiple uses, which trips
            # NVWSInsertTmemAref::TmemAccessDag::build's hasOneUse() assertion
            # on Triton 3.6.0 / sm_100.
            tail_even = tl.dot_scaled(
                q_even_vals,
                q_even_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                fast_math=True,
                rhs_k_pack=True,
            )
            tail_odd = tl.dot_scaled(
                q_odd_vals,
                q_odd_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                fast_math=True,
                rhs_k_pack=True,
            )
            scores = scores + tail_even + tail_odd
        else:
            tail_packed_offsets = tl.arange(0, TAIL_BLOCK_K // 2)
            tail_scale_offsets = tl.arange(0, TAIL_BLOCK_K // FP4_BLOCK)
            q_elem_offsets = q_start + tail_packed_offsets * 2
            q_group_offsets = q_elem_offsets // FP4_BLOCK
            k_group_offsets = tl.where(
                q_group_offsets < non_residual_groups,
                q_group_offsets,
                non_residual_groups + (q_group_offsets - non_residual_groups) // 2,
            )
            byte_offsets_in_group = (q_elem_offsets % FP4_BLOCK) // 2
            packed_q_cols = q_start // 2 + tail_packed_offsets
            packed_k_cols = k_group_offsets * (FP4_BLOCK // 2) + byte_offsets_in_group
            mask_k = q_elem_offsets < Q_HEAD_D
            safe_packed_q_cols = tl.where(mask_k, packed_q_cols, 0)
            safe_packed_k_cols = tl.where(mask_k, packed_k_cols, 0)
            q_vals = tl.load(
                q_fp4_ptr
                + safe_q_rows[:, None] * q_fp4_s0
                + safe_packed_q_cols[None, :] * q_fp4_s1,
                mask=mask_k[None, :] if ASSUME_FULL_HEADS else mask_h[:, None] & mask_k[None, :],
                other=0,
            )
            k_vals = tl.load(
                kv_cache_ptr
                + safe_physical_page * kv_s0
                + token_offsets[:, None].to(tl.int64) * kv_s2
                + safe_packed_k_cols[None, :] * kv_s4,
                mask=mask_k[None, :]
                if ASSUME_VALID_PAGES
                else valid_physical_page & mask_k[None, :],
                other=0,
            )

            q_sf_cols = q_start // FP4_BLOCK + tail_scale_offsets
            k_sf_cols = tl.where(
                q_sf_cols < non_residual_groups,
                q_sf_cols,
                non_residual_groups + (q_sf_cols - non_residual_groups) // 2,
            )
            mask_sf = q_sf_cols < Q_SF_PER_TOKEN
            safe_q_sf_cols = tl.where(mask_sf, q_sf_cols, 0)
            safe_k_sf_cols = tl.where(mask_sf, k_sf_cols, 0)
            q_sf_offsets = _fp4_mla_swizzled_sf_offset(
                safe_q_rows[:, None], safe_q_sf_cols[None, :], Q_SF_PER_TOKEN
            )
            k_sf_offsets = _fp4_mla_swizzled_sf_offset(
                token_offsets[:, None], safe_k_sf_cols[None, :], K_SF_PER_TOKEN
            )
            q_scales = tl.load(q_sf_ptr + q_sf_offsets)
            k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets)
            scores = tl.dot_scaled(
                q_vals,
                q_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )

    return scores


@triton.jit
def _fp4_mla_attention_page_stats_kernel(
    page_max_ptr,
    page_sum_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    q_global_scale_ptr,
    page_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    num_pages,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    page_stats_s0,
    page_stats_s1,
    p_s0,
    p_s1,
    p_num_rows,
    q_num_rows,
    sm_scale,
    local_layer,
    pscale_s0,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    PACK_PROBS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    USE_PER_PAGE_SCALE: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    """Page-stats fused QK + softmax-stats + (optional) FP4 P pack.

    Each program owns one (query, head_block, page). Probs are quantized
    against the per-page max; the page-max correction
    ``exp(page_max - global_max) / denom`` is folded into ``p_sf`` later by
    ``_fp4_mla_attention_prob_scale_kernel``.
    """
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = query_idx * NUM_HEADS
    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_start = page_rel * PAGE_SIZE
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    out_offsets = query_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h

    page_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    page_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if USE_TMA_DATA_LOAD and PACK_PROBS and ASSUME_FULL_HEADS and ASSUME_VALID_PAGES:
        tl.assume(p_s0 % 8 == 0)
        tl.assume(p_s1 == 1)
        p_desc = tl.make_tensor_descriptor(
            p_fp4_ptr,
            shape=[p_num_rows, PAGE_SIZE // 2],
            strides=[p_s0, p_s1],
            block_shape=[BLOCK_H, PAGE_SIZE // 2],
        )
    if ASSUME_FULL_PAGES or page_start < kv_len:
        scores = _fp4_mla_qk_scores_tile(
            q_fp4_ptr,
            q_sf_ptr,
            kv_cache_ptr,
            sf_cache_ptr,
            src_page_ids_ptr,
            page_table_start + page_rel,
            q_row_base,
            head_block * BLOCK_H,
            offs_h,
            offs_t,
            q_num_rows,
            q_fp4_s0,
            q_fp4_s1,
            kv_s0,
            kv_s2,
            kv_s4,
            sf_s0,
            page_ids_len,
            num_pages,
            Q_HEAD_D,
            K_HEAD_D,
            Q_RESIDUAL_D,
            FP4_BLOCK,
            Q_SF_PER_TOKEN,
            K_SF_PER_TOKEN,
            BLOCK_H,
            BLOCK_T,
            BLOCK_K,
            FULL_BLOCK_END,
            TAIL_BLOCK_K,
            NUM_HEADS,
            USE_TMA_DATA_LOAD,
            ASSUME_FULL_HEADS,
            ASSUME_VALID_PAGES,
        )
        if ASSUME_FULL_PAGES:
            valid_t = tl.full([BLOCK_T], True, dtype=tl.int1)
        else:
            valid_t = page_start + offs_t < kv_len
        global_scale = tl.load(global_scale_ptr)
        if USE_PER_PAGE_SCALE:
            # Independent dynamic Q scale and per-page (K/V shared) KV scale.
            # page_gscale also folds into the stored P scale below so the per-page
            # V scaling cancels inside the fused PV dot.
            compact_page = page_table_start + page_rel
            if ASSUME_VALID_PAGES:
                phys_page = tl.load(src_page_ids_ptr + compact_page).to(tl.int64)
            else:
                valid_cp = (compact_page >= 0) & (compact_page < page_ids_len)
                phys_page = tl.load(
                    src_page_ids_ptr + tl.where(valid_cp, compact_page, 0),
                    mask=valid_cp,
                    other=0,
                ).to(tl.int64)
                phys_page = tl.where((phys_page >= 0) & (phys_page < num_pages), phys_page, 0)
            page_gscale = tl.load(page_scale_ptr + local_layer * pscale_s0 + phys_page)
            q_gscale = tl.load(q_global_scale_ptr)
            qk_scale = sm_scale / (q_gscale * page_gscale)
        else:
            # Static scale: global_scale == 1.0, so page_gscale == 1.0 makes the
            # stored-P fold below a no-op and qk_scale == sm_scale.
            page_gscale = global_scale
            qk_scale = sm_scale / (global_scale * global_scale)
        if ASSUME_FULL_HEADS and ASSUME_FULL_PAGES:
            scores = scores * qk_scale
            page_max = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max[:, None]) * _LOG2_E)
            page_sum = tl.sum(exp_scores, axis=1)
        else:
            scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
            page_max = tl.max(scores, axis=1)
            safe_page_max = tl.where(mask_h, page_max, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max[:, None]) * _LOG2_E)
            exp_scores = tl.where(mask_h[:, None] & valid_t[None, :], exp_scores, 0.0)
            page_sum = tl.sum(exp_scores, axis=1)

        if PACK_PROBS:
            grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
            amax = tl.max(grouped_probs, axis=2)
            inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
            # Fold 1/page_gscale into the stored P block scale (page_gscale == 1.0
            # in the static path, so this is a no-op there). Combined with V's
            # baked page_gscale, the page scale cancels in the PV dot and the end
            # out_scale = 1/(global_scale*P_GLOBAL_SCALE) = 1/P_GLOBAL_SCALE stays.
            stored_scale = tl.where(
                amax > 0.0,
                tl.minimum(amax * (P_GLOBAL_SCALE / 6.0) / page_gscale, 448.0),
                1.0,
            )
            scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
            pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
            even_probs, odd_probs = tl.split(pairs)
            packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

            if not ASSUME_VALID_PAGES:
                valid_compact_page = (page_table_start + page_rel >= 0) & (
                    page_table_start + page_rel < page_ids_len
                )
            p_page = query_idx * MAX_PAGES + page_rel
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            ).to(tl.int64)
            scale_cols = tl.arange(0, SF_PER_PAGE)
            if ASSUME_FULL_HEADS and ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
                sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
                    p_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
                )
            else:
                sf_offsets = _fp4_mla_swizzled_sf_offset(
                    safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
                )
            if ASSUME_FULL_HEADS:
                if ASSUME_VALID_PAGES:
                    tl.store(p_sf_ptr + sf_offsets, stored_scale)
                else:
                    tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=valid_compact_page)
            else:
                tl.store(
                    p_sf_ptr + sf_offsets,
                    stored_scale,
                    mask=mask_h[:, None]
                    if ASSUME_VALID_PAGES
                    else valid_compact_page & mask_h[:, None],
                )

            byte_offsets = tl.arange(0, FP4_BLOCK // 2)
            byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
            if USE_TMA_DATA_LOAD and ASSUME_FULL_HEADS and ASSUME_VALID_PAGES:
                p_desc.store(
                    [(p_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
                    tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
                )
            elif ASSUME_FULL_HEADS:
                if ASSUME_VALID_PAGES:
                    tl.store(
                        p_fp4_ptr
                        + safe_p_rows[:, None, None] * p_s0
                        + byte_cols[None, :, :] * p_s1,
                        packed,
                    )
                else:
                    tl.store(
                        p_fp4_ptr
                        + safe_p_rows[:, None, None] * p_s0
                        + byte_cols[None, :, :] * p_s1,
                        packed,
                        mask=valid_compact_page,
                    )
            else:
                tl.store(
                    p_fp4_ptr + safe_p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                    packed,
                    mask=mask_h[:, None, None]
                    if ASSUME_VALID_PAGES
                    else valid_compact_page & mask_h[:, None, None],
                )

    if ASSUME_FULL_HEADS:
        tl.store(page_max_ptr + out_offsets, page_max)
        tl.store(page_sum_ptr + out_offsets, page_sum)
    else:
        tl.store(page_max_ptr + out_offsets, page_max, mask=mask_h)
        tl.store(page_sum_ptr + out_offsets, page_sum, mask=mask_h)


@triton.jit
def _fp4_mla_attention_reduce_stats_kernel(
    max_ptr,
    denom_ptr,
    page_max_ptr,
    page_sum_ptr,
    num_pages,
    stats_s0,
    page_stats_s0,
    page_stats_s1,
    NUM_HEADS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    """Combine per-page max/sum into a global max + denom per (query, head).

    Keep the page dimension in a loop instead of a 2D [pages, heads] vector.
    Large decode batches can push max pages above 256, where the bulk vector
    form becomes too large for a single Triton program.
    """
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)

    max_score = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    for page_idx in tl.range(0, MAX_PAGES):
        page_valid = page_idx < num_pages
        page_offsets = gen_idx * page_stats_s0 + page_idx * page_stats_s1 + safe_offs_h
        page_max = tl.load(
            page_max_ptr + page_offsets, mask=mask_h & page_valid, other=-float("inf")
        )
        max_score = tl.maximum(max_score, page_max)

    safe_max = tl.where(max_score > -float("inf"), max_score, 0.0)
    denom = tl.zeros((BLOCK_H,), tl.float32)
    for page_idx in tl.range(0, MAX_PAGES):
        page_valid = page_idx < num_pages
        page_offsets = gen_idx * page_stats_s0 + page_idx * page_stats_s1 + safe_offs_h
        page_max = tl.load(
            page_max_ptr + page_offsets, mask=mask_h & page_valid, other=-float("inf")
        )
        page_sum = tl.load(page_sum_ptr + page_offsets, mask=mask_h & page_valid, other=0.0)
        weights = tl.math.exp2((page_max - safe_max) * _LOG2_E)
        denom += tl.where(page_sum > 0.0, page_sum * weights, 0.0)

    tl.store(max_ptr + gen_idx * stats_s0 + safe_offs_h, max_score, mask=mask_h)
    tl.store(denom_ptr + gen_idx * stats_s0 + safe_offs_h, denom, mask=mask_h)


@triton.jit
def _fp4_mla_attention_prob_scale_kernel(
    p_sf_ptr,
    max_ptr,
    denom_ptr,
    page_max_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    stats_s0,
    page_stats_s0,
    page_stats_s1,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    """Apply per-page softmax correction by scaling p_sf in place."""
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    page_start = page_rel * PAGE_SIZE
    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
        if page_start >= kv_len:
            return

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if not ASSUME_VALID_PAGES:
        if (compact_page < 0) | (compact_page >= page_ids_len):
            return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    page_max = tl.load(
        page_max_ptr + query_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
        mask=mask_h,
        other=-float("inf"),
    )
    max_score = tl.load(max_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    factor = tl.where(denom > 0.0, tl.math.exp2((page_max - max_score) * _LOG2_E) / denom, 0.0)

    p_page = query_idx * MAX_PAGES + page_rel
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = (
        p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
    ).to(tl.int64)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    if ASSUME_FULL_HEADS and ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
        sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            p_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    else:
        sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    scales = tl.load(p_sf_ptr + sf_offsets, mask=mask_h[:, None], other=1.0).to(tl.float32)
    tl.store(p_sf_ptr + sf_offsets, scales * factor[:, None], mask=mask_h[:, None])


@triton.jit
def _fp4_mla_attention_pv_kernel(
    out_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    kv_cache_ptr,
    v_sf_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    num_pages,
    out_s0,
    out_s1,
    out_s2,
    out_num_rows,
    p_s0,
    p_s1,
    p_num_rows,
    kv_s0,
    kv_s2,
    kv_s4,
    vsf_s0,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_TMA_P_LOAD: tl.constexpr,
    USE_TMA_V_LOAD: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_FULL_V: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    PAGE_SPLIT: tl.constexpr = 1,
    PAGES_PER_SPLIT: tl.constexpr = 0,
    PARTIAL_OUT: tl.constexpr = False,
    partial_out_ptr=None,
    partial_s0: tl.constexpr = 0,
    partial_s1: tl.constexpr = 0,
    partial_s2: tl.constexpr = 0,
    partial_s3: tl.constexpr = 0,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    # When PAGE_SPLIT > 1 we encode (dim_block, split_idx) into program_id(2).
    # The outer loop over pages is partitioned across split_idx programs so the
    # grid grows by PAGE_SPLIT× — this lifts the bs<=32 PV grid out of the
    # 0.5-wave-per-SM regime that the ncu report flagged.
    prog2 = tl.program_id(2)
    if PAGE_SPLIT > 1:
        dim_block = prog2 // PAGE_SPLIT
        split_idx = prog2 - dim_block * PAGE_SPLIT
    else:
        dim_block = prog2
        split_idx = 0
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    if ASSUME_FULL_V:
        mask_v = tl.full([BLOCK_V], True, dtype=tl.int1)
        safe_offs_v = offs_v
    else:
        mask_v = offs_v < V_HEAD_D
        safe_offs_v = tl.where(mask_v, offs_v, 0)
    packed_t = tl.arange(0, PAGE_SIZE // 2)
    scale_cols = tl.arange(0, PAGE_SIZE // FP4_BLOCK)
    even_t = packed_t * 2
    odd_t = even_t + 1
    v_packed_offsets = safe_offs_v // 2
    v_use_high_nibble = (safe_offs_v & 1) != 0
    if ASSUME_FULL_V and BLOCK_V == 128:
        v_sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            dim_block, offs_v[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    else:
        v_sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_offs_v[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    if USE_TMA_P_LOAD:
        tl.assume(p_s0 % 8 == 0)
        tl.assume(p_s1 == 1)
        p_desc = tl.make_tensor_descriptor(
            p_fp4_ptr,
            shape=[p_num_rows, PAGE_SIZE // 2],
            strides=[p_s0, p_s1],
            block_shape=[BLOCK_H, PAGE_SIZE // 2],
        )
    if USE_TMA_V_LOAD and ASSUME_FULL_HEADS and ASSUME_FULL_V:
        tl.assume(out_s1 % 8 == 0)
        tl.assume(out_s2 == 1)
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[out_num_rows, V_HEAD_D],
            strides=[out_s1, out_s2],
            block_shape=[BLOCK_H, BLOCK_V],
        )
    if USE_TMA_V_LOAD:
        tl.assume(kv_s0 % 8 == 0)
        tl.assume(kv_s2 % 8 == 0)
        tl.assume(kv_s4 == 1)
        v_desc = tl.make_tensor_descriptor(
            kv_cache_ptr,
            shape=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            block_shape=[1, PAGE_SIZE, BLOCK_V // 2],
        )

    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    if PAGE_SPLIT > 1:
        page_lo = split_idx * PAGES_PER_SPLIT
        page_hi = tl.minimum(page_lo + PAGES_PER_SPLIT, MAX_PAGES)
    else:
        page_lo = 0
        page_hi = MAX_PAGES
    for page_rel in tl.range(page_lo, page_hi, num_stages=PV_LOOP_STAGES):
        page_start = page_rel * PAGE_SIZE
        if ASSUME_FULL_PAGES or page_start < kv_len:
            compact_page = page_table_start + page_rel
            if ASSUME_VALID_PAGES:
                safe_compact_page = compact_page
                physical_page = tl.load(src_page_ids_ptr + safe_compact_page).to(tl.int64)
                safe_physical_page = physical_page
            else:
                valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
                safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
                physical_page = tl.load(
                    src_page_ids_ptr + safe_compact_page, mask=valid_compact_page, other=-1
                ).to(tl.int64)
                valid_physical_page = (
                    valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
                )
                safe_physical_page = tl.where(valid_physical_page, physical_page, 0)

            p_page = query_idx * MAX_PAGES + page_rel
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            ).to(tl.int64)
            if USE_TMA_P_LOAD:
                p_vals = p_desc.load([(p_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0])
            else:
                p_vals = tl.load(
                    p_fp4_ptr + safe_p_rows[:, None] * p_s0 + packed_t[None, :] * p_s1,
                    mask=mask_h[:, None]
                    if ASSUME_VALID_PAGES
                    else valid_compact_page & mask_h[:, None],
                    other=0,
                )
            p_sf_offsets = _fp4_mla_swizzled_sf_offset(
                safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
            )
            p_scales = tl.load(p_sf_ptr + p_sf_offsets)

            if ASSUME_FULL_PAGES:
                valid_even_t = tl.full([PAGE_SIZE // 2], True, dtype=tl.int1)
                valid_odd_t = tl.full([PAGE_SIZE // 2], True, dtype=tl.int1)
            else:
                valid_even_t = page_start + even_t < kv_len
                valid_odd_t = page_start + odd_t < kv_len
            if USE_TMA_V_LOAD:
                v_tile = v_desc.load(
                    [
                        safe_physical_page.to(tl.int32),
                        0,
                        (dim_block * (BLOCK_V // 2)).to(tl.int32),
                    ]
                )
                v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
                if not ASSUME_VALID_PAGES:
                    v_tile = tl.where(valid_physical_page, v_tile, 0)
                v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
                even_packed, odd_packed = tl.split(v_pairs)
                if not ASSUME_FULL_PAGES:
                    even_packed = tl.where(valid_even_t[None, :], even_packed, 0)
                    odd_packed = tl.where(valid_odd_t[None, :], odd_packed, 0)
                low_vals = _fp4_pack_low_nibbles(even_packed, odd_packed)
                high_vals = _fp4_pack_high_nibbles(even_packed, odd_packed)
                v_vals = tl.reshape(
                    tl.join(low_vals, high_vals).permute(0, 2, 1),
                    (BLOCK_V, PAGE_SIZE // 2),
                )
            else:
                even_packed = tl.load(
                    kv_cache_ptr
                    + safe_physical_page * kv_s0
                    + even_t[None, :].to(tl.int64) * kv_s2
                    + v_packed_offsets[:, None] * kv_s4,
                    mask=(mask_v[:, None] & valid_even_t[None, :])
                    if ASSUME_VALID_PAGES
                    else (valid_physical_page & mask_v[:, None] & valid_even_t[None, :]),
                    other=0,
                )
                odd_packed = tl.load(
                    kv_cache_ptr
                    + safe_physical_page * kv_s0
                    + odd_t[None, :].to(tl.int64) * kv_s2
                    + v_packed_offsets[:, None] * kv_s4,
                    mask=(mask_v[:, None] & valid_odd_t[None, :])
                    if ASSUME_VALID_PAGES
                    else (valid_physical_page & mask_v[:, None] & valid_odd_t[None, :]),
                    other=0,
                )
                even_low = even_packed & 0x0F
                even_high = (even_packed >> 4) & 0x0F
                odd_low = odd_packed & 0x0F
                even_nibble = tl.where(v_use_high_nibble[:, None], even_high, even_low)
                odd_nibble = tl.where(v_use_high_nibble[:, None], odd_packed >> 4, odd_low)
                v_vals = even_nibble | (odd_nibble << 4)
            v_scales = tl.load(v_sf_ptr + safe_physical_page * vsf_s0 + v_sf_offsets)
            acc = tl.dot_scaled(
                p_vals,
                p_scales,
                "e2m1",
                v_vals.T,
                v_scales,
                "e2m1",
                acc=acc,
                fast_math=True,
                rhs_k_pack=True,
            )

    if PARTIAL_OUT:
        # Write the unscaled partial accumulator to a float32 workspace; the
        # reduce-PV kernel sums splits and applies out_scale + dtype cast.
        # Layout: partial_out[query_idx, split_idx, head_offset, v_offset].
        base = (
            query_idx * partial_s0
            + split_idx * partial_s1
            + safe_offs_h[:, None] * partial_s2
            + safe_offs_v[None, :] * partial_s3
        )
        if ASSUME_FULL_HEADS and ASSUME_FULL_V:
            tl.store(partial_out_ptr + base, acc)
        else:
            tl.store(partial_out_ptr + base, acc, mask=mask_h[:, None] & mask_v[None, :])
    elif ASSUME_FULL_HEADS and ASSUME_FULL_V:
        out_vals = acc * out_scale
        if USE_TMA_V_LOAD:
            if out_ptr.dtype.element_ty == tl.bfloat16:
                out_vals = out_vals.to(tl.bfloat16)
            elif out_ptr.dtype.element_ty == tl.float16:
                out_vals = out_vals.to(tl.float16)
            out_desc.store(
                [
                    (query_idx * NUM_HEADS + head_block * BLOCK_H).to(tl.int32),
                    (dim_block * BLOCK_V).to(tl.int32),
                ],
                out_vals,
            )
        else:
            tl.store(
                out_ptr + query_idx * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
                out_vals,
            )
    else:
        tl.store(
            out_ptr
            + query_idx * out_s0
            + safe_offs_h[:, None] * out_s1
            + safe_offs_v[None, :] * out_s2,
            acc * out_scale,
            mask=mask_h[:, None] & mask_v[None, :],
        )


@triton.jit
def _fp4_mla_attention_pv_reduce_kernel(
    out_ptr,
    partial_ptr,
    global_scale_ptr,
    out_s0,
    out_s1,
    out_s2,
    partial_s0,
    partial_s1,
    partial_s2,
    partial_s3,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SPLIT: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_V: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    if ASSUME_FULL_V:
        mask_v = tl.full([BLOCK_V], True, dtype=tl.int1)
        safe_offs_v = offs_v
    else:
        mask_v = offs_v < V_HEAD_D
        safe_offs_v = tl.where(mask_v, offs_v, 0)
    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for split_idx in tl.static_range(0, PAGE_SPLIT):
        base = (
            query_idx * partial_s0
            + split_idx * partial_s1
            + safe_offs_h[:, None] * partial_s2
            + safe_offs_v[None, :] * partial_s3
        )
        acc += tl.load(partial_ptr + base, mask=mask_h[:, None] & mask_v[None, :], other=0.0)
    out_vals = acc * out_scale
    if out_ptr.dtype.element_ty == tl.bfloat16:
        out_vals = out_vals.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float16:
        out_vals = out_vals.to(tl.float16)
    tl.store(
        out_ptr
        + query_idx * out_s0
        + safe_offs_h[:, None] * out_s1
        + safe_offs_v[None, :] * out_s2,
        out_vals,
        mask=mask_h[:, None] & mask_v[None, :],
    )


# ---------------------------------------------------------------------------
# Per-page dynamic-scale store path (triton only)
#
# Two passes per decode step over the *active* page (the page that holds the
# current step's new tokens):
#   Pass A (_fp4_mla_page_scale_gen_kernel): compute the page amax over the
#     shared K/V latent (old tokens read from the BF16 staging pool, new tokens
#     from latent_cache) and write page_gscale = P_GLOBAL_SCALE / page_amax into
#     the [num_layers, num_pages] fp32 page-scale pool.
#   Pass B (_fp4_mla_page_requant_gen_kernel): re-quantize *every* FP4 tile of
#     the active page from the same (staging + latent) source, baking page_gscale
#     into the stored K and V block scales. Completed pages are never revisited,
#     so their scale is frozen at the value computed on the step that filled them.
#
# Both passes assume the step's new tokens land in a single page (always true
# for 1-token decode; the Python dispatch guards the MTP boundary-cross case).
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_mla_page_scale_gen_kernel(
    page_scale_ptr,
    stage_pool_ptr,
    latent_cache_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    num_seq_slots,
    num_pages,
    num_layers,
    local_layer,
    page_size,
    pscale_s0,
    pool_s0,
    pool_s1,
    lc_s0,
    lc_s1,
    HEAD_D: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    PAGE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return

    kv_len = tl.load(kv_lens_ptr + seq_idx)
    gen_len = tl.load(prompt_lens_ptr + seq_idx)
    if (kv_len <= 0) | (gen_len <= 0):
        return
    first_new_pos = kv_len - gen_len
    active_page = (kv_len - 1) // page_size
    # New tokens must all land in the active page (boundary-cross guarded by
    # the Python dispatch; bail defensively here too).
    if first_new_pos // page_size != active_page:
        return
    page_pos_start = active_page * page_size
    fill = kv_len - page_pos_start

    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
    physical_page_offset = page_start + active_page
    if (
        (physical_page_offset < page_start)
        | (physical_page_offset >= page_end)
        | (physical_page_offset < 0)
        | (physical_page_offset >= page_ids_len)
    ):
        return
    physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
    if (physical_page < 0) | (physical_page >= num_pages):
        return
    seq_slot = tl.load(seq_slots_ptr + seq_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_D
    safe_d = tl.where(mask_d, offs_d, 0)
    amax = 0.0
    for tile_idx in tl.range(0, PAGE_SLOTS // FP4_BLOCK):
        token_offsets = tile_idx * FP4_BLOCK + tl.arange(0, FP4_BLOCK)
        abs_pos = page_pos_start + token_offsets
        valid = token_offsets < fill
        from_latent = abs_pos >= first_new_pos
        slot = abs_pos % PAGE_SLOTS
        stage_vals = tl.load(
            stage_pool_ptr
            + seq_slot * pool_s0
            + local_layer * pool_s1
            + slot[:, None] * POOL_HEAD_D
            + safe_d[None, :],
            mask=valid[:, None] & (~from_latent)[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        latent_tok = seq_idx * gen_len + (abs_pos - first_new_pos)
        safe_latent = tl.where(valid & from_latent, latent_tok, 0).to(tl.int64)
        latent_vals = tl.load(
            latent_cache_ptr + safe_latent[:, None] * lc_s0 + safe_d[None, :] * lc_s1,
            mask=valid[:, None] & from_latent[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        vals = stage_vals + latent_vals
        amax = tl.maximum(amax, tl.max(tl.abs(vals)))

    gscale = tl.where(amax > 0.0, P_GLOBAL_SCALE / amax, 1.0)
    tl.store(page_scale_ptr + local_layer * pscale_s0 + physical_page, gscale)


@triton.jit
def _fp4_mla_page_requant_gen_kernel(
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    stage_pool_ptr,
    latent_cache_ptr,
    page_scale_ptr,
    seq_slots_ptr,
    kv_lens_ptr,
    prompt_lens_ptr,
    page_ids_ptr,
    paged_kv_indptr_ptr,
    page_ids_len,
    indptr_len,
    num_seq_slots,
    num_pages,
    num_layers,
    local_layer,
    page_size,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    pool_s0,
    pool_s1,
    lc_s0,
    lc_s1,
    vsf_s0,
    vsf_s1,
    pscale_s0,
    HEAD_D: tl.constexpr,
    POOL_HEAD_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SLOTS: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    if (local_layer < 0) | (local_layer >= num_layers):
        return
    if seq_idx + 1 >= indptr_len:
        return

    kv_len = tl.load(kv_lens_ptr + seq_idx)
    gen_len = tl.load(prompt_lens_ptr + seq_idx)
    if gen_len <= 0:
        return
    first_new_pos = kv_len - gen_len
    active_page = (kv_len - 1) // page_size
    if first_new_pos // page_size != active_page:
        return
    # Re-quantize every tile of the active page (page_gscale changed), not just
    # the tiles that the new tokens touch.
    block_base_pos = active_page * page_size + tile_idx * FP4_BLOCK
    if block_base_pos >= kv_len:
        return

    page_pos = block_base_pos - active_page * page_size
    page_start = tl.load(paged_kv_indptr_ptr + seq_idx).to(tl.int64)
    page_end = tl.load(paged_kv_indptr_ptr + seq_idx + 1).to(tl.int64)
    physical_page_offset = page_start + active_page
    if (
        (page_pos < 0)
        | (page_pos >= page_size)
        | (physical_page_offset < page_start)
        | (physical_page_offset >= page_end)
        | (physical_page_offset < 0)
        | (physical_page_offset >= page_ids_len)
    ):
        return
    physical_page = tl.load(page_ids_ptr + physical_page_offset).to(tl.int64)
    if (physical_page < 0) | (physical_page >= num_pages):
        return
    seq_slot = tl.load(seq_slots_ptr + seq_idx).to(tl.int64)
    if (seq_slot < 0) | (seq_slot >= num_seq_slots):
        return

    page_gscale = tl.load(page_scale_ptr + local_layer * pscale_s0 + physical_page)

    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    token_offsets = tl.arange(0, FP4_BLOCK)
    even_d = dim_block * FP4_BLOCK + byte_offsets * 2
    odd_d = even_d + 1
    all_d = dim_block * FP4_BLOCK + tl.arange(0, FP4_BLOCK)
    mask_even_d = even_d < HEAD_D
    mask_odd_d = odd_d < HEAD_D
    mask_all_d = all_d < HEAD_D
    safe_even_d = tl.where(mask_even_d, even_d, 0)
    safe_odd_d = tl.where(mask_odd_d, odd_d, 0)
    safe_all_d = tl.where(mask_all_d, all_d, 0)

    abs_positions = block_base_pos + token_offsets
    valid_tokens = abs_positions < kv_len
    from_latent = abs_positions >= first_new_pos
    slot = abs_positions % PAGE_SLOTS
    new_token_offsets = abs_positions - first_new_pos
    latent_tokens = seq_idx * gen_len + new_token_offsets
    safe_latent_tokens = tl.where(valid_tokens & from_latent, latent_tokens, 0).to(tl.int64)

    stage_even = tl.load(
        stage_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + slot[:, None] * POOL_HEAD_D
        + safe_even_d[None, :],
        mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    stage_odd = tl.load(
        stage_pool_ptr
        + seq_slot * pool_s0
        + local_layer * pool_s1
        + slot[:, None] * POOL_HEAD_D
        + safe_odd_d[None, :],
        mask=valid_tokens[:, None] & (~from_latent)[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    latent_even = tl.load(
        latent_cache_ptr + safe_latent_tokens[:, None] * lc_s0 + safe_even_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & from_latent[:, None] & mask_even_d[None, :],
        other=0.0,
    ).to(tl.float32)
    latent_odd = tl.load(
        latent_cache_ptr + safe_latent_tokens[:, None] * lc_s0 + safe_odd_d[None, :] * lc_s1,
        mask=valid_tokens[:, None] & from_latent[:, None] & mask_odd_d[None, :],
        other=0.0,
    ).to(tl.float32)
    even_values = stage_even + latent_even
    odd_values = stage_odd + latent_odd

    amax_per_token = tl.maximum(
        tl.max(tl.abs(even_values), axis=1),
        tl.max(tl.abs(odd_values), axis=1),
    )
    tile_amax = tl.max(amax_per_token, axis=0)
    # K consumes scales as [token, dim-block], V as [dim, token-block]. Only the
    # compressed-KV prefix has both views; tail K-only dims keep K's per-token
    # scale. page_gscale is the shared per-page global scale.
    shared_tile = dim_block * FP4_BLOCK < V_HEAD_D
    tile_scale = tl.where(tile_amax > 0.0, tile_amax / 6.0, 1.0)
    token_scale = tl.where(amax_per_token > 0.0, amax_per_token / 6.0, 1.0)
    local_scale = tl.where(shared_tile, tile_scale, token_scale)
    stored_scale = local_scale * page_gscale
    v_stored_scale = tile_scale * page_gscale

    low = _fp4_e2m1_quantize(even_values / local_scale[:, None])
    high = _fp4_e2m1_quantize(odd_values / local_scale[:, None])
    packed = low | (high << 4)

    packed_cols = dim_block * (FP4_BLOCK // 2) + byte_offsets
    page_positions = page_pos + token_offsets
    kv_base = physical_page * kv_s0
    tl.store(
        kv_cache_ptr + kv_base + page_positions[:, None] * kv_s2 + packed_cols[None, :] * kv_s4,
        packed,
        mask=valid_tokens[:, None] & mask_even_d[None, :],
    )

    k_sf_offsets = _fp4_mla_swizzled_sf_offset(page_positions, dim_block, SF_PER_TOKEN)
    tl.store(sf_cache_ptr + physical_page * sf_s0 + k_sf_offsets, stored_scale, mask=valid_tokens)

    token_scale_col = page_pos // FP4_BLOCK
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_all_d, token_scale_col, SF_PER_PAGE)
    v_sf_base = tl.cast(local_layer, tl.int64) * tl.cast(
        vsf_s0, tl.int64
    ) + physical_page * tl.cast(vsf_s1, tl.int64)
    tl.store(
        v_sf_ptr + v_sf_base + sf_offsets.to(tl.int64),
        v_stored_scale,
        mask=mask_all_d & (all_d < V_HEAD_D),
    )
