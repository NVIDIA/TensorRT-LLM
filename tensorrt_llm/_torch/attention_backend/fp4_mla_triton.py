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
* Optional prepacked-V PV path using ``tl.make_tensor_descriptor`` only. It
  stores V as ``[page, dim-block, v, packed-token-pair]`` so PV can skip the
  per-query V transpose/repack inside the page loop.
* Pipelined PV loop via ``tl.range(..., num_stages=PV_LOOP_STAGES)``.
"""

from typing import Any, Optional

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
def _fp4_mla_attention_v_repack_kernel(
    v_packed_ptr,
    kv_cache_ptr,
    num_pages,
    kv_s0,
    kv_s2,
    kv_s4,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    page_idx = tl.program_id(0)
    dim_block = tl.program_id(1)

    tl.assume(kv_s0 % 8 == 0)
    tl.assume(kv_s2 % 8 == 0)
    tl.assume(kv_s4 == 1)
    v_desc = tl.make_tensor_descriptor(
        kv_cache_ptr,
        shape=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        block_shape=[1, PAGE_SIZE, BLOCK_V // 2],
    )
    v_tile = v_desc.load(
        [
            page_idx.to(tl.int32),
            0,
            (dim_block * (BLOCK_V // 2)).to(tl.int32),
        ]
    )
    v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
    v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
    even_packed, odd_packed = tl.split(v_pairs)
    low_vals = _fp4_pack_low_nibbles(even_packed, odd_packed)
    high_vals = _fp4_pack_high_nibbles(even_packed, odd_packed)
    v_vals = tl.reshape(
        tl.join(low_vals, high_vals).permute(0, 2, 1),
        (BLOCK_V, PAGE_SIZE // 2),
    )

    out_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    row_base = (page_idx * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
    out_desc.store([row_base.to(tl.int32), 0], v_vals)


@triton.jit
def _fp4_mla_attention_v_repack_pages_kernel(
    v_packed_ptr,
    kv_cache_ptr,
    page_ids_ptr,
    num_pages,
    kv_s0,
    kv_s2,
    kv_s4,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    page_list_idx = tl.program_id(0)
    dim_block = tl.program_id(1)
    page_idx = tl.load(page_ids_ptr + page_list_idx).to(tl.int64)

    tl.assume(kv_s0 % 8 == 0)
    tl.assume(kv_s2 % 8 == 0)
    tl.assume(kv_s4 == 1)
    v_desc = tl.make_tensor_descriptor(
        kv_cache_ptr,
        shape=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        block_shape=[1, PAGE_SIZE, BLOCK_V // 2],
    )
    v_tile = v_desc.load(
        [
            page_idx.to(tl.int32),
            0,
            (dim_block * (BLOCK_V // 2)).to(tl.int32),
        ]
    )
    v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
    v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
    even_packed, odd_packed = tl.split(v_pairs)
    low_vals = _fp4_pack_low_nibbles(even_packed, odd_packed)
    high_vals = _fp4_pack_high_nibbles(even_packed, odd_packed)
    v_vals = tl.reshape(
        tl.join(low_vals, high_vals).permute(0, 2, 1),
        (BLOCK_V, PAGE_SIZE // 2),
    )

    out_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    row_base = (page_idx * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
    out_desc.store([row_base.to(tl.int32), 0], v_vals)


def fp4_mla_repack_v_cache(
    v_packed: Any,
    kv_cache: Any,
    page_ids: Optional[Any] = None,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int = 128,
    kernel_occupancy: int = 8,
    kernel_num_stages: int = 1,
) -> None:
    """Populate the public-Triton V-packed auxiliary cache."""
    if v_head_dim % block_v != 0:
        raise ValueError(f"v_head_dim={v_head_dim} must be divisible by block_v={block_v}.")
    if kv_cache.ndim < 5:
        raise ValueError(
            f"kv_cache must expose the paged FP4 layout, got shape={tuple(kv_cache.shape)}."
        )
    num_pages = kv_cache.shape[0]
    num_dim_blocks = triton.cdiv(v_head_dim, block_v)
    launch_meta = {
        "occupancy": int(kernel_occupancy),
        "num_stages": int(kernel_num_stages),
    }
    if page_ids is None:
        if num_pages == 0:
            return
        _fp4_mla_attention_v_repack_kernel[(num_pages, num_dim_blocks)](
            v_packed,
            kv_cache,
            num_pages,
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            V_HEAD_D=v_head_dim,
            PAGE_SIZE=page_size,
            BLOCK_V=block_v,
            **launch_meta,
        )
        return

    if page_ids.numel() == 0:
        return
    _fp4_mla_attention_v_repack_pages_kernel[(page_ids.numel(), num_dim_blocks)](
        v_packed,
        kv_cache,
        page_ids,
        num_pages,
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        V_HEAD_D=v_head_dim,
        PAGE_SIZE=page_size,
        BLOCK_V=block_v,
        **launch_meta,
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
        # Static per-layer scales: QK encodes q * q_gscale and k * kv_gscale,
        # so divide the scores by (q_gscale * kv_gscale). When Q and KV share
        # one global scale, this reduces to global_scale^2.
        # PV applies the final KV global_scale divisor separately.
        q_gscale = tl.load(q_global_scale_ptr)
        qk_scale = sm_scale / (q_gscale * global_scale)
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
            # Keep P scales static-like. PV cancels each page's V scale before
            # accumulating pages together.
            stored_scale = tl.where(
                amax > 0.0,
                tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0),
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
def _fp4_mla_attention_page_stats_grouped_kernel(
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
    GROUP_PAGES: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    PAGE_LOOP_STAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    """Grouped page-stats QK + softmax-stats + FP4 P pack.

    Functionally identical to ``_fp4_mla_attention_page_stats_kernel`` for the
    perfect decode shape (NUM_HEADS == BLOCK_H, TMA + PACK_PROBS, the standard
    640/576/64 residual-Q layout) but each program owns one
    ``(query, head_block, page_group)`` and walks ``GROUP_PAGES`` pages in a
    pipelined loop. Q (and its scales) and the TMA descriptors are loaded once
    and reused across the group, eliminating the per-page Q reload and the tiny
    per-CTA prologue that made the one-page-per-CTA kernel work-bound at long
    context. Per-page outputs (page_max/page_sum and packed P) are written
    exactly as the one-page kernel writes them, so every downstream stage is
    unchanged.
    """
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_group = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    head_start = head_block * BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = query_idx * NUM_HEADS

    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)

    global_scale = tl.load(global_scale_ptr)
    q_gscale = tl.load(q_global_scale_ptr)
    qk_scale = sm_scale / (q_gscale * global_scale)

    residual_groups = Q_RESIDUAL_D // FP4_BLOCK
    non_residual_groups = K_HEAD_D // FP4_BLOCK - residual_groups

    # ---- Hoisted, page-independent index tensors and Q tiles ----
    # Main window (q_start == 0): the first FULL_BLOCK_END elements sit entirely
    # in the non-residual region, so Q and K map 1:1 and load contiguously.
    # Q rows are global rows (query_idx * NUM_HEADS + head); the swizzle's
    # row_group term selects this query's scale/tail block, so q_row_base must
    # be folded in (the main q_vals descriptor load already does this via its
    # row coordinate). Keep int32 to match the one-page kernel -- int64 swizzle
    # math is emulated and was measured ~2x slower; the max global row index
    # (num_queries * NUM_HEADS * stride) stays well within int32.
    q_rows = q_row_base + offs_h
    scale_offsets = tl.arange(0, BLOCK_K // FP4_BLOCK)
    q_sf_cols = scale_offsets
    q_sf_offsets = _fp4_mla_swizzled_sf_offset(q_rows[:, None], q_sf_cols[None, :], Q_SF_PER_TOKEN)
    k_sf_offsets_main = _fp4_mla_swizzled_sf_offset(
        offs_t[:, None], q_sf_cols[None, :], K_SF_PER_TOKEN
    )
    q_scales = tl.load(q_sf_ptr + q_sf_offsets)

    # Tail window (q_start == FULL_BLOCK_END): the residual-Q groups, each of
    # which maps onto a duplicated K residual group.
    tail_packed_offsets = tl.arange(0, TAIL_BLOCK_K // 2)
    tail_scale_offsets = tl.arange(0, TAIL_BLOCK_K // FP4_BLOCK)
    qt_elem = FULL_BLOCK_END + tail_packed_offsets * 2
    qt_group = qt_elem // FP4_BLOCK
    kt_group = tl.where(
        qt_group < non_residual_groups,
        qt_group,
        non_residual_groups + (qt_group - non_residual_groups) // 2,
    )
    byte_t = (qt_elem % FP4_BLOCK) // 2
    packed_qt_cols = FULL_BLOCK_END // 2 + tail_packed_offsets
    packed_kt_cols = kt_group * (FP4_BLOCK // 2) + byte_t
    qt_sf_cols = FULL_BLOCK_END // FP4_BLOCK + tail_scale_offsets
    kt_sf_cols = tl.where(
        qt_sf_cols < non_residual_groups,
        qt_sf_cols,
        non_residual_groups + (qt_sf_cols - non_residual_groups) // 2,
    )
    qt_sf_offsets = _fp4_mla_swizzled_sf_offset(
        q_rows[:, None], qt_sf_cols[None, :], Q_SF_PER_TOKEN
    )
    kt_sf_offsets = _fp4_mla_swizzled_sf_offset(
        offs_t[:, None], kt_sf_cols[None, :], K_SF_PER_TOKEN
    )
    q_tail_scales = tl.load(q_sf_ptr + qt_sf_offsets)
    q_tail_vals = tl.load(
        q_fp4_ptr + q_rows[:, None] * q_fp4_s0 + packed_qt_cols[None, :] * q_fp4_s1
    )

    tl.assume(q_fp4_s0 % 8 == 0)
    tl.assume(q_fp4_s1 == 1)
    tl.assume(kv_s0 % 8 == 0)
    tl.assume(kv_s2 % 8 == 0)
    tl.assume(kv_s4 == 1)
    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
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
    p_desc = tl.make_tensor_descriptor(
        p_fp4_ptr,
        shape=[p_num_rows, PAGE_SIZE // 2],
        strides=[p_s0, p_s1],
        block_shape=[BLOCK_H, PAGE_SIZE // 2],
    )
    q_vals = q_desc.load([(q_row_base + head_start).to(tl.int32), 0])

    scale_cols = tl.arange(0, SF_PER_PAGE)
    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]

    page_lo = page_group * GROUP_PAGES
    page_hi = page_lo + GROUP_PAGES
    for page_rel in tl.range(page_lo, page_hi, num_stages=PAGE_LOOP_STAGES):
        if page_rel < MAX_PAGES:
            page_start = page_rel * PAGE_SIZE
            page_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
            page_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
            if ASSUME_FULL_PAGES or page_start < kv_len:
                compact_page = page_table_start + page_rel
                if ASSUME_VALID_PAGES:
                    physical_page = tl.load(src_page_ids_ptr + compact_page).to(tl.int64)
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

                k_vals = k_desc.load([safe_physical_page.to(tl.int32), 0, 0])
                k_vals = tl.reshape(k_vals, (BLOCK_T, BLOCK_K // 2))
                if not ASSUME_VALID_PAGES:
                    k_vals = tl.where(valid_physical_page, k_vals, 0)
                k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets_main)
                scores = tl.dot_scaled(
                    q_vals,
                    q_scales,
                    "e2m1",
                    k_vals.T,
                    k_scales,
                    "e2m1",
                    fast_math=True,
                    rhs_k_pack=True,
                )

                kt_ptrs = (
                    kv_cache_ptr
                    + safe_physical_page * kv_s0
                    + offs_t[:, None].to(tl.int64) * kv_s2
                    + packed_kt_cols[None, :] * kv_s4
                )
                if ASSUME_VALID_PAGES:
                    kt_vals = tl.load(kt_ptrs)
                else:
                    kt_vals = tl.load(kt_ptrs, mask=valid_physical_page, other=0)
                kt_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + kt_sf_offsets)
                scores = tl.dot_scaled(
                    q_tail_vals,
                    q_tail_scales,
                    "e2m1",
                    kt_vals.T,
                    kt_scales,
                    "e2m1",
                    acc=scores,
                    fast_math=True,
                    rhs_k_pack=True,
                )

                if ASSUME_FULL_PAGES:
                    scores = scores * qk_scale
                    page_max = tl.max(scores, axis=1)
                    exp_scores = tl.math.exp2((scores - page_max[:, None]) * _LOG2_E)
                    page_sum = tl.sum(exp_scores, axis=1)
                else:
                    valid_t = page_start + offs_t < kv_len
                    scores = tl.where(valid_t[None, :], scores * qk_scale, -float("inf"))
                    page_max = tl.max(scores, axis=1)
                    exp_scores = tl.math.exp2((scores - page_max[:, None]) * _LOG2_E)
                    exp_scores = tl.where(valid_t[None, :], exp_scores, 0.0)
                    page_sum = tl.sum(exp_scores, axis=1)

                grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
                amax = tl.max(grouped_probs, axis=2)
                inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
                stored_scale = tl.where(
                    amax > 0.0,
                    tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0),
                    1.0,
                )
                scaled_probs = grouped_probs * tl.reshape(
                    inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1)
                )
                pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
                even_probs, odd_probs = tl.split(pairs)
                packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

                p_page = query_idx * MAX_PAGES + page_rel
                if ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
                    sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
                        p_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
                    )
                else:
                    p_rows = (p_page * NUM_HEADS + offs_h).to(tl.int64)
                    sf_offsets = _fp4_mla_swizzled_sf_offset(
                        p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
                    )
                if ASSUME_VALID_PAGES:
                    tl.store(p_sf_ptr + sf_offsets, stored_scale)
                    p_desc.store(
                        [(p_page * NUM_HEADS + head_start).to(tl.int32), 0],
                        tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
                    )
                else:
                    tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=valid_compact_page)
                    p_rows = (p_page * NUM_HEADS + offs_h).to(tl.int64)
                    tl.store(
                        p_fp4_ptr + p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                        packed,
                        mask=valid_compact_page,
                    )

            out_offsets = query_idx * page_stats_s0 + page_rel * page_stats_s1 + offs_h
            tl.store(page_max_ptr + out_offsets, page_max)
            tl.store(page_sum_ptr + out_offsets, page_sum)


@triton.jit
def _fp4_mla_attention_page_stats_mtp_kernel(
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
    occupancy: tl.constexpr = 1,
):
    """MTP-fused page-stats: one CTA owns (seq, head_block, page) and processes
    all QUERY_LEN_PER_SEQ linear-MTP query rows of the sequence, loading the
    page's K (and K scales) once and reusing it across the q_len QK matmuls.

    The per-query-row kernel reloads K once per query row (q_len times per
    page); at decode the QK is load-latency bound (one K load feeds one MMA),
    so amortizing the K load over q_len rows lifts the load:MMA ratio. Per-row
    outputs are written identically to the one-page kernel's masked path
    (ASSUME_FULL_PAGES/VALID_PAGES are always False for q_len>1), so all
    downstream stages are unchanged. Restricted to the perfect decode shape.
    """
    seq_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2)
    head_start = head_block * BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    page_start = page_rel * PAGE_SIZE

    kv_len_base = tl.load(kv_lens_ptr + seq_idx)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    q_gscale = tl.load(q_global_scale_ptr)
    qk_scale = sm_scale / (q_gscale * global_scale)

    residual_groups = Q_RESIDUAL_D // FP4_BLOCK
    non_residual_groups = K_HEAD_D // FP4_BLOCK - residual_groups

    # ---- r-independent index tensors (main + residual-tail column maps) ----
    scale_offsets = tl.arange(0, BLOCK_K // FP4_BLOCK)
    q_sf_cols = scale_offsets
    k_sf_offsets_main = _fp4_mla_swizzled_sf_offset(
        offs_t[:, None], q_sf_cols[None, :], K_SF_PER_TOKEN
    )
    tail_packed_offsets = tl.arange(0, TAIL_BLOCK_K // 2)
    tail_scale_offsets = tl.arange(0, TAIL_BLOCK_K // FP4_BLOCK)
    qt_elem = FULL_BLOCK_END + tail_packed_offsets * 2
    qt_group = qt_elem // FP4_BLOCK
    kt_group = tl.where(
        qt_group < non_residual_groups,
        qt_group,
        non_residual_groups + (qt_group - non_residual_groups) // 2,
    )
    byte_t = (qt_elem % FP4_BLOCK) // 2
    packed_qt_cols = FULL_BLOCK_END // 2 + tail_packed_offsets
    packed_kt_cols = kt_group * (FP4_BLOCK // 2) + byte_t
    qt_sf_cols = FULL_BLOCK_END // FP4_BLOCK + tail_scale_offsets
    kt_sf_cols = tl.where(
        qt_sf_cols < non_residual_groups,
        qt_sf_cols,
        non_residual_groups + (qt_sf_cols - non_residual_groups) // 2,
    )
    kt_sf_offsets = _fp4_mla_swizzled_sf_offset(
        offs_t[:, None], kt_sf_cols[None, :], K_SF_PER_TOKEN
    )
    scale_cols = tl.arange(0, SF_PER_PAGE)
    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]

    tl.assume(q_fp4_s0 % 8 == 0)
    tl.assume(q_fp4_s1 == 1)
    tl.assume(kv_s0 % 8 == 0)
    tl.assume(kv_s2 % 8 == 0)
    tl.assume(kv_s4 == 1)
    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
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
    # p_desc = tl.make_tensor_descriptor(
    #     p_fp4_ptr,
    #     shape=[p_num_rows, PAGE_SIZE // 2],
    #     strides=[p_s0, p_s1],
    #     block_shape=[BLOCK_H, PAGE_SIZE // 2],
    # )

    # ---- Load this page's K once (shared across all query rows). ----
    compact_page = page_table_start + page_rel
    valid_compact_page = (compact_page >= 0) & (compact_page < page_ids_len)
    safe_compact_page = tl.where(valid_compact_page, compact_page, 0)
    physical_page = tl.load(
        src_page_ids_ptr + safe_compact_page, mask=valid_compact_page, other=-1
    ).to(tl.int64)
    valid_physical_page = valid_compact_page & (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.where(valid_physical_page, physical_page, 0)

    k_vals = k_desc.load([safe_physical_page.to(tl.int32), 0, 0])
    k_vals = tl.reshape(k_vals, (BLOCK_T, BLOCK_K // 2))
    k_vals = tl.where(valid_physical_page, k_vals, 0)
    k_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + k_sf_offsets_main)
    kt_vals = tl.load(
        kv_cache_ptr
        + safe_physical_page * kv_s0
        + offs_t[:, None].to(tl.int64) * kv_s2
        + packed_kt_cols[None, :] * kv_s4,
        mask=valid_physical_page,
        other=0,
    )
    kt_scales = tl.load(sf_cache_ptr + safe_physical_page * sf_s0 + kt_sf_offsets)

    for r in tl.static_range(QUERY_LEN_PER_SEQ):
        query_idx_r = seq_idx * QUERY_LEN_PER_SEQ + r
        kv_len_r = tl.maximum(kv_len_base - (QUERY_LEN_PER_SEQ - 1 - r), 0)
        q_row_base_r = query_idx_r * NUM_HEADS
        q_rows_r = q_row_base_r + offs_h
        page_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        page_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        if page_start < kv_len_r:
            q_vals = q_desc.load([(q_row_base_r + head_start).to(tl.int32), 0])
            q_sf_offsets = _fp4_mla_swizzled_sf_offset(
                q_rows_r[:, None], q_sf_cols[None, :], Q_SF_PER_TOKEN
            )
            q_scales = tl.load(q_sf_ptr + q_sf_offsets)
            scores = tl.dot_scaled(
                q_vals,
                q_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                fast_math=True,
                rhs_k_pack=True,
            )
            qt_sf_offsets = _fp4_mla_swizzled_sf_offset(
                q_rows_r[:, None], qt_sf_cols[None, :], Q_SF_PER_TOKEN
            )
            q_tail_scales = tl.load(q_sf_ptr + qt_sf_offsets)
            q_tail_vals = tl.load(
                q_fp4_ptr + q_rows_r[:, None] * q_fp4_s0 + packed_qt_cols[None, :] * q_fp4_s1
            )
            scores = tl.dot_scaled(
                q_tail_vals,
                q_tail_scales,
                "e2m1",
                kt_vals.T,
                kt_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )

            valid_t = page_start + offs_t < kv_len_r
            scores = tl.where(valid_t[None, :], scores * qk_scale, -float("inf"))
            page_max = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max[:, None]) * _LOG2_E)
            exp_scores = tl.where(valid_t[None, :], exp_scores, 0.0)
            page_sum = tl.sum(exp_scores, axis=1)

            grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
            amax = tl.max(grouped_probs, axis=2)
            inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
            stored_scale = tl.where(
                amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0
            )
            scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
            pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
            even_probs, odd_probs = tl.split(pairs)
            packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

            p_page = query_idx_r * MAX_PAGES + page_rel
            p_rows = (p_page * NUM_HEADS + offs_h).to(tl.int64)
            sf_offsets = _fp4_mla_swizzled_sf_offset(
                p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
            )
            tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=valid_compact_page)
            tl.store(
                p_fp4_ptr + p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                packed,
                mask=valid_compact_page,
            )

        out_offsets = query_idx_r * page_stats_s0 + page_rel * page_stats_s1 + offs_h
        tl.store(page_max_ptr + out_offsets, page_max)
        tl.store(page_sum_ptr + out_offsets, page_sum)


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
def _fp4_mla_attention_group_reduce_stats_kernel(
    group_max_ptr,
    group_sum_ptr,
    page_max_ptr,
    page_sum_ptr,
    num_pages,
    group_stats_s0,
    group_stats_s1,
    page_stats_s0,
    page_stats_s1,
    NUM_HEADS: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    """First level of a two-level softmax-stats reduction.

    Each program owns one ``(query, head_block, page_group)`` and combines the
    ``GROUP_PAGES`` per-page ``(max, sum)`` pairs in its group into a single
    online-softmax partial ``(group_max, group_denom)``. ``group_denom`` is the
    page sums rescaled to ``group_max``.  A small follow-up combine
    (``_fp4_mla_attention_reduce_stats_kernel`` over the compact group buffer)
    folds the groups into the global ``(max, denom)``.

    This parallelizes the page reduction across the grid's third axis so the
    decode-stats reduction is no longer a handful of CTAs each serially walking
    every page (the bottleneck the single-level reduce hit at small batch).
    """
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    group_idx = tl.program_id(2)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)

    group_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    group_sum = tl.zeros((BLOCK_H,), tl.float32)
    page_lo = group_idx * GROUP_PAGES
    page_hi = page_lo + GROUP_PAGES
    for page_idx in tl.range(page_lo, page_hi, num_stages=PIPELINE_STAGES):
        page_valid = page_idx < num_pages
        page_offsets = gen_idx * page_stats_s0 + page_idx * page_stats_s1 + safe_offs_h
        page_max = tl.load(
            page_max_ptr + page_offsets, mask=mask_h & page_valid, other=-float("inf")
        )
        page_sum = tl.load(page_sum_ptr + page_offsets, mask=mask_h & page_valid, other=0.0)
        next_group_max = tl.maximum(group_max, page_max)
        # Guard the rescale deltas so an empty accumulator (group_sum == 0,
        # group_max == -inf) or an empty page (page_sum == 0) never feeds
        # (-inf) - (-inf) = NaN into exp2.
        old_delta = tl.where(group_sum > 0.0, group_max - next_group_max, 0.0)
        new_delta = tl.where(page_sum > 0.0, page_max - next_group_max, 0.0)
        group_sum = group_sum * tl.math.exp2(old_delta * _LOG2_E) + page_sum * tl.math.exp2(
            new_delta * _LOG2_E
        )
        group_max = next_group_max

    out_offsets = gen_idx * group_stats_s0 + group_idx * group_stats_s1 + safe_offs_h
    tl.store(group_max_ptr + out_offsets, group_max, mask=mask_h)
    tl.store(group_sum_ptr + out_offsets, group_sum, mask=mask_h)


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
    v_packed_ptr,
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
    USE_PREPACKED_V: tl.constexpr,
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
    if not PARTIAL_OUT and USE_TMA_V_LOAD and ASSUME_FULL_HEADS and ASSUME_FULL_V:
        tl.assume(out_s1 % 8 == 0)
        tl.assume(out_s2 == 1)
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[out_num_rows, V_HEAD_D],
            strides=[out_s1, out_s2],
            block_shape=[BLOCK_H, BLOCK_V],
        )
    if USE_PREPACKED_V:
        v_packed_desc = tl.make_tensor_descriptor(
            v_packed_ptr,
            shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
            strides=[PAGE_SIZE // 2, 1],
            block_shape=[BLOCK_V, PAGE_SIZE // 2],
        )
    elif USE_TMA_V_LOAD:
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
            if USE_PREPACKED_V:
                v_row = (safe_physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
                v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
                if not ASSUME_VALID_PAGES:
                    v_vals = tl.where(valid_physical_page, v_vals, 0)
            elif USE_TMA_V_LOAD:
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
def _fp4_mla_attention_pv_prepacked_v_kernel(
    out_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    v_packed_ptr,
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
    USE_TMA_OUT_STORE: tl.constexpr,
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
    if not PARTIAL_OUT and USE_TMA_OUT_STORE and ASSUME_FULL_HEADS and ASSUME_FULL_V:
        tl.assume(out_s1 % 8 == 0)
        tl.assume(out_s2 == 1)
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[out_num_rows, V_HEAD_D],
            strides=[out_s1, out_s2],
            block_shape=[BLOCK_H, BLOCK_V],
        )
    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
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
            if ASSUME_FULL_HEADS and NUM_HEADS == 128 and BLOCK_H == 128:
                p_sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
                    p_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
                )
            else:
                p_sf_offsets = _fp4_mla_swizzled_sf_offset(
                    safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
                )
            p_scales = tl.load(p_sf_ptr + p_sf_offsets)
            v_row = (safe_physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
            v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
            if not ASSUME_VALID_PAGES:
                v_vals = tl.where(valid_physical_page, v_vals, 0)
            v_scales = tl.load(v_sf_ptr + safe_physical_page * vsf_s0 + v_sf_offsets)
            acc = tl.dot_scaled(
                v_vals,
                v_scales,
                "e2m1",
                p_vals.T,
                p_scales,
                "e2m1",
                acc=acc,
                fast_math=True,
                rhs_k_pack=True,
            )

    out_vals = acc.T
    if PARTIAL_OUT:
        base = (
            query_idx * partial_s0
            + split_idx * partial_s1
            + safe_offs_h[:, None] * partial_s2
            + safe_offs_v[None, :] * partial_s3
        )
        tl.store(partial_out_ptr + base, out_vals, mask=mask_h[:, None] & mask_v[None, :])
    elif ASSUME_FULL_HEADS and ASSUME_FULL_V:
        out_vals = out_vals * out_scale
        if USE_TMA_OUT_STORE:
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
            out_vals * out_scale,
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
