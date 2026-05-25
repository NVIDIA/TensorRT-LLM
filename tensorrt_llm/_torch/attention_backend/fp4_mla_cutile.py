# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FP4 MLA paged decode attention using Triton.

The kernels are adapted from TensorRT-LLM's FP4 MLA decode path.  This module
exposes the attention path for already-packed FP4 Q/K/V tensors and swizzled
FP8 block-scale tensors; quantization and KV-cache update helpers remain outside
this internal op.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

FP4_BLOCK_SIZE = 16
FP4_MLA_P_GLOBAL_SCALE = 448.0 * 6.0


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _swizzled_scale_size(rows: int, logical_cols: int) -> int:
    scale_cols = _ceil_div(logical_cols, FP4_BLOCK_SIZE)
    padded_cols = _ceil_div(scale_cols, 4) * 4
    return _ceil_div(rows, 128) * 128 * padded_cols


def _get_kv_cache_strides(kv_cache: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    if kv_cache.dim() == 3:
        num_pages, page_size, packed_dim = kv_cache.shape
        return (
            num_pages,
            page_size,
            packed_dim,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
        )
    if kv_cache.dim() >= 5:
        num_pages = kv_cache.shape[0]
        page_size = kv_cache.shape[2]
        packed_dim = kv_cache.shape[4]
        return (
            num_pages,
            page_size,
            packed_dim,
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
        )
    raise ValueError(
        "kv_cache must be shaped (num_pages, page_size, packed_dim) or (num_pages, ..., page_size, ..., packed_dim)."
    )


def _workspace_tensor(
    workspace: Optional[torch.Tensor],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                f"Cannot allocate {name} while capturing a CUDA graph. "
                "Pass a preallocated workspace tensor."
            )
        return torch.empty(shape, dtype=dtype, device=device)

    invalid = (
        workspace.dtype != dtype
        or workspace.device != device
        or len(workspace.shape) != len(shape)
        or any(workspace.shape[idx] < dim for idx, dim in enumerate(shape))
    )
    if invalid:
        raise ValueError(
            f"{name} workspace must have shape at least {shape}, dtype={dtype}, "
            f"and device={device}; got shape={tuple(workspace.shape)}, "
            f"dtype={workspace.dtype}, device={workspace.device}."
        )

    slices = tuple(slice(0, dim) for dim in shape)
    return workspace[slices]


@triton.jit
def _fp4_mla_swizzled_sf_offset(row_idx, col_idx, SF_PER_TOKEN: tl.constexpr):
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
def _fp4_e2m1_quantize_packed(even, odd):
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
def _fp4_pack_low_nibbles(even_packed, odd_packed):
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

    if (
        USE_TMA_DATA_LOAD
        and ASSUME_FULL_HEADS
        and ASSUME_VALID_PAGES
        and Q_HEAD_D == 640
        and K_HEAD_D == 576
        and Q_RESIDUAL_D == 64
        and BLOCK_H == 128
        and BLOCK_T == 128
        and BLOCK_K == 512
        and FULL_BLOCK_END == 512
        and TAIL_BLOCK_K == 128
    ):
        tl.assume(q_fp4_s0 % 8 == 0)
        tl.assume(q_fp4_s1 == 1)
        tl.assume(kv_s0 % 8 == 0)
        tl.assume(kv_s2 % 8 == 0)
        tl.assume(kv_s4 == 1)
        q_desc = tl.make_tensor_descriptor(
            q_fp4_ptr,
            shape=[q_num_rows, Q_HEAD_D // 2],
            strides=[q_fp4_s0, q_fp4_s1],
            block_shape=[BLOCK_H, 256],
        )
        k_desc = tl.make_tensor_descriptor(
            kv_cache_ptr,
            shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            block_shape=[1, BLOCK_T, 256],
        )
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
        q_sf_full_view = tl.ext.make_view(
            base=q_sf_ptr,
            shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
            strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
            tile_shape=[1, 8, 2, 256],
            tile_dim_map=[0, 1, 2, 3],
        )
        k_sf_full_view = tl.ext.make_view(
            base=sf_cache_ptr,
            shapes=[num_pages, 1, ((K_SF_PER_TOKEN + 3) // 4), 2, 256],
            strides=[sf_s0, 128 * (((K_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
            tile_shape=[1, 1, 8, 2, 256],
            tile_dim_map=[0, 1, 2, 3, 4],
        )
        k_sf_tail_view = tl.ext.make_view(
            base=sf_cache_ptr,
            shapes=[num_pages, 1, ((K_SF_PER_TOKEN + 3) // 4), 2, 256],
            strides=[sf_s0, 128 * (((K_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
            tile_shape=[1, 1, 1, 2, 256],
            tile_dim_map=[0, 1, 2, 3, 4],
        )

        full_q_vals = q_desc.load([(q_row_base + head_start).to(tl.int32), 0])
        full_k_vals = k_desc.load([safe_physical_page.to(tl.int32), 0, 0])
        full_k_vals = tl.reshape(full_k_vals, (BLOCK_T, 256))
        q_row_group = q_row_base // 128
        full_q_scales = tl.ext.load_view_tko(q_sf_full_view, [q_row_group.to(tl.int32), 0, 0, 0])
        full_q_scales = full_q_scales.reshape([1, 8, 32, 4, 4]).trans(0, 3, 2, 1, 4)
        full_q_scales = full_q_scales.reshape([BLOCK_H, 32])
        full_k_scales = tl.ext.load_view_tko(
            k_sf_full_view, [safe_physical_page.to(tl.int32), 0, 0, 0, 0]
        )
        full_k_scales = full_k_scales.reshape([1, 1, 8, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        full_k_scales = full_k_scales.reshape([BLOCK_T, 32])
        scores = tl.dot_scaled(
            full_q_vals,
            full_q_scales,
            "e2m1",
            full_k_vals.T,
            full_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )

        tail_k_vals = k_tail_desc.load([safe_physical_page.to(tl.int32), 0, 256])
        tail_k_vals = tl.reshape(tail_k_vals, (BLOCK_T, 32))
        tail_k_scales = tl.ext.load_view_tko(
            k_sf_tail_view, [safe_physical_page.to(tl.int32), 0, 8, 0, 0]
        )
        tail_k_scales = tail_k_scales.reshape([1, 1, 1, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        tail_k_scales = tail_k_scales.reshape([BLOCK_T, 4])
        q_tail_vals = q_tail_desc.load([(q_row_base + head_start).to(tl.int32), 256])
        # Map Q tail groups [0, 1, ..., 7] onto K tail groups [0, 0, 1, 1, ..., 3, 3].
        q_tail_vals = q_tail_vals.reshape([BLOCK_H, 4, 2, 8]).trans(0, 1, 3, 2)
        q_even_vals, q_odd_vals = tl.split(q_tail_vals)
        q_even_vals = q_even_vals.reshape([BLOCK_H, 32])
        q_odd_vals = q_odd_vals.reshape([BLOCK_H, 32])

        q_tail_sf_cols = 32 + tl.arange(0, 8)
        q_tail_sf_offsets = _fp4_mla_swizzled_sf_offset(
            q_rows[:, None], q_tail_sf_cols[None, :], Q_SF_PER_TOKEN
        )
        q_tail_scales = tl.load(q_sf_ptr + q_tail_sf_offsets)
        q_tail_scales = q_tail_scales.reshape([BLOCK_H, 4, 2])
        q_even_scales, q_odd_scales = tl.split(q_tail_scales)
        scores = tl.dot_scaled(
            q_even_vals,
            q_even_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q_odd_vals,
            q_odd_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        return scores

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
    for q_start in tl.range(0, FULL_BLOCK_END, BLOCK_K):
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
        if Q_RESIDUAL_D == 64 and TAIL_BLOCK_K == 128:
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
            scores = tl.dot_scaled(
                q_even_vals,
                q_even_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )
            scores = tl.dot_scaled(
                q_odd_vals,
                q_odd_scales,
                "e2m1",
                k_vals.T,
                k_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )
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
def _fp4_mla_attention_stats_kernel(
    max_ptr,
    denom_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
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
    stats_s0,
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
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = gen_idx * NUM_HEADS
    kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    for page_rel in tl.range(0, MAX_PAGES):
        page_start = page_rel * PAGE_SIZE
        if ASSUME_FULL_PAGES or page_start < kv_len:
            compact_page = page_table_start + page_rel
            scores = _fp4_mla_qk_scores_tile(
                q_fp4_ptr,
                q_sf_ptr,
                kv_cache_ptr,
                sf_cache_ptr,
                src_page_ids_ptr,
                compact_page,
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
                scores = tl.where(mask_h[:, None], scores * qk_scale, -float("inf"))
            else:
                valid_t = page_start + offs_t < kv_len
                scores = tl.where(
                    mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf")
                )
            page_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, page_max)
            denom = denom * tl.math.exp2((max_score - new_max) * 1.4426950408889634) + tl.sum(
                tl.math.exp2((scores - new_max[:, None]) * 1.4426950408889634), axis=1
            )
            max_score = new_max

    tl.store(max_ptr + gen_idx * stats_s0 + safe_offs_h, max_score, mask=mask_h)
    tl.store(denom_ptr + gen_idx * stats_s0 + safe_offs_h, denom, mask=mask_h)


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
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = gen_idx * NUM_HEADS
    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_start = page_rel * PAGE_SIZE
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    out_offsets = gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h

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
        qk_scale = sm_scale / (global_scale * global_scale)
        if ASSUME_FULL_HEADS and ASSUME_FULL_PAGES:
            scores = scores * qk_scale
            page_max = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max[:, None]) * 1.4426950408889634)
            page_sum = tl.sum(exp_scores, axis=1)
        else:
            scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
            page_max = tl.max(scores, axis=1)
            safe_page_max = tl.where(mask_h, page_max, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(mask_h[:, None] & valid_t[None, :], exp_scores, 0.0)
            page_sum = tl.sum(exp_scores, axis=1)

        if PACK_PROBS:
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

            if ASSUME_VALID_PAGES:
                safe_compact_page = page_table_start + page_rel
            else:
                valid_compact_page = (page_table_start + page_rel >= 0) & (
                    page_table_start + page_rel < page_ids_len
                )
                safe_compact_page = tl.where(valid_compact_page, page_table_start + page_rel, 0)
            p_rows = safe_compact_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows
                if ASSUME_FULL_HEADS
                else tl.where(mask_h, p_rows, safe_compact_page * NUM_HEADS)
            )
            scale_cols = tl.arange(0, SF_PER_PAGE)
            if ASSUME_FULL_HEADS and ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
                sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
                    safe_compact_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
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
                    [(safe_compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
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
    stats_s0,
    page_stats_s0,
    page_stats_s1,
    NUM_HEADS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    for page_rel in tl.range(0, MAX_PAGES):
        page_max = tl.load(
            page_max_ptr + gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
            mask=mask_h,
            other=-float("inf"),
        )
        max_score = tl.maximum(max_score, page_max)

    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for page_rel in tl.range(0, MAX_PAGES):
        page_max = tl.load(
            page_max_ptr + gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
            mask=mask_h,
            other=-float("inf"),
        )
        page_sum = tl.load(
            page_sum_ptr + gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
            mask=mask_h,
            other=0.0,
        )
        denom += tl.where(
            page_sum > 0.0,
            page_sum * tl.math.exp2((page_max - max_score) * 1.4426950408889634),
            0.0,
        )

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
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2)

    page_start = page_rel * PAGE_SIZE
    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + gen_idx)
        if page_start >= kv_len:
            return

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
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
        page_max_ptr + gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
        mask=mask_h,
        other=-float("inf"),
    )
    max_score = tl.load(max_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    factor = tl.where(
        denom > 0.0, tl.math.exp2((page_max - max_score) * 1.4426950408889634) / denom, 0.0
    )

    p_rows = compact_page * NUM_HEADS + offs_h
    safe_p_rows = (
        p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, compact_page * NUM_HEADS)
    )
    scale_cols = tl.arange(0, SF_PER_PAGE)
    if ASSUME_FULL_HEADS and ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
        sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            compact_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    else:
        sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    scales = tl.load(p_sf_ptr + sf_offsets, mask=mask_h[:, None], other=1.0).to(tl.float32)
    tl.store(p_sf_ptr + sf_offsets, scales * factor[:, None], mask=mask_h[:, None])


@triton.jit
def _fp4_mla_attention_prob_store_page_kernel(
    probs_ptr,
    max_ptr,
    denom_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_rel,
    page_ids_len,
    num_pages,
    probs_s0,
    probs_s1,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    stats_s0,
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
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)

    kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_start = page_rel * PAGE_SIZE
    if (not ASSUME_FULL_PAGES) and page_start >= kv_len:
        return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, PAGE_SIZE)
    if ASSUME_FULL_PAGES:
        valid_t = tl.full([PAGE_SIZE], True, dtype=tl.int1)
    else:
        valid_t = page_start + offs_t < kv_len
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return
    q_row_base = gen_idx * NUM_HEADS

    scores = _fp4_mla_qk_scores_tile(
        q_fp4_ptr,
        q_sf_ptr,
        kv_cache_ptr,
        sf_cache_ptr,
        src_page_ids_ptr,
        compact_page,
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
        PAGE_SIZE,
        BLOCK_K,
        FULL_BLOCK_END,
        TAIL_BLOCK_K,
        NUM_HEADS,
        USE_TMA_DATA_LOAD,
        ASSUME_FULL_HEADS,
        ASSUME_VALID_PAGES,
    )
    max_score = tl.load(max_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=1.0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    denom_valid = denom > 0.0
    safe_denom = tl.where(denom_valid, denom, 1.0)
    safe_max = tl.where(denom_valid, max_score, 0.0)
    scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
    probs = tl.math.exp2((scores - safe_max[:, None]) * 1.4426950408889634) / safe_denom[:, None]
    probs = tl.where(mask_h[:, None] & valid_t[None, :] & denom_valid[:, None], probs, 0.0)

    prob_rows = gen_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, gen_idx * NUM_HEADS)
    tl.store(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + offs_t[None, :] * probs_s1,
        probs,
        mask=mask_h[:, None],
    )


@triton.jit
def _fp4_mla_attention_prob_pack_page_kernel(
    p_fp4_ptr,
    p_sf_ptr,
    probs_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_rel,
    page_ids_len,
    p_s0,
    p_s1,
    probs_s0,
    probs_s1,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    token_group = tl.program_id(1)
    head_block = tl.program_id(2)

    kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_start = page_rel * PAGE_SIZE
    if (not ASSUME_FULL_PAGES) and page_start >= kv_len:
        return

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
    else:
        mask_h = offs_h < NUM_HEADS
    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    token_base = token_group * FP4_BLOCK
    even_t = token_base + byte_offsets * 2
    odd_t = even_t + 1
    valid_even = page_start + even_t < kv_len
    valid_odd = page_start + odd_t < kv_len

    prob_rows = gen_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, gen_idx * NUM_HEADS)
    even_probs = tl.load(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + even_t[None, :] * probs_s1,
        mask=mask_h[:, None] & valid_even[None, :],
        other=0.0,
    )
    odd_probs = tl.load(
        probs_ptr + safe_prob_rows[:, None] * probs_s0 + odd_t[None, :] * probs_s1,
        mask=mask_h[:, None] & valid_odd[None, :],
        other=0.0,
    )
    amax = tl.maximum(tl.max(tl.abs(even_probs), axis=1), tl.max(tl.abs(odd_probs), axis=1))
    local_scale = tl.where(amax > 0.0, amax / 6.0, 1.0)
    stored_scale = tl.where(amax > 0.0, tl.minimum(local_scale * P_GLOBAL_SCALE, 448.0), 1.0)

    p_rows = compact_page * NUM_HEADS + offs_h
    safe_p_rows = tl.where(mask_h, p_rows, compact_page * NUM_HEADS)
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_p_rows, token_group, SF_PER_PAGE)
    tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=mask_h)

    even_quant = _fp4_e2m1_quantize(even_probs / local_scale[:, None])
    odd_quant = _fp4_e2m1_quantize(odd_probs / local_scale[:, None])
    packed = even_quant | (odd_quant << 4)
    byte_cols = token_group * (FP4_BLOCK // 2) + byte_offsets
    tl.store(
        p_fp4_ptr + safe_p_rows[:, None] * p_s0 + byte_cols[None, :] * p_s1,
        packed,
        mask=mask_h[:, None],
    )


@triton.jit
def _fp4_mla_attention_prob_pack_page_fused_kernel(
    p_fp4_ptr,
    p_sf_ptr,
    max_ptr,
    denom_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_rel,
    page_ids_len,
    num_pages,
    p_s0,
    p_s1,
    q_fp4_s0,
    q_fp4_s1,
    kv_s0,
    kv_s2,
    kv_s4,
    sf_s0,
    stats_s0,
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
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    PAGE_REL_FROM_GRID: tl.constexpr = False,
    ASSUME_FULL_HEADS: tl.constexpr = False,
    ASSUME_FULL_PAGES: tl.constexpr = False,
    ASSUME_VALID_PAGES: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    if PAGE_REL_FROM_GRID:
        page_rel = tl.program_id(2)

    kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_start = page_rel * PAGE_SIZE
    if (not ASSUME_FULL_PAGES) and page_start >= kv_len:
        return

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    if ASSUME_FULL_HEADS:
        mask_h = tl.full([BLOCK_H], True, dtype=tl.int1)
        safe_offs_h = offs_h
    else:
        mask_h = offs_h < NUM_HEADS
        safe_offs_h = tl.where(mask_h, offs_h, 0)
    offs_t = tl.arange(0, PAGE_SIZE)
    if ASSUME_FULL_PAGES:
        valid_t = tl.full([PAGE_SIZE], True, dtype=tl.int1)
    else:
        valid_t = page_start + offs_t < kv_len
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return
    q_row_base = gen_idx * NUM_HEADS

    scores = _fp4_mla_qk_scores_tile(
        q_fp4_ptr,
        q_sf_ptr,
        kv_cache_ptr,
        sf_cache_ptr,
        src_page_ids_ptr,
        compact_page,
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
        PAGE_SIZE,
        BLOCK_K,
        FULL_BLOCK_END,
        TAIL_BLOCK_K,
        NUM_HEADS,
        USE_TMA_DATA_LOAD,
        ASSUME_FULL_HEADS,
        ASSUME_VALID_PAGES,
    )
    max_score = tl.load(max_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + gen_idx * stats_s0 + safe_offs_h, mask=mask_h, other=1.0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    denom_valid = denom > 0.0
    safe_denom = tl.where(denom_valid, denom, 1.0)
    safe_max = tl.where(denom_valid, max_score, 0.0)
    scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
    probs = tl.math.exp2((scores - safe_max[:, None]) * 1.4426950408889634) / safe_denom[:, None]
    probs = tl.where(mask_h[:, None] & valid_t[None, :] & denom_valid[:, None], probs, 0.0)

    grouped_probs = tl.reshape(probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
    amax = tl.max(tl.abs(grouped_probs), axis=2)
    local_scale = tl.where(amax > 0.0, amax / 6.0, 1.0)
    stored_scale = tl.where(amax > 0.0, tl.minimum(local_scale * P_GLOBAL_SCALE, 448.0), 1.0)
    scaled_probs = grouped_probs / tl.reshape(local_scale, (BLOCK_H, SF_PER_PAGE, 1))
    pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
    even_probs, odd_probs = tl.split(pairs)
    packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

    p_rows = compact_page * NUM_HEADS + offs_h
    safe_p_rows = tl.where(mask_h, p_rows, compact_page * NUM_HEADS)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    sf_offsets = _fp4_mla_swizzled_sf_offset(safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE)
    tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=mask_h[:, None])

    byte_offsets = tl.arange(0, FP4_BLOCK // 2)
    byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
    tl.store(
        p_fp4_ptr + safe_p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
        packed,
        mask=mask_h[:, None, None],
    )


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
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
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

    if (
        USE_TMA_P_LOAD
        and USE_TMA_V_LOAD
        and ASSUME_FULL_HEADS
        and ASSUME_FULL_PAGES
        and ASSUME_FULL_V
        and ASSUME_VALID_PAGES
        and NUM_HEADS == 128
        and V_HEAD_D == 512
        and PAGE_SIZE == 128
        and BLOCK_H == 128
        and BLOCK_V == 128
        and SF_PER_PAGE == 8
    ):
        p_view = tl.ext.make_view(
            base=p_fp4_ptr,
            shapes=[p_num_rows, PAGE_SIZE // 2],
            strides=[p_s0, p_s1],
            tile_shape=[BLOCK_H, PAGE_SIZE // 2],
            tile_dim_map=[0, 1],
        )
        p_sf_view = tl.ext.make_view(
            base=p_sf_ptr,
            shapes=[p_num_rows // 128, SF_PER_PAGE // 4, 2, 256],
            strides=[128 * (((SF_PER_PAGE + 3) // 4) * 4), 512, 256, 1],
            tile_shape=[1, SF_PER_PAGE // 4, 2, 256],
            tile_dim_map=[0, 1, 2, 3],
        )
        v_sf_view = tl.ext.make_view(
            base=v_sf_ptr,
            shapes=[num_pages, V_HEAD_D // 128, SF_PER_PAGE // 4, 2, 256],
            strides=[vsf_s0, 128 * (((SF_PER_PAGE + 3) // 4) * 4), 512, 256, 1],
            tile_shape=[1, 1, SF_PER_PAGE // 4, 2, 256],
            tile_dim_map=[0, 1, 2, 3, 4],
        )
        v_view = tl.ext.make_view(
            base=kv_cache_ptr,
            shapes=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            tile_shape=[1, PAGE_SIZE, BLOCK_V // 2],
            tile_dim_map=[0, 1, 2],
        )
        page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
        global_scale = tl.load(global_scale_ptr)
        out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
        acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
        for page_rel in tl.range(0, MAX_PAGES, num_stages=PV_LOOP_STAGES):
            compact_page = page_table_start + page_rel
            physical_page = tl.load(src_page_ids_ptr + compact_page).to(tl.int64)

            p_vals = tl.ext.load_view_tko(
                p_view,
                [(compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
            )
            p_vals = p_vals.to(tl.uint8, bitcast=True)
            p_scales = tl.ext.load_view_tko(p_sf_view, [compact_page.to(tl.int32), 0, 0, 0])
            p_scales = p_scales.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
            p_scales = p_scales.reshape([BLOCK_H, SF_PER_PAGE])

            v_tile = tl.ext.load_view_tko(
                v_view,
                [
                    physical_page.to(tl.int32),
                    0,
                    (dim_block * (BLOCK_V // 2)).to(tl.int32),
                ],
            )
            v_tile = v_tile.to(tl.uint8, bitcast=True)
            v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
            v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
            even_packed, odd_packed = tl.split(v_pairs)
            low_vals = _fp4_pack_low_nibbles(even_packed, odd_packed)
            high_vals = _fp4_pack_high_nibbles(even_packed, odd_packed)
            v_vals = tl.reshape(
                tl.join(low_vals, high_vals).permute(0, 2, 1),
                (BLOCK_V, PAGE_SIZE // 2),
            )
            v_scales = tl.ext.load_view_tko(
                v_sf_view,
                [
                    physical_page.to(tl.int32),
                    dim_block,
                    0,
                    0,
                    0,
                ],
            )
            v_scales = v_scales.reshape([1, 1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
            v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
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

        out_vals = acc * out_scale
        if out_ptr.dtype.element_ty == tl.bfloat16:
            out_vals = out_vals.to(tl.bfloat16)
        elif out_ptr.dtype.element_ty == tl.float16:
            out_vals = out_vals.to(tl.float16)
        out_desc.store(
            [
                (gen_idx * NUM_HEADS + head_block * BLOCK_H).to(tl.int32),
                (dim_block * BLOCK_V).to(tl.int32),
            ],
            out_vals,
        )
        return

    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + gen_idx)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for page_rel in tl.range(0, MAX_PAGES, num_stages=PV_LOOP_STAGES):
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

            p_rows = safe_compact_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows
                if ASSUME_FULL_HEADS
                else tl.where(mask_h, p_rows, safe_compact_page * NUM_HEADS)
            )
            if USE_TMA_P_LOAD:
                p_vals = p_desc.load(
                    [(safe_compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0]
                )
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

    if ASSUME_FULL_HEADS and ASSUME_FULL_V:
        out_vals = acc * out_scale
        if USE_TMA_V_LOAD:
            if out_ptr.dtype.element_ty == tl.bfloat16:
                out_vals = out_vals.to(tl.bfloat16)
            elif out_ptr.dtype.element_ty == tl.float16:
                out_vals = out_vals.to(tl.float16)
            out_desc.store(
                [
                    (gen_idx * NUM_HEADS + head_block * BLOCK_H).to(tl.int32),
                    (dim_block * BLOCK_V).to(tl.int32),
                ],
                out_vals,
            )
        else:
            tl.store(
                out_ptr + gen_idx * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
                out_vals,
            )
    else:
        tl.store(
            out_ptr
            + gen_idx * out_s0
            + safe_offs_h[:, None] * out_s1
            + safe_offs_v[None, :] * out_s2,
            acc * out_scale,
            mask=mask_h[:, None] & mask_v[None, :],
        )


def fp4_mla_paged_attention_internal(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    kv_lens: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    *,
    sm_scale: float,
    num_heads: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    page_size: Optional[int] = None,
    q_residual_dim: int = 0,
    p_global_scale: float = FP4_MLA_P_GLOBAL_SCALE,
    block_h: int = 128,
    block_k: Optional[int] = None,
    block_v: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    max_pages: Optional[int] = None,
    page_pipeline_streams: Optional[int] = None,
    kernel_occupancy: Optional[int] = None,
    kernel_num_ctas: Optional[int] = None,
    kernel_num_stages: Optional[int] = None,
    kernel_num_warps: Optional[int] = None,
    pv_loop_stages: int = 1,
    parallel_page_stats: Optional[bool] = None,
    fused_prob_pack: Optional[bool] = None,
    use_tma_data_load: Optional[bool] = None,
    fused_prob_pack_single_launch: Optional[bool] = None,
    pack_prob_in_page_stats: Optional[bool] = None,
    assume_full_pages: Optional[bool] = None,
    assume_valid_pages: Optional[bool] = None,
    p_fp4_workspace: Optional[torch.Tensor] = None,
    p_sf_workspace: Optional[torch.Tensor] = None,
    p_probs_workspace: Optional[torch.Tensor] = None,
    max_scores_workspace: Optional[torch.Tensor] = None,
    denom_workspace: Optional[torch.Tensor] = None,
    page_max_workspace: Optional[torch.Tensor] = None,
    page_sum_workspace: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    if not hasattr(tl, "dot_scaled"):
        raise NotImplementedError(
            "fp4_mla_paged_attention requires a Triton build with tl.dot_scaled."
        )
    if not q_fp4.is_cuda:
        raise ValueError("q_fp4 must be a CUDA tensor.")
    if q_fp4.dtype != torch.uint8 or kv_cache.dtype != torch.uint8:
        raise TypeError("q_fp4 and kv_cache must be packed FP4 tensors with dtype torch.uint8.")
    if global_scale.numel() < 1:
        raise ValueError("global_scale must contain at least one element.")
    if q_fp4.dim() == 3:
        inferred_num_gen, inferred_num_heads, packed_q_dim = q_fp4.shape
        if num_heads is not None and num_heads != inferred_num_heads:
            raise ValueError(
                f"num_heads={num_heads} does not match q_fp4.shape[1]={inferred_num_heads}."
            )
        num_gen = inferred_num_gen
        num_heads = inferred_num_heads
        q_fp4_2d = q_fp4.reshape(num_gen * num_heads, packed_q_dim)
    elif q_fp4.dim() == 2:
        if num_heads is None:
            raise ValueError("num_heads is required when q_fp4 is 2D.")
        if q_fp4.shape[0] % num_heads != 0:
            raise ValueError("q_fp4.shape[0] must be divisible by num_heads.")
        num_gen = q_fp4.shape[0] // num_heads
        packed_q_dim = q_fp4.shape[1]
        q_fp4_2d = q_fp4
    else:
        raise ValueError("q_fp4 must be 2D or 3D.")

    num_pages, inferred_page_size, packed_k_dim, kv_s0, kv_s2, kv_s4 = _get_kv_cache_strides(
        kv_cache
    )
    if page_size is None:
        page_size = inferred_page_size
    if page_size != inferred_page_size:
        raise ValueError(
            f"page_size={page_size} does not match kv_cache page dimension {inferred_page_size}."
        )
    if page_size % FP4_BLOCK_SIZE != 0:
        raise ValueError(f"page_size must be divisible by {FP4_BLOCK_SIZE}.")

    q_head_dim = packed_q_dim * 2
    k_head_dim = packed_k_dim * 2
    if q_residual_dim < 0 or q_residual_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(f"q_residual_dim must be a non-negative multiple of {FP4_BLOCK_SIZE}.")
    if q_head_dim - q_residual_dim != k_head_dim:
        raise ValueError(
            f"q_head_dim - q_residual_dim must match K head dim: {q_head_dim} - {q_residual_dim} != {k_head_dim}."
        )
    if q_head_dim % FP4_BLOCK_SIZE != 0 or k_head_dim % FP4_BLOCK_SIZE != 0:
        raise ValueError(f"Q/K head dims must be divisible by {FP4_BLOCK_SIZE}.")
    if v_head_dim is None:
        v_head_dim = k_head_dim
    if v_head_dim <= 0 or v_head_dim > k_head_dim:
        raise ValueError(f"v_head_dim must be in (0, {k_head_dim}], got {v_head_dim}.")
    q_sf_flat = q_sf.contiguous().view(-1)
    if sf_cache.shape[0] < num_pages or v_sf.shape[0] < num_pages:
        raise ValueError("sf_cache and v_sf must have a leading physical-page dimension.")
    if q_sf_flat.numel() < _swizzled_scale_size(num_gen * num_heads, q_head_dim):
        raise ValueError("q_sf is too small for the swizzled Q scale layout.")
    if sf_cache.numel() < sf_cache.shape[0] * _swizzled_scale_size(page_size, k_head_dim):
        raise ValueError("sf_cache is too small for the swizzled K scale layout.")
    if v_sf.numel() < v_sf.shape[0] * _swizzled_scale_size(v_head_dim, page_size):
        raise ValueError("v_sf is too small for the swizzled V scale layout.")

    if output is None:
        output = torch.empty(
            (num_gen, num_heads, v_head_dim), dtype=output_dtype, device=q_fp4.device
        )
    elif output.shape != (num_gen, num_heads, v_head_dim):
        raise ValueError(
            f"output must have shape {(num_gen, num_heads, v_head_dim)}, got {tuple(output.shape)}."
        )

    if num_gen == 0:
        return output
    triton_backend = "nvt"
    if block_k is None:
        block_k = 512 if triton_backend == "nvt" else 256
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    if max_pages is None:
        if paged_kv_indptr_decode.numel() >= num_gen + 1:
            page_counts = paged_kv_indptr_decode[1 : num_gen + 1] - paged_kv_indptr_decode[:num_gen]
            max_pages = int(page_counts.max().item()) if page_counts.numel() > 0 else 0
        else:
            max_pages = _ceil_div(int(kv_lens[:num_gen].max().item()), page_size)
    if max_pages <= 0:
        output.zero_()
        return output

    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = k_head_dim // FP4_BLOCK_SIZE
    sf_per_page = page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)
    assume_full_heads = num_heads % block_h == 0
    assume_full_v = v_head_dim % block_v == 0
    if assume_full_pages is None:
        assume_full_pages = False
    assume_full_pages = bool(assume_full_pages)
    if assume_valid_pages is None:
        assume_valid_pages = False
    assume_valid_pages = bool(assume_valid_pages)
    total_p_rows = max(src_page_ids.numel() * num_heads, 1)
    if page_pipeline_streams is None:
        if triton_backend == "nvt" and max_pages >= 8 and num_gen >= 128:
            page_pipeline_streams = 2
        else:
            page_pipeline_streams = 1
    page_pipeline_streams = max(1, min(int(page_pipeline_streams), max_pages))
    launch_meta = {}
    if kernel_occupancy is None and triton_backend == "nvt":
        kernel_occupancy = 2
    if kernel_occupancy is not None:
        launch_meta["occupancy"] = int(kernel_occupancy)
    if kernel_num_ctas is not None:
        launch_meta["num_ctas"] = int(kernel_num_ctas)
    if kernel_num_stages is not None:
        launch_meta["num_stages"] = int(kernel_num_stages)
    if kernel_num_warps is not None:
        launch_meta["num_warps"] = int(kernel_num_warps)
    if fused_prob_pack is None:
        fused_prob_pack = triton_backend == "nvt"
    if fused_prob_pack_single_launch is None:
        fused_prob_pack_single_launch = triton_backend == "nvt" and max_pages >= 8
    if use_tma_data_load is None:
        use_tma_data_load = triton_backend == "nvt"
    use_tma_data_load = bool(use_tma_data_load and hasattr(tl, "make_tensor_descriptor"))
    if use_tma_data_load:
        # Device-side descriptors may need Triton's allocator for descriptor scratch storage.
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device=q_fp4.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    p_fp4 = _workspace_tensor(
        p_fp4_workspace,
        (total_p_rows, page_size // 2),
        dtype=torch.uint8,
        device=q_fp4.device,
        name="p_fp4",
    )
    p_sf = _workspace_tensor(
        p_sf_workspace,
        (_swizzled_scale_size(total_p_rows, page_size),),
        dtype=q_sf.dtype,
        device=q_fp4.device,
        name="p_sf",
    )
    if fused_prob_pack:
        p_probs = None
    else:
        p_probs_shape = (max(num_gen * num_heads, 1), page_size)
        if page_pipeline_streams > 1:
            p_probs = _workspace_tensor(
                p_probs_workspace,
                (page_pipeline_streams, *p_probs_shape),
                dtype=torch.float32,
                device=q_fp4.device,
                name="p_probs",
            )
        else:
            p_probs = _workspace_tensor(
                p_probs_workspace,
                p_probs_shape,
                dtype=torch.float32,
                device=q_fp4.device,
                name="p_probs",
            )
    max_scores = _workspace_tensor(
        max_scores_workspace,
        (num_gen, num_heads),
        dtype=torch.float32,
        device=q_fp4.device,
        name="max_scores",
    )
    denom = _workspace_tensor(
        denom_workspace,
        (num_gen, num_heads),
        dtype=torch.float32,
        device=q_fp4.device,
        name="denom",
    )

    if parallel_page_stats is None:
        parallel_page_stats = triton_backend == "nvt" and max_pages >= 8
    if pack_prob_in_page_stats is None:
        pack_prob_in_page_stats = parallel_page_stats and fused_prob_pack
    pack_prob_in_page_stats = bool(
        pack_prob_in_page_stats and parallel_page_stats and fused_prob_pack
    )
    if parallel_page_stats:
        page_stats_shape = (num_gen, max_pages, num_heads)
        page_max = _workspace_tensor(
            page_max_workspace,
            page_stats_shape,
            dtype=torch.float32,
            device=q_fp4.device,
            name="page_max",
        )
        page_sum = _workspace_tensor(
            page_sum_workspace,
            page_stats_shape,
            dtype=torch.float32,
            device=q_fp4.device,
            name="page_sum",
        )
        _fp4_mla_attention_page_stats_kernel[(num_gen, num_head_blocks, max_pages)](
            page_max,
            page_sum,
            p_fp4,
            p_sf,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            src_page_ids.shape[0],
            num_pages,
            q_fp4_2d.stride(0),
            q_fp4_2d.stride(1),
            kv_s0,
            kv_s2,
            kv_s4,
            sf_cache.stride(0),
            page_max.stride(0),
            page_max.stride(1),
            p_fp4.stride(0),
            p_fp4.stride(1),
            p_fp4.shape[0],
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=block_h,
            BLOCK_T=page_size,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            PACK_PROBS=pack_prob_in_page_stats,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **launch_meta,
        )
        _fp4_mla_attention_reduce_stats_kernel[(num_gen, num_head_blocks)](
            max_scores,
            denom,
            page_max,
            page_sum,
            max_scores.stride(0),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=max_pages,
            BLOCK_H=block_h,
            **launch_meta,
        )
        if pack_prob_in_page_stats:
            _fp4_mla_attention_prob_scale_kernel[(num_gen, num_head_blocks, max_pages)](
                p_sf,
                max_scores,
                denom,
                page_max,
                paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                max_scores.stride(0),
                page_max.stride(0),
                page_max.stride(1),
                NUM_HEADS=num_heads,
                PAGE_SIZE=page_size,
                SF_PER_PAGE=sf_per_page,
                BLOCK_H=block_h,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **launch_meta,
            )
    else:
        _fp4_mla_attention_stats_kernel[(num_gen, num_head_blocks)](
            max_scores,
            denom,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            src_page_ids.shape[0],
            num_pages,
            q_fp4_2d.stride(0),
            q_fp4_2d.stride(1),
            kv_s0,
            kv_s2,
            kv_s4,
            sf_cache.stride(0),
            max_scores.stride(0),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            MAX_PAGES=max_pages,
            BLOCK_H=block_h,
            BLOCK_T=page_size,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **launch_meta,
        )

    def _launch_prob_page(page_rel: int, p_probs_slot: Optional[torch.Tensor] = None):
        if fused_prob_pack:
            _fp4_mla_attention_prob_pack_page_fused_kernel[(num_gen, num_head_blocks)](
                p_fp4,
                p_sf,
                max_scores,
                denom,
                q_fp4_2d,
                q_sf_flat,
                kv_cache,
                sf_cache,
                global_scale,
                src_page_ids,
                paged_kv_indptr_decode,
                kv_lens,
                page_rel,
                src_page_ids.shape[0],
                num_pages,
                p_fp4.stride(0),
                p_fp4.stride(1),
                q_fp4_2d.stride(0),
                q_fp4_2d.stride(1),
                kv_s0,
                kv_s2,
                kv_s4,
                sf_cache.stride(0),
                max_scores.stride(0),
                q_fp4_2d.shape[0],
                sm_scale=sm_scale,
                NUM_HEADS=num_heads,
                Q_HEAD_D=q_head_dim,
                K_HEAD_D=k_head_dim,
                Q_RESIDUAL_D=q_residual_dim,
                PAGE_SIZE=page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                Q_SF_PER_TOKEN=q_sf_per_token,
                K_SF_PER_TOKEN=k_sf_per_token,
                SF_PER_PAGE=sf_per_page,
                P_GLOBAL_SCALE=p_global_scale,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                FULL_BLOCK_END=full_block_end,
                TAIL_BLOCK_K=tail_block_k,
                USE_TMA_DATA_LOAD=use_tma_data_load,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **launch_meta,
            )
            return
        assert p_probs_slot is not None
        _fp4_mla_attention_prob_store_page_kernel[(num_gen, num_head_blocks)](
            p_probs_slot,
            max_scores,
            denom,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            page_rel,
            src_page_ids.shape[0],
            num_pages,
            p_probs_slot.stride(0),
            p_probs_slot.stride(1),
            q_fp4_2d.stride(0),
            q_fp4_2d.stride(1),
            kv_s0,
            kv_s2,
            kv_s4,
            sf_cache.stride(0),
            max_scores.stride(0),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **launch_meta,
        )
        _fp4_mla_attention_prob_pack_page_kernel[(num_gen, sf_per_page, num_head_blocks)](
            p_fp4,
            p_sf,
            p_probs_slot,
            paged_kv_indptr_decode,
            kv_lens,
            page_rel,
            src_page_ids.shape[0],
            p_fp4.stride(0),
            p_fp4.stride(1),
            p_probs_slot.stride(0),
            p_probs_slot.stride(1),
            NUM_HEADS=num_heads,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=block_h,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            **launch_meta,
        )

    if pack_prob_in_page_stats:
        pass
    elif fused_prob_pack and fused_prob_pack_single_launch:
        _fp4_mla_attention_prob_pack_page_fused_kernel[(num_gen, num_head_blocks, max_pages)](
            p_fp4,
            p_sf,
            max_scores,
            denom,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            0,
            src_page_ids.shape[0],
            num_pages,
            p_fp4.stride(0),
            p_fp4.stride(1),
            q_fp4_2d.stride(0),
            q_fp4_2d.stride(1),
            kv_s0,
            kv_s2,
            kv_s4,
            sf_cache.stride(0),
            max_scores.stride(0),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            PAGE_REL_FROM_GRID=True,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            **launch_meta,
        )
    elif page_pipeline_streams == 1:
        for page_rel in range(max_pages):
            _launch_prob_page(page_rel, p_probs)
    else:
        current_stream = torch.cuda.current_stream(q_fp4.device)
        streams = [torch.cuda.Stream(device=q_fp4.device) for _ in range(page_pipeline_streams)]
        for stream in streams:
            stream.wait_stream(current_stream)
        for page_rel in range(max_pages):
            stream_idx = page_rel % page_pipeline_streams
            with torch.cuda.stream(streams[stream_idx]):
                if fused_prob_pack:
                    _launch_prob_page(page_rel)
                else:
                    assert p_probs is not None
                    _launch_prob_page(page_rel, p_probs[stream_idx])
        for stream in streams:
            current_stream.wait_stream(stream)

    num_dim_blocks = triton.cdiv(v_head_dim, block_v)
    _fp4_mla_attention_pv_kernel[
        (
            num_gen,
            num_head_blocks,
            num_dim_blocks,
        )
    ](
        output,
        p_fp4,
        p_sf,
        kv_cache,
        v_sf,
        global_scale,
        src_page_ids,
        paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        num_pages,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.shape[0] * output.shape[1],
        p_fp4.stride(0),
        p_fp4.stride(1),
        p_fp4.shape[0],
        kv_s0,
        kv_s2,
        kv_s4,
        v_sf.stride(0),
        NUM_HEADS=num_heads,
        V_HEAD_D=v_head_dim,
        PAGE_SIZE=page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_PAGE=sf_per_page,
        MAX_PAGES=max_pages,
        P_GLOBAL_SCALE=p_global_scale,
        BLOCK_H=block_h,
        BLOCK_V=block_v,
        USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
        USE_TMA_V_LOAD=use_tma_data_load and v_head_dim % block_v == 0,
        PV_LOOP_STAGES=int(pv_loop_stages),
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_FULL_V=assume_full_v,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **launch_meta,
    )
    return output


fp4_mla_paged_attention = fp4_mla_paged_attention_internal
