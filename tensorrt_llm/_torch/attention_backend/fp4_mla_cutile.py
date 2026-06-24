# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FP4 MLA paged decode attention using Triton.

The kernels are adapted from TensorRT-LLM's FP4 MLA decode path.  This module
exposes the attention path for already-packed FP4 Q/K/V tensors and swizzled
FP8 block-scale tensors; quantization and KV-cache update helpers remain outside
this internal op.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

FP4_BLOCK_SIZE = 16
FP4_MLA_P_GLOBAL_SCALE = 448.0 * 6.0


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


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
                f"Cannot allocate {name} while capturing a CUDA graph. Pass a preallocated workspace tensor."
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
def _fp4_e2m1_to_f32(nibble):
    magnitude = nibble & 0x7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    sign = (nibble & 0x8) != 0
    return tl.where(sign, -value, value)


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
def _fp4_pack_nibbles(even_packed, odd_packed):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b32 lo;
            .reg .b32 hi;
            and.b32 lo, $2, 15;
            and.b32 hi, $3, 15;
            shl.b32 hi, hi, 4;
            or.b32 $0, lo, hi;

            shr.u32 lo, $2, 4;
            and.b32 lo, lo, 15;
            and.b32 hi, $3, 240;
            or.b32 $1, lo, hi;
        }
        """,
        constraints="=r,=r,r,r",
        args=[even_packed, odd_packed],
        dtype=(tl.uint8, tl.uint8),
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
    v_view = tl.ext.make_view(
        base=kv_cache_ptr,
        shapes=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        tile_shape=[1, PAGE_SIZE, BLOCK_V // 2],
        tile_dim_map=[0, 1, 2],
    )
    v_tile = tl.ext.load_view_tko(
        v_view,
        [
            page_idx.to(tl.int32),
            0,
            (dim_block * (BLOCK_V // 2)).to(tl.int32),
        ],
    )
    v_tile = v_tile.to(tl.uint8, bitcast=True)
    v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
    v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
    even_packed, odd_packed = tl.split(v_pairs)
    low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
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
    v_view = tl.ext.make_view(
        base=kv_cache_ptr,
        shapes=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        tile_shape=[1, PAGE_SIZE, BLOCK_V // 2],
        tile_dim_map=[0, 1, 2],
    )
    v_tile = tl.ext.load_view_tko(
        v_view,
        [
            page_idx.to(tl.int32),
            0,
            (dim_block * (BLOCK_V // 2)).to(tl.int32),
        ],
    )
    v_tile = v_tile.to(tl.uint8, bitcast=True)
    v_tile = tl.reshape(v_tile, (PAGE_SIZE, BLOCK_V // 2))
    v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
    even_packed, odd_packed = tl.split(v_pairs)
    low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
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
    v_packed: torch.Tensor,
    kv_cache: torch.Tensor,
    page_ids: Optional[torch.Tensor] = None,
    *,
    v_head_dim: int,
    page_size: int,
    block_v: int = 128,
    kernel_occupancy: int = 8,
    kernel_num_stages: int = 1,
) -> None:
    """Populate the V-packed auxiliary cache consumed by the prepacked PV kernel."""
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
    token_start = tl.min(token_offsets, axis=0)
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
            k_vals = k_desc.load(
                [safe_physical_page.to(tl.int32), token_start.to(tl.int32), q_start // 2]
            )
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
                        token_start.to(tl.int32),
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
    QUERY_LEN_PER_SEQ: tl.constexpr,
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
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
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
    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)

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

    tl.store(max_ptr + query_idx * stats_s0 + safe_offs_h, max_score, mask=mask_h)
    tl.store(denom_ptr + query_idx * stats_s0 + safe_offs_h, denom, mask=mask_h)


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
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_BY_QUERY: tl.constexpr,
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
            if P_BY_QUERY:
                p_page = query_idx * MAX_PAGES + page_rel
            else:
                p_page = safe_compact_page
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            )
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
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len: tl.constexpr,
    num_pages: tl.constexpr,
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    page_stats_s0: tl.constexpr,
    page_stats_s1: tl.constexpr,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
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
    P_BY_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    PACK_PROBS: tl.constexpr,
    GROUP_REDUCE_STATS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    MASK_MTP_FINAL_PAGE_ONLY: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr = 2,
    ALLOW_PARTIAL_GROUPS: tl.constexpr = False,
    PAGE_GROUP_OFFSET: tl.constexpr = 0,
    DUPLICATE_TAIL_K: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_group = tl.program_id(2)
    logical_page_group = page_group + PAGE_GROUP_OFFSET
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    q_row_base = query_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    if ASSUME_FULL_PAGES:
        kv_len = 0
    elif MASK_MTP_FINAL_PAGE_ONLY:
        kv_len = MAX_PAGES * PAGE_SIZE - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

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
        block_shape=[BLOCK_H, 32],
    )
    k_tail_desc = tl.make_tensor_descriptor(
        kv_cache_ptr,
        shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        block_shape=[1, BLOCK_T, 32],
    )
    p_desc = tl.make_tensor_descriptor(
        p_fp4_ptr,
        shape=[p_num_rows, PAGE_SIZE // 2],
        strides=[p_s0, p_s1],
        block_shape=[BLOCK_H, PAGE_SIZE // 2],
    )
    q_sf_full_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 8, 2, 256],
        tile_dim_map=[0, 1, 2, 3],
    )
    q_sf_tail_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 1, 2, 256],
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

    q_row_start = (q_row_base + head_block * BLOCK_H).to(tl.int32)
    q_row_group = q_row_base // 128
    full_q_vals = q_desc.load([q_row_start, 0])
    full_q_scales = tl.ext.load_view_tko(q_sf_full_view, [q_row_group.to(tl.int32), 0, 0, 0])
    full_q_scales = full_q_scales.reshape([1, 8, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    full_q_scales = full_q_scales.reshape([BLOCK_H, 32])
    q0_vals = q_tail_desc.load([q_row_start, 256])
    q1_vals = q_tail_desc.load([q_row_start, 288])
    q0_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group.to(tl.int32), 8, 0, 0])
    q0_scales = q0_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q0_scales = q0_scales.reshape([BLOCK_H, 4])
    q1_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group.to(tl.int32), 9, 0, 0])
    q1_scales = q1_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q1_scales = q1_scales.reshape([BLOCK_H, 4])

    group_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = logical_page_group * GROUP_PAGES + page_group_off
        page_start = page_rel * PAGE_SIZE
        valid_group_page = page_rel < MAX_PAGES
        if not ASSUME_FULL_PAGES:
            valid_group_page = valid_group_page & (page_start < kv_len)
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        full_k_vals = k_desc.load([physical_page.to(tl.int32), 0, 0])
        full_k_vals = tl.reshape(full_k_vals, (BLOCK_T, 256))
        full_k_scales = tl.ext.load_view_tko(
            k_sf_full_view, [physical_page.to(tl.int32), 0, 0, 0, 0]
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

        tail_k_vals = k_tail_desc.load([physical_page.to(tl.int32), 0, 256])
        tail_k_vals = tl.reshape(tail_k_vals, (BLOCK_T, 32))
        tail_k_scales = tl.ext.load_view_tko(
            k_sf_tail_view, [physical_page.to(tl.int32), 0, 8, 0, 0]
        )
        tail_k_scales = tail_k_scales.reshape([1, 1, 1, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        tail_k_scales = tail_k_scales.reshape([BLOCK_T, 4])
        if DUPLICATE_TAIL_K:
            q_tail_vals = tl.join(q0_vals, q1_vals).permute(0, 2, 1)
            q_tail_vals = q_tail_vals.reshape([BLOCK_H, 64])
            q_tail_scales = tl.join(q0_scales, q1_scales).permute(0, 2, 1)
            q_tail_scales = q_tail_scales.reshape([BLOCK_H, 8])
            tail_k_groups = tail_k_vals.reshape([BLOCK_T, 4, 8])
            tail_k_dup_vals = tl.join(tail_k_groups, tail_k_groups).permute(0, 1, 3, 2)
            tail_k_dup_vals = tail_k_dup_vals.reshape([BLOCK_T, 64])
            tail_k_dup_scales = tl.join(tail_k_scales, tail_k_scales).permute(0, 2, 1)
            tail_k_dup_scales = tail_k_dup_scales.reshape([BLOCK_T, 8])
            scores = tl.dot_scaled(
                q_tail_vals,
                q_tail_scales,
                "e2m1",
                tail_k_dup_vals.T,
                tail_k_dup_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )
        else:
            scores = tl.dot_scaled(
                q0_vals,
                q0_scales,
                "e2m1",
                tail_k_vals.T,
                tail_k_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )
            scores = tl.dot_scaled(
                q1_vals,
                q1_scales,
                "e2m1",
                tail_k_vals.T,
                tail_k_scales,
                "e2m1",
                acc=scores,
                fast_math=True,
                rhs_k_pack=True,
            )

        scores = scores * qk_scale
        full_softmax_page = ASSUME_FULL_PAGES or (
            MASK_MTP_FINAL_PAGE_ONLY and page_rel < MAX_PAGES - 1
        )
        if ASSUME_FULL_HEADS and (
            (ASSUME_FULL_PAGES and not ALLOW_PARTIAL_GROUPS) or full_softmax_page
        ):
            page_max = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max[:, None]) * 1.4426950408889634)
        else:
            if ASSUME_FULL_PAGES:
                valid_t = tl.full([BLOCK_T], True, dtype=tl.int1)
            else:
                valid_t = page_start + offs_t < kv_len
            scores = tl.where(valid_group_page & valid_t[None, :], scores, -float("inf"))
            page_max = tl.max(scores, axis=1)
            safe_page_max = tl.where(valid_group_page, page_max, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(valid_group_page & valid_t[None, :], exp_scores, 0.0)
        page_sum = tl.sum(exp_scores, axis=1)
        if GROUP_REDUCE_STATS:
            next_group_max = tl.maximum(group_max, page_max)
            old_delta = tl.where(group_sum > 0.0, group_max - next_group_max, 0.0)
            new_delta = tl.where(page_sum > 0.0, page_max - next_group_max, 0.0)
            group_sum = group_sum * tl.math.exp2(
                old_delta * 1.4426950408889634
            ) + page_sum * tl.math.exp2(new_delta * 1.4426950408889634)
            group_max = next_group_max

        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        stored_scale = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

        out_offsets = query_idx * page_stats_s0 + page_rel * page_stats_s1 + offs_h
        tl.store(
            page_max_ptr + out_offsets, page_max, mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS)
        )
        if not GROUP_REDUCE_STATS:
            tl.store(
                page_sum_ptr + out_offsets,
                page_sum,
                mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            )

        if P_BY_QUERY:
            p_page = query_idx * MAX_PAGES + page_rel
        else:
            p_page = safe_compact_page
        sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            p_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
        tl.store(
            p_sf_ptr + sf_offsets, stored_scale, mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS)
        )
        if ALLOW_PARTIAL_GROUPS:
            byte_offsets = tl.arange(0, FP4_BLOCK // 2)
            byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
            safe_p_rows = p_page * NUM_HEADS + offs_h
            tl.store(
                p_fp4_ptr + safe_p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                packed,
                mask=valid_group_page,
            )
        else:
            p_desc.store(
                [(p_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
                tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
            )
    if GROUP_REDUCE_STATS:
        group_max_offsets = (
            query_idx * page_stats_s0 + (logical_page_group * 2) * page_stats_s1 + offs_h
        )
        group_sum_offsets = group_max_offsets + page_stats_s1
        tl.store(page_sum_ptr + group_max_offsets, group_max)
        tl.store(page_sum_ptr + group_sum_offsets, group_sum)


@triton.jit
def _fp4_mla_attention_page_stats_grouped_mtp_pair_kernel(
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
    page_ids_len: tl.constexpr,
    num_pages: tl.constexpr,
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    page_stats_s0: tl.constexpr,
    page_stats_s1: tl.constexpr,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
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
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    PACK_PROBS: tl.constexpr,
    GROUP_REDUCE_STATS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr = 2,
    ALLOW_PARTIAL_GROUPS: tl.constexpr = False,
    Q_PER_GROUP: tl.constexpr = 2,
    DUPLICATE_TAIL_K: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    seq_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    num_query_groups = QUERY_LEN_PER_SEQ // Q_PER_GROUP
    page_group = combo // num_query_groups
    query_group = combo - page_group * num_query_groups
    query_offset0 = query_group * Q_PER_GROUP
    query_offset1 = query_offset0 + 1
    query_idx0 = seq_idx * QUERY_LEN_PER_SEQ + query_offset0
    query_idx1 = query_idx0 + 1

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    q_row_base0 = query_idx0 * NUM_HEADS
    q_row_base1 = query_idx1 * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    if ASSUME_FULL_PAGES:
        kv_len0 = 0
        kv_len1 = 0
    else:
        kv_len_base = tl.load(kv_lens_ptr + seq_idx)
        kv_len0 = tl.maximum(kv_len_base - (QUERY_LEN_PER_SEQ - 1 - query_offset0), 0)
        kv_len1 = tl.maximum(kv_len_base - (QUERY_LEN_PER_SEQ - 1 - query_offset1), 0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

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
        block_shape=[BLOCK_H, 32],
    )
    k_tail_desc = tl.make_tensor_descriptor(
        kv_cache_ptr,
        shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        block_shape=[1, BLOCK_T, 32],
    )
    p_desc = tl.make_tensor_descriptor(
        p_fp4_ptr,
        shape=[p_num_rows, PAGE_SIZE // 2],
        strides=[p_s0, p_s1],
        block_shape=[BLOCK_H, PAGE_SIZE // 2],
    )
    q_sf_full_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 8, 2, 256],
        tile_dim_map=[0, 1, 2, 3],
    )
    q_sf_tail_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 1, 2, 256],
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

    q_row_start0 = (q_row_base0 + head_block * BLOCK_H).to(tl.int32)
    q_row_group0 = q_row_base0 // 128
    full_q_vals0 = q_desc.load([q_row_start0, 0])
    full_q_scales0 = tl.ext.load_view_tko(q_sf_full_view, [q_row_group0.to(tl.int32), 0, 0, 0])
    full_q_scales0 = full_q_scales0.reshape([1, 8, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    full_q_scales0 = full_q_scales0.reshape([BLOCK_H, 32])
    q0_tail0_vals = q_tail_desc.load([q_row_start0, 256])
    q0_tail1_vals = q_tail_desc.load([q_row_start0, 288])
    q0_tail0_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group0.to(tl.int32), 8, 0, 0])
    q0_tail0_scales = q0_tail0_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q0_tail0_scales = q0_tail0_scales.reshape([BLOCK_H, 4])
    q0_tail1_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group0.to(tl.int32), 9, 0, 0])
    q0_tail1_scales = q0_tail1_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q0_tail1_scales = q0_tail1_scales.reshape([BLOCK_H, 4])

    q_row_start1 = (q_row_base1 + head_block * BLOCK_H).to(tl.int32)
    q_row_group1 = q_row_base1 // 128
    full_q_vals1 = q_desc.load([q_row_start1, 0])
    full_q_scales1 = tl.ext.load_view_tko(q_sf_full_view, [q_row_group1.to(tl.int32), 0, 0, 0])
    full_q_scales1 = full_q_scales1.reshape([1, 8, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    full_q_scales1 = full_q_scales1.reshape([BLOCK_H, 32])
    q1_tail0_vals = q_tail_desc.load([q_row_start1, 256])
    q1_tail1_vals = q_tail_desc.load([q_row_start1, 288])
    q1_tail0_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group1.to(tl.int32), 8, 0, 0])
    q1_tail0_scales = q1_tail0_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q1_tail0_scales = q1_tail0_scales.reshape([BLOCK_H, 4])
    q1_tail1_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group1.to(tl.int32), 9, 0, 0])
    q1_tail1_scales = q1_tail1_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q1_tail1_scales = q1_tail1_scales.reshape([BLOCK_H, 4])

    group_max0 = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_sum0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    group_max1 = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_sum1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        page_start = page_rel * PAGE_SIZE
        valid_group_page_base = page_rel < MAX_PAGES
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page_base, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page_base | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        full_k_vals = k_desc.load([physical_page.to(tl.int32), 0, 0])
        full_k_vals = tl.reshape(full_k_vals, (BLOCK_T, 256))
        full_k_scales = tl.ext.load_view_tko(
            k_sf_full_view, [physical_page.to(tl.int32), 0, 0, 0, 0]
        )
        full_k_scales = full_k_scales.reshape([1, 1, 8, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        full_k_scales = full_k_scales.reshape([BLOCK_T, 32])
        tail_k_vals = k_tail_desc.load([physical_page.to(tl.int32), 0, 256])
        tail_k_vals = tl.reshape(tail_k_vals, (BLOCK_T, 32))
        tail_k_scales = tl.ext.load_view_tko(
            k_sf_tail_view, [physical_page.to(tl.int32), 0, 8, 0, 0]
        )
        tail_k_scales = tail_k_scales.reshape([1, 1, 1, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        tail_k_scales = tail_k_scales.reshape([BLOCK_T, 4])

        scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        scores = tl.dot_scaled(
            full_q_vals0,
            full_q_scales0,
            "e2m1",
            full_k_vals.T,
            full_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q0_tail0_vals,
            q0_tail0_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q0_tail1_vals,
            q0_tail1_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = scores * qk_scale
        valid_group_page0 = valid_group_page_base
        if not ASSUME_FULL_PAGES:
            valid_group_page0 = valid_group_page0 & (page_start < kv_len0)
        if ASSUME_FULL_PAGES:
            valid_t0 = tl.full([BLOCK_T], True, dtype=tl.int1)
        else:
            valid_t0 = page_start + offs_t < kv_len0
        if ASSUME_FULL_HEADS and ASSUME_FULL_PAGES and not ALLOW_PARTIAL_GROUPS:
            page_max0 = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max0[:, None]) * 1.4426950408889634)
        else:
            scores = tl.where(valid_group_page0 & valid_t0[None, :], scores, -float("inf"))
            page_max0 = tl.max(scores, axis=1)
            safe_page_max0 = tl.where(valid_group_page0, page_max0, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max0[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(valid_group_page0 & valid_t0[None, :], exp_scores, 0.0)
        page_sum0 = tl.sum(exp_scores, axis=1)
        if GROUP_REDUCE_STATS:
            next_group_max0 = tl.maximum(group_max0, page_max0)
            old_delta0 = tl.where(group_sum0 > 0.0, group_max0 - next_group_max0, 0.0)
            new_delta0 = tl.where(page_sum0 > 0.0, page_max0 - next_group_max0, 0.0)
            group_sum0 = group_sum0 * tl.math.exp2(
                old_delta0 * 1.4426950408889634
            ) + page_sum0 * tl.math.exp2(new_delta0 * 1.4426950408889634)
            group_max0 = next_group_max0
        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        stored_scale = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)
        out_offsets0 = query_idx0 * page_stats_s0 + page_rel * page_stats_s1 + offs_h
        tl.store(
            page_max_ptr + out_offsets0,
            page_max0,
            mask=valid_group_page0 | (not ALLOW_PARTIAL_GROUPS),
        )
        if not GROUP_REDUCE_STATS:
            tl.store(
                page_sum_ptr + out_offsets0,
                page_sum0,
                mask=valid_group_page0 | (not ALLOW_PARTIAL_GROUPS),
            )
        p_page0 = query_idx0 * MAX_PAGES + page_rel
        sf_offsets0 = _fp4_mla_swizzled_sf_offset_row_block(
            p_page0, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
        tl.store(
            p_sf_ptr + sf_offsets0,
            stored_scale,
            mask=valid_group_page0 | (not ALLOW_PARTIAL_GROUPS),
        )
        if ALLOW_PARTIAL_GROUPS:
            byte_offsets = tl.arange(0, FP4_BLOCK // 2)
            byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
            safe_p_rows0 = p_page0 * NUM_HEADS + offs_h
            tl.store(
                p_fp4_ptr + safe_p_rows0[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                packed,
                mask=valid_group_page0,
            )
        else:
            p_desc.store(
                [(p_page0 * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
                tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
            )

        scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        scores = tl.dot_scaled(
            full_q_vals1,
            full_q_scales1,
            "e2m1",
            full_k_vals.T,
            full_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q1_tail0_vals,
            q1_tail0_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q1_tail1_vals,
            q1_tail1_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = scores * qk_scale
        valid_group_page1 = valid_group_page_base
        if not ASSUME_FULL_PAGES:
            valid_group_page1 = valid_group_page1 & (page_start < kv_len1)
        if ASSUME_FULL_PAGES:
            valid_t1 = tl.full([BLOCK_T], True, dtype=tl.int1)
        else:
            valid_t1 = page_start + offs_t < kv_len1
        if ASSUME_FULL_HEADS and ASSUME_FULL_PAGES and not ALLOW_PARTIAL_GROUPS:
            page_max1 = tl.max(scores, axis=1)
            exp_scores = tl.math.exp2((scores - page_max1[:, None]) * 1.4426950408889634)
        else:
            scores = tl.where(valid_group_page1 & valid_t1[None, :], scores, -float("inf"))
            page_max1 = tl.max(scores, axis=1)
            safe_page_max1 = tl.where(valid_group_page1, page_max1, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max1[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(valid_group_page1 & valid_t1[None, :], exp_scores, 0.0)
        page_sum1 = tl.sum(exp_scores, axis=1)
        if GROUP_REDUCE_STATS:
            next_group_max1 = tl.maximum(group_max1, page_max1)
            old_delta1 = tl.where(group_sum1 > 0.0, group_max1 - next_group_max1, 0.0)
            new_delta1 = tl.where(page_sum1 > 0.0, page_max1 - next_group_max1, 0.0)
            group_sum1 = group_sum1 * tl.math.exp2(
                old_delta1 * 1.4426950408889634
            ) + page_sum1 * tl.math.exp2(new_delta1 * 1.4426950408889634)
            group_max1 = next_group_max1
        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        stored_scale = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)
        out_offsets1 = query_idx1 * page_stats_s0 + page_rel * page_stats_s1 + offs_h
        tl.store(
            page_max_ptr + out_offsets1,
            page_max1,
            mask=valid_group_page1 | (not ALLOW_PARTIAL_GROUPS),
        )
        if not GROUP_REDUCE_STATS:
            tl.store(
                page_sum_ptr + out_offsets1,
                page_sum1,
                mask=valid_group_page1 | (not ALLOW_PARTIAL_GROUPS),
            )
        p_page1 = query_idx1 * MAX_PAGES + page_rel
        sf_offsets1 = _fp4_mla_swizzled_sf_offset_row_block(
            p_page1, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
        tl.store(
            p_sf_ptr + sf_offsets1,
            stored_scale,
            mask=valid_group_page1 | (not ALLOW_PARTIAL_GROUPS),
        )
        if ALLOW_PARTIAL_GROUPS:
            byte_offsets = tl.arange(0, FP4_BLOCK // 2)
            byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
            safe_p_rows1 = p_page1 * NUM_HEADS + offs_h
            tl.store(
                p_fp4_ptr + safe_p_rows1[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                packed,
                mask=valid_group_page1,
            )
        else:
            p_desc.store(
                [(p_page1 * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
                tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
            )

    if GROUP_REDUCE_STATS:
        group_max_offsets0 = query_idx0 * page_stats_s0 + (page_group * 2) * page_stats_s1 + offs_h
        group_sum_offsets0 = group_max_offsets0 + page_stats_s1
        tl.store(page_sum_ptr + group_max_offsets0, group_max0)
        tl.store(page_sum_ptr + group_sum_offsets0, group_sum0)
        group_max_offsets1 = query_idx1 * page_stats_s0 + (page_group * 2) * page_stats_s1 + offs_h
        group_sum_offsets1 = group_max_offsets1 + page_stats_s1
        tl.store(page_sum_ptr + group_max_offsets1, group_max1)
        tl.store(page_sum_ptr + group_sum_offsets1, group_sum1)


@triton.jit
def _fp4_mla_attention_page_stats_grouped_generic_kernel(
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
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    page_stats_s0: tl.constexpr,
    page_stats_s1: tl.constexpr,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
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
    GROUP_REDUCE_STATS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    MASK_MTP_FINAL_PAGE_ONLY: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr = 2,
    ALLOW_PARTIAL_GROUPS: tl.constexpr = False,
    PAGE_GROUP_OFFSET: tl.constexpr = 0,
    DUPLICATE_TAIL_K: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_group = tl.program_id(2)
    logical_page_group = page_group + PAGE_GROUP_OFFSET

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    q_row_base = gen_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
    p_desc = tl.make_tensor_descriptor(
        p_fp4_ptr,
        shape=[p_num_rows, PAGE_SIZE // 2],
        strides=[p_s0, p_s1],
        block_shape=[BLOCK_H, PAGE_SIZE // 2],
    )

    group_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = logical_page_group * GROUP_PAGES + page_group_off
        valid_group_page = page_rel < MAX_PAGES
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)

        scores = _fp4_mla_qk_scores_tile(
            q_fp4_ptr,
            q_sf_ptr,
            kv_cache_ptr,
            sf_cache_ptr,
            src_page_ids_ptr,
            safe_compact_page,
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
        scores = scores * qk_scale
        page_max = tl.max(scores, axis=1)
        if ALLOW_PARTIAL_GROUPS:
            page_max = tl.where(valid_group_page, page_max, -float("inf"))
            safe_page_max = tl.where(valid_group_page, page_max, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_max[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(valid_group_page, exp_scores, 0.0)
        else:
            exp_scores = tl.math.exp2((scores - page_max[:, None]) * 1.4426950408889634)
        page_sum = tl.sum(exp_scores, axis=1)
        if GROUP_REDUCE_STATS:
            next_group_max = tl.maximum(group_max, page_max)
            old_delta = tl.where(group_sum > 0.0, group_max - next_group_max, 0.0)
            new_delta = tl.where(page_sum > 0.0, page_max - next_group_max, 0.0)
            group_sum = group_sum * tl.math.exp2(
                old_delta * 1.4426950408889634
            ) + page_sum * tl.math.exp2(new_delta * 1.4426950408889634)
            group_max = next_group_max

        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        stored_scale = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

        out_offsets = gen_idx * page_stats_s0 + page_rel * page_stats_s1 + offs_h
        tl.store(
            page_max_ptr + out_offsets, page_max, mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS)
        )
        if not GROUP_REDUCE_STATS:
            tl.store(
                page_sum_ptr + out_offsets,
                page_sum,
                mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            )

        safe_p_rows = safe_compact_page * NUM_HEADS + offs_h
        sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
        )
        tl.store(
            p_sf_ptr + sf_offsets, stored_scale, mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS)
        )
        if ALLOW_PARTIAL_GROUPS:
            byte_offsets = tl.arange(0, FP4_BLOCK // 2)
            byte_cols = scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
            tl.store(
                p_fp4_ptr + safe_p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
                packed,
                mask=valid_group_page,
            )
        else:
            p_desc.store(
                [(compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
                tl.reshape(packed, (BLOCK_H, PAGE_SIZE // 2)),
            )

    if GROUP_REDUCE_STATS:
        group_max_offsets = (
            gen_idx * page_stats_s0 + (logical_page_group * 2) * page_stats_s1 + offs_h
        )
        group_sum_offsets = group_max_offsets + page_stats_s1
        tl.store(page_sum_ptr + group_max_offsets, group_max)
        tl.store(page_sum_ptr + group_sum_offsets, group_sum)


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
    GROUP_REDUCE_STATS: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    NUM_PAGE_GROUPS: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < NUM_HEADS
    safe_offs_h = tl.where(mask_h, offs_h, 0)

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    if GROUP_REDUCE_STATS:
        for group_rel in tl.range(0, NUM_PAGE_GROUPS):
            group_max = tl.load(
                page_sum_ptr
                + gen_idx * page_stats_s0
                + (group_rel * 2) * page_stats_s1
                + safe_offs_h,
                mask=mask_h,
                other=-float("inf"),
            )
            max_score = tl.maximum(max_score, group_max)
    else:
        for page_rel in tl.range(0, MAX_PAGES):
            page_max = tl.load(
                page_max_ptr + gen_idx * page_stats_s0 + page_rel * page_stats_s1 + safe_offs_h,
                mask=mask_h,
                other=-float("inf"),
            )
            max_score = tl.maximum(max_score, page_max)

    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if GROUP_REDUCE_STATS:
        for group_rel in tl.range(0, NUM_PAGE_GROUPS):
            group_max = tl.load(
                page_sum_ptr
                + gen_idx * page_stats_s0
                + (group_rel * 2) * page_stats_s1
                + safe_offs_h,
                mask=mask_h,
                other=-float("inf"),
            )
            group_sum = tl.load(
                page_sum_ptr
                + gen_idx * page_stats_s0
                + (group_rel * 2 + 1) * page_stats_s1
                + safe_offs_h,
                mask=mask_h,
                other=0.0,
            )
            denom += tl.where(
                group_sum > 0.0,
                group_sum * tl.math.exp2((group_max - max_score) * 1.4426950408889634),
                0.0,
            )
    else:
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
def _fp4_mla_attention_page_stats_half_grouped_kernel(
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
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    page_stats_s0: tl.constexpr,
    page_stats_s1: tl.constexpr,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
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
    GROUP_REDUCE_STATS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    MAX_TILES: tl.constexpr,
    GROUP_TILES: tl.constexpr,
    TILES_PER_PAGE: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr = False,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    tile_group = tl.program_id(2)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    local_t = tl.arange(0, BLOCK_T)
    local_scale_cols = tl.arange(0, BLOCK_T // FP4_BLOCK)
    q_row_base = gen_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)

    group_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for group_off in tl.range(0, GROUP_TILES):
        tile_rel = tile_group * GROUP_TILES + group_off
        valid_tile = tile_rel < MAX_TILES
        page_rel = tile_rel // TILES_PER_PAGE
        tile_in_page = tile_rel - page_rel * TILES_PER_PAGE
        token_base = tile_in_page * BLOCK_T
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_tile, compact_page, page_table_start)

        scores = _fp4_mla_qk_scores_tile(
            q_fp4_ptr,
            q_sf_ptr,
            kv_cache_ptr,
            sf_cache_ptr,
            src_page_ids_ptr,
            safe_compact_page,
            q_row_base,
            head_block * BLOCK_H,
            offs_h,
            token_base + local_t,
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
        scores = scores * qk_scale
        tile_max = tl.max(scores, axis=1)
        tile_max = tl.where(valid_tile, tile_max, -float("inf"))
        safe_tile_max = tl.where(valid_tile, tile_max, 0.0)
        exp_scores = tl.math.exp2((scores - safe_tile_max[:, None]) * 1.4426950408889634)
        exp_scores = tl.where(valid_tile, exp_scores, 0.0)
        tile_sum = tl.sum(exp_scores, axis=1)
        if GROUP_REDUCE_STATS:
            next_group_max = tl.maximum(group_max, tile_max)
            old_delta = tl.where(group_sum > 0.0, group_max - next_group_max, 0.0)
            new_delta = tl.where(tile_sum > 0.0, tile_max - next_group_max, 0.0)
            group_sum = group_sum * tl.math.exp2(
                old_delta * 1.4426950408889634
            ) + tile_sum * tl.math.exp2(new_delta * 1.4426950408889634)
            group_max = next_group_max

        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, BLOCK_T // FP4_BLOCK, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        stored_scale = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        scaled_probs = grouped_probs * tl.reshape(
            inv_local_scale, (BLOCK_H, BLOCK_T // FP4_BLOCK, 1)
        )
        pairs = tl.reshape(scaled_probs, (BLOCK_H, BLOCK_T // FP4_BLOCK, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        packed = _fp4_e2m1_quantize_packed(even_probs, odd_probs)

        out_offsets = gen_idx * page_stats_s0 + tile_rel * page_stats_s1 + offs_h
        tl.store(page_max_ptr + out_offsets, tile_max, mask=valid_tile)
        if not GROUP_REDUCE_STATS:
            tl.store(page_sum_ptr + out_offsets, tile_sum, mask=valid_tile)

        sf_cols = tile_in_page * (BLOCK_T // FP4_BLOCK) + local_scale_cols
        sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            safe_compact_page, offs_h[:, None], sf_cols[None, :], SF_PER_PAGE
        )
        tl.store(p_sf_ptr + sf_offsets, stored_scale, mask=valid_tile)

        byte_offsets = tl.arange(0, FP4_BLOCK // 2)
        byte_cols = (
            token_base // 2 + local_scale_cols[:, None] * (FP4_BLOCK // 2) + byte_offsets[None, :]
        )
        p_rows = safe_compact_page * NUM_HEADS + offs_h
        tl.store(
            p_fp4_ptr + p_rows[:, None, None] * p_s0 + byte_cols[None, :, :] * p_s1,
            packed,
            mask=valid_tile,
        )

    if GROUP_REDUCE_STATS:
        group_max_offsets = gen_idx * page_stats_s0 + (tile_group * 2) * page_stats_s1 + offs_h
        group_sum_offsets = group_max_offsets + page_stats_s1
        tl.store(page_sum_ptr + group_max_offsets, group_max)
        tl.store(page_sum_ptr + group_sum_offsets, group_sum)


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
    P_BY_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
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
    factor = tl.where(
        denom > 0.0, tl.math.exp2((page_max - max_score) * 1.4426950408889634) / denom, 0.0
    )

    if P_BY_QUERY:
        p_page = query_idx * MAX_PAGES + page_rel
    else:
        p_page = compact_page
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
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
def _fp4_mla_attention_prob_scale_half_kernel(
    p_sf_ptr,
    max_ptr,
    denom_ptr,
    tile_max_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    stats_s0,
    tile_stats_s0,
    tile_stats_s1,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_TILES: tl.constexpr,
    TILES_PER_PAGE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    tile_rel = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ
    page_rel = tile_rel // TILES_PER_PAGE
    tile_in_page = tile_rel - page_rel * TILES_PER_PAGE

    page_start = page_rel * PAGE_SIZE + tile_in_page * BLOCK_T
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

    tile_max = tl.load(
        tile_max_ptr + query_idx * tile_stats_s0 + tile_rel * tile_stats_s1 + safe_offs_h,
        mask=mask_h,
        other=-float("inf"),
    )
    max_score = tl.load(max_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    factor = tl.where(
        denom > 0.0, tl.math.exp2((tile_max - max_score) * 1.4426950408889634) / denom, 0.0
    )

    scale_cols = tile_in_page * (BLOCK_T // 16) + tl.arange(0, BLOCK_T // 16)
    if ASSUME_FULL_HEADS and ASSUME_VALID_PAGES and NUM_HEADS == 128 and BLOCK_H == 128:
        sf_offsets = _fp4_mla_swizzled_sf_offset_row_block(
            compact_page, offs_h[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    else:
        p_rows = compact_page * NUM_HEADS + offs_h
        safe_p_rows = (
            p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, compact_page * NUM_HEADS)
        )
        sf_offsets = _fp4_mla_swizzled_sf_offset(
            safe_p_rows[:, None], scale_cols[None, :], SF_PER_PAGE
        )
    scales = tl.load(p_sf_ptr + sf_offsets, mask=mask_h[:, None], other=1.0).to(tl.float32)
    tl.store(p_sf_ptr + sf_offsets, scales * factor[:, None], mask=mask_h[:, None])


@triton.jit
def _fp4_mla_attention_prob_scale_from_group_stats_kernel(
    p_sf_ptr,
    page_max_ptr,
    page_sum_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len,
    page_stats_s0,
    page_stats_s1,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_BY_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_PAGE_GROUPS: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    PAGE_REL_OFFSET: tl.constexpr = 0,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_rel = tl.program_id(2) + PAGE_REL_OFFSET
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
    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    for group_rel in tl.range(0, NUM_PAGE_GROUPS):
        group_max = tl.load(
            page_sum_ptr
            + query_idx * page_stats_s0
            + (group_rel * 2) * page_stats_s1
            + safe_offs_h,
            mask=mask_h,
            other=-float("inf"),
        )
        max_score = tl.maximum(max_score, group_max)

    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for group_rel in tl.range(0, NUM_PAGE_GROUPS):
        group_max = tl.load(
            page_sum_ptr
            + query_idx * page_stats_s0
            + (group_rel * 2) * page_stats_s1
            + safe_offs_h,
            mask=mask_h,
            other=-float("inf"),
        )
        group_sum = tl.load(
            page_sum_ptr
            + query_idx * page_stats_s0
            + (group_rel * 2 + 1) * page_stats_s1
            + safe_offs_h,
            mask=mask_h,
            other=0.0,
        )
        denom += tl.where(
            group_sum > 0.0,
            group_sum * tl.math.exp2((group_max - max_score) * 1.4426950408889634),
            0.0,
        )

    factor = tl.where(
        denom > 0.0, tl.math.exp2((page_max - max_score) * 1.4426950408889634) / denom, 0.0
    )

    if P_BY_QUERY:
        p_page = query_idx * MAX_PAGES + page_rel
    else:
        p_page = compact_page
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
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
    QUERY_LEN_PER_SEQ: tl.constexpr,
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
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
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
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return
    q_row_base = query_idx * NUM_HEADS

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
    max_score = tl.load(max_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=1.0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)
    denom_valid = denom > 0.0
    safe_denom = tl.where(denom_valid, denom, 1.0)
    safe_max = tl.where(denom_valid, max_score, 0.0)
    scores = tl.where(mask_h[:, None] & valid_t[None, :], scores * qk_scale, -float("inf"))
    probs = tl.math.exp2((scores - safe_max[:, None]) * 1.4426950408889634) / safe_denom[:, None]
    probs = tl.where(mask_h[:, None] & valid_t[None, :] & denom_valid[:, None], probs, 0.0)

    prob_rows = query_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, query_idx * NUM_HEADS)
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
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_BY_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    token_group = tl.program_id(1)
    head_block = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    page_start = page_rel * PAGE_SIZE
    if (not ASSUME_FULL_PAGES) and page_start >= kv_len:
        return

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
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

    prob_rows = query_idx * NUM_HEADS + offs_h
    safe_prob_rows = tl.where(mask_h, prob_rows, query_idx * NUM_HEADS)
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

    if P_BY_QUERY:
        p_page = query_idx * MAX_PAGES + page_rel
    else:
        p_page = compact_page
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = tl.where(mask_h, p_rows, p_page * NUM_HEADS)
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
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_BY_QUERY: tl.constexpr,
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
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    if PAGE_REL_FROM_GRID:
        page_rel = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
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
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    compact_page = page_table_start + page_rel
    if (compact_page < 0) | (compact_page >= page_ids_len):
        return
    q_row_base = query_idx * NUM_HEADS

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
    max_score = tl.load(max_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=0.0)
    denom = tl.load(denom_ptr + query_idx * stats_s0 + safe_offs_h, mask=mask_h, other=1.0)
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

    if P_BY_QUERY:
        p_page = query_idx * MAX_PAGES + page_rel
    else:
        p_page = compact_page
    p_rows = p_page * NUM_HEADS + offs_h
    safe_p_rows = tl.where(mask_h, p_rows, p_page * NUM_HEADS)
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
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_BY_QUERY: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_TMA_P_LOAD: tl.constexpr,
    USE_TMA_V_LOAD: tl.constexpr,
    USE_TMA_OUT_STORE: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_FULL_V: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
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
    if USE_TMA_OUT_STORE and ASSUME_FULL_HEADS and ASSUME_FULL_V:
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

    can_use_view_pv_fast_path = (
        USE_TMA_P_LOAD
        and USE_TMA_V_LOAD
        and USE_TMA_OUT_STORE
        and ASSUME_FULL_HEADS
        and ASSUME_FULL_PAGES
        and ASSUME_FULL_V
        and ASSUME_VALID_PAGES
        and not P_BY_QUERY
        and NUM_HEADS == 128
        and V_HEAD_D == 512
        and PAGE_SIZE == 128
        and BLOCK_H == 128
        and (BLOCK_V == 128 or BLOCK_V == 256)
        and SF_PER_PAGE == 8
    )
    if can_use_view_pv_fast_path:
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
            tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
            tile_dim_map=[0, 1, 2, 3, 4],
        )
        v_view = tl.ext.make_view(
            base=kv_cache_ptr,
            shapes=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            tile_shape=[1, PAGE_SIZE, BLOCK_V // 2],
            tile_dim_map=[0, 1, 2],
        )
        page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
        global_scale = tl.load(global_scale_ptr)
        out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
        acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
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
            low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
            v_vals = tl.reshape(
                tl.join(low_vals, high_vals).permute(0, 2, 1),
                (BLOCK_V, PAGE_SIZE // 2),
            )
            v_scales = tl.ext.load_view_tko(
                v_sf_view,
                [
                    physical_page.to(tl.int32),
                    dim_block * (BLOCK_V // 128),
                    0,
                    0,
                    0,
                ],
            )
            v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
                0, 1, 4, 3, 2, 5
            )
            v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
            acc = tl.ext.dot_scaled(
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

        out_vals = acc.T * out_scale
        if out_ptr.dtype.element_ty == tl.bfloat16:
            out_vals = out_vals.to(tl.bfloat16)
        elif out_ptr.dtype.element_ty == tl.float16:
            out_vals = out_vals.to(tl.float16)
        if USE_TMA_OUT_STORE:
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
        return

    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
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

            if P_BY_QUERY:
                p_page = query_idx * MAX_PAGES + page_rel
            else:
                p_page = safe_compact_page
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            )
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
                low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
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
            acc * out_scale,
            mask=mask_h[:, None] & mask_v[None, :],
        )


@triton.jit
def _fp4_mla_attention_pv_prepacked_v_kernel(
    out_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    kv_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    max_ptr,
    denom_ptr,
    page_max_ptr,
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
    stats_s0,
    page_stats_s0,
    page_stats_s1,
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
    P_BY_QUERY: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_TMA_P_LOAD: tl.constexpr,
    USE_TMA_V_LOAD: tl.constexpr,
    USE_TMA_OUT_STORE: tl.constexpr,
    USE_PREPACKED_V: tl.constexpr,
    PV_M_PACKED_V: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    PV_APPLY_PROB_SCALE: tl.constexpr,
    PV_SCALE_IN_SF: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_FULL_PAGES: tl.constexpr,
    ASSUME_FULL_V: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
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
    if USE_TMA_OUT_STORE and ASSUME_FULL_HEADS and ASSUME_FULL_V:
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
    if USE_PREPACKED_V:
        v_packed_desc = tl.make_tensor_descriptor(
            v_packed_ptr,
            shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
            strides=[PAGE_SIZE // 2, 1],
            block_shape=[BLOCK_V, PAGE_SIZE // 2],
        )
    can_use_view_pv_fast_path = (
        USE_TMA_P_LOAD
        and USE_TMA_V_LOAD
        and USE_TMA_OUT_STORE
        and ASSUME_FULL_HEADS
        and ASSUME_FULL_PAGES
        and ASSUME_FULL_V
        and ASSUME_VALID_PAGES
        and not P_BY_QUERY
        and NUM_HEADS == 128
        and V_HEAD_D == 512
        and PAGE_SIZE == 128
        and BLOCK_H == 128
        and (BLOCK_V == 128 or BLOCK_V == 256)
        and SF_PER_PAGE == 8
    )
    if can_use_view_pv_fast_path:
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
            tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
            tile_dim_map=[0, 1, 2, 3, 4],
        )
        v_view = tl.ext.make_view(
            base=kv_cache_ptr,
            shapes=[num_pages, PAGE_SIZE, V_HEAD_D // 2],
            strides=[kv_s0, kv_s2, kv_s4],
            tile_shape=[1, PAGE_SIZE, BLOCK_V // 2],
            tile_dim_map=[0, 1, 2],
        )
        page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
        global_scale = tl.load(global_scale_ptr)
        out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
        if PV_APPLY_PROB_SCALE:
            max_score = tl.load(max_ptr + query_idx * stats_s0 + offs_h)
            denom = tl.load(denom_ptr + query_idx * stats_s0 + offs_h)
        else:
            max_score = tl.zeros((BLOCK_H,), dtype=tl.float32)
            denom = tl.full((BLOCK_H,), 1.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for page_rel in tl.range(0, MAX_PAGES, num_stages=PV_LOOP_STAGES):
            compact_page = page_table_start + page_rel
            physical_page = tl.load(src_page_ids_ptr + compact_page).to(tl.int64)
            if P_BY_QUERY:
                p_page = query_idx * MAX_PAGES + page_rel
            else:
                p_page = compact_page

            p_vals = tl.ext.load_view_tko(
                p_view,
                [(p_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
            )
            p_vals = p_vals.to(tl.uint8, bitcast=True)
            p_scales = tl.ext.load_view_tko(p_sf_view, [p_page.to(tl.int32), 0, 0, 0])
            p_scales = p_scales.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
            p_scales = p_scales.reshape([BLOCK_H, SF_PER_PAGE])

            if USE_PREPACKED_V:
                v_row = (physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
                v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
            else:
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
                if not PV_M_PACKED_V:
                    v_pairs = tl.reshape(v_tile.T, (BLOCK_V // 2, PAGE_SIZE // 2, 2))
                    even_packed, odd_packed = tl.split(v_pairs)
                    low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
                    v_vals = tl.reshape(
                        tl.join(low_vals, high_vals).permute(0, 2, 1),
                        (BLOCK_V, PAGE_SIZE // 2),
                    )
            v_scales = tl.ext.load_view_tko(
                v_sf_view,
                [
                    physical_page.to(tl.int32),
                    dim_block * (BLOCK_V // 128),
                    0,
                    0,
                    0,
                ],
            )
            v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
                0, 1, 4, 3, 2, 5
            )
            v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
            if PV_M_PACKED_V and not USE_PREPACKED_V:
                acc = tl.ext.dot_scaled(
                    v_tile.T,
                    v_scales,
                    "e2m1",
                    p_vals.T,
                    p_scales,
                    "e2m1",
                    acc=acc,
                    fast_math=True,
                    lhs_k_pack=False,
                    rhs_k_pack=True,
                )
            else:
                if PV_APPLY_PROB_SCALE:
                    page_max = tl.load(
                        page_max_ptr + query_idx * page_stats_s0 + page_rel * page_stats_s1 + offs_h
                    )
                    factor = tl.where(
                        denom > 0.0,
                        tl.math.exp2((page_max - max_score) * 1.4426950408889634) / denom,
                        0.0,
                    )
                    if PV_SCALE_IN_SF:
                        p_scales_with_factor = (p_scales.to(tl.float32) * factor[:, None]).to(
                            tl.float8e4nv
                        )
                        acc = tl.ext.dot_scaled(
                            v_vals,
                            v_scales,
                            "e2m1",
                            p_vals.T,
                            p_scales_with_factor,
                            "e2m1",
                            acc=acc,
                            fast_math=True,
                            rhs_k_pack=True,
                        )
                    else:
                        page_acc = tl.ext.dot_scaled(
                            v_vals,
                            v_scales,
                            "e2m1",
                            p_vals.T,
                            p_scales,
                            "e2m1",
                            acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                            fast_math=True,
                            rhs_k_pack=True,
                        )
                        acc += page_acc * factor[None, :]
                else:
                    acc = tl.ext.dot_scaled(
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

        out_vals = acc.T * out_scale
        if out_ptr.dtype.element_ty == tl.bfloat16:
            out_vals = out_vals.to(tl.bfloat16)
        elif out_ptr.dtype.element_ty == tl.float16:
            out_vals = out_vals.to(tl.float16)
        if USE_TMA_OUT_STORE:
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
        return

    if ASSUME_FULL_PAGES:
        kv_len = 0
    else:
        kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
        kv_len = tl.maximum(kv_len, 0)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
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

            if P_BY_QUERY:
                p_page = query_idx * MAX_PAGES + page_rel
            else:
                p_page = safe_compact_page
            p_rows = p_page * NUM_HEADS + offs_h
            safe_p_rows = (
                p_rows if ASSUME_FULL_HEADS else tl.where(mask_h, p_rows, p_page * NUM_HEADS)
            )
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
                low_vals, high_vals = _fp4_pack_nibbles(even_packed, odd_packed)
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
            acc * out_scale,
            mask=mask_h[:, None] & mask_v[None, :],
        )


@triton.jit
def _fp4_mla_attention_pv_mtp_pair_prepacked_v_kernel(
    out_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    num_pages,
    out_s0,
    out_s1,
    out_s2,
    p_s0,
    p_s1,
    p_num_rows: tl.constexpr,
    vsf_s0: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    Q_PER_GROUP: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    seq_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    query_group = combo // NUM_DIM_BLOCKS
    dim_block = combo - query_group * NUM_DIM_BLOCKS
    query_base = seq_idx * QUERY_LEN_PER_SEQ + query_group * Q_PER_GROUP
    query_idx0 = query_base
    query_idx1 = query_base + 1

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)

    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
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
        tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
        tile_dim_map=[0, 1, 2, 3, 4],
    )
    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    acc0 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    for page_rel in tl.range(0, MAX_PAGES, num_stages=PV_LOOP_STAGES):
        compact_page = page_table_start + page_rel
        physical_page = tl.load(src_page_ids_ptr + compact_page).to(tl.int64)

        p_page0 = query_idx0 * MAX_PAGES + page_rel
        p_vals0 = tl.ext.load_view_tko(
            p_view,
            [(p_page0 * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
        )
        p_vals0 = p_vals0.to(tl.uint8, bitcast=True)
        p_scales0 = tl.ext.load_view_tko(p_sf_view, [p_page0.to(tl.int32), 0, 0, 0])
        p_scales0 = p_scales0.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
        p_scales0 = p_scales0.reshape([BLOCK_H, SF_PER_PAGE])

        p_page1 = query_idx1 * MAX_PAGES + page_rel
        p_vals1 = tl.ext.load_view_tko(
            p_view,
            [(p_page1 * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
        )
        p_vals1 = p_vals1.to(tl.uint8, bitcast=True)
        p_scales1 = tl.ext.load_view_tko(p_sf_view, [p_page1.to(tl.int32), 0, 0, 0])
        p_scales1 = p_scales1.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
        p_scales1 = p_scales1.reshape([BLOCK_H, SF_PER_PAGE])

        v_row = (physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
        v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
        v_scales = tl.ext.load_view_tko(
            v_sf_view,
            [
                physical_page.to(tl.int32),
                dim_block * (BLOCK_V // 128),
                0,
                0,
                0,
            ],
        )
        v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
            0, 1, 4, 3, 2, 5
        )
        v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])

        acc0 = tl.ext.dot_scaled(
            v_vals,
            v_scales,
            "e2m1",
            p_vals0.T,
            p_scales0,
            "e2m1",
            acc=acc0,
            fast_math=True,
            rhs_k_pack=True,
        )
        acc1 = tl.ext.dot_scaled(
            v_vals,
            v_scales,
            "e2m1",
            p_vals1.T,
            p_scales1,
            "e2m1",
            acc=acc1,
            fast_math=True,
            rhs_k_pack=True,
        )

    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    out_vals0 = acc0.T * out_scale
    out_vals1 = acc1.T * out_scale
    if out_ptr.dtype.element_ty == tl.bfloat16:
        out_vals0 = out_vals0.to(tl.bfloat16)
        out_vals1 = out_vals1.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float16:
        out_vals0 = out_vals0.to(tl.float16)
        out_vals1 = out_vals1.to(tl.float16)
    tl.store(
        out_ptr + query_idx0 * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
        out_vals0,
    )
    tl.store(
        out_ptr + query_idx1 * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
        out_vals1,
    )


@triton.jit
def _fp4_mla_attention_online_qkpv_group_kernel(
    partial_o_ptr,
    partial_m_ptr,
    partial_l_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    num_pages: tl.constexpr,
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    vsf_s0: tl.constexpr,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    pm_s0: tl.constexpr,
    pm_s1: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
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
    BLOCK_V: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr,
    FP4_PV: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    page_group = combo // NUM_DIM_BLOCKS
    dim_block = combo - page_group * NUM_DIM_BLOCKS

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    packed_t = tl.arange(0, PAGE_SIZE // 2)
    scale_cols = tl.arange(0, SF_PER_PAGE)
    q_row_base = gen_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

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
        block_shape=[BLOCK_H, 32],
    )
    k_tail_desc = tl.make_tensor_descriptor(
        kv_cache_ptr,
        shape=[num_pages, BLOCK_T, K_HEAD_D // 2],
        strides=[kv_s0, kv_s2, kv_s4],
        block_shape=[1, BLOCK_T, 32],
    )
    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    q_sf_full_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 8, 2, 256],
        tile_dim_map=[0, 1, 2, 3],
    )
    q_sf_tail_view = tl.ext.make_view(
        base=q_sf_ptr,
        shapes=[q_num_rows // 128, ((Q_SF_PER_TOKEN + 3) // 4), 2, 256],
        strides=[128 * (((Q_SF_PER_TOKEN + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, 1, 2, 256],
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
    q_row_start = (q_row_base + head_block * BLOCK_H).to(tl.int32)
    q_row_group = q_row_base // 128
    full_q_vals = q_desc.load([q_row_start, 0])
    full_q_scales = tl.ext.load_view_tko(q_sf_full_view, [q_row_group.to(tl.int32), 0, 0, 0])
    full_q_scales = full_q_scales.reshape([1, 8, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    full_q_scales = full_q_scales.reshape([BLOCK_H, 32])
    q0_vals = q_tail_desc.load([q_row_start, 256])
    q1_vals = q_tail_desc.load([q_row_start, 288])
    q0_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group.to(tl.int32), 8, 0, 0])
    q0_scales = q0_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q0_scales = q0_scales.reshape([BLOCK_H, 4])
    q1_scales = tl.ext.load_view_tko(q_sf_tail_view, [q_row_group.to(tl.int32), 9, 0, 0])
    q1_scales = q1_scales.reshape([1, 1, 32, 4, 4]).trans(0, 3, 2, 1, 4)
    q1_scales = q1_scales.reshape([BLOCK_H, 4])

    group_m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    group_o = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        valid_group_page = page_rel < MAX_PAGES
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        scores = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        full_k_vals = k_desc.load([physical_page.to(tl.int32), 0, 0])
        full_k_vals = tl.reshape(full_k_vals, (BLOCK_T, 256))
        full_k_scales = tl.ext.load_view_tko(
            k_sf_full_view, [physical_page.to(tl.int32), 0, 0, 0, 0]
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

        tail_k_vals = k_tail_desc.load([physical_page.to(tl.int32), 0, 256])
        tail_k_vals = tl.reshape(tail_k_vals, (BLOCK_T, 32))
        tail_k_scales = tl.ext.load_view_tko(
            k_sf_tail_view, [physical_page.to(tl.int32), 0, 8, 0, 0]
        )
        tail_k_scales = tail_k_scales.reshape([1, 1, 1, 32, 4, 4]).trans(0, 1, 4, 3, 2, 5)
        tail_k_scales = tail_k_scales.reshape([BLOCK_T, 4])
        scores = tl.dot_scaled(
            q0_vals,
            q0_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )
        scores = tl.dot_scaled(
            q1_vals,
            q1_scales,
            "e2m1",
            tail_k_vals.T,
            tail_k_scales,
            "e2m1",
            acc=scores,
            fast_math=True,
            rhs_k_pack=True,
        )

        scores = scores * qk_scale
        page_m = tl.max(scores, axis=1)
        if ALLOW_PARTIAL_GROUPS:
            page_m = tl.where(valid_group_page, page_m, -float("inf"))
            safe_page_m = tl.where(valid_group_page, page_m, 0.0)
            exp_scores = tl.math.exp2((scores - safe_page_m[:, None]) * 1.4426950408889634)
            exp_scores = tl.where(valid_group_page, exp_scores, 0.0)
        else:
            exp_scores = tl.math.exp2((scores - page_m[:, None]) * 1.4426950408889634)
        page_l = tl.sum(exp_scores, axis=1)

        v_row = (physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
        v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
        if FP4_PV:
            grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
            amax = tl.max(grouped_probs, axis=2)
            inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
            p_scales = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
            p_scales = p_scales.to(tl.float8e4nv)
            scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
            pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
            even_probs, odd_probs = tl.split(pairs)
            p_vals = tl.reshape(
                _fp4_e2m1_quantize_packed(even_probs, odd_probs), (BLOCK_H, PAGE_SIZE // 2)
            )
            v_scale_offsets = _fp4_mla_swizzled_sf_offset(
                offs_v[:, None], scale_cols[None, :], SF_PER_PAGE
            )
            v_scales = tl.load(v_sf_ptr + physical_page * vsf_s0 + v_scale_offsets)
            page_o = tl.ext.dot_scaled(
                v_vals,
                v_scales,
                "e2m1",
                p_vals.T,
                p_scales,
                "e2m1",
                acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                fast_math=True,
                rhs_k_pack=True,
            )
        else:
            v_scale_cols = packed_t // (FP4_BLOCK // 2)
            v_scale_offsets = _fp4_mla_swizzled_sf_offset(
                offs_v[:, None], v_scale_cols[None, :], SF_PER_PAGE
            )
            v_scales = tl.load(v_sf_ptr + physical_page * vsf_s0 + v_scale_offsets).to(tl.float32)
            v_low = _fp4_e2m1_to_f32(v_vals & 0x0F) * v_scales
            v_high = _fp4_e2m1_to_f32((v_vals >> 4) & 0x0F) * v_scales
            exp_pairs = tl.reshape(exp_scores, (BLOCK_H, PAGE_SIZE // 2, 2))
            exp_even, exp_odd = tl.split(exp_pairs)
            page_o = tl.dot(v_low, exp_even.T, out_dtype=tl.float32) + tl.dot(
                v_high, exp_odd.T, out_dtype=tl.float32
            )

        next_m = tl.maximum(group_m, page_m)
        old_delta = tl.where(group_l > 0.0, group_m - next_m, 0.0)
        new_delta = tl.where(page_l > 0.0, page_m - next_m, 0.0)
        old_scale = tl.math.exp2(old_delta * 1.4426950408889634)
        new_scale = tl.math.exp2(new_delta * 1.4426950408889634)
        group_o = group_o * old_scale[None, :] + page_o * new_scale[None, :]
        group_l = group_l * old_scale + page_l * new_scale
        group_m = next_m

    partial_o_offsets = (
        page_group * po_s0 + gen_idx * po_s1 + offs_h[:, None] * po_s2 + offs_v[None, :] * po_s3
    )
    tl.store(partial_o_ptr + partial_o_offsets, group_o.T)
    if dim_block == 0:
        partial_ml_offsets = page_group * pm_s0 + gen_idx * pm_s1 + offs_h
        tl.store(partial_m_ptr + partial_ml_offsets, group_m)
        tl.store(partial_l_ptr + partial_ml_offsets, group_l)


@triton.jit
def _fp4_mla_pv_page_o_prepacked_raw(
    v_sf_ptr,
    v_packed_ptr,
    physical_page,
    dim_block: tl.constexpr,
    p_vals,
    p_scales,
    num_pages: tl.constexpr,
    vsf_s0: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    packed_t = tl.arange(0, PAGE_SIZE // 2)
    num_dim_blocks: tl.constexpr = V_HEAD_D // BLOCK_V
    v_sf_view = tl.ext.make_view(
        base=v_sf_ptr,
        shapes=[num_pages, V_HEAD_D // 128, SF_PER_PAGE // 4, 2, 256],
        strides=[vsf_s0, 128 * (((SF_PER_PAGE + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
        tile_dim_map=[0, 1, 2, 3, 4],
    )
    v_row = (physical_page * num_dim_blocks + dim_block) * BLOCK_V
    v_vals = tl.load(
        v_packed_ptr + (v_row + offs_v[:, None]) * (PAGE_SIZE // 2) + packed_t[None, :]
    )
    v_scales = tl.ext.load_view_tko(
        v_sf_view,
        [
            physical_page.to(tl.int32),
            dim_block * (BLOCK_V // 128),
            0,
            0,
            0,
        ],
    )
    v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
        0, 1, 4, 3, 2, 5
    )
    v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
    return tl.ext.dot_scaled(
        v_vals,
        v_scales,
        "e2m1",
        p_vals.T,
        p_scales,
        "e2m1",
        acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
        fast_math=True,
        rhs_k_pack=True,
    )


@triton.jit
def _fp4_mla_attention_gen_qkpv_group_kernel(
    partial_o_ptr,
    partial_m_ptr,
    partial_l_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len: tl.constexpr,
    num_pages: tl.constexpr,
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    vsf_s0: tl.constexpr,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    pm_s0: tl.constexpr,
    pm_s1: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    page_group = tl.program_id(2)
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    q_row_base = query_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

    group_m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    group_o0 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    group_o1 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    group_o2 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    group_o3 = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)

    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        page_start = page_rel * PAGE_SIZE
        valid_group_page = page_rel < MAX_PAGES
        valid_page_tokens = valid_group_page & (page_start < kv_len)
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        scores = _fp4_mla_qk_scores_tile(
            q_fp4_ptr,
            q_sf_ptr,
            kv_cache_ptr,
            sf_cache_ptr,
            src_page_ids_ptr,
            safe_compact_page,
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
        valid_t = page_start + offs_t < kv_len
        scores = tl.where(valid_page_tokens & valid_t[None, :], scores * qk_scale, -float("inf"))
        page_m = tl.max(scores, axis=1)
        safe_page_m = tl.where(valid_page_tokens, page_m, 0.0)
        exp_scores = tl.math.exp2((scores - safe_page_m[:, None]) * 1.4426950408889634)
        exp_scores = tl.where(valid_page_tokens & valid_t[None, :], exp_scores, 0.0)
        page_l = tl.sum(exp_scores, axis=1)

        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        p_scales = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        p_scales = p_scales.to(tl.float8e4nv)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        p_vals = tl.reshape(
            _fp4_e2m1_quantize_packed(even_probs, odd_probs), (BLOCK_H, PAGE_SIZE // 2)
        )

        page_o0 = _fp4_mla_pv_page_o_prepacked_raw(
            v_sf_ptr,
            v_packed_ptr,
            physical_page,
            0,
            p_vals,
            p_scales,
            num_pages,
            vsf_s0,
            V_HEAD_D,
            PAGE_SIZE,
            SF_PER_PAGE,
            BLOCK_H,
            BLOCK_V,
        )
        page_o1 = _fp4_mla_pv_page_o_prepacked_raw(
            v_sf_ptr,
            v_packed_ptr,
            physical_page,
            1,
            p_vals,
            p_scales,
            num_pages,
            vsf_s0,
            V_HEAD_D,
            PAGE_SIZE,
            SF_PER_PAGE,
            BLOCK_H,
            BLOCK_V,
        )
        page_o2 = _fp4_mla_pv_page_o_prepacked_raw(
            v_sf_ptr,
            v_packed_ptr,
            physical_page,
            2,
            p_vals,
            p_scales,
            num_pages,
            vsf_s0,
            V_HEAD_D,
            PAGE_SIZE,
            SF_PER_PAGE,
            BLOCK_H,
            BLOCK_V,
        )
        page_o3 = _fp4_mla_pv_page_o_prepacked_raw(
            v_sf_ptr,
            v_packed_ptr,
            physical_page,
            3,
            p_vals,
            p_scales,
            num_pages,
            vsf_s0,
            V_HEAD_D,
            PAGE_SIZE,
            SF_PER_PAGE,
            BLOCK_H,
            BLOCK_V,
        )

        next_m = tl.maximum(group_m, page_m)
        old_delta = tl.where(group_l > 0.0, group_m - next_m, 0.0)
        new_delta = tl.where(page_l > 0.0, page_m - next_m, 0.0)
        old_scale = tl.math.exp2(old_delta * 1.4426950408889634)
        new_scale = tl.math.exp2(new_delta * 1.4426950408889634)
        group_o0 = group_o0 * old_scale[None, :] + page_o0 * new_scale[None, :]
        group_o1 = group_o1 * old_scale[None, :] + page_o1 * new_scale[None, :]
        group_o2 = group_o2 * old_scale[None, :] + page_o2 * new_scale[None, :]
        group_o3 = group_o3 * old_scale[None, :] + page_o3 * new_scale[None, :]
        group_l = group_l * old_scale + page_l * new_scale
        group_m = next_m

    offs_v = tl.arange(0, BLOCK_V)
    partial_base = page_group * po_s0 + query_idx * po_s1 + offs_h[:, None] * po_s2
    tl.store(partial_o_ptr + partial_base + offs_v[None, :] * po_s3, group_o0.T)
    tl.store(partial_o_ptr + partial_base + (BLOCK_V + offs_v)[None, :] * po_s3, group_o1.T)
    tl.store(partial_o_ptr + partial_base + (2 * BLOCK_V + offs_v)[None, :] * po_s3, group_o2.T)
    tl.store(partial_o_ptr + partial_base + (3 * BLOCK_V + offs_v)[None, :] * po_s3, group_o3.T)
    partial_ml_offsets = page_group * pm_s0 + query_idx * pm_s1 + offs_h
    tl.store(partial_m_ptr + partial_ml_offsets, group_m)
    tl.store(partial_l_ptr + partial_ml_offsets, group_l)


@triton.jit
def _fp4_mla_attention_mtp_fused_qkpv_group_kernel(
    partial_o_ptr,
    partial_m_ptr,
    partial_l_ptr,
    q_fp4_ptr,
    q_sf_ptr,
    kv_cache_ptr,
    sf_cache_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    kv_lens_ptr,
    page_ids_len: tl.constexpr,
    num_pages: tl.constexpr,
    q_fp4_s0: tl.constexpr,
    q_fp4_s1: tl.constexpr,
    kv_s0: tl.constexpr,
    kv_s2: tl.constexpr,
    kv_s4: tl.constexpr,
    sf_s0: tl.constexpr,
    vsf_s0: tl.constexpr,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    pm_s0: tl.constexpr,
    pm_s1: tl.constexpr,
    q_num_rows: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    Q_HEAD_D: tl.constexpr,
    K_HEAD_D: tl.constexpr,
    Q_RESIDUAL_D: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FP4_BLOCK: tl.constexpr,
    Q_SF_PER_TOKEN: tl.constexpr,
    K_SF_PER_TOKEN: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    QUERY_LEN_PER_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_BLOCK_END: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr,
    SPLIT_PV_K: tl.constexpr,
    USE_TMA_DATA_LOAD: tl.constexpr,
    ASSUME_FULL_HEADS: tl.constexpr,
    ASSUME_VALID_PAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    query_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    page_group = combo // NUM_DIM_BLOCKS
    dim_block = combo - page_group * NUM_DIM_BLOCKS
    seq_idx = query_idx // QUERY_LEN_PER_SEQ
    query_offset = query_idx - seq_idx * QUERY_LEN_PER_SEQ

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    q_row_base = query_idx * NUM_HEADS
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + seq_idx).to(tl.int64)
    kv_len = tl.load(kv_lens_ptr + seq_idx) - (QUERY_LEN_PER_SEQ - 1 - query_offset)
    kv_len = tl.maximum(kv_len, 0)
    global_scale = tl.load(global_scale_ptr)
    qk_scale = sm_scale / (global_scale * global_scale)

    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * NUM_DIM_BLOCKS * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    v_sf_view = tl.ext.make_view(
        base=v_sf_ptr,
        shapes=[num_pages, V_HEAD_D // 128, SF_PER_PAGE // 4, 2, 256],
        strides=[vsf_s0, 128 * (((SF_PER_PAGE + 3) // 4) * 4), 512, 256, 1],
        tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
        tile_dim_map=[0, 1, 2, 3, 4],
    )

    group_m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    group_l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    group_o = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        page_start = page_rel * PAGE_SIZE
        valid_group_page = page_rel < MAX_PAGES
        valid_page_tokens = valid_group_page & (page_start < kv_len)
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        scores = _fp4_mla_qk_scores_tile(
            q_fp4_ptr,
            q_sf_ptr,
            kv_cache_ptr,
            sf_cache_ptr,
            src_page_ids_ptr,
            safe_compact_page,
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
        valid_t = page_start + offs_t < kv_len
        scores = tl.where(valid_page_tokens & valid_t[None, :], scores * qk_scale, -float("inf"))
        page_m = tl.max(scores, axis=1)
        safe_page_m = tl.where(valid_page_tokens, page_m, 0.0)
        exp_scores = tl.math.exp2((scores - safe_page_m[:, None]) * 1.4426950408889634)
        exp_scores = tl.where(valid_page_tokens & valid_t[None, :], exp_scores, 0.0)
        page_l = tl.sum(exp_scores, axis=1)

        grouped_probs = tl.reshape(exp_scores, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK))
        amax = tl.max(grouped_probs, axis=2)
        inv_local_scale = tl.where(amax > 0.0, 6.0 / amax, 1.0)
        p_scales = tl.where(amax > 0.0, tl.minimum(amax * (P_GLOBAL_SCALE / 6.0), 448.0), 1.0)
        p_scales = p_scales.to(tl.float8e4nv)
        scaled_probs = grouped_probs * tl.reshape(inv_local_scale, (BLOCK_H, SF_PER_PAGE, 1))
        pairs = tl.reshape(scaled_probs, (BLOCK_H, SF_PER_PAGE, FP4_BLOCK // 2, 2))
        even_probs, odd_probs = tl.split(pairs)
        p_vals = tl.reshape(
            _fp4_e2m1_quantize_packed(even_probs, odd_probs), (BLOCK_H, PAGE_SIZE // 2)
        )

        v_row = (physical_page * NUM_DIM_BLOCKS + dim_block) * BLOCK_V
        v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
        v_scales = tl.ext.load_view_tko(
            v_sf_view,
            [
                physical_page.to(tl.int32),
                dim_block * (BLOCK_V // 128),
                0,
                0,
                0,
            ],
        )
        v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
            0, 1, 4, 3, 2, 5
        )
        v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
        if SPLIT_PV_K:
            v_val_halves = tl.reshape(v_vals, (BLOCK_V, 2, PAGE_SIZE // 4)).trans(0, 2, 1)
            v_vals0, v_vals1 = tl.split(v_val_halves)
            v_scale_halves = tl.reshape(v_scales, (BLOCK_V, 2, SF_PER_PAGE // 2)).trans(0, 2, 1)
            v_scales0, v_scales1 = tl.split(v_scale_halves)
            p_val_halves = tl.reshape(p_vals, (BLOCK_H, 2, PAGE_SIZE // 4)).trans(0, 2, 1)
            p_vals0, p_vals1 = tl.split(p_val_halves)
            p_scale_halves = tl.reshape(p_scales, (BLOCK_H, 2, SF_PER_PAGE // 2)).trans(0, 2, 1)
            p_scales0, p_scales1 = tl.split(p_scale_halves)
            page_o = tl.ext.dot_scaled(
                v_vals0,
                v_scales0,
                "e2m1",
                p_vals0.T,
                p_scales0,
                "e2m1",
                acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                fast_math=True,
                rhs_k_pack=True,
            )
            page_o = tl.ext.dot_scaled(
                v_vals1,
                v_scales1,
                "e2m1",
                p_vals1.T,
                p_scales1,
                "e2m1",
                acc=page_o,
                fast_math=True,
                rhs_k_pack=True,
            )
        else:
            page_o = tl.ext.dot_scaled(
                v_vals,
                v_scales,
                "e2m1",
                p_vals.T,
                p_scales,
                "e2m1",
                acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                fast_math=True,
                rhs_k_pack=True,
            )

        next_m = tl.maximum(group_m, page_m)
        old_delta = tl.where(group_l > 0.0, group_m - next_m, 0.0)
        new_delta = tl.where(page_l > 0.0, page_m - next_m, 0.0)
        old_scale = tl.math.exp2(old_delta * 1.4426950408889634)
        new_scale = tl.math.exp2(new_delta * 1.4426950408889634)
        group_o = group_o * old_scale[None, :] + page_o * new_scale[None, :]
        group_l = group_l * old_scale + page_l * new_scale
        group_m = next_m

    partial_o_offsets = (
        page_group * po_s0 + query_idx * po_s1 + offs_h[:, None] * po_s2 + offs_v[None, :] * po_s3
    )
    tl.store(partial_o_ptr + partial_o_offsets, group_o.T)
    partial_ml_offsets = page_group * pm_s0 + query_idx * pm_s1 + offs_h
    tl.store(partial_m_ptr + partial_ml_offsets, group_m)
    tl.store(partial_l_ptr + partial_ml_offsets, group_l)


@triton.jit
def _fp4_mla_attention_online_qkpv_reduce_kernel(
    out_ptr,
    partial_o_ptr,
    partial_m_ptr,
    partial_l_ptr,
    global_scale_ptr,
    out_s0,
    out_s1,
    out_s2,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    pm_s0: tl.constexpr,
    pm_s1: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    NUM_PAGE_GROUPS: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FP4_PV: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)

    global_m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    for group_idx in tl.range(0, NUM_PAGE_GROUPS):
        group_m = tl.load(partial_m_ptr + group_idx * pm_s0 + gen_idx * pm_s1 + offs_h)
        global_m = tl.maximum(global_m, group_m)

    global_l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for group_idx in tl.range(0, NUM_PAGE_GROUPS):
        group_m = tl.load(partial_m_ptr + group_idx * pm_s0 + gen_idx * pm_s1 + offs_h)
        group_l = tl.load(partial_l_ptr + group_idx * pm_s0 + gen_idx * pm_s1 + offs_h)
        scale = tl.where(
            group_l > 0.0, tl.math.exp2((group_m - global_m) * 1.4426950408889634), 0.0
        )
        partial_o = tl.load(
            partial_o_ptr
            + group_idx * po_s0
            + gen_idx * po_s1
            + offs_h[:, None] * po_s2
            + offs_v[None, :] * po_s3
        )
        acc += partial_o * scale[:, None]
        global_l += group_l * scale

    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / global_scale
    if FP4_PV:
        out_scale = out_scale / P_GLOBAL_SCALE
    safe_l = tl.where(global_l > 0.0, global_l, 1.0)
    out_vals = (acc / safe_l[:, None]) * out_scale
    if out_ptr.dtype.element_ty == tl.bfloat16:
        out_vals = out_vals.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float16:
        out_vals = out_vals.to(tl.float16)
    tl.store(
        out_ptr + gen_idx * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
        out_vals,
    )


@triton.jit
def _fp4_mla_attention_pv_group_partial_prepacked_v_kernel(
    partial_o_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    v_sf_ptr,
    v_packed_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    num_pages,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    po_num_rows: tl.constexpr,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    vsf_s0: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    page_group = combo // NUM_DIM_BLOCKS
    dim_block = combo - page_group * NUM_DIM_BLOCKS

    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)

    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
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
        tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
        tile_dim_map=[0, 1, 2, 3, 4],
    )
    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )
    tl.assume(po_s2 % 8 == 0)
    tl.assume(po_s3 == 1)
    partial_o_desc = tl.make_tensor_descriptor(
        partial_o_ptr,
        shape=[po_num_rows, V_HEAD_D],
        strides=[po_s2, po_s3],
        block_shape=[BLOCK_H, BLOCK_V],
    )

    acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES, num_stages=PV_LOOP_STAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        valid_group_page = page_rel < MAX_PAGES
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        p_vals = tl.ext.load_view_tko(
            p_view,
            [(safe_compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
        )
        p_vals = p_vals.to(tl.uint8, bitcast=True)
        p_scales = tl.ext.load_view_tko(p_sf_view, [safe_compact_page.to(tl.int32), 0, 0, 0])
        p_scales = p_scales.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
        p_scales = p_scales.reshape([BLOCK_H, SF_PER_PAGE])

        v_row = (physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
        v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
        v_scales = tl.ext.load_view_tko(
            v_sf_view,
            [
                physical_page.to(tl.int32),
                dim_block * (BLOCK_V // 128),
                0,
                0,
                0,
            ],
        )
        v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
            0, 1, 4, 3, 2, 5
        )
        v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
        if ALLOW_PARTIAL_GROUPS:
            page_acc = tl.ext.dot_scaled(
                v_vals,
                v_scales,
                "e2m1",
                p_vals.T,
                p_scales,
                "e2m1",
                acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                fast_math=True,
                rhs_k_pack=True,
            )
            acc += tl.where(valid_group_page, page_acc, 0.0)
        else:
            acc = tl.ext.dot_scaled(
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

    partial_o_desc.store(
        [
            (page_group * (po_s0 // po_s2) + gen_idx * (po_s1 // po_s2) + head_block * BLOCK_H).to(
                tl.int32
            ),
            (dim_block * BLOCK_V).to(tl.int32),
        ],
        acc.T,
    )


@triton.jit
def _fp4_mla_attention_pv_group_partial_reduce_kernel(
    out_ptr,
    partial_o_ptr,
    page_sum_ptr,
    global_scale_ptr,
    out_s0,
    out_s1,
    out_s2,
    po_s0: tl.constexpr,
    po_s1: tl.constexpr,
    po_s2: tl.constexpr,
    po_s3: tl.constexpr,
    po_num_rows: tl.constexpr,
    page_stats_s0: tl.constexpr,
    page_stats_s1: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    NUM_PAGE_GROUPS: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    tl.assume(po_s2 % 8 == 0)
    tl.assume(po_s3 == 1)
    partial_o_desc = tl.make_tensor_descriptor(
        partial_o_ptr,
        shape=[po_num_rows, V_HEAD_D],
        strides=[po_s2, po_s3],
        block_shape=[BLOCK_H, BLOCK_V],
    )

    global_m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    for group_idx in tl.range(0, NUM_PAGE_GROUPS):
        group_m = tl.load(
            page_sum_ptr + gen_idx * page_stats_s0 + (group_idx * 2) * page_stats_s1 + offs_h
        )
        global_m = tl.maximum(global_m, group_m)

    global_l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for group_idx in tl.range(0, NUM_PAGE_GROUPS):
        group_m = tl.load(
            page_sum_ptr + gen_idx * page_stats_s0 + (group_idx * 2) * page_stats_s1 + offs_h
        )
        group_l = tl.load(
            page_sum_ptr + gen_idx * page_stats_s0 + (group_idx * 2 + 1) * page_stats_s1 + offs_h
        )
        scale = tl.where(
            group_l > 0.0, tl.math.exp2((group_m - global_m) * 1.4426950408889634), 0.0
        )
        partial_o = partial_o_desc.load(
            [
                (
                    group_idx * (po_s0 // po_s2) + gen_idx * (po_s1 // po_s2) + head_block * BLOCK_H
                ).to(tl.int32),
                (dim_block * BLOCK_V).to(tl.int32),
            ]
        )
        acc += partial_o * scale[:, None]
        global_l += group_l * scale

    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    safe_l = tl.where(global_l > 0.0, global_l, 1.0)
    out_vals = (acc / safe_l[:, None]) * out_scale
    if out_ptr.dtype.element_ty == tl.bfloat16:
        out_vals = out_vals.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float16:
        out_vals = out_vals.to(tl.float16)
    tl.store(
        out_ptr + gen_idx * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
        out_vals,
    )


@triton.jit
def _fp4_mla_attention_pv_atomic_split_prepacked_v_kernel(
    out_acc_ptr,
    p_fp4_ptr,
    p_sf_ptr,
    v_sf_ptr,
    v_packed_ptr,
    global_scale_ptr,
    src_page_ids_ptr,
    paged_kv_indptr_decode_ptr,
    num_pages,
    out_s0,
    out_s1,
    out_s2,
    p_s0: tl.constexpr,
    p_s1: tl.constexpr,
    p_num_rows: tl.constexpr,
    vsf_s0: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SF_PER_PAGE: tl.constexpr,
    P_GLOBAL_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    GROUP_PAGES: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    ALLOW_PARTIAL_GROUPS: tl.constexpr,
    PV_LOOP_STAGES: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    combo = tl.program_id(2)
    page_group = combo // NUM_DIM_BLOCKS
    dim_block = combo - page_group * NUM_DIM_BLOCKS

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    page_table_start = tl.load(paged_kv_indptr_decode_ptr + gen_idx).to(tl.int64)

    tl.assume(p_s0 % 8 == 0)
    tl.assume(p_s1 == 1)
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
        tile_shape=[1, BLOCK_V // 128, SF_PER_PAGE // 4, 2, 256],
        tile_dim_map=[0, 1, 2, 3, 4],
    )
    v_packed_desc = tl.make_tensor_descriptor(
        v_packed_ptr,
        shape=[num_pages * (V_HEAD_D // BLOCK_V) * BLOCK_V, PAGE_SIZE // 2],
        strides=[PAGE_SIZE // 2, 1],
        block_shape=[BLOCK_V, PAGE_SIZE // 2],
    )

    acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
    for page_group_off in tl.range(0, GROUP_PAGES, num_stages=PV_LOOP_STAGES):
        page_rel = page_group * GROUP_PAGES + page_group_off
        valid_group_page = page_rel < MAX_PAGES
        compact_page = page_table_start + page_rel
        safe_compact_page = tl.where(valid_group_page, compact_page, page_table_start)
        physical_page = tl.load(
            src_page_ids_ptr + safe_compact_page,
            mask=valid_group_page | (not ALLOW_PARTIAL_GROUPS),
            other=0,
        ).to(tl.int64)

        p_vals = tl.ext.load_view_tko(
            p_view,
            [(safe_compact_page * NUM_HEADS + head_block * BLOCK_H).to(tl.int32), 0],
        )
        p_vals = p_vals.to(tl.uint8, bitcast=True)
        p_scales = tl.ext.load_view_tko(p_sf_view, [safe_compact_page.to(tl.int32), 0, 0, 0])
        p_scales = p_scales.reshape([1, SF_PER_PAGE // 4, 32, 4, 4]).trans(0, 3, 2, 1, 4)
        p_scales = p_scales.reshape([BLOCK_H, SF_PER_PAGE])

        v_row = (physical_page * (V_HEAD_D // BLOCK_V) + dim_block) * BLOCK_V
        v_vals = v_packed_desc.load([v_row.to(tl.int32), 0])
        v_scales = tl.ext.load_view_tko(
            v_sf_view,
            [
                physical_page.to(tl.int32),
                dim_block * (BLOCK_V // 128),
                0,
                0,
                0,
            ],
        )
        v_scales = v_scales.reshape([1, BLOCK_V // 128, SF_PER_PAGE // 4, 32, 4, 4]).trans(
            0, 1, 4, 3, 2, 5
        )
        v_scales = v_scales.reshape([BLOCK_V, SF_PER_PAGE])
        if ALLOW_PARTIAL_GROUPS:
            page_acc = tl.ext.dot_scaled(
                v_vals,
                v_scales,
                "e2m1",
                p_vals.T,
                p_scales,
                "e2m1",
                acc=tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32),
                fast_math=True,
                rhs_k_pack=True,
            )
            acc += tl.where(valid_group_page, page_acc, 0.0)
        else:
            acc = tl.ext.dot_scaled(
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

    global_scale = tl.load(global_scale_ptr)
    out_scale = 1.0 / (global_scale * P_GLOBAL_SCALE)
    out_vals = acc.T * out_scale
    tl.atomic_add(
        out_acc_ptr + gen_idx * out_s0 + offs_h[:, None] * out_s1 + offs_v[None, :] * out_s2,
        out_vals,
        sem="relaxed",
    )


@triton.jit
def _fp4_mla_attention_cast_acc_kernel(
    out_ptr,
    out_acc_ptr,
    out_s0,
    out_s1,
    out_s2,
    acc_s0,
    acc_s1,
    acc_s2,
    NUM_HEADS: tl.constexpr,
    V_HEAD_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    occupancy: tl.constexpr = 1,
):
    gen_idx = tl.program_id(0)
    head_block = tl.program_id(1)
    dim_block = tl.program_id(2)
    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_v = dim_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_h = offs_h < NUM_HEADS
    mask_v = offs_v < V_HEAD_D
    safe_h = tl.where(mask_h, offs_h, 0)
    safe_v = tl.where(mask_v, offs_v, 0)

    vals = tl.load(
        out_acc_ptr + gen_idx * acc_s0 + safe_h[:, None] * acc_s1 + safe_v[None, :] * acc_s2,
        mask=mask_h[:, None] & mask_v[None, :],
        other=0.0,
    )
    if out_ptr.dtype.element_ty == tl.bfloat16:
        vals = vals.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float16:
        vals = vals.to(tl.float16)
    tl.store(
        out_ptr + gen_idx * out_s0 + safe_h[:, None] * out_s1 + safe_v[None, :] * out_s2,
        vals,
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
    page_stats_group_size: Optional[int] = None,
    assume_full_pages: Optional[bool] = None,
    assume_full_pages_except_mtp_tail: bool = False,
    assume_valid_pages: Optional[bool] = None,
    query_len_per_seq: int = 1,
    prepack_v_for_pv: bool = False,
    use_prepacked_v_for_pv: bool = False,
    p_fp4_workspace: Optional[torch.Tensor] = None,
    p_sf_workspace: Optional[torch.Tensor] = None,
    v_packed_workspace: Optional[torch.Tensor] = None,
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

    if query_len_per_seq <= 0:
        raise ValueError(f"query_len_per_seq must be positive, got {query_len_per_seq}.")
    if num_gen % query_len_per_seq != 0:
        raise ValueError(
            "q_fp4 query rows must be divisible by query_len_per_seq, got "
            f"{num_gen} rows and query_len_per_seq={query_len_per_seq}."
        )
    num_gen_seqs = num_gen // query_len_per_seq

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
    env_block_h = _env_int("TRTLLM_FP4_MLA_BLOCK_H")
    env_block_k = _env_int("TRTLLM_FP4_MLA_BLOCK_K")
    env_block_v = _env_int("TRTLLM_FP4_MLA_BLOCK_V")
    env_pv_block_h = _env_int("TRTLLM_FP4_MLA_PV_BLOCK_H")
    env_pv_loop_stages = _env_int("TRTLLM_FP4_MLA_PV_LOOP_STAGES")
    env_occupancy = _env_int("TRTLLM_FP4_MLA_OCCUPANCY")
    env_num_ctas = _env_int("TRTLLM_FP4_MLA_NUM_CTAS")
    env_num_warps = _env_int("TRTLLM_FP4_MLA_NUM_WARPS")
    env_num_stages = _env_int("TRTLLM_FP4_MLA_NUM_STAGES")
    env_page_stats_num_ctas = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_CTAS")
    env_page_stats_num_warps = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_WARPS")
    env_page_stats_num_stages = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_STAGES")
    env_pv_num_ctas = _env_int("TRTLLM_FP4_MLA_PV_NUM_CTAS")
    env_pv_num_warps = _env_int("TRTLLM_FP4_MLA_PV_NUM_WARPS")
    env_pv_num_stages = _env_int("TRTLLM_FP4_MLA_PV_NUM_STAGES")
    env_group_pages = _env_int("TRTLLM_FP4_MLA_GROUP_PAGES")
    env_group_reduce_stats = _env_int("TRTLLM_FP4_MLA_GROUP_REDUCE_STATS")
    env_page_pipeline_streams = _env_int("TRTLLM_FP4_MLA_PAGE_PIPELINE_STREAMS")
    env_online_qkpv = _env_int("TRTLLM_FP4_MLA_ONLINE_QKPV")
    env_online_qkpv_group_pages = _env_int("TRTLLM_FP4_MLA_ONLINE_QKPV_GROUP_PAGES")
    env_online_qkpv_max_batch = _env_int("TRTLLM_FP4_MLA_ONLINE_QKPV_MAX_BATCH")
    env_online_qkpv_fp4_pv = _env_int("TRTLLM_FP4_MLA_ONLINE_QKPV_FP4_PV")
    env_gen_qkpv = _env_int("TRTLLM_FP4_MLA_GEN_QKPV")
    env_gen_qkpv_group_pages = _env_int("TRTLLM_FP4_MLA_GEN_QKPV_GROUP_PAGES")
    env_gen_qkpv_block_h = _env_int("TRTLLM_FP4_MLA_GEN_QKPV_BLOCK_H")
    env_gen_qkpv_partial_dtype = os.environ.get("TRTLLM_FP4_MLA_GEN_QKPV_PARTIAL_DTYPE", "").lower()
    env_mtp_fused_qkpv = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV")
    env_mtp_fused_qkpv_group_pages = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV_GROUP_PAGES")
    env_mtp_fused_qkpv_block_h = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV_BLOCK_H")
    env_mtp_fused_qkpv_block_v = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV_BLOCK_V")
    env_mtp_fused_qkpv_split_pv_k = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV_SPLIT_PV_K")
    env_mtp_fused_qkpv_occupancy = _env_int("TRTLLM_FP4_MLA_MTP_FUSED_QKPV_OCCUPANCY")
    env_mtp_page_stats_pair = _env_int("TRTLLM_FP4_MLA_MTP_PAGE_STATS_PAIR")
    env_mtp_pv_pair = _env_int("TRTLLM_FP4_MLA_MTP_PV_PAIR")
    env_mtp_split_tail_group = _env_int("TRTLLM_FP4_MLA_MTP_SPLIT_TAIL_GROUP")
    env_mtp_final_page_fast_path = _env_int("TRTLLM_FP4_MLA_MTP_FINAL_PAGE_FAST_PATH")
    env_group_pv = _env_int("TRTLLM_FP4_MLA_GROUP_PV")
    env_group_pv_max_batch = _env_int("TRTLLM_FP4_MLA_GROUP_PV_MAX_BATCH")
    env_pv_atomic_split = _env_int("TRTLLM_FP4_MLA_PV_ATOMIC_SPLIT")
    env_pv_atomic_group_pages = _env_int("TRTLLM_FP4_MLA_PV_ATOMIC_GROUP_PAGES")
    env_pv_apply_prob_scale = _env_int("TRTLLM_FP4_MLA_PV_APPLY_PROB_SCALE")
    env_pv_scale_in_sf = _env_int("TRTLLM_FP4_MLA_PV_SCALE_IN_SF")
    env_scale_from_group_stats = _env_int("TRTLLM_FP4_MLA_SCALE_FROM_GROUP_STATS")
    env_duplicate_tail_k = _env_int("TRTLLM_FP4_MLA_DUPLICATE_TAIL_K")
    env_debug_page_stats_pack = _env_int("TRTLLM_FP4_MLA_DEBUG_PAGE_STATS_PACK")
    env_debug_stop_after_page_stats = _env_int("TRTLLM_FP4_MLA_DEBUG_STOP_AFTER_PAGE_STATS")
    env_mtp_page_stats_pair_group_pages = _env_int("TRTLLM_FP4_MLA_MTP_PAGE_STATS_PAIR_GROUP_PAGES")
    env_pv_tma = _env_int("TRTLLM_FP4_MLA_PV_TMA")
    env_pv_p_tma = _env_int("TRTLLM_FP4_MLA_PV_P_TMA")
    env_pv_v_tma = _env_int("TRTLLM_FP4_MLA_PV_V_TMA")
    env_pv_out_tma = _env_int("TRTLLM_FP4_MLA_PV_OUT_TMA")
    if env_block_h is not None:
        block_h = env_block_h
    if block_k is None:
        block_k = env_block_k or (512 if triton_backend == "nvt" else 256)
    elif env_block_k is not None:
        block_k = env_block_k
    if env_block_v is not None:
        block_v = env_block_v
    if env_pv_loop_stages is not None:
        pv_loop_stages = env_pv_loop_stages
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    if max_pages is None:
        if paged_kv_indptr_decode.numel() >= num_gen_seqs + 1:
            page_counts = (
                paged_kv_indptr_decode[1 : num_gen_seqs + 1] - paged_kv_indptr_decode[:num_gen_seqs]
            )
            max_pages = int(page_counts.max().item()) if page_counts.numel() > 0 else 0
        else:
            max_pages = _ceil_div(int(kv_lens[:num_gen_seqs].max().item()), page_size)
    if max_pages <= 0:
        output.zero_()
        return output

    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = k_head_dim // FP4_BLOCK_SIZE
    sf_per_page = page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)
    assume_full_heads = num_heads % block_h == 0
    pv_block_h = env_pv_block_h if env_pv_block_h is not None else block_h
    if pv_block_h <= 0:
        raise ValueError(f"TRTLLM_FP4_MLA_PV_BLOCK_H must be positive, got {pv_block_h}.")
    if pv_block_h not in (64, 128):
        raise ValueError(
            f"TRTLLM_FP4_MLA_PV_BLOCK_H currently supports 64 or 128, got {pv_block_h}."
        )
    num_pv_head_blocks = triton.cdiv(num_heads, pv_block_h)
    assume_full_pv_heads = num_heads % pv_block_h == 0
    assume_full_v = v_head_dim % block_v == 0
    if assume_full_pages is None:
        assume_full_pages = False
    assume_full_pages = bool(assume_full_pages) and query_len_per_seq == 1
    mask_mtp_final_page_only = (
        bool(assume_full_pages_except_mtp_tail)
        and (
            env_mtp_final_page_fast_path != 0 if env_mtp_final_page_fast_path is not None else True
        )
        and query_len_per_seq > 1
        and query_len_per_seq <= page_size
    )
    if assume_valid_pages is None:
        assume_valid_pages = False
    assume_valid_pages = bool(assume_valid_pages)
    if (
        not assume_valid_pages
        and assume_full_pages
        and src_page_ids.numel() == num_gen_seqs * max_pages
    ):
        # Full decode pages with an exactly-sized page table do not need the
        # sentinel/physical-page validity masks. Keeping this inference inside
        # the kernel wrapper lets the framework call path stay unchanged.
        assume_valid_pages = True
    p_by_query = query_len_per_seq != 1
    total_p_pages = num_gen * max_pages if p_by_query else src_page_ids.numel()
    total_p_rows = max(total_p_pages * num_heads, 1)
    if page_pipeline_streams is None and env_page_pipeline_streams is not None:
        page_pipeline_streams = env_page_pipeline_streams
    if page_pipeline_streams is None:
        if triton_backend == "nvt" and max_pages >= 8 and num_gen >= 128:
            page_pipeline_streams = 2
        else:
            page_pipeline_streams = 1
    page_pipeline_streams = max(1, min(int(page_pipeline_streams), max_pages))
    launch_meta = {}
    explicit_kernel_occupancy = kernel_occupancy is not None or env_occupancy is not None
    if kernel_occupancy is None:
        if env_occupancy is not None:
            kernel_occupancy = env_occupancy
        elif triton_backend == "nvt":
            kernel_occupancy = 8
    if kernel_occupancy is not None:
        launch_meta["occupancy"] = int(kernel_occupancy)
    if kernel_num_ctas is not None:
        launch_meta["num_ctas"] = int(kernel_num_ctas)
    elif env_num_ctas is not None:
        launch_meta["num_ctas"] = int(env_num_ctas)
    if kernel_num_stages is None:
        if env_num_stages is not None:
            kernel_num_stages = env_num_stages
        elif triton_backend == "nvt":
            kernel_num_stages = 1
    if kernel_num_stages is not None:
        launch_meta["num_stages"] = int(kernel_num_stages)
    if kernel_num_warps is None and env_num_warps is not None:
        kernel_num_warps = env_num_warps
    if kernel_num_warps is not None:
        launch_meta["num_warps"] = int(kernel_num_warps)
    page_stats_launch_meta = dict(launch_meta)
    if env_page_stats_num_ctas is not None:
        page_stats_launch_meta["num_ctas"] = int(env_page_stats_num_ctas)
    if env_page_stats_num_stages is not None:
        page_stats_launch_meta["num_stages"] = int(env_page_stats_num_stages)
    if env_page_stats_num_warps is not None:
        page_stats_launch_meta["num_warps"] = int(env_page_stats_num_warps)
    elif kernel_num_warps is None and triton_backend == "nvt":
        page_stats_launch_meta["num_warps"] = 8
    pv_launch_meta = dict(launch_meta)
    if env_pv_num_ctas is not None:
        pv_launch_meta["num_ctas"] = int(env_pv_num_ctas)
    if env_pv_num_stages is not None:
        pv_launch_meta["num_stages"] = int(env_pv_num_stages)
    if env_pv_num_warps is not None:
        pv_launch_meta["num_warps"] = int(env_pv_num_warps)
    gen_qkpv_launch_meta = dict(page_stats_launch_meta)
    if not explicit_kernel_occupancy:
        gen_qkpv_launch_meta.pop("occupancy", None)
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
    use_pv_tma_data_load = use_tma_data_load and (env_pv_tma is None or env_pv_tma != 0)
    use_pv_p_tma_data_load = use_pv_tma_data_load and (env_pv_p_tma is None or env_pv_p_tma != 0)
    use_pv_v_tma_data_load = use_pv_tma_data_load and (env_pv_v_tma is None or env_pv_v_tma != 0)
    use_pv_out_tma_data_store = use_pv_tma_data_load and (
        env_pv_out_tma is None or env_pv_out_tma != 0
    )

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
    num_dim_blocks = triton.cdiv(v_head_dim, block_v)
    v_repack_block_v = block_v
    num_repack_dim_blocks = num_dim_blocks
    env_prepack_v = os.environ.get("TRTLLM_FP4_MLA_PREPACK_V")
    can_address_all_compact_pages = src_page_ids.numel() == num_gen_seqs * max_pages
    allow_partial_page_prepack_v_for_pv = (
        p_by_query
        and can_address_all_compact_pages
        and (bool(prepack_v_for_pv) or bool(use_prepacked_v_for_pv))
    )
    auto_prepack_v_for_pv = (
        triton_backend == "nvt"
        and use_tma_data_load
        and assume_full_heads
        and (assume_full_pages or allow_partial_page_prepack_v_for_pv)
        and (assume_valid_pages or allow_partial_page_prepack_v_for_pv)
        and v_head_dim == 512
        and page_size == 128
        and block_h == 128
        and block_v in (128, 256)
        and sf_per_page == 8
    )
    if not prepack_v_for_pv and not use_prepacked_v_for_pv:
        prepack_v_for_pv = (
            auto_prepack_v_for_pv
            if env_prepack_v is None
            else env_prepack_v == "1" and auto_prepack_v_for_pv
        )
    wants_prepacked_v_for_pv = bool(prepack_v_for_pv) or bool(use_prepacked_v_for_pv)
    if use_prepacked_v_for_pv and v_packed_workspace is None:
        raise ValueError("use_prepacked_v_for_pv requires v_packed_workspace to be provided.")
    can_use_prepacked_v_for_pv = (
        wants_prepacked_v_for_pv
        and use_tma_data_load
        and assume_full_heads
        and (assume_full_pages or allow_partial_page_prepack_v_for_pv)
        and (assume_valid_pages or allow_partial_page_prepack_v_for_pv)
        and v_head_dim == 512
        and page_size == 128
        and block_h in (64, 128)
        and block_v in (128, 256)
        and sf_per_page == 8
    )
    if wants_prepacked_v_for_pv and not can_use_prepacked_v_for_pv:
        raise ValueError(
            "prepacked V PV path requires TMA, full valid pages or explicit qlen>1 prepack, "
            "v_head_dim=512, page_size=128, block_h in (64, 128), block_v in (128, 256), and sf_per_page=8."
        )
    if can_use_prepacked_v_for_pv:
        v_packed = _workspace_tensor(
            v_packed_workspace,
            (num_pages * num_dim_blocks * block_v, page_size // 2),
            dtype=torch.uint8,
            device=q_fp4.device,
            name="v_packed",
        )
    else:
        v_packed = kv_cache
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
    v_repack_stream = None
    debug_timing = os.environ.get("TRTLLM_FP4_MLA_DEBUG_TIMING") == "1"
    debug_events = []

    def _debug_mark(label: str) -> None:
        if not debug_timing:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        debug_events.append((label, event))

    def _debug_report() -> None:
        if not debug_timing or len(debug_events) < 2:
            return
        torch.cuda.synchronize(q_fp4.device)
        parts = []
        for (start_label, start_event), (end_label, end_event) in zip(
            debug_events, debug_events[1:]
        ):
            parts.append(f"{start_label}->{end_label}={start_event.elapsed_time(end_event):.3f}ms")
        print("[fp4_mla_timing] " + " ".join(parts), flush=True)

    _debug_mark("start")
    if bool(prepack_v_for_pv) and can_use_prepacked_v_for_pv:
        current_stream = torch.cuda.current_stream(q_fp4.device)
        v_repack_stream = torch.cuda.Stream(device=q_fp4.device)
        v_repack_stream.wait_stream(current_stream)
        with torch.cuda.stream(v_repack_stream):
            _fp4_mla_attention_v_repack_kernel[(num_pages, num_repack_dim_blocks)](
                v_packed,
                kv_cache,
                num_pages,
                kv_s0,
                kv_s2,
                kv_s4,
                V_HEAD_D=v_head_dim,
                PAGE_SIZE=page_size,
                BLOCK_V=v_repack_block_v,
                **launch_meta,
            )

    mtp_fused_block_h = env_mtp_fused_qkpv_block_h if env_mtp_fused_qkpv_block_h is not None else 32
    if env_mtp_fused_qkpv == 1 and mtp_fused_block_h not in (16, 32, 64, 128):
        raise ValueError(
            "TRTLLM_FP4_MLA_MTP_FUSED_QKPV_BLOCK_H currently supports 16, 32, 64, or 128, "
            f"got {mtp_fused_block_h}."
        )
    mtp_fused_block_v = (
        env_mtp_fused_qkpv_block_v if env_mtp_fused_qkpv_block_v is not None else v_head_dim
    )
    if env_mtp_fused_qkpv == 1 and mtp_fused_block_v not in (128, 256, 512):
        raise ValueError(
            "TRTLLM_FP4_MLA_MTP_FUSED_QKPV_BLOCK_V currently supports 128, 256, or 512, "
            f"got {mtp_fused_block_v}."
        )
    if env_mtp_fused_qkpv == 1 and v_head_dim % mtp_fused_block_v != 0:
        raise ValueError(
            f"TRTLLM_FP4_MLA_MTP_FUSED_QKPV_BLOCK_V={mtp_fused_block_v} must divide v_head_dim={v_head_dim}."
        )
    num_mtp_fused_dim_blocks = triton.cdiv(v_head_dim, mtp_fused_block_v)
    mtp_fused_group_pages = (
        env_mtp_fused_qkpv_group_pages if env_mtp_fused_qkpv_group_pages is not None else 128
    )
    mtp_fused_group_pages = max(1, min(int(mtp_fused_group_pages), max_pages))
    mtp_fused_launch_meta = dict(page_stats_launch_meta)
    mtp_fused_reduce_meta = dict(pv_launch_meta)
    if env_mtp_fused_qkpv_occupancy is not None:
        mtp_fused_launch_meta["occupancy"] = int(env_mtp_fused_qkpv_occupancy)
        mtp_fused_reduce_meta["occupancy"] = int(env_mtp_fused_qkpv_occupancy)
    elif env_occupancy is None:
        mtp_fused_launch_meta["occupancy"] = 1
        mtp_fused_reduce_meta["occupancy"] = 1
    can_use_mtp_fused_qkpv = (
        env_mtp_fused_qkpv == 1
        and p_by_query
        and query_len_per_seq == 4
        and can_use_prepacked_v_for_pv
        and can_address_all_compact_pages
        and use_tma_data_load
        and num_heads % mtp_fused_block_h == 0
        and num_heads == 128
        and q_head_dim == 640
        and k_head_dim == 576
        and q_residual_dim == 64
        and v_head_dim == 512
        and page_size == 128
        and block_k == 512
        and full_block_end == 512
        and tail_block_k == 128
        and block_v in (128, 256)
        and sf_per_page == 8
    )
    if can_use_mtp_fused_qkpv:
        if v_repack_stream is not None:
            torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)
            v_repack_stream = None
        num_mtp_fused_page_groups = _ceil_div(max_pages, mtp_fused_group_pages)
        num_mtp_fused_head_blocks = triton.cdiv(num_heads, mtp_fused_block_h)
        mtp_fused_partial_o = _workspace_tensor(
            None,
            (num_mtp_fused_page_groups, num_gen, num_heads, v_head_dim),
            dtype=torch.float32,
            device=q_fp4.device,
            name="mtp_fused_partial_o",
        )
        mtp_fused_partial_m = _workspace_tensor(
            None,
            (num_mtp_fused_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="mtp_fused_partial_m",
        )
        mtp_fused_partial_l = _workspace_tensor(
            None,
            (num_mtp_fused_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="mtp_fused_partial_l",
        )
        _fp4_mla_attention_mtp_fused_qkpv_group_kernel[
            (
                num_gen,
                num_mtp_fused_head_blocks,
                num_mtp_fused_page_groups * num_mtp_fused_dim_blocks,
            )
        ](
            mtp_fused_partial_o,
            mtp_fused_partial_m,
            mtp_fused_partial_l,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            v_sf,
            v_packed,
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
            v_sf.stride(0),
            mtp_fused_partial_o.stride(0),
            mtp_fused_partial_o.stride(1),
            mtp_fused_partial_o.stride(2),
            mtp_fused_partial_o.stride(3),
            mtp_fused_partial_m.stride(0),
            mtp_fused_partial_m.stride(1),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            V_HEAD_D=v_head_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=p_global_scale,
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            BLOCK_H=mtp_fused_block_h,
            BLOCK_T=page_size,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            BLOCK_V=mtp_fused_block_v,
            MAX_PAGES=max_pages,
            GROUP_PAGES=mtp_fused_group_pages,
            NUM_DIM_BLOCKS=num_mtp_fused_dim_blocks,
            ALLOW_PARTIAL_GROUPS=max_pages % mtp_fused_group_pages != 0,
            SPLIT_PV_K=env_mtp_fused_qkpv_split_pv_k == 1,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=True,
            ASSUME_VALID_PAGES=True,
            **mtp_fused_launch_meta,
        )
        _fp4_mla_attention_online_qkpv_reduce_kernel[
            (num_gen, num_mtp_fused_head_blocks, num_mtp_fused_dim_blocks)
        ](
            output,
            mtp_fused_partial_o,
            mtp_fused_partial_m,
            mtp_fused_partial_l,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            mtp_fused_partial_o.stride(0),
            mtp_fused_partial_o.stride(1),
            mtp_fused_partial_o.stride(2),
            mtp_fused_partial_o.stride(3),
            mtp_fused_partial_m.stride(0),
            mtp_fused_partial_m.stride(1),
            NUM_HEADS=num_heads,
            V_HEAD_D=v_head_dim,
            NUM_PAGE_GROUPS=num_mtp_fused_page_groups,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=mtp_fused_block_h,
            BLOCK_V=mtp_fused_block_v,
            FP4_PV=True,
            **mtp_fused_reduce_meta,
        )
        return output

    gen_qkpv_block_h = env_gen_qkpv_block_h if env_gen_qkpv_block_h is not None else 32
    gen_qkpv_group_pages = env_gen_qkpv_group_pages if env_gen_qkpv_group_pages is not None else 128
    gen_qkpv_group_pages = max(1, min(int(gen_qkpv_group_pages), max_pages))
    can_use_gen_qkpv = (
        env_gen_qkpv == 1
        and p_by_query
        and query_len_per_seq == 4
        and can_use_prepacked_v_for_pv
        and can_address_all_compact_pages
        and use_tma_data_load
        and num_heads % gen_qkpv_block_h == 0
        and gen_qkpv_block_h in (16, 32, 64)
        and num_heads == 128
        and q_head_dim == 640
        and k_head_dim == 576
        and q_residual_dim == 64
        and v_head_dim == 512
        and page_size == 128
        and block_k == 512
        and full_block_end == 512
        and tail_block_k == 128
        and block_v == 128
        and sf_per_page == 8
    )
    if env_gen_qkpv == 1 and not can_use_gen_qkpv:
        raise ValueError(
            "TRTLLM_FP4_MLA_GEN_QKPV=1 requires qlen=4, prepacked V, "
            "num_heads=128, q/k/v dims 640/576/512, page_size=128, "
            "BLOCK_K=512, BLOCK_V=128, and GEN_QKPV_BLOCK_H in (16, 32, 64)."
        )
    if can_use_gen_qkpv:
        if v_repack_stream is not None:
            torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)
            v_repack_stream = None
        num_gen_qkpv_page_groups = _ceil_div(max_pages, gen_qkpv_group_pages)
        gen_qkpv_partial_dtype = torch.float32
        if env_gen_qkpv_partial_dtype in ("bf16", "bfloat16"):
            gen_qkpv_partial_dtype = torch.bfloat16
        gen_qkpv_partial_o = _workspace_tensor(
            None,
            (num_gen_qkpv_page_groups, num_gen, num_heads, v_head_dim),
            dtype=gen_qkpv_partial_dtype,
            device=q_fp4.device,
            name="gen_qkpv_partial_o",
        )
        gen_qkpv_partial_m = _workspace_tensor(
            None,
            (num_gen_qkpv_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="gen_qkpv_partial_m",
        )
        gen_qkpv_partial_l = _workspace_tensor(
            None,
            (num_gen_qkpv_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="gen_qkpv_partial_l",
        )
        _fp4_mla_attention_gen_qkpv_group_kernel[
            (num_gen, triton.cdiv(num_heads, gen_qkpv_block_h), num_gen_qkpv_page_groups)
        ](
            gen_qkpv_partial_o,
            gen_qkpv_partial_m,
            gen_qkpv_partial_l,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            v_sf,
            v_packed,
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
            v_sf.stride(0),
            gen_qkpv_partial_o.stride(0),
            gen_qkpv_partial_o.stride(1),
            gen_qkpv_partial_o.stride(2),
            gen_qkpv_partial_o.stride(3),
            gen_qkpv_partial_m.stride(0),
            gen_qkpv_partial_m.stride(1),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            V_HEAD_D=v_head_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            Q_SF_PER_TOKEN=q_sf_per_token,
            K_SF_PER_TOKEN=k_sf_per_token,
            SF_PER_PAGE=sf_per_page,
            P_GLOBAL_SCALE=p_global_scale,
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            BLOCK_H=gen_qkpv_block_h,
            BLOCK_T=page_size,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            BLOCK_V=block_v,
            MAX_PAGES=max_pages,
            GROUP_PAGES=gen_qkpv_group_pages,
            ALLOW_PARTIAL_GROUPS=max_pages % gen_qkpv_group_pages != 0,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=True,
            ASSUME_VALID_PAGES=True,
            **gen_qkpv_launch_meta,
        )
        _fp4_mla_attention_online_qkpv_reduce_kernel[
            (num_gen, triton.cdiv(num_heads, gen_qkpv_block_h), num_dim_blocks)
        ](
            output,
            gen_qkpv_partial_o,
            gen_qkpv_partial_m,
            gen_qkpv_partial_l,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            gen_qkpv_partial_o.stride(0),
            gen_qkpv_partial_o.stride(1),
            gen_qkpv_partial_o.stride(2),
            gen_qkpv_partial_o.stride(3),
            gen_qkpv_partial_m.stride(0),
            gen_qkpv_partial_m.stride(1),
            NUM_HEADS=num_heads,
            V_HEAD_D=v_head_dim,
            NUM_PAGE_GROUPS=num_gen_qkpv_page_groups,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=gen_qkpv_block_h,
            BLOCK_V=block_v,
            FP4_PV=True,
            **pv_launch_meta,
        )
        return output

    online_qkpv_max_batch = (
        env_online_qkpv_max_batch if env_online_qkpv_max_batch is not None else 32
    )
    can_use_online_qkpv = (
        env_online_qkpv == 1
        and can_use_prepacked_v_for_pv
        and query_len_per_seq == 1
        and num_gen <= online_qkpv_max_batch
        and use_tma_data_load
        and assume_full_heads
        and assume_full_pages
        and assume_valid_pages
        and num_heads == 128
        and q_head_dim == 640
        and k_head_dim == 576
        and q_residual_dim == 64
        and v_head_dim == 512
        and page_size == 128
        and block_h in (64, 128)
        and block_k == 512
        and full_block_end == 512
        and tail_block_k == 128
        and block_v in (128, 256)
        and sf_per_page == 8
    )
    if can_use_online_qkpv:
        if v_repack_stream is not None:
            torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)
            v_repack_stream = None
        online_group_pages = (
            env_online_qkpv_group_pages if env_online_qkpv_group_pages is not None else 128
        )
        online_group_pages = max(1, min(int(online_group_pages), max_pages))
        num_online_page_groups = _ceil_div(max_pages, online_group_pages)
        online_partial_o = _workspace_tensor(
            None,
            (num_online_page_groups, num_gen, num_heads, v_head_dim),
            dtype=torch.float32,
            device=q_fp4.device,
            name="online_partial_o",
        )
        online_partial_m = _workspace_tensor(
            None,
            (num_online_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="online_partial_m",
        )
        online_partial_l = _workspace_tensor(
            None,
            (num_online_page_groups, num_gen, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
            name="online_partial_l",
        )
        _fp4_mla_attention_online_qkpv_group_kernel[
            (num_gen, num_head_blocks, num_dim_blocks * num_online_page_groups)
        ](
            online_partial_o,
            online_partial_m,
            online_partial_l,
            q_fp4_2d,
            q_sf_flat,
            kv_cache,
            sf_cache,
            v_sf,
            v_packed,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            num_pages,
            q_fp4_2d.stride(0),
            q_fp4_2d.stride(1),
            kv_s0,
            kv_s2,
            kv_s4,
            sf_cache.stride(0),
            v_sf.stride(0),
            online_partial_o.stride(0),
            online_partial_o.stride(1),
            online_partial_o.stride(2),
            online_partial_o.stride(3),
            online_partial_m.stride(0),
            online_partial_m.stride(1),
            q_fp4_2d.shape[0],
            sm_scale=sm_scale,
            NUM_HEADS=num_heads,
            Q_HEAD_D=q_head_dim,
            K_HEAD_D=k_head_dim,
            Q_RESIDUAL_D=q_residual_dim,
            V_HEAD_D=v_head_dim,
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
            BLOCK_V=block_v,
            MAX_PAGES=max_pages,
            GROUP_PAGES=online_group_pages,
            NUM_DIM_BLOCKS=num_dim_blocks,
            ALLOW_PARTIAL_GROUPS=max_pages % online_group_pages != 0,
            FP4_PV=env_online_qkpv_fp4_pv == 1,
            **page_stats_launch_meta,
        )
        _fp4_mla_attention_online_qkpv_reduce_kernel[(num_gen, num_head_blocks, num_dim_blocks)](
            output,
            online_partial_o,
            online_partial_m,
            online_partial_l,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            online_partial_o.stride(0),
            online_partial_o.stride(1),
            online_partial_o.stride(2),
            online_partial_o.stride(3),
            online_partial_m.stride(0),
            online_partial_m.stride(1),
            NUM_HEADS=num_heads,
            V_HEAD_D=v_head_dim,
            NUM_PAGE_GROUPS=num_online_page_groups,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            FP4_PV=env_online_qkpv_fp4_pv == 1,
            **pv_launch_meta,
        )
        return output

    if parallel_page_stats is None:
        parallel_page_stats = triton_backend == "nvt" and max_pages >= 8
    if pack_prob_in_page_stats is None:
        pack_prob_in_page_stats = parallel_page_stats and fused_prob_pack
    pack_prob_in_page_stats = bool(
        pack_prob_in_page_stats and parallel_page_stats and fused_prob_pack
    )
    debug_pack_prob_in_page_stats = (
        pack_prob_in_page_stats
        if env_debug_page_stats_pack is None
        else env_debug_page_stats_pack != 0
    )
    page_stats_group_sizes = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    if page_stats_group_size is None and env_group_pages is not None:
        page_stats_group_size = env_group_pages
    if page_stats_group_size is None:
        if p_by_query and query_len_per_seq > 1 and max_pages >= 512:
            target_group_pages = 512
        elif num_gen <= 32:
            target_group_pages = max(8, 8 * num_gen)
        elif num_gen >= 256:
            target_group_pages = 256
        else:
            target_group_pages = max(8, min(512, 4 * num_gen))
        target_group_pages = min(max_pages, target_group_pages)
        page_stats_group_size = next(
            (
                group_size
                for group_size in reversed(page_stats_group_sizes)
                if group_size <= target_group_pages
            ),
            8,
        )
    else:
        page_stats_group_size = int(page_stats_group_size)
    if env_mtp_page_stats_pair == 1 and p_by_query and query_len_per_seq == 4:
        if env_mtp_page_stats_pair_group_pages is not None:
            page_stats_group_size = int(env_mtp_page_stats_pair_group_pages)
        else:
            # The two-query MTP page-stats kernel duplicates the QK/prob-pack
            # work inside each page group. Keeping the group small avoids very
            # large TileIR kernels while still sharing K loads across query pairs.
            page_stats_group_size = min(page_stats_group_size, 2)
    grouped_storage_full_pages = assume_full_pages or (p_by_query and can_address_all_compact_pages)
    grouped_assume_valid_pages = assume_valid_pages or (
        p_by_query and can_address_all_compact_pages
    )
    can_group_page_stats = (
        page_stats_group_size in page_stats_group_sizes
        and triton_backend == "nvt"
        and parallel_page_stats
        and (pack_prob_in_page_stats or env_debug_stop_after_page_stats == 1)
        and use_tma_data_load
        and assume_full_heads
        and grouped_storage_full_pages
        and grouped_assume_valid_pages
        and num_heads == 128
        and q_head_dim == 640
        and k_head_dim == 576
        and q_residual_dim == 64
        and page_size == 128
        and block_h == 128
        and block_k == 512
        and full_block_end == 512
        and tail_block_k == 128
        and sf_per_page == 8
    )
    page_stats_group_size = page_stats_group_size if can_group_page_stats else 1
    num_page_groups = (
        _ceil_div(max_pages, page_stats_group_size) if page_stats_group_size > 1 else max_pages
    )
    allow_partial_page_groups = page_stats_group_size > 1 and max_pages % page_stats_group_size != 0
    group_reduce_stats = (
        env_group_reduce_stats != 0
        if env_group_reduce_stats is not None
        else triton_backend == "nvt"
    ) and page_stats_group_size > 1
    page_stats_max_entries = max_pages
    page_stats_num_groups = num_page_groups
    pv_apply_prob_scale = False
    page_max = None
    page_sum = None
    if parallel_page_stats:
        page_stats_shape = (num_gen, page_stats_max_entries, num_heads)
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
        can_use_mtp_page_stats_pair = (
            env_mtp_page_stats_pair == 1
            and p_by_query
            and query_len_per_seq == 4
            and can_address_all_compact_pages
            and block_h == 128
            and use_tma_data_load
            and assume_full_heads
            and grouped_storage_full_pages
            and grouped_assume_valid_pages
            and num_heads == 128
            and q_head_dim == 640
            and k_head_dim == 576
            and q_residual_dim == 64
            and page_size == 128
            and block_k == 512
            and full_block_end == 512
            and tail_block_k == 128
            and sf_per_page == 8
        )
        split_tail_group_enabled = (
            env_mtp_split_tail_group != 0 if env_mtp_split_tail_group is not None else False
        )
        can_split_mtp_tail_group = (
            split_tail_group_enabled
            and not can_use_mtp_page_stats_pair
            and p_by_query
            and query_len_per_seq > 1
            and page_stats_group_size > 1
            and num_page_groups > 1
            and (pack_prob_in_page_stats or env_debug_stop_after_page_stats == 1)
            and group_reduce_stats
            and can_address_all_compact_pages
            and block_h == 128
            and use_tma_data_load
            and assume_full_heads
            and grouped_storage_full_pages
            and grouped_assume_valid_pages
            and num_heads == 128
            and q_head_dim == 640
            and k_head_dim == 576
            and q_residual_dim == 64
            and page_size == 128
            and block_k == 512
            and full_block_end == 512
            and tail_block_k == 128
            and sf_per_page == 8
        )
        if page_stats_group_size > 1 or can_use_mtp_page_stats_pair:
            if can_use_mtp_page_stats_pair:
                _fp4_mla_attention_page_stats_grouped_mtp_pair_kernel[
                    (
                        num_gen_seqs,
                        num_head_blocks,
                        num_page_groups * (query_len_per_seq // 2),
                    )
                ](
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
                    QUERY_LEN_PER_SEQ=query_len_per_seq,
                    BLOCK_H=block_h,
                    BLOCK_T=page_size,
                    BLOCK_K=block_k,
                    FULL_BLOCK_END=full_block_end,
                    TAIL_BLOCK_K=tail_block_k,
                    USE_TMA_DATA_LOAD=use_tma_data_load,
                    PACK_PROBS=debug_pack_prob_in_page_stats,
                    GROUP_REDUCE_STATS=group_reduce_stats,
                    ASSUME_FULL_HEADS=assume_full_heads,
                    ASSUME_FULL_PAGES=assume_full_pages,
                    ASSUME_VALID_PAGES=grouped_assume_valid_pages,
                    MAX_PAGES=max_pages,
                    GROUP_PAGES=page_stats_group_size,
                    ALLOW_PARTIAL_GROUPS=allow_partial_page_groups,
                    Q_PER_GROUP=2,
                    DUPLICATE_TAIL_K=env_duplicate_tail_k == 1,
                    **page_stats_launch_meta,
                )
            else:
                page_stats_grouped_kernel = (
                    _fp4_mla_attention_page_stats_grouped_kernel
                    if block_h == 128
                    else _fp4_mla_attention_page_stats_grouped_generic_kernel
                )
                if can_split_mtp_tail_group:
                    page_group_launches = (
                        (num_page_groups - 1, True, False, 0),
                        (1, False, allow_partial_page_groups, num_page_groups - 1),
                    )
                else:
                    page_group_launches = (
                        (num_page_groups, assume_full_pages, allow_partial_page_groups, 0),
                    )
                for (
                    grid_page_groups,
                    launch_assume_full_pages,
                    launch_allow_partial_groups,
                    page_group_offset,
                ) in page_group_launches:
                    if grid_page_groups <= 0:
                        continue
                    page_stats_grouped_kernel[(num_gen, num_head_blocks, grid_page_groups)](
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
                        QUERY_LEN_PER_SEQ=query_len_per_seq,
                        P_BY_QUERY=p_by_query,
                        BLOCK_H=block_h,
                        BLOCK_T=page_size,
                        BLOCK_K=block_k,
                        FULL_BLOCK_END=full_block_end,
                        TAIL_BLOCK_K=tail_block_k,
                        USE_TMA_DATA_LOAD=use_tma_data_load,
                        PACK_PROBS=debug_pack_prob_in_page_stats,
                        GROUP_REDUCE_STATS=group_reduce_stats,
                        ASSUME_FULL_HEADS=assume_full_heads,
                        ASSUME_FULL_PAGES=launch_assume_full_pages,
                        MASK_MTP_FINAL_PAGE_ONLY=mask_mtp_final_page_only,
                        ASSUME_VALID_PAGES=grouped_assume_valid_pages,
                        MAX_PAGES=max_pages,
                        GROUP_PAGES=page_stats_group_size,
                        ALLOW_PARTIAL_GROUPS=launch_allow_partial_groups,
                        PAGE_GROUP_OFFSET=page_group_offset,
                        DUPLICATE_TAIL_K=env_duplicate_tail_k == 1,
                        **page_stats_launch_meta,
                    )
        else:
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
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_BY_QUERY=p_by_query,
                BLOCK_H=block_h,
                BLOCK_T=page_size,
                BLOCK_K=block_k,
                FULL_BLOCK_END=full_block_end,
                TAIL_BLOCK_K=tail_block_k,
                USE_TMA_DATA_LOAD=use_tma_data_load,
                PACK_PROBS=debug_pack_prob_in_page_stats,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **page_stats_launch_meta,
            )
        _debug_mark("page_stats")
        if env_debug_stop_after_page_stats == 1:
            _debug_report()
            return output
        group_pv_max_batch = env_group_pv_max_batch if env_group_pv_max_batch is not None else 64
        can_use_group_pv = (
            env_group_pv == 1
            and page_stats_group_size > 1
            and group_reduce_stats
            and can_use_prepacked_v_for_pv
            and query_len_per_seq == 1
            and not p_by_query
            and num_gen <= group_pv_max_batch
            and use_tma_data_load
            and assume_full_heads
            and assume_full_pages
            and assume_valid_pages
            and num_heads == 128
            and v_head_dim == 512
            and page_size == 128
            and block_h == 128
            and block_v in (128, 256)
            and sf_per_page == 8
        )
        if can_use_group_pv:
            if v_repack_stream is not None:
                torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)
                v_repack_stream = None
            group_pv_partial_dtype = torch.float32
            if os.environ.get("TRTLLM_FP4_MLA_GROUP_PV_PARTIAL_DTYPE", "").lower() in (
                "bf16",
                "bfloat16",
            ):
                group_pv_partial_dtype = torch.bfloat16
            group_partial_o = _workspace_tensor(
                None,
                (num_page_groups, num_gen, num_heads, v_head_dim),
                dtype=group_pv_partial_dtype,
                device=q_fp4.device,
                name="group_partial_o",
            )
            _fp4_mla_attention_pv_group_partial_prepacked_v_kernel[
                (num_gen, num_head_blocks, num_dim_blocks * num_page_groups)
            ](
                group_partial_o,
                p_fp4,
                p_sf,
                v_sf,
                v_packed,
                src_page_ids,
                paged_kv_indptr_decode,
                num_pages,
                group_partial_o.stride(0),
                group_partial_o.stride(1),
                group_partial_o.stride(2),
                group_partial_o.stride(3),
                group_partial_o.shape[0] * group_partial_o.shape[1] * group_partial_o.shape[2],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=v_head_dim,
                PAGE_SIZE=page_size,
                SF_PER_PAGE=sf_per_page,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                MAX_PAGES=max_pages,
                GROUP_PAGES=page_stats_group_size,
                NUM_DIM_BLOCKS=num_dim_blocks,
                ALLOW_PARTIAL_GROUPS=allow_partial_page_groups,
                PV_LOOP_STAGES=int(pv_loop_stages),
                **pv_launch_meta,
            )
            _fp4_mla_attention_pv_group_partial_reduce_kernel[
                (num_gen, num_head_blocks, num_dim_blocks)
            ](
                output,
                group_partial_o,
                page_sum,
                global_scale,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                group_partial_o.stride(0),
                group_partial_o.stride(1),
                group_partial_o.stride(2),
                group_partial_o.stride(3),
                group_partial_o.shape[0] * group_partial_o.shape[1] * group_partial_o.shape[2],
                page_sum.stride(0),
                page_sum.stride(1),
                NUM_HEADS=num_heads,
                V_HEAD_D=v_head_dim,
                NUM_PAGE_GROUPS=num_page_groups,
                P_GLOBAL_SCALE=p_global_scale,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                **pv_launch_meta,
            )
            return output
        scale_from_group_stats = (
            (
                env_scale_from_group_stats == 1
                or (env_scale_from_group_stats is None and num_gen >= 64)
            )
            and pack_prob_in_page_stats
            and group_reduce_stats
            and page_stats_group_size > 1
        )
        if not scale_from_group_stats:
            _fp4_mla_attention_reduce_stats_kernel[(num_gen, num_head_blocks)](
                max_scores,
                denom,
                page_max,
                page_sum,
                max_scores.stride(0),
                page_max.stride(0),
                page_max.stride(1),
                NUM_HEADS=num_heads,
                MAX_PAGES=page_stats_max_entries,
                BLOCK_H=block_h,
                GROUP_REDUCE_STATS=group_reduce_stats,
                GROUP_PAGES=page_stats_group_size,
                NUM_PAGE_GROUPS=page_stats_num_groups,
                **page_stats_launch_meta,
            )
        pv_apply_prob_scale = (
            env_pv_apply_prob_scale == 1
            and pack_prob_in_page_stats
            and not scale_from_group_stats
            and can_use_prepacked_v_for_pv
            and query_len_per_seq == 1
            and not p_by_query
            and page_max is not None
        )
        if pack_prob_in_page_stats and not pv_apply_prob_scale:
            if scale_from_group_stats:
                prob_scale_assume_valid_pages = (
                    grouped_assume_valid_pages if mask_mtp_final_page_only else assume_valid_pages
                )

                def _launch_prob_scale_from_group_stats(
                    grid_pages: int,
                    launch_assume_full_pages: bool,
                    page_rel_offset: int,
                ) -> None:
                    if grid_pages <= 0:
                        return
                    _fp4_mla_attention_prob_scale_from_group_stats_kernel[
                        (num_gen, num_head_blocks, grid_pages)
                    ](
                        p_sf,
                        page_max,
                        page_sum,
                        paged_kv_indptr_decode,
                        kv_lens,
                        src_page_ids.shape[0],
                        page_max.stride(0),
                        page_max.stride(1),
                        NUM_HEADS=num_heads,
                        PAGE_SIZE=page_size,
                        SF_PER_PAGE=sf_per_page,
                        QUERY_LEN_PER_SEQ=query_len_per_seq,
                        MAX_PAGES=max_pages,
                        P_BY_QUERY=p_by_query,
                        BLOCK_H=block_h,
                        NUM_PAGE_GROUPS=num_page_groups,
                        ASSUME_FULL_HEADS=assume_full_heads,
                        ASSUME_FULL_PAGES=launch_assume_full_pages,
                        ASSUME_VALID_PAGES=prob_scale_assume_valid_pages,
                        PAGE_REL_OFFSET=page_rel_offset,
                        **page_stats_launch_meta,
                    )

                if mask_mtp_final_page_only:
                    _launch_prob_scale_from_group_stats(max_pages - 1, True, 0)
                    _launch_prob_scale_from_group_stats(1, False, max_pages - 1)
                else:
                    _launch_prob_scale_from_group_stats(max_pages, assume_full_pages, 0)
            else:
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
                    QUERY_LEN_PER_SEQ=query_len_per_seq,
                    MAX_PAGES=max_pages,
                    P_BY_QUERY=p_by_query,
                    BLOCK_H=block_h,
                    ASSUME_FULL_HEADS=assume_full_heads,
                    ASSUME_FULL_PAGES=assume_full_pages,
                    ASSUME_VALID_PAGES=assume_valid_pages,
                    **page_stats_launch_meta,
                )
        _debug_mark("prob_scale")
        can_use_pv_atomic_split = (
            env_pv_atomic_split == 1
            and pack_prob_in_page_stats
            and can_use_prepacked_v_for_pv
            and query_len_per_seq == 1
            and not p_by_query
            and use_tma_data_load
            and assume_full_heads
            and assume_full_pages
            and assume_valid_pages
            and assume_full_pv_heads
            and pv_block_h == 128
            and num_heads == 128
            and v_head_dim == 512
            and page_size == 128
            and block_h == 128
            and block_v in (128, 256)
            and sf_per_page == 8
        )
        if can_use_pv_atomic_split:
            if v_repack_stream is not None:
                torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)
                v_repack_stream = None
            pv_atomic_group_pages = (
                env_pv_atomic_group_pages
                if env_pv_atomic_group_pages is not None
                else page_stats_group_size
            )
            pv_atomic_group_pages = max(1, min(int(pv_atomic_group_pages), max_pages))
            num_pv_atomic_page_groups = _ceil_div(max_pages, pv_atomic_group_pages)
            out_acc = _workspace_tensor(
                None,
                (num_gen, num_heads, v_head_dim),
                dtype=torch.float32,
                device=q_fp4.device,
                name="pv_atomic_out_acc",
            )
            out_acc.zero_()
            _fp4_mla_attention_pv_atomic_split_prepacked_v_kernel[
                (num_gen, num_pv_head_blocks, num_dim_blocks * num_pv_atomic_page_groups)
            ](
                out_acc,
                p_fp4,
                p_sf,
                v_sf,
                v_packed,
                global_scale,
                src_page_ids,
                paged_kv_indptr_decode,
                num_pages,
                out_acc.stride(0),
                out_acc.stride(1),
                out_acc.stride(2),
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=v_head_dim,
                PAGE_SIZE=page_size,
                SF_PER_PAGE=sf_per_page,
                P_GLOBAL_SCALE=p_global_scale,
                BLOCK_H=pv_block_h,
                BLOCK_V=block_v,
                MAX_PAGES=max_pages,
                GROUP_PAGES=pv_atomic_group_pages,
                NUM_DIM_BLOCKS=num_dim_blocks,
                ALLOW_PARTIAL_GROUPS=max_pages % pv_atomic_group_pages != 0,
                PV_LOOP_STAGES=int(pv_loop_stages),
                **pv_launch_meta,
            )
            _fp4_mla_attention_cast_acc_kernel[(num_gen, num_pv_head_blocks, num_dim_blocks)](
                output,
                out_acc,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                out_acc.stride(0),
                out_acc.stride(1),
                out_acc.stride(2),
                NUM_HEADS=num_heads,
                V_HEAD_D=v_head_dim,
                BLOCK_H=pv_block_h,
                BLOCK_V=block_v,
                **pv_launch_meta,
            )
            return output
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
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            BLOCK_H=block_h,
            BLOCK_T=page_size,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **page_stats_launch_meta,
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
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_BY_QUERY=p_by_query,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                FULL_BLOCK_END=full_block_end,
                TAIL_BLOCK_K=tail_block_k,
                USE_TMA_DATA_LOAD=use_tma_data_load,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **page_stats_launch_meta,
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
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **page_stats_launch_meta,
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
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_BY_QUERY=p_by_query,
            BLOCK_H=block_h,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            **page_stats_launch_meta,
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
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_BY_QUERY=p_by_query,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            FULL_BLOCK_END=full_block_end,
            TAIL_BLOCK_K=tail_block_k,
            USE_TMA_DATA_LOAD=use_tma_data_load,
            PAGE_REL_FROM_GRID=True,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            **page_stats_launch_meta,
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

    if v_repack_stream is not None:
        torch.cuda.current_stream(q_fp4.device).wait_stream(v_repack_stream)

    if can_use_prepacked_v_for_pv:
        mtp_pv_pair_enabled = env_mtp_pv_pair != 0 if env_mtp_pv_pair is not None else True
        can_use_mtp_pv_pair = (
            mtp_pv_pair_enabled
            and p_by_query
            and query_len_per_seq == 4
            and can_address_all_compact_pages
            and num_heads == 128
            and v_head_dim == 512
            and page_size == 128
            and pv_block_h == 128
            and block_v in (128, 256)
            and sf_per_page == 8
            and assume_full_pv_heads
            and assume_full_v
        )
        if can_use_mtp_pv_pair:
            _fp4_mla_attention_pv_mtp_pair_prepacked_v_kernel[
                (
                    num_gen_seqs,
                    num_pv_head_blocks,
                    num_dim_blocks * (query_len_per_seq // 2),
                )
            ](
                output,
                p_fp4,
                p_sf,
                v_sf,
                v_packed,
                global_scale,
                src_page_ids,
                paged_kv_indptr_decode,
                num_pages,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=v_head_dim,
                PAGE_SIZE=page_size,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=p_global_scale,
                BLOCK_H=pv_block_h,
                BLOCK_V=block_v,
                NUM_DIM_BLOCKS=num_dim_blocks,
                Q_PER_GROUP=2,
                PV_LOOP_STAGES=int(pv_loop_stages),
                **pv_launch_meta,
            )
            _debug_mark("pv")
            _debug_report()
            return output
        _fp4_mla_attention_pv_prepacked_v_kernel[
            (
                num_gen,
                num_pv_head_blocks,
                num_dim_blocks,
            )
        ](
            output,
            p_fp4,
            p_sf,
            kv_cache,
            v_sf,
            v_packed,
            global_scale,
            max_scores,
            denom,
            page_max if page_max is not None else max_scores,
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
            max_scores.stride(0),
            page_max.stride(0) if page_max is not None else max_scores.stride(0),
            page_max.stride(1) if page_max is not None else max_scores.stride(0),
            kv_s0,
            kv_s2,
            kv_s4,
            v_sf.stride(0),
            NUM_HEADS=num_heads,
            V_HEAD_D=v_head_dim,
            PAGE_SIZE=page_size,
            FP4_BLOCK=FP4_BLOCK_SIZE,
            SF_PER_PAGE=sf_per_page,
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_BY_QUERY=p_by_query,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=pv_block_h,
            BLOCK_V=block_v,
            USE_TMA_P_LOAD=use_pv_p_tma_data_load and assume_full_pv_heads and assume_valid_pages,
            USE_TMA_V_LOAD=use_pv_v_tma_data_load and v_head_dim % block_v == 0,
            USE_TMA_OUT_STORE=use_pv_out_tma_data_store
            and (not p_by_query or env_pv_out_tma == 1)
            and assume_full_pv_heads
            and assume_full_v,
            USE_PREPACKED_V=True,
            PV_M_PACKED_V=False,
            PV_LOOP_STAGES=int(pv_loop_stages),
            PV_APPLY_PROB_SCALE=pv_apply_prob_scale,
            PV_SCALE_IN_SF=env_pv_scale_in_sf == 1,
            ASSUME_FULL_HEADS=assume_full_pv_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_FULL_V=assume_full_v,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **pv_launch_meta,
        )
        _debug_mark("pv")
    else:
        num_dim_blocks = triton.cdiv(v_head_dim, block_v)
        _fp4_mla_attention_pv_kernel[
            (
                num_gen,
                num_pv_head_blocks,
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
            QUERY_LEN_PER_SEQ=query_len_per_seq,
            MAX_PAGES=max_pages,
            P_BY_QUERY=p_by_query,
            P_GLOBAL_SCALE=p_global_scale,
            BLOCK_H=pv_block_h,
            BLOCK_V=block_v,
            USE_TMA_P_LOAD=use_pv_p_tma_data_load and assume_full_pv_heads and assume_valid_pages,
            USE_TMA_V_LOAD=use_pv_v_tma_data_load and v_head_dim % block_v == 0,
            USE_TMA_OUT_STORE=use_pv_out_tma_data_store
            and (not p_by_query or env_pv_out_tma == 1)
            and assume_full_pv_heads
            and assume_full_v,
            PV_LOOP_STAGES=int(pv_loop_stages),
            ASSUME_FULL_HEADS=assume_full_pv_heads,
            ASSUME_FULL_PAGES=assume_full_pages,
            ASSUME_FULL_V=assume_full_v,
            ASSUME_VALID_PAGES=assume_valid_pages,
            **pv_launch_meta,
        )
        _debug_mark("pv")
    _debug_report()
    return output


fp4_mla_paged_attention = fp4_mla_paged_attention_internal
