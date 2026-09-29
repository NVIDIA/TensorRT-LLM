# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register, TMEM, and SMEM layout helpers for MLA decode."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import StageInfo

from ..throughput_latency_1cta.config import MlaConfig
from .constants import (
    O_STAGE_COPY_SEGMENT_BYTES,
    SMEM_ROW_BYTES,
    SMEM_ROW_BYTE_SHIFT,
    SMEM_VECTOR_BYTES,
    SMEM_VECTOR_BYTE_SHIFT,
    STSM_MATRICES_PER_WARP,
    STSM_MATRICES_PER_WARP_SHIFT,
    STSM_MATRIX_LANES,
    STSM_MATRIX_LANE_SHIFT,
    STSM_ROW_BLOCK_ROWS,
    STSM_WARPS_PER_SLICE,
    STSM_WARPS_PER_SLICE_SHIFT,
    SWIZZLE_ROW_MASK,
    TCGEN05_SECOND_PANEL_ADDR_OFFSET,
)
from .math import ceil_div, mma_k_step_for_qkv


# Per-task cache tuple slots.  Tasks share these indices when passing common
# CTA-local coordinates and runtime sequence length into resource work methods.
_TASK_CACHE_TMEM_BASE_OFFSET = 0
_TASK_CACHE_WARP_GRP_THREAD_IDX = 1
_TASK_CACHE_WARP_IDX = 2
_TASK_CACHE_LANE_IDX = 3
_TASK_CACHE_SEQ_LEN_KV = 4


def num_softmax_scale_groups(cfg: MlaConfig) -> int:
    """Return the number of independent softmax scale groups per thread."""
    if cfg.kernel_variant == "keeps_mma_ab":
        return 1
    return max(cfg.tile_size_q // 4, 4)


def num_s_regs_per_thread(cfg: MlaConfig) -> int:
    """Return the number of score registers carried by each softmax thread."""
    if cfg.kernel_variant == "keeps_mma_ab":
        return cfg.tile_size_kv // 2
    return num_softmax_scale_groups(cfg) * 4


def softmax_scratch_words(cfg: MlaConfig) -> int:
    """Return the shared scratch words needed for softmax reductions."""
    if cfg.kernel_variant == "keeps_mma_ab":
        return 2 * 384
    return 4 * num_softmax_scale_groups(cfg)


def num_packed_p_regs(cfg: MlaConfig) -> int:
    """Return the number of packed P registers per softmax thread."""
    q_repeats = max(cfg.tile_size_q // 8, 1)
    return (2 if cfg.is_fp8_qkv() else 4) * q_repeats


def num_o_reg_pairs(cfg: MlaConfig) -> int:
    """Return the number of packed O register pairs handled per thread."""
    return 2 * max(max(cfg.tile_size_q, 16) // 8, 1)


def num_o_repeats(cfg: MlaConfig) -> int:
    """Return the O repeat count for TMEM loads and GMEM stores."""
    return max(max(cfg.tile_size_q, 16) // 8, 1)


def num_q_repeats(cfg: MlaConfig) -> int:
    """Return the Q/head repeat count for a warpgroup row tile."""
    return max(cfg.tile_size_q // 8, 1)


def num_o_stsm_row_blocks(cfg: MlaConfig) -> int:
    """Return the number of 16-row STSM blocks needed for one O tile."""
    return max(cfg.tile_size_q // 16, 1)


def num_o_tmem_loads_per_stage(cfg: MlaConfig) -> int:
    """Return the number of TMEM O load groups per output stage."""
    bf16_equivalent_bytes = cfg.tile_size_q * cfg.head_dim_per_stage_v * 2
    bf16_equivalent_segments = max(1, ceil_div(bf16_equivalent_bytes, 2048))
    return max(bf16_equivalent_segments // num_o_stsm_row_blocks(cfg), 1)


def num_fp8_output_regs(cfg: MlaConfig) -> int:
    """Return the number of packed FP8 output registers per stage."""
    return max((cfg.tile_size_q * cfg.head_dim_per_stage_v) // 512, 1)


def q_p_desc_k_block_wrap_bytes(cfg: MlaConfig) -> int:
    """Return Q/P descriptor K-wrap distance in bytes."""
    k_block_bytes = mma_k_step_for_qkv(cfg) * cfg.qkv_dtype_bytes
    k_blocks_per_smem_row = 128 // k_block_bytes
    return cfg.tile_size_q * 128 - (k_blocks_per_smem_row - 1) * k_block_bytes


def q_p_desc_k_block_wrap_units(cfg: MlaConfig) -> int:
    """Return Q/P descriptor K-wrap distance in 16-byte units."""
    return q_p_desc_k_block_wrap_bytes(cfg) // 16


def tma_inner_dim_elems(cfg: MlaConfig) -> int:
    """Return the maximum contiguous TMA inner dimension in Q/K/V elements."""
    return 128 // cfg.qkv_dtype_bytes


def tma_page_token_elems(cfg: MlaConfig) -> int:
    """Return the number of page-offset tokens represented by one TMA page."""
    return cfg.num_tokens_per_page


@cute.jit
def p_stsm_smem_offset_bytes(
    local_warp_idx: Int32,
    lane_idx: Int32,
    stsm_group_idx: int = 0,
    stsm_row_block_idx: int = 0,
    tile_size_q: int = 16,
):
    """Return the SMEM byte offset for one packed P stmatrix store lane."""
    # STSM maps four 8-lane matrices per warp.  Offsets below express the
    # tcgen05 128B-row / 16B-column swizzled SMEM layout used for packed P.
    slice_idx = local_warp_idx // Int32(STSM_WARPS_PER_SLICE)
    warp_idx_in_slice = local_warp_idx % Int32(STSM_WARPS_PER_SLICE)
    mtx_idx = lane_idx // Int32(STSM_MATRIX_LANES)
    thr_row_idx = lane_idx % Int32(STSM_MATRIX_LANES)
    if cutlass.const_expr(tile_size_q == 8):
        mtx_row_idx = Int32(0)
        mtx_col_idx = warp_idx_in_slice * Int32(STSM_MATRICES_PER_WARP) + mtx_idx
    else:
        mtx_row_idx = mtx_idx // Int32(STSM_WARPS_PER_SLICE)
        mtx_col_idx = (
            warp_idx_in_slice * Int32(STSM_MATRICES_PER_WARP)
            + (mtx_idx % Int32(STSM_WARPS_PER_SLICE))
            + Int32(stsm_group_idx * STSM_WARPS_PER_SLICE)
        )
    return (
        slice_idx * Int32(tile_size_q * SMEM_ROW_BYTES)
        + Int32(stsm_row_block_idx * STSM_ROW_BLOCK_ROWS * SMEM_ROW_BYTES)
        + mtx_row_idx * Int32(STSM_MATRIX_LANES * SMEM_ROW_BYTES)
        + thr_row_idx * Int32(SMEM_ROW_BYTES)
        + ((mtx_col_idx ^ thr_row_idx) * Int32(SMEM_VECTOR_BYTES))
    )


@cute.jit
def o_stage_stsm_and_copy_offsets(
    cfg: MlaConfig,
    warp_grp_thread_idx: Int32,
    local_warp_idx: Int32,
    lane_idx: Int32,
    stsm_group_idx: int = 0,
    copy_segment_idx: int = 0,
):
    """Return SMEM store, load, row, and column offsets for one O stage."""
    # The copy path treats each thread as a 16B vector lane.  A 2048B segment is
    # one 16-row x 128B SMEM tile, matching the STSM/TMEM load granularity.
    base_offset = (warp_grp_thread_idx << Int32(SMEM_VECTOR_BYTE_SHIFT)) + Int32(
        copy_segment_idx * O_STAGE_COPY_SEGMENT_BYTES
    )
    smem_row_idx = base_offset >> Int32(SMEM_ROW_BYTE_SHIFT)
    thr_row_idx = lane_idx & Int32(SWIZZLE_ROW_MASK)
    mtx_idx = lane_idx >> Int32(STSM_MATRIX_LANE_SHIFT)
    # Swizzle the 16B vector column by the low three row bits to match the 128B
    # shared-memory layout expected by vectorized GMEM stores.
    load_smem_offset = base_offset ^ (
        (smem_row_idx & Int32(SWIZZLE_ROW_MASK)) << Int32(SMEM_VECTOR_BYTE_SHIFT)
    )
    if cutlass.const_expr(
        cfg.head_dim_per_stage_v * cfg.partial_o_dtype_bytes > SMEM_ROW_BYTES
    ):
        slice_idx = local_warp_idx >> Int32(STSM_WARPS_PER_SLICE_SHIFT)
        warp_idx_in_slice = local_warp_idx & Int32(STSM_WARPS_PER_SLICE - 1)
        if cutlass.const_expr(cfg.tile_size_q == 8):
            mtx_row_idx = Int32(0)
            mtx_col_idx = (
                warp_idx_in_slice << Int32(STSM_MATRICES_PER_WARP_SHIFT)
            ) + mtx_idx
        else:
            mtx_row_idx = mtx_idx >> Int32(STSM_WARPS_PER_SLICE_SHIFT)
            mtx_col_idx = (
                (warp_idx_in_slice << Int32(STSM_MATRICES_PER_WARP_SHIFT))
                + (mtx_idx & Int32(STSM_WARPS_PER_SLICE - 1))
                + Int32(stsm_group_idx * STSM_WARPS_PER_SLICE)
            )
        smem_offset_bytes = (
            Int32(copy_segment_idx * O_STAGE_COPY_SEGMENT_BYTES)
            + slice_idx * Int32(cfg.tile_size_q * SMEM_ROW_BYTES)
            + (mtx_row_idx * Int32(STSM_MATRIX_LANES) + thr_row_idx)
            * Int32(SMEM_ROW_BYTES)
            + ((mtx_col_idx ^ thr_row_idx) * Int32(SMEM_VECTOR_BYTES))
        )
        dst_row_idx = smem_row_idx % Int32(cfg.tile_size_q)
        dst_col_offset = (smem_row_idx // Int32(cfg.tile_size_q)) * Int32(
            SMEM_ROW_BYTES
        ) + (base_offset & Int32(SMEM_ROW_BYTES - 1))
    else:
        mtx_row_idx = mtx_idx >> Int32(STSM_WARPS_PER_SLICE_SHIFT)
        mtx_col_idx = mtx_idx & Int32(STSM_WARPS_PER_SLICE - 1)
        seg_col_idx = (
            (local_warp_idx << Int32(STSM_WARPS_PER_SLICE_SHIFT)) + mtx_col_idx
        ) ^ thr_row_idx
        if cutlass.const_expr(stsm_group_idx != 0):
            seg_col_idx = seg_col_idx + Int32(stsm_group_idx * STSM_WARPS_PER_SLICE)
        smem_offset_bytes = (
            mtx_row_idx * Int32(STSM_MATRIX_LANES) + thr_row_idx
        ) * Int32(SMEM_ROW_BYTES) + seg_col_idx * Int32(SMEM_VECTOR_BYTES)
        dst_row_idx = smem_row_idx
        dst_col_offset = base_offset & Int32(SMEM_ROW_BYTES - 1)
    return smem_offset_bytes, load_smem_offset, dst_row_idx, dst_col_offset


@cute.jit
def local_q_head_idx_for_scale(
    cfg: MlaConfig,
    col_group_idx: Int32,
    scale_idx,
):
    """Return the local Q/head row controlled by a softmax scale group."""
    # Scale groups are interleaved as pairs across columns, then advanced in
    # groups of eight rows to match the packed softmax register layout.
    local_head_idx = col_group_idx * Int32(STSM_WARPS_PER_SLICE) + (
        Int32(scale_idx) & Int32(STSM_WARPS_PER_SLICE - 1)
    )
    local_head_idx = local_head_idx + (
        Int32(scale_idx) >> Int32(STSM_WARPS_PER_SLICE_SHIFT)
    ) * Int32(STSM_MATRIX_LANES)
    return local_head_idx


def q_stage_elements(cfg: MlaConfig, stage_idx: int) -> int:
    """Return the number of Q elements stored for one QK head-dim stage."""
    return cfg.tile_size_q * cfg.qk_head_stage_width(stage_idx)


def kv_stage_elements(cfg: MlaConfig, stage_idx: int) -> int:
    """Return the number of K elements stored for one QK head-dim stage."""
    return cfg.tile_size_kv * cfg.qk_head_stage_width(stage_idx)


def v_stage_elements(cfg: MlaConfig, stage_idx: int) -> int:
    """Return the number of V elements stored for one V head-dim stage."""
    return cfg.tile_size_kv * cfg.v_head_stage_width(stage_idx)


def q_stage_smem_element_offset(cfg: MlaConfig, stage_idx: int) -> int:
    """Return the Q SMEM element offset for one QK head-dim stage."""
    return stage_idx * cfg.tile_size_q * cfg.head_dim_per_stage_kv


def kv_stage_smem_element_offset(cfg: MlaConfig, stage_idx: int) -> int:
    """Return the K/V SMEM element offset for one QK head-dim stage."""
    return stage_idx * cfg.tile_size_kv * cfg.head_dim_per_stage_kv


@cute.jit
def head_dim_cta_offset_v(cfg: MlaConfig, cta_idx_head_dim_v):
    """Return the V head-dim element offset for a split head-dim CTA."""
    if cutlass.const_expr(cta_idx_head_dim_v is None):
        return Int32(0)
    return Int32(cta_idx_head_dim_v) * Int32(cfg.head_dim_per_cta_v)


def o_stage_tmem_col_offset(
    cfg: MlaConfig,
    o_stage_idx,
    v_stage_idx: int,
):
    """Return the TMEM column offset for one O pipeline and V head-dim stage."""
    if cfg.kernel_variant == "keeps_mma_ab" and cfg.head_dim_per_cta_v > 256:
        return (
            o_stage_idx * Int32(2 * cfg.tmem_o_buffer_cols)
            + Int32((v_stage_idx % 2) * cfg.tmem_o_buffer_cols)
            + Int32((v_stage_idx // 2) * TCGEN05_SECOND_PANEL_ADDR_OFFSET)
        )
    return o_stage_idx * Int32(cfg.tmem_o_buffer_cols * cfg.v_head_dim_stages) + Int32(
        v_stage_idx * cfg.tmem_o_buffer_cols
    )


def smem_array(context, alloc: SmemAllocation, dtype, count: int):
    """Create a shared-memory array view for an allocation, or None if absent."""
    if context is None or context.smem_base is None or alloc is None:
        return None
    return cutlass.Array(
        context.smem_base.data_ptr() + alloc.offset,
        dtype=dtype,
        shape=(count,),
        addrspace=3,
    )


@cute.jit
def decode_gen_task_cache(stage_info: StageInfo):
    """Return the cached per-task thread values, or a zero cache fallback."""
    if cutlass.const_expr(stage_info.task_cache is None):
        zero = Int32(0)
        return (zero, zero, zero, zero, zero, zero, zero, zero)
    return stage_info.task_cache
