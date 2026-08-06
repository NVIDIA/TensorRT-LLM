# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""SMEM/GMEM output helpers for FMHA decode TS resources.

Holds the transposed FP8 STSM stores, the SMEM→GMEM 16-byte vector copy,
the 16-bit O-reorg offset math, the P-STSM offset math, and partial-O
load helpers used by ``SmemPResource`` and ``TmemCorrResource``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    Constexpr,
    _q_physical_output_row,
    _q_row_is_valid_for_seq,
)


@cute.jit
def _keeps_p_smem_block_offset_bytes(
    cfg: Constexpr[FmhaDecodeConfig], row_idx: Int32, col_idx: Int32
) -> Int32:
    """JIT form of the Keeps P SWIZZLE_128B vector-block address."""
    dtype_bytes = cfg.q_dtype_bytes
    chunk_cols = 128 // dtype_bytes
    chunk_idx = col_idx // Int32(chunk_cols)
    col_in_chunk = col_idx - chunk_idx * Int32(chunk_cols)
    byte_col = col_in_chunk * Int32(dtype_bytes)
    return (
        chunk_idx * Int32(cfg.tile_size_q * 128)
        + row_idx * Int32(128)
        + (((byte_col >> Int32(4)) ^ (row_idx & Int32(0x7))) << Int32(4))
    )


@cute.jit
def _fp8_stsm_smem_dst(
    smem_base_i32: cutlass.Array,
    warp_grp_thread_idx: Int32,
    num_trans_rows: int,
    num_trans_cols: int,
    stsm_idx: int,
) -> cute.Pointer:
    """Return the swizzled SMEM destination for one FP8 transposed STSM."""
    # Compute the swizzled SMEM address used by transposed 8-bit STSM. The
    # mapping writes lane fragments in the same layout the later vector GMEM
    # copy expects.
    num_rows = Int32(num_trans_rows)
    num_bytes_per_row = Int32(num_trans_cols)
    num_rows_per_128b = Int32(128) // num_bytes_per_row
    num_segs_per_warp_per_row = num_bytes_per_row // Int32(16 * 4)
    num_stsm_per_row = max(8 // 32, 1)
    num_mtx_per_col = num_rows // Int32(8)
    warp_idx = warp_grp_thread_idx >> Int32(5)
    lane_idx = warp_grp_thread_idx & Int32(0x1F)
    thr_row_idx = lane_idx & Int32(0x7)
    mtx_idx = lane_idx >> Int32(3)
    mtx_row_idx = mtx_idx % num_mtx_per_col
    mtx_col_idx = mtx_idx // num_mtx_per_col

    stsm_row_idx = Int32(stsm_idx % num_stsm_per_row)
    stsm_col_idx = Int32(stsm_idx // num_stsm_per_row)
    xor_mask = thr_row_idx // num_rows_per_128b
    seg_col_idx = (
        warp_idx * num_segs_per_warp_per_row + mtx_col_idx + stsm_col_idx
    ) ^ xor_mask
    smem_offset = (
        mtx_row_idx * Int32(8) + thr_row_idx + stsm_row_idx * Int32(32)
    ) * num_bytes_per_row + seg_col_idx * Int32(16)
    return (smem_base_i32.subview((smem_offset >> Int32(2)))).data_ptr()


@cute.jit
def _store_transposed_smem8b_x1(
    smem_base_i32: cutlass.Array,
    reg0: Int32,
    warp_grp_thread_idx: Int32,
    num_trans_rows: int,
    num_trans_cols: int,
    stsm_idx: int = 0,
) -> None:
    """Store one packed 8-bit register with transposed stmatrix."""
    smem_dst = _fp8_stsm_smem_dst(
        smem_base_i32,
        warp_grp_thread_idx,
        num_trans_rows,
        num_trans_cols,
        stsm_idx,
    )
    prims.inline_ptx_hl(
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [{$r0}], {{$r1}};",
        read_only_args=[smem_dst, reg0],
    )


@cute.jit
def _store_transposed_smem8b_x2(
    smem_base_i32: cutlass.Array,
    reg0: Int32,
    reg1: Int32,
    warp_grp_thread_idx: Int32,
    num_trans_rows: int,
    num_trans_cols: int,
    stsm_idx: int = 0,
) -> None:
    """Store two packed 8-bit registers with transposed stmatrix."""
    smem_dst = _fp8_stsm_smem_dst(
        smem_base_i32,
        warp_grp_thread_idx,
        num_trans_rows,
        num_trans_cols,
        stsm_idx,
    )
    prims.inline_ptx_hl(
        "stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [{$r0}], {{$r1}, {$r2}};",
        read_only_args=[smem_dst, reg0, reg1],
    )


@cute.jit
def _store_transposed_smem8b_x4(
    smem_base_i32: cutlass.Array,
    reg0: Int32,
    reg1: Int32,
    reg2: Int32,
    reg3: Int32,
    warp_grp_thread_idx: Int32,
    num_trans_rows: int,
    num_trans_cols: int,
    stsm_idx: int = 0,
) -> None:
    """Store four packed 8-bit registers with transposed stmatrix."""
    smem_dst = _fp8_stsm_smem_dst(
        smem_base_i32,
        warp_grp_thread_idx,
        num_trans_rows,
        num_trans_cols,
        stsm_idx,
    )
    prims.inline_ptx_hl(
        "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[smem_dst, reg0, reg1, reg2, reg3],
    )


@cute.jit
def _store_transposed_smem8b(
    smem_base_i32: cutlass.Array,
    regs: cutlass.Array,
    warp_grp_thread_idx: Int32,
    num_trans_rows: int,
    num_trans_cols: int,
    num_regs: int,
) -> None:
    """Dispatch FP8 transposed stores for the register count owned by a lane."""
    # STSM's swizzled row is at most 128 bytes. Wider outputs are laid out as
    # consecutive 128-byte head-dimension bands, each containing every Q row.
    # This keeps the existing D64/D128 mapping and makes D256 independent of
    # TileQ instead of accidentally treating the second band as extra Q rows.
    assert num_trans_rows >= 8 and num_trans_rows % 8 == 0
    assert num_trans_cols > 0 and (
        num_trans_cols in (64, 128) or num_trans_cols % 128 == 0
    )
    num_col_bands = max((num_trans_cols + 127) // 128, 1)
    band_cols = min(num_trans_cols, 128)
    assert num_regs % num_col_bands == 0
    regs_per_band = num_regs // num_col_bands
    assert regs_per_band in (1, 2, 4, 8)
    for band_idx in cutlass.range_constexpr(num_col_bands):
        band_smem_base = smem_base_i32.subview(
            band_idx * num_trans_rows * band_cols // 4
        )
        reg_base = band_idx * regs_per_band
        if cutlass.const_expr(regs_per_band == 1):
            _store_transposed_smem8b_x1(
                band_smem_base,
                regs[reg_base],
                warp_grp_thread_idx,
                num_trans_rows,
                band_cols,
            )
        elif cutlass.const_expr(regs_per_band == 2):
            _store_transposed_smem8b_x2(
                band_smem_base,
                regs[reg_base],
                regs[reg_base + 1],
                warp_grp_thread_idx,
                num_trans_rows,
                band_cols,
            )
        else:
            _store_transposed_smem8b_x4(
                band_smem_base,
                regs[reg_base],
                regs[reg_base + 1],
                regs[reg_base + 2],
                regs[reg_base + 3],
                warp_grp_thread_idx,
                num_trans_rows,
                band_cols,
            )
            if cutlass.const_expr(regs_per_band > 4):
                _store_transposed_smem8b_x4(
                    band_smem_base,
                    regs[reg_base + 4],
                    regs[reg_base + 5],
                    regs[reg_base + 6],
                    regs[reg_base + 7],
                    warp_grp_thread_idx,
                    num_trans_rows,
                    band_cols,
                    1,
                )


def _fp8_smem_load_xor_bytes_for_shape(
    headdim: int, smem_row_idx: int | Int32
) -> int | Int32:
    """Return the byte XOR that reverses the FP8 STSM row swizzle.

    A 128-byte swizzle atom packs multiple D64 rows.  The store-side XOR is
    derived from the row inside each eight-row matrix, not the global packed
    row number; including matrix-row bits would flip bit 6 and exchange the two
    D64 rows sharing one 128-byte atom.
    """
    num_bytes_per_smem_row = min(headdim, 128)
    num_packed_smem_rows = 128 // num_bytes_per_smem_row
    return ((smem_row_idx % Int32(8)) // Int32(num_packed_smem_rows)) * Int32(16)


@cute.jit
def _copy_transposed_smem8b_to_gmem(
    smem_base_i32: cutlass.Array,
    o_ptr: cute.Pointer,
    cfg: Constexpr[FmhaDecodeConfig],
    logical_h_k_idx: Int32,
    logical_b_idx: Int32,
    logical_q_group_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    seq_len_q: Int32,
    q_token_offset: Int32,
    warp_grp_thread_idx: Int32,
    full_tile_rows: Constexpr[bool] = False,
) -> None:
    """Copy a transposed FP8 SMEM tile to the logical GMEM output layout."""
    # Reload the transposed 8-bit SMEM tile as contiguous 16-byte vectors and
    # store the vectors into the logical output row.
    headdim = cfg.headdim
    tile_size_q = cfg.tile_size_q
    num_bytes_per_smem_row = Int32(headdim if headdim <= 128 else 128)
    num_copy_segments = max((tile_size_q * headdim + 2047) // 2048, 1)
    for copy_segment_idx in cutlass.range_constexpr(num_copy_segments):
        base_offset = warp_grp_thread_idx * Int32(16) + Int32(copy_segment_idx * 2048)
        smem_row_idx = base_offset // num_bytes_per_smem_row
        load_smem_offset = base_offset ^ _fp8_smem_load_xor_bytes_for_shape(
            headdim, smem_row_idx
        )
        dst_row_idx = smem_row_idx
        dst_col_offset = base_offset % num_bytes_per_smem_row
        if headdim > 128:
            dst_row_idx = smem_row_idx % Int32(tile_size_q)
            dst_col_offset = (
                smem_row_idx // Int32(tile_size_q)
            ) * num_bytes_per_smem_row + (base_offset % num_bytes_per_smem_row)
        physical_dst_row_idx = _q_physical_output_row(
            cfg,
            h_r,
            num_heads_kv,
            logical_b_idx,
            logical_h_k_idx,
            logical_q_group_idx,
            dst_row_idx,
            q_token_offset,
        )
        valid_output_row = _q_row_is_valid_for_seq(
            cfg,
            h_r,
            logical_q_group_idx,
            dst_row_idx,
            seq_len_q,
        )
        if cutlass.const_expr(full_tile_rows):
            smem_src_full = (
                smem_base_i32.subview((load_smem_offset >> Int32(2)))
            ).data_ptr()
            # The flattened row is intentionally Int32, but the byte address
            # may exceed 2 GiB for a valid packed-Q output.  Widen before the
            # row-stride product so it cannot wrap in signed 32-bit arithmetic.
            dst_row_base_full = Int64(physical_dst_row_idx) * Int64(headdim)
            dst_ptr_full = cutlass.inttoptr(
                o_ptr.toint() + dst_row_base_full + Int64(dst_col_offset),
                mem_space=1,
                dtype=Int32,
            )
            dst_ptr_full.store(smem_src_full.load(count=4, alignment=16), alignment=16)
        else:
            if valid_output_row:
                smem_src_guarded = (
                    smem_base_i32.subview((load_smem_offset >> Int32(2)))
                ).data_ptr()
                dst_row_base_guarded = Int64(physical_dst_row_idx) * Int64(headdim)
                dst_ptr_guarded = cutlass.inttoptr(
                    o_ptr.toint() + dst_row_base_guarded + Int64(dst_col_offset),
                    mem_space=1,
                    dtype=Int32,
                )
                dst_ptr_guarded.store(
                    smem_src_guarded.load(count=4, alignment=16), alignment=16
                )


@cute.jit
def _transposed_smem128x16b_stsm_offset_bytes(
    num_rows: Constexpr[int],
    local_warp_idx: Int32,
    lane_idx: Int32,
    stsm_group_idx: int = 0,
) -> Int32:
    """Compute the SMEM byte offset for one transposed stmatrix store."""
    # Store transposed 128x16b rows. Once there are multiple TMEM load repeats, STSM
    # groups advance both row and column sub-tiles.
    num_tmem_load_reps = max(num_rows // 8, 1)
    if cutlass.const_expr(num_tmem_load_reps == 1):
        num_mtx_per_row_per_stsm = 4
        num_mtx_per_col_per_stsm = 1
    else:
        num_mtx_per_row_per_stsm = 2
        num_mtx_per_col_per_stsm = 2
    num_stsm_per_col = max(num_tmem_load_reps // num_mtx_per_col_per_stsm, 1)
    stsm_row_group_idx = stsm_group_idx // num_stsm_per_col
    stsm_col_group_idx = stsm_group_idx % num_stsm_per_col

    slice_idx = local_warp_idx // Int32(2)
    warp_idx_in_slice = local_warp_idx % Int32(2)
    mtx_idx = lane_idx // Int32(8)
    mtx_row_idx = mtx_idx // Int32(num_mtx_per_row_per_stsm)
    thr_row_idx = lane_idx % Int32(8)
    mtx_col_idx = (
        warp_idx_in_slice * Int32(4)
        + (mtx_idx % Int32(num_mtx_per_row_per_stsm))
        + Int32(stsm_row_group_idx * num_mtx_per_row_per_stsm)
    )
    return (
        slice_idx * Int32(num_rows * 128)
        + (
            mtx_row_idx * Int32(8)
            + thr_row_idx
            + Int32(stsm_col_group_idx * num_mtx_per_col_per_stsm * 8)
        )
        * Int32(128)
        + ((mtx_col_idx ^ thr_row_idx) * Int32(16))
    )


@cute.jit
def _fp16_o_reorg_offsets(
    cfg: FmhaDecodeConfig,
    warp_grp_thread_idx: Int32,
    local_warp_idx: Int32,
    lane_idx: Int32,
    stsm_group_idx: int = 0,
    copy_segment_idx: int = 0,
) -> tuple[Int32, Int32, Int32, Int32]:
    """Return the SMEM reorg offsets for a 16-bit O/partial-O row."""

    o_stage_dtype_bytes = cfg.o_dtype_bytes
    if cutlass.const_expr(cfg.use_split_kv and cfg.use_fp8_output):
        # Split-KV partial O is staged in 16-bit form even when final O is FP8.
        o_stage_dtype_bytes = 2
    base_offset = (warp_grp_thread_idx << Int32(4)) + Int32(copy_segment_idx * 2048)
    smem_row_idx = base_offset >> Int32(7)
    slice_idx = local_warp_idx >> Int32(1)
    warp_idx_in_slice = local_warp_idx & Int32(1)
    thr_row_idx = lane_idx & Int32(0x7)
    if cutlass.const_expr(cfg.headdim * 2 > 128):
        if cutlass.const_expr(cfg.tile_size_q >= 16):
            stsm_per_head_dim_stage = max(cfg.tile_size_q // 8, 1)
            head_dim_stage_idx = stsm_group_idx // stsm_per_head_dim_stage
            stage_stsm_group_idx = stsm_group_idx % stsm_per_head_dim_stage
            smem_offset_bytes = Int32(
                head_dim_stage_idx
                * cfg.tile_size_q
                * cfg.head_dim_kv_stage
                * o_stage_dtype_bytes
            ) + _transposed_smem128x16b_stsm_offset_bytes(
                cfg.tile_size_q,
                local_warp_idx,
                lane_idx,
                stage_stsm_group_idx,
            )
        else:
            mtx_col_idx_256b = (warp_idx_in_slice << Int32(2)) + (
                (lane_idx >> Int32(3)) & Int32(0x3)
            )
            smem_offset_bytes = (
                slice_idx * Int32(8 * 128)
                + Int32(stsm_group_idx * 16 * 128)
                + thr_row_idx * Int32(128)
                + ((mtx_col_idx_256b ^ thr_row_idx) * Int32(16))
            )
        load_smem_offset = base_offset ^ ((smem_row_idx & Int32(0x7)) << Int32(4))
        dst_row_idx = smem_row_idx % Int32(cfg.tile_size_q)
        dst_col_offset = (smem_row_idx // Int32(cfg.tile_size_q)) * Int32(128) + (
            base_offset & Int32(0x7F)
        )
    else:
        mtx_idx = lane_idx >> Int32(3)
        mtx_row_idx = mtx_idx >> Int32(1)
        mtx_col_idx = mtx_idx & Int32(1)
        seg_col_idx = ((local_warp_idx << Int32(1)) + mtx_col_idx) ^ thr_row_idx
        smem_row = mtx_row_idx * Int32(8) + thr_row_idx + Int32(stsm_group_idx * 16)
        smem_offset_bytes = smem_row * Int32(128) + seg_col_idx * Int32(16)
        load_smem_offset = base_offset ^ ((smem_row_idx & Int32(0x7)) << Int32(4))
        dst_row_idx = smem_row_idx
        dst_col_offset = base_offset & Int32(0x7F)
    return smem_offset_bytes, load_smem_offset, dst_row_idx, dst_col_offset


@cute.jit
def _p_stsm_smem_offset_bytes(
    local_warp_idx: Int32,
    lane_idx: Int32,
    stsm_group_idx: int = 0,
    tile_size_q: int = 16,
) -> Int32:
    """Return the SMEM byte offset used when storing P with stmatrix."""
    return _transposed_smem128x16b_stsm_offset_bytes(
        tile_size_q, local_warp_idx, lane_idx, stsm_group_idx
    )


@cute.jit
def _load_partial_o_vec8_as_f32(
    regs_i32: cutlass.Array, use_bf16_partial: Constexpr[bool]
) -> cutlass.Array:
    """Convert one packed split-KV partial-O vector to FP32 values."""
    regs_vec = cutlass.Vector.from_elements(
        (regs_i32[0], regs_i32[1], regs_i32[2], regs_i32[3]), Int32
    )
    if cutlass.const_expr(use_bf16_partial):
        return regs_vec.bitcast(BFloat16).to(Float32)
    return regs_vec.bitcast(Float16).to(Float32)
