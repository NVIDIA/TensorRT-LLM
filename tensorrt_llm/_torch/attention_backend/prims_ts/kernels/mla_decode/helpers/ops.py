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

"""Reusable low-level operations for MLA decode TS kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as cprims

from .constants import (
    OCTET_LANES,
    SMEM_ROW_BYTES,
    SMEM_VECTOR_BYTES,
    SMEM_WORD_BYTE_SHIFT,
    STSM_MATRIX_LANE_SHIFT,
    STSM_X4_REG_COUNT,
    SOFTMAX_SCRATCH_SUM_WORD_OFFSET,
    TCGEN05_16X32BX2_BF16_P_STRIDE,
    TCGEN05_16X32BX2_FP8_P_STRIDE,
    TCGEN05_SECOND_PANEL_ADDR_OFFSET,
    WARP_LANES,
    WARP_LANE_MASK,
    WARP_LANE_SHIFT,
    WARP_REDUCTION_BFLY_DISTANCES,
)


primitives_inline_ptx = cprims.inline_ptx
"""Public primitive inline-PTX entry point used by MLA helper ops."""

inline_ptx = cute.arch.inline_ptx
"""CuTe inline PTX entry point used by MLA helper ops."""


@cute.jit
def freeze_smem_descriptor(desc):
    """Copy an SMEM descriptor through a register to prevent rematerialization."""
    return inline_ptx(
        "mov.b64 {$w0}, {$r0};",
        write_only_types=[Int64],
        read_only_args=[desc],
    )


@cute.jit
def warp_idx_from_warpgroup_thread(warp_grp_thread_idx):
    """Return the warp index inside a warpgroup thread-id range."""
    return warp_grp_thread_idx >> Int32(WARP_LANE_SHIFT)


@cute.jit
def lane_idx_from_thread(thread_idx):
    """Return the lane index inside one CUDA warp."""
    return thread_idx & Int32(WARP_LANE_MASK)


@cute.jit
def tcgen05_second_panel_addr(base_addr):
    """Return the second TMEM panel address for split S/P/O panels."""
    # tcgen05 TMEM addresses encode the row in the high 16 bits.  Split-panel
    # layouts keep the column fixed and advance only that row field.
    return base_addr + Int32(TCGEN05_SECOND_PANEL_ADDR_OFFSET)


@cute.jit
def tcgen05_panel_addr(base_addr, panel_idx):
    """Return a TMEM address offset by ``panel_idx`` split panels."""
    # Panel index arithmetic uses the same high-16-bit row advance as the
    # second-panel helper so call sites do not open-code TMEM address packing.
    return base_addr + Int32(panel_idx * TCGEN05_SECOND_PANEL_ADDR_OFFSET)


@cute.jit
def softmax_sum_state_ptr(state_ptr):
    """Return the softmax-sum scratch panel paired with a max scratch pointer."""
    # The online-softmax scratch allocation stores max and sum panels in one
    # Uint32 array.  The sum pointer is offset in scratch words, not bytes.
    return state_ptr + Int32(SOFTMAX_SCRATCH_SUM_WORD_OFFSET)


@cutlass.dsl_user_op
def vector_from_scalars(values, dtype, *, loc=None, ip=None):
    """Pack scalar register values into a DSL vector."""
    return cutlass.Vector.from_elements(
        tuple(dtype(value) for value in values),
        dtype,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    """Return the maximum of two values as Float32."""
    return Float32(
        cute.math.max(Float32(a), Float32(b), ftz=True, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def warp_reduce_max_f32(val, *, loc=None, ip=None):
    """Reduce a Float32 value to the warp maximum with butterfly shuffles."""
    val = Float32(val)
    for dist in WARP_REDUCTION_BFLY_DISTANCES:
        val = fmax_f32(
            val,
            Float32(
                cprims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=val,
                    offset=dist,
                    mask_and_clamp=0x1F,
                    kind=cprims.Shfl.BFLY,
                    loc=loc,
                    ip=ip,
                )
            ),
            loc=loc,
            ip=ip,
        )
    return val


@cutlass.dsl_user_op
def warp_reduce_sum_f32(val, *, loc=None, ip=None):
    """Reduce a Float32 value to the warp sum with butterfly shuffles."""
    val = Float32(val)
    for dist in WARP_REDUCTION_BFLY_DISTANCES:
        val = Float32(
            val
            + cprims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=val,
                offset=dist,
                mask_and_clamp=0x1F,
                kind=cprims.Shfl.BFLY,
                loc=loc,
                ip=ip,
            )
        )
    return val


@cutlass.dsl_user_op
def tcgen05_ld_16x32bx2_f32(
    tmem_addr,
    *,
    num: cutlass.Constexpr[int],
    offset,
    loc=None,
    ip=None,
):
    """Load Float32 registers from a split 16x32bx2 TMEM tile."""
    result = cprims.tcgen05_ld(
        cprims.Tcgen05LdStShape.SHAPE_16X32BX2,
        tmem_addr,
        num=num,
        offset=Int64(offset),
        loc=loc,
        ip=ip,
    )
    if num == 1:
        return result[0]
    return result


@cutlass.dsl_user_op
def tcgen05_st_16x32bx2_f32(
    tmem_addr,
    value,
    *,
    offset,
    loc=None,
    ip=None,
):
    """Store Float32 registers to a split 16x32bx2 TMEM tile."""
    cprims.tcgen05_st(
        cprims.Tcgen05LdStShape.SHAPE_16X32BX2,
        tmem_addr,
        value,
        offset=Int64(offset),
        loc=loc,
        ip=ip,
    )


@cute.jit
def pack_float4_to_fp8_e4m3(v0: Float32, v1: Float32, v2: Float32, v3: Float32):
    """Pack four Float32 values into one E4M3 x4 register."""
    # Spell the canonical pair conversions through CuTe's public inline-PTX
    # operation so the generated f8x2 instruction receives only its supported
    # operands.
    return inline_ptx(
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "  mov.b32 {$w0}, {lo, hi};\n"
        "}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def fp8_log2_quant_scale():
    """Return log2(448) for the E4M3 P scaling convention."""
    return Float32(8.8073549)


@cute.jit
def fp8_quant_scale_rcp():
    """Return the reciprocal of the shared E4M3 probability scale."""
    return Float32(1.0 / 448.0)


@cute.jit
def fp8_stsm_smem_dst(
    smem_base_i32,
    warp_grp_thread_idx,
    num_trans_rows,
    num_trans_cols,
    stsm_idx: cutlass.Constexpr[int],
):
    """Return byte-transposed STSM destination for FP8 P/O staging."""

    num_rows = Int32(num_trans_rows)
    num_bytes_per_row = Int32(num_trans_cols)
    num_rows_per_smem_row = Int32(SMEM_ROW_BYTES) // num_bytes_per_row
    num_segs_per_warp_per_row = num_bytes_per_row // Int32(
        SMEM_VECTOR_BYTES * STSM_X4_REG_COUNT
    )
    # One STSM group covers one row for the current 8-row MMA fragment.
    num_stsm_per_row = 1
    num_mtx_per_col = num_rows // Int32(OCTET_LANES)
    warp_idx = warp_idx_from_warpgroup_thread(warp_grp_thread_idx)
    lane_idx = lane_idx_from_thread(warp_grp_thread_idx)
    thr_row_idx = lane_idx & Int32(OCTET_LANES - 1)
    mtx_idx = lane_idx >> Int32(STSM_MATRIX_LANE_SHIFT)
    mtx_row_idx = mtx_idx % num_mtx_per_col
    mtx_col_idx = mtx_idx // num_mtx_per_col

    # STSM writes one 8-row matrix per lane group.  The destination is computed
    # in bytes so the same helper works for the x1/x2/x4 instruction variants;
    # the final pointer conversion switches back to Uint32 word addressing.
    stsm_row_idx = Int32(stsm_idx % num_stsm_per_row)
    stsm_col_idx = Int32(stsm_idx // num_stsm_per_row)
    xor_mask = thr_row_idx // num_rows_per_smem_row
    seg_col_idx = (
        warp_idx * num_segs_per_warp_per_row + mtx_col_idx + stsm_col_idx
    ) ^ xor_mask
    smem_offset = (
        mtx_row_idx * Int32(OCTET_LANES)
        + thr_row_idx
        + stsm_row_idx * Int32(WARP_LANES)
    ) * num_bytes_per_row + seg_col_idx * Int32(SMEM_VECTOR_BYTES)
    return smem_base_i32.data_ptr(smem_offset >> Int32(SMEM_WORD_BYTE_SHIFT))


@cute.jit
def store_transposed_smem8b_x1(
    smem_base_i32,
    reg0: Int32,
    warp_grp_thread_idx,
    num_trans_rows,
    num_trans_cols,
    stsm_idx: cutlass.Constexpr[int] = 0,
):
    """Store one FP8 STSM register into transposed SMEM layout."""
    smem_dst = fp8_stsm_smem_dst(
        smem_base_i32, warp_grp_thread_idx, num_trans_rows, num_trans_cols, stsm_idx
    )
    primitives_inline_ptx(
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [{$r0}], {{$r1}};",
        read_only_args=[smem_dst, reg0],
    )


@cute.jit
def store_transposed_smem8b_x2(
    smem_base_i32,
    reg0: Int32,
    reg1: Int32,
    warp_grp_thread_idx,
    num_trans_rows,
    num_trans_cols,
    stsm_idx: cutlass.Constexpr[int] = 0,
):
    """Store two FP8 STSM registers into transposed SMEM layout."""
    smem_dst = fp8_stsm_smem_dst(
        smem_base_i32, warp_grp_thread_idx, num_trans_rows, num_trans_cols, stsm_idx
    )
    primitives_inline_ptx(
        "stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [{$r0}], {{$r1}, {$r2}};",
        read_only_args=[smem_dst, reg0, reg1],
    )


@cute.jit
def store_transposed_smem8b_x4(
    smem_base_i32,
    reg0: Int32,
    reg1: Int32,
    reg2: Int32,
    reg3: Int32,
    warp_grp_thread_idx,
    num_trans_rows,
    num_trans_cols,
    stsm_idx: cutlass.Constexpr[int] = 0,
):
    """Store four FP8 STSM registers into transposed SMEM layout."""
    smem_dst = fp8_stsm_smem_dst(
        smem_base_i32, warp_grp_thread_idx, num_trans_rows, num_trans_cols, stsm_idx
    )
    primitives_inline_ptx(
        "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[smem_dst, reg0, reg1, reg2, reg3],
    )


@cute.jit
def tcgen05_store_p_16x32bx2_x16(tmem_addr, regs_p, start_idx: cutlass.Constexpr[int]):
    """Store 16 packed BF16 P registers to a split TMEM-P tile."""
    # The stride immediate is the BF16 lane-to-column spacing for the split
    # 16x32bx2 P layout; keep it named because the FP8 path uses a different
    # immediate with the same instruction shape.
    inline_ptx(
        "tcgen05.st.sync.aligned.16x32bx2.x16.b32 "
        f"[{{$r0}}], {TCGEN05_16X32BX2_BF16_P_STRIDE}, "
        "{ {$r1}, {$r2}, {$r3}, {$r4}, {$r5}, {$r6}, {$r7}, {$r8}, "
        "{$r9}, {$r10}, {$r11}, {$r12}, {$r13}, {$r14}, {$r15}, {$r16} };",
        read_only_args=[
            tmem_addr,
            regs_p[start_idx + 0],
            regs_p[start_idx + 1],
            regs_p[start_idx + 2],
            regs_p[start_idx + 3],
            regs_p[start_idx + 4],
            regs_p[start_idx + 5],
            regs_p[start_idx + 6],
            regs_p[start_idx + 7],
            regs_p[start_idx + 8],
            regs_p[start_idx + 9],
            regs_p[start_idx + 10],
            regs_p[start_idx + 11],
            regs_p[start_idx + 12],
            regs_p[start_idx + 13],
            regs_p[start_idx + 14],
            regs_p[start_idx + 15],
        ],
    )


@cute.jit
def tcgen05_store_p_fp8_16x32bx2_x16(tmem_addr, regs_p):
    """Store 16 packed E4M3 P registers to a split TMEM-P tile."""
    # FP8 P uses twice as many elements per byte vector as BF16, so the TMEM
    # column stride immediate is smaller even though the register count matches.
    inline_ptx(
        "tcgen05.st.sync.aligned.16x32bx2.x16.b32 "
        f"[{{$r0}}], {TCGEN05_16X32BX2_FP8_P_STRIDE}, "
        "{ {$r1}, {$r2}, {$r3}, {$r4}, {$r5}, {$r6}, {$r7}, {$r8}, "
        "{$r9}, {$r10}, {$r11}, {$r12}, {$r13}, {$r14}, {$r15}, {$r16} };",
        read_only_args=[
            tmem_addr,
            regs_p[0],
            regs_p[1],
            regs_p[2],
            regs_p[3],
            regs_p[4],
            regs_p[5],
            regs_p[6],
            regs_p[7],
            regs_p[8],
            regs_p[9],
            regs_p[10],
            regs_p[11],
            regs_p[12],
            regs_p[13],
            regs_p[14],
            regs_p[15],
        ],
    )


@cute.jit
def float_to_u32_bits(val):
    """Return the raw Uint32 bit pattern for a Float32 value."""
    return cprims.mov_b32(val, target_type=Uint32)


@cute.jit
def u32_bits_to_float(val: Uint32):
    """Return the Float32 value represented by raw Uint32 bits."""
    return cprims.mov_b32(val, target_type=Float32)
