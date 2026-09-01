# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hand-encoded Rubin (sm107) block-scaled UTCOMMA for the non-swapAB gather fc1
# MoE kernel.  Emits ``tcgen05.mma...mxf4nvf4.block_scale.block16`` as literal
# PTX (via ``llvm.inline_asm``) so the instruction descriptor (``idesc``) is
# fully under our control -- specifically so bit 26 (SFA layout: 0 =
# SFA_32dp_4xCopy, 1 = SFA_128dp_Unique) can be set.
#
# Mechanism copied from megaMOE swapAB ``dynamic_mainloop.py`` (the user's
# confirmed Rubin-correct reference).  Differences here (non-swapAB fc1):
#   * A = tokens (M), B = weights (N).  n_dim is STATIC (= mma_inst_n >> 3),
#     folded into the static idesc base -- no dynamic-token n_dim.
#   * bit 26 (sfa_layout) is exposed as a build-time knob.
#   * Per-atom entry (kernel already loops k-blocks externally), single MMA
#     issued per call.
#
# idesc bit layout (SM107 OMMA table, Sm107Mma.h):
#   bit [ 3]     k_size upper
#   bit [ 4, 6)  b_sf_id   (runtime, OR'd from SFB tmem addr top bits)
#   bit [ 7,10)  a_format  (1 = NVFP4 mxf4nvf4)
#   bit [10,12)  b_format  (1 = NVFP4)
#   bit [15]     a_major   (0 = K-major)
#   bit [16]     b_major   (0 = K-major)
#   bit [17,23)  n_dim     (N >> 3, static for non-swapAB)
#   bit [23,26)  scale_format (0 = UE4M3 SF)
#   bit [26]     sfa_layout (0 = 32dp_4xCopy, 1 = 128dp_Unique)  <-- the knob
#   bit [27,29)  m_dim     (M >> 4 placed at bit 24 == M >> 7 placed at bit 27)
#   bit [29,31)  a_sf_id   (runtime, OR'd from SFA tmem addr top bits)
#   bit [31]     k_size lower

from typing import Optional

import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op, Int32, Boolean
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin
from cutlass._mlir.dialects import llvm


_BIT_A_FORMAT = 7  # width 3
_BIT_B_FORMAT = 10  # width 2
_BIT_A_MAJOR = 15  # width 1
_BIT_B_MAJOR = 16  # width 1
_BIT_N_DIM = 17  # width 6
_BIT_SCALE_FORMAT = 23  # width 3
_BIT_SFA_LAYOUT = 26  # width 1  <-- 0 = 32dp_4xCopy, 1 = 128dp_Unique
_BIT_M_DIM = 24  # m_dim = umma_m >> 4 placed here (== umma_m >> 7 at bit 27)
_BIT_A_SF_ID = 29  # width 2
_BIT_B_SF_ID = 4  # width 2
_BIT_K_SIZE_LO = 31  # K-size lower bit
_BIT_K_SIZE_HI = 3  # K-size upper bit (Rubin repurposes this reserved bit)

_UMMA_K_NVFP4 = 128
_K_SIZE_FIELD = {64: 0, 96: 1, 128: 2}


def build_static_idesc_base(
    *,
    umma_m: int,
    umma_n: int,
    a_format: int = 1,
    b_format: int = 1,
    a_major: int = 0,
    b_major: int = 0,
    scale_format: int = 0,
    umma_k: int = _UMMA_K_NVFP4,
    sfa_layout: int = 0,
) -> int:
    """Pack the static-field portion of the idesc into a u32.

    Runtime fields (a_sf_id_, b_sf_id_) are left at zero; ``compute_idesc``
    OR's them in from the SF TMEM addresses.  n_dim is static (non-swapAB) and
    folded in here.
    """
    assert umma_m in (64, 128, 256), f"Unsupported UMMA_M={umma_m}"
    assert umma_k in _K_SIZE_FIELD, f"Unsupported UMMA_K={umma_k}"
    assert 0 <= sfa_layout < 2

    m_dim = umma_m >> 4  # placed at bit 24; == (umma_m >> 7) << 27
    n_dim = umma_n >> 3

    desc = 0
    desc |= (a_format & 0x7) << _BIT_A_FORMAT
    desc |= (b_format & 0x3) << _BIT_B_FORMAT
    desc |= (a_major & 0x1) << _BIT_A_MAJOR
    desc |= (b_major & 0x1) << _BIT_B_MAJOR
    desc |= (n_dim & 0x3F) << _BIT_N_DIM
    desc |= (scale_format & 0x7) << _BIT_SCALE_FORMAT
    desc |= (sfa_layout & 0x1) << _BIT_SFA_LAYOUT
    desc |= (m_dim & 0x1F) << _BIT_M_DIM
    k_size_value = _K_SIZE_FIELD[umma_k]
    desc |= (k_size_value & 0x1) << _BIT_K_SIZE_LO
    desc |= ((k_size_value >> 1) & 0x1) << _BIT_K_SIZE_HI
    return desc & 0xFFFFFFFF


@dsl_user_op
def compute_idesc(
    *,
    static_base: int,
    sfa_tmem_addr_i32,
    sfb_tmem_addr_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """OR runtime a_sf_id_ / b_sf_id_ (SF TMEM addr top 2 bits) into the base."""
    idesc = Int32(static_base)
    sfa_top = Int32(sfa_tmem_addr_i32) & Int32(0xC0000000)
    sfb_top = Int32(sfb_tmem_addr_i32) & Int32(0xC0000000)
    idesc = idesc | (sfa_top >> Int32(30 - _BIT_A_SF_ID))
    idesc = idesc | (sfb_top >> Int32(30 - _BIT_B_SF_ID))
    return idesc


def _smem_desc_to_i64(smem_desc_value: ir.Value) -> ir.Value:
    i64_ty = ir.IntegerType.get_signless(64)
    return builtin.unrealized_conversion_cast([i64_ty], [smem_desc_value])


def _tmem_ptr_to_i32(tmem_ptr_value: ir.Value) -> ir.Value:
    i32_ty = ir.IntegerType.get_signless(32)
    return builtin.unrealized_conversion_cast([i32_ty], [tmem_ptr_value])


def _as_value(it) -> ir.Value:
    return it.value if hasattr(it, "value") else it


@dsl_user_op
def _tcgen05_mma_mxf4nvf4_block_scale_block16(
    *,
    cta_group: int,
    d_tmem_i32,
    a_desc_i64,
    b_desc_i64,
    idesc_i32,
    enable_input_d_i32,
    sfa_tmem_i32,
    sfb_tmem_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Emit ``tcgen05.mma.cta_group::{1,2}.kind::mxf4nvf4.block_scale.block16``."""
    assert cta_group in (1, 2), f"cta_group must be 1 or 2, got {cta_group}"
    llvm.inline_asm(
        None,
        [
            d_tmem_i32,
            a_desc_i64,
            b_desc_i64,
            idesc_i32,
            enable_input_d_i32,
            sfa_tmem_i32,
            sfb_tmem_i32,
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $4, 0;\n\t"
        f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4.block_scale.block16"
        ".collector::a::discard "
        "[$0], $1, $2, $3, [$5], [$6], p;\n\t"
        "}\n",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_manual_block_scaled_mma_atom(
    *,
    acc_frag,  # tCtAcc  (V, M_count, N_count)   -- TMEM f32 accumulator
    a_frag,  # tCrA[(None,None,kblk,stage)]   (V, M_count) -- smem desc
    sfa_frag,  # tCtSFA[(None,None,kblk)]        (V, MN_count) -- TMEM sf
    b_frag,  # tCrB[(None,None,kblk,stage)]   (V, N_count) -- smem desc
    sfb_frag,  # tCtSFB[(None,None,kblk)]        (V, MN_count) -- TMEM sf
    static_idesc_base: int,
    accumulate,  # Boolean SSA: D = A@B (+ C) when True
    cta_group: int = 1,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ONE block-scaled UTCOMMA for the current (k-block, stage) atom.

    Mirrors megaMOE ``dynamic_mainloop`` per-atom extraction: slice each
    operand down to its atom (V,), cast to the PTX register types, OR the
    runtime SF-id bits into the static idesc base, then emit the asm.
    """
    a_atom = a_frag[(None, 0)]
    b_atom = b_frag[(None, 0)]
    sfa_atom = sfa_frag[(None, 0)]
    sfb_atom = sfb_frag[(None, 0)]
    acc_atom = acc_frag[(None, 0, 0)]

    a_iter_val = _as_value(a_atom.iterator)
    b_iter_val = _as_value(b_atom.iterator)
    acc_iter_val = _as_value(acc_atom.iterator)
    sfa_iter_val = _as_value(sfa_atom.iterator)
    sfb_iter_val = _as_value(sfb_atom.iterator)

    operand_a = _smem_desc_to_i64(a_iter_val)
    operand_b = _smem_desc_to_i64(b_iter_val)
    operand_sfa_i32 = _tmem_ptr_to_i32(sfa_iter_val)
    operand_sfb_i32 = _tmem_ptr_to_i32(sfb_iter_val)
    operand_acc_i32 = _tmem_ptr_to_i32(acc_iter_val)

    idesc = compute_idesc(
        static_base=static_idesc_base,
        sfa_tmem_addr_i32=operand_sfa_i32,
        sfb_tmem_addr_i32=operand_sfb_i32,
    )

    with cute.arch.elect_one():
        _tcgen05_mma_mxf4nvf4_block_scale_block16(
            cta_group=cta_group,
            d_tmem_i32=operand_acc_i32,
            a_desc_i64=operand_a,
            b_desc_i64=operand_b,
            idesc_i32=idesc.ir_value(),
            enable_input_d_i32=Int32(Boolean(accumulate)).ir_value(),
            sfa_tmem_i32=operand_sfa_i32,
            sfb_tmem_i32=operand_sfb_i32,
        )
