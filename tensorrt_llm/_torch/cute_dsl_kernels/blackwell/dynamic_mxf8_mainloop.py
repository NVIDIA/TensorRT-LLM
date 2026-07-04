# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-N tcgen05 block-scaled MXF8 MMA helpers for SM100."""

from typing import Optional

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass._mlir.dialects import nvvm as _nvvm_raw
from cutlass.cute.arch.nvvm_wrappers import nvvm
from cutlass.cutlass_dsl import Boolean, Int32, Uint32, dsl_user_op

_MXF8_SCALE_VEC_SIZE = getattr(
    _nvvm_raw.Tcgen05MMAScaleVecSize,
    "BLOCK32",
    _nvvm_raw.Tcgen05MMAScaleVecSize.X1,
)


def _align16(x):
    return (Int32(x) + Int32(15)) & Int32(-16)


@dsl_user_op
def compute_non_leader_cta_load_shift(
    *,
    valid_n,
    mma_tiler_n: int,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    return (_align16(valid_n) >> Int32(1)) - Int32(mma_tiler_n // 2)


def _build_static_idesc_base(umma_m: int) -> int:
    # MXF8 E4M3 operands use format 0, UE8M0 scales use scale format 1.
    return (((umma_m >> 4) & 0x1F) << 24) | (1 << 23)


def _as_value(iterator) -> ir.Value:
    return iterator.value if hasattr(iterator, "value") else iterator


def _smem_desc_to_i64(value: ir.Value) -> ir.Value:
    return builtin.unrealized_conversion_cast([ir.IntegerType.get_signless(64)], [value])


def _tmem_ptr_to_i32(value: ir.Value) -> ir.Value:
    return builtin.unrealized_conversion_cast([ir.IntegerType.get_signless(32)], [value])


def _i32_to_tmem_ptr(value: ir.Value) -> ir.Value:
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(ptr_ty, value)


@dsl_user_op
def issue_dynamic_mxf8_mma_tile(
    *,
    acc_tensor,
    a_frag_tile,
    b_frag_tile,
    sfa_tensor,
    sfb_tensor,
    k_tile_idx,
    valid_n,
    sfa_kphase_idx=None,
    sfb_kphase_idx=None,
    mma_tiler_mnk: tuple = (256, 208, 128),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one K=128 MXF8 tile with a runtime UMMA N extent."""
    static_base = _build_static_idesc_base(mma_tiler_mnk[0])
    n_dim = _align16(valid_n) >> Int32(3)
    num_k_inner = mma_tiler_mnk[2] // 32
    group_kind = (
        _nvvm_raw.Tcgen05GroupKind
        if hasattr(_nvvm_raw, "Tcgen05GroupKind")
        else _nvvm_raw.CTAGroupKind
    )

    acc_atom = acc_tensor[(None, 0, 0)]
    acc_i32 = _tmem_ptr_to_i32(_as_value(acc_atom.iterator))
    idesc_base = Uint32(static_base) | (Uint32(n_dim) << Uint32(17))

    packed_sfa_i32 = None
    packed_sfb_i32 = None
    if sfa_kphase_idx is not None:
        packed_sfa_atom = sfa_tensor[(None, 0, sfa_kphase_idx)]
        packed_sfa_i32 = _tmem_ptr_to_i32(_as_value(packed_sfa_atom.iterator))
    if sfb_kphase_idx is not None:
        packed_sfb_atom = sfb_tensor[(None, 0, sfb_kphase_idx)]
        packed_sfb_i32 = _tmem_ptr_to_i32(_as_value(packed_sfb_atom.iterator))

    packed_idesc = idesc_base
    if packed_sfa_i32 is not None:
        packed_idesc = packed_idesc | ((Uint32(packed_sfa_i32) & Uint32(0xC0000000)) >> Uint32(1))
    if packed_sfb_i32 is not None:
        packed_idesc = packed_idesc | ((Uint32(packed_sfb_i32) & Uint32(0xC0000000)) >> Uint32(26))

    for k_inner in range(num_k_inner):
        a_atom = a_frag_tile[(None, 0, k_inner)]
        b_atom = b_frag_tile[(None, 0, k_inner)]
        sfa_idx = k_inner if sfa_kphase_idx is None else sfa_kphase_idx
        sfb_idx = k_inner if sfb_kphase_idx is None else sfb_kphase_idx

        operand_a = _smem_desc_to_i64(_as_value(a_atom.iterator))
        operand_b = _smem_desc_to_i64(_as_value(b_atom.iterator))
        sfa_i32 = packed_sfa_i32
        sfb_i32 = packed_sfb_i32
        idesc = packed_idesc
        if sfa_i32 is None:
            sfa_atom = sfa_tensor[(None, 0, sfa_idx)]
            sfa_i32 = _tmem_ptr_to_i32(_as_value(sfa_atom.iterator))
            idesc = idesc | ((Uint32(sfa_i32) & Uint32(0xC0000000)) >> Uint32(1))
        if sfb_i32 is None:
            sfb_atom = sfb_tensor[(None, 0, sfb_idx)]
            sfb_i32 = _tmem_ptr_to_i32(_as_value(sfb_atom.iterator))
            idesc = idesc | ((Uint32(sfb_i32) & Uint32(0xC0000000)) >> Uint32(26))
        accum = k_tile_idx != 0 if k_inner == 0 else True

        with cute.arch.elect_one():
            nvvm.tcgen05_mma_block_scale(
                mma_kind=_nvvm_raw.Tcgen05MMAKind.MXF8F6F4,
                cta_group=(group_kind.CTA_2 if mma_tiler_mnk[0] == 256 else group_kind.CTA_1),
                d=_i32_to_tmem_ptr(acc_i32),
                a=operand_a,
                b=operand_b,
                idesc=idesc.ir_value(),
                enable_input_d=Boolean(accum).ir_value(),
                scale_a=_i32_to_tmem_ptr(sfa_i32),
                scale_b=_i32_to_tmem_ptr(sfb_i32),
                scale_vec_size=_MXF8_SCALE_VEC_SIZE,
            )
