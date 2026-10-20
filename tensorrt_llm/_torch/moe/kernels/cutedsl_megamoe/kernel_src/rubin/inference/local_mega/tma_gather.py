# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin TMA Gather4 load primitives used by Local MegaMoE."""

import cutlass
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op

_CTA_GROUP_2_MBAR_MASK = 0xFEFFFFFF


@dsl_user_op
def sm107_tma_gather4_load(
    tma_atom,
    smem_dst_ptr,
    mbar_ptr,
    col,
    r0,
    r1,
    r2,
    r3,
    *,
    use_cta_group_2: bool = False,
    mcast_mask=None,
    loc=None,
    ip=None,
):
    """Issue one Rubin TMA TILE_GATHER4 load.

    Selects one of four PTX forms:
      {1CTA, 2CTA} x {no-mcast, multicast::cluster}.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        AddressSpace.generic,
        64,
    )
    desc_cute_ptr = _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
    desc_i64 = desc_cute_ptr.toint().ir_value(loc=loc, ip=ip)

    smem_dst_i32 = cutlass.Int32(smem_dst_ptr.toint()).ir_value(loc=loc, ip=ip)
    mbar_int = cutlass.Int32(mbar_ptr.toint())
    if use_cta_group_2:
        mbar_int = mbar_int & cutlass.Int32(_CTA_GROUP_2_MBAR_MASK)
    mbar_i32 = mbar_int.ir_value(loc=loc, ip=ip)
    col_i32 = cutlass.Int32(col).ir_value(loc=loc, ip=ip)
    r0_i32 = cutlass.Int32(r0).ir_value(loc=loc, ip=ip)
    r1_i32 = cutlass.Int32(r1).ir_value(loc=loc, ip=ip)
    r2_i32 = cutlass.Int32(r2).ir_value(loc=loc, ip=ip)
    r3_i32 = cutlass.Int32(r3).ir_value(loc=loc, ip=ip)
    cache_hint_i64 = cutlass.Int64(0).ir_value(loc=loc, ip=ip)

    use_mcast = mcast_mask is not None
    if use_mcast:
        mcast_mask_i16 = cutlass.Int16(mcast_mask).ir_value(loc=loc, ip=ip)

    scope = "cluster" if use_cta_group_2 or use_mcast else "cta"
    multicast_qualifier = ".multicast::cluster" if use_mcast else ""
    cta_group_qualifier = ".cta_group::2" if use_cta_group_2 else ""
    asm = (
        f"cp.async.bulk.tensor.2d.shared::{scope}.global"
        f".tile::gather4.mbarrier::complete_tx::bytes{multicast_qualifier}"
        f".L2::cache_hint{cta_group_qualifier}"
        " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8"
    )
    operands = [
        smem_dst_i32,
        desc_i64,
        mbar_i32,
        col_i32,
        r0_i32,
        r1_i32,
        r2_i32,
        r3_i32,
    ]
    constraints = "r, l, r, r, r, r, r, r"
    if use_mcast:
        asm += ", $9;"
        operands.extend((mcast_mask_i16, cache_hint_i64))
        constraints += ", h, l"
    else:
        asm += ";"
        operands.append(cache_hint_i64)
        constraints += ", l"
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = ["sm107_tma_gather4_load"]
