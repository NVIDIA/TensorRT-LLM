# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cutlass
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op

# PTX `mbarrier::peer_bit` mask used by TMA gather4 in 2CTA mode: keeps
# all address bits except bit 24 (the peer-CTA bit), so both CTAs' bytes
# flow to the leader CTA's mbar.
_PEER_BIT_MASK = 0xFEFFFFFF


@dsl_user_op
def sm100_tma_gather4_load(
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
    """Issue one TMA TILE_GATHER4 load.

    Emits inline PTX because gather4 has no DSL op. There are four PTX variants:
    {1CTA, 2CTA} x {no-mcast, mcast::cluster}.

    - 2CTA: `.cta_group::2`; mbar peer-bit-masked so both CTAs' bytes flow to
      the leader's mbar.
    - mcast::cluster: adds `.multicast::cluster` + u16 mcast_mask operand.
      All CTAs in the mcast group issue with identical params; HW coalesces
      into one GMEM load + broadcast. Each CTA's mbar receives full tx bytes.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        AddressSpace.generic,
        64,
    )
    desc_cute_ptr = _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
    desc_i64 = desc_cute_ptr.toint().ir_value(loc=loc, ip=ip)

    smem_dst_int = cutlass.Int32(smem_dst_ptr.toint())
    mbar_int = cutlass.Int32(mbar_ptr.toint())
    if use_cta_group_2:
        mbar_int = mbar_int & cutlass.Int32(_PEER_BIT_MASK)
    smem_dst_i32 = smem_dst_int.ir_value(loc=loc, ip=ip)
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

    if use_cta_group_2 and use_mcast:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster"
            ".cta_group::2"
            " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
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
            mcast_mask_i16,
        ]
        constraints = "r, l, r, r, r, r, r, r, h"
    elif use_cta_group_2:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
            " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
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
            cache_hint_i64,
        ]
        constraints = "r, l, r, r, r, r, r, r, l"
    elif use_mcast:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster"
            " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
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
            mcast_mask_i16,
        ]
        constraints = "r, l, r, r, r, r, r, r, h"
    else:
        asm = (
            "cp.async.bulk.tensor.2d.shared::cta.global"
            ".tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
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
            cache_hint_i64,
        ]
        constraints = "r, l, r, r, r, r, r, r, l"

    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def sm100_tcgen05_st_32x32b_x4(
    tmem_addr,
    r0,
    r1,
    r2,
    r3,
    *,
    loc=None,
    ip=None,
):
    """Issue one tcgen05.st.sync.aligned.32x32b.x4.b32.

    Writes 4 32-bit cells per lane to TMEM[lane_offset, col_base..col_base+3].
    Used by SFA transform warps to write LDS+repacked SF data into TMEM,
    bypassing cute.copy auto-partition.
    """
    addr_i32 = cutlass.Uint32(tmem_addr).ir_value(loc=loc, ip=ip)
    r0_i32 = cutlass.Uint32(r0).ir_value(loc=loc, ip=ip)
    r1_i32 = cutlass.Uint32(r1).ir_value(loc=loc, ip=ip)
    r2_i32 = cutlass.Uint32(r2).ir_value(loc=loc, ip=ip)
    r3_i32 = cutlass.Uint32(r3).ir_value(loc=loc, ip=ip)
    asm = "tcgen05.st.sync.aligned.32x32b.x4.b32 [$0], {$1, $2, $3, $4};"
    llvm.inline_asm(
        None,
        [addr_i32, r0_i32, r1_i32, r2_i32, r3_i32],
        asm,
        "r, r, r, r, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
