# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software grid-sync primitive for cuTeDSL kernels.

Replicates the mega_moe `grid_sync` from
`DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh` since cuTeDSL has no
cooperative-launch entrypoint. Uses the canonical phase-flip pattern: a single
u32 slot whose bit 31 (kFinishSumTag = 0x80000000) acts as the round-phase, and
whose low bits accumulate per-CTA arrivals. The first CTA (sm_idx == 0) adds
`(kFinishSumTag - (num_sms - 1))`, every other CTA adds 1, so after all CTAs
arrive the slot equals `kFinishSumTag` (bit 31 toggled vs. its pre-round value),
which both signals completion and primes the slot for the next round.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, dsl_user_op

_FINISH_SUM_TAG = 0x80000000


@dsl_user_op
@cute.jit
def software_grid_sync(counter_ptr,
                       sm_idx,
                       num_sms,
                       tid_in_group,
                       *,
                       num_threads=None,
                       loc=None,
                       ip=None):
    """Grid-wide barrier replacing cooperative_groups::this_grid().sync().

    counter_ptr  : Pointer to a single u32 in global memory, zero-initialised
                   before the first call. Every barrier round leaves it equal
                   to one of {0x80000000, 0} (alternating), so it is reusable
                   across rounds without an explicit reset.
    sm_idx       : The block index this CTA was launched as (typically
                   ``cute.arch.block_idx()[0]``). Used to pick the special
                   delta for sm 0.
    num_sms      : Total number of CTAs participating. Must equal grid_dim.
    tid_in_group : Logical thread index within the calling participant group,
                   0-indexed.  The thread with ``tid_in_group == 0`` is the
                   *leader* that issues the atomic add and spin-wait; all
                   others just wait on the surrounding NamedBarrier.

                   This MUST be the logical (group-relative) tid, NOT the
                   hardware ``%tid.x``.  In fused kernels the dispatch warps
                   may start at a non-zero ``%tid.x`` base (e.g. MegaMoE puts
                   them at warps 8-11 / tid_x [256, 384)).  Reading ``%tid.x``
                   directly inside the PTX would then find no leader and
                   silently degenerate the grid_sync into a CTA-local NB10
                   sandwich -- the entire grid-wide sync becomes a no-op.
                   Caller is responsible for computing this; e.g.
                   ``tid_in_group = local_warp_idx * 32 + lane_idx`` where
                   ``local_warp_idx`` is the warp index *within* the dispatch
                   group (0..3 for the 4 dispatch warps).
    num_threads  : Number of threads participating on the wrapping
                   NamedBarrier.  Default ``None`` -> ``cute.arch.sync_threads()``
                   (= bar.sync 0, full CTA). Pass an explicit count (e.g. 128
                   = number of dispatch threads) when only a subset of the
                   CTA's warps participate in this grid sync; needed when the
                   kernel also launches non-participating warps (e.g. the
                   placeholder epilogue group), because ``bar.sync 0`` without
                   a count waits for every thread in the CTA and would
                   deadlock against those non-participating warps sitting on
                   a different ``bar.sync`` id.
    """
    # NamedBarrier ID 10 (NOT 0!): ID 0 is implicitly used by cuTeDSL's
    # ``cute.arch.sync_threads()`` and may also be touched by various
    # pipeline / mbarrier primitives from concurrent warp groups.  When the
    # caller uses ``num_threads != None`` (= a subset of the CTA), reusing
    # ID 0 can race with other warps' implicit ``bar.sync 0`` and release
    # this grid sync prematurely (e.g. SM-X exits before the SM-0 publish
    # it is supposed to wait for, causing stale GMEM reads downstream).
    # Must match TokenInPullTokenBackPush.dispatch_intra_cta_bar_id.
    if num_threads is None:
        cute.arch.sync_threads(loc=loc, ip=ip)
    else:
        # TODO: Remove this hardcode.
        cute.arch.barrier(barrier_id=10,
                          number_of_threads=num_threads,
                          loc=loc,
                          ip=ip)

    # PTX add.u32 treats the operand as a raw 32-bit bit pattern so signed
    # underflow of (kFinishSumTag - (num_sms - 1)) is benign and matches mega_moe.
    if cutlass.const_expr(isinstance(num_sms, int)):
        sm_zero_bits = (_FINISH_SUM_TAG - (num_sms - 1)) & 0xFFFFFFFF
        if cutlass.const_expr(sm_zero_bits >= 0x80000000):
            sm_zero_bits -= 0x100000000
        sm_zero_delta = Int32(sm_zero_bits)
    else:
        sm_zero_delta = Int32(-_FINISH_SUM_TAG) - (Int32(num_sms) - Int32(1))
    other_delta = Int32(1)

    # Accept either a Pointer (which has `.toint()`) or a cute.Tensor (which
    # exposes the underlying pointer via `.iterator`).
    if cutlass.const_expr(hasattr(counter_ptr, "iterator")):
        counter_ptr = counter_ptr.iterator
    # $0=counter_ptr, $1=sm_idx, $2=sm_zero_delta, $3=other_delta,
    # $4=tid_in_group (leader predicate source).
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(sm_idx).ir_value(),
            sm_zero_delta.ir_value(),
            other_delta.ir_value(),
            Int32(tid_in_group).ir_value(),
        ],
        ("{\n\t"
         ".reg .b32 %delta; .reg .b32 %old; .reg .b32 %cur;\n\t"
         ".reg .pred %not_leader; .reg .pred %is_sm0; .reg .pred %waiting;\n\t"
         "setp.ne.u32 %not_leader, $4, 0;\n\t"
         "@%not_leader bra DONE;\n\t"
         "setp.eq.u32 %is_sm0, $1, 0;\n\t"
         "selp.b32 %delta, $2, $3, %is_sm0;\n\t"
         "atom.release.gpu.global.add.u32 %old, [$0], %delta;\n\t"
         "SPIN:\n\t"
         "ld.acquire.gpu.global.b32 %cur, [$0];\n\t"
         "xor.b32 %cur, %cur, %old;\n\t"
         "and.b32 %cur, %cur, 0x80000000;\n\t"
         "setp.eq.u32 %waiting, %cur, 0;\n\t"
         "@%waiting bra SPIN;\n\t"
         "DONE:\n\t"
         "}"),
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )

    if cutlass.const_expr(num_threads is None):
        cute.arch.sync_threads(loc=loc, ip=ip)
    else:
        cute.arch.barrier(barrier_id=10,
                          number_of_threads=num_threads,
                          loc=loc,
                          ip=ip)
