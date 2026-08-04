# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX wrappers (TMA 1D load/store, fns.b32, raw-int64 peer ops) for the cuTeDSL dispatch kernel."""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

# CUTLASS ``cute::TMA::CacheHintSm100`` policy descriptors
# (`copy_sm90_desc.hpp:193-196`). At large per-launch transfer sizes
# (e.g. 1+ GB/launch for T=32k batches) the L2 (~128 MB on B200/GB200) is
# heavily over-subscribed, so cache hints matter. mega_moe defaults TMA
# loads to EVICT_FIRST (single-use peer data) and TMA stores to
# EVICT_NORMAL (just-pulled data will be re-read shortly by the GEMM
# consumer). cuTeDSL was passing hint=0 (undefined policy) on loads and
# omitting the cache_hint operand entirely on stores -- both lead to L2
# pollution at large batch.
_TMA_CACHE_HINT_EVICT_NORMAL = 0x1000000000000000
_TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def tma_load_1d(dst_smem, src_gmem, mbar_smem, num_bytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_1d_raw(dst_smem,
                    src_gmem_addr: Int64,
                    mbar_smem,
                    num_bytes,
                    *,
                    loc=None,
                    ip=None):
    """Variant of ``tma_load_1d`` that takes a raw int64 GMEM byte address.

    Used for cross-rank TMA load via ``peer_rank_ptr_mapper.map`` style: source
    address is computed dynamically as ``peer_rank_ptr_mapper.map(local_iter.toint(),
    dst_rank, element_offset)``, bypassing per-tensor constexpr fanout.
    """
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem_addr.ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_1d(dst_gmem, src_smem, num_bytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_NORMAL).ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
        "[$0], [$1], $2, $3;",
        "l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fns_b32(mask: Int32, base: Int32, n: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [mask.ir_value(), base.ir_value(),
             n.ir_value()],
            "fns.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        ))


# -----------------------------------------------------------------------------
# Raw-int64-pointer GMEM ops for the cross-rank ``peer_rank_ptr_mapper.map`` pattern.
# -----------------------------------------------------------------------------
# Each helper takes a 64-bit byte address (local iterator's ``.toint()`` +
# peer-rank offset + element offset) and emits the corresponding PTX op
# without requiring a cute.Tensor / iterator wrap. Mirrors mega_moe's pattern
# of computing ``peer_rank_ptr_mapper.map(local_ptr, dst_rank)`` and dereferencing it
# directly (sym_buffer.cuh:34-37).


@dsl_user_op
def ldg_b32_raw(addr: Int64, *, loc=None, ip=None) -> Int32:
    """``ld.global.u32`` via raw int64 byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value()],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def ldg_f32_raw(addr: Int64, *, loc=None, ip=None) -> Float32:
    """``ld.global.f32`` via raw int64 byte address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [addr.ir_value()],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def stg_b32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """``st.global.u32`` via raw int64 byte address."""
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """``st.global.u64`` via raw int64 byte address."""
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_u64_raw(addr: Int64,
                                val: Int64,
                                *,
                                loc=None,
                                ip=None) -> None:
    """``red.release.sys.global.add.u64`` via raw int64 byte address.

    Fire-and-forget atomic add (no return value) -- mega_moe uses this
    pattern for ``expert_recv_count_sum`` cross-rank publish where the
    fetched-old is unused (sm100_fp8_fp4_mega_moe.cuh:511-513).
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.release.sys.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_s32_raw(addr: Int64,
                                val: Int32,
                                *,
                                loc=None,
                                ip=None) -> None:
    """``red.release.sys.global.add.s32`` via raw int64 byte address.

    Used for the NVLink barrier signal cross-rank fan-out (matches
    mega_moe ``ptx::red_add_rel_sys`` at barrier.cuh:50).
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_add_release_sys_u32_raw(addr: Int64,
                                      val: Int32,
                                      *,
                                      loc=None,
                                      ip=None) -> None:
    """``red.async.release.sys.global.add.u32`` via raw int64 byte address.

    sm_90+ async reduction — fire-and-forget; the issuing SM does NOT
    wait for the L2/HBM round-trip. Same architectural release/sys
    semantics as the synchronous form, but the SM can continue issuing
    instructions immediately. Used in cuTeDSL dispatch_pull V2 to
    replace the synchronous ``atom.release.gpu.global.add.u32`` for
    l1_arrival_count, eliminating the per-token atomic-wait stall.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.async.release.sys.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def read_clock64(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        ))
