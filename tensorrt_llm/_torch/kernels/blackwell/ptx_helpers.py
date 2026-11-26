# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device helpers shared by the Blackwell CuTe DSL kernels (the decode MoE engines, the low-M MoE FC2, the MoE
router GEMM): inline PTX and small tracing helpers.

The helpers stay inline PTX where ``cute.arch`` has an equivalent operation (``barrier``, ``exp2``, ``rcp_approx``,
``cp_async_commit_group``, ``load`` / ``store`` / ``atomic_add`` on pointers): those lower to other MLIR / LLVM
operations and change the engines' generated PTX, while ``nvvm.inline_ptx`` is a side-effecting ``asm`` the compiler
neither merges nor moves across another, which the engines' cross-CTA protocols (counter polls, release / acquire
pairs) and their bit-exact epilogues are written against.
"""

from __future__ import annotations

from typing import Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import tcgen05

from .moe_decode_front import _add

# A tree of value indices: an index, or a pair of trees summed left + right.
SumTree = Union[int, Tuple["SumTree", "SumTree"]]


# ----------------------------------------------------------------------------------------------------------------
# Global memory
# ----------------------------------------------------------------------------------------------------------------
def _ld_s32(addr: Int64) -> Int32:
    return cute.arch.inline_ptx(
        "ld.global.s32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[addr]
    )


def _ld_relaxed_u32(addr: Int64) -> Int32:
    return cute.arch.inline_ptx(
        "ld.relaxed.gpu.global.u32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[addr]
    )


def _ld_acquire_u32(addr: Int64) -> Int32:
    return cute.arch.inline_ptx(
        "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];", write_only_types=[Int32], read_only_args=[addr]
    )


def _ld_nc_f32(addr: Int64) -> cutlass.Float32:
    return cute.arch.inline_ptx(
        "ld.global.nc.f32 {$w0}, [{$r0}];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )


def _st_relaxed_u32(addr: Int64, v: Int32) -> None:
    cute.arch.inline_ptx("st.relaxed.gpu.global.u32 [{$r0}], {$r1};", read_only_args=[addr, v])


def _st_global_b32(addr: Int64, v: Int32) -> None:
    cute.arch.inline_ptx("st.global.b32 [{$r0}], {$r1};", read_only_args=[addr, v])


def _red_release_add(addr: Int64, v: Int32) -> None:
    cute.arch.inline_ptx("red.release.gpu.global.add.u32 [{$r0}], {$r1};", read_only_args=[addr, v])


def _atom_add_acq_rel(addr: Int64, v: Int32) -> Int32:
    return cute.arch.inline_ptx(
        "atom.acq_rel.gpu.global.add.u32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[Int32],
        read_only_args=[addr, v],
    )


def _atom_add_relaxed(addr: Int64, v: Int32) -> Int32:
    return cute.arch.inline_ptx(
        "atom.relaxed.gpu.global.add.u32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[Int32],
        read_only_args=[addr, v],
    )


def _policy_evict_first() -> Int64:
    """An L2 cache policy: evict first (createpolicy.fractional, fraction 1.0)."""
    return cute.arch.inline_ptx(
        "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;", write_only_types=[Int64]
    )


def _policy_evict_last() -> Int64:
    """An L2 cache policy: evict last (createpolicy.fractional, fraction 1.0)."""
    return cute.arch.inline_ptx(
        "createpolicy.fractional.L2::evict_last.b64 {$w0}, 1.0;", write_only_types=[Int64]
    )


def _st_global_b32_hint(addr: Int64, v: Int32, pol: Int64) -> None:
    cute.arch.inline_ptx(
        "st.global.L2::cache_hint.b32 [{$r0}], {$r1}, {$r2};", read_only_args=[addr, v, pol]
    )


def _cp_async_16(dst: Int32, src: Int64) -> None:
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16;", read_only_args=[dst, src]
    )


def _cp_async_arrive_noinc(bar: Int32) -> None:
    """Arrive on bar once this thread's earlier cp.async copies complete (the barrier's count includes it)."""
    cute.arch.inline_ptx(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [{$r0}];", read_only_args=[bar]
    )


def _cp_async_commit() -> None:
    cute.arch.inline_ptx("cp.async.commit_group;")


def _bar_sync(bar_id: Int32, n: int) -> None:
    """Named barrier bar_id (a register) over n threads."""
    cute.arch.inline_ptx(f"bar.sync {{$r0}}, {n};", read_only_args=[bar_id])


# ----------------------------------------------------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------------------------------------------------
def _pack_bf16x2_rn(lo: cutlass.Float32, hi: cutlass.Float32) -> Int32:
    """{bf16(hi), bf16(lo)} by cvt.rn.bf16x2.f32 (no saturation)."""
    return cute.arch.inline_ptx(
        "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};", write_only_types=[Int32], read_only_args=[lo, hi]
    )


def _mul_rn(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[cutlass.Float32], read_only_args=[a, b]
    )


def _add_rn(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[cutlass.Float32], read_only_args=[a, b]
    )


def _rcp(a: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.inline_ptx(
        "rcp.approx.ftz.f32 {$w0}, {$r0};", write_only_types=[cutlass.Float32], read_only_args=[a]
    )


def _ex2(a: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.inline_ptx(
        "ex2.approx.ftz.f32 {$w0}, {$r0};", write_only_types=[cutlass.Float32], read_only_args=[a]
    )


def _scale_pack_bf16x2(lo: cutlass.Float32, hi: cutlass.Float32, s: cutlass.Float32) -> Int32:
    """{bf16_sat(hi * s), bf16_sat(lo * s)}: one IEEE mul.rn.f32 each, then cvt.rn.satfinite.bf16x2.f32 (the
    trtllm-gen FC2 cubins' FMUL2 + F2FP.SATFINITE.BF16.F32.PACK_AB)."""
    return cute.arch.inline_ptx(
        "{ .reg .f32 a, b; mul.rn.f32 a, {$r0}, {$r2}; mul.rn.f32 b, {$r1}, {$r2};"
        " cvt.rn.satfinite.bf16x2.f32 {$w0}, b, a; }",
        write_only_types=[Int32],
        read_only_args=[lo, hi, s],
    )


def sum_tree(vals: list, tree: SumTree) -> cutlass.Float32:
    """``vals`` summed (``add.f32``) in the order of ``tree`` (nested pairs of indices, e.g.
    ``moe_bf16_engine._butterfly_sum``)."""
    if isinstance(tree, int):
        return vals[tree]
    return _add(sum_tree(vals, tree[0]), sum_tree(vals, tree[1]))


# ----------------------------------------------------------------------------------------------------------------
# Tensor memory
# ----------------------------------------------------------------------------------------------------------------
def s2t_copy(
    sSF: cute.Tensor, tSF: cute.Tensor, cta_group: tcgen05.CtaGroup, sf_dtype: type
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    """The shared memory -> TMEM copy of block scales ``sSF`` into ``tSF`` (``tcgen05.cp`` 4x32x128b): the tiled
    copy, its shared-memory descriptor source and its TMEM destination."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t
