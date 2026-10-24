# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi-K3 KDA ``[f_a | b] -> f_b`` decode projection (SM100/SM103/SM107).

One launch computes, for ``x`` of ``1 <= M <= 512`` token rows,

* ``bfa = bf16(x @ W_bfa^T)`` (``W_bfa`` = ``[f_a_proj; b_proj]`` rows, FP32 accumulation);
* ``beta = bfa[:, 128 : 128 + H]``;
* ``forget_gate = bf16(bfa[:, :128] @ W_fb^T)`` (FP32 accumulation),

i.e. what a ``[f_a | b]`` GEMM, a slice copy and an ``f_b`` GEMM produce in three to five
launches.

The kernel runs on a second stream next to the KDA ``qkvg`` GEMM, which fills 96-120 SMs with
~220 KB of shared memory each (clusters of 4-5 CTAs). It therefore uses 27-28 large CTAs that fit
in the SMs the GEMM leaves free; more CTAs (or small ones that land between the GEMM's clusters)
delay the GEMM.

Design:

* GEMM 1, ``M <= 64`` (``mma.sync``). CTA ``(g, r, s)`` owns token group ``g`` (``8 * ntg``
  tokens), row group ``r`` (``9 / R`` 16-row tiles of ``W_bfa``, rows past ``N1`` zero-filled by
  TMA) and the ``s``-th slice of K in 64-wide chunks. One thread TMA-loads the CTA's whole weight
  slice (before ``griddepcontrol.wait``) and activation slice up front, one mbarrier per chunk.
  MMA warps tile the CTA's rows x tokens (``mt`` x ``nt`` m16n8k16 tiles each, rows on the MMA M
  side) and split the chunks into ``kg`` groups, summed through shared memory in a fixed order.
* GEMM 1, larger M (``tcgen05`` UMMA). CTA ``(g, s)`` owns 128 tokens (UMMA M) and all ``N1``
  rows (UMMA N = 144) of the ``s``-th K slice: the weight slice is TMA-loaded up front, the
  activation streams through a ring, the accumulator goes TMEM -> shared memory -> partial slot.
* Hand-off. Every CTA stores one FP32 partial tile ``[tokens, N1]`` into slots of a per-device
  workspace that are filled with a sentinel (``0xffffffff``, which no FP32 operation produces);
  consumers poll the data itself, so no fence, flag or counter sits between the producers' stores
  and the consumers' loads. The workspace has two halves used by alternate launches. A control
  warp per CTA takes a ticket from a packed 64-bit counter (high half: launch number, low half:
  CTAs of the launch that have arrived; the last arrival moves it to the next launch), caches the
  scalar parameters in shared memory, warms L2 with a share of ``W_fb``, and refills with the
  sentinel its share of the region the previous launch dirtied in the other half.
* Tail. The CTAs with the last ``T`` ranks (``T`` = tail work items) become workers. Worker
  ``(tb, rb)`` TMA-loads its ``W_fb`` row block, polls the ``R * S`` partials of its tokens (in
  register-sized batches with all loads of a batch in flight), sums them in split order, rounds
  ``f_a`` to BF16 into shared memory (worker ``rb == 0`` also writes ``beta``) and computes its
  forget-gate rows with ``mma.sync``. A worker only waits for CTAs with higher ranks or that
  already run, so progress needs ``T + 1`` co-resident CTAs, not grid-wide co-residency.
* Staged tail (UMMA GEMM 1, the default; ``FabConfig.staged``). Level 1: every CTA sums an equal
  share of the ``[token, 4-row unit]`` partials (same split order, same rounding) into BF16
  ``f_a`` rows of a sentinel-filled staging region after the partial slots, and writes ``beta``;
  meanwhile the workers' ``W_fb`` blocks stream into shared memory. Level 2: each worker
  (``8 ntb`` tokens x ``16 mtb`` rows) polls its staged rows into the MMA operand tile and runs
  the same ``mma.sync`` chain. A worker no longer reads ``S`` FP32 copies of its tokens nor all
  of ``W_fb``, and every CTA shares the reduction; the results are bitwise the one-level tail's.
* Launches sharing a device's counter and workspace must not overlap (stream order; with PDL the
  ticket and the refill happen after ``griddepcontrol.wait``).

``f_a`` mode (:func:`k3_kda_fab_fa`) runs the same GEMM 1 and hand-off but stops at
``bfa = bf16([f_a | b])``: every CTA becomes a worker that sums an equal share of the
``[token, 4-row unit]`` partials (same split order, same rounding, so bitwise the ``f_a`` and
``beta`` of the full kernel) and stores them as BF16 rows. The KDA decode then computes the
forget gate itself (``kda_mtp_decode_v2`` ``FUSED_FB``), which removes ``W_fb``, the f_b MMA and
the forget-gate round trip from this kernel. A worker polls the partials of every CTA, so all
(at most 28) CTAs must become co-resident; next to the ``qkvg`` GEMM, which does not depend on
this kernel, they do at the latest when that GEMM retires.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

#: Width of ``f_a`` (the ``W_fb`` reduction size).
FA_DIM = 128
#: Largest token count.
MAX_M = 512
#: Largest ``[f_a | b]`` row count (nine 16-row MMA tiles).
MAX_N1 = 144

_K_CHUNK = 64
_ROW_BYTES = _K_CHUNK * 2
_M_TILE = 16
_N_TILE = 8
_N1_TILES = MAX_N1 // _M_TILE
#: UMMA GEMM 1: tokens per CTA (UMMA M) and tensor-memory columns (144 accumulator columns).
_UMMA_M = 128
_UMMA_TMEM_COLS = 256
#: Row pitch (floats) of the UMMA accumulator staging tile (conflict-free 16-byte accesses).
_UMMA_STAGE_PITCH = MAX_N1 + 4
#: The tail probes one unit per producer before loading every partial when a thread loads at
#: least this many partial units per round.
_PROBE_MIN_LOADS = 32
#: Partial floats a tail thread keeps in registers per polling round.
_POLL_FLOATS = 128
#: The tail stages its forget-gate tile in shared memory for 16-byte stores at this size.
_STAGE_MIN_ELEMS = 4096
_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")
#: L2 eviction hint for the streamed weight (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: Partial-tile workspace per device: two halves used by alternate launches.
_WS_HALF_BYTES = 8 << 20
#: Device state words: packed ticket, then the dirtied bytes of each workspace half.
_STATE_WORDS = 8
_TICKET_WORD, _DIRTY_WORD = 0, 1
#: The sentinel that marks an empty FP32 workspace word.
_SENTINEL = -1
#: UMMA tilings use the staged tail (``FabConfig.staged``; bitwise the one-level tail's results):
#: ``TRTLLM_K3_KDA_FAB_STAGED`` = 1 (default) or 2 (the level-1 variant), 0 for the one-level tail.
STAGED_TAIL = int(os.environ.get("TRTLLM_K3_KDA_FAB_STAGED", "1"))
if STAGED_TAIL not in (0, 1, 2):
    raise ValueError(f"TRTLLM_K3_KDA_FAB_STAGED must be 0, 1 or 2, got {STAGED_TAIL}")

# Shared-memory cache of scalar kernel parameters (constant-cache misses in the tail are slow).
(_P_M, _P_BETA, _P_FG, _P_HALF, _P_RANK, _P_PREV, _P_OTHER) = range(7)
_NUM_PARAMS = 8


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class FabConfig:
    """Tiling of one specialization.

    Attributes:
        ntg: GEMM-1 token tiles (of 8) per CTA.
        r1: GEMM-1 row groups (1, 3 or 9).
        s: GEMM-1 K splits.
        mt: 16-row MMA tiles per warp (divides ``9 // r1``).
        nt: 8-token MMA tiles per warp.
        kg: K groups of warps (summed through shared memory).
        ntb: Tail token tiles per worker (divides ``ntg``).
        mtb: Tail ``W_fb`` 16-row tiles per worker.
        umma: GEMM 1 on ``tcgen05`` UMMA (128-token CTAs: ``ntg = 16``, ``r1 = 1``; ``mt``,
            ``nt`` and ``kg`` unused) instead of ``mma.sync``.
        staged: Two-level tail (UMMA only): every CTA sums an equal share of the partials into a
            BF16 ``f_a`` staging buffer (and ``beta``), then the workers (tiled by ``ntb`` /
            ``mtb``) apply ``f_b`` to the staged rows. Same sums, roundings and MMAs as the
            one-level tail, so bitwise the same results. 1: each thread polls its partial units
            with global loads; 2: the CTA bulk-copies the partial rows of its share (one copy per
            K split) into shared memory and sums them there.
    """

    ntg: int
    r1: int
    s: int
    mt: int
    nt: int
    kg: int
    ntb: int
    mtb: int
    umma: bool = False
    staged: int = 0


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _ldsm_x4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _mma_bf16(a, b0, b1, c, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (BF16 inputs, FP32 accumulate)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (*a, b0, b1)] + [_fv(x, loc=loc, ip=ip) for x in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_bf16x4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Round four FP32 values to BF16 (nearest even) and store them as 8 bytes."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 lo, hi;\ncvt.rn.bf16x2.f32 lo, $2, $1;\ncvt.rn.bf16x2.f32 hi, $4, $3;\n"
        "st.shared.v2.b32 [$0], {lo, hi};\n}",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "st.shared.b64 [$0], $1;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b64 $0, [$1];", "=l,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ld_relaxed_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_global_bf16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_bf16x4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Round four FP32 values to BF16 (nearest even) and store them as 8 bytes."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 lo, hi;\ncvt.rn.bf16x2.f32 lo, $2, $1;\ncvt.rn.bf16x2.f32 hi, $4, $3;\n"
        "st.global.v2.b32 [$0], {lo, hi};\n}",
        "l,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_bf16x4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Round four FP32 values to BF16 (nearest even) and store them as 8 bytes (relaxed, gpu)."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 lo, hi;\ncvt.rn.bf16x2.f32 lo, $2, $1;\ncvt.rn.bf16x2.f32 hi, $4, $3;\n"
        "st.relaxed.gpu.global.v2.b32 [$0], {lo, hi};\n}",
        "l,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """``nbytes`` (a multiple of 16) from global ``src`` to shared ``dst``, completing on ``mbar``."""
    _asm(
        None,
        [
            _iv(dst, loc=loc, ip=ip),
            _lv(src, loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_add_u64(addr, value, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_add_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_u64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.relaxed.gpu.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _all_valid4(v0, v1, v2, v3, *, loc=None, ip=None):
    """1 if none of the four FP32 words is the empty-slot sentinel ``0xffffffff``."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n.reg .pred p;\n.reg .b32 t;\n"
            "mov.b32 t, $1;\nsetp.ne.b32 p, t, -1;\n"
            "mov.b32 t, $2;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "mov.b32 t, $3;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "mov.b32 t, $4;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _min_s32(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _prefetch_l2(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_bf16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.shared.b16 [$0], h;\n}",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_s2g_16(dst, src, *, loc=None, ip=None):
    """16 bytes from shared ``src`` to global ``dst``."""
    _asm(
        None,
        [_lv(dst, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip)],
        "{\n.reg .b32 a, b, c, d;\nld.shared.v4.b32 {a, b, c, d}, [$1];\n"
        "st.global.v4.b32 [$0], {a, b, c, d};\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _tmem_ld16(taddr, *, loc=None, ip=None):
    """16 consecutive FP32 accumulator columns of this thread's TMEM lane."""
    regs = ", ".join(f"${i}" for i in range(16))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [_iv(taddr, loc=loc, ip=ip)],
        "tcgen05.fence::after_thread_sync;\n"
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16];\n"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * 16) + ",r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 16, loc=loc, ip=ip)


@dsl_user_op
def _tmem_fence_before_sync(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class _FabTail:
    """Control warp, hand-off and tail shared by the GEMM-1 variants (see the module docstring).

    Subclasses set the geometry attributes (``num_ctas``, ``items``, ``tg``, ``threads``, ...),
    run :meth:`control_warp` and :meth:`refill` on one warp, store their partial tiles into the
    workspace half cached in shared memory and then run :meth:`tail` on the worker CTAs.
    """

    def _init_common(
        self, cfg, n_tiles, n1, n2, k, num_heads, use_pdl, geo, fa_mode=False, warm_w2=True
    ):
        self.cfg = cfg
        self.n_tiles = n_tiles
        self.n1 = n1
        self.n2 = n2
        self.k = k
        self.num_heads = num_heads
        self.use_pdl = use_pdl
        self.fa_mode = fa_mode
        # Bytes of W_fb the control warps warm in L2 (the tail's operand; in f_a mode the KDA
        # decode's, whose per-head TMA load would otherwise come from HBM).
        self.warm_bytes = n2 * FA_DIM * 2 if warm_w2 else 0
        for name, value in geo.items():
            setattr(self, name, value)
        if fa_mode:
            # Every CTA is a worker over an equal share of the [token, 4-row unit] partials.
            self.items = self.num_ctas
            units_max = n_tiles * _N_TILE * (n1 // 4)
            self.fa_units_per_thread = _ceil_div(_ceil_div(units_max, self.items), self.threads)
        if fa_mode and cfg.staged:
            raise ValueError("f_a mode has no f_b tail to stage")
        self.dirty_bytes = self.groups * cfg.s * self.tg * n1 * 4
        if cfg.staged:
            # BF16 f_a rows [groups * tg, 128] after the partial slots, sentinel-refilled with them.
            self.stage_off = self.dirty_bytes
            self.dirty_bytes += self.groups * self.tg * FA_DIM * 2
            units_max = n_tiles * _N_TILE * (n1 // 4)
            self.red_units_per_thread = _ceil_div(_ceil_div(units_max, self.num_ctas), self.threads)
            # 16-byte vectors (8 f_a columns) of the worker's staged rows per thread.
            self.stage_vecs_per_thread = _ceil_div(
                cfg.ntb * _N_TILE * (FA_DIM * 2 // 16), self.threads
            )
            # Bulk level 1: CTA (g, j) of a token group sums tokens [j l1_tpc, (j + 1) l1_tpc).
            self.l1_tpc = _ceil_div(self.tg, cfg.s)
            self.l1_units_per_thread = _ceil_div(self.l1_tpc * (n1 // 4), self.threads)
        self.beta_units = _ceil_div(num_heads, 4)
        # Tail units (4 consecutive rows of one token) a thread polls at once.
        self.units_per_thread = _ceil_div(
            cfg.ntb * _N_TILE * (FA_DIM // 4 + self.beta_units), self.threads
        )

    @cute.jit
    def make_w2_tma(self, w2_ptr: cute.Pointer):
        """TMA atom of ``W_fb`` boxes and their shared-memory layout (the tail's row block)."""
        mW2 = cute.make_tensor(w2_ptr, cute.make_layout((self.n2, FA_DIM), stride=(FA_DIM, 1)))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW2,
            cute.tile_to_shape(atom, (self.w2_box_rows, _K_CHUNK), order=(0, 1)),
            (self.w2_box_rows, _K_CHUNK),
        )
        layout = cute.tile_to_shape(atom, (self.w2_box_rows, _K_CHUNK, self.w2_boxes), (0, 1, 2))
        return tma_atom, tma_tensor, layout

    @cute.jit
    def control_warp(
        self,
        params: Int32,
        bidx: Int32,
        lane: Int32,
        m: Int32,
        w2_addr: Int64,
        beta_addr: Int64,
        fg_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
    ):
        """Ticket, parameter cache (in shared memory) and ``W_fb`` L2 warm-up."""
        if lane == 0:
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            ticket = _atom_add_u64(state_addr + Int64(_TICKET_WORD * 8), Int64(1))
            launch = ticket >> 32
            rank = ticket & Int64(0xFFFFFFFF)
            if rank == self.num_ctas - 1:
                # Last arrival: clear the arrivals and advance the launch number in one add.
                _red_add_u64(state_addr + Int64(_TICKET_WORD * 8), Int64((1 << 32) - self.num_ctas))
            parity = launch & Int64(1)
            prev = _ld_relaxed_u64(state_addr + Int64((_DIRTY_WORD + 1 - parity) * 8))
            if rank == 0:
                _st_relaxed_u64(
                    state_addr + Int64((_DIRTY_WORD + parity) * 8), Int64(self.dirty_bytes)
                )
            cached = (
                Int64(m),
                beta_addr,
                fg_addr,
                ws_addr + parity * _WS_HALF_BYTES,
                rank,
                prev,
                ws_addr + (Int64(1) - parity) * _WS_HALF_BYTES,
            )
            for i in cutlass.range_constexpr(len(cached)):
                _sts_b64(params + i * 8, cached[i])
            # Warm L2 with this CTA's share of W_fb for the tail.
            total = self.warm_bytes
            if cutlass.const_expr(total > 0):
                share = _ceil_div(_ceil_div(total, self.num_ctas), 16) * 16
                off = bidx * share
                if off < total:
                    nbytes = Int32(share)
                    if off + share > total:
                        nbytes = total - off
                    _prefetch_l2(w2_addr + Int64(off), nbytes)
        cute.arch.sync_warp()

    @cute.jit
    def refill(self, params: Int32, lane: Int32):
        """Refill this CTA's share of what the previous launch dirtied in the other half."""
        prev = _lds_b64(params + _P_PREV * 8)
        other = _lds_b64(params + _P_OTHER * 8)
        rank = _lds_b64(params + _P_RANK * 8)
        share = ((prev + self.num_ctas - 1) // self.num_ctas + 15) // 16 * 16
        begin = rank * share
        end = begin + share
        if end > prev:
            end = prev
        off = begin + Int64(lane * 16)
        while off < end:
            _st_sentinel_v4(other + off)
            off = off + 512

    @cute.jit
    def issue_w2(
        self,
        sW2: cute.Tensor,
        w2_full,
        tma_atom_w2: cute.CopyAtom,
        mW2t: cute.Tensor,
        worker: Int32,
    ):
        """Warp 0: the worker's ``W_fb`` row block -> shared memory: TMA boxes of
        ``w2_box_rows`` x 64 columns (128B swizzle), rows past ``N2`` zero-filled."""
        row0 = (worker // self.token_blocks) * (self.cfg.mtb * _M_TILE)
        # The block reuses shared memory last accessed through the generic proxy.
        _fence_proxy_async_shared()
        gW2 = cute.local_tile(mW2t, (self.w2_box_rows, _K_CHUNK), (None, None))
        tW2s, tW2g = cpasync.tma_partition(
            tma_atom_w2,
            0,
            cute.make_layout(1),
            cute.group_modes(sW2, 0, 2),
            cute.group_modes(gW2, 0, 2),
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(w2_full, self.w2_smem_bytes)
        box0 = row0 // self.w2_box_rows
        for b in cutlass.range_constexpr(self.w2_boxes):
            cute.copy(
                tma_atom_w2,
                tW2g[(None, box0 + b // 2, b % 2)],
                tW2s[(None, b)],
                tma_bar_ptr=w2_full,
            )

    @cute.jit
    def reduce_staged(self, params: Int32, tidx: Int32, rank: Int32, m: Int32, half: Int64):
        """Staged tail, level 1 (every CTA): sum this CTA's share of the ``[token, 4-row unit]``
        partials in split order (as :meth:`tail`), store ``f_a`` rounded to BF16 into the staging
        rows and ``beta`` into its output."""
        S = self.cfg.s
        beta_addr = _lds_b64(params + _P_BETA * 8)
        stage_base = half + Int64(self.stage_off)
        upt = self.n1 // 4
        total = m * upt
        per = (total + self.num_ctas - 1) // self.num_ctas
        begin = rank * per
        end = begin + per
        if end > total:
            end = total
        slot_stride = self.tg * self.n1 * 4
        U = self.red_units_per_thread
        batch = max(1, _POLL_FLOATS // (4 * S))
        for b0 in cutlass.range_constexpr(0, U, batch):
            nb = min(batch, U - b0)
            valid = []
            srcs = []
            for i in cutlass.range_constexpr(nb):
                idx = begin + tidx + (b0 + i) * self.threads
                t = idx // upt
                u = idx - t * upt
                g = t // self.tg
                valid.append(Int32(idx < end))
                srcs.append(
                    half + Int64(((g * S * self.tg + t - g * self.tg) * self.n1 + 4 * u) * 4)
                )
            parts = cute.make_rmem_tensor((nb * S * 4,), Float32)
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                # All loads of the batch are in flight before the first sentinel check.
                for i in cutlass.range_constexpr(nb):
                    if valid[i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            v = _ld_relaxed_v4_f32(srcs[i] + Int64(sp * slot_stride))
                            for e in cutlass.range_constexpr(4):
                                parts[(i * S + sp) * 4 + e] = v[e]
                for i in cutlass.range_constexpr(nb):
                    if valid[i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            o = (i * S + sp) * 4
                            ready = ready & _all_valid4(
                                parts[o], parts[o + 1], parts[o + 2], parts[o + 3]
                            )
            for i in cutlass.range_constexpr(nb):
                if valid[i] != 0:
                    idx = begin + tidx + (b0 + i) * self.threads
                    t = idx // upt
                    u = idx - t * upt
                    v = [parts[i * S * 4 + e] for e in range(4)]
                    for sp in cutlass.range_constexpr(1, S):
                        for e in cutlass.range_constexpr(4):
                            v[e] = v[e] + parts[(i * S + sp) * 4 + e]
                    if u < FA_DIM // 4:
                        _st_relaxed_bf16x4(stage_base + Int64((t * FA_DIM + 4 * u) * 2), *v)
                    else:
                        for e in cutlass.range_constexpr(4):
                            h = 4 * (u - FA_DIM // 4) + e
                            if h < self.num_heads:
                                _st_global_bf16(
                                    beta_addr + Int64((t * self.num_heads + h) * 2), v[e]
                                )

    @cute.jit
    def reduce_staged_bulk(
        self,
        l1: Int32,
        red_full,
        vote: Int32,
        params: Int32,
        tidx: Int32,
        rank: Int32,
        m: Int32,
        half: Int64,
    ):
        """Staged tail, level 1, bulk variant: CTA ``rank = (g, j)`` copies rows ``[j l1_tpc,
        (j + 1) l1_tpc)`` of token group ``g`` from every K-split slot into shared memory (one
        bulk copy per slot), repeats the copies until no word holds the sentinel, then sums each
        ``[token, 4-row unit]`` in split order like :meth:`reduce_staged`."""
        S = self.cfg.s
        tg = self.tg
        n1 = self.n1
        tpc = self.l1_tpc
        row_bytes = n1 * 4
        upt = n1 // 4
        beta_addr = _lds_b64(params + _P_BETA * 8)
        stage_base = half + Int64(self.stage_off)
        g = rank // S
        t_begin = (rank - g * S) * tpc
        nt = Int32(tpc)
        if t_begin + nt > tg:
            nt = tg - t_begin
        if g * tg + t_begin + nt > m:
            nt = m - g * tg - t_begin
        if nt < 0:
            nt = Int32(0)
        nbytes = nt * row_bytes
        src0 = half + Int64((g * S * tg + t_begin) * row_bytes)
        U = self.l1_units_per_thread
        parts = cute.make_rmem_tensor((U * S * 4,), Float32)
        rnd = Int32(0)
        done = Int32(nt == 0)
        while done == 0:
            if tidx == 0:
                # The buffer was last read through the generic proxy (previous round).
                _fence_proxy_async_shared()
                cute.arch.mbarrier_arrive_and_expect_tx(red_full, S * nbytes)
                for sp in cutlass.range_constexpr(S):
                    _bulk_g2s(
                        l1 + sp * (tpc * row_bytes),
                        src0 + Int64(sp * tg * row_bytes),
                        nbytes,
                        red_full.toint(),
                    )
            cute.arch.mbarrier_wait(red_full, rnd & 1)
            ok = Int32(1)
            for it in cutlass.range_constexpr(U):
                idx = tidx + it * self.threads
                tl = idx // upt
                u = idx - tl * upt
                if idx < nt * upt:
                    for sp in cutlass.range_constexpr(S):
                        v = _lds_v4_f32(l1 + (sp * tpc + tl) * row_bytes + 16 * u)
                        for e in cutlass.range_constexpr(4):
                            parts[(it * S + sp) * 4 + e] = v[e]
            for it in cutlass.range_constexpr(U):
                idx = tidx + it * self.threads
                if idx < nt * upt:
                    for sp in cutlass.range_constexpr(S):
                        o = (it * S + sp) * 4
                        ok = ok & _all_valid4(parts[o], parts[o + 1], parts[o + 2], parts[o + 3])
            # CTA vote: any sentinel left -> copy again (word rnd & 1; the other one is reset
            # for the next round before anyone can write it).
            if ok == 0:
                _sts_b32(vote + (rnd & 1) * 4, 1)
            cute.arch.barrier()
            bad = _lds_b32(vote + (rnd & 1) * 4)
            if tidx == 0:
                _sts_b32(vote + ((rnd + 1) & 1) * 4, 0)
            cute.arch.barrier()
            done = Int32(bad == 0)
            rnd = rnd + 1
        for it in cutlass.range_constexpr(U):
            idx = tidx + it * self.threads
            tl = idx // upt
            u = idx - tl * upt
            if idx < nt * upt:
                t = g * tg + t_begin + tl
                v = [parts[it * S * 4 + e] for e in range(4)]
                for sp in cutlass.range_constexpr(1, S):
                    for e in cutlass.range_constexpr(4):
                        v[e] = v[e] + parts[(it * S + sp) * 4 + e]
                if u < FA_DIM // 4:
                    _st_relaxed_bf16x4(stage_base + Int64((t * FA_DIM + 4 * u) * 2), *v)
                else:
                    for e in cutlass.range_constexpr(4):
                        h = 4 * (u - FA_DIM // 4) + e
                        if h < self.num_heads:
                            _st_global_bf16(beta_addr + Int64((t * self.num_heads + h) * 2), v[e])

    @cute.jit
    def tail(
        self,
        region: Int32,
        sW2: cute.Tensor,
        w2_full,
        tma_atom_w2: cute.CopyAtom,
        mW2t: cute.Tensor,
        params: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        worker: Int32,
        m: Int32,
        half: Int64,
    ):
        cfg = self.cfg
        fg_addr = _lds_b64(params + _P_FG * 8)
        tb = worker % self.token_blocks
        rb = worker // self.token_blocks
        sw2 = region
        sfa = sw2 + self.w2_smem_bytes
        tb_rows = cfg.ntb * _N_TILE
        sout = sfa + tb_rows * FA_DIM * 2
        row0 = rb * (cfg.mtb * _M_TILE)
        t0 = tb * tb_rows
        if cutlass.const_expr(cfg.staged):
            # The W_fb block is already on its way (issued before the level-1 reduction); the
            # block's f_a rows come from the staging buffer: poll each 16-byte vector until it
            # holds no sentinel, then place it into the swizzled MMA operand tile.
            stage_base = half + Int64(self.stage_off)
            V = self.stage_vecs_per_thread
            vec_per_row = FA_DIM * 2 // 16
            vvalid = []
            vsrcs = []
            for it in cutlass.range_constexpr(V):
                idx = tidx + it * self.threads
                tl = idx // vec_per_row
                j = idx % vec_per_row
                t = t0 + tl
                vvalid.append(Int32(idx < tb_rows * vec_per_row) * Int32(t < m))
                vsrcs.append(stage_base + Int64((t * FA_DIM + 8 * j) * 2))
            vals = cute.make_rmem_tensor((V * 4,), Float32)
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                for it in cutlass.range_constexpr(V):
                    if vvalid[it] != 0:
                        v = _ld_relaxed_v4_f32(vsrcs[it])
                        for e in cutlass.range_constexpr(4):
                            vals[it * 4 + e] = v[e]
                for it in cutlass.range_constexpr(V):
                    if vvalid[it] != 0:
                        o = it * 4
                        ready = ready & _all_valid4(vals[o], vals[o + 1], vals[o + 2], vals[o + 3])
            for it in cutlass.range_constexpr(V):
                if vvalid[it] != 0:
                    idx = tidx + it * self.threads
                    tl = idx // vec_per_row
                    j = idx % vec_per_row
                    dst = sfa + (j // 8) * (tb_rows * _ROW_BYTES) + tl * _ROW_BYTES
                    dst = dst + (((j % 8) ^ (tl & 7)) << 4)
                    _sts_v4_f32(dst, *[vals[it * 4 + e] for e in range(4)])
        else:
            if warp == 0:
                self.issue_w2(sW2, w2_full, tma_atom_w2, mW2t, worker)
            self.tail_from_partials(sfa, params, tidx, warp, lane, rb, t0, m, half)
        cute.arch.mbarrier_wait(w2_full, 0)
        cute.arch.barrier()
        self.fb_mma_store(sw2, sfa, sout, fg_addr, tidx, warp, lane, row0, t0, m)

    @cute.jit
    def tail_from_partials(
        self,
        sfa: Int32,
        params: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        rb: Int32,
        t0: Int32,
        m: Int32,
        half: Int64,
    ):
        """f_a (and, for ``rb == 0``, beta) of the block's tokens: poll the ``R * S`` partials of
        each 4-row unit until no slot holds the sentinel, then sum them in split order."""
        cfg = self.cfg
        S = cfg.s
        beta_addr = _lds_b64(params + _P_BETA * 8)
        tb_rows = cfg.ntb * _N_TILE
        g = t0 // self.tg
        upt = FA_DIM // 4 + self.beta_units * Int32(rb == 0)
        total = tb_rows * upt
        slot_stride = self.tg * self.n1 * 4
        U = self.units_per_thread
        valid = []
        srcs = []
        for it in cutlass.range_constexpr(U):
            idx = tidx + it * self.threads
            tl = idx // upt
            u = idx % upt
            # Unit u covers rows [4 u, 4 u + 4): f_a for u < 32, beta (rows 128 + ...) after.
            row = 4 * u
            t = t0 + tl
            valid.append(Int32(idx < total) * Int32(t < m))
            srcs.append(half + Int64(((g * S * self.tg + t - g * self.tg) * self.n1 + row) * 4))
        # Cheap wait first: lane i of warp 0 polls one unit of producer slot i (split, row
        # group) until it is set; the full loads below then rarely need a second round.
        if cutlass.const_expr(U * S >= _PROBE_MIN_LOADS):
            self.probe(warp, lane, t0, tb_rows, g, m, half)
        # Units are polled in batches of `batch` (all their loads in flight together) so the
        # partials of a batch fit in registers.
        batch = max(1, _POLL_FLOATS // (4 * S))
        for b0 in cutlass.range_constexpr(0, U, batch):
            nb = min(batch, U - b0)
            parts = cute.make_rmem_tensor((nb * S * 4,), Float32)
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                # All loads of the batch are in flight before the first sentinel check.
                for i in cutlass.range_constexpr(nb):
                    if valid[b0 + i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            v = _ld_relaxed_v4_f32(srcs[b0 + i] + Int64(sp * slot_stride))
                            for e in cutlass.range_constexpr(4):
                                parts[(i * S + sp) * 4 + e] = v[e]
                for i in cutlass.range_constexpr(nb):
                    if valid[b0 + i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            o = (i * S + sp) * 4
                            ready = ready & _all_valid4(
                                parts[o], parts[o + 1], parts[o + 2], parts[o + 3]
                            )
            for i in cutlass.range_constexpr(nb):
                if valid[b0 + i] != 0:
                    idx = tidx + (b0 + i) * self.threads
                    tl = idx // upt
                    u = idx % upt
                    t = t0 + tl
                    v = [parts[i * S * 4 + e] for e in range(4)]
                    for sp in cutlass.range_constexpr(1, S):
                        for e in cutlass.range_constexpr(4):
                            v[e] = v[e] + parts[(i * S + sp) * 4 + e]
                    if u < FA_DIM // 4:
                        kc = u // 16
                        unit = (u % 16) // 2
                        dst = sfa + kc * (tb_rows * _ROW_BYTES) + tl * _ROW_BYTES
                        dst = dst + ((unit ^ (tl & 7)) << 4) + (u % 2) * 8
                        _sts_bf16x4(dst, *v)
                    else:
                        for e in cutlass.range_constexpr(4):
                            h = 4 * (u - FA_DIM // 4) + e
                            if h < self.num_heads:
                                _st_global_bf16(
                                    beta_addr + Int64((t * self.num_heads + h) * 2), v[e]
                                )

    @cute.jit
    def fb_mma_store(
        self,
        sw2: Int32,
        sfa: Int32,
        sout: Int32,
        fg_addr: Int64,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        row0: Int32,
        t0: Int32,
        m: Int32,
    ):
        """``forget_gate[block tokens, block rows] = f_a @ W_fb^T`` from the operand tiles in
        shared memory, rounded to BF16 and stored."""
        cfg = self.cfg
        blk_rows = cfg.mtb * _M_TILE
        tb_rows = cfg.ntb * _N_TILE
        out_pitch = blk_rows + 8
        stage = tb_rows * blk_rows >= _STAGE_MIN_ELEMS
        # forget_gate[block tokens, block rows] = f_a @ W_fb^T.
        mat = lane >> 3
        r = lane & 7
        a_unit = mat >> 1
        b_unit = mat & 1
        ntb = cfg.ntb
        b_rows = [
            ((2 * p + (mat >> 1) * min(1, ntb - 1 - 2 * p)) * _N_TILE + r) * _ROW_BYTES
            for p in range(_ceil_div(ntb, 2))
        ]
        # The block's f_a fragments are loaded once per warp; a warp's m-tiles (warp, warp +
        # num_warps, ...) accumulate independently.
        nk = FA_DIM // 16
        bfrag = cute.make_rmem_tensor((nk * _ceil_div(ntb, 2) * 4,), Int32)
        for kk in cutlass.range_constexpr(nk):
            b_chunk = sfa + (kk // 4) * (tb_rows * _ROW_BYTES)
            for p in cutlass.range_constexpr(_ceil_div(ntb, 2)):
                b = _ldsm_x4(b_chunk + b_rows[p] + ((((kk % 4) * 2 + b_unit) ^ r) << 4))
                for i in cutlass.range_constexpr(4):
                    bfrag[(kk * _ceil_div(ntb, 2) + p) * 4 + i] = b[i]
        mpw = _ceil_div(cfg.mtb, self.num_warps)
        acc = cute.make_rmem_tensor((mpw * ntb * 4,), Float32)
        acc.fill(0.0)
        a_rows = []
        for w in cutlass.range_constexpr(mpw):
            # Warps past the block's last m-tile redo it (their results are not stored).
            mi = _min_s32(warp + w * self.num_warps, cfg.mtb - 1)
            row_l = mi * _M_TILE + (mat & 1) * 8 + r
            a_rows.append(
                sw2
                + (row_l // self.w2_box_rows) * 2 * (self.w2_box_rows * _ROW_BYTES)
                + (row_l % self.w2_box_rows) * _ROW_BYTES
            )
        for kk in cutlass.range_constexpr(nk):
            for w in cutlass.range_constexpr(mpw):
                a = _ldsm_x4(
                    a_rows[w]
                    + (kk // 4) * (self.w2_box_rows * _ROW_BYTES)
                    + ((((kk % 4) * 2 + a_unit) ^ r) << 4)
                )
                for j in cutlass.range_constexpr(ntb):
                    p = j // 2
                    q = j % 2
                    o = (w * ntb + j) * 4
                    base = (kk * _ceil_div(ntb, 2) + p) * 4
                    d = _mma_bf16(
                        a,
                        bfrag[base + 2 * q],
                        bfrag[base + 2 * q + 1],
                        (acc[o], acc[o + 1], acc[o + 2], acc[o + 3]),
                    )
                    for e in cutlass.range_constexpr(4):
                        acc[o + e] = d[e]
        for w in cutlass.range_constexpr(mpw):
            mi = warp + w * self.num_warps
            if mi < cfg.mtb:
                for j in cutlass.range_constexpr(ntb):
                    for e in cutlass.range_constexpr(4):
                        tl = j * _N_TILE + 2 * (lane & 3) + (e & 1)
                        row_l = mi * _M_TILE + (lane >> 2) + (e >> 1) * 8
                        val = acc[(w * ntb + j) * 4 + e]
                        if cutlass.const_expr(stage):
                            _sts_bf16(sout + (tl * out_pitch + row_l) * 2, val)
                        elif t0 + tl < m and row0 + row_l < self.n2:
                            _st_global_bf16(
                                fg_addr + Int64(((t0 + tl) * self.n2 + row0 + row_l) * 2), val
                            )
        if cutlass.const_expr(stage):
            # Coalesced 16-byte stores of the staged [tokens, rows] tile.
            cute.arch.barrier()
            for idx in cutlass.range(tidx, tb_rows * (blk_rows // 8), self.threads, unroll=1):
                tl = idx // (blk_rows // 8)
                uu = idx % (blk_rows // 8)
                t = t0 + tl
                row = row0 + uu * 8
                if t < m and row < self.n2:
                    _copy_s2g_16(
                        fg_addr + Int64((t * self.n2 + row) * 2),
                        sout + (tl * out_pitch + uu * 8) * 2,
                    )

    @cute.jit
    def tail_fa(self, params: Int32, tidx: Int32, worker: Int32, m: Int32, half: Int64):
        """f_a mode: sum this worker's share of the partials and store ``bf16`` rows of ``bfa``.

        Unit ``idx`` is rows ``[4 u, 4 u + 4)`` of token ``t`` (``idx = t * N1 / 4 + u``); its
        ``S`` split partials are polled until set and summed in split order, as in :meth:`tail`.
        """
        S = self.cfg.s
        out_addr = _lds_b64(params + _P_FG * 8)
        upt = self.n1 // 4
        total = m * upt
        per = (total + self.items - 1) // self.items
        begin = worker * per
        end = begin + per
        if end > total:
            end = total
        slot_stride = self.tg * self.n1 * 4
        U = self.fa_units_per_thread
        batch = max(1, _POLL_FLOATS // (4 * S))
        for b0 in cutlass.range_constexpr(0, U, batch):
            nb = min(batch, U - b0)
            valid = []
            srcs = []
            dsts = []
            for i in cutlass.range_constexpr(nb):
                idx = begin + tidx + (b0 + i) * self.threads
                t = idx // upt
                u = idx - t * upt
                g = t // self.tg
                valid.append(Int32(idx < end))
                srcs.append(
                    half + Int64(((g * S * self.tg + t - g * self.tg) * self.n1 + 4 * u) * 4)
                )
                dsts.append(out_addr + Int64((t * self.n1 + 4 * u) * 2))
            parts = cute.make_rmem_tensor((nb * S * 4,), Float32)
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                # All loads of the batch are in flight before the first sentinel check.
                for i in cutlass.range_constexpr(nb):
                    if valid[i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            v = _ld_relaxed_v4_f32(srcs[i] + Int64(sp * slot_stride))
                            for e in cutlass.range_constexpr(4):
                                parts[(i * S + sp) * 4 + e] = v[e]
                for i in cutlass.range_constexpr(nb):
                    if valid[i] != 0:
                        for sp in cutlass.range_constexpr(S):
                            o = (i * S + sp) * 4
                            ready = ready & _all_valid4(
                                parts[o], parts[o + 1], parts[o + 2], parts[o + 3]
                            )
            for i in cutlass.range_constexpr(nb):
                if valid[i] != 0:
                    v = [parts[i * S * 4 + e] for e in range(4)]
                    for sp in cutlass.range_constexpr(1, S):
                        for e in cutlass.range_constexpr(4):
                            v[e] = v[e] + parts[(i * S + sp) * 4 + e]
                    _st_global_bf16x4(dsts[i], *v)

    @cute.jit
    def probe(
        self, warp: Int32, lane: Int32, t0: Int32, tb_rows: int, g: Int32, m: Int32, half: Int64
    ):
        """Lane i of warp 0 polls one unit of producer slot i (split, row group) until set."""
        cfg = self.cfg
        S = cfg.s
        if warp == 0:
            if lane < S * cfg.r1:
                sp = lane // cfg.r1
                row_end = (lane - sp * cfg.r1 + 1) * self.rows_cta
                if row_end > self.n1:
                    row_end = Int32(self.n1)
                t_last = t0 + tb_rows - 1
                if t_last > m - 1:
                    t_last = m - 1
                probe = half + Int64(
                    (((g * S + sp) * self.tg + t_last - g * self.tg) * self.n1 + row_end - 4) * 4
                )
                v = _ld_relaxed_v4_f32(probe)
                ok = _all_valid4(*v)
                while ok == 0:
                    v = _ld_relaxed_v4_f32(probe)
                    ok = _all_valid4(*v)
        cute.arch.barrier()


class K3KdaFabKernel(_FabTail):
    """``beta``, ``forget_gate`` from ``x``, ``W_bfa`` and ``W_fb`` (see the module docstring).

    Args:
        cfg: Tiling.
        n_tiles: Token tiles of 8 (``ceil(M / 8)``) of the specialization.
        n1: Rows of ``W_bfa`` (``128 + H`` padded to a multiple of 8, at most 144).
        n2: Rows of ``W_fb`` (a multiple of 16).
        k: Hidden size (a multiple of 64).
        num_heads: ``H``; ``beta`` is rows ``[128, 128 + H)`` of ``bfa``.
        use_pdl: Launch with programmatic dependent launch.
        fa_mode: Stop at ``bfa`` (written through the forget-gate address; see the module
            docstring).
        warm_w2: Warm L2 with ``W_fb`` (``n2`` rows at ``w2_addr``).
    """

    def __init__(
        self,
        cfg: FabConfig,
        n_tiles: int,
        n1: int,
        n2: int,
        k: int,
        num_heads: int,
        use_pdl: bool,
        fa_mode: bool = False,
        warm_w2: bool = True,
    ):
        validate_config(cfg, n_tiles, n1, n2, k, num_heads)
        self._init_common(
            cfg,
            n_tiles,
            n1,
            n2,
            k,
            num_heads,
            use_pdl,
            _geometry(cfg, n_tiles, n2, k),
            fa_mode,
            warm_w2,
        )

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        m: Int32,
        w2_addr: Int64,
        beta_addr: Int64,
        fg_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(w1_ptr, cute.make_layout((self.n1, self.k), stride=(self.k, 1)))
        mX = cute.make_tensor(x_ptr, cute.make_layout((m, self.k), stride=(self.k, 1)))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        w_layout = cute.tile_to_shape(atom, (self.rows_cta, _K_CHUNK, self.max_chunks), (0, 1, 2))
        x_layout = cute.tile_to_shape(atom, (self.tg, _K_CHUNK, self.max_chunks), (0, 1, 2))
        tma_atom_w, tma_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW,
            cute.tile_to_shape(atom, (self.rows_cta, _K_CHUNK), order=(0, 1)),
            (self.rows_cta, _K_CHUNK),
        )
        tma_atom_x, tma_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.tile_to_shape(atom, (self.tg, _K_CHUNK), order=(0, 1)),
            (self.tg, _K_CHUNK),
        )
        tma_atom_w2, tma_w2, w2_layout = self.make_w2_tma(w2_ptr)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.max_chunks]
            w2_full: cute.struct.MemRange[cutlass.Int64, 1]
            params: cute.struct.MemRange[cutlass.Int64, _NUM_PARAMS]
            # The weight and activation chunk buffers; once the GEMM-1 MMAs are done, the same
            # bytes hold the warp-partial reduction and then the tail's operands.
            sW: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, self.sw_elems], 1024]
            sX: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, self.sx_elems], 1024]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_w,
            tma_w,
            tma_atom_x,
            tma_x,
            w_layout,
            x_layout,
            tma_atom_w2,
            tma_w2,
            w2_layout,
            m,
            w2_addr,
            beta_addr,
            fg_addr,
            ws_addr,
            state_addr,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mWt: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mXt: cute.Tensor,
        w_layout: cute.ComposedLayout,
        x_layout: cute.ComposedLayout,
        tma_atom_w2: cute.CopyAtom,
        mW2t: cute.Tensor,
        w2_layout: cute.ComposedLayout,
        m: Int32,
        w2_addr: Int64,
        beta_addr: Int64,
        fg_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
    ):
        cfg = self.cfg
        S = cfg.s
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        group = bidx % self.groups
        rest = bidx // self.groups
        rgrp = rest % cfg.r1
        split = rest // cfg.r1
        base_chunks = self.num_k_chunks // S
        rem_chunks = self.num_k_chunks % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr()
        params = storage.params.data_ptr().toint()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.max_chunks):
                    cute.arch.mbarrier_init(mbar + i, 1)
                cute.arch.mbarrier_init(storage.w2_full.data_ptr(), 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- GEMM-1 loads: the weight never depends on the previous grid. ----
        if warp == 0:
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)
            sW = storage.sW.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
            gW = cute.local_tile(mWt, (self.rows_cta, _K_CHUNK), (None, None))
            gX = cute.local_tile(mXt, (self.tg, _K_CHUNK), (None, None))
            tWs, tWg = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
            tXs, tXg = cpasync.tma_partition(
                tma_atom_x,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2),
            )
            for c in cutlass.range(chunk_count, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + c, self.w_chunk_bytes + self.x_chunk_bytes
                    )
                cute.copy(
                    tma_atom_w,
                    tWg[(None, rgrp, chunk_start + c)],
                    tWs[(None, c)],
                    tma_bar_ptr=mbar + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            for c in cutlass.range(chunk_count, unroll=1):
                cute.copy(
                    tma_atom_x,
                    tXg[(None, group, chunk_start + c)],
                    tXs[(None, c)],
                    tma_bar_ptr=mbar + c,
                )

        if warp == self.num_mma_warps:
            self.control_warp(
                params, bidx, lane, m, w2_addr, beta_addr, fg_addr, ws_addr, state_addr
            )
            self.refill(params, lane)
        else:
            self.gemm1(storage, mbar, warp, lane, chunk_count)

        # ---- Warp partials were written to shared memory; sum them in K-group order and
        # store the CTA's FP32 partial tile into the current workspace half. ----
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()
        m = Int32(_lds_b64(params + _P_M * 8))
        half = _lds_b64(params + _P_HALF * 8)
        red = storage.sW.data_ptr().toint()
        units = self.rows_cta // 4
        part_base = half + Int64((group * S + split) * self.tg * self.n1 * 4)
        for o in cutlass.range(tidx, self.tg * units, self.threads, unroll=1):
            token = o // units
            u = o % units
            row = rgrp * self.rows_cta + 4 * u
            if group * self.tg + token < m and row < self.n1:
                v = list(_lds_v4_f32(red + (token * self.red_stride + 4 * u) * 4))
                for w in cutlass.range_constexpr(1, cfg.kg):
                    v2 = _lds_v4_f32(red + ((w * self.tg + token) * self.red_stride + 4 * u) * 4)
                    for e in cutlass.range_constexpr(4):
                        v[e] = v[e] + v2[e]
                _st_relaxed_v4_f32(part_base + Int64((token * self.n1 + row) * 4), *v)
        cute.arch.barrier()

        # ---- Tail: the last T ranks are workers. ----
        worker = Int32(_lds_b64(params + _P_RANK * 8)) - (self.num_ctas - self.items)
        sW2 = storage.sW.get_tensor(w2_layout.outer, swizzle=w2_layout.inner)
        if cutlass.const_expr(self.fa_mode):
            self.tail_fa(params, tidx, worker, m, half)
        elif worker >= 0:
            self.tail(
                red,
                sW2,
                storage.w2_full.data_ptr(),
                tma_atom_w2,
                mW2t,
                params,
                tidx,
                warp,
                lane,
                worker,
                m,
                half,
            )

    @cute.jit
    def gemm1(self, storage, mbar, warp: Int32, lane: Int32, chunk_count: Int32):
        """GEMM-1 MMAs of one MMA warp; its partial tile goes to shared memory."""
        cfg = self.cfg
        mt = cfg.mt
        nt = cfg.nt
        wk = warp % cfg.kg
        wmn = warp // cfg.kg
        wn = wmn % self.warps_n
        wm = wmn // self.warps_n
        acc = cute.make_rmem_tensor((mt * nt * 4,), Float32)
        acc.fill(0.0)
        w_base = storage.sW.data_ptr().toint()
        x_base = storage.sX.data_ptr().toint()
        mat = lane >> 3
        r = lane & 7
        a_unit = mat >> 1
        b_unit = mat & 1
        a_rows = [((wm * mt + i) * _M_TILE + (mat & 1) * 8 + r) * _ROW_BYTES for i in range(mt)]
        # Token tiles of this warp, clamped to the CTA's last tile (clamped tiles are not stored).
        tiles = [_min_s32(wn * nt + j, cfg.ntg - 1) for j in range(nt)]
        # ldmatrix.x4 row addresses of each tile pair (an odd last tile loads itself twice).
        b_rows = []
        for p in cutlass.range_constexpr(_ceil_div(nt, 2)):
            second = tiles[min(2 * p + 1, nt - 1)]
            tile = tiles[2 * p] + (mat >> 1) * (second - tiles[2 * p])
            b_rows.append((tile * _N_TILE + r) * _ROW_BYTES)
        for c in cutlass.range(wk, chunk_count, cfg.kg, unroll=1):
            cute.arch.mbarrier_wait(mbar + c, 0)
            a_chunk = w_base + c * self.w_chunk_bytes
            x_chunk = x_base + c * self.x_chunk_bytes
            for ks in cutlass.range_constexpr(_K_CHUNK // 16):
                a = [
                    _ldsm_x4(a_chunk + a_rows[i] + (((ks * 2 + a_unit) ^ r) << 4))
                    for i in range(mt)
                ]
                for p in cutlass.range_constexpr(_ceil_div(nt, 2)):
                    b = _ldsm_x4(x_chunk + b_rows[p] + (((ks * 2 + b_unit) ^ r) << 4))
                    for q in cutlass.range_constexpr(min(2, nt - 2 * p)):
                        j = 2 * p + q
                        for i in cutlass.range_constexpr(mt):
                            o = (i * nt + j) * 4
                            d = _mma_bf16(
                                a[i],
                                b[2 * q],
                                b[2 * q + 1],
                                (acc[o], acc[o + 1], acc[o + 2], acc[o + 3]),
                            )
                            for e in cutlass.range_constexpr(4):
                                acc[o + e] = d[e]
        # All MMA warps must be done with the chunk buffers before they become the reduction
        # buffer (named barrier over the MMA warps only).
        cute.arch.barrier(barrier_id=1, number_of_threads=32 * self.num_mma_warps)
        red = w_base
        for j in cutlass.range_constexpr(nt):
            if wn * nt + j < cfg.ntg:
                for i in cutlass.range_constexpr(mt):
                    for e in cutlass.range_constexpr(4):
                        token = (wn * nt + j) * _N_TILE + 2 * (lane & 3) + (e & 1)
                        row = (wm * mt + i) * _M_TILE + (lane >> 2) + (e >> 1) * 8
                        _sts_f32(
                            red + ((wk * self.tg + token) * self.red_stride + row) * 4,
                            acc[(i * nt + j) * 4 + e],
                        )


class K3KdaFabUmmaKernel(_FabTail):
    """GEMM 1 on ``tcgen05`` UMMA for larger M; same hand-off and tail as :class:`K3KdaFabKernel`.

    CTA ``(g, s)`` owns 128 tokens (UMMA M: ``x`` is operand A) and every ``W_bfa`` row (UMMA N =
    144: operand B, rows past ``N1`` zero-filled by TMA) of the ``s``-th K slice in 64-wide
    chunks. Warp 0 TMA-loads the CTA's whole weight slice before ``griddepcontrol.wait`` and
    streams the activation chunks through a ring; warp 1 allocates tensor memory and issues the
    UMMAs; warp 2 is the control warp; warps 4-7 copy the accumulator (one token per thread) into
    the partial slots.

    Args: see :class:`K3KdaFabKernel` (``cfg.ntg`` must be 16 and ``cfg.r1`` 1).
    """

    _THREADS = 256
    _CONTROL_WARP = 2
    _EPI_WARP0 = 4

    def __init__(
        self,
        cfg: FabConfig,
        n_tiles: int,
        n1: int,
        n2: int,
        k: int,
        num_heads: int,
        use_pdl: bool,
        fa_mode: bool = False,
        warm_w2: bool = True,
    ):
        validate_config(cfg, n_tiles, n1, n2, k, num_heads)
        self._init_common(
            cfg,
            n_tiles,
            n1,
            n2,
            k,
            num_heads,
            use_pdl,
            _geometry(cfg, n_tiles, n2, k),
            fa_mode,
            warm_w2,
        )
        self.mma_tiler = (_UMMA_M, MAX_N1, _K_CHUNK)

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        m: Int32,
        w2_addr: Int64,
        beta_addr: Int64,
        fg_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(
            w1_ptr, cute.make_layout((self.n1, self.k, 1), stride=(self.k, 1, self.n1 * self.k))
        )
        mX = cute.make_tensor(
            x_ptr, cute.make_layout((m, self.k, 1), stride=(self.k, 1, m * self.k))
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.x_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.max_chunks
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mX,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_w2, tma_w2, w2_layout = self.make_w2_tma(w2_ptr)

        @cute.struct
        class SharedStorage:
            w_full: cute.struct.MemRange[cutlass.Int64, self.max_chunks]
            w2_full: cute.struct.MemRange[cutlass.Int64, 1]
            x_full: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            x_empty: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_ready: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_free: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
            red_full: cute.struct.MemRange[cutlass.Int64, 1]
            red_vote: cute.struct.MemRange[cutlass.Int32, 2]
            params: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, _NUM_PARAMS], 16]
            # Weight chunks, then the activation ring; the tail reuses both.
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, max(cute.cosize(b_layout.outer), self.sw_elems)
                ],
                1024,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, max(cute.cosize(a_layout.outer), self.sx_elems)
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            a_layout,
            b_layout,
            tma_atom_w2,
            tma_w2,
            w2_layout,
            m,
            w2_addr,
            beta_addr,
            fg_addr,
            ws_addr,
            state_addr,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(self._THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        tma_atom_w2: cute.CopyAtom,
        mW2t: cute.Tensor,
        w2_layout: cute.ComposedLayout,
        m: Int32,
        w2_addr: Int64,
        beta_addr: Int64,
        fg_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
    ):
        S = self.cfg.s
        stages = self.x_stages
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        group = bidx % self.groups
        split = bidx // self.groups
        base_chunks = self.num_k_chunks // S
        rem_chunks = self.num_k_chunks % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        w_full = storage.w_full.data_ptr()
        x_full = storage.x_full.data_ptr()
        x_empty = storage.x_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        tmem_ready = storage.tmem_ready.data_ptr()
        tmem_free = storage.tmem_free.data_ptr()
        tmem_holding = storage.tmem_holding.data_ptr()
        params = storage.params.data_ptr().toint()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.max_chunks):
                    cute.arch.mbarrier_init(w_full + i, 1)
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(x_full + i, 1)
                    cute.arch.mbarrier_init(x_empty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, 4)
                cute.arch.mbarrier_init(storage.w2_full.data_ptr(), 1)
                cute.arch.mbarrier_init(storage.red_full.data_ptr(), 1)
                vote = storage.red_vote.data_ptr().toint()
                _sts_b32(vote, 0)
                _sts_b32(vote + 4, 0)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp == 0:
            # Weight first (independent of the previous grid), then the activation ring.
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            thr_mma = tiled_mma.get_slice(0)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            for c in cutlass.range(chunk_count, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(w_full + c, self.w_chunk_bytes)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, chunk_start + c, 0)],
                    tBsB[(None, c)],
                    tma_bar_ptr=w_full + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            # Dependents may launch only once this CTA holds its tensor memory: a dependent CTA
            # that took the columns first and then waited on this grid would deadlock it.
            cute.arch.mbarrier_wait(tmem_ready, 0)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
                _griddepcontrol_wait()
            for c in cutlass.range(chunk_count, unroll=1):
                stage = c % stages
                if c >= stages:
                    cute.arch.mbarrier_wait(x_empty + stage, ((c // stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(x_full + stage, self.x_chunk_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, group, chunk_start + c, 0)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=x_full + stage,
                )
        elif warp == 1:
            cute.arch.alloc_tmem(_UMMA_TMEM_COLS, tmem_holding)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for c in cutlass.range(chunk_count, unroll=1):
                stage = c % stages
                cute.arch.mbarrier_wait(w_full + c, 0)
                cute.arch.mbarrier_wait(x_full + stage, (c // stages) & 1)
                for kp in cutlass.range_constexpr(num_kphases):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, stage)],
                        tCrB[(None, None, kp, c)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(x_empty + stage)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
            cute.arch.mbarrier_wait(tmem_free, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, _UMMA_TMEM_COLS)
        elif warp == self._CONTROL_WARP:
            self.control_warp(
                params, bidx, lane, m, w2_addr, beta_addr, fg_addr, ws_addr, state_addr
            )
            self.refill(params, lane)
        elif warp >= self._EPI_WARP0:
            # TMEM lane = token: warp 4 + i reads lanes [32 i, 32 i + 32), all 144 columns.
            cute.arch.mbarrier_wait(tmem_ready, 0)
            tmem_base = _lds_b32(tmem_holding.toint())
            token = (warp - self._EPI_WARP0) * 32 + lane
            lane_base = tmem_base + (((warp - self._EPI_WARP0) * 32) << 16)
            # The accumulator goes through shared memory (the drained weight chunks) so the
            # partial-tile stores below are whole lines: one token row of N1 floats per thread.
            stage = storage.sB.data_ptr().toint() + token * (_UMMA_STAGE_PITCH * 4)
            cute.arch.mbarrier_wait(acc_full, 0)
            for j in cutlass.range_constexpr(MAX_N1 // 16):
                v = _tmem_ld16(lane_base + j * 16)
                for q in cutlass.range_constexpr(4):
                    if cutlass.const_expr(j * 16 + q * 4 < self.n1):
                        _sts_v4_f32(
                            stage + (j * 16 + q * 4) * 4,
                            v[4 * q],
                            v[4 * q + 1],
                            v[4 * q + 2],
                            v[4 * q + 3],
                        )
            _tmem_fence_before_sync()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_free)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()
        # The staged tile -> the CTA's partial slot, by every thread.
        m = Int32(_lds_b64(params + _P_M * 8))
        half = _lds_b64(params + _P_HALF * 8)
        units = self.n1 // 4
        rows = m - group * _UMMA_M
        if rows > _UMMA_M:
            rows = Int32(_UMMA_M)
        part_base = half + Int64(((group * S + split) * _UMMA_M) * self.n1 * 4)
        stage0 = storage.sB.data_ptr().toint()
        for o in cutlass.range(tidx, rows * units, self._THREADS, unroll=4):
            t = o // units
            u = o % units
            w = _lds_v4_f32(stage0 + (t * _UMMA_STAGE_PITCH + 4 * u) * 4)
            _st_relaxed_v4_f32(part_base + Int64((t * self.n1 + 4 * u) * 4), *w)
        cute.arch.barrier()
        m = Int32(_lds_b64(params + _P_M * 8))
        half = _lds_b64(params + _P_HALF * 8)
        rank = Int32(_lds_b64(params + _P_RANK * 8))
        worker = rank - (self.num_ctas - self.items)
        region = storage.sB.data_ptr().toint()
        sW2 = storage.sB.get_tensor(w2_layout.outer, swizzle=w2_layout.inner)
        if cutlass.const_expr(self.cfg.staged):
            # W_fb streams in while every CTA reduces its share of the partials.
            if worker >= 0:
                if warp == 0:
                    self.issue_w2(sW2, storage.w2_full.data_ptr(), tma_atom_w2, mW2t, worker)
            if cutlass.const_expr(self.cfg.staged == 2):
                self.reduce_staged_bulk(
                    region + self.l1_off,
                    storage.red_full.data_ptr(),
                    storage.red_vote.data_ptr().toint(),
                    params,
                    tidx,
                    rank,
                    m,
                    half,
                )
            else:
                self.reduce_staged(params, tidx, rank, m, half)
        if cutlass.const_expr(self.fa_mode):
            self.tail_fa(params, tidx, worker, m, half)
        elif worker >= 0:
            self.tail(
                region,
                sW2,
                storage.w2_full.data_ptr(),
                tma_atom_w2,
                mW2t,
                params,
                tidx,
                warp,
                lane,
                worker,
                m,
                half,
            )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _geometry(cfg: FabConfig, n_tiles: int, n2: int, k: int) -> dict:
    tg = cfg.ntg * _N_TILE
    groups = _ceil_div(n_tiles, cfg.ntg)
    num_k_chunks = k // _K_CHUNK
    max_chunks = _ceil_div(num_k_chunks, cfg.s)
    token_blocks = _ceil_div(n_tiles, cfg.ntb)
    row_blocks = _ceil_div(n2 // _M_TILE, cfg.mtb)
    w2_smem_bytes = cfg.mtb * _M_TILE * FA_DIM * 2
    w2_box_rows = min(128, cfg.mtb * _M_TILE)
    fa_smem_bytes = cfg.ntb * _N_TILE * FA_DIM * 2
    out_smem_bytes = cfg.ntb * _N_TILE * (cfg.mtb * _M_TILE + 8) * 2
    geo = {
        "tg": tg,
        "groups": groups,
        "num_k_chunks": num_k_chunks,
        "max_chunks": max_chunks,
        "token_blocks": token_blocks,
        "row_blocks": row_blocks,
        "items": token_blocks * row_blocks,
        "num_ctas": groups * cfg.r1 * cfg.s,
        "w2_smem_bytes": w2_smem_bytes,
        "w2_box_rows": w2_box_rows,
        "w2_boxes": (cfg.mtb * _M_TILE // w2_box_rows) * (FA_DIM // _K_CHUNK),
        "workspace_bytes": groups * cfg.s * tg * MAX_N1 * 4
        + bool(cfg.staged) * groups * tg * FA_DIM * 2,
        # Bulk level 1: the share's partial rows of every K split, after the tail's operands.
        "l1_off": _ceil_div(w2_smem_bytes + fa_smem_bytes + out_smem_bytes, 128) * 128,
        "l1_bytes": (cfg.staged == 2) * cfg.s * _ceil_div(tg, cfg.s) * MAX_N1 * 4,
    }
    if cfg.umma:
        w_chunk_bytes = MAX_N1 * _ROW_BYTES
        x_chunk_bytes = _UMMA_M * _ROW_BYTES
        sw_bytes = max_chunks * w_chunk_bytes
        budget = _SMEM_CAPACITY - sw_bytes - 4096
        x_stages = max(1, min(max_chunks, budget // x_chunk_bytes))
        sx_bytes = x_stages * x_chunk_bytes
        alias_need = max(
            geo["l1_off"] + geo["l1_bytes"] if cfg.staged == 2 else 0,
            w2_smem_bytes + fa_smem_bytes + out_smem_bytes,
            _UMMA_M * _UMMA_STAGE_PITCH * 4,
        )
        sw_bytes = max(sw_bytes, _ceil_div(max(alias_need - sx_bytes, 0), 1024) * 1024)
        geo.update(
            rows_cta=MAX_N1,
            num_mma_warps=0,
            num_warps=K3KdaFabUmmaKernel._THREADS // 32,
            threads=K3KdaFabUmmaKernel._THREADS,
            x_stages=x_stages,
            w_chunk_bytes=w_chunk_bytes,
            x_chunk_bytes=x_chunk_bytes,
            sw_elems=sw_bytes // 2,
            sx_elems=sx_bytes // 2,
            smem_bytes=sw_bytes
            + sx_bytes
            + 8 * (max_chunks + 2 * x_stages + 4)
            + 8 * _NUM_PARAMS
            + 2048,
        )
        return geo
    mtc = _N1_TILES // cfg.r1
    rows_cta = mtc * _M_TILE
    warps_m = mtc // cfg.mt
    warps_n = _ceil_div(cfg.ntg, cfg.nt)
    num_mma_warps = warps_m * warps_n * cfg.kg
    red_stride = rows_cta + 4
    w_chunk_bytes = rows_cta * _ROW_BYTES
    x_chunk_bytes = tg * _ROW_BYTES
    red_bytes = cfg.kg * tg * red_stride * 4
    sw_bytes = max_chunks * w_chunk_bytes
    sx_bytes = max_chunks * x_chunk_bytes
    alias_need = max(red_bytes, w2_smem_bytes + fa_smem_bytes + out_smem_bytes)
    sw_bytes = max(sw_bytes, _ceil_div(max(alias_need - sx_bytes, 0), 1024) * 1024)
    geo.update(
        mtc=mtc,
        rows_cta=rows_cta,
        warps_m=warps_m,
        warps_n=warps_n,
        num_mma_warps=num_mma_warps,
        # One control warp (ticket, parameters, workspace refill) besides the MMA warps.
        num_warps=num_mma_warps + 1,
        threads=32 * (num_mma_warps + 1),
        red_stride=red_stride,
        w_chunk_bytes=w_chunk_bytes,
        x_chunk_bytes=x_chunk_bytes,
        sw_elems=sw_bytes // 2,
        sx_elems=sx_bytes // 2,
        smem_bytes=sw_bytes + sx_bytes + 8 * max_chunks + 8 * _NUM_PARAMS + 2048,
    )
    return geo


def validate_config(cfg: FabConfig, n_tiles: int, n1: int, n2: int, k: int, num_heads: int) -> None:
    """Raise ``ValueError`` if ``cfg`` cannot run the given problem."""
    if not 1 <= n_tiles <= MAX_M // _N_TILE:
        raise ValueError(f"n_tiles={n_tiles} must be in [1, {MAX_M // _N_TILE}]")
    if n1 % 8 or not FA_DIM < n1 <= MAX_N1 or not 1 <= num_heads <= n1 - FA_DIM:
        raise ValueError(f"N1={n1} / num_heads={num_heads} unsupported")
    if n2 % _M_TILE or k % _K_CHUNK:
        raise ValueError(f"N2={n2} must be a multiple of 16 and K={k} of 64")
    if cfg.umma:
        if cfg.ntg != _UMMA_M // _N_TILE or cfg.r1 != 1:
            raise ValueError(f"UMMA tiling needs ntg={_UMMA_M // _N_TILE} and r1=1: {cfg}")
    elif cfg.staged:
        raise ValueError(f"the staged tail needs the UMMA GEMM 1: {cfg}")
    elif cfg.r1 not in (1, 3, 9) or (_N1_TILES // cfg.r1) % cfg.mt:
        raise ValueError(f"r1={cfg.r1} / mt={cfg.mt} invalid")
    elif cfg.ntg < 1 or cfg.nt < 1 or cfg.nt > cfg.ntg or cfg.kg < 1:
        raise ValueError(f"invalid token tiling {cfg}")
    if cfg.ntb < 1 or cfg.ntg % cfg.ntb or cfg.mtb < 1:
        raise ValueError(f"invalid tail tiling {cfg}")
    if not 1 <= cfg.s <= k // _K_CHUNK:
        raise ValueError(f"s={cfg.s} invalid for K={k}")
    geo = _geometry(cfg, n_tiles, n2, k)
    if (cfg.mtb * _M_TILE) % min(128, cfg.mtb * _M_TILE):
        raise ValueError(f"{cfg}: the W_fb block must be whole 128-row boxes")
    if cfg.s * cfg.r1 > 32:
        raise ValueError(f"{cfg}: more than 32 producer slots per token group")
    if geo["num_warps"] > 16:
        raise ValueError(f"{cfg} needs {geo['num_warps']} warps")
    if geo["items"] > geo["num_ctas"]:
        raise ValueError(f"{cfg}: {geo['items']} tail items > {geo['num_ctas']} CTAs")
    if geo["smem_bytes"] > _SMEM_CAPACITY:
        raise ValueError(f"{cfg} needs {geo['smem_bytes']} B of shared memory")
    if geo["workspace_bytes"] > _WS_HALF_BYTES:
        raise ValueError(f"{cfg} needs {geo['workspace_bytes']} B of workspace")


def _base_config(n_tiles: int) -> FabConfig:
    if n_tiles == 1:
        return FabConfig(ntg=1, r1=3, s=9, mt=1, nt=1, kg=4, ntb=1, mtb=2)
    if n_tiles == 2:
        return FabConfig(ntg=2, r1=3, s=9, mt=1, nt=2, kg=4, ntb=1, mtb=4)
    if n_tiles <= 4:
        return FabConfig(ntg=n_tiles, r1=3, s=9, mt=3, nt=2, kg=4, ntb=1, mtb=8)
    if n_tiles <= 8:
        return FabConfig(ntg=n_tiles, r1=3, s=9, mt=3, nt=4, kg=2, ntb=1, mtb=16)
    umma = {"ntg": _UMMA_M // _N_TILE, "r1": 1, "mt": 1, "nt": 1, "kg": 1, "umma": True}
    if n_tiles <= 16:
        return FabConfig(s=28, ntb=1 if n_tiles <= 14 else 2, mtb=24, **umma)
    return FabConfig(s=14, ntb=1 if n_tiles <= 28 else 2, mtb=48, **umma)


def _staged_config(cfg: FabConfig, n_tiles: int, n2: int, k: int, staged: int) -> FabConfig:
    """``cfg`` with the staged tail and the worker tiling with the least ``f_b`` work per worker
    (then the most tokens per worker) that fits the CTAs; ``cfg`` itself if none fits."""
    best = None
    for ntb in (1, 2, 4):
        for mtb in (8, 16, 24, 48):
            cand = dataclasses.replace(cfg, staged=staged, ntb=ntb, mtb=mtb)
            try:
                validate_config(cand, n_tiles, MAX_N1, n2, k, 1)
            except ValueError:
                continue
            key = (ntb * mtb, -ntb)
            if best is None or key < best[0]:
                best = (key, cand)
    return cfg if best is None else best[1]


@functools.cache
def default_config(
    n_tiles: int, n2: int = 768, k: int = 7168, staged: Optional[int] = None
) -> FabConfig:
    """Tiling for ``ceil(M / 8) = n_tiles``, tuned on GB300 next to the concurrent ``qkvg`` GEMM.

    The concurrent ``qkvg`` GEMM (``k3_fp8w_lowm_gemm``, clusters of 4-5 CTAs) occupies 96-120 of
    the 152 SMs; more than ~28 CTAs here break its single-wave cluster placement. So the grid is
    27 CTAs (``mma.sync``, M <= 64) or 28 CTAs (UMMA, larger M), with at most one tail item each.
    Other shapes keep the tiling with at most ``k / 64`` K splits and enough ``W_fb`` rows per
    tail item to cover ``n2`` with the CTAs available. With :data:`STAGED_TAIL` the UMMA tilings
    use the staged (two-level) tail (``staged`` overrides it; f_a mode passes ``False``).
    """
    cfg = _base_config(n_tiles)
    cfg = dataclasses.replace(cfg, s=min(cfg.s, k // _K_CHUNK))
    staged = STAGED_TAIL if staged is None else int(staged)
    if cfg.umma and staged:
        return _staged_config(cfg, n_tiles, n2, k, staged)
    geo = _geometry(cfg, n_tiles, n2, k)
    row_blocks = max(1, geo["num_ctas"] // geo["token_blocks"])
    mtb = max(cfg.mtb, _ceil_div(n2 // _M_TILE, row_blocks))
    if mtb > 128 // _M_TILE:
        mtb = _ceil_div(mtb, 128 // _M_TILE) * (128 // _M_TILE)
    return dataclasses.replace(cfg, mtb=mtb)


@functools.cache
def _compiled(
    cfg: FabConfig,
    n_tiles: int,
    n1: int,
    n2: int,
    k: int,
    num_heads: int,
    use_pdl: bool,
    fa_mode: bool = False,
    warm_w2: bool = True,
):
    kernel_cls = K3KdaFabUmmaKernel if cfg.umma else K3KdaFabKernel
    kernel = kernel_cls(cfg, n_tiles, n1, n2, k, num_heads, use_pdl, fa_mode, warm_w2)
    dummy = 1 << 20
    ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    addr = Int64(dummy)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel, ptr, ptr, ptr, Int32(8 * n_tiles), addr, addr, addr, addr, addr, stream
    )


_STATE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-device ``(state words, sentinel-filled partial workspace)``.

    Created outside CUDA graph capture (a capture would allocate it from the graph's private
    pool); :func:`can_use_k3_kda_fab` rejects capturing callers until it exists.
    """
    index = _device_index(device)
    state = _STATE.get(index)
    if state is None:
        words = torch.zeros(_STATE_WORDS, dtype=torch.int64, device=device)
        ws = torch.full((2 * _WS_HALF_BYTES // 4,), _SENTINEL, dtype=torch.int32, device=device)
        state = (words, ws)
        _STATE[index] = state
    return state


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    # SM100 / SM103 (Blackwell) and SM107 (Rubin).
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def _check_inputs(x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int):
    for name, t in (("x", x), ("w_bfa", w_bfa), ("w_fb", w_fb)):
        if t.dtype != torch.bfloat16 or t.dim() != 2 or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous 2D bfloat16 tensor")
        if not t.is_cuda or t.device != x.device or t.data_ptr() % 16:
            raise ValueError(f"{name} must be a 16-byte aligned tensor on x's CUDA device")
    m, k = x.shape
    n1 = w_bfa.shape[0]
    n2 = w_fb.shape[0]
    if w_bfa.shape[1] != k or w_fb.shape[1] != FA_DIM:
        raise ValueError(
            f"shape mismatch: x {tuple(x.shape)}, w_bfa {tuple(w_bfa.shape)}, "
            f"w_fb {tuple(w_fb.shape)}"
        )
    if not 1 <= m <= MAX_M:
        raise ValueError(f"M={m} must be in [1, {MAX_M}]")
    if n1 % 8 or not FA_DIM < n1 <= MAX_N1 or not 1 <= num_heads <= n1 - FA_DIM:
        raise ValueError(f"N1={n1} / num_heads={num_heads} unsupported")
    if n2 % _M_TILE or k % _K_CHUNK:
        raise ValueError(f"N2={n2} must be a multiple of 16 and K={k} of 64")


def run_k3_kda_fab(
    x: torch.Tensor,
    w_bfa: torch.Tensor,
    w_fb: torch.Tensor,
    num_heads: int,
    beta: torch.Tensor,
    forget_gate: torch.Tensor,
    *,
    config: FabConfig | None = None,
    use_pdl: bool = True,
) -> None:
    """Launch into ``beta`` (``bf16 [M, H]``) and ``forget_gate`` (``bf16 [M, N2]``)."""
    m, k = x.shape
    n_tiles = _ceil_div(m, _N_TILE)
    cfg = config if config is not None else default_config(n_tiles, w_fb.shape[0], k)
    compiled = _compiled(cfg, n_tiles, w_bfa.shape[0], w_fb.shape[0], k, num_heads, use_pdl)
    words, ws = _device_state(x.device)
    compiled(
        make_ptr(cutlass.BFloat16, w_bfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, w_fb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(m),
        Int64(w_fb.data_ptr()),
        Int64(beta.data_ptr()),
        Int64(forget_gate.data_ptr()),
        Int64(ws.data_ptr()),
        Int64(words.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


def run_k3_kda_fab_fa(
    x: torch.Tensor,
    w_bfa: torch.Tensor,
    num_heads: int,
    bfa: torch.Tensor,
    w_fb: torch.Tensor | None = None,
    *,
    config: FabConfig | None = None,
    use_pdl: bool = True,
) -> None:
    """f_a mode: launch into ``bfa`` (``bf16 [M, N1]`` contiguous): ``bf16(x @ W_bfa^T)``, bitwise
    the ``f_a`` and ``beta`` of :func:`run_k3_kda_fab` with the same ``config``. ``w_fb``
    (``bf16 [N2, 128]``, optional) is only warmed in L2 for the decode that applies it; ``N2``
    (768 without it) also selects the default tiling like in the full kernel."""
    m, k = x.shape
    n_tiles = _ceil_div(m, _N_TILE)
    n2 = w_fb.shape[0] if w_fb is not None else 768
    cfg = config if config is not None else default_config(n_tiles, n2, k, staged=False)
    compiled = _compiled(
        cfg, n_tiles, w_bfa.shape[0], n2, k, num_heads, use_pdl, True, w_fb is not None
    )
    words, ws = _device_state(x.device)
    w1 = make_ptr(cutlass.BFloat16, w_bfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    w2_addr = w_fb.data_ptr() if w_fb is not None else w_bfa.data_ptr()
    compiled(
        w1,
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        # The W_fb TMA operand is unused in f_a mode: any 16-byte aligned buffer of at least
        # N2 x 128 BF16 (W_bfa without w_fb).
        w1,
        Int32(m),
        Int64(w2_addr),
        Int64(bfa.data_ptr()),
        Int64(bfa.data_ptr()),
        Int64(ws.data_ptr()),
        Int64(words.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


@torch.library.custom_op("trtllm::k3_kda_fab_fa", mutates_args=())
def k3_kda_fab_fa(
    x: torch.Tensor, w_bfa: torch.Tensor, num_heads: int, w_fb: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``bfa = bf16(x @ w_bfa^T)`` (``bf16 [M, N1]``) by the fused fab kernel's GEMM 1 and
    reduction (f_a mode): columns ``[0, 128)`` are ``f_a``, ``[128, 128 + num_heads)`` ``beta``,
    bitwise as inside :func:`k3_kda_fab_proj` (same ``W_fb`` row count). ``w_fb`` is only warmed
    in L2. Accepts what :func:`can_use_k3_kda_fab_fa` does."""
    _check_fa_inputs(x, w_bfa, num_heads, w_fb)
    bfa = torch.empty((x.shape[0], w_bfa.shape[0]), dtype=torch.bfloat16, device=x.device)
    run_k3_kda_fab_fa(x, w_bfa, num_heads, bfa, w_fb)
    return bfa


@k3_kda_fab_fa.register_fake
def _(
    x: torch.Tensor, w_bfa: torch.Tensor, num_heads: int, w_fb: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_bfa.shape[0]))


def _check_fa_inputs(
    x: torch.Tensor, w_bfa: torch.Tensor, num_heads: int, w_fb: torch.Tensor | None = None
) -> None:
    """:func:`_check_inputs` with an optional ``W_fb`` operand."""
    if w_fb is not None:
        _check_inputs(x, w_bfa, w_fb, num_heads)
        return
    for name, t in (("x", x), ("w_bfa", w_bfa)):
        if t.dtype != torch.bfloat16 or t.dim() != 2 or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous 2D bfloat16 tensor")
        if not t.is_cuda or t.device != x.device or t.data_ptr() % 16:
            raise ValueError(f"{name} must be a 16-byte aligned tensor on x's CUDA device")
    m, k = x.shape
    n1 = w_bfa.shape[0]
    if w_bfa.shape[1] != k or not 1 <= m <= MAX_M or k % _K_CHUNK:
        raise ValueError(f"unsupported shapes: x {tuple(x.shape)}, w_bfa {tuple(w_bfa.shape)}")
    if n1 % 8 or not FA_DIM < n1 <= MAX_N1 or not 1 <= num_heads <= n1 - FA_DIM:
        raise ValueError(f"N1={n1} / num_heads={num_heads} unsupported")


def can_use_k3_kda_fab_fa(
    x: torch.Tensor, w_bfa: torch.Tensor, num_heads: int, w_fb: torch.Tensor | None = None
) -> bool:
    """Whether :func:`k3_kda_fab_fa` accepts these inputs (non-raising check)."""
    if not x.is_cuda or not _is_supported_device(x.device.index):
        return False
    if torch.cuda.is_current_stream_capturing() and _device_index(x.device) not in _STATE:
        return False
    try:
        _check_fa_inputs(x, w_bfa, num_heads, w_fb)
        n_tiles = _ceil_div(x.shape[0], _N_TILE)
        k = x.shape[1]
        n2 = w_fb.shape[0] if w_fb is not None else 768
        validate_config(
            default_config(n_tiles, n2, k, staged=False), n_tiles, w_bfa.shape[0], n2, k, num_heads
        )
    except ValueError:
        return False
    return True


@torch.library.custom_op("trtllm::k3_kda_fab_proj", mutates_args=())
def k3_kda_fab_proj(
    x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(beta, forget_gate)`` of the KDA ``[f_a | b] -> f_b`` projection in one launch.

    Args:
        x: ``bf16 [M, K]``, ``1 <= M <= 512``, ``K`` a multiple of 64.
        w_bfa: ``bf16 [N1, K]``: ``f_a_proj`` rows ``[0, 128)``, ``b_proj`` rows
            ``[128, 128 + num_heads)``, zero padding (``N1`` a multiple of 8, at most 144).
        w_fb: ``bf16 [N2, 128]`` (``f_b_proj``), ``N2`` a multiple of 16.
        num_heads: Rows of ``b_proj``.

    Returns:
        ``beta`` ``bf16 [M, num_heads]`` and ``forget_gate`` ``bf16 [M, N2]``, equal to
        ``bfa = bf16(x @ w_bfa^T)``, ``bfa[:, 128:128 + num_heads]`` and
        ``bf16(bfa[:, :128] @ w_fb^T)`` (FP32 accumulation).
    """
    _check_inputs(x, w_bfa, w_fb, num_heads)
    beta = torch.empty((x.shape[0], num_heads), dtype=torch.bfloat16, device=x.device)
    forget_gate = torch.empty((x.shape[0], w_fb.shape[0]), dtype=torch.bfloat16, device=x.device)
    run_k3_kda_fab(x, w_bfa, w_fb, num_heads, beta, forget_gate)
    return beta, forget_gate


@k3_kda_fab_proj.register_fake
def _(
    x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((x.shape[0], num_heads)),
        x.new_empty((x.shape[0], w_fb.shape[0])),
    )


def can_use_k3_kda_fab(
    x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
) -> bool:
    """Whether :func:`k3_kda_fab_proj` accepts these inputs (non-raising check)."""
    if not x.is_cuda or not _is_supported_device(x.device.index):
        return False
    if torch.cuda.is_current_stream_capturing() and _device_index(x.device) not in _STATE:
        return False
    try:
        _check_inputs(x, w_bfa, w_fb, num_heads)
        n_tiles = _ceil_div(x.shape[0], _N_TILE)
        n2, k = w_fb.shape[0], x.shape[1]
        validate_config(default_config(n_tiles, n2, k), n_tiles, w_bfa.shape[0], n2, k, num_heads)
    except ValueError:
        return False
    return True


__all__ = [
    "FA_DIM",
    "FabConfig",
    "K3KdaFabKernel",
    "MAX_M",
    "MAX_N1",
    "can_use_k3_kda_fab",
    "can_use_k3_kda_fab_fa",
    "default_config",
    "k3_kda_fab_fa",
    "k3_kda_fab_proj",
    "run_k3_kda_fab",
    "run_k3_kda_fab_fa",
    "validate_config",
]
