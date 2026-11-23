# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE routing for decode batches (SM100/SM103/SM107): top-16 + ``moe_sort`` in one launch.

``k3_route_tail`` replaces ``noaux_tc_op`` -> ``moe_sort`` (and, given split-K partial logits,
the split-K reduction before them) by one launch on a cluster of eight small CTAs, so the
routing occupies few SMs next to the side-stream kernels it runs with.

* Input: FP32 router logits ``[M, 896]`` (``1 <= M <= 2048``), or ``S`` split-K partials
  ``[S, R, 896]`` (``R >= M``) that are summed in split order.
* Top-k, bit-identical to ``noaux_tc_op`` on the same logits (selection, order, weights). A
  team of ``G`` warps routes one token, lane ``g`` of the team owning experts
  ``j * 32 G + g``. Each lane forms the order-preserving bits of ``sigmoid(logit) + bias`` (the
  high word of ``noaux_tc_op``'s 64-bit key, whose low word ``65535 - expert`` breaks ties
  toward smaller experts). In each warp the 16th largest lane maximum is a lower bound of the
  token's 16th largest score; the team keeps the largest of these bounds. Every expert at or
  above it (usually about 20 per token) is a candidate: warps compact theirs into shared
  memory and rank them by their full keys. If a warp has more than 32 (64 for one-warp teams)
  candidates (heavy ties), its team's first warp instead runs 16 exact arg-max rounds over all
  896 keys. The weights are the ``sigmoid`` scores renormalized like ``noaux_tc_op``.
* Sort (``moe_sort`` metadata of one expert-parallel rank): CTA ``c`` of the cluster owns a
  contiguous block of at most 32 tokens and marks, per local expert, a token bitmask in shared
  memory, then stores it into every CTA of the cluster (distributed shared memory). After a
  cluster barrier each CTA scans the per-expert tile counts (``ceil(count / 128)`` tiles in
  local-expert order) and writes the rows of its own tokens: rows of an expert follow the token
  order, which is what ``moe_sort``'s single-CTA kernels produce (its cluster kernel orders rows
  by atomics). CTA 0 writes the tile arrays.
* ``256 < M <= 2048`` (:class:`K3RouteWideKernel`): independent CTAs of 8-32 tokens run the
  same top-k; each publishes per-expert token masks (and the top-k slot of each masked token)
  to a persistent workspace, and the last CTA to arrive (arrival counter) writes the same
  ``moe_sort`` outputs, rows again in ascending token order.

The cluster kernels keep no state between launches; the multi-CTA kernel resets its arrival
counter before it exits. Either way a captured CUDA graph replays them as is.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_fused_router import (
    NUM_EXPERTS,
    TOP_K,
    _asm,
    _atom_add_relaxed_gpu,
    _div_rn_f32,
    _fence_acq_rel_gpu,
    _fv,
    _iv,
    _lds_b32,
    _lds_f32,
    _lv,
    _popc,
    _red_shared_or,
    _renormalize,
    _st_global_b32,
    _st_global_f32,
    _st_relaxed_b32,
    _sts_b32,
    _sts_v4_f32,
    _unit_scale_fast_path,
    _unpack,
    _warp_inclusive_sum,
    _warp_sum_f32,
    sort_buffer_sizes,
)

#: Largest token count served by the tail.
MAX_M = 256
#: The cluster kernel picks the candidates on ``tanh.approx`` scores and evaluates ``noaux_tc``'s
#: exact sigmoid only for them (the accurate ``tanhf`` of all 896 experts dominates the top-k).
#: The selection stays bit-identical: see :meth:`K3RouteTailKernel.select_token`.
#: Score margin of that approximate candidate pass, at least twice the largest difference between
#: an approximate and an exact score: ``tanh.approx.f32`` is within 2^-10.987 of tanh, i.e. the
#: sigmoid within 2.5e-4, plus a few FP32 roundings of values below 2.
APPROX_MARGIN = 6.0e-4
#: Largest cluster (portable cluster size).
MAX_CTAS = 8
#: Tokens per CTA are capped by the 32-bit token mask of the sort.
_MAX_TOKENS_PER_CTA = 32
#: Expert slots per lane: 7 blocks of 128 experts x 4 consecutive experts.
_BLOCKS = NUM_EXPERTS // 128
_SLOTS = _BLOCKS * 4
#: Bytes of one compacted candidate: ``{65535 - expert, key high word, sigmoid bits, 0}``.
_CAND_BYTES = 16
#: Tile size of the CuteDSL MoE (``tile_tokens_dim``).
TILE = 128

assert NUM_EXPERTS % 128 == 0


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
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
def _bar_sync_named(barrier_id, num_threads: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(barrier_id, loc=loc, ip=ip)],
        f"bar.sync $0, {num_threads};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_nc_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _score_bits(score, *, loc=None, ip=None):
    """Order-preserving bits of an FP32 score (the high word of ``noaux_tc``'s key)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(score, loc=loc, ip=ip)],
            "{\n.reg .b32 b, m;\nmov.b32 b, $1;\nshr.s32 m, b, 31;\n"
            "or.b32 m, m, 0x80000000;\nxor.b32 $0, b, m;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _score_value(bits, *, loc=None, ip=None):
    """Inverse of :func:`_score_bits`: the FP32 score of order-preserving bits."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(bits, loc=loc, ip=ip)],
            "{\n.reg .b32 m;\nshr.s32 m, $1, 31;\nnot.b32 m, m;\n"
            "or.b32 m, m, 0x80000000;\nxor.b32 m, $1, m;\nmov.b32 $0, m;\n}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _tanh_approx(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_fv(x, loc=loc, ip=ip)], "tanh.approx.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


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
def _max_u32(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_16th_largest_u32(value, lane, *, loc=None, ip=None):
    """16th largest of the 32 lane values (unsigned), in every lane.

    Bitonic-sorts the lower 16 lanes descending and the upper 16 ascending; the pairwise maxima
    of lanes ``i`` and ``i + 16`` are then the 16 largest values, whose minimum is the answer.
    """
    body = "{\n.reg .b32 x, o, mx, mn, t, u;\n.reg .pred pj, pk, km;\nmov.b32 x, $1;\n"
    k = 2
    while k <= 16:
        j = k // 2
        while j >= 1:
            body += f"shfl.sync.bfly.b32 o, x, {j}, 0x1f, 0xffffffff;\n"
            body += "max.u32 mx, x, o;\nmin.u32 mn, x, o;\n"
            # Keep the maximum iff the lane is the lower one of its pair in a descending block.
            body += f"and.b32 t, $2, {j};\nsetp.eq.u32 pj, t, 0;\n"
            body += f"and.b32 u, $2, {k};\nsetp.eq.u32 pk, u, 0;\nxor.pred km, pj, pk;\n"
            body += "selp.b32 x, mn, mx, km;\n"
            j //= 2
        k *= 2
    body += "shfl.sync.bfly.b32 o, x, 16, 0x1f, 0xffffffff;\nmax.u32 x, x, o;\n"
    body += "redux.sync.min.u32 $0, x, 0xffffffff;\n}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            body,
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_ge_u32(values, threshold, *, loc=None, ip=None):
    """Number of ``values`` (unsigned) at or above ``threshold``."""
    n = len(values)
    body = "{\n.reg .pred p;\nmov.u32 $0, 0;\n"
    for i in range(n):
        body += f"setp.ge.u32 p, ${2 + i}, $1;\n@p add.u32 $0, $0, 1;\n"
    body += "}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(threshold, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in values],
            body,
            "=&r,r" + ",r" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_candidates(addr, his, los, sigs, threshold, *, loc=None, ip=None):
    """Append the slots with ``hi >= threshold`` as 16-byte records at ``addr``; returns the end.

    Record: ``{65535 - expert, hi, sigmoid bits, 0}`` (``ld.shared.b64`` of its first eight
    bytes is ``noaux_tc``'s key).
    """
    n = len(his)
    body = "{\n.reg .pred p;\n.reg .b32 z;\nmov.b32 z, 0;\nmov.u32 $0, $1;\n"
    for i in range(n):
        h, lo, s = 3 + i, 3 + n + i, 3 + 2 * n + i
        body += (
            f"setp.ge.u32 p, ${h}, $2;\n"
            f"@p st.shared.v4.b32 [$0], {{${lo}, ${h}, ${s}, z}};\n"
            "@p add.u32 $0, $0, 16;\n"
        )
    body += "}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(threshold, loc=loc, ip=ip)]
            + [_iv(v, loc=loc, ip=ip) for v in his]
            + [_iv(v, loc=loc, ip=ip) for v in los]
            + [_fv(v, loc=loc, ip=ip) for v in sigs],
            body,
            "=&r,r,r" + ",r" * (2 * n) + ",f" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_greater4(acc, addr, key, *, loc=None, ip=None):
    """``acc`` plus how many of the four candidate keys at ``addr`` (16-byte records) exceed
    ``key``."""
    body = "{\n.reg .b64 a, b, c, d;\n.reg .pred p, q, r, s;\n"
    body += "ld.shared.b64 a, [$2];\nld.shared.b64 b, [$2+16];\n"
    body += "ld.shared.b64 c, [$2+32];\nld.shared.b64 d, [$2+48];\n"
    body += "setp.gt.u64 p, a, $3;\nsetp.gt.u64 q, b, $3;\nsetp.gt.u64 r, c, $3;\nsetp.gt.u64 s, d, $3;\n"
    body += "mov.u32 $0, $1;\n@p add.u32 $0, $0, 1;\n@q add.u32 $0, $0, 1;\n"
    body += "@r add.u32 $0, $0, 1;\n@s add.u32 $0, $0, 1;\n}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(acc, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip), _lv(key, loc=loc, ip=ip)],
            body,
            "=&r,r,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_dsmem_b32(addr, rank, value, *, loc=None, ip=None):
    """32-bit store into the shared memory of cluster CTA ``rank`` at its offset ``addr``."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(rank, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .b32 r;\nmapa.shared::cluster.u32 r, $0, $1;\nst.shared::cluster.b32 [r], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_zero16(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 z;\nmov.b32 z, 0;\nst.shared.v4.b32 [$0], {z, z, z, z};\n}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b64_or_zero(valid, addr, *, loc=None, ip=None):
    """``*addr`` if ``valid != 0`` else 0."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b64 $0, 0;\n@p ld.shared.b64 $0, [$2];\n}",
            "=&l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _syncwarp(*, loc=None, ip=None):
    _asm(None, [], "bar.warp.sync 0xffffffff;", "", loc=loc, ip=ip)


@dsl_user_op
def _shfl_idx_b32(value, src_lane, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(src_lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_f32(value, src_lane, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(src_lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_max_u32(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_min_u32(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "redux.sync.min.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _argmax_scan4(
    best_hi, best_slot, best_sig, removed, his, sigs, slot0: int, *, loc=None, ip=None
):
    """Fold four slots (``slot0..slot0+3``) into a lane's running best ``(hi, slot, sigmoid)``:
    the largest ``hi`` whose bit in ``removed`` is clear, the first slot on ties (slots are in
    expert order, so the first is the smallest expert)."""
    body = "{\n.reg .pred p;\n.reg .b32 h, b;\nmov.u32 $0, $3;\nmov.u32 $1, $4;\nmov.b32 $2, $5;\n"
    for i in range(4):
        body += (
            f"bfe.u32 b, $6, {slot0 + i}, 1;\n"
            "setp.eq.u32 p, b, 0;\n"
            f"selp.b32 h, ${7 + i}, 0, p;\n"
            "setp.gt.u32 p, h, $0;\n"
            "@p mov.u32 $0, h;\n"
            f"@p mov.u32 $1, {slot0 + i};\n"
            f"@p mov.b32 $2, ${11 + i};\n"
        )
    body += "}"
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.f32()]),
        [
            _iv(best_hi, loc=loc, ip=ip),
            _iv(best_slot, loc=loc, ip=ip),
            _fv(best_sig, loc=loc, ip=ip),
            _iv(removed, loc=loc, ip=ip),
        ]
        + [_iv(v, loc=loc, ip=ip) for v in his]
        + [_fv(v, loc=loc, ip=ip) for v in sigs],
        body,
        "=&r,=&r,=&f,r,r,f,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _cluster_arrive_release(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _low_bits_mask(n, *, loc=None, ip=None):
    """``(1 << n) - 1`` for ``0 <= n < 32``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(n, loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nshl.b32 t, 1, $1;\nsub.u32 $0, t, 1;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bit(n, *, loc=None, ip=None):
    """``1 << n`` for ``0 <= n < 32``."""
    return Int32(
        _asm(T.i32(), [_iv(n, loc=loc, ip=ip)], "shl.b32 $0, 1, $1;", "=r,r", loc=loc, ip=ip)
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class K3RouteTailKernel:
    """Top-16 + ``moe_sort`` of up to 256 tokens on one cluster of ``num_ctas`` CTAs.

    The warps of a CTA form teams of ``warps_per_token`` warps; a team routes one token at a
    time, lane ``g`` of the team (``g < 32 * warps_per_token``) owning experts
    ``j * 32 * warps_per_token + g``.

    Args:
        num_ctas: CTAs (cluster size), ``1..8``; host side guarantees ``ceil(M / num_ctas)
            <= 32``.
        num_warps: Warps per CTA.
        warps_per_token: Warps routing one token (1, 2, 4 or 7).
        num_splits: Split-K partial logits summed in split order (1 = plain logits).
        with_sort: Also produce the ``moe_sort`` outputs.
        use_pdl: Launch with programmatic dependent launch (the logits come from the
            previous kernel).
        local_cap: Largest number of local experts served.
    """

    def __init__(
        self,
        num_ctas: int,
        num_warps: int,
        warps_per_token: int,
        num_splits: int,
        with_sort: bool,
        use_pdl: bool,
        local_cap: int = 224,
    ):
        if not 1 <= num_ctas <= MAX_CTAS:
            raise ValueError(f"num_ctas={num_ctas} must be in [1, {MAX_CTAS}]")
        if warps_per_token not in (1, 2, 4, 7) or num_warps % warps_per_token:
            raise ValueError(f"warps_per_token={warps_per_token} must be 1, 2, 4 or 7")
        if num_warps // warps_per_token > 15 and warps_per_token > 1:
            raise ValueError("at most 15 multi-warp teams (named barriers 1..15)")
        if not 1 <= num_splits <= 64:
            raise ValueError(f"num_splits={num_splits} must be in [1, 64]")
        self.num_ctas = num_ctas
        self.num_warps = num_warps
        self.threads = 32 * num_warps
        self.team_warps = warps_per_token
        self.team_lanes = 32 * warps_per_token
        self.teams = num_warps // warps_per_token
        self.slots = NUM_EXPERTS // self.team_lanes
        # Candidate records per warp (plus zero padding to a multiple of 4); more fall back to
        # the exact arg-max rounds.
        self.region_cap = 64 if warps_per_token == 1 else 32
        self.region_slots = self.region_cap + 4
        self.local_cap = local_cap
        self.num_splits = num_splits
        self.with_sort = with_sort
        self.use_pdl = use_pdl
        # Local experts handled per thread in the sort scan (contiguous).
        self.experts_per_thread = (local_cap + self.threads - 1) // self.threads
        self.max_tiles_per_expert = (MAX_M + TILE - 1) // TILE
        # Skip candidate-ranking rounds in which the warp has no candidate.
        self.skip_empty_rank_rounds = False
        # Approximate candidate pass (cluster kernel only).
        self.approx = type(self) is K3RouteTailKernel

    @cute.jit
    def __call__(
        self,
        m: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        logits_out_addr: Int64,
        stream: cuda.CUstream,
    ):
        teams = self.teams

        @cute.struct
        class SharedStorage:
            bias: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, NUM_EXPERTS], 16]
            mask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            # Every CTA's masks, broadcast over distributed shared memory.
            masks: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_ctas * self.local_cap], 16
            ]
            base: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _MAX_TOKENS_PER_CTA * TOP_K], 16
            ]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_warps * self.region_slots * 4], 16
            ]
            top: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * TOP_K * 2], 16]
            # Per team and warp: threshold, then candidate count.
            xchg: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * 16], 16]
            warp_tot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.num_warps], 16]

        self.shared_storage = SharedStorage

        kernel = self.kernel(
            m,
            rows,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            scale_bits,
            logits_out_addr,
        )
        if cutlass.const_expr(self.num_ctas > 1):
            kernel.launch(
                grid=(self.num_ctas, 1, 1),
                block=(self.threads, 1, 1),
                cluster=(self.num_ctas, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            kernel.launch(
                grid=(1, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        logits_out_addr: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        rank, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.bias.data_ptr().toint()
        mask_smem = storage.mask.data_ptr().toint()
        masks_smem = storage.masks.data_ptr().toint()
        base_smem = storage.base.data_ptr().toint()
        ids_smem = storage.ids.data_ptr().toint()
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = storage.cand.data_ptr().toint() + team * (
            self.team_warps * self.region_slots * _CAND_BYTES
        )
        top_smem = storage.top.data_ptr().toint() + team * (TOP_K * 8)
        xchg_smem = storage.xchg.data_ptr().toint() + team * 64
        warp_tot_smem = storage.warp_tot.data_ptr().toint()

        per_cta = (m + self.num_ctas - 1) // self.num_ctas
        t_begin = rank * per_cta
        t_end = t_begin + per_cta
        if t_end > m:
            t_end = m

        if cutlass.const_expr(self.with_sort):
            for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
                _sts_b32(mask_smem + j * 4, Int32(0))
        # The bias is a weight: stage it in shared memory before waiting on the previous grid.
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
            _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            self.topk_token(
                token,
                token - t_begin,
                team,
                team_warp,
                lane,
                rows,
                logits_addr,
                bias_smem,
                ids_addr,
                vals_addr,
                scale_bits,
                logits_out_addr,
                cand_smem,
                top_smem,
                xchg_smem,
                ids_smem,
                mask_smem,
                local_offset,
                local_num,
            )
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.with_sort):
            cute.arch.barrier()
            if cutlass.const_expr(self.num_ctas > 1):
                # Broadcast this CTA's token masks into every CTA of the cluster.
                for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
                    word = _lds_b32(mask_smem + j * 4)
                    dst = masks_smem + (rank * self.local_cap + j) * 4
                    for c in cutlass.range_constexpr(self.num_ctas):
                        _st_dsmem_b32(dst, Int32(c), word)
                _cluster_arrive_release()
                _cluster_wait_acquire()
            self.sort(
                tidx,
                warp,
                lane,
                rank,
                t_begin,
                t_end,
                mask_smem,
                masks_smem,
                base_smem,
                ids_smem,
                warp_tot_smem,
                e2p_addr,
                p2e_addr,
                tile_expert_addr,
                tile_limit_addr,
                total_addr,
                num_tiles_addr,
                local_offset,
                local_num,
            )

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """Logit ``s * stride + g`` (``s < n``) of the first split of the token whose logits start
        at ``row``."""
        return [_ld_f32(row + Int64((s * stride + g) * 4)) for s in range(n)]

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        """Logits ``128 j + 4 lane .. + 3`` (``j < n``) of the first split of the token whose
        logits start at ``row``, concatenated. Not ``cute.jit`` (the DSL preprocessor does not
        rewrite this method), so the loop is a plain Python loop, unrolled at trace time."""
        logit = []
        for blk in range(n):
            logit.extend(_ld_v4_f32(row + Int64((blk * 128 + lane * 4) * 4)))
        return logit

    @cute.jit
    def team_sync(self, team: Int32):
        if cutlass.const_expr(self.team_warps > 1):
            _bar_sync_named(team + 1, self.team_lanes)
        else:
            _syncwarp()

    @cute.jit
    def topk_token(
        self,
        token: Int32,
        token_local: Int32,
        team: Int32,
        team_warp: Int32,
        lane: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_smem: Int32,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        logits_out_addr: Int64,
        cand_smem: Int32,
        top_smem: Int32,
        xchg_smem: Int32,
        ids_smem: Int32,
        mask_smem: Int32,
        local_offset: Int32,
        local_num: Int32,
    ):
        """Top-16 of one token by one team; its first warp writes ids/weights and mask bits."""
        expert, weight = self.select_token(
            token,
            team,
            team_warp,
            lane,
            rows,
            logits_addr,
            bias_smem,
            scale_bits,
            logits_out_addr,
            cand_smem,
            top_smem,
            xchg_smem,
        )
        if team_warp == 0:
            if lane < TOP_K:
                out = (token * TOP_K + lane) * 4
                _st_global_b32(ids_addr + Int64(out), expert)
                _st_global_f32(vals_addr + Int64(out), weight)
                if cutlass.const_expr(self.with_sort):
                    _sts_b32(ids_smem + (token_local * TOP_K + lane) * 4, expert)
                    local = expert - local_offset
                    if local >= 0 and local < local_num:
                        _red_shared_or(mask_smem + local * 4, _bit(token_local))

    @cute.jit
    def select_token(
        self,
        token: Int32,
        team: Int32,
        team_warp: Int32,
        lane: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_smem: Int32,
        scale_bits: Int64,
        logits_out_addr: Int64,
        cand_smem: Int32,
        top_smem: Int32,
        xchg_smem: Int32,
    ):
        """Top-16 of one token by one team: returns ``(expert, weight)``, the ``lane``-th
        selection in lanes ``0..15`` of the team's first warp."""
        n_slots = self.slots
        stride = self.team_lanes
        g = team_warp * 32 + lane
        row = logits_addr + Int64(token) * Int64(NUM_EXPERTS * 4)
        split_stride = Int64(rows) * Int64(NUM_EXPERTS * 4)
        logit = self.load_logits(row, g, stride, n_slots)
        for sp in cutlass.range_constexpr(1, self.num_splits):
            for s in cutlass.range_constexpr(n_slots):
                logit[s] = logit[s] + _ld_f32(row + split_stride * sp + Int64((s * stride + g) * 4))
        if logits_out_addr != 0:
            out_row = logits_out_addr + Int64(token) * Int64(NUM_EXPERTS * 4)
            for s in cutlass.range_constexpr(n_slots):
                _st_global_f32(out_row + Int64((s * stride + g) * 4), logit[s])
        bias = [_lds_f32(bias_smem + (s * stride + g) * 4) for s in range(n_slots)]
        if cutlass.const_expr(self.approx):
            # Approximate scores a_e, |a_e - s_e| <= delta (APPROX_MARGIN = 2 delta) for the exact
            # scores s_e. Records keep the logit (in the sigmoid field) for the exact pass below.
            sig = [
                _tanh_approx(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
                for s in range(n_slots)
            ]
        else:
            sig = [
                cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
                for s in range(n_slots)
            ]
        his = [_score_bits(sig[s] + bias[s]) for s in range(n_slots)]
        lane_max = his[0]
        for s in cutlass.range_constexpr(1, n_slots):
            lane_max = _max_u32(lane_max, his[s])
        # A lower bound of the 16th largest score: the 16th largest lane maximum of a warp (16
        # of the warp's experts score at least that much); the team takes the largest.
        threshold = _warp_16th_largest_u32(lane_max, lane)
        if cutlass.const_expr(self.team_warps > 1):
            if lane == 0:
                _sts_b32(xchg_smem + team_warp * 4, threshold)
            self.team_sync(team)
            for w in cutlass.range_constexpr(self.team_warps):
                threshold = _max_u32(threshold, _lds_b32(xchg_smem + w * 4))
        if cutlass.const_expr(self.approx):
            # The threshold t is at most the 16th largest approximate score A16, and A16 <= T* +
            # delta for the 16th largest exact score T* (more than 15 approximate scores above
            # T* + delta would mean more than 15 exact scores above T*). Every expert of the exact
            # top 16 has a_e >= s_e - delta >= T* - delta >= t - 2 delta: all are candidates, and
            # ranking the candidates by their exact keys gives noaux_tc's selection and order.
            threshold = _score_bits(_score_value(threshold) - Float32(APPROX_MARGIN))
        count = Int32(0)
        for c4 in cutlass.range_constexpr(0, n_slots, 4):
            count = count + _count_ge_u32(his[c4 : min(c4 + 4, n_slots)], threshold)
        inclusive = _warp_inclusive_sum(count)
        num_mine = _shfl_idx_b32(inclusive, Int32(31))
        cap = self.region_cap
        region_bytes = self.region_slots * _CAND_BYTES
        region = cand_smem + team_warp * region_bytes
        if num_mine <= cap:
            los = [Int32(65535 - s * stride) - g for s in range(n_slots)]
            dst = region + (inclusive - count) * _CAND_BYTES
            # Approximate pass: the record's sigmoid field carries the logit for the exact pass.
            carried = sig
            if cutlass.const_expr(self.approx):
                carried = logit
            for c4 in cutlass.range_constexpr(0, n_slots, 4):
                q = slice(c4, min(c4 + 4, n_slots))
                dst = _store_candidates(dst, his[q], los[q], carried[q], threshold)
            # Zero keys after the last candidate, so the ranking reads groups of four.
            if lane < 4:
                _sts_zero16(region + (num_mine + lane) * _CAND_BYTES)
        if cutlass.const_expr(self.approx):
            if num_mine <= cap:
                # Exact pass over this warp's candidates: noaux_tc's sigmoid and key, in place.
                _syncwarp()
                for c0 in cutlass.range_constexpr(cap // 32):
                    c = lane + c0 * 32
                    if c < num_mine:
                        rec = region + c * _CAND_BYTES
                        e = 65535 - _lds_b32(rec)
                        sg = cute.math.tanh(_lds_f32(rec + 8) * Float32(0.5)) * Float32(
                            0.5
                        ) + Float32(0.5)
                        _sts_b32(rec + 4, _score_bits(sg + _lds_f32(bias_smem + e * 4)))
                        _sts_f32(rec + 8, sg)
        if cutlass.const_expr(self.team_warps > 1):
            if lane == 0:
                _sts_b32(xchg_smem + 32 + team_warp * 4, num_mine)
        self.team_sync(team)
        counts = [num_mine]
        overflow = Int32(0)
        if num_mine > cap:
            overflow = Int32(1)
        if cutlass.const_expr(self.team_warps > 1):
            counts = [_lds_b32(xchg_smem + 32 + w * 4) for w in range(self.team_warps)]
            for w in cutlass.range_constexpr(self.team_warps):
                if counts[w] > cap:
                    overflow = Int32(1)

        expert = Int32(0)
        score = Float32(0.0)
        if overflow == 0:
            # Rank this warp's candidates against all candidates of the team.
            for c0 in cutlass.range_constexpr(cap // 32):
                c = lane + c0 * 32
                valid = Int32(0)
                if c < num_mine:
                    valid = Int32(1)
                key = _lds_b64_or_zero(valid, region + c * _CAND_BYTES)
                r = Int32(0)
                for w in cutlass.range_constexpr(self.team_warps):
                    other = cand_smem + w * region_bytes
                    limit = counts[w]
                    if cutlass.const_expr(c0 > 0 and self.skip_empty_rank_rounds):
                        # No candidate of this warp in this round: skip the count.
                        if num_mine <= c0 * 32:
                            limit = Int32(0)
                    for o in cutlass.range(0, limit, 4, unroll=1):
                        r = _count_greater4(r, other + o * _CAND_BYTES, key)
                if valid != 0 and r < TOP_K:
                    _sts_b32(top_smem + r * 8, 65535 - (Int32(key) & 0xFFFF))
                    _sts_b32(top_smem + r * 8 + 4, _lds_b32(region + c * _CAND_BYTES + 8))
            self.team_sync(team)
            if team_warp == 0 and lane < TOP_K:
                expert = _lds_b32(top_smem + lane * 8)
                score = _lds_f32(top_smem + lane * 8 + 4)
        else:
            if team_warp == 0:
                expert, score = self.topk_rounds(row, split_stride, bias_smem, lane)

        weight = Float32(0.0)
        if team_warp == 0:
            total = _warp_sum_f32(score)
            # Exact FP32 fast path of noaux_tc's FP64 expression (see k3_fused_router).
            if _unit_scale_fast_path(scale_bits, total) != 0:
                weight = _div_rn_f32(score, total)
            else:
                weight = _renormalize(score, total, scale_bits)
        return expert, weight

    @cute.jit
    def topk_rounds(self, row: Int64, split_stride: Int64, bias_smem: Int32, lane: Int32):
        """Exact fallback by one warp: 16 arg-max rounds over all 896 keys of the token (lane
        ``l`` owns experts ``128 j + 4 l + i``); lane ``k`` gets the ``k``-th expert."""
        logit = self.load_logits4(row, lane, _BLOCKS)
        for sp in cutlass.range_constexpr(1, self.num_splits):
            for blk in cutlass.range_constexpr(_BLOCKS):
                part = _ld_v4_f32(row + split_stride * sp + Int64((blk * 128 + lane * 4) * 4))
                for i in cutlass.range_constexpr(4):
                    logit[blk * 4 + i] = logit[blk * 4 + i] + part[i]
        sig = [
            cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
            for s in range(_SLOTS)
        ]
        his = []
        for blk in cutlass.range_constexpr(_BLOCKS):
            b = _lds_v4_f32(bias_smem + (blk * 128 + lane * 4) * 4)
            for i in cutlass.range_constexpr(4):
                his.append(_score_bits(sig[blk * 4 + i] + b[i]))
        removed = Int32(0)
        expert = Int32(0)
        score = Float32(0.0)
        for r in cutlass.range_constexpr(TOP_K):
            best_hi = Int32(0)
            best_slot = Int32(0)
            best_sig = Float32(0.0)
            for blk in cutlass.range_constexpr(_BLOCKS):
                q = slice(blk * 4, blk * 4 + 4)
                best_hi, best_slot, best_sig = _argmax_scan4(
                    best_hi, best_slot, best_sig, removed, his[q], sig[q], blk * 4
                )
            top_hi = _redux_max_u32(best_hi)
            # Slot s of this lane is expert 128 (s / 4) + 4 lane + s % 4.
            cand = Int32(0xFFFF)
            if best_hi == top_hi:
                cand = (best_slot >> 2) * 128 + lane * 4 + (best_slot & 3)
            winner = _redux_min_u32(cand)
            owner = (winner & 127) >> 2
            if lane == owner:
                removed = removed | _bit(best_slot)
            win_sig = _shfl_idx_f32(best_sig, owner)
            if lane == r:
                expert = winner
                score = win_sig
        return expert, score

    @cute.jit
    def sort(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        rank: Int32,
        t_begin: Int32,
        t_end: Int32,
        mask_smem: Int32,
        masks_smem: Int32,
        base_smem: Int32,
        ids_smem: Int32,
        warp_tot_smem: Int32,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
    ):
        """``moe_sort`` metadata from the token masks of all CTAs of the cluster."""
        ept = self.experts_per_thread
        j0 = tidx * ept
        counts = []
        before = []
        for q in cutlass.range_constexpr(ept):
            j = j0 + q
            jj = j
            if j >= local_num:
                jj = Int32(0)
            words = []
            for c in cutlass.range_constexpr(self.num_ctas):
                if cutlass.const_expr(self.num_ctas > 1):
                    words.append(_lds_b32(masks_smem + (c * self.local_cap + jj) * 4))
                else:
                    words.append(_lds_b32(mask_smem + jj * 4))
            cnt = Int32(0)
            pre = Int32(0)
            for c in cutlass.range_constexpr(self.num_ctas):
                n = _popc(words[c])
                cnt = cnt + n
                if c < rank:
                    pre = pre + n
            if j >= local_num:
                cnt = Int32(0)
            counts.append(cnt)
            before.append(pre)
        tiles = [(counts[q] + (TILE - 1)) >> 7 for q in range(ept)]
        thread_tiles = tiles[0]
        for q in cutlass.range_constexpr(1, ept):
            thread_tiles = thread_tiles + tiles[q]
        inclusive = _warp_inclusive_sum(thread_tiles)
        if lane == 31:
            _sts_b32(warp_tot_smem + warp * 4, inclusive)
        cute.arch.barrier()
        tile = inclusive - thread_tiles
        num_tiles = Int32(0)
        for w in cutlass.range_constexpr(self.num_warps):
            cnt_w = _lds_b32(warp_tot_smem + w * 4)
            if w < warp:
                tile = tile + cnt_w
            num_tiles = num_tiles + cnt_w
        for q in cutlass.range_constexpr(ept):
            j = j0 + q
            if tiles[q] != 0:
                _sts_b32(base_smem + j * 4, tile * TILE + before[q])
                if rank == 0:
                    for t in cutlass.range_constexpr(self.max_tiles_per_expert):
                        if t < tiles[q]:
                            limit = tile * TILE + counts[q]
                            if limit > (tile + t + 1) * TILE:
                                limit = (tile + t + 1) * TILE
                            _st_global_b32(tile_expert_addr + Int64((tile + t) * 4), j)
                            _st_global_b32(tile_limit_addr + Int64((tile + t) * 4), limit)
            tile = tile + tiles[q]
        if rank == 0 and tidx == 0:
            _st_global_b32(total_addr, num_tiles * TILE)
            _st_global_b32(num_tiles_addr, num_tiles)
        cute.arch.barrier()

        n_entries = (t_end - t_begin) * TOP_K
        for e in cutlass.range(tidx, n_entries, self.threads, unroll=1):
            token_local = e >> 4
            expanded = t_begin * TOP_K + e
            local = _lds_b32(ids_smem + e * 4) - local_offset
            perm = Int32(-1)
            if local >= 0 and local < local_num:
                word = _lds_b32(mask_smem + local * 4)
                perm = _lds_b32(base_smem + local * 4) + _popc(word & _low_bits_mask(token_local))
                _st_global_b32(p2e_addr + Int64(perm * 4), expanded)
            _st_global_b32(e2p_addr + Int64(expanded * 4), perm)


# ---------------------------------------------------------------------------
# Multi-CTA routing (M > 256)
# ---------------------------------------------------------------------------

#: Largest token count of the multi-CTA routing (:class:`K3RouteWideKernel`).
WIDE_MAX_M = 2048
#: Largest number of local experts the multi-CTA sort serves (expert parallelism >= 4).
WIDE_LOCAL_CAP = 224
#: Cells (local experts x CTAs) of the sort's shared-memory table (32 KB): the sort serves
#: ``local_num_experts * ceil(M / tokens_per_cta) <= 8192``.
WIDE_TABLE_CELLS = 8192
_WIDE_MAX_CTAS = WIDE_MAX_M // 8
#: Workspace of the multi-CTA sort: arrival counter, then per CTA its number of local
#: selections, the ``[CTA][local expert]`` token masks, and per CTA a list of its local
#: selections (``tokens_per_cta * 16`` slots).
_WIDE_COUNT_OFFSET = 128
_WIDE_MASK_OFFSET = _WIDE_COUNT_OFFSET + _WIDE_MAX_CTAS * 4
_WIDE_LIST_OFFSET = _WIDE_MASK_OFFSET + WIDE_TABLE_CELLS * 4
_WIDE_WORKSPACE_BYTES = _WIDE_LIST_OFFSET + WIDE_MAX_M * TOP_K * 4
#: Smem words of the top-k phase per region (aliased by the sort's table): bias, then per team.
_WIDE_BIAS_WORDS = NUM_EXPERTS
#: Cells per step of the sort's per-expert scan.
_SCAN_CHUNK = 8
#: Shared-memory budget of the compact sort tables of 8-token CTAs: ``CTAs x L4`` mask bytes
#: (``L4``: local experts rounded up to 4), then 16-bit rows-before per 8-CTA block and expert;
#: the size of the top-k scratch of an 8-warp CTA, which they alias.
WIDE_COMPACT_TABLE_BYTES = 13824


@dsl_user_op
def _ld_relaxed_b32_or_zero(valid, addr, *, loc=None, ip=None):
    """``*addr`` (relaxed, GPU scope) if ``valid != 0`` else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b32 $0, 0;\n"
            "@p ld.relaxed.gpu.global.b32 $0, [$2];\n}",
            "=&r,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_b32_or_zero(valid, addr, *, loc=None, ip=None):
    """Shared ``*addr`` if ``valid != 0`` else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b32 $0, 0;\n@p ld.shared.b32 $0, [$2];\n}",
            "=&r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_u8_or_zero(valid, addr, *, loc=None, ip=None):
    """Shared byte ``*addr`` if ``valid != 0`` else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b32 $0, 0;\n@p ld.shared.u8 $0, [$2];\n}",
            "=&r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared.u16 [$0], h;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_u16(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_b32_if(valid, addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.shared.b32 [$1], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_shared_add(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


class K3RouteWideKernel(K3RouteTailKernel):
    """Top-16 (+ ``moe_sort``) of up to :data:`WIDE_MAX_M` tokens on a grid of independent CTAs.

    CTA ``c`` routes tokens ``[T c, T c + T)`` (``T = tokens_per_cta``) with the teams of
    :class:`K3RouteTailKernel` (same selection code, so the same top-k). The sort is done by the
    CTA that finishes last:

    * Each CTA marks, per local expert ``j``, a ``T``-bit mask of its tokens that selected ``j``
      in shared memory, appends each such selection ``(j, token, slot k)`` to its list in the
      workspace and stores ``expanded_idx_to_permuted_idx = -1`` for its selections of other
      ranks' experts. It then publishes the masks (``[c][j]``) and its list length and takes a
      ticket from an arrival counter (release fence first).
    * The CTA that draws the last ticket resets the counter and builds ``moe_sort``'s metadata:
      it loads the whole mask table and the first ``2 T`` list entries of every CTA in one round
      of loads (later entries, rare at EP16, are loaded as needed), then, in shared memory,
      turns each expert's masks into token counts and row offsets (rows of an expert in
      ascending token order), lays out the tiles like :class:`K3RouteTailKernel`, and writes
      the rows of all listed selections. 16-token CTAs keep a 32-bit cell per (CTA, expert): the
      expert's rows in earlier CTAs above the mask. 8-token CTAs keep byte masks plus the rows
      before each block of 8 CTAs (:data:`WIDE_COMPACT_TABLE_BYTES`), aliased with the top-k
      scratch, so a 256-thread CTA needs about 14 KB of shared memory and can share an SM with
      a persistent GEMM CTA.

    No CTA waits on another, so the grid needs no co-residency. Launches sharing the workspace
    must not overlap (stream order provides that); each launch leaves the counter at zero, so a
    captured CUDA graph replays as is.

    Args:
        tokens_per_cta: Tokens per CTA: 8 (compact sort tables) or 16, or 32 (top-k only).
        num_warps: Warps per CTA: 8 or 16 (with the sort), or 4 (top-k only).
        warps_per_token: Warps routing one token (1, 2, 4 or 7).
        num_splits: Split-K partial logits summed in split order (1 = plain logits).
        with_sort: Also produce the ``moe_sort`` outputs.
        use_pdl: Launch with programmatic dependent launch.
        local_cap: Largest number of local experts served (at most :data:`WIDE_LOCAL_CAP`).
    """

    def __init__(
        self,
        tokens_per_cta: int,
        num_warps: int,
        warps_per_token: int,
        num_splits: int,
        with_sort: bool,
        use_pdl: bool,
        local_cap: int = WIDE_LOCAL_CAP,
    ):
        super().__init__(1, num_warps, warps_per_token, num_splits, with_sort, use_pdl, local_cap)
        if tokens_per_cta not in ((8, 16) if with_sort else (8, 16, 32)):
            raise ValueError(f"tokens_per_cta={tokens_per_cta} must be 8 or 16 (32 top-k only)")
        if num_warps not in ((8, 16) if with_sort else (4, 8, 16)):
            raise ValueError(f"num_warps={num_warps} must be 8 or 16 (4 top-k only)")
        if local_cap > WIDE_LOCAL_CAP:
            raise ValueError(f"local_cap={local_cap} must be at most {WIDE_LOCAL_CAP}")
        self.tokens = tokens_per_cta
        self.max_ctas = WIDE_MAX_M // tokens_per_cta
        # Sort: mask-table words per thread; list entries per CTA loaded up front (twice the
        # tokens, about twice the expected local selections at EP16), 32 / list_slots CTAs per
        # warp-wide load.
        self.compact = with_sort and tokens_per_cta == 8
        self.table_words = WIDE_COMPACT_TABLE_BYTES // 4 if self.compact else WIDE_TABLE_CELLS
        self.table_items = (self.table_words + self.threads - 1) // self.threads
        self.list_slots = min(2 * tokens_per_cta, 32)
        self.lists_per_load = 32 // self.list_slots
        self.list_loads = (self.max_ctas + num_warps * self.lists_per_load - 1) // (
            num_warps * self.lists_per_load
        )
        self.cand_words = num_warps * self.region_slots * 4
        self.topk_words = (
            _WIDE_BIAS_WORDS + self.cand_words + self.teams * TOP_K * 2 + self.teams * 16
        )
        self.skip_empty_rank_rounds = True

    @cute.jit
    def __call__(
        self,
        m: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        ws_addr: Int64,
        stream: cuda.CUstream,
    ):
        scratch_words = self.topk_words
        if cutlass.const_expr(self.with_sort):
            scratch_words = max(scratch_words, self.table_words)

        @cute.struct
        class SharedStorage:
            # Top-k phase: bias, candidate regions, top-16 and exchange slots; the sort's
            # [CTA][local expert] table aliases them.
            scratch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, scratch_words], 16]
            mask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            # Sort: first tile per local expert.
            tile: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            warp_tot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.num_warps], 16]
            # [0]: last-CTA flag, [1]: number of local selections of this CTA.
            flag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.shared_storage = SharedStorage

        num_ctas = (m + (self.tokens - 1)) // self.tokens
        self.kernel(
            m,
            rows,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            scale_bits,
            ws_addr,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        rows: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        ws_addr: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        scratch = storage.scratch.data_ptr().toint()
        mask_smem = storage.mask.data_ptr().toint()
        tile_smem = storage.tile.data_ptr().toint()
        warp_tot_smem = storage.warp_tot.data_ptr().toint()
        flag_smem = storage.flag.data_ptr().toint()
        bias_smem = scratch
        cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
        top_all = cand_all + self.cand_words * 4
        xchg_all = top_all + self.teams * TOP_K * 8
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = cand_all + team * (self.team_warps * self.region_slots * _CAND_BYTES)
        top_smem = top_all + team * (TOP_K * 8)
        xchg_smem = xchg_all + team * 64

        num_ctas = (m + (self.tokens - 1)) // self.tokens
        t_begin = cta * self.tokens
        t_end = t_begin + self.tokens
        if t_end > m:
            t_end = m
        ws_count = ws_addr + Int64(_WIDE_COUNT_OFFSET)
        ws_mask = ws_addr + Int64(_WIDE_MASK_OFFSET)
        ws_list = ws_addr + Int64(_WIDE_LIST_OFFSET)
        list_base = ws_list + Int64(cta * (self.tokens * TOP_K * 4))

        if cutlass.const_expr(self.with_sort):
            for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
                _sts_b32(mask_smem + j * 4, Int32(0))
            if tidx == 0:
                _sts_b32(flag_smem + 4, Int32(0))
        # The bias is a weight: stage it in shared memory before waiting on the previous grid.
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
            _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            expert, weight = self.select_token(
                token,
                team,
                team_warp,
                lane,
                rows,
                logits_addr,
                bias_smem,
                scale_bits,
                Int64(0),
                cand_smem,
                top_smem,
                xchg_smem,
            )
            if team_warp == 0:
                if lane < TOP_K:
                    out = (token * TOP_K + lane) * 4
                    _st_global_b32(ids_addr + Int64(out), expert)
                    _st_global_f32(vals_addr + Int64(out), weight)
                    if cutlass.const_expr(self.with_sort):
                        local = expert - local_offset
                        token_local = token - t_begin
                        if local >= 0 and local < local_num:
                            _red_shared_or(mask_smem + local * 4, _bit(token_local))
                            pos = _atom_shared_add(flag_smem + 4, Int32(1))
                            _st_global_b32(
                                list_base + Int64(pos * 4),
                                (local << 12) | (token_local << 4) | lane,
                            )
                        else:
                            _st_global_b32(e2p_addr + Int64(out), Int32(-1))
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.with_sort):
            cute.arch.barrier()
            if cutlass.const_expr(self.compact):
                # Four 8-bit masks per word: the table is [CTA][local expert] bytes.
                words = (local_num + 3) // 4
                for w in cutlass.range(tidx, words, self.threads, unroll=1):
                    word = Int32(0)
                    for b in cutlass.range_constexpr(4):
                        if 4 * w + b < local_num:
                            word = word | (_lds_b32(mask_smem + (4 * w + b) * 4) << (8 * b))
                    _st_global_b32(ws_mask + Int64((cta * words + w) * 4), word)
            else:
                for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
                    _st_global_b32(
                        ws_mask + Int64((cta * local_num + j) * 4), _lds_b32(mask_smem + j * 4)
                    )
            if tidx == 0:
                _st_global_b32(ws_count + Int64(cta * 4), _lds_b32(flag_smem + 4))
            cute.arch.barrier()
            if tidx == 0:
                _fence_acq_rel_gpu()
                ticket = _atom_add_relaxed_gpu(ws_addr, Int32(1))
                last = Int32(0)
                if ticket == num_ctas - 1:
                    _fence_acq_rel_gpu()
                    _st_relaxed_b32(ws_addr, Int32(0))
                    last = Int32(1)
                _sts_b32(flag_smem, last)
            cute.arch.barrier()
            if _lds_b32(flag_smem) != 0:
                self.wide_sort(
                    tidx,
                    warp,
                    lane,
                    num_ctas,
                    ws_count,
                    ws_mask,
                    ws_list,
                    scratch,
                    tile_smem,
                    warp_tot_smem,
                    e2p_addr,
                    p2e_addr,
                    tile_expert_addr,
                    tile_limit_addr,
                    total_addr,
                    num_tiles_addr,
                    local_num,
                )

    @cute.jit
    def wide_sort(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        num_ctas: Int32,
        ws_count: Int64,
        ws_mask: Int64,
        ws_list: Int64,
        table_smem: Int32,
        tile_smem: Int32,
        warp_tot_smem: Int32,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_num: Int32,
    ):
        """``moe_sort`` metadata from the published masks and lists of all CTAs (last CTA)."""
        list_words = self.tokens * TOP_K
        # Compact: byte table [CTA][row_bytes], then the 16-bit block table [CTA / 8][expert].
        row_bytes = 4 * ((local_num + 3) // 4)
        block_smem = table_smem + ((num_ctas * row_bytes + 15) // 16) * 16
        cells = num_ctas * local_num
        if cutlass.const_expr(self.compact):
            cells = num_ctas * (row_bytes // 4)
        # One round of loads: the [CTA][local expert] mask table, then the list length and the
        # first list_slots entries of each CTA (lane l of a warp-wide load: CTA
        # lists_per_load * (warp + q * num_warps) + l / list_slots, entry l % list_slots).
        table = []
        for i in cutlass.range_constexpr(self.table_items):
            idx = tidx + i * self.threads
            valid = Int32(0)
            if idx < cells:
                valid = Int32(1)
            table.append(_ld_relaxed_b32_or_zero(valid, ws_mask + Int64(idx * 4)))
        slots = self.list_slots
        slot = lane % slots
        counts = []
        entries = []
        for q in cutlass.range_constexpr(self.list_loads):
            c = (warp + q * self.num_warps) * self.lists_per_load + lane // slots
            valid = Int32(0)
            if c < num_ctas:
                valid = Int32(1)
            counts.append(_ld_relaxed_b32_or_zero(valid, ws_count + Int64(c * 4)))
            entries.append(
                _ld_relaxed_b32_or_zero(valid, ws_list + Int64((c * list_words + slot) * 4))
            )
        for i in cutlass.range_constexpr(self.table_items):
            idx = tidx + i * self.threads
            valid = Int32(0)
            if idx < cells:
                valid = Int32(1)
            _sts_b32_if(valid, table_smem + idx * 4, table[i])
        cute.arch.barrier()

        # Thread per local expert: token count, and the expert's rows in earlier CTAs per cell
        # (stored above the cell's mask, which has at most 16 bits) or, compact, per block of
        # 8 CTAs.
        ept = self.experts_per_thread
        j0 = tidx * ept
        totals = []
        for q in cutlass.range_constexpr(ept):
            j = j0 + q
            run = Int32(0)
            if cutlass.const_expr(self.compact):
                if j < local_num:
                    for c0 in cutlass.range(0, num_ctas, 8, unroll=1):
                        _sts_u16(block_smem + ((c0 >> 3) * local_num + j) * 2, run)
                        masks = []
                        for u in cutlass.range_constexpr(8):
                            valid = Int32(0)
                            if c0 + u < num_ctas:
                                valid = Int32(1)
                            masks.append(
                                _lds_u8_or_zero(valid, table_smem + (c0 + u) * row_bytes + j)
                            )
                        for u in cutlass.range_constexpr(8):
                            run = run + _popc(masks[u])
            else:
                if j < local_num:
                    run = self.scan_cells(j, num_ctas, local_num, table_smem)
            totals.append(run)

        # Tile layout: ceil(count / 128) tiles per local expert, in local-expert order.
        tiles = [(totals[q] + (TILE - 1)) >> 7 for q in range(ept)]
        thread_tiles = tiles[0]
        for q in cutlass.range_constexpr(1, ept):
            thread_tiles = thread_tiles + tiles[q]
        inclusive = _warp_inclusive_sum(thread_tiles)
        if lane == 31:
            _sts_b32(warp_tot_smem + warp * 4, inclusive)
        cute.arch.barrier()
        tile = inclusive - thread_tiles
        num_tiles = Int32(0)
        for w in cutlass.range_constexpr(self.num_warps):
            cnt_w = _lds_b32(warp_tot_smem + w * 4)
            if w < warp:
                tile = tile + cnt_w
            num_tiles = num_tiles + cnt_w
        for q in cutlass.range_constexpr(ept):
            if j0 + q < local_num:
                _sts_b32(tile_smem + (j0 + q) * 4, tile * TILE)
            if tiles[q] != 0:
                end = tile * TILE + totals[q]
                for t in cutlass.range(tiles[q], unroll=1):
                    limit = (tile + t + 1) * TILE
                    if limit > end:
                        limit = end
                    _st_global_b32(tile_expert_addr + Int64((tile + t) * 4), j0 + q)
                    _st_global_b32(tile_limit_addr + Int64((tile + t) * 4), limit)
            tile = tile + tiles[q]
        if tidx == 0:
            _st_global_b32(total_addr, num_tiles * TILE)
            _st_global_b32(num_tiles_addr, num_tiles)
        cute.arch.barrier()

        # Rows of the listed selections: entry (j, token t, slot k) of CTA c lands in row
        # first_row(j) + rows of j in CTAs before c + tokens of the mask before t.
        for q in cutlass.range_constexpr(self.list_loads):
            c = (warp + q * self.num_warps) * self.lists_per_load + lane // slots
            n = counts[q]
            if c < num_ctas:
                if slot < n:
                    self.place_entry(
                        entries[q],
                        c,
                        local_num,
                        row_bytes,
                        table_smem,
                        block_smem,
                        tile_smem,
                        e2p_addr,
                        p2e_addr,
                    )
                for i in cutlass.range(slot + slots, n, slots, unroll=1):
                    entry = _ld_relaxed_b32_or_zero(
                        Int32(1), ws_list + Int64((c * list_words + i) * 4)
                    )
                    self.place_entry(
                        entry,
                        c,
                        local_num,
                        row_bytes,
                        table_smem,
                        block_smem,
                        tile_smem,
                        e2p_addr,
                        p2e_addr,
                    )

    @cute.jit
    def scan_cells(self, j: Int32, num_ctas: Int32, local_num: Int32, table_smem: Int32):
        """Rewrite expert ``j``'s 32-bit cells as ``rows in earlier CTAs << 16 | mask``;
        returns the expert's token count."""
        run = Int32(0)
        for c0 in cutlass.range(0, num_ctas, _SCAN_CHUNK, unroll=1):
            words = []
            valids = []
            for u in cutlass.range_constexpr(_SCAN_CHUNK):
                valid = Int32(0)
                if c0 + u < num_ctas:
                    valid = Int32(1)
                valids.append(valid)
                words.append(_lds_b32_or_zero(valid, table_smem + ((c0 + u) * local_num + j) * 4))
            for u in cutlass.range_constexpr(_SCAN_CHUNK):
                _sts_b32_if(
                    valids[u],
                    table_smem + ((c0 + u) * local_num + j) * 4,
                    (run << 16) | words[u],
                )
                run = run + _popc(words[u])
        return run

    @cute.jit
    def place_entry(
        self,
        entry: Int32,
        c: Int32,
        local_num: Int32,
        row_bytes: Int32,
        table_smem: Int32,
        block_smem: Int32,
        tile_smem: Int32,
        e2p_addr: Int64,
        p2e_addr: Int64,
    ):
        """Write the row of selection ``entry`` (``j << 12 | token << 4 | slot``) of CTA ``c``."""
        j = entry >> 12
        t = (entry >> 4) & 0xFF
        k = entry & 15
        rank = _lds_b32(tile_smem + j * 4)
        if cutlass.const_expr(self.compact):
            rank = rank + _lds_u16(block_smem + ((c >> 3) * local_num + j) * 2)
            # Tokens of j in the block's earlier CTAs, then in this CTA before t.
            first = c - (c & 7)
            masks = []
            for u in cutlass.range_constexpr(7):
                valid = Int32(0)
                if first + u < c:
                    valid = Int32(1)
                masks.append(_lds_u8_or_zero(valid, table_smem + (first + u) * row_bytes + j))
            mine = _lds_u8_or_zero(Int32(1), table_smem + c * row_bytes + j)
            for u in cutlass.range_constexpr(7):
                rank = rank + _popc(masks[u])
            rank = rank + _popc(mine & _low_bits_mask(t))
        else:
            cell = _lds_b32(table_smem + (c * local_num + j) * 4)
            rank = rank + (cell >> 16) + _popc(cell & 0xFFFF & _low_bits_mask(t))
        expanded = (c * self.tokens + t) * TOP_K + k
        _st_global_b32(e2p_addr + Int64(expanded * 4), rank)
        _st_global_b32(p2e_addr + Int64(rank * 4), expanded)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

#: Local-expert capacity of the common build (expert parallelism >= 4); more use a larger one.
_SMALL_LOCAL_CAP = 224


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def supports_device(device: torch.device) -> bool:
    """Whether the routing kernels run on ``device`` (SM100, SM103 or SM107)."""
    return device.type == "cuda" and _is_supported_device(_device_index(device))


def default_config(m: int) -> tuple[int, int, int]:
    """``(num_ctas, warps_per_token, num_warps)`` for ``m`` tokens (at most 32 tokens per CTA).

    Measured on GB300 (standalone, behind a producer kernel): eight 8-warp CTAs with two-warp
    teams up to 32 tokens, one-warp teams up to 64; 16-warp CTAs above.
    """
    if m <= 32:
        return MAX_CTAS, 2, 8
    if m <= 64:
        return MAX_CTAS, 1, 8
    return MAX_CTAS, 1, 16


@functools.cache
def _compiled_tail(
    num_ctas: int,
    num_warps: int,
    warps_per_token: int,
    num_splits: int,
    with_sort: bool,
    use_pdl: bool,
    local_cap: int,
):
    kernel = K3RouteTailKernel(
        num_ctas, num_warps, warps_per_token, num_splits, with_sort, use_pdl, local_cap
    )
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(num_ctas),
        Int32(num_ctas),
        *([dummy] * 10),
        Int32(0),
        Int32(NUM_EXPERTS),
        Int64(0),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _check_inputs(logits: torch.Tensor, e_score_correction_bias: torch.Tensor) -> None:
    if logits.dtype != torch.float32 or logits.dim() not in (2, 3):
        raise ValueError("logits must be a 2D [M, 896] or 3D [S, R, 896] float32 tensor")
    if logits.shape[-1] != NUM_EXPERTS or not logits.is_contiguous():
        raise ValueError(f"logits must be contiguous with {NUM_EXPERTS} columns")
    if (
        e_score_correction_bias.dtype != torch.float32
        or e_score_correction_bias.numel() != NUM_EXPERTS
        or not e_score_correction_bias.is_contiguous()
    ):
        raise ValueError(f"e_score_correction_bias must be a contiguous float32 [{NUM_EXPERTS}]")
    for t in (logits, e_score_correction_bias):
        if not t.is_cuda or t.device != logits.device or t.data_ptr() % 16:
            raise ValueError("inputs must be 16-byte aligned tensors on the same CUDA device")


def can_use_k3_route_tail(
    num_tokens: int, logits: torch.Tensor, local_num_experts: int | None = None
) -> bool:
    """Whether the routing tail serves ``num_tokens`` tokens of these logits (non-raising).

    Above :data:`MAX_M` tokens the sort needs :func:`wide_sort_fits` (``local_num_experts=None``
    skips that check).
    """
    if num_tokens > MAX_M and local_num_experts is not None:
        if not wide_sort_fits(num_tokens, local_num_experts):
            return False
    return (
        logits.is_cuda
        and _is_supported_device(logits.device.index)
        and 1 <= num_tokens <= WIDE_MAX_M
        and logits.dtype == torch.float32
        and logits.shape[-1] == NUM_EXPERTS
    )


def run_k3_route_tail(
    logits: torch.Tensor,
    num_tokens: int,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sort_outputs: tuple[torch.Tensor, ...] | None,
    local_expert_offset: int,
    local_num_experts: int,
    *,
    logits_out: torch.Tensor | None = None,
    num_ctas: int | None = None,
    warps_per_token: int | None = None,
    num_warps: int | None = None,
    use_pdl: bool = True,
) -> None:
    """Launch the routing tail on the current stream (inputs are not re-validated).

    Args:
        logits: FP32 ``[M, 896]`` logits or ``[S, R, 896]`` split-K partials (``R >= M``).
        num_tokens: ``M``, ``1 <= M <= 256``.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        topk_ids: Output ``int32 [M, 16]``.
        topk_weights: Output ``fp32 [M, 16]``.
        sort_outputs: ``moe_sort`` outputs ``(tile_idx_to_expert_idx, tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
            total_num_padded_tokens, num_non_exiting_tiles)`` (``int32``), or ``None`` for
            top-k only.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.
        logits_out: Optional output ``fp32 [M, 896]``: the (summed) logits.
        num_ctas: Cluster size (default from :func:`default_config`).
        warps_per_token: Warps routing one token (default from :func:`default_config`).
        num_warps: Warps per CTA (default from :func:`default_config`).
        use_pdl: Launch with programmatic dependent launch.
    """
    m = num_tokens
    default_ctas, default_team, default_warps = default_config(m)
    if num_ctas is None:
        num_ctas = default_ctas
    if warps_per_token is None:
        warps_per_token = default_team
    if num_warps is None:
        num_warps = default_warps
    if (m + num_ctas - 1) // num_ctas > _MAX_TOKENS_PER_CTA:
        raise ValueError(f"M={m} needs more than {num_ctas} CTAs")
    splits = 1 if logits.dim() == 2 else logits.shape[0]
    rows = logits.shape[0] if logits.dim() == 2 else logits.shape[1]
    with_sort = sort_outputs is not None
    local_cap = _SMALL_LOCAL_CAP if local_num_experts <= _SMALL_LOCAL_CAP else NUM_EXPERTS
    compiled = _compiled_tail(
        num_ctas, num_warps, warps_per_token, splits, with_sort, use_pdl, local_cap
    )
    if with_sort:
        tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_outputs)
    else:
        tile_expert = tile_limit = e2p = p2e = total = num_tiles = 0
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
        Int32(rows),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(local_expert_offset),
        Int32(local_num_experts),
        Int64(scale_bits),
        Int64(logits_out.data_ptr() if logits_out is not None else 0),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


#: Local-expert capacity of the multi-CTA build for expert parallelism >= 14 (EP16: 56).
_WIDE_SMALL_LOCAL_CAP = 64
#: Largest token count routed with 8-token CTAs by default (see :func:`default_wide_config`).
_WIDE_COMPACT_MAX_M = 1024


def wide_table_fits(m: int, local_num_experts: int, tokens_per_cta: int) -> bool:
    """Whether the multi-CTA sort's shared-memory tables hold ``m`` tokens over
    ``local_num_experts`` local experts in CTAs of ``tokens_per_cta`` tokens."""
    ctas = (m + tokens_per_cta - 1) // tokens_per_cta
    if tokens_per_cta == 8:
        row_bytes = 4 * ((local_num_experts + 3) // 4)
        table = (ctas * row_bytes + 15) // 16 * 16
        return table + (ctas + 7) // 8 * local_num_experts * 2 <= WIDE_COMPACT_TABLE_BYTES
    return local_num_experts * ctas <= WIDE_TABLE_CELLS


def wide_sort_fits(m: int, local_num_experts: int) -> bool:
    """Whether the multi-CTA sort serves ``m`` tokens over ``local_num_experts`` local experts
    (see :func:`wide_table_fits`; CTAs of 8 or 16 tokens)."""
    return (
        1 <= m <= WIDE_MAX_M
        and 1 <= local_num_experts <= WIDE_LOCAL_CAP
        and (wide_table_fits(m, local_num_experts, 8) or wide_table_fits(m, local_num_experts, 16))
    )


def default_wide_config(m: int, local_num_experts: int = 56) -> tuple[int, int, int]:
    """``(tokens_per_cta, num_warps, warps_per_token)`` of the multi-CTA routing for ``m``
    tokens (``MAX_M < m <= WIDE_MAX_M``).

    Measured on GB300 (EP16): up to 1024 tokens, 8-warp CTAs of 8 tokens, whose compact sort
    tables let them share SMs with the persistent latent_down GEMM running beside the routing
    (M = 896: 23.6 instead of 28.7 us beside it, the same alone). Above, and whenever those
    tables do not fit, 16-warp CTAs of 16 tokens, one token per warp (M = 1344 alone: 9.5
    instead of 11.3 us, where the last CTA sorts the masks of 84 instead of 168 CTAs).
    """
    if m <= _WIDE_COMPACT_MAX_M and wide_table_fits(m, local_num_experts, 8):
        return 8, 8, 1
    return 16, 16, 1


@functools.cache
def _compiled_wide(
    tokens_per_cta: int,
    num_warps: int,
    warps_per_token: int,
    num_splits: int,
    with_sort: bool,
    use_pdl: bool,
    local_cap: int,
):
    kernel = K3RouteWideKernel(
        tokens_per_cta, num_warps, warps_per_token, num_splits, with_sort, use_pdl, local_cap
    )
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(MAX_M + 1),
        Int32(MAX_M + 1),
        *([dummy] * 10),
        Int32(0),
        Int32(local_cap),
        Int64(0),
        dummy,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_WIDE_WORKSPACES: dict[int, torch.Tensor] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def wide_workspace(device: torch.device) -> torch.Tensor:
    """Persistent per-device state of the multi-CTA sort (arrival counter + mask slots).

    Between launches the counter is zero; each launch restores that. It must be created
    outside CUDA graph capture (a capture would allocate it from the graph's private pool).
    """
    index = _device_index(device)
    workspace = _WIDE_WORKSPACES.get(index)
    if workspace is None:
        workspace = torch.zeros(
            _WIDE_WORKSPACE_BYTES // 4, dtype=torch.int32, device=torch.device("cuda", index)
        )
        _WIDE_WORKSPACES[index] = workspace
    return workspace


def wide_workspace_ready(device: torch.device) -> bool:
    """Whether this device's multi-CTA sort workspace exists (see :func:`wide_workspace`)."""
    return _device_index(device) in _WIDE_WORKSPACES


def run_k3_route_wide(
    logits: torch.Tensor,
    num_tokens: int,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sort_outputs: tuple[torch.Tensor, ...] | None,
    local_expert_offset: int,
    local_num_experts: int,
    *,
    tokens_per_cta: int | None = None,
    num_warps: int | None = None,
    warps_per_token: int | None = None,
    use_pdl: bool = True,
    workspace: torch.Tensor | None = None,
) -> None:
    """Launch the multi-CTA routing on the current stream (inputs are not re-validated).

    Same arguments and outputs as :func:`run_k3_route_tail` (without ``logits_out``) for
    ``1 <= num_tokens <= WIDE_MAX_M`` and, with ``sort_outputs``, ``local_num_experts <=
    WIDE_LOCAL_CAP`` and :func:`wide_table_fits`. The configuration defaults come from
    :func:`default_wide_config`. ``workspace``: the device's :func:`wide_workspace` (default:
    looked up here).
    """
    m = num_tokens
    default_tokens, default_warps, default_team = default_wide_config(m, local_num_experts)
    if tokens_per_cta is None:
        tokens_per_cta = default_tokens
    if num_warps is None:
        num_warps = default_warps
    if warps_per_token is None:
        warps_per_token = default_team
    if not 1 <= m <= WIDE_MAX_M:
        raise ValueError(f"num_tokens={m} must be in [1, {WIDE_MAX_M}]")
    with_sort = sort_outputs is not None
    if with_sort and (
        local_num_experts > WIDE_LOCAL_CAP
        or not wide_table_fits(m, local_num_experts, tokens_per_cta)
    ):
        raise ValueError(
            f"the multi-CTA sort serves at most {WIDE_LOCAL_CAP} local experts and "
            "tables within wide_table_fits"
        )
    splits = 1 if logits.dim() == 2 else logits.shape[0]
    rows = logits.shape[0] if logits.dim() == 2 else logits.shape[1]
    local_cap = (
        _WIDE_SMALL_LOCAL_CAP if local_num_experts <= _WIDE_SMALL_LOCAL_CAP else WIDE_LOCAL_CAP
    )
    compiled = _compiled_wide(
        tokens_per_cta, num_warps, warps_per_token, splits, with_sort, use_pdl, local_cap
    )
    if with_sort:
        tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_outputs)
        if workspace is None:
            workspace = wide_workspace(logits.device)
        workspace = workspace.data_ptr()
    else:
        tile_expert = tile_limit = e2p = p2e = total = num_tiles = workspace = 0
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
        Int32(rows),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(local_expert_offset),
        Int32(local_num_experts),
        Int64(scale_bits),
        Int64(workspace),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


def _route_tail_outputs(
    logits: torch.Tensor,
    m: int,
    e_score_correction_bias: torch.Tensor,
    local_num_experts: int,
    tile_tokens_dim: int,
) -> tuple[torch.Tensor, ...]:
    """Fresh ``(topk_ids, topk_weights, *moe_sort outputs)`` for ``m`` tokens."""
    device = logits.device
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num_experts, tile_tokens_dim)
    i32 = dict(dtype=torch.int32, device=device)
    return (
        torch.empty((m, TOP_K), **i32),
        torch.empty((m, TOP_K), dtype=torch.float32, device=device),
        torch.empty((max_tiles,), **i32),
        torch.empty((max_tiles,), **i32),
        torch.empty((m, TOP_K), **i32),
        torch.empty((max_permuted,), **i32),
        torch.empty((1,), **i32),
        torch.empty((1,), **i32),
    )


def k3_route_tail(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int = TILE,
    num_tokens: int = -1,
    workspace: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """``noaux_tc`` top-16 + ``moe_sort`` of FP32 router logits in one launch.

    Args:
        logits: ``fp32 [M, 896]`` router logits, or ``[S, R, 896]`` split-K partials whose
            first ``M`` rows are summed in split order.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.
        tile_tokens_dim: Must be 128.
        num_tokens: ``M`` (default: all rows of ``logits``), ``1 <= M <= 2048``; above 256
            tokens (multi-CTA kernel) see :func:`wide_sort_fits`.
        workspace: Above 256 tokens, the device's persistent multi-CTA sort workspace
            (:func:`wide_workspace`; default: looked up here). Unused at or below 256 tokens.

    Returns:
        ``(topk_ids int32 [M, 16], topk_weights fp32 [M, 16], tile_idx_to_expert_idx,
        tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
        total_num_padded_tokens, num_non_exiting_tiles)``. Top-k equals ``noaux_tc_op`` on the
        same logits bit for bit; the sort outputs have ``moe_sort``'s shapes and dtypes, only
        valid entries are written (as in ``moe_sort``), and the rows of an expert are in
        ascending token order.
    """
    _check_inputs(logits, e_score_correction_bias)
    if tile_tokens_dim != TILE:
        raise ValueError(f"tile_tokens_dim must be {TILE}")
    rows = logits.shape[0] if logits.dim() == 2 else logits.shape[1]
    m = rows if num_tokens < 0 else num_tokens
    if not 1 <= m <= min(rows, WIDE_MAX_M):
        raise ValueError(f"num_tokens={m} must be in [1, {min(rows, WIDE_MAX_M)}]")
    if not (
        0 <= local_expert_offset
        and 1 <= local_num_experts
        and local_expert_offset + local_num_experts <= NUM_EXPERTS
    ):
        raise ValueError("invalid local expert range")
    if m > MAX_M and not wide_sort_fits(m, local_num_experts):
        raise ValueError(
            f"above {MAX_M} tokens the sort serves at most {WIDE_LOCAL_CAP} local experts and "
            "tables within wide_sort_fits"
        )
    outs = _route_tail_outputs(
        logits, m, e_score_correction_bias, local_num_experts, tile_tokens_dim
    )
    if m <= MAX_M:
        run_k3_route_tail(
            logits,
            m,
            e_score_correction_bias,
            routed_scaling_factor,
            outs[0],
            outs[1],
            outs[2:],
            local_expert_offset,
            local_num_experts,
        )
    else:
        run_k3_route_wide(
            logits,
            m,
            e_score_correction_bias,
            routed_scaling_factor,
            outs[0],
            outs[1],
            outs[2:],
            local_expert_offset,
            local_num_experts,
            workspace=workspace,
        )
    return outs


__all__ = [
    "MAX_M",
    "WIDE_LOCAL_CAP",
    "WIDE_MAX_M",
    "WIDE_COMPACT_TABLE_BYTES",
    "WIDE_TABLE_CELLS",
    "K3RouteTailKernel",
    "K3RouteWideKernel",
    "can_use_k3_route_tail",
    "default_config",
    "default_wide_config",
    "k3_route_tail",
    "run_k3_route_tail",
    "run_k3_route_wide",
    "supports_device",
    "wide_sort_fits",
    "wide_table_fits",
    "wide_workspace",
    "wide_workspace_ready",
]
