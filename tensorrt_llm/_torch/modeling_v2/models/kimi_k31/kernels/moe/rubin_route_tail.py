# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 / K3.1 decode MoE routing (SM100/SM103/SM107): top-16 + ``moe_sort`` in one launch.

``k3_route_tail`` replaces ``noaux_tc_op`` -> ``moe_sort`` by one launch on a cluster of eight
small CTAs, so the routing occupies few SMs next to the side-stream kernels it runs with.

* Input: FP32 router logits ``[M, 896]`` (``1 <= M <= 256``).
* Top-k, bit-identical to ``noaux_tc_op`` on the same logits (selection, order, weights). A
  team of ``G`` warps routes one token, lane ``g`` of the team owning experts
  ``j * 32 G + g``. Each lane forms the order-preserving bits of ``sigmoid(logit) + bias`` (the
  high word of ``noaux_tc_op``'s 64-bit key, whose low word ``65535 - expert`` breaks ties
  toward smaller experts). In each warp the 16th largest lane maximum is a lower bound of the
  token's 16th largest score; the team keeps the largest of these bounds. Every expert at or
  above it (usually about 20 per token) is a candidate: warps compact theirs into shared
  memory and rank them by their full keys. If a warp has more than 32 (64 for one-warp teams)
  candidates (heavy ties), its team's first warp instead runs 16 exact arg-max rounds over all
  896 keys. The candidates are picked on ``tanh.approx`` scores and ranked by the exact ones
  (:meth:`K3RouteTailKernel.select_token`). The weights are the ``sigmoid`` scores
  renormalized like ``noaux_tc_op``.
* Sort (``moe_sort`` metadata of one expert-parallel rank): CTA ``c`` of the cluster owns a
  contiguous block of at most 32 tokens and marks, per local expert, a token bitmask in shared
  memory, then stores it into every CTA of the cluster (distributed shared memory). After a
  cluster barrier each CTA scans the per-expert tile counts (``ceil(count / 128)`` tiles in
  local-expert order) and writes the rows of its own tokens: rows of an expert follow the token
  order, which is what ``moe_sort``'s single-CTA kernels produce (its cluster kernel orders rows
  by atomics). CTA 0 writes the tile arrays.
* Top-k only, ``256 < M <= 2048`` (:class:`K3RouteWideKernel`, :func:`run_k3_route_wide`):
  independent CTAs of 16 tokens run the same top-k.

The kernels keep no state between launches, so a captured CUDA graph replays them as is.
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

from .rubin_fused_router import (
    NUM_EXPERTS,
    TOP_K,
    _asm,
    _div_rn_f32,
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
#: Score margin of the approximate candidate pass, at least twice the largest difference between
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


def _parse_route_tail_cfg(spec: str) -> tuple[tuple[int, int, int, int], ...]:
    """Bands ``(max_m, num_ctas, num_warps, warps_per_token)`` of a routing tail band list
    (:data:`HELIX8_ROUTE_TAIL_CFG`).

    ``spec`` is a comma list of ``<max_m>:<num_ctas>x<num_warps>x<warps_per_token>`` with strictly
    ascending ``max_m`` (at most :data:`MAX_M`); empty means no override. Each band must be a
    configuration :class:`K3RouteTailKernel` accepts for every token count it covers.
    """
    bands: list[tuple[int, int, int, int]] = []
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        try:
            max_m_text, cfg_text = item.split(":")
            ctas_text, warps_text, team_text = cfg_text.lower().split("x")
            band = (int(max_m_text), int(ctas_text), int(warps_text), int(team_text))
        except ValueError as err:
            raise ValueError(
                f"routing tail bands: {item!r} is not <max_m>:<ctas>x<warps>x<team>"
            ) from err
        max_m, ctas, warps, team = band
        if bands and max_m <= bands[-1][0]:
            raise ValueError(f"routing tail bands: max_m must ascend ({item!r})")
        if not 1 <= max_m <= MAX_M:
            raise ValueError(f"routing tail bands: max_m must be in [1, {MAX_M}] ({item!r})")
        if not 1 <= ctas <= MAX_CTAS or (max_m + ctas - 1) // ctas > _MAX_TOKENS_PER_CTA:
            raise ValueError(
                f"routing tail bands: {item!r} needs 1..{MAX_CTAS} CTAs with at most "
                f"{_MAX_TOKENS_PER_CTA} tokens per CTA"
            )
        if team not in (1, 2, 4, 7) or not 1 <= warps <= 32 or warps % team:
            raise ValueError(
                f"routing tail bands: {item!r} needs teams of 1, 2, 4 or 7 warps that "
                "divide the 1..32 warps of a CTA"
            )
        if team > 1 and warps // team > 15:
            raise ValueError(f"routing tail bands: at most 15 multi-warp teams ({item!r})")
        bands.append(band)
    return tuple(bands)


#: Launch configuration of the cluster routing tail (:func:`run_k3_route_tail`) by token count on
#: K3.1's helix8 generation server on SM107 (``route_tail_bands == "helix8"``), instead
#: of :func:`default_config`, whose choice up to 128 tokens was measured on GB300. Comma-separated
#: ``<max_m>:<num_ctas>x<num_warps>x<warps_per_token>`` bands in ascending ``max_m``: the first
#: band with ``m <= max_m`` applies, token counts above the last band keep
#: :func:`default_config`. They were measured on SM107 helix8 generation steps of 8 to 24
#: requests. Every configuration runs the same per-token algorithm, so the routing is
#: bit-identical. The multi-CTA kernel above 256 tokens is not affected; every other server keeps
#: :func:`default_config`.
HELIX8_ROUTE_TAIL_CFG = "32:8x8x2,48:8x12x2,64:8x8x1,96:8x12x1,128:8x16x1,256:8x32x1"
HELIX8_ROUTE_TAIL_BANDS = _parse_route_tail_cfg(HELIX8_ROUTE_TAIL_CFG)
_ROUTE_TAIL_CFG_LOGGED: set = set()


def _route_tail_bands(route_tail_bands: str = "") -> tuple[tuple[int, int, int, int], ...]:
    """The routing tail bands ``route_tail_bands`` names: :data:`HELIX8_ROUTE_TAIL_BANDS` for
    ``"helix8"`` (K3.1's helix8 generation server), none for ``""`` (:func:`default_config` for
    every token count)."""
    bands = route_tail_bands
    if bands == "helix8":
        return HELIX8_ROUTE_TAIL_BANDS
    if bands:
        raise ValueError(f'route_tail_bands must be "" or "helix8", got {bands!r}')
    return ()


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
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


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
        with_sort: Also produce the ``moe_sort`` outputs.
        local_cap: Largest number of local experts served.

    The launch is a programmatic dependent launch (the logits come from the previous kernel).
    """

    def __init__(
        self,
        num_ctas: int,
        num_warps: int,
        warps_per_token: int,
        with_sort: bool,
        local_cap: int = 224,
    ):
        if not 1 <= num_ctas <= MAX_CTAS:
            raise ValueError(f"num_ctas={num_ctas} must be in [1, {MAX_CTAS}]")
        if warps_per_token not in (1, 2, 4, 7) or num_warps % warps_per_token:
            raise ValueError(f"warps_per_token={warps_per_token} must be 1, 2, 4 or 7")
        if num_warps // warps_per_token > 15 and warps_per_token > 1:
            raise ValueError("at most 15 multi-warp teams (named barriers 1..15)")
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
        self.with_sort = with_sort
        # Local experts handled per thread in the sort scan (contiguous).
        self.experts_per_thread = (local_cap + self.threads - 1) // self.threads
        self.max_tiles_per_expert = (MAX_M + TILE - 1) // TILE
        # Skip candidate-ranking rounds in which the warp has no candidate.
        self.skip_empty_rank_rounds = False
        # Candidates picked on tanh.approx scores (see select_token).
        self.approx = True

    @cute.jit
    def __call__(
        self,
        m: Int32,
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

        self.kernel(
            m,
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
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.num_ctas, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
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
        if cutlass.const_expr(self.with_sort):
            # Distributed shared memory may only be accessed once every CTA of the cluster runs: arrive here, wait
            # before the mask broadcast.
            _cluster_arrive_relaxed()
        _griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            self.topk_token(
                token,
                token - t_begin,
                team,
                team_warp,
                lane,
                logits_addr,
                bias_smem,
                ids_addr,
                vals_addr,
                scale_bits,
                cand_smem,
                top_smem,
                xchg_smem,
                ids_smem,
                mask_smem,
                local_offset,
                local_num,
            )
        _griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.with_sort):
            cute.arch.barrier()
            # Every CTA of the cluster has started (arrive after the bias staging).
            _cluster_wait_acquire()
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
        """Logit ``s * stride + g`` (``s < n``) of the token whose logits start at ``row``. A
        subclass may read them differently (the sharded MoE front polls them)."""
        return [_ld_f32(row + Int64((s * stride + g) * 4)) for s in range(n)]

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        """Logits ``128 j + 4 lane .. + 3`` (``j < n``) of the token whose logits start at
        ``row``, concatenated. Not ``cute.jit`` (the DSL preprocessor does not rewrite this
        method), so the loop is a plain Python loop, unrolled at trace time."""
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
        logits_addr: Int64,
        bias_smem: Int32,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
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
            logits_addr,
            bias_smem,
            scale_bits,
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
        logits_addr: Int64,
        bias_smem: Int32,
        scale_bits: Int64,
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
        logit = self.load_logits(row, g, stride, n_slots)
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
                expert, score = self.topk_rounds(row, bias_smem, lane)

        weight = Float32(0.0)
        if team_warp == 0:
            total = _warp_sum_f32(score)
            # Exact FP32 fast path of noaux_tc's FP64 expression (see rubin_fused_router).
            if _unit_scale_fast_path(scale_bits, total) != 0:
                weight = _div_rn_f32(score, total)
            else:
                weight = _renormalize(score, total, scale_bits)
        return expert, weight

    @cute.jit
    def topk_rounds(self, row: Int64, bias_smem: Int32, lane: Int32):
        """Exact fallback by one warp: 16 arg-max rounds over all 896 keys of the token (lane
        ``l`` owns experts ``128 j + 4 l + i``); lane ``k`` gets the ``k``-th expert."""
        logit = self.load_logits4(row, lane, _BLOCKS)
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
                words.append(_lds_b32(masks_smem + (c * self.local_cap + jj) * 4))
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
# Multi-CTA top-k (M > 256)
# ---------------------------------------------------------------------------

#: Largest token count of the multi-CTA top-k (:class:`K3RouteWideKernel`).
WIDE_MAX_M = 2048
#: Tokens (and warps) per CTA of the multi-CTA top-k: one token per warp.
_WIDE_TOKENS_PER_CTA = 16
#: Smem words of the bias, ahead of the per-team regions.
_WIDE_BIAS_WORDS = NUM_EXPERTS


class K3RouteWideKernel(K3RouteTailKernel):
    """Top-16 of up to :data:`WIDE_MAX_M` tokens on a grid of independent CTAs (no sort).

    CTA ``c`` routes tokens ``[T c, T c + T)`` (``T = 16``, one token per warp) with the teams of
    :class:`K3RouteTailKernel` (same selection code, so the same top-k), picking the candidates
    on the exact scores. No CTA waits on another, so the grid needs no co-residency, and the
    kernel keeps no state between launches.
    """

    def __init__(self):
        super().__init__(1, _WIDE_TOKENS_PER_CTA, 1, False)
        self.approx = False
        self.tokens = _WIDE_TOKENS_PER_CTA
        self.cand_words = self.num_warps * self.region_slots * 4
        self.topk_words = (
            _WIDE_BIAS_WORDS + self.cand_words + self.teams * TOP_K * 2 + self.teams * 16
        )
        self.skip_empty_rank_rounds = True

    @cute.jit
    def __call__(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            # Bias, candidate regions, top-16 and exchange slots.
            scratch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.topk_words], 16]

        self.shared_storage = SharedStorage

        num_ctas = (m + (self.tokens - 1)) // self.tokens
        self.kernel(
            m,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            scale_bits,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.scratch.data_ptr().toint()
        cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
        top_all = cand_all + self.cand_words * 4
        xchg_all = top_all + self.teams * TOP_K * 8
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = cand_all + team * (self.team_warps * self.region_slots * _CAND_BYTES)
        top_smem = top_all + team * (TOP_K * 8)
        xchg_smem = xchg_all + team * 64

        t_begin = cta * self.tokens
        t_end = t_begin + self.tokens
        if t_end > m:
            t_end = m

        # The bias is a weight: stage it in shared memory before waiting on the previous grid.
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
            _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        _griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            expert, weight = self.select_token(
                token,
                team,
                team_warp,
                lane,
                logits_addr,
                bias_smem,
                scale_bits,
                cand_smem,
                top_smem,
                xchg_smem,
            )
            if team_warp == 0:
                if lane < TOP_K:
                    out = (token * TOP_K + lane) * 4
                    _st_global_b32(ids_addr + Int64(out), expert)
                    _st_global_f32(vals_addr + Int64(out), weight)
        _griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

#: Local-expert capacity of the common build (expert parallelism >= 4); more use a larger one.
_SMALL_LOCAL_CAP = 224


def default_config(m: int) -> tuple[int, int, int]:
    """``(num_ctas, warps_per_token, num_warps)`` for ``m`` tokens (at most 32 tokens per CTA).

    Measured on GB300 (standalone, behind a producer kernel): eight 8-warp CTAs with two-warp
    teams up to 32 tokens, one-warp teams up to 64; 16-warp CTAs above (the helix8 bands of
    :data:`HELIX8_ROUTE_TAIL_CFG` take 32-warp CTAs above 128 tokens instead).
    """
    if m <= 32:
        return MAX_CTAS, 2, 8
    if m <= 64:
        return MAX_CTAS, 1, 8
    return MAX_CTAS, 1, 16


def tail_config(m: int, route_tail_bands: str = "") -> tuple[int, int, int]:
    """``(num_ctas, warps_per_token, num_warps)`` of :func:`run_k3_route_tail` for ``m`` tokens.

    The band of ``route_tail_bands`` (:func:`_route_tail_bands`) covering ``m`` (``"helix8"``:
    K3.1's helix8 generation server), else :func:`default_config` (``""``: always
    :func:`default_config`).
    """
    for max_m, num_ctas, num_warps, warps_per_token in _route_tail_bands(route_tail_bands):
        if m <= max_m:
            if max_m not in _ROUTE_TAIL_CFG_LOGGED:
                _ROUTE_TAIL_CFG_LOGGED.add(max_m)
                from tensorrt_llm.logger import logger

                logger.info(
                    f"Kimi K3.1 helix8 default: routing tail up to {max_m} tokens on {num_ctas} "
                    f"CTAs x {num_warps} warps, {warps_per_token}-warp teams"
                )
            return num_ctas, warps_per_token, num_warps
    return default_config(m)


@functools.cache
def _compiled_tail(
    num_ctas: int,
    num_warps: int,
    warps_per_token: int,
    with_sort: bool,
    local_cap: int,
):
    kernel = K3RouteTailKernel(num_ctas, num_warps, warps_per_token, with_sort, local_cap)
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(num_ctas),
        *([dummy] * 10),
        Int32(0),
        Int32(NUM_EXPERTS),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _check_inputs(logits: torch.Tensor, e_score_correction_bias: torch.Tensor) -> None:
    if logits.dtype != torch.float32 or logits.dim() != 2:
        raise ValueError("logits must be a 2D [M, 896] float32 tensor")
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


def run_k3_route_tail(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sort_outputs: tuple[torch.Tensor, ...] | None,
    local_expert_offset: int,
    local_num_experts: int,
    *,
    route_tail_bands: str = "",
) -> None:
    """Launch the routing tail on the current stream (inputs are not re-validated).

    Args:
        logits: FP32 ``[M, 896]`` logits, ``1 <= M <= 256``.
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
        route_tail_bands: The launch configuration bands by token count, ``"helix8"`` or
            ``""`` (see :func:`tail_config`).

    The candidates are picked on ``tanh.approx`` scores and only they get ``noaux_tc``'s exact
    sigmoid (the accurate ``tanhf`` of all 896 experts dominates the top-k); the selection stays
    bit-identical: see :meth:`K3RouteTailKernel.select_token`.
    """
    m = logits.shape[0]
    num_ctas, warps_per_token, num_warps = tail_config(m, route_tail_bands)
    if (m + num_ctas - 1) // num_ctas > _MAX_TOKENS_PER_CTA:
        raise ValueError(f"M={m} needs more than {num_ctas} CTAs")
    with_sort = sort_outputs is not None
    local_cap = _SMALL_LOCAL_CAP if local_num_experts <= _SMALL_LOCAL_CAP else NUM_EXPERTS
    compiled = _compiled_tail(num_ctas, num_warps, warps_per_token, with_sort, local_cap)
    if with_sort:
        tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_outputs)
    else:
        tile_expert = tile_limit = e2p = p2e = total = num_tiles = 0
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
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
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


@functools.cache
def _compiled_wide():
    dummy = Int64(1 << 20)
    return cute.compile(
        K3RouteWideKernel(),
        Int32(MAX_M + 1),
        *([dummy] * 4),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_k3_route_wide(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Launch the multi-CTA top-k on the current stream (inputs are not re-validated): the
    arguments and top-k outputs of :func:`run_k3_route_tail` without the sort, for
    ``1 <= M <= WIDE_MAX_M``."""
    m = logits.shape[0]
    if not 1 <= m <= WIDE_MAX_M:
        raise ValueError(f"num_tokens={m} must be in [1, {WIDE_MAX_M}]")
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    _compiled_wide()(
        Int32(m),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(scale_bits),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


def _route_tail_outputs(
    logits: torch.Tensor,
    m: int,
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
    route_tail_bands: str = "",
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
        logits: ``fp32 [M, 896]`` router logits, ``1 <= M <= 256``.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.
        tile_tokens_dim: Must be 128.
        route_tail_bands: See :func:`run_k3_route_tail`.

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
    m = logits.shape[0]
    if not 1 <= m <= MAX_M:
        raise ValueError(f"num_tokens={m} must be in [1, {MAX_M}]")
    if not (
        0 <= local_expert_offset
        and 1 <= local_num_experts
        and local_expert_offset + local_num_experts <= NUM_EXPERTS
    ):
        raise ValueError("invalid local expert range")
    outs = _route_tail_outputs(logits, m, local_num_experts, tile_tokens_dim)
    run_k3_route_tail(
        logits,
        e_score_correction_bias,
        routed_scaling_factor,
        outs[0],
        outs[1],
        outs[2:],
        local_expert_offset,
        local_num_experts,
        route_tail_bands=route_tail_bands,
    )
    return outs


__all__ = [
    "MAX_M",
    "WIDE_MAX_M",
    "K3RouteTailKernel",
    "K3RouteWideKernel",
    "default_config",
    "k3_route_tail",
    "run_k3_route_tail",
    "run_k3_route_wide",
]
