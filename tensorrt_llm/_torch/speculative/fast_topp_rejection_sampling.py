# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused rejection sampling with a top-p (nucleus) filter for one-model block drafters.

:mod:`fast_rejection_sampling` covers ``AdvancedSamplingMode.NO_TOPK_NO_TOPP`` (temperature-only
sampling). This module adds ``AdvancedSamplingMode.NO_TOPK``: the target and the draft distributions
are both filtered by top-p, as the reference does (``softmax`` then flashinfer's AIR
``top_p_renorm_probs`` on the target rows and on the draft rows), and then the same chain rejection
sampling runs on the renormalised nuclei. The Philox streams, the verify / draft layout and the
buffers are those of ``fast_rejection_sampling``, unchanged.

Nucleus semantics (flashinfer AIR top-p renorm): with ``w_v = exp(z_v - max z)``, ``S = sum w`` and
``z = logits / T``, the pivot is the first token, in descending order of ``z``, at which the
cumulative mass reaches ``top_p * S``; every token with ``z >= z_pivot`` (ties included) is kept and
renormalised: ``p~_v = w_v / Z`` with ``Z`` the kept mass. Rows with ``top_p >= 1`` keep everything.

Nucleus search, no sort of the vocabulary: the rows are processed by a pipeline of kernels with one
program per ``(row, chunk of the vocabulary)`` (hundreds of programs per step, instead of one program
per row), and every kernel after the search reads only the per-row nucleus record ``rec`` (``z_pivot``,
``Z``, ``S``, status, ``max z``, ``1/T``, padding flag), so the target rows, the draft rows and the
verify kernels classify every token by the same comparison ``z >= z_pivot``:

1. :func:`_topp_stats_kernel`: per chunk ``(max z, sum exp(z - max z), number of tokens equal to the
   maximum)``. The last program of a row (an atomic ticket) combines the partials in index order into
   the row descriptor ``rd``: ``max z``, ``S``, ``target = top_p * S``, the search mode (the row keeps
   the whole vocabulary / its maximum alone reaches the target / a search is needed) and the window
   of the histogram. The pivot is bounded: the tokens at or below it carry more than ``(1 - p) * S``
   of the mass and each weighs at most ``exp(-depth)``, so ``depth < ln(V / ((1 - p) * S))``.
2. :func:`_topp_hist_kernel`: counts the tokens of ``[max z - depth, max z]`` in ``NB`` equal-width
   bins (integer atomics). Counts bound each bin's mass (``count * [w_low, w_high]``) and so the
   cumulative mass; the last program of the row selects the *zone* (:func:`_zone_from_hist`): either
   every token at or above ``z_lo`` is a candidate (mode 1) or the pivot lies among the tokens of
   ``[z_lo, z_hi]`` and the mass above ``z_hi`` is summed exactly (mode 2), when at most ``CAP``
   candidates result.
3. :func:`_topp_gather_kernel`: each chunk compacts its zone tokens (at most ``CHUNK_CAP``, counted
   exactly) and, in mode 2, the mass above and inside the zone. A chunk is a set of interleaved
   256-token segments spread over the vocabulary (the probable tokens of a language model cluster at
   the lowest ids, which would overflow one contiguous chunk long before ``CAP`` candidates).
4. :func:`_topp_resolve_kernel` (one program per row): concatenates the candidates, sorts them
   (three size classes) and takes as pivot the first one, in descending order, at which the running mass
   (started at the exact mass above the zone) reaches the target. ``Z`` is the mass of every token at or
   above the pivot (ties included, no special case). A list longer than ``bin_min`` is first narrowed
   (:func:`_bin_select_resolve`): its ``z`` range is split into ``sub_bins`` equal bins, the exact (fixed
   order) mass of every bin is summed, and only the bin where the running mass from the top reaches the
   target is sorted, starting from the mass of the bins above; a token's bin depends on its ``z`` alone,
   so a tied group is never split, and a sort of thousands of candidates (tens of microseconds of
   latency) shrinks to a few hundred. A zone whose tokens are all equal needs no sort; a row whose
   maximum alone reaches the target keeps the tokens tied with it (``Z`` from the tie count) without
   any pass over the row.
5. :func:`_topp_nucleus_kernel`, the exact fallback: one program per row that still has status
   pending (a chunk overflowed, more than ``CAP`` candidates, or the candidate mass is short of the
   target) refines the histogram zone pass by pass (each refinement narrows the zone by about
   ``NB / 5``). A violated certificate (only possible for non-finite input) or exhausted iterations
   set ``status != 0``; the row then keeps every token at or above the lowest ``z`` of the last
   certified zone (a superset of the exact nucleus), the whole vocabulary when no zone was certified.
   After its first pass the search knows the exact mass inside the window, so it bounds the mass
   below the window by ``S - mass above - mass inside`` instead of by counting each token below at the
   window's edge weight. Rows the pipeline resolved exit at once.

Exactness. The record is exact up to the rounding of float32 mass sums: a zone is accepted only when
the mass above it is below the target and the mass through it reaches the target, both compared with
a relative tolerance ``_MASS_TOL`` (the rounding of a float32 sum of up to ~1e5 weights); only a token
boundary within that tolerance of the target can differ from the exact-arithmetic nucleus, and there
flashinfer's own float32 ternary search is arbitrary as well. The pivot is always the stored ``z`` of
a token of the row, so no comparison involves computed thresholds.

Determinism. The only atomics are integer (tickets and histogram counts, order independent); every
floating-point sum is of fixed-order per-chunk partials combined by one program in index order, so
the result is a pure function of the logits: tensor-parallel ranks holding the same logits find
bit-identical nuclei and draws. The ticket counters and the histograms live in one control buffer
that is zeroed before every call (a buffer first allocated inside a CUDA-graph capture is not
reliably zero until the first replay, so the kernels do not reset counters themselves).

The per-call scratch (partials, descriptors, records, candidate lists) comes from the grow-only
workspace, whose retired buffers are never freed, so CUDA graphs captured earlier stay valid.
"""

import dataclasses
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .fast_rejection_sampling import (
    _WORKSPACE,
    _check_draft_args,
    _check_request_args,
    _curand_uniform,
    _inv_temperature,
    _load_scaled,
    _markov_feedback_emb_kernel,
    _num_chunks,
    _pick_chunk,
    _rs_draft_pick_kernel,
)

# Row descriptor ``rd[row]`` (16 x fp32), written by the last stats program of the row:
#   0 1/T (0 for T = 0)       1 padding row (0 / 1)      2 max z        3 S = sum exp(z - max z)
#   4 search mode: 0 the row keeps the whole vocabulary, 1 the maximum alone reaches the target, 2 a
#     nucleus search is needed                           5 lo, 6 depth: the histogram window [lo, max z]
#   7 target = top_p * S                                 8 number of tokens tied with the maximum
_RD_WIDTH = 16
# ``rec[row]`` (8 x fp32) = (z_pivot, Z, S, status, max z, 1/T, padding, spare): the nucleus of one row,
# everything the draft-write and verify kernels need.
_REC_WIDTH = 8
STATUS_OK = 0
STATUS_INVALID_ZONE = 1
STATUS_NOT_RESOLVED = 2
STATUS_PENDING = 3  # handed to the single-program kernel
# Vocab elements per program of the draft-write / verify kernels (the chunking of the draft
# proposals of ``fast_rejection_sampling``).
_CHUNK = 4096
_NUM_WARPS = 8
# Largest top_p below one that the nucleus search handles (1 - 2^-24 is the largest float32 < 1).
_MAX_TOP_P = 0.99999994
# Relative size of the float32 rounding of a sum of up to ~1e5 weights (tree sums over tiles, then over
# tiles): a zone whose bracket of the target fails by less than this is rounding, not an error.
_MASS_TOL = tl.constexpr(4e-6)


@dataclasses.dataclass(frozen=True)
class NucleusConfig:
    """Compile-time shape of the nucleus search kernels (``bins``, ``cap`` powers of two).

    ``fast`` enables the chunk-parallel pipeline (stats, histogram, gather, resolve kernels); the
    single-program-per-row kernel (``tile``, ``bins``, ``iters``, ``warps``) then only runs for the rows
    that pipeline hands over (``fast=False`` runs it for every row)."""

    tile: int = 8192  # single-program kernel: vocab elements per loop iteration of a pass
    bins: int = 512  # single-program kernel: histogram bins
    cap: int = 4096  # candidate tokens classified by one sort
    cap_small: int = 256  # candidate lists up to this size use the cheapest sort
    iters: int = 6  # single-program kernel: refinement iterations before the row falls back
    warps: int = 16  # single-program kernel
    fast: bool = True
    hist_bins: int = 512  # chunk-parallel histogram bins
    chunk_cap: int = 256  # candidates one gather program can hand to the resolve step
    pipe_warps: int = 8  # histogram / gather kernels
    resolve_warps: int = 8  # resolve kernel
    hist_maxnreg: int = 0  # register cap of the histogram kernel (0: compiler's choice)
    gather_maxnreg: int = 0  # register cap of the gather kernel
    hg_chunk: int = 4096  # vocab elements per histogram / gather program
    stats_chunk: int = 8192  # vocab elements per stats program
    cap_mid: int = 1024  # candidate lists up to this size use the middle sort
    zone_small: int = 0  # > 0: gather only a zone of at most this many tokens, not all above it
    bin_min: int = 256  # candidate lists longer than this are narrowed to one bin before the sort
    sub_bins: int = 32  # equal-width bins of that narrowing (a power of two)
    sub_slab: int = 128  # candidates per iteration of the bin-mass loop (a power of two)
    gather_seg: int = 256  # tokens per gathered segment, spread over the vocabulary (0: one chunk)

    def __post_init__(self):
        seg = self.gather_seg
        if seg < 0 or (seg > 0 and (seg & (seg - 1) or self.hg_chunk % seg)):
            raise ValueError(
                f"gather_seg {seg} must be 0 or a power of two dividing hg_chunk {self.hg_chunk}"
            )

    @property
    def small(self) -> int:
        return min(self.cap_small, self.cap)

    @property
    def mid(self) -> int:
        return min(max(self.cap_mid, self.small), self.cap)


DEFAULT_NUCLEUS = NucleusConfig()


@triton.jit
def _topp_hist_pass(
    logits_ptr,
    row,
    stride_row,
    bias_ptr,
    stride_bias,
    inv_t,
    is_pad,
    vocab,
    lo,
    hi,
    TILE: tl.constexpr,
    NB: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Counts of the row's tokens with ``lo <= z <= hi`` in ``NB`` equal-width bins of ``[lo, hi]``."""
    cnt = tl.zeros([NB], tl.int32)
    scale = NB / tl.maximum(hi - lo, 1e-30)
    for t in range(0, vocab, TILE):
        offs = t + tl.arange(0, TILE)
        mask = offs < vocab
        z = _load_scaled(
            logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad,
            HAS_BIAS,
        )  # fmt: skip
        inside = mask & (z >= lo) & (z <= hi)
        j = tl.minimum(tl.maximum((z - lo) * scale, 0.0), NB - 1.0)
        cnt += tl.histogram(j.to(tl.int32), NB, mask=inside)
    return cnt


@triton.jit
def _sort_resolve(zz_row_ptr, n_mem, a_above, target, big_m, SIZE: tl.constexpr):
    """Pivot and kept mass of a zone of ``n_mem <= SIZE`` tokens stored at ``zz_row_ptr``.

    ``a_above`` is the mass of the tokens above the zone. Returns ``(z_pivot, kept, ok)``. If the
    running mass falls short of the target by float rounding (``_MASS_TOL``), the zone's last token is
    the pivot. Lanes past ``n_mem`` are padding (``-inf``, weight 0): a parallel scan may round their
    running mass to the other side of the target than the last real lane's, so only real lanes can
    be the pivot."""
    oi = tl.arange(0, SIZE)
    zv = tl.load(zz_row_ptr + oi, mask=oi < n_mem, other=float("-inf"))
    zs = tl.sort(zv, descending=True)
    ws = tl.exp(zs - big_m)
    cum = a_above + tl.cumsum(ws, axis=0)
    k = tl.min(tl.where((cum >= target) & (oi < n_mem), oi, SIZE), axis=0)
    total = a_above + tl.sum(ws, axis=0)
    k = tl.where((k >= SIZE) & (total >= target - _MASS_TOL * target), n_mem - 1, k)
    z_pivot = tl.sum(tl.where(oi == k, zs, 0.0), axis=0)
    kept = a_above + tl.sum(tl.where(zs >= z_pivot, ws, 0.0), axis=0)
    return z_pivot, kept, (k < SIZE).to(tl.int32)


@triton.jit
def _zone_certified(a_above, m_zone, n_mem, target):
    """Whether a zone of ``n_mem`` tokens, with the exact mass ``a_above`` above it and ``m_zone``
    inside it, brackets the target: the mass above it does not exceed the target and the mass through
    it reaches it (both to the float32 rounding of the sums, ``_MASS_TOL``), so the pivot is one of its
    tokens. ``<=`` keeps a zone that starts at the maximum (``a_above = 0``) valid for ``top_p = 0``,
    whose target is 0."""
    tol = _MASS_TOL * target
    return (a_above <= target + tol) & (a_above + m_zone >= target - tol) & (n_mem > 0)


@triton.jit
def _bin_select_resolve(
    dense_row_ptr,
    sub_row_ptr,
    n_mem,
    a_above,
    target,
    big_m,
    CAP: tl.constexpr,
    CAP_SMALL: tl.constexpr,
    CAP_MID: tl.constexpr,
    NS: tl.constexpr,
    SLAB: tl.constexpr,
):
    """:func:`_sort_resolve` for a long candidate list (``n_mem <= CAP`` tokens at ``dense_row_ptr``)
    without sorting all of it: the candidates are split into ``NS`` equal-width bins of their ``z``
    range, the exact mass of every bin is summed (fixed order), the bin where the running mass from the
    top reaches the target holds the pivot, and only that bin's tokens (copied to ``sub_row_ptr``) are
    sorted, starting from the mass of the bins above. A token's bin is a function of its ``z`` alone, so
    a tied group is never split. Returns ``(z_pivot, kept, ok)`` like :func:`_sort_resolve`; ``ok`` is
    0 when the chosen bin is empty (the running mass crossed the target by rounding only)."""
    oi = tl.arange(0, CAP)
    valid = oi < n_mem
    zv = tl.load(dense_row_ptr + oi, mask=valid, other=float("-inf"))
    lo_b = tl.min(tl.where(valid, zv, float("inf")), axis=0)
    hi_b = tl.max(zv, axis=0)
    scale = NS / tl.maximum(hi_b - lo_b, 1e-30)
    bins = tl.arange(0, NS)
    acc = tl.zeros([NS, SLAB], tl.float32)
    for s in range(0, n_mem, SLAB):
        oj = s + tl.arange(0, SLAB)
        vj = oj < n_mem
        zs = tl.load(dense_row_ptr + oj, mask=vj, other=lo_b)
        js = tl.minimum(((zs - lo_b) * scale).to(tl.int32), NS - 1)
        ws = tl.where(vj, tl.exp(zs - big_m), 0.0)
        acc += tl.where(js[None, :] == bins[:, None], ws[None, :], 0.0)
    sm = tl.sum(acc, axis=1)
    cum_top = a_above + tl.cumsum(sm, axis=0, reverse=True)
    total = a_above + tl.sum(sm, axis=0)
    j_star = tl.max(tl.where(cum_top >= target, bins, -1), axis=0)
    # The mass is short of the target by rounding only: the pivot is in the lowest bin.
    j_star = tl.where((j_star < 0) & (total >= target - _MASS_TOL * target), 0, j_star)
    a_sub = a_above + tl.sum(tl.where(bins > j_star, sm, 0.0), axis=0)
    jv = tl.minimum(((tl.where(valid, zv, lo_b) - lo_b) * scale).to(tl.int32), NS - 1)
    mem = valid & (jv == j_star)
    mem_i = mem.to(tl.int32)
    tl.store(sub_row_ptr + tl.cumsum(mem_i, axis=0) - 1, zv, mask=mem)
    n_sub = tl.sum(mem_i, axis=0)
    tl.debug_barrier()
    if n_sub <= CAP_SMALL:
        zp, zk, ok = _sort_resolve(sub_row_ptr, n_sub, a_sub, target, big_m, CAP_SMALL)
    elif n_sub <= CAP_MID:
        zp, zk, ok = _sort_resolve(sub_row_ptr, n_sub, a_sub, target, big_m, CAP_MID)
    else:
        zp, zk, ok = _sort_resolve(sub_row_ptr, n_sub, a_sub, target, big_m, CAP)
    return zp, zk, tl.where((n_sub > 0) & (j_star >= 0), ok, 0)


@triton.jit
def _zone_from_hist(
    cnt,
    big_m,
    big_s,
    target,
    lo,
    hi,
    vocab,
    CAP: tl.constexpr,
    SMALL_ZONE: tl.constexpr,
    NB: tl.constexpr,
):
    """What the chunk-parallel resolve step has to look at, from the bin counts (tokens of
    ``[lo, hi]``, ``NB`` equal bins). Returns ``(z_lo, z_hi, mode, found)``:

    * mode 1: every token with ``z >= z_lo`` is a candidate (at most ``CAP`` of them) and holds the
      pivot, so the resolve step sorts them (``z_hi`` is +inf);
    * mode 2: the pivot lies among the at most ``CAP`` tokens of ``[z_lo, z_hi]``; the mass above
      ``z_hi`` is summed exactly (``SMALL_ZONE > 0`` prefers it when the zone holds at most that many
      tokens and the tokens above it do not);
    * mode 0: neither (an inconsistent histogram, or more than ``CAP`` tokens in the zone).
    """
    idx = tl.arange(0, NB)
    dlt = (hi - lo) / NB
    e_lo = lo + idx.to(tl.float32) * dlt
    e_hi = lo + (idx + 1).to(tl.float32) * dlt
    w_max = tl.exp(tl.minimum(e_hi, big_m) - big_m)
    w_min = tl.exp(e_lo - big_m)
    cf = cnt.to(tl.float32)
    f_min = cf * w_min
    f_max = cf * w_max
    # The maximum itself weighs exactly 1 and sits in the top bin.
    top = idx == NB - 1
    f_min = tl.where(top & (cnt > 0), 1.0 + (cf - 1.0) * w_min, f_min)
    # Mass from the top through bin j: bounded from above by the bins' mass (top-down) and by S minus
    # the bins below (bottom-up), from below likewise; tokens below `lo` weigh at most exp(lo - max).
    n_below = vocab - tl.sum(cnt, axis=0)
    below_max = n_below.to(tl.float32) * tl.exp(lo - big_m)
    bot_min = tl.cumsum(f_min, axis=0) - f_min
    bot_max = tl.cumsum(f_max, axis=0) - f_max
    ub = tl.minimum(tl.cumsum(f_max, axis=0, reverse=True), big_s - bot_min)
    lb = tl.maximum(tl.cumsum(f_min, axis=0, reverse=True), big_s - bot_max - below_max)
    j_up = tl.max(tl.where(ub >= target, idx, -1), axis=0)
    j_dn = tl.max(tl.where(lb >= target, idx, -1), axis=0)
    # The pivot is not below the lowest populated bin either.
    j_first = tl.min(tl.where(cnt > 0, idx, NB), axis=0)
    jlo = tl.maximum(tl.maximum(j_dn - 1, 0), j_first)
    jhi = tl.minimum(j_up + 1, NB - 1)
    z_lo = lo + jlo.to(tl.float32) * dlt
    z_hi = tl.where(jhi == NB - 1, hi, lo + (jhi + 1).to(tl.float32) * dlt)
    n_up = tl.sum(tl.where(idx >= jlo, cnt, 0), axis=0)
    n_zone = tl.sum(tl.where((idx >= jlo) & (idx <= jhi), cnt, 0), axis=0)
    found = j_up >= 0
    mode1 = found & (n_up <= CAP)
    if SMALL_ZONE > 0:
        mode1 = mode1 & ((n_up <= SMALL_ZONE) | (n_zone > SMALL_ZONE))
    mode2 = found & (~mode1) & (n_zone <= CAP)
    z_hi = tl.where(mode1, float("inf"), z_hi)
    mode = tl.where(mode1, 1.0, tl.where(mode2, 2.0, 0.0))
    return z_lo, z_hi, mode, found


@triton.jit
def _rd_load(rd_ptr, row):
    """The row descriptor: ``(1/T, padding, max z, S, mode, lo, depth, target, ties at the max)``."""
    base = rd_ptr + row * 16
    return (
        tl.load(base),
        tl.load(base + 1) > 0.5,
        tl.load(base + 2),
        tl.load(base + 3),
        tl.load(base + 4),
        tl.load(base + 5),
        tl.load(base + 6),
        tl.load(base + 7),
        tl.load(base + 8),
    )


@triton.jit
def _topp_stats_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    temps_ptr,
    temps_div,
    topp_ptr,
    topp_div,
    slot_ptr,
    dummy_row,
    rows_per_req,
    pmax_ptr,
    psum_ptr,
    pties_ptr,
    ctl_ptr,
    ctl_stride,
    rd_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Per chunk, ``(max, sum exp(x - max))`` plus the number of tokens equal to the chunk's maximum;
    the last program of a row (an atomic ticket; ``ctl`` is zeroed before every call) combines the
    partials, in a fixed order, into the row descriptor (:data:`_RD_WIDTH`) that every later kernel
    reads instead."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    inv_t = _inv_temperature(tl.load(temps_ptr + row // temps_div))
    is_pad = tl.load(slot_ptr + row // rows_per_req) == dummy_row
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = _load_scaled(
        logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad, HAS_BIAS
    )
    x = tl.where(mask, x, float("-inf"))
    m = tl.max(x, 0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    s = tl.sum(tl.where(mask, tl.exp(x - safe_m), 0.0), 0)
    ties = tl.sum(tl.where(mask & (x == m), 1, 0), 0)
    tl.store(pmax_ptr + row * num_chunks + ch, m)
    tl.store(psum_ptr + row * num_chunks + ch, s)
    tl.store(pties_ptr + row * num_chunks + ch, ties)
    tl.debug_barrier()
    ticket = tl.atomic_add(ctl_ptr + row * ctl_stride, 1)
    if ticket == num_chunks - 1:
        cidx = tl.arange(0, NC_POW2)
        cmask = cidx < num_chunks
        pm = tl.load(pmax_ptr + row * num_chunks + cidx, mask=cmask, other=float("-inf"),
                     cache_modifier=".cg")  # fmt: skip
        ps = tl.load(
            psum_ptr + row * num_chunks + cidx, mask=cmask, other=0.0, cache_modifier=".cg"
        )
        pt = tl.load(pties_ptr + row * num_chunks + cidx, mask=cmask, other=0, cache_modifier=".cg")
        big_m = tl.max(pm, 0)
        safe = tl.where(big_m == float("-inf"), 0.0, big_m)
        scale = tl.where(cmask, tl.exp(pm - safe), 0.0)
        big_s = tl.sum(ps * scale, 0)
        ties_max = tl.sum(tl.where(cmask & (pm == big_m), pt, 0), axis=0).to(tl.float32)
        top_p = tl.load(topp_ptr + row // topp_div)
        tp = tl.minimum(tl.maximum(top_p, 0.0), 0.99999994)
        active = (~is_pad) & (inv_t != 0.0) & (top_p < 1.0) & (big_m > float("-inf"))
        target = tp * big_s
        # The pivot lies within `depth` of the maximum: the tokens at or below it carry more than
        # (1 - p) * S of the mass and each weighs at most exp(-depth), so depth < ln(V / ((1 - p) S));
        # S >= 1 for every row with a finite maximum.
        depth = tl.log(vocab.to(tl.float32) / ((1.0 - tp) * tl.maximum(big_s, 1.0))) + 0.25
        mode = tl.where(active, tl.where(target <= 1.0, 1.0, 2.0), 0.0)
        base = rd_ptr + row * 16
        tl.store(base, inv_t)
        tl.store(base + 1, is_pad.to(tl.float32))
        tl.store(base + 2, big_m)
        tl.store(base + 3, big_s)
        tl.store(base + 4, mode)
        tl.store(base + 5, big_m - depth)
        tl.store(base + 6, depth)
        tl.store(base + 7, target)
        tl.store(base + 8, ties_max)


@triton.jit
def _topp_hist_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    rd_ptr,
    ctl_ptr,
    ctl_stride,
    zone_ptr,
    vocab,
    num_chunks,
    NB: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
    SMALL_ZONE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Counts of the tokens of ``[max z - depth, max z]`` in ``NB`` equal-width bins (one program per
    ``(row, chunk)``, integer atomics into the row's histogram, which lives in ``ctl`` after the row's
    two tickets). The last program of a row (an atomic ticket) selects the zone from the counts
    (:func:`_zone_from_hist`) and publishes it in ``zone``. Rows that keep the whole vocabulary, and
    rows whose maximum alone reaches the target, do nothing."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    mode = tl.load(rd_ptr + row * 16 + 4)
    if mode > 1.5:
        inv_t, is_pad, big_m, big_s, _mode, lo, depth, target, _ties = _rd_load(rd_ptr, row)
        offs = ch * CHUNK + tl.arange(0, CHUNK)
        mask = offs < vocab
        z = _load_scaled(
            logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad,
            HAS_BIAS,
        )  # fmt: skip
        jf = (z - lo) * (NB / depth)
        cnt = tl.histogram(tl.minimum(jf, NB - 1.0).to(tl.int32), NB, mask=mask & (jf >= 0.0))
        idx = tl.arange(0, NB)
        ctl = ctl_ptr + row * ctl_stride
        tl.atomic_add(ctl + 2 + idx, cnt, mask=cnt > 0)
        tl.debug_barrier()
        ticket = tl.atomic_add(ctl + 1, 1)
        if ticket == num_chunks - 1:
            total = tl.load(ctl + 2 + idx, cache_modifier=".cg")
            z_lo, z_hi, zmode, found = _zone_from_hist(
                total, big_m, big_s, target, lo, big_m, vocab, CAP, SMALL_ZONE, NB
            )
            zone = zone_ptr + row * 4
            tl.store(zone, z_lo)
            tl.store(zone + 1, z_hi)
            tl.store(zone + 2, zmode)
            tl.store(zone + 3, tl.where(found, 3.0, 2.0))


@triton.jit
def _topp_gather_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    rd_ptr,
    zone_ptr,
    cand_ptr,
    cnt_ptr,
    part_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    CHUNK_CAP: tl.constexpr,
    SEG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Gathers the tokens of the zone (one program per ``(row, chunk)``; at most ``CHUNK_CAP`` per
    chunk, counted exactly); in mode 2 it also sums the mass above the zone and inside it.

    With ``SEG > 0`` a program reads ``CHUNK / SEG`` segments of ``SEG`` contiguous tokens spaced
    ``num_chunks * SEG`` apart instead of one contiguous chunk: the probable tokens of a language
    model cluster at the lowest ids (frequent tokens come first in a BPE vocabulary), and a contiguous
    chunk then overflows ``CHUNK_CAP`` for a nucleus of a few hundred tokens, which sends the row to
    the slow single-program kernel; interleaved, every program holds about the same share."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    if tl.load(rd_ptr + row * 16 + 4) > 1.5:
        zone = zone_ptr + row * 4
        mode = tl.load(zone + 2)
        if mode > 0.0:
            z_lo = tl.load(zone)
            z_hi = tl.load(zone + 1)
            inv_t, is_pad, big_m, _s, _mode, _lo, _depth, _target, _ties = _rd_load(rd_ptr, row)
            if SEG > 0:
                i = tl.arange(0, CHUNK)
                offs = tl.max_contiguous((ch + (i // SEG) * num_chunks) * SEG + (i % SEG), SEG)
            else:
                offs = ch * CHUNK + tl.arange(0, CHUNK)
            mask = offs < vocab
            z = _load_scaled(
                logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad,
                HAS_BIAS,
            )  # fmt: skip
            mem = mask & (z >= z_lo) & (z <= z_hi)
            mem_i = mem.to(tl.int32)
            n = tl.sum(mem_i, axis=0)
            tl.store(cnt_ptr + row * num_chunks + ch, n)
            if mode > 1.5:
                pbase = part_ptr + (row * num_chunks + ch) * 4
                w = tl.exp(z - big_m)
                tl.store(pbase, tl.sum(tl.where(mask & (z > z_hi), w, 0.0), axis=0))
                tl.store(pbase + 1, tl.sum(tl.where(mem, w, 0.0), axis=0))
                tl.store(pbase + 2, tl.min(tl.where(mem, z, float("inf")), axis=0))
                tl.store(pbase + 3, tl.max(tl.where(mem, z, float("-inf")), axis=0))
            if n > 0:
                pos = tl.cumsum(mem_i, axis=0) - 1
                tl.store(
                    cand_ptr + (row * num_chunks + ch) * CHUNK_CAP + pos, z,
                    mask=mem & (pos < CHUNK_CAP),
                )  # fmt: skip


@triton.jit
def _topp_resolve_kernel(
    rd_ptr,
    num_chunks,
    zone_ptr,
    cand_ptr,
    cnt_ptr,
    part_ptr,
    dense_ptr,
    sub_ptr,
    rec_ptr,
    CHUNK_CAP: tl.constexpr,
    CAP: tl.constexpr,
    CAP_MID: tl.constexpr,
    CAP_SMALL: tl.constexpr,
    BIN_MIN: tl.constexpr,
    NS: tl.constexpr,
    SLAB: tl.constexpr,
    NC_POW2: tl.constexpr,
):
    """The nucleus record of one row (one program per row) from what the earlier kernels left:

    * rows that keep the whole vocabulary: ``z_pivot = -inf``;
    * rows whose maximum alone reaches the target: the pivot is the maximum and the nucleus the
      tokens tied with it (their number comes from the stats kernel), no pass over the row;
    * otherwise the gathered candidates are sorted and the pivot is the first one, in descending
      order, at which the running mass (started at the mass above the zone) reaches the target; the
      kept mass includes every token tied with it. That is exact when the list holds every token of
      the zone (no chunk overflowed, at most ``CAP`` in all) and the zone brackets the target (mass
      above it below, mass through it at least the target; mode 1: the list's mass reaches it);
      otherwise the record stays pending for the single-program kernel.
    """
    row = tl.program_id(0).to(tl.int64)
    inv_t, is_pad, big_m, big_s, row_mode, _lo, _depth, target, ties = _rd_load(rd_ptr, row)
    z_pivot = big_m * 0.0 - float("inf")
    z_kept = big_s
    status = big_m * 0.0
    code = big_m * 0.0
    cidx = tl.arange(0, NC_POW2)
    cmask = cidx < num_chunks
    zone = zone_ptr + row * 4
    if (row_mode > 0.5) & (row_mode < 1.5):
        # The maximum alone reaches the target: the nucleus is the tokens tied with it.
        z_pivot = big_m
        z_kept = ties
        code = big_m * 0.0 + 9.0
    elif row_mode > 1.5:
        status = big_m * 0.0 + 3.0
        mode = tl.load(zone + 2)
        code = tl.load(zone + 3)
        if mode > 0.0:
            c = tl.load(cnt_ptr + row * num_chunks + cidx, mask=cmask, other=0)
            n_tot = tl.sum(c, axis=0)
            c_max = tl.max(c, axis=0)
            fits = (c_max <= CHUNK_CAP) & (n_tot <= CAP) & (n_tot > 0)
            a_above = big_m * 0.0
            m_zone = big_m * 0.0
            z_min = big_m * 0.0 + float("inf")
            z_max = big_m * 0.0 - float("inf")
            valid = n_tot > 0
            if mode > 1.5:
                pb = part_ptr + (row * num_chunks + cidx) * 4
                a_above = tl.sum(tl.load(pb, mask=cmask, other=0.0), axis=0)
                m_zone = tl.sum(tl.load(pb + 1, mask=cmask, other=0.0), axis=0)
                z_min = tl.min(tl.load(pb + 2, mask=cmask, other=float("inf")), axis=0)
                z_max = tl.max(tl.load(pb + 3, mask=cmask, other=float("-inf")), axis=0)
                valid = _zone_certified(a_above, m_zone, n_tot, target)
            code = tl.where(c_max > CHUNK_CAP, 4.0, 5.0)
            code = tl.where(valid, code, 6.0)
            if valid & (z_max <= z_min) & (mode > 1.5):
                # One tied group: it holds the pivot and every token of the zone is kept.
                z_pivot = z_min
                z_kept = a_above + m_zone
                status = big_m * 0.0
                code = big_m * 0.0 + 7.0
            elif valid & fits:
                # Dense list: chunk c's candidates go after those of the chunks before it.
                excl = tl.cumsum(c, axis=0) - c
                kk = tl.arange(0, CHUNK_CAP)
                keep = (kk[None, :] < c[:, None]) & cmask[:, None]
                vals = tl.load(
                    cand_ptr + (row * num_chunks + cidx[:, None]) * CHUNK_CAP + kk[None, :],
                    mask=keep, other=float("-inf"),
                )  # fmt: skip
                tl.store(dense_ptr + row * CAP + excl[:, None] + kk[None, :], vals, mask=keep)
                tl.debug_barrier()
                if n_tot <= CAP_SMALL:
                    zp, zk, ok = _sort_resolve(
                        dense_ptr + row * CAP, n_tot, a_above, target, big_m, CAP_SMALL
                    )
                elif n_tot <= BIN_MIN:
                    if n_tot <= CAP_MID:
                        zp, zk, ok = _sort_resolve(
                            dense_ptr + row * CAP, n_tot, a_above, target, big_m, CAP_MID
                        )
                    else:
                        zp, zk, ok = _sort_resolve(
                            dense_ptr + row * CAP, n_tot, a_above, target, big_m, CAP
                        )
                else:
                    zp, zk, ok = _bin_select_resolve(
                        dense_ptr + row * CAP, sub_ptr + row * CAP, n_tot, a_above, target, big_m,
                        CAP, CAP_SMALL, CAP_MID, NS, SLAB,
                    )  # fmt: skip
                code = tl.where(ok != 0, tl.where(mode > 1.5, 8.0, 1.0), 6.0)
                if ok != 0:
                    z_pivot = zp
                    z_kept = zk
                    status = big_m * 0.0
    tl.store(zone + 3, code)
    base = rec_ptr + row * 8
    tl.store(base, z_pivot)
    tl.store(base + 1, z_kept)
    tl.store(base + 2, big_s)
    tl.store(base + 3, status)
    tl.store(base + 4, big_m)
    tl.store(base + 5, inv_t)
    tl.store(base + 6, is_pad.to(tl.float32))


@triton.jit
def _topp_nucleus_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    rd_ptr,
    zz_ptr,
    rec_ptr,
    vocab,
    TILE: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    CAP_SMALL: tl.constexpr,
    ITERS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Exact single-program search for the rows the chunk-parallel pipeline did not resolve (and, with
    ``fast=False``, for every row): refines the histogram zone pass by pass, in one program per row.
    Rows with status 0 in ``rec`` keep their record."""
    row = tl.program_id(0).to(tl.int64)
    inv_t, is_pad, big_m, big_s, row_mode, lo, _depth, target, _ties = _rd_load(rd_ptr, row)

    # Rows the chunk-parallel pipeline resolved keep their record (status 0).
    need = tl.load(rec_ptr + row * 8 + 3) != 0.0
    # Defaults: keep the whole vocabulary (padding rows, T = 0, top_p >= 1, or no certified zone).
    z_pivot = big_m * 0.0 - float("inf")
    z_kept = big_s
    status = big_m * 0.0
    if need & (row_mode > 0.5):
        hi = big_m
        a0 = big_m * 0.0  # mass of the tokens above [lo, hi]
        n_above = 0  # their number
        m_win = (
            big_m * 0.0 - 1.0
        )  # exact mass of the tokens inside [lo, hi] once a pass has summed it
        cnt = tl.zeros([NB], tl.int32)
        have_cnt = 0
        found = 0
        failed = 0
        z_pivot = big_m * 0.0 - float("inf")
        z_kept = big_s
        # Fallback nucleus if the search ends without a pivot: every token at or above the lowest z of
        # the last certified zone (a superset of the exact nucleus), the whole vocabulary before any.
        z_fb = big_m * 0.0 - float("inf")
        kept_fb = big_s
        zrow = row * CAP
        idx = tl.arange(0, NB)
        for _it in range(ITERS):
            if (found == 0) & (failed == 0):
                if have_cnt == 0:
                    cnt = _topp_hist_pass(
                        logits_ptr, row, stride_row, bias_ptr, stride_bias, inv_t, is_pad, vocab,
                        lo, hi, TILE, NB, HAS_BIAS,
                    )  # fmt: skip
                    have_cnt = 1
                # Zone selection from the bin counts: bounds on the cumulative mass from the top.
                dlt = (hi - lo) / NB
                e_lo = lo + idx.to(tl.float32) * dlt
                e_hi = lo + (idx + 1).to(tl.float32) * dlt
                w_max = tl.exp(tl.minimum(e_hi, big_m) - big_m)
                w_min = tl.exp(e_lo - big_m)
                cf = cnt.to(tl.float32)
                f_min = cf * w_min
                f_max = cf * w_max
                # Mass from the top through bin j: bounded from above by the bins' mass (top-down) and
                # by S minus the bins below (bottom-up); the bottom-up bounds are the tight ones when
                # the target is near S. Tokens below `lo` weigh at most exp(lo - max) each.
                n_below = vocab - n_above - tl.sum(cnt, axis=0)
                below_max = n_below.to(tl.float32) * tl.exp(lo - big_m)
                # Once the window's mass is known exactly so is the mass below it: S - a0 - m_win (and
                # the three sums' rounding). The count bound above charges every token below the
                # window at its edge weight, which is hopelessly loose for a window cut into the
                # tokens of a huge tie group (their weight is ~1, the tokens below weigh ~0).
                below_exact = tl.maximum(big_s - a0 - m_win, 0.0) + _MASS_TOL * big_s
                below_max = tl.where(m_win >= 0.0, tl.minimum(below_max, below_exact), below_max)
                bot_min = tl.cumsum(f_min, axis=0) - f_min
                bot_max = tl.cumsum(f_max, axis=0) - f_max
                ub = tl.minimum(a0 + tl.cumsum(f_max, axis=0, reverse=True), big_s - bot_min)
                lb = tl.maximum(
                    a0 + tl.cumsum(f_min, axis=0, reverse=True), big_s - bot_max - below_max
                )
                j_up = tl.max(tl.where(ub >= target, idx, -1), axis=0)
                j_dn = tl.max(tl.where(lb >= target, idx, -1), axis=0)
                # The pivot is not below the lowest populated bin either.
                j_first = tl.min(tl.where(cnt > 0, idx, NB), axis=0)
                jlo = tl.maximum(tl.maximum(j_dn - 1, 0), j_first)
                jhi = tl.minimum(j_up + 1, NB - 1)
                z_lo = lo + jlo.to(tl.float32) * dlt
                z_hi = tl.where(jhi == NB - 1, hi, lo + (jhi + 1).to(tl.float32) * dlt)
                n_est = tl.sum(tl.where((idx >= jlo) & (idx <= jhi), cnt, 0), axis=0)
                if j_up < 0:
                    failed = 1
                else:
                    # Exact pass: mass above the zone, zone mass / extent, and either the zone's
                    # tokens (compaction) or, for a big zone, its histogram (refinement).
                    compact = n_est <= CAP - 16
                    a_above = big_m * 0.0
                    c_above = 0
                    m_zone = big_m * 0.0
                    n_mem = 0
                    z_min = big_m * 0.0 + float("inf")
                    z_max = big_m * 0.0 - float("inf")
                    cnt_new = tl.zeros([NB], tl.int32)
                    scale2 = NB / tl.maximum(z_hi - z_lo, 1e-30)
                    for t in range(0, vocab, TILE):
                        offs = t + tl.arange(0, TILE)
                        mask = offs < vocab
                        z = _load_scaled(
                            logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask,
                            inv_t, is_pad, HAS_BIAS,
                        )  # fmt: skip
                        above = mask & (z > z_hi)
                        mem = mask & (z >= z_lo) & (z <= z_hi)
                        w = tl.exp(z - big_m)
                        a_above += tl.sum(tl.where(above, w, 0.0), axis=0)
                        c_above += tl.sum(above.to(tl.int32), axis=0)
                        m_zone += tl.sum(tl.where(mem, w, 0.0), axis=0)
                        mem_i = mem.to(tl.int32)
                        if compact:
                            pos = n_mem + tl.cumsum(mem_i, axis=0) - 1
                            keep = mem & (pos < CAP)
                            tl.store(zz_ptr + zrow + pos, z, mask=keep)
                        else:
                            j2 = tl.minimum(tl.maximum((z - z_lo) * scale2, 0.0), NB - 1.0)
                            cnt_new += tl.histogram(j2.to(tl.int32), NB, mask=mem)
                        n_mem += tl.sum(mem_i, axis=0)
                        z_min = tl.minimum(z_min, tl.min(tl.where(mem, z, float("inf")), axis=0))
                        z_max = tl.maximum(z_max, tl.max(tl.where(mem, z, float("-inf")), axis=0))
                    tl.debug_barrier()
                    valid = _zone_certified(a_above, m_zone, n_mem, target)
                    if valid:
                        if z_max <= z_min:
                            # One tied group: the pivot, and every token of the zone, is kept.
                            z_pivot = z_min
                            z_kept = a_above + m_zone
                            found = 1
                        elif compact & (n_mem <= CAP):
                            # Sort the zone's tokens by descending z: the pivot is the first one at
                            # which the running mass (from a_above) reaches the target; every token
                            # at or above its z (ties included) is kept.
                            if n_mem <= CAP_SMALL:
                                z_pivot, z_kept, ok_sort = _sort_resolve(
                                    zz_ptr + zrow, n_mem, a_above, target, big_m, CAP_SMALL
                                )
                            else:
                                z_pivot, z_kept, ok_sort = _sort_resolve(
                                    zz_ptr + zrow, n_mem, a_above, target, big_m, CAP
                                )
                            found = ok_sort
                            failed = 1 - ok_sort
                        else:
                            # Zone too big: refine inside it. The zone is certified, so every token at
                            # or above its lowest z holds the nucleus: the fallback if refining stalls.
                            z_fb = z_min
                            kept_fb = a_above + m_zone
                            m_win = m_zone
                            a0 = a_above
                            n_above = c_above
                            if compact:
                                lo = z_min
                                hi = z_max
                                have_cnt = 0
                            else:
                                lo = z_lo
                                hi = z_hi
                                cnt = cnt_new
                                have_cnt = 1
                    else:
                        failed = 1
        if found == 0:
            z_pivot = z_fb
            z_kept = kept_fb
            status = tl.where(failed == 1, 1.0, 2.0)
    if need:
        base = rec_ptr + row * 8
        tl.store(base, z_pivot)
        tl.store(base + 1, z_kept)
        tl.store(base + 2, big_s)
        tl.store(base + 3, status)
        tl.store(base + 4, big_m)
        tl.store(base + 5, inv_t)
        tl.store(base + 6, is_pad.to(tl.float32))


@triton.jit
def _topp_draft_write_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    slot_ptr,
    rows_per_req,
    step0,
    rec_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    csum_ptr,
    clast_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """``q~ = [z >= z_pivot] * exp(z - max) / Z`` of one (row, chunk), written into the slot row."""
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    req = row // rows_per_req
    step = step0 + row % rows_per_req
    rec = rec_ptr + row * 8
    z_pivot = tl.load(rec)
    inv_z = tl.math.div_rn(1.0, tl.load(rec + 1))
    big_m = tl.load(rec + 4)
    inv_t = tl.load(rec + 5)
    is_pad = tl.load(rec + 6) > 0.5
    slot = tl.load(slot_ptr + req)
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    z = _load_scaled(
        logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad, HAS_BIAS
    )  # fmt: skip
    q = tl.where(mask & (z >= z_pivot), tl.exp(z - big_m) * inv_z, 0.0)
    tl.store(probs_ptr + slot * stride_slot + step * stride_step + offs, q, mask=mask)
    tl.store(csum_ptr + row * num_chunks + ch, tl.sum(q, 0))
    last = tl.max(tl.where(q > 0.0, offs, -1), 0)
    tl.store(clast_ptr + row * num_chunks + ch, last)


@triton.jit
def _topp_target_prob(z, z_pivot, big_m, inv_z):
    """The target nucleus probability of a token with scaled logit ``z``."""
    return tl.where(z >= z_pivot, tl.exp(z - big_m) * inv_z, 0.0)


@triton.jit
def _topp_accept_pos(
    tlogits_ptr,
    stride_row,
    rec_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    req,
    slot,
    seed,
    offset,
    vocab,
    NUM_DRAFT: tl.constexpr,
):
    """flashinfer chain_speculative_sampling accept loop on the nucleus distributions: position
    ``i`` is accepted when ``u_i * q~(d_i) < p~(d_i)``; returns the first rejected position."""
    pos = NUM_DRAFT
    for i in tl.static_range(NUM_DRAFT):
        row = req * (NUM_DRAFT + 1) + i
        rec = rec_ptr + row * 8
        z_pivot = tl.load(rec)
        inv_z = tl.math.div_rn(1.0, tl.load(rec + 1))
        big_m = tl.load(rec + 4)
        inv_t = tl.load(rec + 5)
        is_pad = tl.load(rec + 6) > 0.5
        d = tl.load(draft_ptr + req * NUM_DRAFT + i).to(tl.int64)
        d = tl.minimum(tl.maximum(d, 0), vocab - 1)
        x = tl.load(tlogits_ptr + row * stride_row + d)
        z = tl.where(is_pad, 0.0, x * inv_t)
        p_t = _topp_target_prob(z, z_pivot, big_m, inv_z)
        q_d = tl.load(probs_ptr + slot * stride_slot + i * stride_step + d)
        u = _curand_uniform(seed, offset, req, i)
        reject = (pos == NUM_DRAFT) & (~(u * q_d < p_t))
        pos = tl.where(reject, i, pos)
    return pos


@triton.jit
def _topp_verify_resid_kernel(
    tlogits_ptr,
    stride_row,
    slot_ptr,
    rec_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    seed_ptr,
    offset_ptr,
    pos_ptr,
    rsum_ptr,
    rlast_ptr,
    vocab,
    num_chunks,
    NUM_DRAFT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    seed = tl.load(seed_ptr + req)
    offset = tl.load(offset_ptr + req)
    slot = tl.load(slot_ptr + req)
    pos = _topp_accept_pos(
        tlogits_ptr,
        stride_row,
        rec_ptr,
        probs_ptr,
        stride_slot,
        stride_step,
        draft_ptr,
        req,
        slot,
        seed,
        offset,
        vocab,
        NUM_DRAFT,
    )
    row = req * (NUM_DRAFT + 1) + pos
    rec = rec_ptr + row * 8
    z_pivot = tl.load(rec)
    inv_z = tl.math.div_rn(1.0, tl.load(rec + 1))
    big_m = tl.load(rec + 4)
    inv_t = tl.load(rec + 5)
    is_pad = tl.load(rec + 6) > 0.5
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    z = tl.where(is_pad, 0.0, x * inv_t)
    p_t = _topp_target_prob(z, z_pivot, big_m, inv_z)
    has_draft = pos < NUM_DRAFT
    q_d = tl.load(
        probs_ptr + slot * stride_slot + tl.minimum(pos, NUM_DRAFT - 1) * stride_step + offs,
        mask=mask & has_draft,
        other=0.0,
    )
    res = tl.where(mask, tl.maximum(p_t - q_d, 0.0), 0.0)
    tl.store(rsum_ptr + req * num_chunks + ch, tl.sum(res, 0))
    tl.store(rlast_ptr + req * num_chunks + ch, tl.max(tl.where(res > 0.0, offs, -1), 0))
    if ch == 0:
        tl.store(pos_ptr + req, pos)


@triton.jit
def _topp_verify_pick_kernel(
    tlogits_ptr,
    stride_row,
    slot_ptr,
    rec_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    seed_ptr,
    offset_ptr,
    pos_ptr,
    rsum_ptr,
    rlast_ptr,
    out_tokens_ptr,
    out_num_ptr,
    vocab,
    num_chunks,
    NUM_DRAFT: tl.constexpr,
    CHUNK: tl.constexpr,
    KP1_POW2: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    seed = tl.load(seed_ptr + req)
    offset = tl.load(offset_ptr + req)
    slot = tl.load(slot_ptr + req)
    pos = tl.load(pos_ptr + req)
    has_draft = pos < NUM_DRAFT
    # Draw K, after the accept draws 0..K-1, for the residual and the bonus token alike.
    u = _curand_uniform(seed, offset, req, NUM_DRAFT)
    total = 0.0
    for k in range(num_chunks):
        total = total + tl.load(rsum_ptr + req * num_chunks + k)
    u = u * total
    kstar, base, last = _pick_chunk(rsum_ptr, rlast_ptr, req, num_chunks, u)
    row = req * (NUM_DRAFT + 1) + pos
    rec = rec_ptr + row * 8
    z_pivot = tl.load(rec)
    inv_z = tl.math.div_rn(1.0, tl.load(rec + 1))
    big_m = tl.load(rec + 4)
    inv_t = tl.load(rec + 5)
    is_pad = tl.load(rec + 6) > 0.5
    kk = tl.maximum(kstar, 0)
    offs = kk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    z = tl.where(is_pad, 0.0, x * inv_t)
    p_t = _topp_target_prob(z, z_pivot, big_m, inv_z)
    q_d = tl.load(
        probs_ptr + slot * stride_slot + tl.minimum(pos, NUM_DRAFT - 1) * stride_step + offs,
        mask=mask & has_draft,
        other=0.0,
    )
    res = tl.where(mask, tl.maximum(p_t - q_d, 0.0), 0.0)
    cdf = base + tl.cumsum(res, 0)
    hit = (res > 0.0) & (cdf > u)
    first = tl.min(tl.where(hit, offs, 2147483647), 0)
    chunk_last = tl.max(tl.where(res > 0.0, offs, -1), 0)
    token = tl.where(first < 2147483647, first, chunk_last)
    token = tl.where(kstar >= 0, token, last)
    token = tl.where(token >= 0, token, 0)
    cols = tl.arange(0, KP1_POW2)
    drafts = tl.load(draft_ptr + req * NUM_DRAFT + cols, mask=cols < NUM_DRAFT, other=0)
    out = tl.where(cols < pos, drafts, tl.where(cols == pos, token, -1))
    tl.store(
        out_tokens_ptr + req * (NUM_DRAFT + 1) + cols, out.to(tl.int32), mask=cols < NUM_DRAFT + 1
    )
    tl.store(out_num_ptr + req, (pos + 1).to(tl.int32))


# Control words per row, zeroed before every call: the stats ticket, the histogram ticket, then the
# histogram bins.
_CTL_TICKETS = 2


def _topp_row_stats(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    temps: torch.Tensor,
    temps_div: int,
    top_ps: torch.Tensor,
    topp_div: int,
    slot_ids: torch.Tensor,
    dummy_row: int,
    rows_per_req: int,
    tag: str,
    cfg: NucleusConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pmax, psum, pties, rd, ctl)``: the per-chunk partials, the row descriptors (:data:`_RD_WIDTH`;
    the last program of each row combines the partials) and the zeroed control words of the nucleus
    pipeline."""
    rows, vocab = logits.shape
    nc = triton.cdiv(vocab, cfg.stats_chunk)
    dev = logits.device
    pmax = _WORKSPACE.get(tag + ".pmax", (rows, nc), torch.float32, dev)
    psum = _WORKSPACE.get(tag + ".psum", (rows, nc), torch.float32, dev)
    pties = _WORKSPACE.get(tag + ".pties", (rows, nc), torch.int32, dev)
    rd = _WORKSPACE.get(tag + ".rd", (rows, _RD_WIDTH), torch.float32, dev)
    ctl_width = _CTL_TICKETS + cfg.hist_bins
    ctl = _WORKSPACE.get(tag + ".ctl", (rows, ctl_width), torch.int32, dev)
    if rows > 0:
        ctl.zero_()
        _topp_stats_kernel[(rows, nc)](
            logits,
            logits.stride(0),
            bias if bias is not None else logits,
            bias.stride(0) if bias is not None else 0,
            temps,
            temps_div,
            top_ps,
            topp_div,
            slot_ids,
            dummy_row,
            rows_per_req,
            pmax,
            psum,
            pties,
            ctl,
            ctl_width,
            rd,
            vocab,
            nc,
            CHUNK=cfg.stats_chunk,
            NC_POW2=triton.next_power_of_2(nc),
            HAS_BIAS=bias is not None,
            num_warps=_NUM_WARPS,
        )
    return pmax, psum, pties, rd, ctl


def _nucleus_records(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    temps: torch.Tensor,
    temps_div: int,
    top_ps: torch.Tensor,
    topp_div: int,
    slot_ids: torch.Tensor,
    dummy_row: int,
    rows_per_req: int,
    tag: str,
    cfg: NucleusConfig,
) -> torch.Tensor:
    """The nucleus of every row of ``logits``: ``rec [rows, 8]`` fp32 = ``(z_pivot, Z, S, status,
    max z, 1/T, padding, -)`` (workspace-backed; valid until the next call with this ``tag``).

    The chunk-parallel pipeline (stats, histogram, gather, resolve) handles every row whose candidate
    list fits; the single-program kernel runs over the rest (status pending) and is the exact
    fallback for every case."""
    rows, vocab = logits.shape
    dev = logits.device
    _pmax, _psum, _pties, rd, ctl = _topp_row_stats(
        logits,
        bias,
        temps,
        temps_div,
        top_ps,
        topp_div,
        slot_ids,
        dummy_row,
        rows_per_req,
        tag,
        cfg,
    )
    rec = _WORKSPACE.get(tag + ".rec", (rows, _REC_WIDTH), torch.float32, dev)
    zone = _WORKSPACE.get(tag + ".zone", (rows, 4), torch.float32, dev)
    zz = _WORKSPACE.get(tag + ".zz", (rows, cfg.cap), torch.float32, dev)
    if rows == 0:
        return rec
    has_bias = bias is not None
    bias_arg = bias if has_bias else logits
    bias_stride = bias.stride(0) if has_bias else 0
    if cfg.fast:
        ncg = triton.cdiv(vocab, cfg.hg_chunk)
        ncg_pow2 = triton.next_power_of_2(ncg)
        cand = _WORKSPACE.get(tag + ".cand", (rows, ncg, cfg.chunk_cap), torch.float32, dev)
        cnt = _WORKSPACE.get(tag + ".cnt", (rows, ncg), torch.int32, dev)
        part = _WORKSPACE.get(tag + ".part", (rows, ncg, 4), torch.float32, dev)
        dense = _WORKSPACE.get(tag + ".dense", (rows, cfg.cap), torch.float32, dev)
        _topp_hist_kernel[(rows, ncg)](
            logits, logits.stride(0), bias_arg, bias_stride, rd, ctl, ctl.stride(0), zone, vocab, ncg,
            NB=cfg.hist_bins, CHUNK=cfg.hg_chunk, CAP=cfg.cap, SMALL_ZONE=cfg.zone_small,
            HAS_BIAS=has_bias, num_warps=cfg.pipe_warps, **_reg_cap(cfg.hist_maxnreg),
        )  # fmt: skip
        _topp_gather_kernel[(rows, ncg)](
            logits, logits.stride(0), bias_arg, bias_stride, rd, zone, cand, cnt, part, vocab, ncg,
            CHUNK=cfg.hg_chunk, CHUNK_CAP=cfg.chunk_cap, SEG=cfg.gather_seg, HAS_BIAS=has_bias,
            num_warps=cfg.pipe_warps, **_reg_cap(cfg.gather_maxnreg),
        )  # fmt: skip
        _topp_resolve_kernel[(rows,)](
            rd, ncg, zone, cand, cnt, part, dense, zz, rec,
            CHUNK_CAP=cfg.chunk_cap, CAP=cfg.cap, CAP_MID=cfg.mid, CAP_SMALL=cfg.small,
            BIN_MIN=cfg.bin_min, NS=cfg.sub_bins, SLAB=cfg.sub_slab,
            NC_POW2=ncg_pow2, num_warps=cfg.resolve_warps,
        )  # fmt: skip
    else:
        rec[:, 3].fill_(float(STATUS_PENDING))
    _topp_nucleus_kernel[(rows,)](
        logits, logits.stride(0), bias_arg, bias_stride, rd, zz, rec, vocab,
        TILE=cfg.tile, NB=cfg.bins, CAP=cfg.cap, CAP_SMALL=cfg.small, ITERS=cfg.iters,
        HAS_BIAS=has_bias, num_warps=cfg.warps,
    )  # fmt: skip
    return rec


def _reg_cap(maxnreg: int) -> dict:
    """Launch option capping a kernel's registers per thread (0: the compiler's choice)."""
    return {"maxnreg": maxnreg} if maxnreg > 0 else {}


def fast_topp_nucleus(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    cfg: Optional[NucleusConfig] = None,
    return_path: bool = False,
):
    """Nucleus parameters of independent rows (testing / inspection entry point).

    Args:
        logits: ``[rows, vocab]`` fp32.
        temperatures: ``[rows]`` fp32.
        top_ps: ``[rows]`` fp32.
    Returns:
        ``[rows, 4]`` fp32 ``(z_pivot, Z, S, status)`` per row (a copy): keep ``z >= z_pivot`` with
        ``z = logits / T`` (as ``_load_scaled``); ``Z`` is the kept mass in units of ``exp(z -
        max z)``, ``S`` the total; ``status`` is 0 unless the search ended without a pivot (non-finite input or an
        adversarial tie structure): the row then keeps a superset of the nucleus.
        With ``return_path`` also ``[rows]`` fp32 path codes: 1 the chunk-parallel gather resolved the
        row, 8 the zone sums did, 7 a tied zone did; 2 no zone; 3 zone too big; 4 a chunk
        overflowed; 5 candidates not in 1..cap; 6 their mass is short of the target (codes 2-6: the
        single-program kernel resolved it; stale for rows that keep the whole vocabulary, and
        meaningless with ``cfg.fast=False``).
    """
    cfg = DEFAULT_NUCLEUS if cfg is None else cfg
    rows, _ = logits.shape
    slot_ids = torch.zeros(rows, dtype=torch.int64, device=logits.device)
    rec = _nucleus_records(
        logits, None, temperatures, 1, top_ps, 1, slot_ids, -1, 1, "nuc_test", cfg
    )
    rec = rec[:, :4]
    if not return_path:
        return rec.clone()
    zone = _WORKSPACE.get("nuc_test.zone", (rows, 4), torch.float32, logits.device)
    return rec.clone(), zone[:, 3].clone()


def _check_top_ps(top_ps: torch.Tensor, numel: int) -> None:
    assert top_ps.dtype == torch.float32 and top_ps.dim() == 1 and top_ps.stride(0) == 1
    assert top_ps.numel() >= numel


def _topp_draft_rows(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
    step0: int,
    steps_total: int,
    tokens: torch.Tensor,
    cfg: NucleusConfig,
) -> None:
    """Sample ``rows / num_gens`` draft positions (from ``step0``) per request from their nuclei,
    writing the proposals to ``draft_probs[slot, step]`` and the tokens to ``tokens[req, step]``."""
    rows, vocab = logits.shape
    num_gens = slot_ids.shape[0]
    rows_per_req = rows // num_gens
    _check_request_args(temperatures, slot_ids, seed, offset, num_gens, num_gens)
    _check_top_ps(top_ps, num_gens)
    nc, _nc_pow2 = _num_chunks(vocab)
    rec = _nucleus_records(
        logits, bias, temperatures, rows_per_req, top_ps, rows_per_req, slot_ids, dummy_row,
        rows_per_req, "topp.draft", cfg,
    )  # fmt: skip
    csum = _WORKSPACE.get("topp.draft.csum", (rows, nc), torch.float32, logits.device)
    clast = _WORKSPACE.get("topp.draft.clast", (rows, nc), torch.int32, logits.device)
    _topp_draft_write_kernel[(rows, nc)](
        logits,
        logits.stride(0),
        bias if bias is not None else logits,
        bias.stride(0) if bias is not None else 0,
        slot_ids,
        rows_per_req,
        step0,
        rec,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        csum,
        clast,
        vocab,
        nc,
        CHUNK=_CHUNK,
        HAS_BIAS=bias is not None,
        num_warps=_NUM_WARPS,
    )
    _rs_draft_pick_kernel[(rows,)](
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        slot_ids,
        rows_per_req,
        step0,
        steps_total,
        csum,
        clast,
        seed,
        offset,
        step_offset,
        tokens,
        vocab,
        nc,
        CHUNK=_CHUNK,
        num_warps=_NUM_WARPS,
    )


def fast_topp_rs_draft_block(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
    cfg: Optional[NucleusConfig] = None,
) -> torch.Tensor:
    """Sample one block of draft tokens from their top-p nuclei and store the proposal rows.

    Args:
        logits: ``[num_gens * K, vocab]`` fp32 draft logits, request-major (row ``b * K + i``).
        temperatures, top_ps: ``[num_gens]`` fp32 per-request temperature and top-p.
        slot_ids: ``[num_gens]`` int64 ``draft_probs`` slot of each request.
        dummy_row: slot of CUDA-graph padding requests (their logits are treated as zeros);
            ``-1`` when there is none.
        draft_probs: ``[slots, max_draft_len, vocab]`` fp32; rows ``[slot, 0:K, :]`` are written.
        seed, offset: ``[num_gens]`` int64 per-request Philox seed and offset (the draft stream
            is keyed ``seed[b] ^ DRAFT_KEY_XOR``).
        step_offset: added to ``offset[b]`` (1 for the block draft sampler).
    Returns:
        ``[num_gens * K]`` int32 tokens, distributed as the top-p nucleus of ``softmax(logits / T)``.
    """
    cfg = DEFAULT_NUCLEUS if cfg is None else cfg
    rows, _ = logits.shape
    num_gens = slot_ids.shape[0]
    tokens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return tokens
    _check_draft_args(logits, draft_probs)
    k = rows // num_gens
    assert k * num_gens == rows and draft_probs.shape[1] >= k
    _topp_draft_rows(
        logits, None, temperatures, top_ps, slot_ids, dummy_row, draft_probs, seed, offset,
        step_offset, 0, k, tokens, cfg,
    )  # fmt: skip
    return tokens


def fast_topp_rs_draft_markov_feedback(
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
    cfg: Optional[NucleusConfig] = None,
) -> torch.Tensor:
    """DSpark vanilla-Markov block chain that feeds back the SAMPLED token, under top-p.

    Position ``i`` proposes the top-p nucleus of ``softmax((base_i + markov_w2 @ markov_w1[x_{i-1}])
    / T)``, with ``x_{-1}`` the anchor and ``x_{i-1}`` the token drawn from the nucleus at position
    ``i - 1``: the chain sees the distribution the target samples from (a nucleus, not the full
    softmax). ``q~_i`` is stored for verification, so rejection sampling stays lossless.

    Args:
        base_logits: ``[num_gens, K, vocab]`` fp32 shared-lm_head block logits.
        anchor_tokens: ``[num_gens]`` last accepted token per request.
        markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights.
        top_ps: ``[num_gens]`` fp32 per-request top-p.
        Other arguments as :func:`fast_topp_rs_draft_block`.
    Returns:
        ``[num_gens, K]`` int32 tokens.
    """
    cfg = DEFAULT_NUCLEUS if cfg is None else cfg
    num_gens, k, vocab = base_logits.shape
    tokens = torch.empty((num_gens, k), dtype=torch.int32, device=base_logits.device)
    if num_gens == 0 or k == 0:
        return tokens
    _check_draft_args(base_logits, draft_probs)
    assert draft_probs.shape[1] >= k
    rank = markov_w1.shape[1]
    anchor = anchor_tokens.contiguous()
    emb = _WORKSPACE.get("topp.markov.emb", (num_gens, rank), markov_w1.dtype, base_logits.device)
    for step in range(k):
        _markov_feedback_emb_kernel[(num_gens,)](
            tokens,
            k,
            step,
            anchor,
            markov_w1,
            emb,
            vocab,
            RANK=rank,
            BLOCK_R=triton.next_power_of_2(rank),
            num_warps=4,
        )
        bias = torch.nn.functional.linear(emb, markov_w2)
        _topp_draft_rows(
            base_logits[:, step], bias, temperatures, top_ps, slot_ids, dummy_row, draft_probs,
            seed, offset, step_offset, step, k, tokens, cfg,
        )  # fmt: skip
    return tokens


def fast_topp_rs_verify(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    cfg: Optional[NucleusConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample the generation requests' drafts against the target's top-p nuclei.

    Args:
        logits: ``[num_gens * (K + 1), vocab]`` fp32 target logits, request-major.
        temperatures, top_ps: ``[num_gens * (K + 1)]`` fp32 per-row temperature and top-p.
        draft_tokens: ``[num_gens, K]`` int32 draft tokens (unit inner stride).
        slot_ids: ``[num_gens]`` int64 ``draft_probs`` slot of each request.
        dummy_row: slot of CUDA-graph padding requests (their logits are treated as zeros).
        draft_probs: ``[slots, >= K, vocab]`` fp32 nucleus proposals stored by the previous draft step.
        seed, offset: ``[num_gens]`` int64 per-request Philox seed and offset.
    Returns:
        ``accepted_tokens [num_gens, K + 1]`` int32 (accepted drafts, the drawn token, then
        ``-1``) and ``num_accepted [num_gens]`` int32 (accepted drafts + 1).
    """
    cfg = DEFAULT_NUCLEUS if cfg is None else cfg
    num_gens, k = draft_tokens.shape
    rows, vocab = logits.shape
    device = logits.device
    accepted = torch.empty((num_gens, k + 1), dtype=torch.int32, device=device)
    num_acc = torch.empty((num_gens,), dtype=torch.int32, device=device)
    if num_gens == 0:
        return accepted, num_acc
    assert rows == num_gens * (k + 1), (rows, num_gens, k)
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert draft_probs.dtype == torch.float32 and draft_probs.stride(2) == 1
    assert draft_probs.shape[2] == vocab and draft_probs.shape[1] >= k
    assert draft_tokens.stride(1) == 1
    _check_request_args(temperatures, slot_ids, seed, offset, rows, num_gens)
    _check_top_ps(top_ps, rows)
    nc, _nc_pow2 = _num_chunks(vocab)
    rec = _nucleus_records(
        logits, None, temperatures, 1, top_ps, 1, slot_ids, dummy_row, k + 1, "topp.target", cfg
    )
    pos = _WORKSPACE.get("topp.verify.pos", (num_gens,), torch.int64, device)
    rsum = _WORKSPACE.get("topp.verify.rsum", (num_gens, nc), torch.float32, device)
    rlast = _WORKSPACE.get("topp.verify.rlast", (num_gens, nc), torch.int32, device)
    _topp_verify_resid_kernel[(num_gens, nc)](
        logits,
        logits.stride(0),
        slot_ids,
        rec,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        seed,
        offset,
        pos,
        rsum,
        rlast,
        vocab,
        nc,
        NUM_DRAFT=k,
        CHUNK=_CHUNK,
        num_warps=_NUM_WARPS,
    )
    _topp_verify_pick_kernel[(num_gens,)](
        logits,
        logits.stride(0),
        slot_ids,
        rec,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        seed,
        offset,
        pos,
        rsum,
        rlast,
        accepted,
        num_acc,
        vocab,
        nc,
        NUM_DRAFT=k,
        CHUNK=_CHUNK,
        KP1_POW2=triton.next_power_of_2(k + 1),
        num_warps=_NUM_WARPS,
    )
    return accepted, num_acc
