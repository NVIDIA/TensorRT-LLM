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
"""BF16 near-tie tooling for the draft_tail decision-exactness study.

* Native decisions: the 8-shard native oracle (``draft_tail_oracle``) on BF16 logits
  computed by ``F.linear`` (nvjet) at a given GEMM row count M.
* Exact-rounding decisions: native semantics (largest BF16 logit, lowest id
  on ties) applied to logits computed in FP64 and rounded once to BF16.
* Adversarial near-ties: real hidden rows changed in a few BF16 elements so
  the FP64 logits of their two best tokens straddle one BF16 rounding
  midpoint at a chosen distance (in FP32 ulps at the midpoint). There the
  BF16 decision depends on the GEMM's FP32 accumulation order.
"""

import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft_tail_oracle as oracle  # noqa: E402

# ---------------------------------------------------------------------------
# BF16 grid helpers (finite values only)
# ---------------------------------------------------------------------------


def bf16_step(x: torch.Tensor, k) -> torch.Tensor:
    """BF16 values ``k`` grid steps above ``x`` (BF16), along the real line;
    -0 and +0 are one point. ``k`` is an int or a broadcastable int tensor."""
    bits = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    mag = bits & 0x7FFF
    ordv = torch.where((bits & 0x8000) != 0, -mag, mag) + k
    out = torch.where(ordv >= 0, ordv, 0x8000 | (-ordv))
    out = torch.where(out >= 0x8000, out - 0x10000, out)
    return out.to(torch.int16).view(torch.bfloat16)


def bf16_round_exact(x: torch.Tensor) -> torch.Tensor:
    """Round FP64 ``x`` to BF16 once (nearest, ties to even), avoiding the
    double rounding of FP64 -> FP32 -> BF16."""
    x = x.double()
    trunc = (
        (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
    )  # toward zero, from FP32
    # FP32 rounding of x may have crossed a BF16 grid point: step back.
    lo = torch.where(
        trunc.double().abs() > x.abs(), bf16_step(trunc, torch.where(x > 0, -1, 1)), trunc
    )
    hi = bf16_step(lo, torch.where(x >= 0, 1, -1))
    lo_d, hi_d = lo.double(), hi.double()
    mid = (lo_d + hi_d) / 2
    away = x.abs() > mid.abs()
    tie = x.abs() == mid.abs()
    lo_even = (lo.view(torch.int16) & 1) == 0
    pick_hi = away | (tie & ~lo_even)
    return torch.where(pick_hi, hi, lo)


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """BF16 ulp at |x| (FP64), for normal values."""
    e = torch.floor(torch.log2(x.abs().double().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)


def fp32_ulp(x: torch.Tensor) -> torch.Tensor:
    e = torch.floor(torch.log2(x.abs().double().clamp_min(1e-30)))
    return torch.pow(2.0, e - 23)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def native_bf16_logits(
    hidden: torch.Tensor, shards: Sequence[torch.Tensor], m: int
) -> List[torch.Tensor]:
    """Per-shard native BF16 logits of ``hidden`` [N, H] computed with GEMMs
    of ``m`` rows (the last chunk is padded with copies of its first row so
    every GEMM has exactly ``m`` rows)."""
    out = [[] for _ in shards]
    n = hidden.shape[0]
    for b in range(0, n, m):
        chunk = hidden[b : b + m]
        live = chunk.shape[0]
        if live < m:
            chunk = torch.cat([chunk, chunk[:1].expand(m - live, -1)])
        for r, w in enumerate(shards):
            out[r].append(F.linear(chunk, w)[:live])
    return [torch.cat(o) for o in out]


def native_tokens(shard_logits: Sequence[torch.Tensor], chunk: int = 512) -> torch.Tensor:
    """8-shard native oracle (``_get_local_max_and_combined`` + gather +
    ``_get_draft_tokens_from_gathered``) over many rows."""
    n = shard_logits[0].shape[0]
    toks = []
    for b in range(0, n, chunk):
        toks.append(oracle.native_tokens_from_logits([lg[b : b + chunk] for lg in shard_logits]))
    return torch.cat(toks)


def bf16_key(values_bf16: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """draft_tail's packed key (ordered BF16 << 32 | ~id) as int64 (no NaN here)."""
    bits = values_bf16.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    ordered = torch.where((bits & 0x8000) != 0, 0x7FFF - (bits & 0x7FFF), bits | 0x8000)
    # Same mapping as orderedBf16 for non-NaN values, up to a constant that
    # does not change the order (-0 == +0 below).
    ordered = torch.where((bits & 0x7FFF) == 0, torch.full_like(bits, 0x7FFF), ordered)
    return (ordered << 32) | (0xFFFFFFFF - ids.to(torch.int64))


def topk_fp64(
    hidden: torch.Tensor, head: torch.Tensor, shard_logits: Sequence[torch.Tensor], k: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ids of the ``k`` largest native BF16 logits per row and their FP64
    logits (sorted descending by FP64 value)."""
    full = torch.cat(shard_logits, dim=-1).float()
    ids = full.topk(k, dim=-1).indices
    w = head[ids].double()  # [N, k, H]
    vals = torch.einsum("nh,nkh->nk", hidden.double(), w)
    order = vals.argsort(dim=-1, descending=True)
    return ids.gather(1, order), vals.gather(1, order)


def exact_tokens(ids: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """Native semantics on once-rounded FP64 logits of the candidates."""
    rounded = bf16_round_exact(vals)
    keys = bf16_key(rounded, ids)
    return ids.gather(1, keys.argmax(dim=1, keepdim=True)).squeeze(1).to(torch.int32)


def near_tie_stats(ids: torch.Tensor, vals: torch.Tensor) -> Dict:
    """Per-row near-tie measures from the FP64 candidates.

    gap_bf16_ulps: (top1 - top2) / BF16 ulp at top1.
    edge_fp32_ulps: for the candidates within one BF16 ulp of the top, the
    smallest distance of any of them to a BF16 rounding midpoint, in FP32
    ulps; small values mean a different FP32 accumulation order can change
    that candidate's BF16 value.
    """
    top = vals[:, :1]
    ulp_b = bf16_ulp(top)
    gap = (vals[:, 0] - vals[:, 1]) / ulp_b[:, 0]
    rounded = bf16_round_exact(vals).double()
    # nearest midpoint: between rounded and its neighbour on x's side
    side = torch.where(vals >= rounded, 1, -1)
    neigh = bf16_step(bf16_round_exact(vals), side).double()
    mid = (rounded + neigh) / 2
    dist = (vals - mid).abs() / fp32_ulp(vals)
    close = (top - vals) <= ulp_b
    edge = torch.where(close, dist, torch.full_like(dist, float("inf")))
    return {
        "gap_bf16_ulps": gap,
        "edge_fp32_ulps": edge.min(dim=1).values,
    }


def draft_tail_full_tokens(
    runner, hidden: torch.Tensor, head: torch.Tensor, variant: int, chunk: int = 8
) -> torch.Tensor:
    """draft_tail fused-kernel decision over the whole vocabulary (one tp_size 1
    shard; equals the 8-shard global decision, see test_draft_tail_draft_tail)."""
    out = []
    for b in range(0, hidden.shape[0], chunk):
        h = hidden[b : b + chunk].contiguous()
        keys = torch.zeros(h.shape[0], dtype=torch.int64, device="cuda")
        runner.run_fused(h, head, 0, keys_out=keys, variant=variant)
        out.append(oracle.keys_to_tokens(keys))
    return torch.cat(out)


def draft_tail_logits_tokens(
    runner, shard_logits: Sequence[torch.Tensor], chunk: int = 8
) -> torch.Tensor:
    """draft_tail logits mode (native BF16 logits + draft_tail), 8 shards emulated."""
    n = shard_logits[0].shape[0]
    out = []
    for b in range(0, n, chunk):
        keys = []
        for r, lg in enumerate(shard_logits):
            k = torch.zeros(min(chunk, n - b), dtype=torch.int64, device="cuda")
            runner.run_from_logits(lg[b : b + chunk].contiguous(), r * lg.shape[1], keys_out=k)
            keys.append(k)
        out.append(oracle.keys_to_tokens(oracle.global_key(keys)))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Adversarial near-ties
# ---------------------------------------------------------------------------

_STEPS = (1, 2, 4, 8, 16, 32)


def construct_straddle(
    hidden: torch.Tensor,
    head: torch.Tensor,
    tok_hi: torch.Tensor,
    tok_lo: torch.Tensor,
    tau_fp32_ulps: torch.Tensor,
    iters: int = 3000,
) -> Dict[str, torch.Tensor]:
    """Move rows ``hidden`` [B, H] (BF16) by BF16 grid steps of single
    elements until the FP64 logit of ``tok_hi`` sits ``tau`` FP32 ulps above
    a BF16 rounding midpoint m and that of ``tok_lo`` ``tau`` below it (tau
    per row, times a random factor in [0.5, 1]); m is the midpoint nearest
    to the mean of the two starting logits. Greedy: each iteration applies,
    per row, the single (element, step) change that most reduces the
    distance to the targets.
    """
    b, hdim = hidden.shape
    h = hidden.clone()
    w_hi = head[tok_hi].double()
    w_lo = head[tok_lo].double()

    def logits(hh):
        hd = hh.double()
        return (hd * w_hi).sum(-1), (hd * w_lo).sum(-1)

    l_hi, l_lo = logits(h)
    mean = (l_hi + l_lo) / 2
    lo_grid = bf16_round_exact(mean)
    lo_grid = torch.where(lo_grid.double() > mean, bf16_step(lo_grid, -1), lo_grid)
    mid = (lo_grid.double() + bf16_step(lo_grid, 1).double()) / 2
    gen = torch.Generator(device=hidden.device).manual_seed(7)
    jitter = 0.5 + 0.5 * torch.rand(2, b, generator=gen, device=hidden.device, dtype=torch.float64)
    ulp = fp32_ulp(mid)
    t_hi = mid + tau_fp32_ulps.double() * ulp * jitter[0]
    t_lo = mid - tau_fp32_ulps.double() * ulp * jitter[1]
    rows = torch.arange(b, device=hidden.device)
    for _ in range(iters):
        r_hi, r_lo = t_hi - l_hi, t_lo - l_lo
        cur = torch.maximum(r_hi.abs(), r_lo.abs())
        done = cur <= 0.05 * tau_fp32_ulps.double() * ulp
        if bool(done.all()):
            break
        best_cost = cur.clone()
        best_idx = torch.full((b,), -1, dtype=torch.int64, device=hidden.device)
        best_step = torch.zeros(b, dtype=torch.int64, device=hidden.device)
        hd = h.double()
        for s in _STEPS:
            for sign in (1, -1):
                moved = bf16_step(h, sign * s).double()
                delta = moved - hd
                c = torch.maximum(
                    (r_hi[:, None] - delta * w_hi).abs(), (r_lo[:, None] - delta * w_lo).abs()
                )
                c = torch.where(torch.isfinite(moved), c, torch.full_like(c, float("inf")))
                val, idx = c.min(dim=1)
                better = val < best_cost
                best_cost = torch.where(better, val, best_cost)
                best_idx = torch.where(better, idx, best_idx)
                best_step = torch.where(better, torch.full_like(best_step, sign * s), best_step)
        move = (best_idx >= 0) & ~done
        if not bool(move.any()):
            break
        sel = rows[move]
        col = best_idx[move]
        h[sel, col] = bf16_step(h[sel, col], best_step[move])
        l_hi, l_lo = logits(h)
    return {
        "hidden": h,
        "mid": mid,
        "l_hi": l_hi,
        "l_lo": l_lo,
        "dist_hi_fp32_ulps": (l_hi - mid) / ulp,
        "dist_lo_fp32_ulps": (l_lo - mid) / ulp,
        "changed_elements": (h != hidden).sum(-1),
    }


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + r) / den
