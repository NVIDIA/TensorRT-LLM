# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared routed-expert phase planners, with target geometry supplied explicitly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MoePlanGeometry:
    top_k: int
    hidden: int
    u1: int
    stages: int
    row2_bytes: int
    item_bytes: int


def make_phase_plan(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    fine: bool = True,
    heavy_fc2_only: bool = False,
    *,
    geometry: MoePlanGeometry,
    fine_ranges_fn: Callable[[list[float], int], list[int]],
) -> tuple[torch.Tensor, int]:
    """The work split of ``m3_moe`` for each distinct-expert count G (4 .. 4 m_max): FC2 32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..) a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work, or (``fine``) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit (:func:`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and the largest block count."""
    gcap = geometry.top_k * m_max
    nb2 = geometry.hidden // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    min_nb = min(nbs)
    no_fc1 = [heavy_fc2_only and nbs[c] > min_nb for c in range(num_ctas)]
    BIG_W = 1 << 40
    blk_bytes = 32 * geometry.row2_bytes
    for g in range(geometry.top_k, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = b0s[c] | nbs[c] << 16
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * geometry.u1 * geometry.stages
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and (not (early0 and p == 0 and (m_max >= 2))):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            for c in range(num_ctas):
                if no_fc1[c]:
                    fc2[c] += BIG_W
            items = [0] * num_ctas
            if total >= geometry.stages * num_ctas:

                def counts_for(limit):
                    return [
                        0 if no_fc1[c] else max(geometry.stages, (limit - f) // geometry.item_bytes)
                        for c, f in enumerate(fc2)
                    ]

                lo, hi = (
                    0,
                    max((f for c, f in enumerate(fc2) if not no_fc1[c]))
                    + total * geometry.item_bytes,
                )
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sum(counts_for(mid)) >= total:
                        hi = mid
                    else:
                        lo = mid + 1
                items = counts_for(lo)
                extra = sum(items) - total
                while extra > 0:
                    c = min(
                        (c for c in range(num_ctas) if items[c] > geometry.stages),
                        key=lambda c: (-(fc2[c] + items[c] * geometry.item_bytes), c),
                    )
                    items[c] -= 1
                    extra -= 1
            elif fine:
                lo_f, hi_f = (0.0, float(max(fc2) + total * geometry.item_bytes))
                for _ in range(100):
                    mid = 0.5 * (lo_f + hi_f)
                    if sum((max(0.0, (mid - f) / geometry.item_bytes) for f in fc2)) >= total:
                        hi_f = mid
                    else:
                        lo_f = mid
                wts = [max(0.0, (hi_f - f) / geometry.item_bytes) for f in fc2]
                sw = sum(wts)
                items = fine_ranges_fn([w * total / sw for w in wts], total)
            else:
                chosen = sorted(
                    sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // geometry.stages]
                )
                for c in chosen:
                    items[c] = geometry.stages
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g, c, p] = i0 | i0 + items[c] << 16
                i0 += items[c]
            assert i0 == total
    return (plan, max(nbs))


def make_phase_plan_tp8(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    fine: bool = True,
    *,
    geometry: MoePlanGeometry,
    fine_ranges_fn: Callable[[list[float], int], list[int]],
) -> tuple[torch.Tensor, int]:
    """The work split of ``m3_moe`` for each distinct-expert count G (4 .. 4 m_max): FC2 32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..) a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work, or (``fine``) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit (:func:`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and the largest block count."""
    gcap = geometry.top_k * m_max
    nb2 = geometry.hidden // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    blk_bytes = 32 * geometry.row2_bytes
    for g in range(geometry.top_k, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = b0s[c] | nbs[c] << 16
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * geometry.u1 * geometry.stages
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and (not (early0 and p == 0 and (m_max >= 2))):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            items = [0] * num_ctas
            if total >= geometry.stages * num_ctas:

                def counts_for(limit):
                    return [max(geometry.stages, (limit - f) // geometry.item_bytes) for f in fc2]

                lo, hi = (0, max(fc2) + total * geometry.item_bytes)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sum(counts_for(mid)) >= total:
                        hi = mid
                    else:
                        lo = mid + 1
                items = counts_for(lo)
                extra = sum(items) - total
                while extra > 0:
                    c = min(
                        (c for c in range(num_ctas) if items[c] > geometry.stages),
                        key=lambda c: (-(fc2[c] + items[c] * geometry.item_bytes), c),
                    )
                    items[c] -= 1
                    extra -= 1
            elif fine:
                lo_f, hi_f = (0.0, float(max(fc2) + total * geometry.item_bytes))
                for _ in range(100):
                    mid = 0.5 * (lo_f + hi_f)
                    if sum((max(0.0, (mid - f) / geometry.item_bytes) for f in fc2)) >= total:
                        hi_f = mid
                    else:
                        lo_f = mid
                wts = [max(0.0, (hi_f - f) / geometry.item_bytes) for f in fc2]
                sw = sum(wts)
                items = fine_ranges_fn([w * total / sw for w in wts], total)
            else:
                chosen = sorted(
                    sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // geometry.stages]
                )
                for c in chosen:
                    items[c] = geometry.stages
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g, c, p] = i0 | i0 + items[c] << 16
                i0 += items[c]
            assert i0 == total
    return (plan, max(nbs))
