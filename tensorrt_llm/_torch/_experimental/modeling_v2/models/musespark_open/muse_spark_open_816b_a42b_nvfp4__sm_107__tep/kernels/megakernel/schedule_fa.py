# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host planner of the megakernel's full-attention stages (``stages/fullAttention.cuh``). Pure Python (no GPU).

Every CTA ``c < units`` runs the attention unit ``c`` first (its shared-memory region is written from kernel entry on,
so nothing may precede it on the CTA); the 128 merge units (one per (token, query head) row) follow on CTAs ``0 ..
127`` (``merge_cta``). The device mirrors ``fa_geometry`` (``faGeom``): the keyed units and their page ranges come from
the context length at run time, never from the host.
"""

from __future__ import annotations

from typing import List, Tuple

from . import schedule

#: Rows of the decode step (tokens x query heads of one rank) = merge units.
ROWS = 128
PAGE = 32
#: ``fa::kMaxUnitPages``: pages one attention unit can hold the block ids of.
MAX_UNIT_PAGES = 256
#: The stock split rule's fewest pages per keyed split (``full_attention_split_kv.MIN_PAGES``).
DEFAULT_MIN_PAGES = 4


def fa_geometry(seq_len: int, units: int, min_pages: int, unit: int) -> Tuple[int, int, int]:
    """``(eff, p0, p1)``: the keyed unit count and unit ``unit``'s pages ``[p0, p1)`` at context ``seq_len``."""
    pages = (seq_len + PAGE - 1) // PAGE
    eff = min(units, (pages + min_pages - 1) // min_pages)
    if unit >= eff:
        return eff, 0, 0
    return eff, unit * pages // eff, (unit + 1) * pages // eff


def max_context(units: int) -> int:
    """The longest context whose every unit fits ``MAX_UNIT_PAGES`` (with all units keyed)."""
    return MAX_UNIT_PAGES * units * PAGE


def plan_fa(ctas: int, units: int, merges: int = ROWS) -> schedule.Plan:
    """The attention units on CTAs ``0 .. units - 1`` (first on each), merge unit ``m`` on CTA
    ``merge_cta(ctas, m)``."""
    assert 1 <= units <= ctas and merges == ROWS and merges <= ctas
    plan: schedule.Plan = [[] for _ in range(ctas)]
    for c in range(units):
        plan[c].append(("FaAttn", c))
    for m in range(merges):
        plan[merge_cta(ctas, m)].append(("FaMerge", m))
    return plan


def merge_cta(ctas: int, m: int, merges: int = ROWS) -> int:
    """The CTA of merge unit ``m``: CTA ``m``. The next launch's lowest CTAs start first (on the SMs the CTAs without a
    merge free early), so they are the ones that carry the merges; at short contexts the CTAs ``eff .. 127`` hold no
    keys and their merges poll from the start."""
    return m


def fa_tables(ctas: int, units: int, region_bytes: int) -> schedule.Tables:
    """Role tables of ``plan_fa``: the attention unit's region (``fa::kRegionBytes``) at offset 0 (dynamic: its own
    code fills it); no unit has a weight copy."""
    plan = plan_fa(ctas, units)
    return schedule.build_tables(plan, lambda st, u: 0, dynamic={"FaAttn": (0, int(region_bytes))})


def unit_pages(seq_len: int, units: int, min_pages: int) -> List[int]:
    """Pages per unit (tests and reports)."""
    return [
        p1 - p0 for (_, p0, p1) in (fa_geometry(seq_len, units, min_pages, u) for u in range(units))
    ]
