# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host planner of the full-attention layer's attention half on the megakernel (``full_half.py``). Pure Python.

The half runs as two launches of the same library:

* launch Q: the library's QKV stage (``schedule.plan_qkv``: ``QkvRun`` on every CTA, 36 ``QkvSum``), the raw
  projection stored into a plain buffer;
* launch F (``plan_full_attention``), per CTA in order:

  - ``FaAttn`` (CTAs ``0 .. units - 1``, first: its region ``[0, fa::kRegionBytes)`` is written from kernel entry);
  - ``FaMerge`` (CTAs ``0 .. 127``): one (token, query head) row, its gated o_proj input as Lamport words into the
    arena (``kIFlagFaOutArena``);
  - ``Clear`` (``schedule.CLEAR_UNITS`` CTAs without a merge unit): the next launch's arena copy;
  - ``AhOproj`` (every CTA, last): 2-3 o_proj tiles into the hop1 slot; the weights are requested when the CTA's
    attention unit ends (its region overlaps the attention unit's).

Both launches need every CTA's shared memory at entry (the QKV run region and the attention unit's ring are each
~175 KB), so one launch cannot hold both without a streamed QKV.
"""

from __future__ import annotations

from typing import List

from . import schedule, schedule_attn, schedule_fa

#: The CTAs that take three o_proj tiles: the last ones (no merge unit there).
THREE_TILE_CTAS = schedule_attn.THREE_TILE_CTAS


def plan_full_attention(
    ctas: int, units: int, three_tile_ctas: int = THREE_TILE_CTAS
) -> schedule.Plan:
    """Launch F's unit lists (see the module docstring)."""
    assert 1 <= units <= ctas and schedule_fa.ROWS <= ctas
    free = list(range(schedule_fa.ROWS, ctas))
    clears = free[: schedule.CLEAR_UNITS]
    assert len(clears) == schedule.CLEAR_UNITS
    three = set(range(ctas - three_tile_ctas, ctas))
    plan: schedule.Plan = [[] for _ in range(ctas)]
    tile = 0
    for c in range(ctas):
        if c < units:
            plan[c].append(("FaAttn", c))
        if c < schedule_fa.ROWS:
            plan[c].append(("FaMerge", c))
        if c in clears:
            plan[c].append(("Clear", clears.index(c)))
        n = 3 if c in three else 2
        plan[c].append(("AhOproj", schedule_attn.oproj_unit(tile, n)))
        tile += n
    assert tile == schedule_attn.O_TILES, (
        f"{tile} o_proj tiles planned, want {schedule_attn.O_TILES}"
    )
    return plan


def unit_bytes_full(stage: str, unit: int) -> int:
    """Weight bytes of a unit of ``plan_full_attention``."""
    if stage == "AhOproj":
        return len(schedule_attn.oproj_tiles(unit)) * schedule_attn.O_TILE_BYTES
    return 0


def full_attention_tables(
    ctas: int, units: int, region_bytes: int, three_tile_ctas: int = THREE_TILE_CTAS
) -> schedule.Tables:
    """Launch F's role tables: the attention unit's region at offset 0 (dynamic), o_proj after it."""
    plan = plan_full_attention(ctas, units, three_tile_ctas)
    return schedule.build_tables(plan, unit_bytes_full, dynamic={"FaAttn": (0, int(region_bytes))})


def oproj_issue_points(tables: schedule.Tables) -> List[int]:
    """Per CTA, the index of the unit after which its o_proj weights are requested (tests)."""
    out = []
    for c, units in enumerate(tables.plan):
        i = [s for s, _ in units].index("AhOproj")
        out.append(int(tables.issue[c, i]))
    return out
