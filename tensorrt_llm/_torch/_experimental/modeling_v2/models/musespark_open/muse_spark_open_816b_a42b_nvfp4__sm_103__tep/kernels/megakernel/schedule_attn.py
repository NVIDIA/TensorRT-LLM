# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host planner of the attention-half megakernel (``stages/attnHalf.cuh``): one launch per sliding-window layer from
the QKV projection to the hop1 slot. Pure Python (no GPU), tested on the CPU.

Per CTA, in order:

* ``QkvRun`` (every CTA, unit = run): a run of the K-slice packed QKV weight (blocks heads first: q | k | v | gate),
  requested at entry. The last ``LATE_CTAS`` CTAs take pure-gate runs, the others the remaining runs in order;
* ``AhQkvFin`` (36 CTAs; q/k/v blocks on early CTAs, gate blocks on late ones): the block's sum, prep and KV append;
* ``AhSwaAttn`` (66 other early CTAs, unit = 33 KV head + block): one 64-key block's partial; blocks 0-31 also merge
  two query rows and apply the norm-gate;
* ``Clear`` (``schedule.CLEAR_UNITS`` early CTAs without a QkvFin or SwaAttn unit): the next launch's arena copy;
* ``AhOproj`` (every CTA, last; unit = first tile | tiles << 10): 3 tiles on ``three_tile_ctas`` CTAs that run nothing
  but the QKV run (and no clear unit), 2 on the others, 512 tiles in all.

Shared memory: the QKV run region ``[0, 178176)``; the attention unit's BF16 tiles ``[98304, 125952)`` and its entry
copies (pool pages) ``[178176, 186368)``; QkvFin's entry copies (RoPE rows) ``[178176, 180224)``; the o_proj weights
``[0, 98304)``, requested when the QKV run ends.
"""

from __future__ import annotations

from typing import List, Sequence

from . import schedule

#: Stage geometry (``stages/attnHalf.cuh``).
KV_HEADS = 2
ATTN_BLOCKS = 33
ATTN_UNITS = KV_HEADS * ATTN_BLOCKS
FIN_UNITS = schedule.QKV_BLOCKS
O_TILES = 512
O_TILE_BYTES = 16 * schedule.BF16_BLOCK
ENTRY_OFF = 178176
ENTRY_BYTES = 8192
SCRATCH_OFF = 98304
#: Units whose own code (or entry copies) fill a fixed region: stage -> (offset, bytes).
DYNAMIC = {
    "AhSwaAttn": (SCRATCH_OFF, ENTRY_OFF + ENTRY_BYTES - SCRATCH_OFF),
    "AhQkvFin": (ENTRY_OFF, 2048),
}
#: The CTAs that take three o_proj tiles (88 x 3 + 124 x 2 = 512).
THREE_TILE_CTAS = 88
#: QkvFin units of the q, k and v blocks (heads-first packed order: q_0..q_15 | k_0 k_1 | v_0 v_1 | g_0..g_15).
FIN_QKV = 20
#: The first gate tile of a K slice in the heads-first order.
GATE_TILE0 = FIN_QKV * schedule.QKV_TILES_PER_BLOCK
#: The last CTAs of the grid, which start last: under PDL behind the served boundary kernel (64 CTAs) they wait for
#: its SMs (so they start after the others). They take only pure-gate QKV runs, the gate QkvFin units and o_proj.
LATE_CTAS = 64


def _spread(pool: Sequence[int], n: int) -> List[int]:
    """``n`` members of ``pool`` evenly spaced over it."""
    assert 0 <= n <= len(pool), (n, len(pool))
    return [pool[(i * len(pool)) // n] for i in range(n)]


def oproj_unit(first_tile: int, tiles: int) -> int:
    """The unit id of o_proj tiles ``[first_tile, first_tile + tiles)``."""
    assert 0 <= first_tile < 1024 and 1 <= tiles <= MAX_O_TILES  # 4 with ``oproj_off_merge``
    return first_tile | (tiles << 10)


# ---- o_proj off the merge CTAs ----
#: ``stages/attnHalf.cuh`` ``kAhOMaxTiles``: a 4-tile unit (128 KiB of weights) only on a CTA without an attention
#: unit (whose scratch starts at ``SCRATCH_OFF``).
MAX_O_TILES = 4
assert MAX_O_TILES * O_TILE_BYTES > SCRATCH_OFF


def _oproj_counts(
    plan: schedule.Plan, ctas: int, off_merge: bool, three_tile_ctas: int
) -> List[int]:
    """o_proj tiles per CTA. Default: 3 on ``three_tile_ctas`` light CTAs, 2 on the others. ``off_merge``:
    none on the 64 CTAs whose attention unit also merges (they reach o_proj only after the last merge, so their o_proj
    set the attention half's end), 4 on as many CTAs without an attention unit as needed, 3 on the rest."""
    busy = [any(st in ("AhQkvFin", "AhSwaAttn", "Clear") for st, _ in p) for p in plan]
    if not off_merge:
        light = [c for c in range(ctas) if not busy[c]]
        three = set(_spread(light, three_tile_ctas))
        return [3 if c in three else 2 for c in range(ctas)]
    merge = [
        any(st == "AhSwaAttn" and u % ATTN_BLOCKS < ATTN_BLOCKS - 1 for st, u in p) for p in plan
    ]
    attn = [any(st == "AhSwaAttn" for st, _ in p) for p in plan]
    hosts = [c for c in range(ctas) if not merge[c]]
    four_ok = [c for c in hosts if not attn[c]]
    n4 = O_TILES - 3 * len(hosts)
    assert 0 <= n4 <= len(four_ok), (n4, len(four_ok))
    four = set(_spread(four_ok, n4))
    return [0 if merge[c] else (4 if c in four else 3) for c in range(ctas)]


# ---- end o_proj off the merge CTAs ----


def oproj_tiles(unit: int) -> range:
    """The tiles of an o_proj unit."""
    return range(unit & 1023, (unit & 1023) + (unit >> 10))


def pure_gate_runs(ctas: int) -> List[int]:
    """The QKV runs whose chunks all lie in gate tiles of the heads-first packed order (``GATE_TILE0`` on in every K
    slice)."""
    seg = schedule.seg_chunks(4)
    out = []
    for r in range(ctas):
        j0, j1 = schedule.qkv_run_bounds(r, ctas)
        if all(
            (g % schedule.QKV_TILES) >= GATE_TILE0 for g in range(j0 // seg, (j1 - 1) // seg + 1)
        ):
            out.append(r)
    return out


def fin_first_hosts(ctas: int, late_ctas: int = LATE_CTAS) -> List[int]:
    """(``MSO_MK_AH`` bit 2) The early CTAs whose QKV run is pure gate under ``plan_attention``'s run placement: a
    q / k / v QkvFin unit there can run first, because its CTA's own run feeds no q / k / v block."""
    early = list(range(ctas - late_ctas))
    late = list(range(ctas - late_ctas, ctas))
    gate_runs = pure_gate_runs(ctas)
    late_runs = _spread(gate_runs, late_ctas)
    early_runs = [r for r in range(ctas) if r not in set(late_runs)]
    run_of = dict(zip(early + late, early_runs + late_runs))
    return [c for c in early if run_of[c] in set(gate_runs)]


def plan_attention(
    ctas: int,
    slices: int = 4,
    three_tile_ctas: int = THREE_TILE_CTAS,
    late_ctas: int = LATE_CTAS,
    oproj_off_merge: bool = False,
    gate_in_merge: bool = False,
    fin_first: bool = False,
    gate_runs_off_fin: bool = False,
) -> schedule.Plan:
    """The attention half's unit lists (see the module docstring). ``oproj_off_merge``: ``_oproj_counts``.

    The ``MSO_MK_AH`` mask (``msoMegakernel.cu``): ``gate_in_merge`` (bit 1; the build must have it) drops the gate
    QkvFin units, because the merging SwaAttn units sum the gate from the gate runs' partials; ``fin_first`` (bit 2)
    puts the 20 q / k / v QkvFin units on ``fin_first_hosts`` as the CTA's first role, ahead of its pure-gate QKV run.

    ``gate_runs_off_fin`` (bit 8; with bit 2) takes those CTAs' pure-gate runs off them: each becomes the second
    QKV run of a plain early CTA (QKV run and o_proj only), so the gate partials the bit-1 merges read wait for no fin.
    """
    assert slices == 4, "the attention half uses the K-slice QKV order"
    early = list(range(ctas - late_ctas))
    late = list(range(ctas - late_ctas, ctas))
    late_runs = _spread(pure_gate_runs(ctas), late_ctas)
    early_runs = [r for r in range(ctas) if r not in set(late_runs)]
    run_of = dict(zip(early + late, early_runs + late_runs))
    gate_fin = late_ctas >= FIN_UNITS - FIN_QKV and not gate_in_merge  # bit 1: no gate fins
    if fin_first:  # bit 2
        hosts = fin_first_hosts(ctas, late_ctas)
        assert len(hosts) >= FIN_QKV, (
            f"fin_first: {len(hosts)} early pure-gate runs for {FIN_QKV} q / k / v fins"
        )
        fin_ctas = _spread(hosts, FIN_QKV)
    else:
        fin_ctas = _spread(early, FIN_QKV if gate_fin or gate_in_merge else FIN_UNITS)
    if gate_fin:
        fin_ctas += _spread(late, FIN_UNITS - FIN_QKV)
    elif not gate_in_merge and fin_first:
        fin_ctas += _spread([c for c in early if c not in set(fin_ctas)], FIN_UNITS - FIN_QKV)
    owners = {c: b for b, c in enumerate(fin_ctas)}
    free = [c for c in early if c not in owners]
    attn = _spread(free, ATTN_UNITS)
    rest = [c for c in free if c not in set(attn)]
    clears = _spread(rest, schedule.CLEAR_UNITS)
    attn_unit = {c: u for u, c in enumerate(attn)}
    clear_unit = {c: u for u, c in enumerate(clears)}
    second_run = {}  # gate_runs_off_fin: host CTA -> the fin-first CTA's pure-gate run it takes as its second run
    if gate_runs_off_fin:
        assert fin_first, "gate_runs_off_fin moves the fin-first runs: it needs fin_first"
        ff_ctas = [c for c in fin_ctas if owners[c] < FIN_QKV]
        plain = [c for c in rest if c not in set(clears)]
        assert len(plain) >= len(ff_ctas), (
            f"gate_runs_off_fin: {len(plain)} plain early CTAs for {len(ff_ctas)} runs"
        )
        second_run = {h: run_of[c] for h, c in zip(_spread(plain, len(ff_ctas)), ff_ctas)}
    plan: schedule.Plan = [[] for _ in range(ctas)]
    for c in range(ctas):
        if (
            fin_first and owners.get(c, FIN_QKV) < FIN_QKV
        ):  # fin_first: the q / k / v fin first, then the gate run
            plan[c].append(("AhQkvFin", owners[c]))
            if not gate_runs_off_fin:  # gate_runs_off_fin: the run is a plain CTA's second run
                plan[c].append(("QkvRun", run_of[c]))
            continue
        plan[c].append(("QkvRun", run_of[c]))
        if c in second_run:  # gate_runs_off_fin
            plan[c].append(("QkvRun", second_run[c]))
        if c in owners:
            plan[c].append(("AhQkvFin", owners[c]))
        if c in attn_unit:
            plan[c].append(("AhSwaAttn", attn_unit[c]))
        if c in clear_unit:
            plan[c].append(("Clear", clear_unit[c]))
    counts = _oproj_counts(plan, ctas, oproj_off_merge, three_tile_ctas)  # oproj_off_merge
    tile = 0
    for c in range(ctas):
        n = counts[c]
        if n:
            plan[c].append(("AhOproj", oproj_unit(tile, n)))
        tile += n
    assert tile == O_TILES, f"{tile} o_proj tiles planned, want {O_TILES}"
    return plan


def unit_bytes_attention(stage: str, unit: int, ctas: int) -> int:
    """Weight bytes of a unit of ``plan_attention``."""
    if stage == "QkvRun":
        return schedule.unit_bytes_qkv(stage, unit, ctas)
    if stage == "AhOproj":
        return len(oproj_tiles(unit)) * O_TILE_BYTES
    return 0


def attention_tables(
    ctas: int,
    slices: int = 4,
    three_tile_ctas: int = THREE_TILE_CTAS,
    late_ctas: int = LATE_CTAS,
    oproj_issue: str = "early",
    oproj: bool = True,
) -> schedule.Tables:
    """The attention half's role tables (the greedy placement; the dynamic regions above).

    Args:
        oproj_issue: ``"early"`` requests a CTA's o_proj weights as soon as its region is free (after the QKV run);
            ``"last"`` after the unit right before o_proj (the QkvFin, attention or clear unit where there is one).
        oproj: False drops the o_proj units (a diagnostic plan: nothing is stored into the slot).
    """
    assert oproj_issue in ("early", "last")
    plan = plan_attention(ctas, slices, three_tile_ctas, late_ctas)
    if not oproj:
        plan = [[r for r in p if r[0] != "AhOproj"] for p in plan]
    tables = schedule.build_tables(
        plan, lambda st, u: unit_bytes_attention(st, u, ctas), dynamic=DYNAMIC
    )
    if oproj_issue == "last":
        for c, p in enumerate(plan):
            i = len(p) - 1
            if p[i][0] == "AhOproj" and int(tables.issue[c, i]) < i - 1:
                tables.issue[c, i] = i - 1
    return tables
