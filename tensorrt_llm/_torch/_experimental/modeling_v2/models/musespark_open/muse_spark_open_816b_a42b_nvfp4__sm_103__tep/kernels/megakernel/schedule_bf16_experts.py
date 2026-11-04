# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host plan of the megakernel's BF16 MoE experts stage (``stages/bf16Experts.cuh``, layers 0 and 61), a CPU mirror of
its device-side cut and of its two-window placement.

The stage is one dynamic unit per CTA (``MoeExpertsBf16``), as the NVFP4 stage: units ``1..64`` also run one final unit
(token ``(u - 1) // 8``, columns ``512 ((u - 1) % 8) ..``). With ``E`` slots the FC1 list has ``E * 4096`` units (slot,
128-row tile of 64, k-block of 64 BF16) and the FC2 list ``E * 2048`` (slot, tile of 32, k-block); every unit is one
16 KiB BlockMajorK slab. CTA ``c`` takes ``[U c // ctas, U (c + 1) // ctas)`` of each list, or whole tiles with
``tile_cut``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from . import schedule

#: Geometry (``stages/bf16Experts_defs.cuh``).
KB = 64
FC1_TILES = 64
FC2_TILES = 32
FC1_UNITS = FC1_TILES * KB
FC2_UNITS = FC2_TILES * KB
UNIT_BYTES = 128 * 128
STAGES = 8
FINAL_UNITS = 64
WINDOW = 64
BLOCK = 32
RANKS = 8
EXPERTS = 256


def plan_bf16_experts(ctas: int) -> schedule.Plan:
    """One ``MoeExpertsBf16`` unit per CTA; CTAs ``0..63`` also carry one final unit each."""
    assert ctas >= FINAL_UNITS
    return [[("MoeExpertsBf16", c + 1 if c < FINAL_UNITS else 0)] for c in range(ctas)]


def bf16_experts_tables(ctas: int) -> schedule.Tables:
    """The role tables of ``plan_bf16_experts`` (the stage writes the whole unit region itself)."""
    return schedule.build_tables(
        plan_bf16_experts(ctas),
        lambda stage, unit: 0,
        dynamic={"MoeExpertsBf16": (0, schedule.SMEM_MAIN)},
    )


def bf16_cut(E: int, ctas: int, tile_cut: bool = False) -> Dict[str, object]:
    """Per list (``fc1``, ``fc2``): every CTA's unit range, and per tile its first CTA and its finishing CTA."""
    out: Dict[str, object] = {}
    for name, per in (("fc1", FC1_UNITS), ("fc2", FC2_UNITS)):
        U = E * per
        tiles = U // KB

        def start(c, U=U, tiles=tiles):
            if U == 0:
                return 0
            return (tiles * c // ctas) * KB if tile_cut else U * c // ctas

        def cta_of(u, U=U, tiles=tiles):
            if tile_cut:
                return (ctas * (u // KB + 1) - 1) // tiles
            return (ctas * (u + 1) - 1) // U

        ranges = [(start(c), start(c + 1)) for c in range(ctas)]
        tile_ctas: List[Tuple[int, int]] = [
            (cta_of(KB * g), cta_of(KB * g + KB - 1)) for g in range(tiles)
        ]
        out[name] = dict(units=U, ranges=ranges, tiles=tile_ctas)
    return out


def check_bf16_cut(E: int, ctas: int, tile_cut: bool = False) -> None:
    """Raises AssertionError unless the cut covers each list exactly once and every CTA leaves at most one unfinished
    tile per list (its last), with the finishing CTA's range holding the tile's last k-block."""
    cut = bf16_cut(E, ctas, tile_cut)
    for name in ("fc1", "fc2"):
        lst = cut[name]
        ranges = lst["ranges"]
        assert ranges[0][0] == 0 and ranges[-1][1] == lst["units"]
        for c in range(ctas - 1):
            assert ranges[c][1] == ranges[c + 1][0]
        for g, (first, fin) in enumerate(lst["tiles"]):
            a, b = ranges[fin]
            assert a <= KB * g + KB - 1 < b, (name, g, fin)
            assert ranges[first][0] <= KB * g < ranges[first][1], (name, g, first)
            for c in range(first, fin):
                assert KB * g <= ranges[c][1] - 1 < KB * g + KB, (name, g, c)
            if tile_cut:
                assert first == fin


def per_cta_bytes(E: int, ctas: int, tile_cut: bool = False) -> List[int]:
    """Weight bytes each CTA streams."""
    cut = bf16_cut(E, ctas, tile_cut)
    return [
        UNIT_BYTES
        * (
            (cut["fc1"]["ranges"][c][1] - cut["fc1"]["ranges"][c][0])
            + (cut["fc2"]["ranges"][c][1] - cut["fc2"]["ranges"][c][0])
        )
        for c in range(ctas)
    ]


def place2(selected: Sequence[int], rank: int) -> Tuple[List[int], List[int], List[int]]:
    """CPU mirror of ``z1::place2`` (the device's integer steps, in order): ``(local ids, loads before, loads after)``
    for the flat selected ids of one step. Equal to ``ring_buddy_assign.reference`` by construction of the rule."""
    present = sorted(set(int(e) for e in selected))
    words = [0] * RANKS
    for e in present:
        words[e // BLOCK] |= 1 << (e % BLOCK)
    load = [bin(w).count("1") for w in words]
    D = sum(load)
    tr = 0
    for r in range(RANKS):
        arc = 0
        for n in range(1, RANKS):
            arc += load[(r - n + 1) % RANKS]
            tr = max(tr, (arc + n) // (n + 1))
    T = max((D + RANKS - 1) // RANKS, tr)
    x = []
    for r in range(RANKS):
        c = 0
        arc = 0
        for n in range(1, RANKS + 1):
            arc += load[(r - n + 1) % RANKS]
            c = max(c, arc - n * T)
        x.append(c)
    local = []
    for e in selected:
        b, i = divmod(int(e), BLOCK)
        above = bin(words[b] >> (i + 1)).count("1")
        runs_on = (b + 1) % RANKS if above < x[b] else b
        if runs_on == rank:
            local.append(((rank - b) % RANKS) * BLOCK + i)
        else:
            local.append(WINDOW + int(e) % (EXPERTS - WINDOW))
    after = [load[r] - x[r] + x[(r - 1) % RANKS] for r in range(RANKS)]
    return local, load, after
