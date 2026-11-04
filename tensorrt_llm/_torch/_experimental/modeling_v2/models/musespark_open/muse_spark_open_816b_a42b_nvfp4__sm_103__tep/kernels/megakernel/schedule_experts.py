# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host plan of the megakernel's MoE experts stage (``stages/experts.cuh``) and a CPU mirror of its device-side cut.

The stage is one dynamic unit per CTA (``MoeExperts``; it owns the whole unit region, its weights are requested by
its own code once the routing is known). Units ``1..64`` also run one final unit (token ``(u - 1) // 8``, columns
``512 ((u - 1) % 8) ..``) after the CTA's expert work; unit ``0`` runs none.

The cut (``experts_cut``) mirrors ``p4a::makeCut`` / ``ctaOfUnit``: with ``E`` slots the FC1 list has ``E * 1024``
units (slot, 128-row tile of 64, k-block of 16) and the FC2 list ``E * 512`` (slot, tile of 32, k-block); CTA ``c``
takes ``[U c // ctas, U (c + 1) // ctas)`` of each list, or whole tiles with ``tile_cut``. A CTA walks the tiles of
its part from the last to the first (``p4a::listUnit``): the last tile's partial is out before the CTA waits for the
partial of its first tile, so the cross-CTA waits are one hop deep.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import schedule

#: Geometry (``stages/experts_defs.cuh``).
KB = 16
FC1_TILES = 64
FC2_TILES = 32
FC1_UNITS = FC1_TILES * KB
FC2_UNITS = FC2_TILES * KB
UNIT_BYTES = 128 * 128 + 128 * 16
STAGES = 9
FINAL_UNITS = 64
MAX_SLOTS = 64


def plan_experts(ctas: int) -> schedule.Plan:
    """One ``MoeExperts`` unit per CTA; CTAs ``0..63`` also carry one final unit each."""
    assert ctas >= FINAL_UNITS
    return [[("MoeExperts", c + 1 if c < FINAL_UNITS else 0)] for c in range(ctas)]


def experts_tables(ctas: int) -> schedule.Tables:
    """The role tables of ``plan_experts`` (the stage writes the whole unit region itself)."""
    return schedule.build_tables(
        plan_experts(ctas),
        lambda stage, unit: 0,
        dynamic={"MoeExperts": (0, schedule.SMEM_MAIN)},
    )


def experts_cut(E: int, ctas: int, tile_cut: bool = False) -> Dict[str, object]:
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


def check_cut(E: int, ctas: int, tile_cut: bool = False) -> None:
    """Raises AssertionError unless the cut covers each list exactly once, every CTA's range is consistent with
    ``cta_of``, and no CTA leaves more than one unfinished tile per list (its last)."""
    cut = experts_cut(E, ctas, tile_cut)
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
                # a contributor's last unit is inside this tile: the tile is its last one
                assert KB * g <= ranges[c][1] - 1 < KB * g + KB, (name, g, c)
            if tile_cut:
                assert first == fin


def per_cta_bytes(E: int, ctas: int, tile_cut: bool = False) -> List[int]:
    """Weight bytes each CTA streams."""
    cut = experts_cut(E, ctas, tile_cut)
    return [
        UNIT_BYTES
        * (
            (cut["fc1"]["ranges"][c][1] - cut["fc1"]["ranges"][c][0])
            + (cut["fc2"]["ranges"][c][1] - cut["fc2"]["ranges"][c][0])
        )
        for c in range(ctas)
    ]


# ---- the stock persistent_experts tile queue (stages/experts.cuh pxSchedule / pxTile, kMoeFlagPxQueue) ----
def px_schedule(E: int, ctas: int) -> Tuple[int, int]:
    """(working CTAs G, early slots) of the stock queue with ``balance = 1`` and ``order = 1``."""
    total = E * (FC1_TILES + FC2_TILES)
    rounds = (total + ctas - 1) // ctas
    G = (total + rounds - 1) // rounds if rounds > 0 else ctas
    early = 0
    fc1_total = E * FC1_TILES
    if fc1_total > 0:
        r_mixed = fc1_total // G
        in_mixed = G * (r_mixed + 1) - fc1_total
        if in_mixed < G and r_mixed + 1 < rounds:
            early = min((in_mixed + FC2_TILES - 1) // FC2_TILES, E, (G * r_mixed) // FC1_TILES)
    return G, early


def px_tile(E: int, G: int, early: int, p: int) -> Tuple[int, int, int]:
    """Queue position ``p`` -> (fc, slot, 128-row tile)."""
    fc1_total = E * FC1_TILES
    if p < fc1_total:
        slot, mt = divmod(p, FC1_TILES)
        if p >= early * FC1_TILES:
            q = p - early * FC1_TILES
            rest = E - early
            mt, r = divmod(q, rest)
            slot = early + r
        return 0, slot, mt
    q = p - fc1_total
    return 1, q // FC2_TILES, q % FC2_TILES


def px_tiles(E: int, ctas: int) -> List[List[Tuple[int, int, int]]]:
    """Per CTA its tiles in order (positions c, c + G, ...)."""
    G, early = px_schedule(E, ctas)
    total = E * (FC1_TILES + FC2_TILES)
    return [
        [px_tile(E, G, early, p) for p in range(c, total, G)] if c < G else [] for c in range(ctas)
    ]


# ---- end tile queue ----
