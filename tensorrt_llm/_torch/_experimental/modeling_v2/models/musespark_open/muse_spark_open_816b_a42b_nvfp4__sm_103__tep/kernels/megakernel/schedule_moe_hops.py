# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host plan of the MoE-side hop stages (``stages/moe_hops.cuh``): which CTA runs which unit, in which order.
Pure Python (no GPU).

One launch is the MoE side of one decoder layer on one rank of 8 (the experts replaced by a stub). With 212 CTAs:

* CTAs ``[0, 64)``: ``MhPool`` (tile ``u // 2`` of the rank's pool slice, K half ``u % 2``), 128 KiB of weights each;
* ``[64, 128)``: ``MhUnpool`` (tile ``u``), 128 KiB each;
* ``[128, 160)``: ``MhRouter`` (expert ``32 rank + u``), 32 KiB each;
* ``[160, 192)``: unit ``u`` = (row ``u // 4``, quarter ``u % 4``) of ``MhOproj`` (stub), ``MhExpStub`` (stub) and
  ``MhFinal``; on the first ``8 U`` of them (U = boundary units per row, a build constant) also the boundary unit
  ``u`` = (row ``u // U``, part ``u % U``) of ``MhH1`` (after ``MhOproj``) and of ``MhF1`` (last) -- the two share the
  CTA and the thread map, so r' passes between them in shared memory; their coefficients are their unit weights;
* ``[192, 212)``: ``MhClear`` (the previous launch's copy of the arena and of the F2' areas).

Every weight-carrying unit's weights are requested at kernel entry (before the grid-dependency wait); the pool,
unpool and router CTAs have the lowest indices, so the next launch's CTAs that become resident first are the ones that
stream the most weight bytes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import schedule

POOL_UNITS = 64
UNPOOL_UNITS = 64
ROUTER_UNITS = 32
ROW_UNITS = 32
CLEAR_UNITS = 20
POOL_UNIT_BYTES = 64 * schedule.BF16_BLOCK
UNPOOL_UNIT_BYTES = 64 * schedule.BF16_BLOCK
ROUTER_UNIT_BYTES = 8192 * 4
#: Boundary units per row supported by the device code (``MSO_MK_MH_ROW_UNITS``), and its default.
ROW_UNIT_CHOICES = (1, 2, 4)
DEFAULT_ROW_UNITS = 4

#: First CTA of each group.
POOL_CTA0 = 0
UNPOOL_CTA0 = POOL_CTA0 + POOL_UNITS
ROUTER_CTA0 = UNPOOL_CTA0 + UNPOOL_UNITS
ROW_CTA0 = ROUTER_CTA0 + ROUTER_UNITS
CLEAR_CTA0 = ROW_CTA0 + ROW_UNITS
assert CLEAR_CTA0 + CLEAR_UNITS == 212

#: The units of one row CTA, in run order (the boundary units only on the first 8 U of them).
ROW_CHAIN = ("MhOproj", "MhH1", "MhExpStub", "MhFinal", "MhF1")


def coef_h1_bytes(row_units: int) -> int:
    """Unit weights of an ``MhH1`` unit: alpha1, beta1 (FP32) and the pre-FFN weight (BF16) of its columns."""
    return (8192 // row_units) * 10


def coef_f1_bytes(row_units: int) -> int:
    """Unit weights of an ``MhF1`` unit: alpha2, beta2 (FP32), the post-FFN gain and the next pre-QKV weight (BF16)."""
    return (8192 // row_units) * 12


def row_chain(u: int, row_units: int) -> List[Tuple[str, int]]:
    """The units of row CTA ``u``."""
    boundary = u < 8 * row_units
    return [(s, u) for s in ROW_CHAIN if boundary or s not in ("MhH1", "MhF1")]


def plan_moe_hops(ctas: int = 212, row_units: int = DEFAULT_ROW_UNITS) -> schedule.Plan:
    """The plan of one launch (see the module docstring)."""
    assert ctas == 212, "the MoE hop plan is laid out for 212 CTAs"
    assert row_units in ROW_UNIT_CHOICES
    # A row's units are one thread-block cluster in the MSO_MK_MH_XCHG = 2 builds (cluster rank = part).
    assert ROW_CTA0 % row_units == 0
    plan: schedule.Plan = [[] for _ in range(ctas)]
    for u in range(POOL_UNITS):
        plan[POOL_CTA0 + u].append(("MhPool", u))
    for u in range(UNPOOL_UNITS):
        plan[UNPOOL_CTA0 + u].append(("MhUnpool", u))
    for u in range(ROUTER_UNITS):
        plan[ROUTER_CTA0 + u].append(("MhRouter", u))
    for u in range(ROW_UNITS):
        plan[ROW_CTA0 + u] += row_chain(u, row_units)
    for u in range(CLEAR_UNITS):
        plan[CLEAR_CTA0 + u].append(("MhClear", u))
    return plan


#: The whole row's post-FFN gain (BF16 [8192]) that an ``MhF1`` unit adds to its weights when it sums F1's first norm
#: over the whole row itself (``MSO_MK_MH_F1FULL``).
F1_GAIN_BYTES = 8192 * 2


def unit_bytes_moe_hops(
    stage: str, unit: int, row_units: int = DEFAULT_ROW_UNITS, f1_full: bool = False
) -> int:
    """Weight bytes of a unit of ``plan_moe_hops``."""
    return {
        "MhPool": POOL_UNIT_BYTES,
        "MhUnpool": UNPOOL_UNIT_BYTES,
        "MhRouter": ROUTER_UNIT_BYTES,
        "MhH1": coef_h1_bytes(row_units),
        "MhF1": coef_f1_bytes(row_units) + (F1_GAIN_BYTES if f1_full and row_units > 1 else 0),
    }.get(stage, 0)


def moe_hops_tables(
    ctas: int = 212, row_units: int = DEFAULT_ROW_UNITS, f1_full: bool = False
) -> schedule.Tables:
    """The role tables of ``plan_moe_hops``."""
    return schedule.build_tables(
        plan_moe_hops(ctas, row_units), lambda st, u: unit_bytes_moe_hops(st, u, row_units, f1_full)
    )


def census(ctas: int = 212, row_units: int = DEFAULT_ROW_UNITS) -> Dict[str, int]:
    return schedule.plan_census(plan_moe_hops(ctas, row_units))


def boundary_unit_columns(unit: int, row_units: int) -> Tuple[int, int, int]:
    """(row, first column, columns) of a boundary unit (``MhH1`` / ``MhF1``)."""
    width = 8192 // row_units
    return unit // row_units, width * (unit % row_units), width
