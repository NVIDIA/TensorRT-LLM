# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host planner of the megakernel (``msoMegakernel.cu``): per-CTA unit lists, shared-memory placement and the
point at which each unit's weights are requested. Pure Python (no GPU), so it is tested on the CPU.

A plan is one list of ``(stage, unit)`` per CTA, run in order. ``build_tables`` turns it into the three int32
``[ctas, MAX_ROLES]`` tables the kernel reads: the role (``stage << 16 | unit``), the unit's shared-memory offset and
its issue point. For a unit whose region is filled only by its weight copy, every candidate offset (0 and the 1 KiB
rounded end of every region already placed) is tried and the one whose latest overlapping earlier unit (the
"blocker") is earliest wins; the issue point is that blocker's index (the weights are requested right after it ends),
or ``ISSUE_AT_ENTRY`` when nothing earlier overlaps (requested at kernel entry, before the grid-dependency wait). A
unit without a weight copy gets ``ISSUE_NONE``. The kernel's QKV cut (``qkv_run_bounds``, ``qkv_cta_of_chunk``) is
mirrored here so the plan and the tests agree with the device code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

#: Stage codes (``mso_mk::Stage``).
STAGES: Dict[str, int] = {
    "QkvRun": 1,
    "QkvSum": 2,
    "Clear": 3,
    "XSsq": 4,
    # ---- attention half (stages/attnHalf.cuh; range 10-29) ----
    "AhQkvFin": 10,
    "AhSwaAttn": 11,
    "AhOproj": 12,
    # ---- end attention half ----
    "TestCopy": 16,
    "TestPub": 17,
    "TestSub": 18,
    "TestHang": 19,
    # ---- full attention (schedule_fa.py) ----
    "FaAttn": 50,
    "FaMerge": 51,
    # ---- end full attention ----
    # ---- MoE experts ----
    "MoeExperts": 60,
    # ---- end MoE experts ----
    # ---- MoE-side hops (schedule_moe_hops.py) ----
    "MhOproj": 80,
    "MhH1": 81,
    "MhRouter": 82,
    "MhPool": 83,
    "MhExpStub": 84,
    "MhFinal": 85,
    "MhUnpool": 86,
    "MhF1": 87,
    "MhClear": 88,
    # ---- end MoE-side hops ----
    # ---- whole-layer integration (layer_mk.py) ----
    "IClearX": 100,
    # ---- end whole-layer integration ----
    # ---- ghost and L2-warm units (layer_mk.py) ----
    "I2bGhost": 101,
    "I2bWarm": 102,
    # ---- end ghost and L2-warm units ----
    # ---- BF16 MoE experts of layers 0 / 61 (schedule_bf16_experts.py) ----
    "MoeExpertsBf16": 130,
    # ---- end BF16 MoE experts ----
}
STAGE_NAMES: Dict[int, str] = {code: name for name, code in STAGES.items()}

#: Units per CTA (``mso_mk::kMaxRoles``).
MAX_ROLES = 16
#: Bytes of unit regions in shared memory (``mso_mk::kSmemMain``).
SMEM_MAIN = 200704
#: Issue points: requested at kernel entry / the unit has no weight copy.
ISSUE_AT_ENTRY = -1
ISSUE_NONE = -2
#: Region alignment of the placement.
REGION_ALIGN = 1024

#: QKV geometry (``mso_mk::kQkv*``): [2304, 8192] in 144 tiles of 16 rows x 128 chunks of 64 columns.
HIDDEN = 8192
QKV_ROWS = 2304
CHUNK_COLS = 64
BF16_BLOCK = 2048
CHUNKS_PER_ROW = HIDDEN // CHUNK_COLS
QKV_TILES = QKV_ROWS // 16
QKV_CHUNKS = QKV_TILES * CHUNKS_PER_ROW
QKV_BLOCK_ROWS = 64
QKV_BLOCKS = QKV_ROWS // QKV_BLOCK_ROWS
QKV_TILES_PER_BLOCK = QKV_BLOCK_ROWS // 16
SSQ_CHUNKS = CHUNKS_PER_ROW
#: Bytes of one arena clear unit and of the arena copy (``mso_mk::kClearUnitBytes``, ``kArenaBytes``).
CLEAR_UNITS = 8

Plan = List[List[Tuple[str, int]]]


def qkv_run_bounds(cta: int, ctas: int) -> Tuple[int, int]:
    """Chunks ``[j0, j1)`` of the packed QKV weight's ``[tile][chunk]`` order that CTA ``cta`` multiplies."""
    return (QKV_CHUNKS * cta) // ctas, (QKV_CHUNKS * (cta + 1)) // ctas


def qkv_cta_of_chunk(chunk: int, ctas: int) -> int:
    """The CTA whose run holds ``chunk`` (``mso_mk::qkvCtaOfChunk``)."""
    return (ctas * (chunk + 1) - 1) // QKV_CHUNKS


def seg_chunks(slices: int) -> int:
    """Chunks of one segment = one (K slice, tile) of the packed order ``[slice][tile][chunk in slice]``."""
    assert slices in (1, 4)
    return CHUNKS_PER_ROW // slices


def qkv_max_slots(ctas: int, slices: int) -> int:
    """Partial slots of one run (segments it can touch; ``mso_mk::kQkvMaxSlots``)."""
    return (-(-QKV_CHUNKS // ctas) - 2) // seg_chunks(slices) + 2


def qkv_tile_runs(tile: int, ctas: int, slices: int = 1) -> List[Tuple[int, int]]:
    """``(cta, slot)`` of every run that covers ``tile`` (in its ``slices`` K slices), in ascending CTA order (the
    summation order). A run's slot k is the k-th segment of the packed order it touches."""
    L = seg_chunks(slices)
    out = []
    for s in range(slices):
        g = s * QKV_TILES + tile
        first = qkv_cta_of_chunk(g * L, ctas)
        last = qkv_cta_of_chunk(g * L + L - 1, ctas)
        out += [(c, g - qkv_run_bounds(c, ctas)[0] // L) for c in range(first, last + 1)]
    return out


def qkv_block_owner(block: int, ctas: int, slices: int = 1) -> int:
    """The CTA that runs ``QkvSum`` of 64-row block ``block``: the one holding the block's last chunk (in the last
    K slice)."""
    L = seg_chunks(slices)
    g = (slices - 1) * QKV_TILES + (block + 1) * QKV_TILES_PER_BLOCK - 1
    return qkv_cta_of_chunk(g * L + L - 1, ctas)


def plan_qkv(
    ctas: int, x_prime: bool = False, clear_cta: Optional[int] = None, slices: int = 1
) -> Plan:
    """The QKV stage on every CTA.

    Every CTA runs its ``QkvRun`` (weights requested at entry); the 36 ``QkvSum`` units follow the run on the CTA
    that holds their block's last chunk. With ``x_prime`` the 128 ``XSsq`` units (one per 64-column chunk of the FP32
    residual) come first on CTAs without a ``QkvSum`` unit. One ``Clear`` unit empties the next launch's arena copy
    after the run on a CTA without a ``QkvSum`` (or ``XSsq``) unit."""
    plan: Plan = [[] for _ in range(ctas)]
    owners = {qkv_block_owner(b, ctas, slices): b for b in range(QKV_BLOCKS)}
    assert len(owners) == QKV_BLOCKS, "two blocks on one CTA"
    free = [c for c in range(ctas) if c not in owners]
    ssq_ctas = free[:SSQ_CHUNKS] if x_prime else []
    assert len(ssq_ctas) == (SSQ_CHUNKS if x_prime else 0), "not enough CTAs for the XSsq units"
    for u, c in enumerate(ssq_ctas):
        plan[c].append(("XSsq", u))
    for c in range(ctas):
        plan[c].append(("QkvRun", c))
    for c, b in owners.items():
        plan[c].append(("QkvSum", b))
    if clear_cta is None:
        rest = [c for c in free if c not in ssq_ctas]
        clear_cta = rest[-1] if rest else free[-1]
    for u in range(CLEAR_UNITS):
        plan[clear_cta].append(("Clear", u))
    return plan


def unit_bytes_qkv(stage: str, unit: int, ctas: int) -> int:
    """Weight bytes of a unit of ``plan_qkv``."""
    if stage == "QkvRun":
        j0, j1 = qkv_run_bounds(unit, ctas)
        return (j1 - j0) * BF16_BLOCK
    return 0


@dataclass
class Tables:
    """The three role tables (int32 ``[ctas, MAX_ROLES]``) and the plan they came from."""

    roles: torch.Tensor
    smem: torch.Tensor
    issue: torch.Tensor
    plan: Plan

    @property
    def ctas(self) -> int:
        return int(self.roles.shape[0])

    def flat(self) -> torch.Tensor:
        """int32 ``[3, ctas, MAX_ROLES]``, the layout after the parameter block."""
        return torch.stack([self.roles, self.smem, self.issue]).contiguous()


def build_tables(
    plan: Plan,
    unit_bytes: Callable[[str, int], int],
    smem_main: int = SMEM_MAIN,
    dynamic: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tables:
    """The role tables of ``plan`` (a greedy placement).

    Args:
        plan: per CTA its ``(stage name, unit)`` list, in run order.
        unit_bytes: weight bytes of ``(stage, unit)``; 0 = no region.
        smem_main: bytes of unit regions.
        dynamic: stage -> ``(offset, bytes)`` for units whose own code writes a fixed region (no weight copy);
            they block later units that overlap it.
    """
    dynamic = dynamic or {}
    ctas = len(plan)
    roles = torch.zeros(ctas, MAX_ROLES, dtype=torch.int32)
    smem = torch.zeros(ctas, MAX_ROLES, dtype=torch.int32)
    issue = torch.full((ctas, MAX_ROLES), ISSUE_NONE, dtype=torch.int32)
    for c, units in enumerate(plan):
        assert len(units) <= MAX_ROLES, (c, units)
        placed: List[Tuple[int, int, int]] = []  # (offset, size, index)
        for i, (stage, unit) in enumerate(units):
            assert 0 <= unit < 1 << 16, (c, stage, unit)
            roles[c, i] = (STAGES[stage] << 16) | unit
            if stage in dynamic:
                off, size = dynamic[stage]
                smem[c, i] = off
                placed.append((off, size, i))
                continue
            size = unit_bytes(stage, unit)
            if size == 0:
                continue
            cands = sorted(
                {0, *[-(-(o + s) // REGION_ALIGN) * REGION_ALIGN for (o, s, _) in placed]}
            )
            best = None
            for off in cands:
                if off + size > smem_main:
                    continue
                blk = max(
                    [j for (o, s, j) in placed if o < off + size and off < o + s],
                    default=ISSUE_AT_ENTRY,
                )
                if best is None or blk < best[0]:
                    best = (blk, off)
            assert best is not None, f"CTA {c}: {stage} {unit} ({size} B) does not fit"
            issue[c, i], smem[c, i] = best
            placed.append((best[1], size, i))
    tables = Tables(roles, smem, issue, plan)
    validate(tables, unit_bytes, smem_main, dynamic)
    return tables


def validate(
    tables: Tables,
    unit_bytes: Callable[[str, int], int],
    smem_main: int = SMEM_MAIN,
    dynamic: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    """Raises AssertionError unless every region is in bounds and 1 KiB aligned, and every unit's region overlaps
    only regions of units that end no later than its issue point (so its copy never lands on live data)."""
    dynamic = dynamic or {}
    for c, units in enumerate(tables.plan):
        regions = []
        for i, (stage, unit) in enumerate(units):
            size = dynamic[stage][1] if stage in dynamic else unit_bytes(stage, unit)
            off = int(tables.smem[c, i])
            iss = int(tables.issue[c, i])
            if size == 0:
                assert iss == ISSUE_NONE, (c, i)
                continue
            assert 0 <= off and off + size <= smem_main, (c, i, off, size)
            assert off % REGION_ALIGN == 0, (c, i, off)
            if stage in dynamic:
                assert iss == ISSUE_NONE, (c, i)
                limit = i  # a dynamic unit's own code writes its region while it runs
            else:
                assert ISSUE_AT_ENTRY <= iss < i, (c, i, iss)
                limit = iss + 1
            for j, (o, s) in regions:
                if o < off + size and off < o + s:
                    assert j < limit, (
                        f"CTA {c}: unit {i} region overlaps unit {j}, issued after {iss}"
                    )
            regions.append((i, (off, size)))
        for i in range(len(units), MAX_ROLES):
            assert int(tables.roles[c, i]) == 0 and int(tables.issue[c, i]) == ISSUE_NONE, (c, i)


def units_of(tables: Tables, stage: str) -> List[Tuple[int, int, int]]:
    """``(cta, role index, unit)`` of every unit of ``stage``."""
    code = STAGES[stage]
    out = []
    for c in range(tables.ctas):
        for i in range(MAX_ROLES):
            r = int(tables.roles[c, i])
            if r == 0:
                break
            if r >> 16 == code:
                out.append((c, i, r & 0xFFFF))
    return out


def plan_census(plan: Sequence[Sequence[Tuple[str, int]]]) -> Dict[str, int]:
    """Units per stage."""
    out: Dict[str, int] = {}
    for units in plan:
        for stage, _ in units:
            out[stage] = out.get(stage, 0) + 1
    return out
