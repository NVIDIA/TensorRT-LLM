# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision policy of adaptive EPLB: the time model, the layout planner and the install decisions,
and the operands of the ops that apply them (route tables, shares, statistics buffers).

The functions compute in a fixed order on inputs that all EP ranks share, so every rank reaches the
same decision. A layout is the expert of every slot (rank-major, ``-1`` when free); a slot's share is
the fraction of its expert's rows (routed tokens) it receives; ``[L, ...]`` arrays have a row per layer.
The module imports numpy only: the operands import torch where they call the ops.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch

_ExpertSlots = list[tuple[int, ...]]


@dataclass(frozen=True)
class ExpertGeometry:
    """``num_experts`` experts and ``num_slots >= num_experts`` slots over ``ep_size`` ranks."""

    num_experts: int
    num_slots: int
    ep_size: int

    @property
    def local_slots(self) -> int:
        return self.num_slots // self.ep_size

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ep_size

    def home_layout(self) -> np.ndarray:
        """``[num_slots]`` int64: rank ``r`` holds expert ``r * local_experts + j`` in local slot ``j``."""
        layout = np.full(self.num_slots, -1, dtype=np.int64)
        experts = np.arange(self.num_experts)
        layout[experts // self.local_experts * self.local_slots + experts % self.local_experts] = (
            experts
        )
        return layout


@dataclass(frozen=True)
class TimeModel:
    """Microseconds of one decode step of one MoE layer on one GPU, given by the MoE backend.

    A GPU's time is ``scale`` times the sum of: ``per_active_slot_us`` for every slot that receives
    rows (streaming its weights), ``per_extra_tile_us`` for every tile of ``tile_rows`` rows a slot
    needs beyond its first, ``per_row_us`` for every row, and ``hot_slot_extra_tile_us`` for every
    tile beyond the first of the GPU's busiest slot. A layer takes its busiest GPU's time, and
    ``route_us`` more while it holds a replica: the route op's path that spreads an expert's rows.
    With ``time_shares``, :func:`solve_shares` minimizes the busiest GPU's time, not its rows.
    ``scales`` holds ``(rows, scale, per_active_slot_us, route_us, time_shares)`` per calibrated
    CUDA graph bucket, ascending by the rows a GPU's tokens route per forward in it (:meth:`at`);
    empty, the fields hold at every size.
    """

    per_active_slot_us: float
    per_extra_tile_us: float
    per_row_us: float
    hot_slot_extra_tile_us: float
    scale: float
    tile_rows: int
    scales: tuple[tuple[float, float, float, float, bool], ...] = ()
    route_us: float = 0.0
    time_shares: bool = False

    def at(self, rows: float) -> TimeModel:
        """The model of the smallest bucket of at least ``rows`` rows per GPU per forward, the
        bucket a step of that size pads to (the largest bucket beyond them)."""
        if not self.scales:
            return self
        i = bisect.bisect_left([size for size, *_ in self.scales], rows)
        _, scale, read, route, timed = self.scales[min(i, len(self.scales) - 1)]
        bucket = dict(per_active_slot_us=read, route_us=route, time_shares=timed)
        return replace(self, scale=scale, scales=(), **bucket)

    @property
    def slope_us(self) -> float:
        """Time of one more row on a slot that fills its tiles."""
        return self.scale * (self.per_extra_tile_us / self.tile_rows + self.per_row_us)

    def layer_time_us(self, rows: np.ndarray, ep_size: int) -> np.ndarray:
        """``[L]`` float64 time of every layer for ``[L, num_slots]`` rows per slot."""
        rows = rows.reshape(rows.shape[0], ep_size, -1)
        tiles = np.ceil(rows / float(self.tile_rows))
        gpu = self.scale * (
            self.per_active_slot_us * (rows > 0).sum(2)
            + self.per_extra_tile_us * np.maximum(tiles - 1.0, 0.0).sum(2)
            + self.per_row_us * rows.sum(2)
            + self.hot_slot_extra_tile_us * np.maximum(tiles.max(2) - 1.0, 0.0)
        )
        return gpu.max(1)

    def balanced_time_us(self, load: np.ndarray, geometry: ExpertGeometry) -> np.ndarray:
        """``[L]`` time of every layer with its ``[L, num_experts]`` rows spread evenly."""
        rows = np.zeros((load.shape[0], geometry.ep_size, geometry.local_slots))
        rows[:, :, : geometry.local_experts] = (load.sum(1) / geometry.num_experts)[:, None, None]
        return self.layer_time_us(rows.reshape(load.shape[0], -1), geometry.ep_size)


def slot_rows(load: np.ndarray, layout: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """``[L, num_slots]`` rows of every slot: its expert's rows times the slot's share."""
    active = layout >= 0
    rows = np.zeros(layout.shape)
    np.multiply(np.take_along_axis(load, np.maximum(layout, 0), 1), shares, out=rows, where=active)
    return rows


# --- Layout planner ---


def active_slots(geometry: ExpertGeometry) -> int:
    """Slots a planned layout may occupy: every GPU keeps as many non-home slots free as it holds
    replicas, so one install can replace all of a GPU's replicas (new replicas are copied into free
    slots while the old ones still serve routes)."""
    spare = geometry.local_slots - geometry.local_experts
    return geometry.num_slots - geometry.ep_size * (spare // 2)


@dataclass(frozen=True)
class Plan:
    """A planned layout: ``expert_slots`` lists the ascending slots of every expert."""

    expert_slots: _ExpertSlots
    layout: np.ndarray


def plan_layout(
    load: Sequence[float] | np.ndarray,
    geometry: ExpertGeometry,
    home_layout: np.ndarray,
    current_layout: np.ndarray,
    total_active_slots: int,
) -> Plan:
    """Plan the next layout of one layer from its ``[num_experts]`` rows.

    Every expert keeps its home slot, and each further slot up to ``total_active_slots`` goes to the
    expert with the most rows per copy. Replicas go, most rows first, to the least loaded GPU that
    does not hold the expert yet. New replicas only take slots that ``current_layout`` leaves free,
    and an installed replica that stays on its GPU keeps its slot.
    """
    num_experts = geometry.num_experts
    rows = np.asarray(load, dtype=np.float64).tolist()
    homes = np.asarray(home_layout).tolist()
    home_slot = [slots[0] for slots in _expert_slots_of(homes, num_experts)]
    is_home = [expert >= 0 for expert in homes]
    proposed = _place_replicas(rows, homes, home_slot, geometry, int(total_active_slots))
    installed = _expert_slots_of(np.asarray(current_layout).tolist(), num_experts)
    fitted = _fit_free_slots(installed, proposed, rows, geometry)
    expert_slots = _one_install(installed, fitted, is_home, geometry)
    layout = np.full(geometry.num_slots, -1, dtype=np.int64)
    for expert, slots in enumerate(expert_slots):
        layout[list(slots)] = expert
    return Plan(expert_slots=expert_slots, layout=layout)


def _expert_slots_of(layout: list[int], num_experts: int) -> _ExpertSlots:
    slots_of: list[list[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(layout):
        if expert >= 0:
            slots_of[expert].append(slot)
    return [tuple(slots) for slots in slots_of]


def _copies(rows: list[float], total: int) -> list[int]:
    """Copies of every expert; without rows the extra slots go to the lowest ids."""
    num_experts = len(rows)
    if total == num_experts:
        return [1] * num_experts
    if not any(rows):
        base, rest = divmod(total, num_experts)
        return [base + (expert < rest) for expert in range(num_experts)]
    copies = [1] * num_experts
    # Max-heap on (rows per copy, expert id): a tie pops the larger id first.
    heap = [(-rows[expert], -expert) for expert in range(num_experts)]
    heapq.heapify(heap)
    for _ in range(total - num_experts):
        _, neg_expert = heapq.heappop(heap)
        expert = -neg_expert
        copies[expert] += 1
        heapq.heappush(heap, (-(rows[expert] / float(copies[expert])), neg_expert))
    return copies


def _place_replicas(
    rows: list[float], homes: list[int], home_slot: list[int], geometry: ExpertGeometry, total: int
) -> _ExpertSlots:
    """Every expert's home slot and its replicas on the free slots of ``homes``."""
    num_experts = geometry.num_experts
    ep_size = geometry.ep_size
    local = geometry.local_slots
    copies = _copies(rows, total)
    free = [[s for s in range(r * local, (r + 1) * local) if homes[s] < 0] for r in range(ep_size)]
    limit = [min(local - geometry.local_experts, len(free[r])) for r in range(ep_size)]
    experts_on_gpu = [
        {e for e in homes[r * local : (r + 1) * local] if e >= 0} for r in range(ep_size)
    ]
    gpu_rows = [
        math.fsum([rows[e] / copies[e] for e in sorted(experts_on_gpu[r])]) for r in range(ep_size)
    ]
    replicas = sorted(
        ((rows[e] / copies[e], e) for e in range(num_experts) for _ in range(copies[e] - 1)),
        key=lambda item: (-item[0], item[1]),
    )
    taken = [0] * ep_size
    placed: list[list[int]] = [[] for _ in range(ep_size)]
    # Min-heap on (rows, GPU) of the GPUs with room left.
    heap = [(gpu_rows[r], r) for r in range(ep_size) if taken[r] < limit[r]]
    heapq.heapify(heap)
    for size, expert in replicas:
        holding = []
        gpu = -1
        while heap:
            entry = heapq.heappop(heap)
            if expert in experts_on_gpu[entry[1]]:
                holding.append(entry)
                continue
            gpu = entry[1]
            break
        if gpu < 0:
            # Every GPU with room already holds the expert: add a copy on the least loaded one.
            gpu = holding.pop(0)[1]
        for entry in holding:
            heapq.heappush(heap, entry)
        experts_on_gpu[gpu].add(expert)
        gpu_rows[gpu] += size
        taken[gpu] += 1
        placed[gpu].append(expert)
        if taken[gpu] < limit[gpu]:
            heapq.heappush(heap, (gpu_rows[gpu], gpu))
    slots: dict[int, list[int]] = {}
    for gpu in range(ep_size):
        for k, expert in enumerate(placed[gpu]):
            slots.setdefault(expert, [home_slot[expert]]).append(free[gpu][k])
    expert_slots: _ExpertSlots = [(slot,) for slot in home_slot]
    for expert, chosen in slots.items():
        expert_slots[expert] = tuple(sorted(chosen))
    return expert_slots


def _changes(
    installed: _ExpertSlots, new: _ExpertSlots, geometry: ExpertGeometry
) -> tuple[list[set[int]], dict[int, list[int]], dict[int, list[tuple[int, int]]], list[list[int]]]:
    """Compare ``installed`` with ``new``: the slots every expert keeps, the experts every GPU gains,
    the ``(expert, slot)`` pairs every GPU loses, and every GPU's free slots under ``installed``."""
    local = geometry.local_slots
    occupied = {slot for slots in installed for slot in slots}
    free = [
        [s for s in range(r * local, (r + 1) * local) if s not in occupied]
        for r in range(geometry.ep_size)
    ]
    kept: list[set[int]] = []
    adds: dict[int, list[int]] = {}
    drops: dict[int, list[tuple[int, int]]] = {}
    for expert, (now, target) in enumerate(zip(installed, new)):
        if now == target:
            kept.append(set(now))
            continue
        now_set, new_set = set(now), set(target)
        kept.append(now_set & new_set)
        for slot in sorted(new_set - now_set):
            adds.setdefault(slot // local, []).append(expert)
        for slot in now_set - new_set:
            drops.setdefault(slot // local, []).append((expert, slot))
    return kept, adds, drops, free


def _fit_free_slots(
    installed: _ExpertSlots, proposed: _ExpertSlots, rows: list[float], geometry: ExpertGeometry
) -> _ExpertSlots:
    """Keep, GPU by GPU, as many new replicas of ``proposed`` as the free slots of ``installed``
    hold, the replicas with the most rows per copy first."""
    local = geometry.local_slots
    kept, adds, drops, free = _changes(installed, proposed, geometry)
    copies = [len(slots) for slots in proposed]

    def order(expert: int) -> tuple[float, int]:
        return (-(rows[expert] / copies[expert]), expert)

    for gpu in range(geometry.ep_size):
        want = sorted(adds.get(gpu, []), key=order)
        room = len(free[gpu])
        for expert, slot in zip(want[:room], free[gpu]):
            kept[expert].add(slot)
        short = len(want) - room
        for expert, slot in sorted(drops.get(gpu, []), key=lambda d: order(d[0]) + (d[1],)):
            if short <= 0:
                break
            if any(s // local == gpu for s in kept[expert]):
                continue
            kept[expert].add(slot)
            short -= 1
    return [tuple(sorted(slots)) for slots in kept]


def _keep_replica_slots(
    installed: _ExpertSlots, proposed: _ExpertSlots, is_home: list[bool], geometry: ExpertGeometry
) -> _ExpertSlots:
    """Move every replica of ``proposed`` to the slot of an installed replica of the same expert on
    the same GPU, when there is one.

    Any other replica keeps its slot if that slot is free, else takes the GPU's lowest free non-home
    slot.
    """
    local = geometry.local_slots
    installed_replicas: dict[tuple[int, int], list[int]] = {}  # (expert, GPU) -> replica slots
    for expert, slots in enumerate(installed):
        if len(slots) > 1:
            for slot in sorted(slots):
                if not is_home[slot]:
                    installed_replicas.setdefault((expert, slot // local), []).append(slot)
    result = list(proposed)
    rebuilt: dict[int, list[int]] = {}
    loose: dict[int, list[tuple[int, int]]] = {}
    taken: dict[int, set[int]] = {}
    for expert, expert_slots in enumerate(proposed):
        if len(expert_slots) == 1:
            continue
        by_gpu: dict[int, list[int]] = {}
        slots = []
        for slot in sorted(expert_slots):
            if is_home[slot]:
                slots.append(slot)
            else:
                by_gpu.setdefault(slot // local, []).append(slot)
        for gpu, want in by_gpu.items():
            have = installed_replicas.get((expert, gpu), [])
            same = [s for s in want if s in have]
            rest_have = [s for s in have if s not in same]
            rest_want = [s for s in want if s not in same]
            paired = same + rest_have[: len(rest_want)]
            slots.extend(paired)
            taken.setdefault(gpu, set()).update(paired)
            if len(rest_want) > len(rest_have):
                loose.setdefault(gpu, []).extend((expert, s) for s in rest_want[len(rest_have) :])
        rebuilt[expert] = slots
    for gpu, items in loose.items():
        used = taken.setdefault(gpu, set())
        moved = []
        for expert, slot in items:
            if slot in used:
                moved.append(expert)
            else:
                used.add(slot)
                rebuilt[expert].append(slot)
        spare = [
            s for s in range(gpu * local, (gpu + 1) * local) if not is_home[s] and s not in used
        ]
        for expert, slot in zip(moved, spare):
            rebuilt[expert].append(slot)
    for expert, slots in rebuilt.items():
        result[expert] = tuple(sorted(slots))
    return result


def _one_install(
    installed: _ExpertSlots, fitted: _ExpertSlots, is_home: list[bool], geometry: ExpertGeometry
) -> _ExpertSlots:
    """Choose the slots of ``fitted`` so that one install reaches it from ``installed``: existing
    replicas keep their slots and new replicas go to free slots."""
    kept, adds, _, free = _changes(
        installed, _keep_replica_slots(installed, fitted, is_home, geometry), geometry
    )
    for gpu, experts in adds.items():
        for expert, slot in zip(experts, free[gpu]):
            kept[expert].add(slot)
    return [tuple(sorted(slots)) for slots in kept]


# --- Install decisions ---


def evaluate_gains(
    load: np.ndarray,
    layout: np.ndarray,
    shares: np.ndarray,
    new_layout: np.ndarray,
    new_shares: np.ndarray,
    model: TimeModel,
    geometry: ExpertGeometry,
) -> np.ndarray:
    """The gain of ``new_layout`` at ``new_shares`` over ``layout`` at ``shares`` of each layer, in
    ms per decode step, for ``load``; an install applies a layout and its shares together."""
    if not len(load):
        return np.zeros(0)

    def time_us(layout: np.ndarray, shares: np.ndarray) -> np.ndarray:  # a replica adds route_us
        moe = model.layer_time_us(slot_rows(load, layout, shares), geometry.ep_size)
        return moe + model.route_us * ((layout >= 0).sum(1) > geometry.num_experts)

    return (time_us(layout, shares) - time_us(new_layout, new_shares)) / 1e3


def choose_layers_to_install(
    gains: dict[int, float], steps_left: float | None, copy_ms: dict[int, float]
) -> list[int]:
    """Layers to install, largest gain first, whose gain (ms per step) over the steps left repays
    their copy (``copy_ms`` per layer). The drift of the rows sets the interval between evaluations
    (:class:`EvaluationScheduler`), not the worth of an install."""
    if steps_left is None:
        return []
    order = sorted(gains.items(), key=lambda kv: (-kv[1], kv[0]))
    return [layer for layer, gain in order if gain * float(steps_left) > copy_ms[layer]]


def agreed_timings(
    per_rank: Sequence[Sequence[tuple[int, int, float]]],
) -> list[tuple[int, int, float]]:
    """``(group, layers, ms)`` of the install groups that every rank timed, at the slowest rank's ms.

    Raises ``ValueError`` when ranks disagree on a group's layers.
    """
    by_rank = [{int(g): (int(n), float(ms)) for g, n, ms in rows} for rows in per_rank]
    if not by_rank:
        return []
    common = sorted(set.intersection(*[set(groups) for groups in by_rank]))
    bad = [g for g in common if len({groups[g][0] for groups in by_rank}) != 1]
    if bad:
        raise ValueError(f"ranks disagree on the layers of install groups {bad[:8]}")
    return [(g, by_rank[0][g][0], max(groups[g][1] for groups in by_rank)) for g in common]


def fixed_install_ms(timings: Sequence[tuple[int, int, float]]) -> float | None:
    """Milliseconds of applying route tables that do not depend on the number of layers: the
    least-squares intercept of the ``(group, layers, ms)`` timings against their layer counts."""
    points = [(n, ms) for _, n, ms in timings]
    if len({n for n, _ in points}) < 2:
        return None
    mean_n = sum(n for n, _ in points) / len(points)
    mean_ms = sum(ms for _, ms in points) / len(points)
    slope = sum((n - mean_n) * (ms - mean_ms) for n, ms in points) / sum(
        (n - mean_n) ** 2 for n, _ in points
    )
    return max(0.0, mean_ms - slope * mean_n)


@dataclass(frozen=True)
class CostGrowth:
    """Growth ``rate`` (ms per call) of the layouts' cost over ``layers`` read in two windows."""

    rate: float | None
    layers: int
    interval: int


@dataclass(frozen=True)
class EvaluationPlan:
    """Calls from an evaluation's apply call to the next evaluation (``interval``) and to the open
    of the next statistics window (``open_after``)."""

    interval: int
    open_after: int


class EvaluationScheduler:
    """Chooses the number of calls between evaluations.

    An evaluation has a fixed cost, and between evaluations the installed layouts fit the rows a
    little worse every call; an interval of ``sqrt(2 * fixed cost / growth per call)`` calls
    minimizes their sum per call. The growth is read from two windows of the same layers: their
    time above the same rows spread evenly, under the current layouts. Growth below
    :meth:`min_resolvable_rate` is not counted.
    """

    def __init__(self, model: TimeModel, geometry: ExpertGeometry) -> None:
        self.model = model
        """The time model of the latest window (:meth:`TimeModel.at` its size)."""
        self._geometry = geometry
        self._previous_load: dict[int, np.ndarray] = {}
        self._previous_open: int | None = None
        self._last_positive_rate: float | None = None

    def read_cost(
        self,
        layers: Sequence[int],
        load: np.ndarray,
        layout: np.ndarray,
        shares: np.ndarray,
        open_call: int,
    ) -> CostGrowth | None:
        """Price this window's rows under ``layout`` and keep them for the next window."""
        if not layers:
            return None
        both = [i for i, layer in enumerate(layers) if layer in self._previous_load]
        growth = CostGrowth(None, 0, 0)
        if both and self._previous_open is not None and open_call > self._previous_open:
            ep = self._geometry.ep_size
            previous = np.zeros_like(load)
            for i in both:
                previous[i] = self._previous_load[layers[i]]
            balanced = self.model.balanced_time_us(load, self._geometry)
            now = self.model.layer_time_us(slot_rows(load, layout, shares), ep) - balanced
            total = previous.sum(1)
            previous *= np.where(total > 0.0, load.sum(1) / np.where(total > 0.0, total, 1.0), 0.0)[
                :, None
            ]
            old = self.model.layer_time_us(slot_rows(previous, layout, shares), ep) - balanced
            cost_now = sum(float(now[i]) for i in both) / 1e3
            cost_prev = sum(float(old[i]) for i in both) / 1e3
            interval = int(open_call) - self._previous_open
            growth = CostGrowth((cost_now - cost_prev) / float(interval), len(both), interval)
        for i, layer in enumerate(layers):
            self._previous_load[layer] = load[i].copy()
        self._previous_open = int(open_call)
        return growth

    def min_resolvable_rate(self, layers: int, interval: int) -> float:
        """The smallest growth per call that a reading of ``layers`` layers over ``interval`` calls
        can show: one more row per layer over the interval, at the time of one row."""
        return layers * self.model.slope_us / 1e3 / interval

    def plan(
        self,
        call: int,
        gather_call: int,
        next_window_calls: int,
        stats_copy_ms: Sequence[float],
        install_timings: Sequence[tuple[int, int, float]],
        growth: CostGrowth | None,
        steps_left: float | None,
    ) -> EvaluationPlan:
        """The plan from the evaluation whose route tables apply at ``call`` to the next one.

        The window was gathered at ``gather_call`` and the next window takes ``next_window_calls``.
        ``stats_copy_ms`` (the statistics copies of past windows) and ``install_timings`` give the
        fixed cost of an evaluation, ``growth`` is this window's :meth:`read_cost` and
        ``steps_left`` the decode steps the running requests still have.
        """
        calls_since_gather = int(call) - int(gather_call)
        min_interval = 1 + int(next_window_calls) + calls_since_gather
        copies = [float(x) for x in stats_copy_ms]
        stats_copy = statistics.median(copies) if copies else None
        install_fixed = fixed_install_ms(install_timings)
        fixed = None if stats_copy is None else stats_copy + (install_fixed or 0.0)
        rate = None
        if growth is not None:
            if growth.rate is not None and growth.rate > self.min_resolvable_rate(
                growth.layers, growth.interval
            ):
                self._last_positive_rate = growth.rate
            rate = self._last_positive_rate
        best = None
        if fixed is not None and rate is not None:
            best = math.sqrt(2.0 * fixed / rate)
        interval = min_interval if best is None else max(min_interval, int(math.ceil(best)))
        if steps_left is not None:
            interval = min(interval, max(min_interval, int(math.ceil(steps_left))))
        open_after = max(1, interval - int(next_window_calls) - calls_since_gather)
        return EvaluationPlan(interval, open_after)


# --- Operands of the ops: the route op (``trtllm::adaptive_eplb_route_into``) maps every routed
# expert to one of its slots by the tables of build_route_tables and adds the routes to statistics
# laid out by statistics_buffers; the record op (``trtllm::adaptive_eplb_record``) adds routes alone;
# the share solver (``trtllm::adaptive_eplb_solve_shares``) splits the rows of replicated experts.

_UINT32_MAX = (1 << 32) - 1


def solve_shares(
    load: np.ndarray, layout: np.ndarray, geometry: ExpertGeometry, model: TimeModel | None = None
) -> np.ndarray:
    """``[num_slots]`` shares of ``layout`` that minimize the busiest GPU's rows of ``load``, or its
    time by ``model`` where that model's ``time_shares`` is set."""
    import torch

    load = np.ascontiguousarray(load, dtype=np.float64)
    slots = np.asarray(layout)
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("layout must hold integers")
    slots = torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int64))
    timed = model is not None and model.time_shares  # then its fields up to the tile rows
    costs = [float(getattr(model, f.name)) for f in fields(model)[:6]] if timed else None
    solve = torch.ops.trtllm.adaptive_eplb_solve_shares
    return solve(torch.from_numpy(load), slots, geometry.ep_size, costs).numpy()


class RouteTables(NamedTuple):
    """Route tables of one layout: copies (``counts``) and first table entry (``starts``) of every
    expert, the slot of every entry, and the Q32 cumulative shares ``cumulative_shares`` of replicated experts."""

    counts: list[int]
    starts: list[int]
    slots: list[int]
    cumulative_shares: list[int]

    def packed(self) -> list[int]:
        """``counts | starts | slots | cumulative_shares`` as int32 values."""
        cumulative_shares = [
            value - (1 << 32) if value >> 31 else value for value in self.cumulative_shares
        ]
        return self.counts + self.starts + self.slots + cumulative_shares


def build_route_tables(
    layout: Sequence[int], shares: Sequence[float] | None, num_experts: int, num_slots: int
) -> RouteTables:
    """Route tables of a ``[num_slots]`` layout and the ``shares`` of its slots; without
    ``shares``, every expert's rows split evenly over its copies."""
    layout = [int(e) for e in (layout.tolist() if hasattr(layout, "tolist") else layout)]
    shares = [1.0] * num_slots if shares is None else shares
    shares = [float(s) for s in (shares.tolist() if hasattr(shares, "tolist") else shares)]
    expert_slots = _expert_slots_of(layout, num_experts)
    counts = [len(copies) for copies in expert_slots]
    if 0 in counts:
        raise ValueError(f"expert {counts.index(0)} has no copy")
    starts = list(itertools.accumulate(counts, initial=0))[:-1]
    slots = [slot for copies in expert_slots for slot in copies] + [-1] * layout.count(-1)
    cumulative_shares = [_UINT32_MAX] * num_slots
    for copies, start in zip(expert_slots, starts):
        weights = [max(0.0, shares[slot]) for slot in copies]
        total = sum(weights)
        if total <= 0.0:
            weights, total = [1.0] * len(copies), float(len(copies))
        acc = 0.0
        bound = 0
        for k in range(len(copies) - 1):
            acc += weights[k]
            bound = min(max(int(acc / total * 4294967296.0 + 0.5), bound), _UINT32_MAX)
            cumulative_shares[start + k] = bound
    return RouteTables(counts, starts, slots, cumulative_shares)


def statistics_buffers(layers: int, num_experts: int, device: torch.device | str) -> dict:
    """Every layer's statistics in one contiguous ``storage`` of bytes, so that a window opens with
    one zeroing and closes with one copy: ``rows`` ``[L, E]`` float32 (the rows routed to every
    expert, exact below 2^24), then ``forwards`` ``[L, 1]`` int64 from the next 8-byte boundary."""
    import torch

    rows_bytes = layers * num_experts * 4
    forwards_at = -(-rows_bytes // 8) * 8
    storage = torch.zeros(forwards_at + layers * 8, dtype=torch.uint8, device=device)
    return dict(
        rows=storage[:rows_bytes].view(torch.float32).view(layers, num_experts),
        forwards=storage[forwards_at:].view(torch.int64).view(layers, 1),
        storage=storage,
    )


class ExpertStatistics(NamedTuple):
    """One layer's statistics in the ops' argument order, and the ``[1]`` flag that the ops read at
    every launch, so that a captured CUDA graph records statistics only while the flag is set."""

    rows: torch.Tensor
    forwards: torch.Tensor
    enabled: torch.Tensor

    @classmethod
    def of(cls, buffers: dict, layer: int, enabled: torch.Tensor) -> ExpertStatistics:
        """The statistics of layer ``layer`` in buffers made by :func:`statistics_buffers`."""
        return cls(buffers["rows"][layer], buffers["forwards"][layer], enabled)

    def record(
        self,
        token_selected_experts: torch.Tensor,
        first_call: bool = True,
        live_row_word: torch.Tensor | None = None,
    ) -> None:
        """Add ``[tokens, top_k]`` expert ids to the statistics, of the live rows of
        ``live_row_word`` only; only a forward's first call counts the forward."""
        import torch

        torch.ops.trtllm.adaptive_eplb_record(
            token_selected_experts, *self, first_call, live_row_word
        )
