# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement learned online from a server's own traffic: in warmup, or for the whole run.

``MoeLoadBalancerConfig.warmup_placement`` keeps the MoE layers static load-balancer layers with
one slot per expert and lets a controller re-place their experts a bounded number of times at
executor-step boundaries: per-layer expert counters run from ``start_step``, a round runs every
``interval_steps`` after it once ``min_rows`` new rows are counted, and the placement is frozen
after ``max_migrations`` rounds. At ``deadline_step`` a last round runs on whatever was counted
since the previous one, then the placement is frozen.

A round, on every EP rank at the same step:

1. the counters are summed over the EP group;
2. rank 0 re-plans every layer from its current placement by swapping slots between heavy and
   light ranks (``plan_layer_swaps``), on the counted picks (``planner="lpt"``) or on each
   expert's modelled MegaMoE cost (``planner="cost"``), and broadcasts it;
3. the moved slots' per-slot tensors go GPU to GPU over a few-CTA NCCL communicator, one group of
   whole permutation cycles at a time through bounded staging, and are written back in place;
4. the balancer's device table and any backend expert -> slot table are rewritten in place and
   every slot that received an expert is checksummed.

Nothing moves to host memory and no buffer is reallocated, so CUDA graphs captured before a round
stay valid. A placement only changes which EP rank computes an expert.

The counters see the live rows of real requests only (``set_live_rows``: no attention-DP dummies or
CUDA-graph padding); with ``min_pos`` only the generation rows of requests that have produced at
least that many tokens, since early decode positions route unlike the rest of a response. Rows whose
router logits are all zero never count.

A swap is taken only if it lowers the heavier of its two ranks by more than ``min_gain`` of the
mean rank, and a round moves at most ``max_moved_fraction`` of a layer's slots: successive rounds
converge towards a balanced placement, and a round on unchanged traffic moves almost nothing.
With ``min_round_gain`` a round whose plan lowers the planner's estimate (slowest rank over the
mean, averaged over layers) by fewer percentage points moves nothing. With ``decay < 1`` the
counters are scaled by it after every round, moved or not, so later rounds follow recent traffic.
With a large ``max_migrations``, no ``deadline_step`` and ``min_pos >= 2`` (a request's first
decode step never counts) the placement follows the served decode traffic for the whole run;
``count_every_steps`` then counts one forward in that many, to keep the per-step host work of the
row masks small.

Hot-expert copies: a layer with extra slots (``num_slots = experts + ep_size * x``, no assignments)
keeps copies of experts, never two copies of an expert on one rank. A round first moves its copies
towards ``target_copies`` (``plan_layer_copy_moves``, one slot of the budget each, a swap two): hot
experts get copies while that lowers the slowest rank, the other extra slots hold copies of the
coldest experts, so the copies follow the hot experts and cost little elsewhere. A copy carries all
of its expert's bytes, so any copy gives the same bits; the backend spreads an expert's picks over
its copies.

Row passes: a layer registers under the row space of its MoE input (``"main"``: the step's tokens).
A drafter whose MoE routes its own rows registers under its pass with ``rows_per_request`` (a DSpark
draft stage: its block rows per generation request), which ``set_live_rows`` bounds and masks like
the main pass. Such a layer is learned from the default contiguous placement, in the same rounds.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import time
import weakref
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

_LOG = "[MoePlace]"
# Rows per forward and pass that the min_pos row mask covers; later rows never count.
_MASK_ROWS = 16384
# Bytes of incoming slots a rank stages per transfer group (its outgoing slots take as much again).
_STAGING_BYTES = 256 << 20
# CTAs of the slot-exchange communicator: NCCL keeps P2P buffers per channel and peer outside torch.
_EXCHANGE_CTAS = 2

# Backends whose expert-indexed state is exactly the per-slot tensors below plus the balancer's
# table. A backend that keeps another expert -> slot table implements
# ``on_expert_placement_changed(global_assignments)``; any other backend is refused.
_VERIFIED_BACKENDS = frozenset({"CuteDslFusedMoE", "CutlassFusedMoE"})

# Per-slot parameters (leading dim = local slots) a migration moves: what the NVFP4 quantization
# methods register per expert, the per-expert activation constants, plus MegaMoE's packed weights.
# Empty placeholders are skipped.
_SLOT_PARAM_NAMES = (
    "w3_w1_weight",
    "w2_weight",
    "w3_w1_bias",
    "w2_bias",
    "w3_w1_weight_scale",
    "w2_weight_scale",
    "fc31_alpha",
    "fc2_alpha",
    "fc31_weight_scale_2",
    "fc2_weight_scale_2",
    "fc31_act_scale",
    "fake_input_scale",
    "mega_fc1_weight",
    "mega_fc1_weight_sf",
    "mega_fc2_weight",
    "mega_fc2_weight_sf",
    "fc1_norm_const",
    "act_alpha",
    "act_beta",
    "act_clamp",
)


@dataclasses.dataclass(frozen=True)
class WarmupPlacementSettings:
    """Schedule and planner of the warmup placement (``MoeWarmupPlacementConfig``)."""

    start_step: int
    interval_steps: int
    max_migrations: int
    min_rows: int = 0
    deadline_step: Optional[int] = None
    planner: str = "lpt"
    min_pos: int = 0
    decay: float = 1.0
    max_moved_fraction: float = 0.1
    min_gain: float = 0.01
    count_every_steps: int = 1
    min_round_gain: float = 0.0

    @classmethod
    def from_config(cls, config) -> Optional["WarmupPlacementSettings"]:
        return None if config is None else cls(**config.model_dump())


class RouteCounters(NamedTuple):
    """One layer's counters for a routing kernel that adds its picks itself (see ``count_routing``)."""

    hist: torch.Tensor  # int32 [num_experts]
    flag: torch.Tensor  # int32 [1]: 0 stops counting (frozen) without another kernel variant
    bound: torch.Tensor  # int32 [1]: rows of the forward at or past it are not live (-1: none)
    mask: Optional[torch.Tensor]  # int8 [_MASK_ROWS] with min_pos: rows that count
    row_offset: int  # the launch's first row in the forward


# ----------------------------------------------------------------------------------- schedule

NONE, CHECK, LAST = 0, 1, 2


def schedule_action(
    settings: WarmupPlacementSettings, step: int, migrations_done: int, next_check: int
) -> Tuple[int, int]:
    """(action, next_check) before the forward of ``step``: a pure function of its arguments, so
    every EP rank decides the same at the same step. Steps without a forward never reach the
    controller, so a boundary is taken at the first forward at or after it."""
    if settings.deadline_step is not None and step >= settings.deadline_step:
        return LAST, next_check
    if step >= next_check:
        k = (step - settings.start_step) // settings.interval_steps + 1
        return CHECK, settings.start_step + k * settings.interval_steps
    return NONE, next_check


# ----------------------------------------------------------------------------------- planners


def rank_loads(load: np.ndarray, slot_expert: Sequence[int], slots_per_rank: int) -> np.ndarray:
    se = np.asarray(slot_expert, dtype=np.int64)
    ranks = np.arange(se.size) // slots_per_rank
    return np.bincount(ranks, weights=load[se], minlength=se.size // slots_per_rank)


# planner="cost": a MegaMoE rank's arrival at the cross-rank tail tracks its active local experts
# (each costs a weight stream and a tile schedule) more than its rows. Least-squares slopes of the
# tail arrival over (active experts, routed picks) per rank, in us, fitted on DEP16 / DEP32 GB300.
_COST_COEFFS = {16: (1.95, 0.0139), 32: (1.15, 0.0166)}


def cost_coeffs(ep_size: int) -> Tuple[float, float]:
    """(us per active expert, us per routed pick) of the nearest fitted EP width (log scale)."""
    near = min(_COST_COEFFS, key=lambda w: (abs(math.log2(w) - math.log2(max(1, ep_size))), w))
    return _COST_COEFFS[near]


def expert_cost(load: np.ndarray, top_k: int, rows: float, a: float, b: float) -> np.ndarray:
    """Expected per-forward cost of each expert, ``a * P(active) + b * lambda``: ``lambda`` is its
    share of ``load`` x ``top_k`` x ``rows`` (expected picks per forward) and ``P(active) = 1 -
    exp(-lambda)`` (Poisson)."""
    load = np.asarray(load, dtype=np.float64)
    total = load.sum()
    if total <= 0 or rows <= 0:
        return np.zeros_like(load)
    lam = load / total * (top_k * rows)
    return a * -np.expm1(-lam) + b * lam


# ----------------------------------------------------------------------------------- transfers


def build_transfers(old: Sequence[int], new: Sequence[int]) -> List[Tuple[int, int]]:
    """(source slot, destination slot) for every slot whose expert changes; the source is where
    the destination's new expert sits now."""
    where = {e: s for s, e in enumerate(old)}
    if sorted(old) != sorted(new) or len(where) != len(old):
        raise ValueError(f"{_LOG} placements must be permutations of the same experts")
    return [(where[e], s) for s, e in enumerate(new) if old[s] != e]


def transfer_groups(
    transfers: Sequence[Tuple[int, int]], slots_per_rank: int, cap_slots: int
) -> List[List[Tuple[int, int]]]:
    """Split a layer's transfers (a slot permutation) into groups of whole cycles, in order of
    each cycle's lowest slot, so that no rank receives more than ``cap_slots`` slots in a group
    (a cycle over the cap is a group of its own). Cycles touch disjoint slots, so the groups apply
    one after another, each staging only its own incoming slots. Deterministic."""
    src_of = {d: s for s, d in transfers}
    seen, cycles = set(), []
    for d in sorted(src_of):
        if d in seen:
            continue
        cyc, x = [], d
        while x not in seen:
            seen.add(x)
            cyc.append((src_of[x], x))
            x = src_of[x]
        cycles.append(cyc)
    groups: List[List[Tuple[int, int]]] = []
    cur: List[Tuple[int, int]] = []
    cur_in: Dict[int, int] = {}
    for cyc in cycles:
        need: Dict[int, int] = {}
        for _, d in cyc:
            need[d // slots_per_rank] = need.get(d // slots_per_rank, 0) + 1
        if cur and any(cur_in.get(r, 0) + n > cap_slots for r, n in need.items()):
            groups.append(cur)
            cur, cur_in = [], {}
        cur.extend(cyc)
        for r, n in need.items():
            cur_in[r] = cur_in.get(r, 0) + n
    if cur:
        groups.append(cur)
    return groups


def is_replicated_cover(
    slot_expert: Sequence[int], num_experts: int, slots_per_rank: int, ep_size: int
) -> bool:
    """A placement with extra slots that holds every expert, never twice on one rank."""
    se = list(slot_expert)
    if len(se) != slots_per_rank * ep_size or len(se) <= num_experts:
        return False
    if set(se) != set(range(num_experts)):
        return False
    return all(
        len(set(se[r * slots_per_rank : (r + 1) * slots_per_rank])) == slots_per_rank
        for r in range(ep_size)
    )


def copy_transfers(
    old: Sequence[int], new: Sequence[int], slots_per_rank: int
) -> List[Tuple[int, int]]:
    """(source slot, destination slot) for every slot whose expert changes in a placement with
    copies: a slot holding the expert now, on the destination's rank when there is one, else copy
    ``destination % copies`` in slot order. Deterministic."""
    where: Dict[int, List[int]] = collections.defaultdict(list)
    for s, e in enumerate(old):
        where[e].append(s)
    out = []
    for d, e in enumerate(new):
        if old[d] == e:
            continue
        srcs = where[e]
        local = [s for s in srcs if s // slots_per_rank == d // slots_per_rank]
        out.append((local[0] if local else srcs[d % len(srcs)], d))
    return out


def _replica_inputs(load: np.ndarray, top_k: int, cost, row_space: int):
    """(per-expert expected load, weight of a copy's share) of a layer with copies: with the cost
    planner the expected picks per forward and the modelled cost, else the counted picks."""
    load = np.asarray(load, dtype=np.float64)
    if cost is None:
        return load, (lambda lam: np.asarray(lam, dtype=np.float64))
    a, b, fwd_rows = cost
    total, rows = load.sum(), fwd_rows[row_space]
    lam = load / total * (top_k * rows) if total > 0 and rows > 0 else np.zeros_like(load)
    return lam, (lambda x: a * -np.expm1(-np.asarray(x)) + b * np.asarray(x))


def _replica_excess(lam: np.ndarray, slot_expert: Sequence[int], slots_per_rank: int, weight):
    se = np.asarray(slot_expert, dtype=np.int64)
    copies = np.bincount(se, minlength=lam.size)
    w = weight(lam[se] / np.maximum(copies[se], 1))
    rl = np.bincount(
        np.arange(se.size) // slots_per_rank, weights=w, minlength=se.size // slots_per_rank
    )
    mean = rl.mean()
    return float((rl.max() - mean) / mean) if mean > 0 else 0.0


def target_copies(lam: np.ndarray, weight, ep_size: int, total_slots: int) -> np.ndarray:
    """Copies per expert of a layer with extra slots: the expert with the heaviest copy gets one more
    while that lowers max(mean rank load, heaviest copy); the slots left over hold second copies of
    the coldest experts, which are rarely active. Deterministic."""
    lam = np.asarray(lam, dtype=np.float64)
    copies = np.ones(lam.size, dtype=np.int64)
    per_copy = np.asarray(weight(lam), dtype=np.float64)
    total = float(per_copy.sum())
    bound = max(total / ep_size, float(per_copy.max()))
    for _ in range(total_slots - lam.size):
        h = int(np.argmax(np.where(copies < ep_size, per_copy, -np.inf)))
        split = float(weight(lam[h] / (copies[h] + 1)))
        new_total = total + split * (copies[h] + 1) - per_copy[h] * copies[h]
        kept = per_copy[h]
        per_copy[h] = split
        new_bound = max(new_total / ep_size, float(per_copy.max()))
        if new_bound >= bound:
            per_copy[h] = kept
            break
        copies[h] += 1
        total, bound = new_total, new_bound
    left = total_slots - int(copies.sum())
    for e in np.lexsort((np.arange(lam.size), lam)):
        if left == 0:
            break
        if copies[e] < ep_size:
            copies[e] += 1
            left -= 1
    return copies


def plan_layer_copy_moves(
    lam: np.ndarray,
    slot_expert: Sequence[int],
    slots_per_rank: int,
    weight,
    max_moves: int,
) -> Tuple[List[int], int]:
    """Churn-bounded re-choice of the copies of a layer with extra slots -> (slot -> expert list,
    moves). A move turns a slot of an expert with more copies than ``target_copies`` gives it into a
    copy of an expert with fewer, on a rank that does not hold it (a slot's load: ``weight(lam_e /
    copies_e)``): the move with the lowest slowest rank, then the lowest sum of squared rank loads,
    taken only if the slowest rank does not get slower. At most ``max_moves`` moves. Deterministic."""
    se = np.asarray(slot_expert, dtype=np.int64).copy()
    lam = np.asarray(lam, dtype=np.float64)
    ranks = se.size // slots_per_rank
    rank_of = np.arange(se.size) // slots_per_rank
    target = target_copies(lam, weight, ranks, se.size)
    copies = np.bincount(se, minlength=lam.size)
    holds = np.zeros((ranks, lam.size), dtype=bool)
    holds[rank_of, se] = True
    per_copy = np.asarray(weight(lam / np.maximum(copies, 1)), dtype=np.float64)
    loads = np.bincount(rank_of, weights=per_copy[se], minlength=ranks)
    rank_ids = np.arange(ranks)
    moves = 0
    while moves < max_moves:
        donors = np.flatnonzero(copies[se] > target[se])
        rec = np.flatnonzero(copies < target)
        if donors.size == 0 or rec.size == 0:
            break
        dex, drank = se[donors], rank_of[donors]
        rec_new = np.asarray(weight(lam[rec] / (copies[rec] + 1)), dtype=np.float64)
        don_new = np.asarray(weight(lam[dex] / (copies[dex] - 1)), dtype=np.float64)
        # Rank load changes [receiver, donor slot, rank]: the receiver's copies get lighter, the
        # donor expert's other copies heavier, and the donor slot's rank swaps one copy for another.
        own = rank_ids[None, :] == drank[:, None]
        rec_holds = holds[:, rec].T
        don_holds = holds[:, dex].T & ~own
        delta = (rec_holds * (rec_new - per_copy[rec])[:, None])[:, None, :] + (
            don_holds * (don_new - per_copy[dex])[:, None]
        )[None, :, :]
        delta += own[None, :, :] * (rec_new[:, None] - per_copy[dex][None, :])[:, :, None]
        new_loads = loads + delta
        peak = new_loads.max(axis=2)
        ok = ~holds[drank[None, :], rec[:, None]] & (peak <= loads.max())
        if not ok.any():
            break
        sq = (new_loads**2).sum(axis=2)
        flat = np.lexsort((sq[ok], peak[ok]))[0]
        i, j = (int(x[flat]) for x in np.nonzero(ok))
        s, e, d, r = int(donors[j]), int(rec[i]), int(dex[j]), int(drank[j])
        se[s] = e
        copies[d] -= 1
        copies[e] += 1
        holds[r, d], holds[r, e] = False, True
        per_copy[d], per_copy[e] = don_new[j], rec_new[i]
        loads = np.bincount(rank_of, weights=per_copy[se], minlength=ranks)
        moves += 1
    return se.tolist(), moves


def plan_layer_swaps(
    slot_weight: np.ndarray,
    slot_expert: Sequence[int],
    slots_per_rank: int,
    max_swaps: int,
    min_gain: float,
) -> Tuple[List[int], int]:
    """Churn-bounded re-placement -> (slot -> expert list, swaps). Each swap exchanges a slot of one
    of the heaviest quarter of the ranks with a slot of one of the lighter half, the pair that most
    lowers the sum of squared rank loads (a rank's load: the sum of its slots' ``slot_weight``),
    taken only if it lowers the heavier rank by more than ``min_gain`` x the mean rank load. Never
    puts two copies of an expert on one rank; at most ``max_swaps`` swaps (2 x ``max_swaps`` slots
    change). Repeated rounds converge towards a balanced placement. Deterministic."""
    experts = np.asarray(slot_expert, dtype=np.int64).reshape(-1, slots_per_rank).copy()
    weight = np.asarray(slot_weight, dtype=np.float64).reshape(-1, slots_per_rank).copy()
    ranks = experts.shape[0]
    copies = np.bincount(experts.reshape(-1)).max(initial=1) > 1
    loads = weight.sum(axis=1)
    mean = float(loads.mean())
    swaps = 0
    while swaps < max_swaps and mean > 0:
        order = np.argsort(-loads, kind="stable")
        heavy = order[: max(1, ranks // 4)]
        light = order[ranks - max(1, ranks // 2) :]
        # A swap relieves the heavier rank by at most half their load gap: closer pairs (less a
        # rounding margin) cannot pass min_gain.
        gap = loads[heavy][:, None] - loads[light][None, :]
        need = 2.0 * min_gain * mean - 1e-9 * loads[heavy][:, None]
        pair_hi, pair_lo = np.nonzero((gap > 0) & (gap >= need))
        if pair_hi.size == 0:
            break
        his, los = heavy[pair_hi], light[pair_lo]
        # shift[p, i, j]: load moving from rank his[p] to rank los[p] by swapping slots i and j; the
        # first best pair in (heavy, light, i, j) order wins.
        shift = weight[his][:, :, None] - weight[los][:, None, :]
        load_hi = loads[his][:, None, None]
        load_lo = loads[los][:, None, None]
        drop = 2.0 * shift * ((load_hi - load_lo) - shift)
        relief = load_hi - np.maximum(load_hi - shift, load_lo + shift)
        ok = relief > min_gain * mean
        if copies:
            # A copy may not join a rank that already holds its expert.
            same = experts[his][:, :, None] == experts[los][:, None, :]
            ok &= ~same.any(axis=2)[:, :, None] & ~same.any(axis=1)[:, None, :]
        drop = np.where(ok, drop, -np.inf)
        p, i, j = np.unravel_index(int(np.argmax(drop)), drop.shape)
        if not np.isfinite(drop[p, i, j]):
            break
        hi, lo, i, j = int(his[p]), int(los[p]), int(i), int(j)
        experts[hi, i], experts[lo, j] = experts[lo, j], experts[hi, i]
        weight[hi, i], weight[lo, j] = weight[lo, j], weight[hi, i]
        loads[hi] = weight[hi].sum()
        loads[lo] = weight[lo].sum()
        swaps += 1
    return experts.reshape(-1).tolist(), swaps


# ----------------------------------------------------------------------------------- counting

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _count_kernel(
        ids_ptr,
        logits_ptr,
        hist_ptr,
        flag_ptr,
        bound_ptr,
        mask_ptr,
        num_rows,
        row_offset,
        logit_stride,
        mask_len,
        TOP_K: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_E: tl.constexpr,
        ROWS: tl.constexpr,
        ROW_MASK: tl.constexpr,
    ):
        bound = tl.load(bound_ptr)
        limit = tl.where(bound >= 0, tl.minimum(bound - row_offset, num_rows), num_rows)
        limit = tl.where(tl.load(flag_ptr) != 0, limit, 0)
        row0 = tl.program_id(0) * ROWS
        offs_e = tl.arange(0, BLOCK_E)
        for i in tl.static_range(ROWS):
            row = row0 + i
            if row < limit:
                live = True
                if ROW_MASK:
                    g = row + row_offset
                    live = (g < mask_len) & (tl.load(mask_ptr + g, mask=g < mask_len, other=0) != 0)
                lg = tl.load(
                    logits_ptr + row * logit_stride + offs_e,
                    mask=offs_e < NUM_EXPERTS,
                    other=0.0,
                )
                live = live & (tl.max(tl.abs(lg.to(tl.float32)), axis=0) > 0.0)
                if live:
                    for k in tl.static_range(TOP_K):
                        e = tl.load(ids_ptr + row * TOP_K + k)
                        if (e >= 0) & (e < NUM_EXPERTS):
                            tl.atomic_add(hist_ptr + e, 1)

except ImportError:  # pragma: no cover - CPU-only environments
    triton = None


def count_routing(
    expert_ids: torch.Tensor, counters: RouteCounters, router_logits: torch.Tensor
) -> None:
    """Add the expert picks of the live rows of ``expert_ids`` [rows, top_k] to ``counters.hist``.

    Row ``i`` (row ``counters.row_offset + i`` of the forward) counts while the flag is set, the row
    is below the live-row bound, its mask byte is set (with a mask) and its logits are not all zero.
    One launch; inside a CUDA graph it exits after one load once the flag is cleared."""
    rows, top_k = expert_ids.shape
    if rows == 0:
        return
    num_experts = counters.hist.numel()
    ids = expert_ids
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    mask = counters.mask
    rows_per_prog = 16
    _count_kernel[(triton.cdiv(rows, rows_per_prog),)](
        ids,
        router_logits,
        counters.hist,
        counters.flag,
        counters.bound,
        mask if mask is not None else ids,
        rows,
        int(counters.row_offset),
        router_logits.stride(0),
        mask.numel() if mask is not None else 0,
        TOP_K=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_E=triton.next_power_of_2(num_experts),
        ROWS=rows_per_prog,
        ROW_MASK=mask is not None,
    )


# ----------------------------------------------------------------------------------- controller


@dataclasses.dataclass
class _Layer:
    index: int
    num_experts: int
    top_k: int
    slots_per_rank: int
    moe: "weakref.ReferenceType"
    balancer: object  # SingleLayerMoeLoadBalancer
    assign: List[int]  # global slot -> expert
    row_space: int = 0
    slot_tensors: Optional[List[torch.Tensor]] = None
    slot_bytes: int = 0
    replicated: bool = False  # extra slots: an expert may hold several slots, never two on a rank
    movable: Optional[bool] = None  # every EP rank can move its per-slot state (None: unchecked)


class WarmupPlacementController:
    """Counters, schedule and migration rounds of the static balancer layers of one instance.

    ``MoeLoadBalancer`` creates it; layers register from ``MoE._init_load_balancer``; ``finalize``
    runs with the balancer's; ``on_step`` runs before every executor forward on every rank.

    Collectives of a round run in the same order on every rank: counter allreduce, live-row
    allreduce (``planner="cost"``), plan broadcast, prepare vote, checksum allreduce (first round),
    slot transfers, failure count. A rank-local error before the vote skips the round everywhere;
    after the vote errors are fatal."""

    def __init__(
        self,
        settings: WarmupPlacementSettings,
        ep_rank: int,
        ep_size: int,
    ):
        self.settings = settings
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.layers: List[_Layer] = []
        self.migrations_done = 0
        self.frozen = False
        self._finalized = False
        self._hist: Optional[torch.Tensor] = None
        self._flag: Optional[torch.Tensor] = None
        self._bounds: List[torch.Tensor] = []  # per row pass: int32 [1] live-row bound
        self._row_mask: Optional[torch.Tensor] = None
        self._mask_ring: List[Tuple[torch.Tensor, torch.cuda.Event]] = []
        self._mask_slot = 0
        self._mask_overflow_logged = False
        self._step = -1
        self._comm = None  # MPI communicator over the EP group, comm rank == EP rank
        self._group = None  # few-CTA NCCL group over the EP group, for the slot transfers
        self._staging: Optional[torch.Tensor] = None
        self._expert_checksums: Optional[np.ndarray] = None
        self._counting_since: Optional[int] = None
        self._passes: Dict[str, int] = {"main": 0}
        self._rows_per_request: Dict[int, Optional[int]] = {0: None}  # other passes, per request
        self._live_rows: Dict[int, collections.deque] = {0: collections.deque(maxlen=32)}
        self._rows_at_last_round = 0
        self._next_check = settings.start_step + settings.interval_steps
        logger.info(f"{_LOG} enabled: {settings}")

    # ------------------------------------------------------------------ registration
    def register_layer(
        self,
        moe,
        balancer,
        num_experts: int,
        top_k: int,
        slots_per_rank: int,
        initial_assignments: Sequence[int],
        row_space: str = "main",
        rows_per_request: Optional[int] = None,
    ) -> Optional[int]:
        """Track a static balancer layer; returns its counter index, or None for a layer whose
        assignment neither holds every expert once nor covers every expert with extra slots, which
        keeps its configured placement. ``row_space`` names the row space of the layer's MoE input
        (``"main"``: the step's tokens); another pass gives its MoE rows per generation request in
        ``rows_per_request`` and starts from the default placement."""
        if self._finalized:
            raise RuntimeError(f"{_LOG} layers must register before the model is finalized")
        one_copy = slots_per_rank * self.ep_size == num_experts and sorted(
            initial_assignments
        ) == list(range(num_experts))
        replicated = not one_copy and is_replicated_cover(
            initial_assignments, num_experts, slots_per_rank, self.ep_size
        )
        if not one_copy and not replicated:
            logger.info(
                f"{_LOG} layer with {num_experts} experts on {slots_per_rank} x {self.ep_size} "
                f"slots keeps its configured placement"
            )
            return None
        if row_space not in self._passes:
            pid = len(self._passes)
            self._passes[row_space] = pid
            self._rows_per_request[pid] = None if row_space == "main" else int(rows_per_request)
            self._live_rows[pid] = collections.deque(maxlen=32)
        layer = _Layer(
            len(self.layers),
            num_experts,
            top_k,
            slots_per_rank,
            weakref.ref(moe),
            balancer,
            list(initial_assignments),
            self._passes[row_space],
            replicated=replicated,
        )
        self.layers.append(layer)
        return layer.index

    def finalize(self) -> None:
        """Allocate the counters and communicators once every layer registered (after the model
        moved to the GPU). Collective over the world: every rank runs it at model load."""
        if self._finalized:
            return
        self._finalized = True
        if not self.layers:
            logger.warning(f"{_LOG} no static one-copy balancer layer registered; nothing to place")
            self.frozen = True
            return
        # Rows padded to 16 bytes, so a kernel may assume aligned counter rows.
        emax = -(-max(layer.num_experts for layer in self.layers) // 4) * 4
        dev = torch.device("cuda", torch.cuda.current_device())
        # Normal (not inference) tensors, so forwards in and outside inference mode may update them.
        with torch.inference_mode(False):
            self._hist = torch.zeros((len(self.layers), emax), dtype=torch.int32, device=dev)
            self._flag = torch.zeros(1, dtype=torch.int32, device=dev)
            self._bounds = [
                torch.full((1,), -1, dtype=torch.int32, device=dev) for _ in self._passes
            ]
            if self.settings.min_pos > 0:
                shape = (len(self._passes), _MASK_ROWS)
                self._row_mask = torch.zeros(shape, dtype=torch.int8, device=dev)
                self._mask_ring = [
                    (
                        maybe_pin_memory(torch.zeros(shape, dtype=torch.int8, device="cpu")),
                        torch.cuda.Event(),
                    )
                    for _ in range(4)
                ]
        self._comm = self._make_ep_comm()
        self._group = self._make_exchange_group()
        s = self.settings
        deadline = "" if s.deadline_step is None else f", a last round at step {s.deadline_step}"
        logger.info(
            f"{_LOG} {len(self.layers)} layers start at their configured placement; they count "
            f"live rows (min_pos {s.min_pos}) from step {s.start_step} and are re-placed every "
            f"{s.interval_steps} steps once {s.min_rows} new rows are counted, at most "
            f"{s.max_migrations} time(s){deadline}"
        )

    def _make_ep_comm(self):
        from tensorrt_llm._utils import mpi_comm

        # The world is the EP group (``_make_exchange_group`` requires pure EP).
        return mpi_comm()

    def _make_exchange_group(self):
        import torch.distributed as dist

        from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

        ensure_cutedsl_torch_distributed(self.ep_size)
        if dist.get_world_size() != self.ep_size:
            raise ValueError(
                f"{_LOG} needs pure expert parallelism (world {dist.get_world_size()} == EP "
                f"{self.ep_size}) for its slot exchange"
            )
        options = dist.ProcessGroupNCCL.Options()
        options.config.min_ctas = 1
        options.config.max_ctas = _EXCHANGE_CTAS
        group = dist.new_group(
            list(range(self.ep_size)), backend="nccl", pg_options=options, sort_ranks=False
        )
        # The first NCCL call of the group involves every rank; later transfers pair only some.
        dist.all_reduce(torch.zeros(1, device=torch.cuda.current_device()), group=group)
        return group

    # ------------------------------------------------------------------ hot-path accessors
    def _counting(self) -> bool:
        return self._hist is not None and not self.frozen

    def route_counters(self, index: int, row_offset: int) -> Optional[RouteCounters]:
        """Layer ``index``'s counters for a routing kernel that counts its own picks, for the whole
        run once finalized (the flag stops it at the freeze, so it keeps one compiled variant)."""
        if self._hist is None:
            return None
        layer = self.layers[index]
        p = layer.row_space
        return RouteCounters(
            self._hist[index, : layer.num_experts],
            self._flag,
            self._bounds[p],
            None if self._row_mask is None else self._row_mask[p],
            row_offset,
        )

    def counters(self, index: int, row_offset: int) -> Optional[RouteCounters]:
        """The same while counting, else None: for a separate count launch that is then skipped."""
        return self.route_counters(index, row_offset) if self._counting() else None

    @property
    def active(self) -> bool:
        """Whether the per-step live-row inputs are still needed."""
        return self._finalized and self._counting()

    def set_live_rows(self, requests, draft_len: int) -> None:
        """Every pass's live-row bound and, with ``min_pos``, row mask for the next forward. Main
        pass: context rows, then ``1 + draft_len`` rows per generation request (``real_rows_of``'s
        order). Another pass: its ``rows_per_request`` per live generation request, in the same
        order; a forward with live context requests does not count there. With ``min_pos`` context
        rows and the rows of generation requests with fewer than ``min_pos`` generated tokens are
        masked out. The masks go up through a pinned ring on the current stream."""
        if not self._counting():
            return
        if self._step % self.settings.count_every_steps:
            for bound in self._bounds:
                bound.fill_(0)
            return
        gens = [req for req in requests.generation_requests if not req.is_dummy]
        has_ctx = any(not req.is_dummy for req in requests.context_requests)
        live = {0: real_rows_of(requests, draft_len)}
        for pid, rpr in self._rows_per_request.items():
            if pid:
                live[pid] = 0 if has_ctx else len(gens) * rpr
        for pid, rows in live.items():
            if self._counting_since is not None:
                self._live_rows[pid].append(rows)
            self._bounds[pid].fill_(rows)
        if self._row_mask is None:
            return
        host, event = self._mask_ring[self._mask_slot]
        event.synchronize()  # this pinned slot's previous upload has finished
        mall = host.numpy()
        mall[:] = 0
        m = mall[0]
        n = m.size
        off = sum(_context_rows(req) for req in requests.context_requests if not req.is_dummy)
        per = 1 + int(draft_len or 0)
        counted = np.fromiter(
            (_generated_tokens(req) >= self.settings.min_pos for req in gens),
            dtype=bool,
            count=len(gens),
        )
        end = off + len(gens) * per
        if off < n:
            m[off : min(end, n)] = np.repeat(counted, per)[: min(end, n) - off]
        if not has_ctx:
            for pid, rpr in self._rows_per_request.items():
                if pid:
                    k = min(len(gens) * rpr, n)
                    mall[pid, :k] = np.repeat(counted, rpr)[:k]
        if end > n and not self._mask_overflow_logged:
            self._mask_overflow_logged = True
            logger.warning(f"{_LOG} {end} live rows > {n} masked rows; later rows do not count")
        self._row_mask.copy_(host, non_blocking=True)
        event.record()
        self._mask_slot = (self._mask_slot + 1) % len(self._mask_ring)

    # ------------------------------------------------------------------ schedule
    def on_step(self, step: int) -> None:
        self._step = step
        if not self._finalized or self.frozen:
            return
        if self._counting_since is None and step >= self.settings.start_step:
            self._flag.fill_(1)
            self._counting_since = step
            logger.info(f"{_LOG} counters on at step {step}")
        action, self._next_check = schedule_action(
            self.settings, step, self.migrations_done, self._next_check
        )
        if action != NONE:
            # Weights made in inference mode can only be updated in place inside it.
            with torch.inference_mode():
                self._round(step, last=action == LAST)
            if action == LAST and not self.frozen:
                self._freeze(step)

    # ------------------------------------------------------------------ one round
    def _round(self, step: int, last: bool = False) -> None:
        """One round; ``last`` (the deadline) runs on any rows counted since the previous one."""
        from mpi4py import MPI

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        local = self._hist.cpu().numpy().astype(np.int64)
        glob = np.empty_like(local)
        self._comm.Allreduce(local, glob, op=MPI.SUM)
        first = self.layers[0]
        rows = int(glob[0, : first.num_experts].sum()) // max(1, first.top_k)
        fresh = rows - self._rows_at_last_round
        if last and fresh == 0:
            logger.warning(
                f"{_LOG} deadline step {step}: no live row past min_pos {self.settings.min_pos} "
                f"was counted since the last of {self.migrations_done} round(s); the placement "
                f"stays as it is"
            )
            return
        if not last and fresh < self.settings.min_rows:
            logger.info(
                f"{_LOG} step {step}: {fresh} live rows past min_pos {self.settings.min_pos} "
                f"counted since the last round (< {self.settings.min_rows}); deferred to the next "
                f"boundary"
            )
            return
        self._check_movable(step)
        cost = self._cost_inputs(step, glob) if self.settings.planner == "cost" else None
        plan, note = None, ""
        if self.ep_rank == 0:
            plan, before, after = [], 0.0, 0.0
            for layer in self.layers:
                if not layer.movable:
                    plan.append(None)
                    continue
                load = glob[layer.index, : layer.num_experts].astype(np.float64)
                spr = layer.slots_per_rank
                if layer.replicated:
                    lam, weight = _replica_inputs(load, layer.top_k, cost, layer.row_space)
                    # Copy moves take one slot of the round's budget each, swaps two.
                    budget = 2 * self._max_swaps(layer)
                    chosen, moves = plan_layer_copy_moves(lam, layer.assign, spr, weight, budget)
                    copies = np.bincount(chosen, minlength=layer.num_experts)
                    slot_weight = weight(lam[chosen] / copies[chosen])
                    new, _ = plan_layer_swaps(
                        slot_weight,
                        chosen,
                        spr,
                        (budget - moves) // 2,
                        self.settings.min_gain,
                    )
                    plan.append(None if new == layer.assign else new)
                    before += _replica_excess(lam, layer.assign, spr, weight)
                    after += _replica_excess(lam, new, spr, weight)
                    continue
                if cost is not None:
                    a, b, fwd_rows = cost
                    load = expert_cost(load, layer.top_k, fwd_rows[layer.row_space], a, b)
                new, _ = plan_layer_swaps(
                    load[layer.assign],
                    layer.assign,
                    spr,
                    self._max_swaps(layer),
                    self.settings.min_gain,
                )
                plan.append(None if new == layer.assign else new)
                before += _excess(load, layer.assign, spr)
                after += _excess(load, new, spr)
            n = max(1, sum(bool(layer.movable) for layer in self.layers))
            note = (
                ", planner swaps: slowest rank over the mean "
                f"{100 * before / n:.1f}% -> {100 * after / n:.1f}%"
            )
            min_round_gain = self.settings.min_round_gain
            if min_round_gain > 0 and 100 * (before - after) / n < min_round_gain:
                plan = None
        plan = self._comm.bcast(plan, root=0)
        if plan is None:
            self._decay_counters(rows, glob)
            logger.info(
                f"{_LOG} step {step}: the plan gains less than {self.settings.min_round_gain} "
                f"points; the placement stays, {(time.perf_counter() - t0) * 1e3:.0f} ms{note}"
            )
            return
        if all(new is None for new in plan):
            # Nothing moves: the round only advances its counters, on every rank alike.
            applied = (0, 0)
        else:
            applied = self.apply_plan(plan, step)
        if applied is None:
            return  # skipped on every rank: the next boundary or the deadline tries again
        moved, nbytes = applied
        self._decay_counters(rows, glob)
        self.migrations_done += 1
        logger.info(
            f"{_LOG} round {self.migrations_done}/{self.settings.max_migrations} at step {step}"
            f"{' (deadline)' if last else ''}: rows {rows}, layers changed "
            f"{sum(p is not None for p in plan)}, slots received {moved} on EP rank "
            f"{self.ep_rank} ({nbytes / 2**20:.0f} MiB), "
            f"{(time.perf_counter() - t0) * 1e3:.0f} ms{note}"
        )
        if self.migrations_done >= self.settings.max_migrations:
            self._freeze(step)

    def _decay_counters(self, rows: int, glob: np.ndarray) -> None:
        """Every counter keeps ``decay`` of its value, so later rounds follow recent traffic; the
        next round's fresh rows count from here."""
        first = self.layers[0]
        self._rows_at_last_round = rows
        if self.settings.decay < 1.0:
            self._hist.copy_(torch.floor(self._hist.double() * self.settings.decay).to(torch.int32))
            kept = np.floor(glob[0, : first.num_experts] * self.settings.decay).sum()
            self._rows_at_last_round = int(kept) // max(1, first.top_k)

    def _max_swaps(self, layer: _Layer) -> int:
        return int(self.settings.max_moved_fraction * len(layer.assign)) // 2

    def _cost_inputs(self, step: int, glob: np.ndarray) -> Tuple[float, float, Dict[int, float]]:
        """planner="cost": (a, b, rows per forward over the EP group for each row pass): the mean
        live rows of each rank's last counted forwards, summed over the ranks; without any, the
        counted rows per counted step. Collective on the EP group."""
        from mpi4py import MPI

        a, b = cost_coeffs(self.ep_size)
        pids = sorted(self._live_rows)
        local = np.array(
            [float(np.mean(self._live_rows[p])) if self._live_rows[p] else 0.0 for p in pids]
        )
        total = np.empty_like(local)
        self._comm.Allreduce(local, total, op=MPI.SUM)
        steps = max(1, step - self._counting_since)
        rows = {}
        for i, pid in enumerate(pids):
            fwd = float(total[i])
            if fwd <= 0:
                first = next((layer for layer in self.layers if layer.row_space == pid), None)
                counted = 0
                if first is not None:
                    counted = int(glob[first.index, : first.num_experts].sum()) // first.top_k
                fwd = counted / steps
            rows[pid] = fwd
        if self.ep_rank == 0:
            logger.info(
                f"{_LOG} step {step}: cost planner a {a:g} us/expert, b {b:g} us/pick, rows per "
                f"forward over EP{self.ep_size}: {rows}"
            )
        return a, b, rows

    def _check_movable(self, step: int) -> None:
        """Once, collectively: a layer whose per-slot state some EP rank cannot move keeps its
        placement; the other layers are still learned."""
        from mpi4py import MPI

        if all(layer.movable is not None for layer in self.layers):
            return
        local = np.ones(len(self.layers), dtype=np.int32)
        reasons = {}
        for layer in self.layers:
            try:
                if layer.slot_tensors is None:
                    layer.slot_tensors, layer.slot_bytes = _resolve_slot_tensors(layer)
                _check_backend(layer)
            except (RuntimeError, ValueError) as e:
                local[layer.index] = 0
                reasons[layer.index] = e
        glob = np.empty_like(local)
        self._comm.Allreduce(local, glob, op=MPI.MIN)
        for layer in self.layers:
            layer.movable = bool(glob[layer.index])
            if not layer.movable:
                why = reasons.get(layer.index, "another EP rank cannot move its per-slot state")
                logger.error(f"{_LOG} step {step}: layer {layer.index} keeps its placement: {why}")

    def apply_plan(
        self, plan: Sequence[Optional[List[int]]], step: int
    ) -> Optional[Tuple[int, int]]:
        """Collective on the EP group: move every layer to ``plan[layer]`` (its new slot -> expert
        list, None to keep it), its per-slot tensors GPU to GPU in place, and rewrite the placement
        tables. Every rank passes the same plan of movable layers. Returns (slots received here,
        bytes received), or None when a rank fails to prepare and the plan is skipped everywhere."""
        from mpi4py import MPI

        if any(new is not None and not layer.movable for layer, new in zip(self.layers, plan)):
            raise ValueError(f"{_LOG} the plan moves a layer that cannot move")
        ok = 1
        try:
            self._prepare_local(plan)
        except (RuntimeError, ValueError) as e:  # incl. torch.cuda.OutOfMemoryError for staging
            logger.error(f"{_LOG} step {step}: prepare failed on EP rank {self.ep_rank}: {e!r}")
            self._staging = None
            ok = 0
        if self._comm.allreduce(ok, op=MPI.MIN) == 0:
            logger.error(f"{_LOG} step {step}: plan skipped on every rank")
            self._staging = None
            return None
        if self._expert_checksums is None:
            self._expert_checksums = self._gather_checksums()
        received = self._migrate(plan)
        torch.cuda.synchronize()
        self._staging = None
        bad = self._verify(received)
        if self._comm.allreduce(bad, op=MPI.SUM):
            raise RuntimeError(
                f"{_LOG} step {step}: slots on EP rank {self.ep_rank} hold the wrong expert after "
                f"migration ({bad} here)"
            )
        nbytes = sum(len(r) * layer.slot_bytes for r, layer in zip(received, self.layers))
        return sum(len(r) for r in received), nbytes

    def _freeze(self, step: int) -> None:
        self._flag.fill_(0)
        for bound in self._bounds:
            bound.fill_(-1)
        torch.cuda.synchronize()
        self.frozen = True
        logger.info(f"{_LOG} frozen at step {step} after {self.migrations_done} round(s)")

    # ------------------------------------------------------------------ migration
    def _layer_groups(self, layer: _Layer, new: List[int]) -> List[List[Tuple[int, int]]]:
        if layer.replicated:
            # One group: every source is read before any slot of the layer is written.
            transfers = copy_transfers(layer.assign, new, layer.slots_per_rank)
            return [transfers] if transfers else []
        transfers = build_transfers(layer.assign, new)
        cap = max(1, _STAGING_BYTES // max(1, layer.slot_bytes))
        return transfer_groups(transfers, layer.slots_per_rank, cap)

    def _prepare_local(self, plan) -> None:
        need = 0
        for layer, new in zip(self.layers, plan):
            if new is None:
                continue
            for group in self._layer_groups(layer, new):
                incoming = sum(1 for _, d in group if d // layer.slots_per_rank == self.ep_rank)
                need = max(need, incoming * layer.slot_bytes)
        # Allocated before the vote, so a rank without the memory skips the round on every rank.
        self._staging = torch.empty(need, dtype=torch.uint8, device="cuda") if need else None

    def _migrate(self, plan) -> List[List[int]]:
        """Per layer, the local slots that received an expert."""
        received = []
        for layer, new in zip(self.layers, plan):
            if new is None:
                received.append([])
                continue
            start = self.ep_rank * layer.slots_per_rank
            transfers = (
                copy_transfers(layer.assign, new, layer.slots_per_rank)
                if layer.replicated
                else build_transfers(layer.assign, new)
            )
            received.append(
                [d - start for _, d in transfers if start <= d < start + layer.slots_per_rank]
            )
            for group in self._layer_groups(layer, new):
                self._migrate_group(layer, group)
            self._commit_tables(layer, new)
        return received

    def _migrate_group(self, layer: _Layer, transfers: List[Tuple[int, int]]) -> None:
        """Move one group of whole cycles: every incoming slot of this rank is staged before any
        slot of the group is overwritten, so the group's sources are intact while it moves."""
        import torch.distributed as dist

        spr, me, nbytes = layer.slots_per_rank, self.ep_rank, layer.slot_bytes
        incoming = sorted([(s, d) for s, d in transfers if d // spr == me], key=lambda x: x[1])
        row = {d: i for i, (_, d) in enumerate(incoming)}
        staging = self._staging
        ops, sends = [], []
        # Both sides of a pair post its messages in destination-slot order, so they match.
        for s, d in sorted(transfers, key=lambda x: x[1]):
            src, dst = s // spr, d // spr
            if src == me and dst == me:
                _pack_slot(layer, s - me * spr, _row(staging, row[d], nbytes))
            elif src == me:
                buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                _pack_slot(layer, s - me * spr, buf)
                sends.append(buf)
                ops.append(dist.P2POp(dist.isend, buf, dst, self._group))
            elif dst == me:
                ops.append(dist.P2POp(dist.irecv, _row(staging, row[d], nbytes), src, self._group))
        if ops:
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        for _, d in incoming:
            _unpack_slot(layer, d - me * spr, _row(staging, row[d], nbytes))

    def _commit_tables(self, layer: _Layer, new: List[int]) -> None:
        torch.cuda.synchronize()  # the balancer writes its device table from its own stream
        layer.balancer.set_initial_weight_assignments(list(new))
        layer.assign = list(new)
        spr = layer.slots_per_rank
        start = self.ep_rank * spr
        for mod in _moe_and_backend(layer.moe()):
            if hasattr(mod, "initial_global_assignments"):
                mod.initial_global_assignments = list(new)
            if hasattr(mod, "initial_local_expert_ids"):
                mod.initial_local_expert_ids = list(new[start : start + spr])
            hook = getattr(mod, "on_expert_placement_changed", None)
            if hook is not None:
                hook(list(new))

    # ------------------------------------------------------------------ verification
    def _gather_checksums(self) -> np.ndarray:
        """Checksum of every expert of every layer, from the slot that holds it now (its first copy).
        Every other copy must carry the same bytes, else the round fails on every rank."""
        from mpi4py import MPI

        emax = max(layer.num_experts for layer in self.layers)
        local = np.zeros((len(self.layers), emax), dtype=np.int64)
        sums = {}
        movable = [layer for layer in self.layers if layer.movable]
        for layer in movable:
            start = self.ep_rank * layer.slots_per_rank
            first: Dict[int, int] = {}
            for slot, expert in enumerate(layer.assign):
                first.setdefault(expert, slot)
            sums[layer.index] = _slot_checksums(layer)
            for j, cs in sums[layer.index].items():
                if first[layer.assign[start + j]] == start + j:
                    local[layer.index, layer.assign[start + j]] = cs
        glob = np.empty_like(local)
        self._comm.Allreduce(local, glob, op=MPI.SUM)
        bad = 0
        for layer in movable:
            start = self.ep_rank * layer.slots_per_rank
            for j, cs in sums[layer.index].items():
                bad += int(cs != glob[layer.index, layer.assign[start + j]])
        if self._comm.allreduce(bad, op=MPI.SUM):
            raise RuntimeError(f"{_LOG} copies of an expert differ in bytes ({bad} slots here)")
        return glob

    def _verify(self, received: List[List[int]]) -> int:
        """Checksum every local slot that received an expert against that expert's checksum."""
        bad = 0
        for layer, slots in zip(self.layers, received):
            start = self.ep_rank * layer.slots_per_rank
            for j, cs in _slot_checksums(layer, slots).items():
                if cs != self._expert_checksums[layer.index, layer.assign[start + j]]:
                    bad += 1
        return bad


def _excess(load: np.ndarray, slot_expert: Sequence[int], slots_per_rank: int) -> float:
    rl = rank_loads(load, slot_expert, slots_per_rank)
    mean = rl.mean()
    return float((rl.max() - mean) / mean) if mean > 0 else 0.0


def _context_rows(req) -> int:
    from tensorrt_llm._torch.pyexecutor.ced_replay import encoder_replay_tokens

    return int(getattr(req, "context_chunk_size", 0) or 0) + encoder_replay_tokens(req)


def _generated_tokens(req) -> int:
    """Tokens a generation request has produced so far (its position in the response)."""
    return int(req.max_beam_num_tokens) - int(req.orig_prompt_len)


def _moe_and_backend(moe) -> List[object]:
    mods = [moe]
    backend = getattr(moe, "backend", None)
    if backend is not None and backend is not moe:
        mods.append(backend)
    return mods


def _weights_owner(moe):
    backend = getattr(moe, "backend", None)
    return backend if backend is not None else moe


def _check_backend(layer: _Layer) -> None:
    owner = _weights_owner(layer.moe())
    if hasattr(owner, "on_expert_placement_changed"):
        return
    if type(owner).__name__ not in _VERIFIED_BACKENDS:
        raise RuntimeError(
            f"{_LOG} backend {type(owner).__name__} may keep expert state this controller cannot "
            f"update (no on_expert_placement_changed); refusing to migrate"
        )


def _resolve_slot_tensors(layer: _Layer) -> Tuple[List[torch.Tensor], int]:
    owner = _weights_owner(layer.moe())
    out = []
    for name in _SLOT_PARAM_NAMES:
        t = getattr(owner, name, None)
        if not isinstance(t, torch.Tensor) or t.dim() == 0 or t.numel() == 0:
            continue
        if t.shape[0] != layer.slots_per_rank:
            continue
        if not t.is_cuda or not t.is_contiguous():
            raise RuntimeError(f"{_LOG} per-slot tensor {name} must be a contiguous CUDA tensor")
        out.append(t.data)
    if not out:
        raise RuntimeError(
            f"{_LOG} layer {layer.index}: no per-slot tensors on {type(owner).__name__}"
        )
    named = [*owner.named_parameters(recurse=False), *owner.named_buffers(recurse=False)]
    unmoved = [
        name
        for name, t in named
        if name not in _SLOT_PARAM_NAMES
        and t.dim() > 0
        and t.numel() > 0
        and t.shape[0] == layer.slots_per_rank
    ]
    if unmoved:
        raise RuntimeError(
            f"{_LOG} layer {layer.index}: {type(owner).__name__} keeps per-slot tensors this "
            f"controller does not move: {unmoved}"
        )
    return out, sum(t[0].numel() * t.element_size() for t in out)


def _row(staging: torch.Tensor, i: int, nbytes: int) -> torch.Tensor:
    return staging[i * nbytes : (i + 1) * nbytes]


def _pack_slot(layer: _Layer, local_slot: int, buf: torch.Tensor) -> None:
    off = 0
    for t in layer.slot_tensors:
        src = t[local_slot].reshape(-1).view(torch.uint8)
        buf[off : off + src.numel()].copy_(src)
        off += src.numel()


def _unpack_slot(layer: _Layer, local_slot: int, buf: torch.Tensor) -> None:
    off = 0
    for t in layer.slot_tensors:
        dst = t[local_slot].reshape(-1).view(torch.uint8)
        dst.copy_(buf[off : off + dst.numel()])
        off += dst.numel()


def _slot_checksums(layer: _Layer, slots: Optional[Sequence[int]] = None) -> Dict[int, int]:
    """Checksums of local slots (all, or ``slots``): per tensor the int64 sum of the slot's int32
    words plus a strided partial sum, combined over tensors. Detects a slot holding another expert,
    stale data or a partial copy, in bounded temporaries and one host read per layer."""
    idx = list(range(layer.slots_per_rank)) if slots is None else sorted(set(slots))
    if not idx:
        return {}
    out = torch.zeros(layer.slots_per_rank, dtype=torch.int64, device="cuda")
    chunk = max(1, (128 << 20) // max(1, layer.slot_bytes))
    for n, t in enumerate(layer.slot_tensors):
        flat_all = t.reshape(t.shape[0], -1).view(torch.uint8)
        for c in range(0, len(idx), chunk):
            sel = torch.tensor(idx[c : c + chunk], device="cuda")
            flat = flat_all.index_select(0, sel)
            words = flat.view(torch.int32) if flat.shape[1] % 4 == 0 else flat.to(torch.int32)
            total = words.sum(dim=1, dtype=torch.int64)
            part = words[:, ::97].sum(dim=1, dtype=torch.int64)
            out[sel] = out[sel] * 1000003 + total * 7919 + part + n
    host = out.cpu().tolist()
    return {j: host[j] for j in idx}


# ----------------------------------------------------------------------------------- process-wide

_controllers: "weakref.WeakSet[WarmupPlacementController]" = weakref.WeakSet()


def register_controller(controller: WarmupPlacementController) -> None:
    _controllers.add(controller)


def on_executor_step(step: int) -> None:
    """Before every executor forward, on every rank, with the executor's step counter."""
    for c in list(_controllers):
        c.on_step(step)


def active() -> bool:
    """Whether a controller still counts (cheap; for per-step hooks)."""
    return any(c.active for c in list(_controllers))


def set_live_rows(requests, draft_len: int) -> None:
    """The live-row bounds (and ``min_pos`` row masks) of the next forward for every controller."""
    for c in list(_controllers):
        if c.active:
            c.set_live_rows(requests, draft_len)


def real_rows_of(requests, draft_len: int) -> int:
    """Live MoE input rows of a scheduled batch: context chunks and generation tokens (1 +
    ``draft_len``) of the non-dummy requests. Attention-DP dummies and CUDA-graph padding requests
    follow the live ones in the token order."""
    rows = 0
    for req in requests.context_requests:
        if not req.is_dummy:
            rows += _context_rows(req)
    for req in requests.generation_requests:
        if not req.is_dummy:
            rows += 1 + int(draft_len or 0)
    return rows
