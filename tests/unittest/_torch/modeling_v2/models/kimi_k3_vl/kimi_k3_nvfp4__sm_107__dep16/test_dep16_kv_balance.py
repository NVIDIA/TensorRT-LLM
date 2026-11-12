# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kimi_k3_nvfp4__sm_107__dep16: the KV-balanced admission planner decides as the c383 reference does.

The oracle is the reference planner (084745cde4 ``pyexecutor/scheduler/adp_kv_balance.py``, run by the
c383 GEN servers with window 32 and 64 wait rounds), copied below without change but for the queue, which is
a list of ``(id, tokens)`` here. The target's planner runs inside the engine's ``DefaultADPRouter``, which
keeps the window, the wait-round count and the queue; both are driven with the same synthetic loads and
must admit the same requests to the same ranks, round after round.
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_only

_DEP16 = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16"
)
_REF_WINDOW = 32
_REF_MAX_WAIT_ITERS = 64


def _ref_plan(
    candidate_tokens: Sequence[int],
    candidate_forced: Sequence[bool],
    rank_tokens: Sequence[int],
    rank_free_slots: Sequence[int],
    max_new: int,
) -> List[Tuple[int, int]]:
    loads = list(rank_tokens)
    free = [max(0, f) for f in rank_free_slots]
    num_ranks = len(loads)
    remaining = list(range(len(candidate_tokens)))
    budget = min(max_new, sum(free), len(remaining))
    plan: List[Tuple[int, int]] = []
    if budget <= 0:
        return plan
    avg_candidate = sum(candidate_tokens) / len(candidate_tokens)
    target = (sum(loads) + avg_candidate * budget) / num_ranks
    for _ in range(budget):
        open_ranks = [r for r in range(num_ranks) if free[r] > 0]
        if not open_ranks or not remaining:
            break
        forced = [i for i in remaining if candidate_forced[i]]
        if forced:
            idx = forced[0]
            rank = min(open_ranks, key=lambda r: (loads[r], r))
        else:
            rank = min(open_ranks, key=lambda r: (loads[r] - target, r))
            want = target - loads[rank]
            idx = min(remaining, key=lambda i: (abs(candidate_tokens[i] - want), i))
        plan.append((idx, rank))
        remaining.remove(idx)
        loads[rank] += candidate_tokens[idx]
        free[rank] -= 1
    return plan


def _ref_select(
    queue: List[Tuple[int, int]],
    rank_tokens: Sequence[int],
    rank_num_requests: Sequence[int],
    max_num_active_requests: int,
    max_new: int,
    wait_rounds: Dict[int, int],
) -> Tuple[List[int], Dict[int, int]]:
    """The reference ``select_kv_balanced_requests`` on a list queue: (admitted ids in queue order,
    id -> rank). Updates ``queue`` and ``wait_rounds`` in place."""
    if max_new <= 0 or not queue:
        return [], {}
    candidates = list(itertools.islice(iter(queue), max(_REF_WINDOW, 1)))
    free = [max_num_active_requests - n for n in rank_num_requests]
    tokens = [n for _, n in candidates]
    forced = [wait_rounds.get(rid, 0) >= _REF_MAX_WAIT_ITERS for rid, _ in candidates]
    plan = _ref_plan(tokens, forced, rank_tokens, free, max_new)
    placement = {candidates[idx][0]: rank for idx, rank in plan}
    queue[:] = [item for item in queue if item[0] not in placement]
    candidate_ids = {rid for rid, _ in candidates}
    for stale_id in [rid for rid in wait_rounds if rid not in candidate_ids]:
        del wait_rounds[stale_id]
    for rid, _ in candidates:
        if rid in placement:
            wait_rounds.pop(rid, None)
        elif plan:
            wait_rounds[rid] = wait_rounds.get(rid, 0) + 1
    chosen = [rid for rid, _ in candidates if rid in placement]
    return chosen, placement


def _planner():
    import importlib

    return importlib.import_module(f"{_DEP16}.modeling").K3Dep16KvBalancedAdmission()


def _random_load(rng: random.Random):
    num_ranks = rng.choice([2, 4, 16])
    slots = rng.choice([1, 4, 16])
    counts = [rng.randint(0, slots) for _ in range(num_ranks)]
    loads = [n * rng.randint(0, 250_000) for n in counts]
    num_candidates = rng.randint(1, 40)
    tokens = [
        rng.choice([rng.randint(1, 4_000), rng.randint(4_000, 400_000)])
        for _ in range(num_candidates)
    ]
    forced = [rng.random() < 0.05 for _ in range(num_candidates)]
    max_new = rng.randint(1, num_ranks * slots)
    return dict(
        candidate_tokens=tokens,
        candidate_forced=forced,
        rank_tokens=loads,
        rank_num_requests=counts,
        max_requests_per_rank=slots,
        max_new_requests=max_new,
        candidate_groups=list(range(num_candidates)),
    )


def test_window_is_the_reference_32():
    planner = _planner()
    assert type(planner).WINDOW == _REF_WINDOW
    for max_new, free in [(1, 0), (4, 3), (64, 256), (1000, 1000)]:
        assert planner.window(max_new, free) == _REF_WINDOW


def test_plan_matches_the_reference_on_synthetic_loads():
    """Every decision equals the reference planner's on 5000 random rank states and windows (forced
    candidates, full ranks, more candidates than slots and the other way round); the admission groups
    the engine passes are ignored, as the reference has none."""
    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import plan_kv_balanced_admission

    planner = _planner()
    rng = random.Random(1111)
    engine_differs = 0
    for _ in range(5000):
        load = _random_load(rng)
        want = _ref_plan(
            load["candidate_tokens"],
            load["candidate_forced"],
            load["rank_tokens"],
            [load["max_requests_per_rank"] - n for n in load["rank_num_requests"]],
            load["max_new_requests"],
        )
        assert planner.plan(**load) == want, load
        engine_differs += plan_kv_balanced_admission(**load) != want
    # The engine's own planner decides differently on these loads: the declaration is what makes
    # the c383 GEN servers admit as the reference did.
    assert engine_differs > 0


def _gen_item(req_id: int, num_tokens: int):
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
    from tensorrt_llm.bindings.executor import RequestType
    from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY

    request = SimpleNamespace(
        num_input_tokens=num_tokens,
        request_type=RequestType.REQUEST_TYPE_GENERATION_ONLY,
        py_scheduling_params=None,
        py_conversation_params=None,
        priority=DEFAULT_REQUEST_PRIORITY,
    )
    return RequestQueueItem(req_id, request)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_admits_as_the_reference_round_after_round(seed):
    """The engine's router with the target's planner and the reference's selection, fed the same
    arrivals and completions over 400 admission rounds of 16 attention-DP ranks, admit the same
    requests to the same ranks in every round, including requests forced after 64 passed-over rounds."""
    from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
        KV_BALANCE_MAX_SKIPPED_ROUNDS,
        DefaultADPRouter,
        RankState,
    )

    assert KV_BALANCE_MAX_SKIPPED_ROUNDS == _REF_MAX_WAIT_ITERS
    # 4 slots per rank: arrivals outpace completions, so the queue backs up past the window and the
    # balance passes over some requests long enough to force them (50 to 70 per seed).
    num_ranks, slots = 16, 4
    dist = MagicMock()
    dist.tp_rank, dist.tp_size, dist.has_cp_helix = 0, num_ranks, False
    router = DefaultADPRouter(
        dist=dist, kv_balanced_admission=True, kv_balanced_admission_planner=_planner()
    )
    queue = FCFSWaitingQueue([])
    ref_queue: List[Tuple[int, int]] = []
    ref_wait: Dict[int, int] = {}
    active: List[List[Tuple[int, int]]] = [[] for _ in range(num_ranks)]
    rng = random.Random(seed)
    next_id = forced_admissions = 0
    for _ in range(400):
        for _ in range(rng.randint(0, 12)):
            # Mostly short requests, now and then a very long one that the balance passes over.
            n = rng.randint(200_000, 900_000) if rng.random() < 0.03 else rng.randint(500, 60_000)
            queue.add_request(_gen_item(next_id, n))
            ref_queue.append((next_id, n))
            next_id += 1
        for rank in range(num_ranks):
            active[rank] = [r for r in active[rank] if rng.random() > 0.08]
        tokens = [sum(n for _, n in reqs) for reqs in active]
        counts = [len(reqs) for reqs in active]
        max_new = num_ranks * slots - sum(counts)
        if max_new <= 0:
            continue
        states = [
            RankState(rank=r, num_active_requests=counts[r], num_active_tokens=tokens[r])
            for r in range(num_ranks)
        ]
        forced_before = {rid for rid, w in ref_wait.items() if w >= _REF_MAX_WAIT_ITERS}
        chosen, placement = router.pop_requests(queue, states, max_new, slots)
        want_ids, want_placement = _ref_select(ref_queue, tokens, counts, slots, max_new, ref_wait)
        assert [item.id for item in chosen] == want_ids
        assert placement == want_placement
        assert [item.id for item in queue] == [rid for rid, _ in ref_queue]
        forced_admissions += len(forced_before & set(want_ids))
        size = {item.id: item.request.num_input_tokens for item in chosen}
        for rid, rank in placement.items():
            active[rank].append((rid, size[rid]))
    assert next_id > 1000
    assert forced_admissions > 0
