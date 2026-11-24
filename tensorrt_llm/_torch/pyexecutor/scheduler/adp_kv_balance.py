# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV-length-balanced admission for attention-DP generation workers.

With attention DP every rank attends over the KV of its own requests, and the
ranks step in lockstep (the MoE exchange waits for the slowest rank), so the
step time follows the rank holding the most KV tokens. At saturation the
default admission refills a freed slot with the next request in FIFO order no
matter how much KV the slot's rank already holds.

This module picks, for each free slot, the waiting request whose prompt length
brings that rank's KV load closest to the mean load the ranks would have once
every free slot is filled. Requests that have waited ``max_wait_iters``
admission rounds at the head window are admitted first, to the least loaded
rank, so a long request cannot starve. The decision depends only on the
waiting queue and the all-gathered rank states, so every rank computes the
same assignment.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..executor_request_queue import RequestQueueItem
    from .waiting_queue import WaitingQueue

# TRTLLM_ADP_KV_BALANCE_COUNT_FIRST=<slack> (e.g. 1.1; unset or 0 = off): place each admitted request on the rank
# with the fewest requests, then the least KV, among the ranks whose KV after the admission stays within slack x
# the mean KV per rank (else on the least-KV rank). An attention-DP decode step follows the busiest rank's request
# count more than its KV, so this trades a little KV spread for a lower busiest-rank count.
KV_BALANCE_COUNT_FIRST_ENV = "TRTLLM_ADP_KV_BALANCE_COUNT_FIRST"


def kv_balance_count_first_slack() -> Optional[float]:
    """The count-first KV slack of ``TRTLLM_ADP_KV_BALANCE_COUNT_FIRST`` (``None`` when off)."""
    raw = os.environ.get(KV_BALANCE_COUNT_FIRST_ENV, "0")
    try:
        slack = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{KV_BALANCE_COUNT_FIRST_ENV} must be a number >= 1 or 0, got {raw!r}"
        ) from error
    if slack == 0:
        return None
    if not math.isfinite(slack) or slack < 1:
        raise ValueError(f"{KV_BALANCE_COUNT_FIRST_ENV} must be a number >= 1 or 0, got {raw!r}")
    return slack


def plan_kv_balanced_admission(
    candidate_tokens: Sequence[int],
    candidate_forced: Sequence[bool],
    rank_tokens: Sequence[int],
    rank_free_slots: Sequence[int],
    max_new: int,
    rank_num_requests: Optional[Sequence[int]] = None,
    count_first_slack: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """Return ``(candidate_index, rank)`` pairs, in admission order.

    Args:
        candidate_tokens: KV tokens each waiting candidate brings (queue order).
        candidate_forced: Whether the candidate has waited too long and must be
            admitted before any other choice.
        rank_tokens: Current KV tokens per rank.
        rank_free_slots: Free request slots per rank.
        max_new: Upper bound on the number of admissions.
        rank_num_requests: Requests per rank (count-first placement only).
        count_first_slack: Count-first placement (``TRTLLM_ADP_KV_BALANCE_COUNT_FIRST``): every admitted
            candidate goes to the rank with the fewest requests, then the least KV, among the ranks whose
            KV after it stays within this factor of the mean KV per rank (else to the least-KV rank). When
            every candidate fits they are placed largest first, else the first ``budget`` in queue order
            (forced first). ``None`` keeps the KV-target placement.
    """
    loads = list(rank_tokens)
    free = [max(0, f) for f in rank_free_slots]
    num_ranks = len(loads)
    remaining = list(range(len(candidate_tokens)))
    budget = min(max_new, sum(free), len(remaining))
    plan: List[Tuple[int, int]] = []
    if budget <= 0:
        return plan

    if count_first_slack is not None:
        counts = list(rank_num_requests) if rank_num_requests is not None else [0] * num_ranks

        def count_first_rank(tokens: int) -> int:
            ranks = [r for r in range(num_ranks) if free[r] > 0]
            ceiling = count_first_slack * (sum(loads) + tokens) / num_ranks
            fit = [r for r in ranks if loads[r] + tokens <= ceiling]
            if fit:
                return min(fit, key=lambda r: (counts[r], loads[r], r))
            return min(ranks, key=lambda r: (loads[r], counts[r], r))

        if budget >= len(remaining):
            order = sorted(remaining, key=lambda i: (-candidate_tokens[i], i))
        else:
            order = sorted(remaining, key=lambda i: (not candidate_forced[i], i))[:budget]
        for idx in order:
            rank = count_first_rank(candidate_tokens[idx])
            plan.append((idx, rank))
            loads[rank] += candidate_tokens[idx]
            counts[rank] += 1
            free[rank] -= 1
        return plan

    # Mean load once every slot we fill holds an average-sized candidate.
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
            # Fill the rank furthest below target first; give it the candidate
            # that lands it closest to the target (ties: earlier in the queue).
            rank = min(open_ranks, key=lambda r: (loads[r] - target, r))
            want = target - loads[rank]
            idx = min(remaining, key=lambda i: (abs(candidate_tokens[i] - want), i))
        plan.append((idx, rank))
        remaining.remove(idx)
        loads[rank] += candidate_tokens[idx]
        free[rank] -= 1
    return plan


def select_kv_balanced_requests(
    waiting_queue: "WaitingQueue",
    rank_tokens: Sequence[int],
    rank_num_requests: Sequence[int],
    max_num_active_requests: int,
    max_new: int,
    window: int,
    max_wait_iters: int,
    wait_rounds: Dict[int, int],
    num_tokens_fn: Callable[["RequestQueueItem"], int],
    count_first_slack: Optional[float] = None,
) -> Optional[Tuple[List["RequestQueueItem"], Dict[int, List["RequestQueueItem"]]]]:
    """Pop a KV-balanced set of requests from ``waiting_queue``.

    Returns ``None`` (caller falls back to the default FIFO path) when the head
    window holds anything but plain single requests. ``wait_rounds`` is the
    caller-owned per-request counter of admission rounds a request spent in the
    head window without being chosen; it is updated in place.
    """
    if max_new <= 0 or not waiting_queue:
        return [], {r: [] for r in range(len(rank_tokens))}
    candidates = list(itertools.islice(iter(waiting_queue), max(window, 1)))
    if any(not item.is_normal_request or item.child_req_ids for item in candidates):
        return None

    free = [max_num_active_requests - n for n in rank_num_requests]
    tokens = [num_tokens_fn(item) for item in candidates]
    forced = [wait_rounds.get(item.id, 0) >= max_wait_iters for item in candidates]
    plan = plan_kv_balanced_admission(
        tokens,
        forced,
        rank_tokens,
        free,
        max_new,
        rank_num_requests=rank_num_requests,
        count_first_slack=count_first_slack,
    )

    assignment: Dict[int, List["RequestQueueItem"]] = {r: [] for r in range(len(rank_tokens))}
    chosen = []
    for idx, rank in plan:
        assignment[rank].append(candidates[idx])
        chosen.append(candidates[idx])
    chosen_ids = {item.id for item in chosen}
    if chosen_ids:
        waiting_queue.remove_by_ids(chosen_ids)
    # Requests only move toward the head, so an id that left the window was
    # admitted or cancelled; drop its counter.
    candidate_ids = {item.id for item in candidates}
    for stale_id in [rid for rid in wait_rounds if rid not in candidate_ids]:
        del wait_rounds[stale_id]
    for item in candidates:
        if item.id in chosen_ids:
            wait_rounds.pop(item.id, None)
        elif plan:
            # Only rounds that admitted somebody count as skipping this one.
            wait_rounds[item.id] = wait_rounds.get(item.id, 0) + 1
    return chosen, assignment
