# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KV-length-balanced attention-DP admission."""

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler.adp_kv_balance import (
    KV_BALANCE_COUNT_FIRST_ENV,
    kv_balance_count_first_slack,
    plan_kv_balanced_admission,
    select_kv_balanced_requests,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue


@dataclass
class _Item:
    id: int
    tokens: int
    child_req_ids: Optional[List[int]] = None
    is_normal_request: bool = True
    request: object = field(default=None)


def _queue(sizes):
    queue = FCFSWaitingQueue()
    for i, tokens in enumerate(sizes):
        queue.add_request(_Item(id=i, tokens=tokens))
    return queue


def test_heavy_rank_gets_short_request_light_rank_gets_long():
    # Rank 0 is heavy, rank 1 light; one slot each.
    plan = plan_kv_balanced_admission(
        candidate_tokens=[500_000, 20_000, 150_000],
        candidate_forced=[False, False, False],
        rank_tokens=[1_500_000, 600_000],
        rank_free_slots=[1, 1],
        max_new=2,
    )
    assigned = {rank: idx for idx, rank in plan}
    assert assigned[1] == 0  # the 500k request goes to the light rank
    assert assigned[0] == 1  # the heavy rank takes the 20k request


def test_forced_request_admitted_first_to_least_loaded_rank():
    plan = plan_kv_balanced_admission(
        candidate_tokens=[900_000, 10_000],
        candidate_forced=[True, False],
        rank_tokens=[100, 50],
        rank_free_slots=[1, 1],
        max_new=1,
    )
    assert plan == [(0, 1)]


def test_respects_slots_and_budget():
    plan = plan_kv_balanced_admission(
        candidate_tokens=[1, 2, 3, 4, 5],
        candidate_forced=[False] * 5,
        rank_tokens=[0, 0, 0],
        rank_free_slots=[0, 2, 0],
        max_new=5,
    )
    assert len(plan) == 2
    assert all(rank == 1 for _, rank in plan)
    assert plan_kv_balanced_admission([1], [False], [0], [1], 0) == []


def test_select_pops_chosen_and_ages_the_rest():
    queue = _queue([800_000, 30_000, 40_000])
    wait_rounds = {}
    chosen, assignment = select_kv_balanced_requests(
        queue,
        rank_tokens=[2_000_000, 2_100_000],
        rank_num_requests=[8, 7],
        max_num_active_requests=8,
        max_new=4,
        window=8,
        max_wait_iters=2,
        wait_rounds=wait_rounds,
        num_tokens_fn=lambda item: item.tokens,
    )
    # Only rank 1 has a slot; target ~2.195M wants ~95k there, so the 40k
    # request (closest) is taken and the 800k head request waits.
    assert [item.id for item in chosen] == [2]
    assert [item.id for item in assignment[1]] == [2]
    assert assignment[0] == []
    assert [item.id for item in queue] == [0, 1]
    assert wait_rounds == {0: 1, 1: 1}

    # After max_wait_iters skipped rounds the long head request is forced in.
    wait_rounds[0] = 2
    chosen, assignment = select_kv_balanced_requests(
        queue,
        rank_tokens=[2_000_000, 2_130_000],
        rank_num_requests=[7, 8],
        max_num_active_requests=8,
        max_new=4,
        window=8,
        max_wait_iters=2,
        wait_rounds=wait_rounds,
        num_tokens_fn=lambda item: item.tokens,
    )
    assert [item.id for item in chosen] == [0]
    assert [item.id for item in assignment[0]] == [0]
    assert 0 not in wait_rounds


def test_falls_back_on_special_items():
    queue = FCFSWaitingQueue()
    queue.add_request(_Item(id=0, tokens=10, is_normal_request=False))
    result = select_kv_balanced_requests(
        queue, [0, 0], [0, 0], 8, 4, 8, 4, {}, lambda item: item.tokens
    )
    assert result is None
    assert len(queue) == 1


class TestCountFirst:
    """TRTLLM_ADP_KV_BALANCE_COUNT_FIRST: each admission goes to the rank with the fewest requests among
    those whose KV stays within slack x the mean KV per rank."""

    def test_slack_parsing(self, monkeypatch):
        monkeypatch.delenv(KV_BALANCE_COUNT_FIRST_ENV, raising=False)
        assert kv_balance_count_first_slack() is None
        monkeypatch.setenv(KV_BALANCE_COUNT_FIRST_ENV, "0")
        assert kv_balance_count_first_slack() is None
        monkeypatch.setenv(KV_BALANCE_COUNT_FIRST_ENV, "1.1")
        assert kv_balance_count_first_slack() == pytest.approx(1.1)
        for bad in ("0.5", "nan", "x"):
            monkeypatch.setenv(KV_BALANCE_COUNT_FIRST_ENV, bad)
            with pytest.raises(ValueError, match=KV_BALANCE_COUNT_FIRST_ENV):
                kv_balance_count_first_slack()

    def test_fewest_requests_within_the_kv_slack(self):
        # Rank 0: light KV, many requests; rank 1: more KV, few requests.
        kwargs = dict(
            candidate_tokens=[10_000],
            candidate_forced=[False],
            rank_tokens=[100_000, 110_000],
            rank_free_slots=[8, 8],
            max_new=1,
            rank_num_requests=[8, 2],
        )
        assert plan_kv_balanced_admission(**kwargs) == [(0, 0)]  # KV target: the lighter rank
        assert plan_kv_balanced_admission(**kwargs, count_first_slack=1.1) == [(0, 1)]
        # Past the slack the request goes to the least-KV rank.
        assert plan_kv_balanced_admission(
            **{**kwargs, "rank_tokens": [100_000, 300_000]}, count_first_slack=1.1
        ) == [(0, 0)]

    def test_partial_window_admits_forced_then_queue_order(self):
        plan = plan_kv_balanced_admission(
            candidate_tokens=[1_000, 2_000, 3_000],
            candidate_forced=[False, False, True],
            rank_tokens=[0, 0],
            rank_free_slots=[1, 1],
            max_new=2,
            rank_num_requests=[0, 0],
            count_first_slack=1.1,
        )
        assert [idx for idx, _ in plan] == [2, 0]
        assert sorted(rank for _, rank in plan) == [0, 1]

    def test_select_passes_the_slack_through(self):
        queue = _queue([10_000])
        chosen, assignment = select_kv_balanced_requests(
            queue,
            rank_tokens=[100_000, 110_000],
            rank_num_requests=[8, 2],
            max_num_active_requests=16,
            max_new=1,
            window=32,
            max_wait_iters=64,
            wait_rounds={},
            num_tokens_fn=lambda item: item.tokens,
            count_first_slack=1.1,
        )
        assert [item.id for item in chosen] == [0]
        assert [item.id for item in assignment[1]] == [0]
