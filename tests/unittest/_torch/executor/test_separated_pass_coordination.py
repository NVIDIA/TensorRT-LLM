# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP ranks agree on context-only and generation-only passes."""

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import SeparatedBatchKind

pytestmark = pytest.mark.cpu_only

PREFILL = SeparatedBatchKind.PREFILL_ONLY
DECODE = SeparatedBatchKind.DECODE_ONLY


class _Dist:
    """Gathers the given per-rank rows; this rank's row is ignored."""

    def __init__(self, rows):
        self.rows = rows

    def tp_allgather_int64(self, values):
        return np.asarray(self.rows, dtype=np.int64)


def _executor(rows, *, balance=True, timeout=3, max_batch_size=4, last=None):
    scheduler = SimpleNamespace(pending_separable_work=lambda active, inflight: (True, 1))
    executor = SimpleNamespace(
        scheduler=scheduler,
        active_requests=[],
        inflight_req_ids=set(),
        dist=_Dist(rows),
        attention_dp_enable_balance=balance,
        attention_dp_time_out_iters=timeout,
        attention_dp_batching_wait_iters=0,
        attention_dp_min_generation_requests=0,
        attention_dp_low_occupancy_timeout_iters=0,
        adp_ctx_waiting_iters_count=0,
        adp_ctx_batching_wait_iters_count=0,
        max_batch_size=max_batch_size,
        _last_separated_batch_kind=last,
    )
    return executor


def _decide(executor):
    return PyExecutor._coordinate_separated_batch_kind(executor)


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([[0, 2], [0, 3]], id="no-context-anywhere"),
        pytest.param([[1, 0], [1, 0]], id="no-generation-anywhere"),
    ],
)
def test_one_kind_of_work_is_an_ordinary_pass(rows):
    executor = _executor(rows, last=PREFILL)
    assert _decide(executor) is None
    assert executor._last_separated_batch_kind is None


def test_context_pass_is_released_when_every_rank_is_ready():
    executor = _executor([[1, 2], [1, 3]])
    assert _decide(executor) is PREFILL
    # The next pass is generation-only, whatever the ranks then hold.
    assert _decide(executor) is DECODE
    assert _decide(executor) is PREFILL


def test_context_waits_for_the_other_ranks_until_the_timeout():
    executor = _executor([[1, 2], [0, 3]], timeout=2)
    assert [_decide(executor) for _ in range(3)] == [DECODE, DECODE, PREFILL]
    assert executor.adp_ctx_waiting_iters_count == 0


def test_full_rank_holds_context_back():
    executor = _executor([[1, 4], [1, 3]], max_batch_size=4, timeout=10)
    assert _decide(executor) is DECODE


def test_rank_without_generation_releases_context_at_once():
    executor = _executor([[1, 0], [0, 3]], timeout=10)
    assert _decide(executor) is PREFILL


def test_without_balancing_context_is_released_at_once():
    executor = _executor([[1, 2], [0, 3]], balance=False)
    assert _decide(executor) is PREFILL
