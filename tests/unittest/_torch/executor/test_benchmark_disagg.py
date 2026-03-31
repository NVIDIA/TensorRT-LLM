# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for benchmark disaggregated serving gating in PyExecutor.

In benchmark disagg mode the GEN executor must defer the forward pass
until all benchmark requests have completed KV transfer.  These tests
cover:
- ``is_benchmark_disagg`` attribute initialisation
- ``_is_benchmark_disagg_fill_complete`` (non-ADP and ADP paths)
- ``can_forward`` gating initialisation and transitions
- Incremental fill convergence when CTX has limited KV cache capacity
- Non-blocking behaviour of ``_prepare_and_schedule_batch``
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gen_request(is_dummy: bool = False) -> Mock:
    """Create a generation request stub with the ``is_attention_dp_dummy`` flag."""
    req = Mock()
    req.is_attention_dp_dummy = is_dummy
    return req


def _make_scheduled_batch(num_gen_requests: int, num_dummy_requests: int = 0) -> ScheduledRequests:
    """Create a ScheduledRequests with generation stubs.

    Args:
        num_gen_requests: Number of real (non-dummy) generation requests.
        num_dummy_requests: Number of ADP dummy generation requests.
    """
    batch = ScheduledRequests()
    batch.generation_requests = [
        _make_gen_request(is_dummy=False) for _ in range(num_gen_requests)
    ] + [_make_gen_request(is_dummy=True) for _ in range(num_dummy_requests)]
    return batch


class MockBenchmarkExecutor:
    """Minimal stub mirroring the PyExecutor attributes used by
    ``_is_benchmark_disagg_fill_complete``, ``_check_benchmark_disagg_gate``,
    and the ``can_forward`` gate.

    Binds the real production methods so tests exercise actual logic
    without needing a fully-initialised executor.
    """

    def __init__(
        self,
        benchmark_req_queues_size: int = 0,
        kv_cache_transceiver=None,
        enable_attention_dp: bool = False,
        tp_size: int = 1,
        rank: int = 0,
        num_fetch_requests: int = 0,
        is_warmup: bool = False,
    ):
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_benchmark_disagg = (
            benchmark_req_queues_size > 0 and kv_cache_transceiver is not None
        )
        self.enable_attention_dp = enable_attention_dp
        self.num_fetch_requests = num_fetch_requests
        self.is_warmup = is_warmup

        self.dist = Mock()
        self.dist.rank = rank
        self.dist.tp_size = tp_size

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    _is_benchmark_disagg_fill_complete = PyExecutor._is_benchmark_disagg_fill_complete
    _check_benchmark_disagg_gate = PyExecutor._check_benchmark_disagg_gate


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (non-ADP)
# ---------------------------------------------------------------------------


class TestFillCompleteNonADP:
    @pytest.mark.parametrize(
        "num_gen_requests, expected",
        [
            pytest.param(4, True, id="meets_threshold"),
            pytest.param(6, True, id="exceeds_threshold"),
            pytest.param(2, False, id="below_threshold"),
            pytest.param(0, False, id="zero_requests"),
        ],
    )
    def test_threshold(self, num_gen_requests, expected):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)

        assert ex._is_benchmark_disagg_fill_complete(batch) is expected

    def test_no_allgather_called_without_adp(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), enable_attention_dp=False
        )
        batch = _make_scheduled_batch(num_gen_requests=4)

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_not_called()

    def test_logs_progress_on_rank_zero(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), rank=0, num_fetch_requests=2
        )
        batch = _make_scheduled_batch(num_gen_requests=1)

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_called_once()
            msg = mock_logger.debug.call_args[0][0]
            assert "fill in progress" in msg
            assert "num_fetched=2" in msg

    def test_no_log_on_non_zero_rank(self):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), rank=1)
        batch = _make_scheduled_batch(num_gen_requests=1)

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (ADP)
# ---------------------------------------------------------------------------


class TestFillCompleteADP:
    @pytest.mark.parametrize(
        "num_gen_requests, allgather_result, expected",
        [
            pytest.param(2, [2, 2, 2, 2], True, id="meets_threshold"),
            pytest.param(3, [3, 3, 3, 3], True, id="exceeds_threshold"),
            pytest.param(1, [1, 1, 1, 0], False, id="below_threshold"),
            pytest.param(5, [5, 1, 1, 1], True, id="uneven_distribution"),
        ],
    )
    def test_threshold(self, num_gen_requests, allgather_result, expected):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=4,
        )
        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)
        ex.dist.tp_allgather.return_value = allgather_result

        assert ex._is_benchmark_disagg_fill_complete(batch) is expected

    def test_allgather_receives_local_count(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=3)
        ex.dist.tp_allgather.return_value = [3, 1]

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_called_once_with(3)

    def test_allgather_excludes_dummy_requests(self):
        """Dummy requests must not inflate the local count sent via allgather."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=3)
        ex.dist.tp_allgather.return_value = [2, 2]

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_called_once_with(2)

    def test_logs_progress_on_rank_zero(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
            num_fetch_requests=3,
        )
        batch = _make_scheduled_batch(num_gen_requests=2)
        ex.dist.tp_allgather.return_value = [2, 1]

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_called_once()
            msg = mock_logger.debug.call_args[0][0]
            assert "total_gen_count=3" in msg
            assert "local=2" in msg

    def test_no_log_on_non_zero_rank(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
            rank=1,
        )
        batch = _make_scheduled_batch(num_gen_requests=1)
        ex.dist.tp_allgather.return_value = [1, 0]

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# ADP regression: mixed real + dummy generation requests
# ---------------------------------------------------------------------------


class TestFillCompleteADPDummyExclusion:
    """Verify that ADP dummy requests do not inflate the fill threshold.

    In ADP, ``_pad_attention_dp_dummy_request`` injects dummy generation
    requests on ranks with no active work.  These dummies must be excluded
    from the ``total_gen_count`` so the ``can_forward`` gate only opens
    after the required number of *real* requests complete KV transfer.
    """

    def test_dummies_do_not_trigger_threshold(self):
        """8 dummies + 0 real must not satisfy a threshold of 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=0, num_dummy_requests=8)
        ex.dist.tp_allgather.return_value = [0, 0]

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_mixed_real_and_dummy_only_counts_real(self):
        """2 real + 3 dummies on each rank: total real = 4, threshold = 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=3)
        ex.dist.tp_allgather.return_value = [2, 2]

        assert ex._is_benchmark_disagg_fill_complete(batch) is True

    def test_mixed_below_threshold(self):
        """1 real + 5 dummies on each rank: total real = 2, threshold = 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=1, num_dummy_requests=5)
        ex.dist.tp_allgather.return_value = [1, 1]

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_non_adp_dummies_excluded(self):
        """Without ADP, dummies should also be excluded from the local count."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=False,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=5)

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_non_adp_real_only_meets_threshold(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=False,
        )
        batch = _make_scheduled_batch(num_gen_requests=4, num_dummy_requests=3)

        assert ex._is_benchmark_disagg_fill_complete(batch) is True


# ---------------------------------------------------------------------------
# can_forward gating  (unit-level, no real executor loop)
# ---------------------------------------------------------------------------


class TestCanForwardGating:
    """Verify can_forward initialisation and state transitions.

    The can_forward gate is shared by _executor_loop and
    _executor_loop_overlap to defer the forward pass in benchmark
    disagg mode until all requests are generation-ready.
    """

    @pytest.mark.parametrize(
        "benchmark_size, transceiver, is_disagg, can_forward",
        [
            pytest.param(0, None, False, True, id="no_benchmark"),
            pytest.param(8, None, False, True, id="benchmark_without_disagg"),
            pytest.param(0, "mock", False, True, id="disagg_without_benchmark"),
            pytest.param(8, "mock", True, False, id="benchmark_and_disagg"),
        ],
    )
    def test_initial_value(self, benchmark_size, transceiver, is_disagg, can_forward):
        kv = Mock() if transceiver == "mock" else None
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=benchmark_size, kv_cache_transceiver=kv
        )
        assert ex.is_benchmark_disagg is is_disagg
        assert (not ex.is_benchmark_disagg) is can_forward

    @pytest.mark.parametrize(
        "num_gen_requests, expected_after_fill",
        [
            pytest.param(4, True, id="complete_fill"),
            pytest.param(2, False, id="incomplete_fill"),
        ],
    )
    def test_transition(self, num_gen_requests, expected_after_fill):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        can_forward = not ex.is_benchmark_disagg
        assert can_forward is False

        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)
        can_forward = ex._is_benchmark_disagg_fill_complete(batch)
        assert can_forward is expected_after_fill

    def test_can_forward_stays_true_once_set(self):
        """can_forward is latching: once True it must not revert."""
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        can_forward = not ex.is_benchmark_disagg

        batch_partial = _make_scheduled_batch(num_gen_requests=4)
        can_forward = ex._is_benchmark_disagg_fill_complete(batch_partial)
        assert can_forward is True

        batch_empty = _make_scheduled_batch(num_gen_requests=0)
        # After can_forward is True, the gate is never re-entered in the
        # real loop (guarded by `if not can_forward`).  Verify that calling
        # the helper again with an empty batch would return False, but
        # can_forward itself is not mutated.
        result = ex._is_benchmark_disagg_fill_complete(batch_empty)
        assert result is False
        assert can_forward is True  # local variable unchanged


# ---------------------------------------------------------------------------
# _check_benchmark_disagg_gate  (consolidated gate helper)
# ---------------------------------------------------------------------------


class TestCheckBenchmarkDisaggGate:
    """Verify the consolidated gate helper used by both executor loops."""

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_opens_when_fill_complete(self, mock_time):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=4)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is True
        assert should_retry is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_blocks_and_sleeps_when_incomplete(self, mock_time):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=1)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is False
        assert should_retry is True
        mock_time.sleep.assert_called_once_with(1)

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_warmup_bypasses_gate(self, mock_time):
        """During warmup, the gate must not block even in benchmark disagg mode."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            is_warmup=True,
        )
        batch = _make_scheduled_batch(num_gen_requests=0)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is False
        assert should_retry is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_already_forwarding_skips_check(self, mock_time):
        """Once can_forward is True, the gate is a no-op."""
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=0)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, True)
        assert can_forward is True
        assert should_retry is False
        mock_time.sleep.assert_not_called()


# ---------------------------------------------------------------------------
# _pad_attention_dp_dummy_request  (benchmark disagg condition)
# ---------------------------------------------------------------------------


def _make_active_request(in_init: bool = False, in_transfer: bool = False) -> Mock:
    """Create an active request stub for _pad_attention_dp_dummy_request."""
    req = Mock()
    req.is_disagg_generation_init_state = in_init
    req.is_disagg_generation_transmission_in_progress = in_transfer
    return req


class MockPadDummyExecutor:
    """Stub mirroring the PyExecutor attributes used by
    ``_pad_attention_dp_dummy_request``.

    Only the benchmark disagg early-return guard and the dummy-addition
    branch are exercised; the rest is mocked out.
    """

    def __init__(
        self,
        *,
        is_benchmark_disagg: bool = False,
        is_warmup: bool = False,
        enable_attention_dp: bool = True,
        kv_cache_transceiver=None,
        active_requests=None,
        expected_num_active_requests: int = 1,
        num_fetch_requests: int = 0,
        tp_size: int = 1,
    ):
        self.is_benchmark_disagg = is_benchmark_disagg
        self.is_warmup = is_warmup
        self.enable_attention_dp = enable_attention_dp
        self.kv_cache_transceiver = kv_cache_transceiver
        self.active_requests = active_requests if active_requests is not None else []
        self.expected_num_active_requests = expected_num_active_requests
        self.num_fetch_requests = num_fetch_requests
        self.max_total_draft_tokens = 0

        self.dist = Mock()
        self.dist.tp_size = tp_size

        self.kv_cache_manager = Mock()
        dummy_req = Mock()
        dummy_req.is_attention_dp_dummy = True
        self.kv_cache_manager.add_dummy_requests.return_value = [dummy_req]

        self.resource_manager = Mock()
        self.resource_manager.get_resource_manager.return_value = None

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    _pad_attention_dp_dummy_request = PyExecutor._pad_attention_dp_dummy_request
    _count_schedulable_active_requests = PyExecutor._count_schedulable_active_requests
    _should_skip_dummy_for_benchmark_disagg = PyExecutor._should_skip_dummy_for_benchmark_disagg


class TestPadAttentionDpDummyBenchmarkDisagg:
    """Verify _pad_attention_dp_dummy_request skips dummy insertion correctly.

    The guard skips dummies when:
    - benchmark disagg is active, not warming up, no active requests, AND
    - either ADP is disabled, or num_fetch_requests >= tp_size (every rank
      has at least one real INIT request).

    When ADP is enabled and num_fetch_requests < tp_size, some ranks have
    zero requests and still need dummies for collective operations.
    """

    def test_skips_dummy_when_adp_and_fetch_ge_tp(self):
        """ADP with num_fetch_requests >= tp_size: all ranks have INIT requests."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
            num_fetch_requests=2,
            tp_size=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_skips_dummy_when_all_requests_in_transfer(self):
        """Requests exist but all are in INIT/transfer state."""
        reqs = [_make_active_request(in_init=True), _make_active_request(in_transfer=True)]
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=reqs,
            expected_num_active_requests=3,
            num_fetch_requests=4,
            tp_size=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_allows_dummy_during_warmup(self):
        """Warmup must bypass the benchmark disagg guard."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            is_warmup=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_allows_dummy_when_not_benchmark_disagg(self):
        """Non-benchmark or non-disagg mode: normal dummy insertion."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=False,
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_allows_dummy_when_active_requests_ready(self):
        """Some requests have completed KV transfer: guard does not trigger."""
        ready_req = _make_active_request(in_init=False, in_transfer=False)
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[ready_req],
            expected_num_active_requests=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_skips_when_adp_disabled(self):
        """_pad_attention_dp_dummy_request early-returns when ADP is off."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            enable_attention_dp=False,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_adp_allows_dummy_when_fetch_below_tp_size(self):
        """With ADP, dummies are needed when fewer requests than TP ranks."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
            num_fetch_requests=1,
            tp_size=4,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_adp_skips_dummy_when_fetch_reaches_tp_size(self):
        """With ADP, skip dummies once every rank has at least one request."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[_make_active_request(in_init=True)],
            expected_num_active_requests=2,
            num_fetch_requests=4,
            tp_size=4,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_adp_allows_dummy_at_zero_fetched(self):
        """With ADP, very first iteration: no requests fetched yet."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
            num_fetch_requests=0,
            tp_size=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()


# ---------------------------------------------------------------------------
# _prepare_and_schedule_batch is non-blocking
# ---------------------------------------------------------------------------


class TestIncrementalFillScenario:
    """Simulate incremental request arrival when CTX has limited KV cache.

    Setup: the CTX server can only send a few requests per iteration
    (limited KV cache), while the GEN server has enough capacity for all
    requests.  The GEN executor must cycle its main loop — fetching a
    batch, servicing KV transfers, checking readiness — so the CTX
    server can free KV cache and make progress incrementally.

    These tests replay the outer-loop logic over multiple iterations
    with controlled request arrival and KV-transfer completion to verify
    the system converges without blocking.
    """

    TOTAL_REQUESTS = 8
    CTX_CAPACITY = 2  # CTX can only release this many per iteration

    def test_gen_side_processes_requests_incrementally(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=self.TOTAL_REQUESTS,
            kv_cache_transceiver=Mock(),
        )

        fetched = 0
        gen_ready = 0

        def simulate_fetch():
            nonlocal fetched
            new = min(self.CTX_CAPACITY, self.TOTAL_REQUESTS - fetched)
            fetched += new
            ex.num_fetch_requests = fetched

        def simulate_kv_transfer():
            nonlocal gen_ready
            gen_ready = fetched

        can_forward = not ex.is_benchmark_disagg
        assert can_forward is False

        iterations = 0
        MAX_ITER = 50

        while not can_forward and iterations < MAX_ITER:
            simulate_fetch()
            simulate_kv_transfer()
            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert fetched == self.TOTAL_REQUESTS
        assert gen_ready == self.TOTAL_REQUESTS
        assert iterations > 1, "Should take multiple iterations (not a single blocking call)"

    def test_kv_transfer_lag_still_converges(self):
        """KV transfers complete one iteration behind request fetching.

        Iteration 1: fetch 2, gen_ready 0
        Iteration 2: fetch 4, gen_ready 2  (transfers from iter 1 complete)
        ...
        This models the realistic case where KV transfer takes time.
        """
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=self.TOTAL_REQUESTS,
            kv_cache_transceiver=Mock(),
        )

        fetched = 0
        gen_ready = 0
        prev_fetched = 0

        can_forward = not ex.is_benchmark_disagg
        iterations = 0
        MAX_ITER = 50

        while not can_forward and iterations < MAX_ITER:
            gen_ready = prev_fetched
            prev_fetched = fetched

            new = min(self.CTX_CAPACITY, self.TOTAL_REQUESTS - fetched)
            fetched += new
            ex.num_fetch_requests = fetched

            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert gen_ready >= self.TOTAL_REQUESTS
        assert iterations > 2, "With transfer lag, should take more iterations than without"

    def test_single_request_at_a_time(self):
        """Worst case: CTX releases exactly 1 request per iteration."""
        total = 4
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=total,
            kv_cache_transceiver=Mock(),
        )

        gen_ready = 0
        can_forward = not ex.is_benchmark_disagg
        iterations = 0

        while not can_forward and iterations < 50:
            gen_ready = min(gen_ready + 1, total)
            ex.num_fetch_requests = gen_ready
            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert iterations == total


class TestPrepareAndScheduleBatchNoBlock:
    """_prepare_and_schedule_batch must not block on request fetching.

    It should call _fetch_and_activate_new_requests exactly once per
    invocation, regardless of benchmark_req_queues_size, so the outer
    executor loop remains free to service KV transfers between iterations.

    NOTE: This test uses ``object.__new__(PyExecutor)`` to bypass __init__
    and manually sets internal attributes.  This is inherently fragile —
    if _prepare_and_schedule_batch gains new attribute references the test
    will fail with AttributeError.  Keep the attribute list below in sync
    with the method's implementation.
    """

    def test_fetch_called_once_even_in_benchmark_disagg(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = 8
        ex.kv_cache_transceiver = Mock()
        ex.is_benchmark_disagg = True
        ex.enable_attention_dp = False
        ex.num_fetch_requests = 0
        ex.dist = Mock(rank=0, tp_size=1)
        ex.is_shutdown = False
        ex._is_warmup = False
        ex.enable_iter_perf_stats = False
        ex.active_requests = []
        ex.waiting_queue = []
        ex.expected_num_active_requests = 0
        ex.drafter = None
        ex.inflight_req_ids = set()
        ex.kv_connector_manager = None
        ex.enable_partial_reuse_for_disagg = False

        mock_fetch = Mock(return_value=[])
        ex._fetch_and_activate_new_requests = mock_fetch
        ex._check_disagg_ctx_schedulable_status = Mock()
        ex._check_disagg_gen_transfer_status = Mock()
        ex._check_kv_transfer_timeout = Mock()
        ex._check_disagg_ctx_cache_transfer_status = Mock()
        ex._pad_attention_dp_dummy_request = Mock()
        ex._schedule = Mock(return_value=(ScheduledRequests(), [], 0))
        ex._prepare_disagg_gen_init = Mock()

        ex._prepare_and_schedule_batch()

        mock_fetch.assert_called_once()
