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
- State-based `_is_benchmark_disagg_fill_complete` predicate
- `can_forward` gating initialisation and transitions
- ADP dummy suppression during fill vs taper-down
- ADP router imbalance regression (nvbug 6071070)
- Non-blocking behaviour of `_prepare_and_schedule_batch`
- Insufficient-KV fail-fast vs transfer-admission backpressure
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_active_request(
    in_init: bool = False,
    in_transfer: bool = False,
    in_error: bool = False,
) -> Mock:
    """Create an active request stub with disagg state flags."""
    req = Mock()
    req.is_disagg_generation_init_state = in_init
    req.is_disagg_generation_transmission_in_progress = in_transfer
    req.state = (
        LlmRequestState.DISAGG_TRANS_ERROR if in_error else LlmRequestState.GENERATION_IN_PROGRESS
    )
    req.is_attention_dp_dummy = False
    return req


def _make_transceiver(transfer_complete: bool = True) -> Mock:
    """Create a KV cache transceiver stub."""
    transceiver = Mock()
    transceiver.check_gen_transfer_complete.return_value = transfer_complete
    return transceiver


class MockBenchmarkExecutor:
    """Minimal stub mirroring the PyExecutor attributes used by
    `_is_benchmark_disagg_fill_complete`, `_check_benchmark_disagg_gate`,
    and the `can_forward` gate.

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
        active_requests=None,
    ):
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_benchmark_disagg = (
            benchmark_req_queues_size > 0 and kv_cache_transceiver is not None
        )
        self._benchmark_fill_phase_active = self.is_benchmark_disagg
        self._sync_disagg_transfer_made_progress = False
        self._benchmark_sync_progress_global = False
        self._fill_admit_cap = 0
        self.enable_attention_dp = enable_attention_dp
        self.num_fetch_requests = num_fetch_requests
        self.is_warmup = is_warmup
        self.active_requests = active_requests if active_requests is not None else []

        self.dist = Mock()
        self.dist.rank = rank
        self.dist.tp_size = tp_size
        self.dist.world_size = tp_size

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    _dist_size = staticmethod(PyExecutor._dist_size)
    _allgather_model_parallel_status = PyExecutor._allgather_model_parallel_status
    _is_benchmark_disagg_fill_complete = PyExecutor._is_benchmark_disagg_fill_complete
    _check_benchmark_disagg_gate = PyExecutor._check_benchmark_disagg_gate


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (state-based predicate)
# ---------------------------------------------------------------------------


class TestFillCompleteStateBased:
    """Test the state-based fill-complete predicate.

    Gate opens when (A) all fetched, (B) all active requests are past
    KV-transfer states, and (C) the transceiver is drained.
    """

    def test_opens_when_all_conditions_met(self):
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(),
            num_fetch_requests=4,
            active_requests=reqs,
        )
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is True

    @pytest.mark.parametrize(
        "num_fetch, reqs, transfer_complete, reason",
        [
            pytest.param(
                2,
                [_make_active_request() for _ in range(2)],
                True,
                "not all fetched",
                id="condition_A_fails",
            ),
            pytest.param(
                4,
                [_make_active_request(), _make_active_request(in_init=True)],
                True,
                "request in INIT",
                id="condition_B_init",
            ),
            pytest.param(
                4,
                [_make_active_request(), _make_active_request(in_transfer=True)],
                True,
                "request in TRANS_IN_PROGRESS",
                id="condition_B_transfer",
            ),
            pytest.param(
                4,
                [_make_active_request() for _ in range(4)],
                False,
                "transceiver has pending receives",
                id="condition_C_fails",
            ),
        ],
    )
    def test_blocked_when_condition_fails(self, num_fetch, reqs, transfer_complete, reason):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=transfer_complete),
            num_fetch_requests=num_fetch,
            active_requests=reqs,
        )
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is False, reason

    def test_no_transceiver_skips_condition_c(self):
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=None,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex.is_benchmark_disagg = True
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is True

    def test_raises_outside_benchmark_disagg(self):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=0)
        with pytest.raises(RuntimeError):
            ex._is_benchmark_disagg_fill_complete(ScheduledRequests())


class TestFillCompleteADP:
    """Test the state-based predicate with ADP (allgather across ranks)."""

    @pytest.mark.parametrize(
        "allgather_result, expected",
        [
            pytest.param([(1, False)] * 4, True, id="all_ranks_ready"),
            pytest.param(
                [(1, False), (1, False), (0, False), (1, False)],
                False,
                id="one_rank_blocked",
            ),
            pytest.param([(0, False)] * 4, False, id="all_ranks_blocked"),
        ],
    )
    def test_global_gate(self, allgather_result, expected):
        reqs = [_make_active_request() for _ in range(2)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=True,
            tp_size=4,
            num_fetch_requests=8,
            active_requests=reqs,
        )
        ex.dist.tp_allgather.return_value = allgather_result
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is expected

    def test_allgather_sends_local_ok_int(self):
        reqs = [_make_active_request() for _ in range(2)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=True,
            tp_size=2,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex.dist.tp_allgather.return_value = [(1, False), (1, False)]
        ex._is_benchmark_disagg_fill_complete(ScheduledRequests())
        ex.dist.tp_allgather.assert_called_once_with((1, False))

    def test_single_rank_skips_model_parallel_allgather(self):
        """A singleton group returns local status without a collective."""
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=False,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex.dist.tp_size = 1
        ex.dist.cp_size = 1
        ex.dist.world_size = 1
        ex._is_benchmark_disagg_fill_complete(ScheduledRequests())
        ex.dist.tp_allgather.assert_not_called()
        ex.dist.tp_cp_allgather.assert_not_called()


class TestFillCompleteADPRouterImbalance:
    """Regression test for nvbug 6071070.

    The old count-based predicate hung when ADP router produced ±1 skew
    (e.g. 31x256 + 1x255 = 8191 < 8192). The new state-based predicate
    must open when all requests are past transfer, regardless of per-rank
    distribution.
    """

    def test_gate_opens_despite_fewer_requests_on_a_rank(self):
        reqs = [_make_active_request() for _ in range(255)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8192,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=True,
            tp_size=32,
            num_fetch_requests=8192,
            active_requests=reqs,
        )
        ex.dist.tp_allgather.return_value = [(1, False)] * 32
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is True

    def test_gate_blocked_when_overflow_requests_in_init(self):
        reqs = [_make_active_request() for _ in range(250)]
        reqs += [_make_active_request(in_init=True) for _ in range(6)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8192,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=True,
            tp_size=32,
            num_fetch_requests=8192,
            active_requests=reqs,
        )
        ex.dist.tp_allgather.return_value = [(0, False)] + [(1, False)] * 31
        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is False

    def test_gate_blocked_when_sync_transfer_failed(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=1,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            num_fetch_requests=1,
            active_requests=[_make_active_request(in_error=True)],
        )

        assert ex._is_benchmark_disagg_fill_complete(ScheduledRequests()) is False


# ---------------------------------------------------------------------------
# can_forward gating  (unit-level, no real executor loop)
# ---------------------------------------------------------------------------


class TestCanForwardGating:
    """Verify can_forward initialisation and state transitions."""

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


# ---------------------------------------------------------------------------
# _check_benchmark_disagg_gate  (consolidated gate helper)
# ---------------------------------------------------------------------------


class TestCheckBenchmarkDisaggGate:
    """Verify the consolidated gate helper used by both executor loops."""

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_opens_when_fill_complete(self, mock_time):
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            num_fetch_requests=4,
            active_requests=reqs,
        )
        assert ex._benchmark_fill_phase_active is True

        can_forward, should_retry = ex._check_benchmark_disagg_gate(ScheduledRequests(), False)
        assert can_forward is True
        assert should_retry is False
        assert ex._benchmark_fill_phase_active is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_retries_with_short_sleep_when_incomplete(self, mock_time):
        reqs = [_make_active_request(in_init=True)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=False),
            num_fetch_requests=2,
            active_requests=reqs,
        )

        can_forward, should_retry = ex._check_benchmark_disagg_gate(ScheduledRequests(), False)
        assert can_forward is False
        assert should_retry is True
        mock_time.sleep.assert_called_once_with(0.1)

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_retries_without_sleep_after_sync_transfer_progress(self, mock_time):
        reqs = [_make_active_request(in_init=True)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=False),
            num_fetch_requests=2,
            active_requests=reqs,
        )
        ex._sync_disagg_transfer_made_progress = True

        can_forward, should_retry = ex._check_benchmark_disagg_gate(ScheduledRequests(), False)

        assert can_forward is False
        assert should_retry is True
        assert ex._sync_disagg_transfer_made_progress is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_skips_sleep_on_all_adp_ranks_when_peer_makes_progress(self, mock_time):
        reqs = [_make_active_request(in_init=True)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=False),
            enable_attention_dp=True,
            tp_size=2,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex.dist.tp_allgather.return_value = [(0, False), (0, True)]

        can_forward, should_retry = ex._check_benchmark_disagg_gate(ScheduledRequests(), False)

        assert can_forward is False
        assert should_retry is True
        ex.dist.tp_allgather.assert_called_once_with((0, False))
        mock_time.sleep.assert_not_called()

    @pytest.mark.parametrize(
        "enable_attention_dp, tp_size, cp_size, gather_name",
        [
            pytest.param(False, 2, 1, "tp_allgather", id="tensor_parallel"),
            pytest.param(False, 1, 2, "tp_cp_allgather", id="context_parallel"),
            pytest.param(True, 2, 2, "tp_cp_allgather", id="attention_dp_with_cp"),
        ],
    )
    def test_gate_waits_for_blocked_model_parallel_peer(
        self, enable_attention_dp, tp_size, cp_size, gather_name
    ):
        """A ready local slice cannot open the gate ahead of a peer.

        Args:
            enable_attention_dp: Whether to simulate attention data parallelism.
            tp_size: Tensor-parallel group size.
            cp_size: Context-parallel group size.
            gather_name: Expected model-parallel allgather method.
        """
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=enable_attention_dp,
            tp_size=tp_size,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex.dist.cp_size = cp_size
        ex.dist.world_size = tp_size * cp_size
        all_rank_status = [(1, False)] * (tp_size * cp_size)
        all_rank_status[-1] = (0, False)
        gather = getattr(ex.dist, gather_name)
        gather.return_value = all_rank_status

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.time") as mock_time:
            can_forward, should_retry = ex._check_benchmark_disagg_gate(ScheduledRequests(), False)

        assert can_forward is False
        assert should_retry is True
        assert ex._benchmark_fill_phase_active is True
        gather.assert_called_once_with((1, False))
        if gather_name == "tp_cp_allgather":
            ex.dist.tp_allgather.assert_not_called()
        mock_time.sleep.assert_called_once_with(0.1)

    @pytest.mark.parametrize(
        "is_warmup, can_forward_in",
        [
            pytest.param(True, False, id="warmup_bypasses_gate"),
            pytest.param(False, True, id="already_forwarding_skips_check"),
        ],
    )
    @patch("tensorrt_llm._torch.pyexecutor.py_executor.time")
    def test_gate_no_op(self, mock_time, is_warmup, can_forward_in):
        """Gate is a no-op during warmup or when can_forward is already True."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(),
            is_warmup=is_warmup,
        )

        can_forward, should_retry = ex._check_benchmark_disagg_gate(
            ScheduledRequests(), can_forward_in
        )
        assert can_forward is can_forward_in
        assert should_retry is False
        mock_time.sleep.assert_not_called()


# ---------------------------------------------------------------------------
# _pad_attention_dp_dummy_request  (benchmark disagg condition)
# ---------------------------------------------------------------------------


class MockPadDummyExecutor:
    """Stub mirroring the PyExecutor attributes used by
    `_pad_attention_dp_dummy_request`.
    """

    def __init__(
        self,
        *,
        is_benchmark_disagg: bool = False,
        benchmark_fill_phase_active: bool | None = None,
        is_warmup: bool = False,
        enable_attention_dp: bool = True,
        kv_cache_transceiver=None,
        active_requests=None,
        expected_num_active_requests: int = 1,
        num_fetch_requests: int = 0,
        benchmark_req_queues_size: int = 8,
        tp_size: int = 1,
    ):
        self.is_benchmark_disagg = is_benchmark_disagg
        self._benchmark_fill_phase_active = (
            benchmark_fill_phase_active
            if benchmark_fill_phase_active is not None
            else is_benchmark_disagg
        )
        self.is_warmup = is_warmup
        self.enable_attention_dp = enable_attention_dp
        self.kv_cache_transceiver = kv_cache_transceiver
        self.active_requests = active_requests if active_requests is not None else []
        self.expected_num_active_requests = expected_num_active_requests
        self.num_fetch_requests = num_fetch_requests
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self.max_total_draft_tokens = 0
        self._adp_dummy_is_gen = True
        self.max_num_tokens = None

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

    During fill (_benchmark_fill_phase_active=True): skip.
    After fill (_benchmark_fill_phase_active=False): normal lifecycle.
    """

    @pytest.mark.parametrize(
        "active_reqs",
        [
            pytest.param([], id="empty_rank"),
            pytest.param(
                [_make_active_request(in_init=True), _make_active_request(in_transfer=True)],
                id="requests_in_transfer",
            ),
        ],
    )
    def test_skips_during_fill_phase(self, active_reqs):
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=active_reqs,
            expected_num_active_requests=max(1, len(active_reqs) + 1),
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    @pytest.mark.parametrize(
        "is_warmup, benchmark_fill_active, is_benchmark_disagg, reason",
        [
            pytest.param(False, False, True, "taper-down", id="after_fill_taper_down"),
            pytest.param(True, True, True, "warmup", id="warmup_bypass"),
            pytest.param(False, False, False, "non-benchmark", id="not_benchmark_disagg"),
        ],
    )
    def test_allows_dummy(self, is_warmup, benchmark_fill_active, is_benchmark_disagg, reason):
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=is_benchmark_disagg,
            benchmark_fill_phase_active=benchmark_fill_active,
            is_warmup=is_warmup,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_no_dummy_needed_when_active_requests_ready(self):
        ready_req = _make_active_request()
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            benchmark_fill_phase_active=False,
            kv_cache_transceiver=Mock(),
            active_requests=[ready_req],
            expected_num_active_requests=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_skips_when_adp_disabled(self):
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            enable_attention_dp=False,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()


# ---------------------------------------------------------------------------
# _prepare_and_schedule_batch is non-blocking
# ---------------------------------------------------------------------------


class TestPrepareAndScheduleBatchNoBlock:
    """_prepare_and_schedule_batch must not block on request fetching.

    NOTE: This test uses `object.__new__(PyExecutor)` to bypass __init__
    and manually sets internal attributes.  Keep the attribute list in sync
    with the method's implementation.
    """

    def test_fetch_called_once_even_in_benchmark_disagg(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = 8
        ex.kv_cache_transceiver = Mock()
        ex.kv_cache_manager = Mock()
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = True
        ex._fill_admit_cap = 0
        ex.enable_attention_dp = False
        ex.num_fetch_requests = 0
        ex.dist = Mock(rank=0, tp_size=1)
        ex.dist.allreduce = Mock(side_effect=lambda v, op=None: v)
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


# ---------------------------------------------------------------------------
# Benchmark fill admission flow control
# ---------------------------------------------------------------------------


class TestBenchmarkFillAdmissionFlowControl:
    """Verify benchmark disagg fill admits requests via a slow-start ramp."""

    @staticmethod
    def _make_executor(tp_size: int = 4, fill_phase_active: bool = True):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.enable_attention_dp = True
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = fill_phase_active
        ex._fill_admit_cap = 0
        ex.benchmark_req_queues_size = 32
        ex.num_fetch_requests = 0
        ex.max_num_active_requests = 8

        ex.dist = Mock()
        ex.dist.tp_size = tp_size
        return ex

    @staticmethod
    def _expected_ramp(tp_size: int, total_max: int) -> list:
        """Build the expected admission-cap sequence: tp_size, 2*tp_size, ...,
        clamped to total_max, then steady at total_max."""
        seq = []
        cap = tp_size
        while cap < total_max:
            seq.append(cap)
            cap = min(cap * 2, total_max)
        seq.append(total_max)
        return seq

    @pytest.mark.parametrize("tp_size", [1, 4, 32])
    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_first_iter_caps_at_tp_size(self, mock_get_from_waiting_queue, tp_size):
        """Iter 0 of the ramp must still cap at tp_size — this is the
        load-bearing iter-0 burst protection (PR #12206 fail-fast)."""
        ex = self._make_executor(tp_size=tp_size)
        waiting_queue = Mock()
        all_ranks_num_active_requests = [0] * tp_size

        ex._pop_from_waiting_queue(
            waiting_queue,
            total_num_active_requests=0,
            all_ranks_num_active_requests=all_ranks_num_active_requests,
        )

        mock_get_from_waiting_queue.assert_called_once()
        _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]

        assert max_new_requests == tp_size, (
            "Iter 0 of the benchmark-fill admission ramp must cap at tp_size "
            "to protect against the simultaneous-DISAGG_GENERATION_INIT burst "
            "that trips PR #12206's fail-fast under ADP router imbalance."
        )
        assert ex._fill_admit_cap == tp_size

    @pytest.mark.parametrize("tp_size", [1, 4, 8, 32])
    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_ramp_doubles_each_iter_and_saturates_at_total_max(
        self, mock_get_from_waiting_queue, tp_size
    ):
        """Each subsequent fill-phase iter doubles the admission cap, and the
        cap saturates at `total_max` (never exceeds the global capacity)."""
        ex = self._make_executor(tp_size=tp_size)
        total_max = ex.dist.tp_size * ex.max_num_active_requests
        expected = self._expected_ramp(tp_size, total_max)

        observed = []
        # Walk a few iters past saturation to confirm the cap stays put.
        for _ in range(len(expected) + 3):
            ex._pop_from_waiting_queue(
                Mock(),
                total_num_active_requests=0,
                all_ranks_num_active_requests=[0] * tp_size,
            )
            _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]
            observed.append(max_new_requests)

        # The ramp matches tp_size, 2*tp_size, ..., total_max.
        assert observed[: len(expected)] == expected, (
            f"ramp mismatch: expected {expected}, observed {observed}"
        )
        # Cap stays at total_max thereafter.
        assert all(c == total_max for c in observed[len(expected) - 1 :])
        # Convergence is logarithmic in total_max / tp_size.
        # (Allow +1 for the saturating final entry.)
        max_iters = (total_max // tp_size).bit_length() + 1
        assert len(expected) <= max_iters

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_no_fill_phase_uses_full_available_capacity(self, mock_get_from_waiting_queue):
        ex = self._make_executor(fill_phase_active=False)
        waiting_queue = Mock()
        all_ranks_num_active_requests = [0, 0, 0, 0]

        ex._pop_from_waiting_queue(
            waiting_queue,
            total_num_active_requests=0,
            all_ranks_num_active_requests=all_ranks_num_active_requests,
        )

        _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]

        assert max_new_requests == ex.dist.tp_size * ex.max_num_active_requests
        assert ex._fill_admit_cap == 0, (
            "When fill phase is inactive, the slow-start cap must remain "
            "untouched so a future fill phase starts the ramp from tp_size."
        )

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_ramp_state_resets_when_fill_phase_ends(self, mock_get_from_waiting_queue):
        """After the fill gate opens (`_fill_admit_cap` reset to 0 by
        `_check_benchmark_disagg_gate`), a *subsequent* fill phase starts
        the ramp from tp_size again rather than wherever the previous ramp
        ended.  Matters for back-to-back benchmarks (e.g. after warmup)."""
        ex = self._make_executor(tp_size=4)

        # Walk a couple of iters into the ramp.
        for _ in range(3):
            ex._pop_from_waiting_queue(
                Mock(),
                total_num_active_requests=0,
                all_ranks_num_active_requests=[0] * 4,
            )
        assert ex._fill_admit_cap > 4

        # Simulate the gate-open path that `_check_benchmark_disagg_gate`
        # takes when fill completes.
        ex._benchmark_fill_phase_active = False
        ex._fill_admit_cap = 0

        # New fill phase starts; the next throttled iter must reseed at tp_size.
        ex._benchmark_fill_phase_active = True
        ex._pop_from_waiting_queue(
            Mock(),
            total_num_active_requests=0,
            all_ranks_num_active_requests=[0] * 4,
        )
        _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]
        assert max_new_requests == 4
        assert ex._fill_admit_cap == 4

    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_warmup_skips_throttle(self, mock_get_from_waiting_queue):
        """Warmup must bypass the slow-start ramp so warmup iters are not
        artificially throttled and `_fill_admit_cap` stays unchanged."""
        ex = self._make_executor(tp_size=4)
        # Bypass the is_warmup property setter (which also pokes model_engine,
        # not present on this stub).
        ex._is_warmup = True

        ex._pop_from_waiting_queue(
            Mock(),
            total_num_active_requests=0,
            all_ranks_num_active_requests=[0] * 4,
        )

        _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]
        assert max_new_requests == ex.dist.tp_size * ex.max_num_active_requests
        assert ex._fill_admit_cap == 0


# ---------------------------------------------------------------------------
# ADP router per-rank cap  (prevents overflow that caused nvbug 6071070)
# ---------------------------------------------------------------------------

_ROUTER_CLS_PATHS = [
    pytest.param(
        "tensorrt_llm._torch.pyexecutor.scheduler.adp_router.DefaultADPRouter",
        id="DefaultADPRouter",
    ),
    pytest.param(
        "tensorrt_llm._torch.pyexecutor.scheduler.adp_router.KVCacheAwareADPRouter",
        id="KVCacheAwareADPRouter",
    ),
]


def _make_router(cls_path: str, tp_size: int = 4):
    """Instantiate a router from its dotted class path."""
    import importlib

    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState

    module_path, cls_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    RouterCls = getattr(mod, cls_name)

    dist = Mock()
    dist.tp_rank = 0
    dist.tp_size = tp_size

    extra_kwargs = {}
    if cls_name == "KVCacheAwareADPRouter":
        kv_mgr = Mock()
        kv_mgr.probe_prefix_match_length = Mock(return_value=0)
        extra_kwargs["kv_cache_manager"] = kv_mgr

    router = RouterCls(dist=dist, **extra_kwargs)
    return router, cls_name, RankState


def _make_unscheduled_request_item(req_id=None):
    """Create a mock RequestQueueItem with no py_scheduling_params."""
    req_item = Mock()
    req_item.request.py_scheduling_params = None
    req_item.request.input_token_ids = [1, 2, 3]
    req_item.id = req_id if req_id is not None else id(req_item)
    return req_item


def _prep_kv_router(router, cls_name, tp_size):
    """Populate prefix-match data for KVCacheAwareADPRouter (no-op for Default)."""
    if cls_name == "KVCacheAwareADPRouter":
        router._all_ranks_prefix_matches = [{} for _ in range(tp_size)]


class TestADPRouterPerRankCap:
    """Verify that the ADP router never proposes expected_num_active_requests
    above max_num_active_requests, even when rank imbalance or bulk arrival
    would produce a larger ceiling-division result.

    This is the root cause of nvbug 6071070: without the cap, the router
    could set expected=257 for a rank with max_batch_size=256, causing the
    heap to push requests onto ranks that can never schedule them.
    """

    @pytest.mark.parametrize("router_cls_path", _ROUTER_CLS_PATHS)
    def test_expected_capped_at_max(self, router_cls_path):
        """When one rank is heavily loaded, expected must not exceed max."""
        from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState

        router, cls_name, _ = _make_router(router_cls_path)
        _prep_kv_router(router, cls_name, 4)

        max_num_active = 4
        all_rank_states = [
            RankState(rank=0, num_active_requests=6, num_active_tokens=100),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
            RankState(rank=2, num_active_requests=0, num_active_tokens=0),
            RankState(rank=3, num_active_requests=0, num_active_tokens=0),
        ]
        requests = [_make_unscheduled_request_item(i) for i in range(3)]

        result, expected = router.route_requests(all_rank_states, requests, max_num_active)

        assert expected == max_num_active, (
            f"expected_num_active_requests should be capped at {max_num_active}, got {expected}"
        )
        assert len(result[0]) == 0, "Overloaded rank 0 (already at 6) must not receive new requests"

    @pytest.mark.parametrize("router_cls_path", _ROUTER_CLS_PATHS)
    def test_no_rank_exceeds_max(self, router_cls_path):
        """Near-capacity: no rank should be assigned beyond max_num_active."""
        from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState

        router, cls_name, _ = _make_router(router_cls_path)
        _prep_kv_router(router, cls_name, 4)

        max_num_active = 256
        all_rank_states = [
            RankState(rank=0, num_active_requests=255, num_active_tokens=1000),
            RankState(rank=1, num_active_requests=254, num_active_tokens=900),
            RankState(rank=2, num_active_requests=250, num_active_tokens=800),
            RankState(rank=3, num_active_requests=253, num_active_tokens=950),
        ]
        requests = [_make_unscheduled_request_item(i) for i in range(10)]

        result, expected = router.route_requests(all_rank_states, requests, max_num_active)

        assert expected <= max_num_active
        for rank, assigned in result.items():
            initial = all_rank_states[rank].num_active_requests
            total = initial + len(assigned)
            assert total <= max_num_active, (
                f"Rank {rank}: {initial} existing + {len(assigned)} new = {total}, "
                f"exceeds cap {max_num_active}"
            )

    @pytest.mark.parametrize("router_cls_path", _ROUTER_CLS_PATHS)
    def test_cap_not_applied_when_below_max(self, router_cls_path):
        """When the natural expected is within max, cap has no effect.

        The natural value differs per router class: DefaultADPRouter computes
        ceil(total/tp_size) directly, whereas KVCacheAwareADPRouter wraps it
        with fair_share_multiplier (default 2.0) to allow more concentration
        for cache affinity.  Both must land below max_num_active_requests so
        that the hard cap is a no-op.
        """
        from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState

        router, cls_name, _ = _make_router(router_cls_path)
        _prep_kv_router(router, cls_name, 4)

        max_num_active = 256
        all_rank_states = [
            RankState(rank=i, num_active_requests=0, num_active_tokens=0) for i in range(4)
        ]
        requests = [_make_unscheduled_request_item(i) for i in range(8)]

        result, expected = router.route_requests(all_rank_states, requests, max_num_active)

        if cls_name == "KVCacheAwareADPRouter":
            # fair_share_multiplier defaults to 2.0, so natural value is
            # ceil(2.0 * ceil(8/4)) = ceil(2.0 * 2) = 4.
            expected_natural = 4
        else:
            # DefaultADPRouter natural value is ceil(8/4) = 2.
            expected_natural = 2
        assert expected == expected_natural, (
            f"{cls_name}: natural expected was {expected_natural}, got {expected}"
        )
        # Hard cap must be a no-op here since natural value << max.
        assert expected < max_num_active
        total_assigned = sum(len(v) for v in result.values())
        assert total_assigned == 8


# ---------------------------------------------------------------------------
# Insufficient-KV fail-fast during benchmark fill
# ---------------------------------------------------------------------------


class TestFailFastDuringBenchmarkFill:
    """Verify that the insufficient-KV fail-fast distinguishes healthy fill
    progress from a stalled fill.

    During healthy fill, the scheduler can fit at least one INIT request for
    KV transfer.  If all benchmark requests have been fetched and the scheduler
    cannot fit any INIT request, the fill gate can never open and we should
    return an explicit error instead of hanging.

    This covers the CI regression where the fill-phase guard suppressed the
    fail-fast forever and
    `test_disaggregated_benchmark_gen_only_insufficient_kv` timed out.
    """

    def _make_executor(
        self,
        *,
        fill_phase_active,
        num_init_requests=3,
        num_fetch_requests=8,
        fitting_init_requests=None,
    ):
        """Build a minimal PyExecutor stub for _prepare_and_schedule_batch."""
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = 8
        ex.kv_cache_transceiver = Mock()
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = fill_phase_active
        ex._fill_admit_cap = 0
        ex.enable_attention_dp = False
        ex.num_fetch_requests = num_fetch_requests
        ex.dist = Mock(rank=0, tp_size=1, cp_size=1, world_size=1)
        ex.dist.allreduce.return_value = 0
        ex.is_shutdown = False
        ex._is_warmup = False
        ex.enable_iter_perf_stats = False
        ex.waiting_queue = []
        ex.expected_num_active_requests = 0
        ex.drafter = None
        ex.inflight_req_ids = set()
        ex.kv_connector_manager = None
        ex.enable_partial_reuse_for_disagg = False

        init_reqs = [_make_active_request(in_init=True) for _ in range(num_init_requests)]
        ready_reqs = [_make_active_request() for _ in range(5)]
        ex.active_requests = init_reqs + ready_reqs

        ex._fetch_and_activate_new_requests = Mock(return_value=[])
        ex._check_disagg_ctx_schedulable_status = Mock()
        ex._check_disagg_gen_transfer_status = Mock()
        ex._check_disagg_gen_cache_transfer_status = Mock()
        ex._check_kv_transfer_timeout = Mock()
        ex._check_disagg_ctx_cache_transfer_status = Mock()
        ex._pad_attention_dp_dummy_request = Mock()
        ex._prepare_disagg_gen_init = Mock()
        ex._handle_errors = Mock()

        scheduled = ScheduledRequests()
        if fitting_init_requests is None:
            fitting_init_requests = []
        ex._schedule = Mock(return_value=(scheduled, fitting_init_requests, 0))

        return ex

    def test_stalled_fill_phase_kills_after_all_requests_fetched(self):
        """A fill phase with no fitting INIT requests is a deadlock."""
        ex = self._make_executor(fill_phase_active=True)

        result, _ = ex._prepare_and_schedule_batch()

        assert result is None, (
            "Fail-fast SHOULD fire during a stalled fill phase — "
            "otherwise the fill gate never opens"
        )
        ex._handle_errors.assert_called_once()

    def test_healthy_fill_phase_does_not_kill(self):
        """During healthy fill, a fitting INIT request means progress exists."""
        fitting_req = _make_active_request(in_init=True)
        ex = self._make_executor(fill_phase_active=True, fitting_init_requests=[fitting_req])

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None, (
            "Fail-fast should NOT fire while the scheduler can still fit an "
            "INIT request for KV transfer"
        )
        ex._handle_errors.assert_not_called()

    def test_partial_transfer_admission_uses_only_admitted_requests(self):
        """The admitted subset is prepared and passed to the idle check."""
        admitted_req = _make_active_request(in_init=True)
        deferred_req = _make_active_request(in_init=True)
        candidates = [admitted_req, deferred_req]
        ex = self._make_executor(fill_phase_active=True, fitting_init_requests=candidates)
        ex._apply_disagg_transfer_admission = Mock(return_value=([admitted_req], False))
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None
        ex._apply_disagg_transfer_admission.assert_called_once_with(candidates)
        ex._prepare_disagg_gen_init.assert_called_once_with([admitted_req])
        ex._check_disagg_transfer_progress_when_idle.assert_called_once_with(
            0, [admitted_req], False, False
        )
        ex._handle_errors.assert_not_called()

    def test_fill_with_no_init_requests_does_not_kill(self):
        """The final fill iteration is ready for the gate, not terminal."""
        ex = self._make_executor(fill_phase_active=True, num_init_requests=0)

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None
        ex._handle_errors.assert_not_called()

    def test_transfer_admission_backpressure_does_not_kill(self, monkeypatch):
        """NVBug 6438658: admission backpressure is not KV exhaustion.

        Args:
            monkeypatch: Pytest fixture used to select asynchronous transfer
                behavior.
        """
        monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
        monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
        fitting_req = _make_active_request(in_init=True)
        ex = self._make_executor(fill_phase_active=True, fitting_init_requests=[fitting_req])
        ex._apply_disagg_transfer_admission = Mock(return_value=([], True))

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None, (
            "Fail-fast should NOT fire when the scheduler fit an INIT request "
            "that transfer admission temporarily deferred"
        )
        ex._apply_disagg_transfer_admission.assert_called_once_with([fitting_req])
        ex._prepare_disagg_gen_init.assert_called_once_with([])
        ex._check_disagg_gen_cache_transfer_status.assert_called_once_with(1)
        ex._check_disagg_ctx_cache_transfer_status.assert_not_called()
        ex._handle_errors.assert_not_called()

    @pytest.mark.parametrize(
        "enable_attention_dp, tp_size, cp_size, gather_name",
        [
            pytest.param(False, 2, 1, "tp_allgather", id="tensor_parallel"),
            pytest.param(False, 1, 2, "tp_cp_allgather", id="context_parallel"),
            pytest.param(True, 2, 2, "tp_cp_allgather", id="attention_dp_with_cp"),
        ],
    )
    def test_model_parallel_peer_terminal_no_fit_kills_all_ranks(
        self, enable_attention_dp, tp_size, cp_size, gather_name
    ):
        """A terminal peer makes every model-parallel rank fail together.

        Args:
            enable_attention_dp: Whether to simulate attention data parallelism.
            tp_size: Tensor-parallel group size.
            cp_size: Context-parallel group size.
            gather_name: Expected model-parallel allgather method.
        """
        fitting_req = _make_active_request(in_init=True)
        ex = self._make_executor(fill_phase_active=True, fitting_init_requests=[fitting_req])
        ex.enable_attention_dp = enable_attention_dp
        ex.dist.tp_size = tp_size
        ex.dist.cp_size = cp_size
        ex.dist.world_size = tp_size * cp_size
        all_rank_status = [(True, False)] * (tp_size * cp_size)
        all_rank_status[-1] = (True, True)
        gather = getattr(ex.dist, gather_name)
        gather.return_value = all_rank_status
        ex._apply_disagg_transfer_admission = Mock(return_value=([], True))
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()

        assert result is None
        gather.assert_called_once_with((True, False))
        if gather_name == "tp_cp_allgather":
            ex.dist.tp_allgather.assert_not_called()
        ex._handle_errors.assert_called_once()
        assert "one or more requests" in ex._handle_errors.call_args.args[0]

    def test_attention_dp_backpressure_without_terminal_peer_does_not_kill(self):
        """Admission backpressure stays non-terminal on every rank."""
        fitting_req = _make_active_request(in_init=True)
        ex = self._make_executor(fill_phase_active=True, fitting_init_requests=[fitting_req])
        ex.enable_attention_dp = True
        ex.dist.tp_size = 2
        ex.dist.world_size = 2
        ex.dist.tp_allgather.return_value = [
            (True, False),
            (True, False),
        ]
        ex._apply_disagg_transfer_admission = Mock(return_value=([], True))
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None
        ex.dist.tp_allgather.assert_called_once_with((True, False))
        ex._handle_errors.assert_not_called()

    def test_model_parallel_waits_until_all_ranks_have_fetched(self):
        """A terminal rank cannot fail peers that are still fetching."""
        ex = self._make_executor(fill_phase_active=True)
        ex.dist.tp_size = 2
        ex.dist.world_size = 2
        ex.dist.tp_allgather.return_value = [
            (True, True),
            (False, False),
        ]
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None
        ex.dist.tp_allgather.assert_called_once_with((True, True))
        ex._handle_errors.assert_not_called()

    def test_mid_fetch_does_not_kill(self):
        """Before all benchmark requests are fetched, keep filling."""
        ex = self._make_executor(fill_phase_active=True, num_fetch_requests=4)

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None, (
            "Fail-fast should NOT fire before the full benchmark queue has been fetched"
        )
        ex._handle_errors.assert_not_called()

    def test_post_fill_skips_fail_fast_vote(self):
        """Decode iterations must not pay for the fill-only collective."""
        ex = self._make_executor(fill_phase_active=False)
        ex.enable_attention_dp = True
        ex.dist.tp_size = 2
        ex.dist.world_size = 2
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None
        ex.dist.tp_allgather.assert_not_called()
        ex._handle_errors.assert_not_called()

    @pytest.mark.parametrize(
        "fill_active, is_warmup, expected_alive",
        [
            pytest.param(True, False, False, id="stalled_fill_kills"),
            pytest.param(False, False, True, id="post_fill_suppresses"),
            pytest.param(False, True, True, id="warmup_suppresses"),
            pytest.param(True, True, True, id="both_suppress"),
        ],
    )
    def test_suppression_matrix(self, fill_active, is_warmup, expected_alive):
        """Parametrized test covering all (fill_phase, warmup) combinations."""
        ex = self._make_executor(fill_phase_active=fill_active)
        ex._is_warmup = is_warmup

        result, _ = ex._prepare_and_schedule_batch()

        if expected_alive:
            assert result is not None, (
                f"fill_active={fill_active}, warmup={is_warmup}: should NOT kill requests"
            )
            ex._handle_errors.assert_not_called()
        else:
            assert result is None, (
                f"fill_active={fill_active}, warmup={is_warmup}: "
                "SHOULD kill requests (genuine KV insufficiency)"
            )
            ex._handle_errors.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end fill phase reproducer
# ---------------------------------------------------------------------------


class TestFillPhaseEndToEnd:
    """End-to-end simulation of the benchmark disagg fill lifecycle.

    Reproduces the exact failure sequence from the CI pipeline:
    1. All requests fetched (num_fetch_requests >= benchmark_req_queues_size)
    2. KV cache full with transferred requests, some INIT requests remain
    3. Scheduler can't fit INIT requests
    4. Verify: fail-fast does NOT fire while scheduler fits INIT requests
    5. Transfers complete, gate opens, fill phase clears
    6. Verify: the fill-only fail-fast vote stops after the gate opens

    This test catches all three bugs we found iteratively:
    - Bug 1: Count-based gate unsatisfiable under ADP router skew
    - Bug 2: Router overshooting max_batch_size
    - Bug 3: Fail-fast must distinguish healthy fill from stalled fill
    """

    TOTAL_REQUESTS = 8
    MAX_BATCH_SIZE = 4
    TP_SIZE = 2

    def _make_executor(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = self.TOTAL_REQUESTS
        ex.kv_cache_transceiver = _make_transceiver(transfer_complete=False)
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = True
        ex._fill_admit_cap = 0
        ex.enable_attention_dp = True
        ex.num_fetch_requests = 0
        ex.max_num_active_requests = self.MAX_BATCH_SIZE
        ex.dist = Mock(rank=0, tp_size=self.TP_SIZE)
        ex.dist.tp_allreduce.return_value = 0
        ex.dist.tp_allgather.return_value = [(0, False)] * self.TP_SIZE
        ex.is_shutdown = False
        ex._is_warmup = False
        ex.enable_iter_perf_stats = False
        ex.waiting_queue = []
        ex.expected_num_active_requests = 0
        ex.drafter = None
        ex.inflight_req_ids = set()
        ex.kv_connector_manager = None
        ex.enable_partial_reuse_for_disagg = False
        ex.active_requests = []

        ex._fetch_and_activate_new_requests = Mock(return_value=[])
        ex._check_disagg_ctx_schedulable_status = Mock()
        ex._check_disagg_gen_transfer_status = Mock()
        ex._check_kv_transfer_timeout = Mock()
        ex._check_disagg_ctx_cache_transfer_status = Mock()
        ex._pad_attention_dp_dummy_request = Mock()
        ex._prepare_disagg_gen_init = Mock()
        ex._handle_errors = Mock()

        scheduled = ScheduledRequests()
        ex._schedule = Mock(return_value=(scheduled, [], 0))

        return ex

    def test_full_lifecycle(self):
        """Simulate the fill → gate-open → post-fill lifecycle."""
        ex = self._make_executor()

        # Phase 0: Fill-phase admission uses the real waiting-queue path and
        # pulls only tp_size requests per iteration, rather than the full
        # tp_size * max_batch_size queue.
        with patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue"
        ) as mock_get_from_waiting_queue:
            ex._pop_from_waiting_queue(
                waiting_queue=Mock(),
                total_num_active_requests=0,
                all_ranks_num_active_requests=[0] * self.TP_SIZE,
            )

        _, max_new_requests = mock_get_from_waiting_queue.call_args.args[:2]
        assert max_new_requests == self.TP_SIZE

        # Phase 1: Fetching requests (gate should not open)
        ex.num_fetch_requests = 4
        batch = ScheduledRequests()
        assert not ex._is_benchmark_disagg_fill_complete(batch), (
            "Gate should not open: only 4/8 requests fetched"
        )

        # Phase 2: All fetched, some still in transfer
        ex.num_fetch_requests = self.TOTAL_REQUESTS
        init_reqs = [_make_active_request(in_init=True) for _ in range(3)]
        ready_reqs = [_make_active_request() for _ in range(5)]
        ex.active_requests = init_reqs + ready_reqs

        ex.dist.tp_allgather = Mock(return_value=[(0, False), (1, False)])
        assert not ex._is_benchmark_disagg_fill_complete(batch), (
            "Gate should not open: 3 requests still in INIT"
        )

        # Phase 2b: Healthy fill keeps making progress, so fail-fast must not
        # fire even though some active requests remain in INIT.
        ex._schedule = Mock(return_value=(ScheduledRequests(), [init_reqs[0]], 0))
        ex.dist.tp_allgather = Mock(return_value=[(True, False), (True, False)])
        result, _ = ex._prepare_and_schedule_batch()
        assert result is not None, (
            "Fail-fast must not kill requests while the scheduler can still fit INIT requests"
        )

        # Phase 3: All transfers complete, gate opens
        for req in init_reqs:
            req.is_disagg_generation_init_state = False
        ex.kv_cache_transceiver.check_gen_transfer_complete.return_value = True
        ex.dist.tp_allgather = Mock(return_value=[(1, False), (1, False)])

        assert ex._is_benchmark_disagg_fill_complete(batch), (
            "Gate should open: all requests past transfer, transceiver complete"
        )

        # Phase 4: Gate opens, fill phase clears
        ex._benchmark_fill_phase_active = False

        # Phase 5: Decode iterations do not re-enter the fill-only vote.
        ex._schedule = Mock(return_value=(ScheduledRequests(), [], 0))
        ex.active_requests = ready_reqs
        ex.dist.tp_allgather = Mock()
        ex._check_disagg_transfer_progress_when_idle = Mock()

        result, _ = ex._prepare_and_schedule_batch()
        assert result is not None
        ex.dist.tp_allgather.assert_not_called()
        ex._handle_errors.assert_not_called()
