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
- State-based ``_is_benchmark_disagg_fill_complete`` predicate
- ``can_forward`` gating initialisation and transitions
- ADP dummy suppression during fill vs taper-down
- ADP router imbalance regression (nvbug 6071070)
- Non-blocking behaviour of ``_prepare_and_schedule_batch``
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_active_request(
    in_init: bool = False,
    in_transfer: bool = False,
) -> Mock:
    """Create an active request stub with disagg state flags."""
    req = Mock()
    req.is_disagg_generation_init_state = in_init
    req.is_disagg_generation_transmission_in_progress = in_transfer
    req.is_attention_dp_dummy = False
    return req


def _make_transceiver(transfer_complete: bool = True) -> Mock:
    """Create a KV cache transceiver stub."""
    transceiver = Mock()
    transceiver.check_gen_transfer_complete.return_value = transfer_complete
    return transceiver


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
        active_requests=None,
    ):
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_benchmark_disagg = (
            benchmark_req_queues_size > 0 and kv_cache_transceiver is not None
        )
        self._benchmark_fill_phase_active = self.is_benchmark_disagg
        self.enable_attention_dp = enable_attention_dp
        self.num_fetch_requests = num_fetch_requests
        self.is_warmup = is_warmup
        self.active_requests = active_requests if active_requests is not None else []

        self.dist = Mock()
        self.dist.rank = rank
        self.dist.tp_size = tp_size

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    _is_benchmark_disagg_fill_complete = PyExecutor._is_benchmark_disagg_fill_complete
    _check_benchmark_disagg_gate = PyExecutor._check_benchmark_disagg_gate


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (state-based predicate)
# ---------------------------------------------------------------------------


class TestFillCompleteStateBased:
    """Test the state-based fill-complete predicate.

    Gate opens when (A) all fetched, (B) all past transfer, (C) no inflight.
    """

    def test_opens_when_all_conditions_met(self):
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
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
            pytest.param([1, 1, 1, 1], True, id="all_ranks_ready"),
            pytest.param([1, 1, 0, 1], False, id="one_rank_blocked"),
            pytest.param([0, 0, 0, 0], False, id="all_ranks_blocked"),
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
        ex.dist.tp_allgather.return_value = [1, 1]
        ex._is_benchmark_disagg_fill_complete(ScheduledRequests())
        ex.dist.tp_allgather.assert_called_once_with(1)

    def test_no_allgather_without_adp(self):
        reqs = [_make_active_request() for _ in range(4)]
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=_make_transceiver(transfer_complete=True),
            enable_attention_dp=False,
            num_fetch_requests=4,
            active_requests=reqs,
        )
        ex._is_benchmark_disagg_fill_complete(ScheduledRequests())
        ex.dist.tp_allgather.assert_not_called()


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
        ex.dist.tp_allgather.return_value = [1] * 32
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
        ex.dist.tp_allgather.return_value = [0] + [1] * 31
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
    ``_pad_attention_dp_dummy_request``.
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

    NOTE: This test uses ``object.__new__(PyExecutor)`` to bypass __init__
    and manually sets internal attributes.  Keep the attribute list in sync
    with the method's implementation.
    """

    def test_fetch_called_once_even_in_benchmark_disagg(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = 8
        ex.kv_cache_transceiver = Mock()
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = True
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


# ---------------------------------------------------------------------------
# Benchmark fill admission flow control
# ---------------------------------------------------------------------------


class TestBenchmarkFillAdmissionFlowControl:
    """Verify benchmark disagg fill admits requests gradually.

    The failing wide-EP Kimi case has ``benchmark_req_queues_size`` equal to
    ``tp_size * max_batch_size``.  Without an explicit fill-phase cap, GEN can
    admit the entire benchmark queue in one iteration, prepare too many KV
    receives before the first forward pass, and hit process-level memory
    pressure.  The desired invariant is non-blocking flow control: each
    executor iteration should still return to the outer loop, but it should
    only admit a small bounded number of new requests during the fill phase.
    """

    @staticmethod
    def _make_executor(tp_size: int = 4, fill_phase_active: bool = True):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.enable_attention_dp = True
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = fill_phase_active
        ex.benchmark_req_queues_size = 32
        ex.num_fetch_requests = 0
        ex.max_num_active_requests = 8

        ex.dist = Mock()
        ex.dist.tp_size = tp_size
        return ex

    @pytest.mark.parametrize("tp_size", [1, 4, 32])
    @patch("tensorrt_llm._torch.pyexecutor.py_executor.get_from_waiting_queue")
    def test_fill_phase_caps_admission_to_tp_size(self, mock_get_from_waiting_queue, tp_size):
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

        assert max_new_requests == ex.dist.tp_size, (
            "Benchmark disagg fill should admit at most tp_size requests per "
            "executor iteration.  Using full global capacity would admit "
            "tp_size * max_batch_size requests at once and recreate the "
            "fill-phase memory-pressure failure."
        )

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
# Fail-fast suppression during fill phase (prevents false-positive kills)
# ---------------------------------------------------------------------------


class TestFailFastSuppressedDuringFill:
    """Verify that the PR #12206 fail-fast (Insufficient KV cache) does not
    fire while the benchmark fill phase is active.

    During fill, INIT requests are expected — they're waiting for KV
    transfer, not for KV allocation.  The fail-fast should only fire after
    the gate opens (fill phase complete), when stuck INIT requests genuinely
    indicate insufficient KV cache.

    This is the root cause of the CI failure on pipeline 49471411: the
    fail-fast fired with "61 request(s) are waiting for KV cache allocation"
    during the fill phase, before the gate had a chance to open.
    """

    def _make_executor(self, *, fill_phase_active, num_init_requests=3):
        """Build a minimal PyExecutor stub for _prepare_and_schedule_batch."""
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        ex = object.__new__(PyExecutor)
        ex.benchmark_req_queues_size = 8
        ex.kv_cache_transceiver = Mock()
        ex.is_benchmark_disagg = True
        ex._benchmark_fill_phase_active = fill_phase_active
        ex.enable_attention_dp = False
        ex.num_fetch_requests = 8
        ex.dist = Mock(rank=0, tp_size=1)
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
        ex._check_kv_transfer_timeout = Mock()
        ex._check_disagg_ctx_cache_transfer_status = Mock()
        ex._pad_attention_dp_dummy_request = Mock()
        ex._prepare_disagg_gen_init = Mock()
        ex._handle_errors = Mock()

        scheduled = ScheduledRequests()
        ex._schedule = Mock(return_value=(scheduled, [], 0))

        return ex

    def test_no_kill_during_fill_phase(self):
        """During fill phase, stuck INIT requests must not trigger fail-fast."""
        ex = self._make_executor(fill_phase_active=True)

        result, _ = ex._prepare_and_schedule_batch()

        assert result is not None, (
            "Fail-fast should NOT fire during fill phase — "
            "INIT requests are expected while KV transfers are in progress"
        )

    def test_kills_after_fill_phase(self):
        """After fill phase completes, stuck INIT requests trigger fail-fast."""
        ex = self._make_executor(fill_phase_active=False)

        result, _ = ex._prepare_and_schedule_batch()

        assert result is None, (
            "Fail-fast SHOULD fire after fill phase — "
            "stuck INIT requests indicate genuine KV insufficiency"
        )

    @pytest.mark.parametrize(
        "fill_active, is_warmup, expected_alive",
        [
            pytest.param(True, False, True, id="fill_phase_suppresses"),
            pytest.param(False, False, False, id="post_fill_kills"),
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
        else:
            assert result is None, (
                f"fill_active={fill_active}, warmup={is_warmup}: "
                "SHOULD kill requests (genuine KV insufficiency)"
            )


# ---------------------------------------------------------------------------
# End-to-end fill phase reproducer
# ---------------------------------------------------------------------------


class TestFillPhaseEndToEnd:
    """End-to-end simulation of the benchmark disagg fill lifecycle.

    Reproduces the exact failure sequence from the CI pipeline:
    1. All requests fetched (num_fetch_requests >= benchmark_req_queues_size)
    2. KV cache full with transferred requests, some INIT requests remain
    3. Scheduler can't fit INIT requests
    4. Verify: fail-fast does NOT fire during fill phase
    5. Transfers complete, gate opens, fill phase clears
    6. Verify: fail-fast DOES fire if INIT requests remain after fill

    This test catches all three bugs we found iteratively:
    - Bug 1: Count-based gate unsatisfiable under ADP router skew
    - Bug 2: Router overshooting max_batch_size
    - Bug 3: Fail-fast firing during fill phase
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
        ex.enable_attention_dp = True
        ex.num_fetch_requests = 0
        ex.max_num_active_requests = self.MAX_BATCH_SIZE
        ex.dist = Mock(rank=0, tp_size=self.TP_SIZE)
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

        ex.dist.tp_allgather = Mock(return_value=[0, 1])
        assert not ex._is_benchmark_disagg_fill_complete(batch), (
            "Gate should not open: 3 requests still in INIT"
        )

        # Phase 2b: Fail-fast must NOT fire during fill phase
        result, _ = ex._prepare_and_schedule_batch()
        assert result is not None, (
            "Fail-fast must not kill requests during fill phase, "
            "even though scheduler can't fit INIT requests"
        )

        # Phase 3: All transfers complete, gate opens
        for req in init_reqs:
            req.is_disagg_generation_init_state = False
        ex.kv_cache_transceiver.check_gen_transfer_complete.return_value = True
        ex.dist.tp_allgather = Mock(return_value=[1, 1])

        assert ex._is_benchmark_disagg_fill_complete(batch), (
            "Gate should open: all requests past transfer, transceiver complete"
        )

        # Phase 4: Gate opens, fill phase clears
        ex._benchmark_fill_phase_active = False

        # Phase 5: After fill, if new INIT requests appear, fail-fast fires
        stuck_req = _make_active_request(in_init=True)
        ex.active_requests = [stuck_req] + ready_reqs

        result, _ = ex._prepare_and_schedule_batch()
        assert result is None, (
            "Fail-fast SHOULD fire after fill phase completes — "
            "stuck INIT requests now indicate genuine KV insufficiency"
        )
