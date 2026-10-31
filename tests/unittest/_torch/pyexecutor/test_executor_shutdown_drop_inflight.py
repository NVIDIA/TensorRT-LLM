# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A requested shutdown that drops in-flight requests instead of draining them.

``drop_inflight_requests_on_shutdown`` lets the executor loop stop while
requests are still active or queued. Without it every case keeps the drain:
the loop stops only once nothing is left and no connector transfer is pending.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import (
    PyExecutor,
    _resolve_drop_inflight_on_shutdown,
)
from tensorrt_llm._torch.pyexecutor.scheduler import create_waiting_queue
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only

# Leaves `_drop_inflight_on_shutdown` unset, as on executors assembled without `__init__`.
_UNSET = object()


class _KeptDraining(Exception):
    """Raised by the first step after the stop check: the loop kept going."""


def _executor(
    *,
    drop_inflight: object = _UNSET,
    active: int = 0,
    queued: int = 0,
    connector: str | None = None,
) -> PyExecutor:
    """An executor asked to shut down, assembled without ``__init__``.

    ``connector`` is None (no connector), "idle", "load" (a prefix load in flight)
    or "transfer" (a save in flight).
    """
    executor = object.__new__(PyExecutor)
    if drop_inflight is not _UNSET:
        executor._drop_inflight_on_shutdown = drop_inflight
    executor.is_shutdown = True
    executor.active_requests = [
        SimpleNamespace(py_request_id=i, state=LlmRequestState.CONTEXT_INIT) for i in range(active)
    ]
    executor.waiting_queue = create_waiting_queue()
    executor.waiting_queue.add_requests(RequestQueueItem(100 + i) for i in range(queued))
    executor.enable_attention_dp = False
    executor.kv_connector_manager = None
    if connector is not None:
        executor.kv_connector_manager = Mock(
            **{"has_pending_loads.return_value": connector == "load"}
        )
        executor.async_transfer_manager = Mock(
            **{"has_any_inflight_requests.return_value": connector == "transfer"}
        )
    # The termination path, down to the resource managers.
    executor.kv_cache_transceiver = None
    executor._disagg_pp_termination_handler = None
    executor._is_kv_manager_v2 = False
    executor.sampler = Mock()
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = set()
    executor._disagg_coordinator = Mock()
    executor.model_engine = None
    executor.draft_model_engine = None
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor.result_wait_queues = {}
    return executor


@pytest.fixture
def log_calls(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Capture what the TRT-LLM logger emits, with every log-once key unused."""
    calls = Mock()
    monkeypatch.setattr(logger, "log", calls)
    monkeypatch.setattr(logger, "_appeared_keys", set())
    return calls


def _messages(log_calls: Mock, severity: str) -> list[str]:
    return [
        " ".join(map(str, c.args[1:])) for c in log_calls.call_args_list if c.args[0] == severity
    ]


@pytest.mark.parametrize(
    ("setting", "pp_size", "disable_overlap_scheduler", "enabled", "warned"),
    [
        (_UNSET, 1, True, False, False),
        (False, 1, True, False, False),
        (True, 1, True, True, False),
        (True, 2, True, False, True),
        (True, 1, False, False, True),
    ],
    ids=["absent", "off", "on", "two-stages", "overlap"],
)
def test_the_executor_takes_the_setting_from_llm_args(
    log_calls: Mock,
    setting: object,
    pp_size: int,
    disable_overlap_scheduler: bool,
    enabled: bool,
    warned: bool,
) -> None:
    """Only one pipeline stage without the overlap scheduler may drop in-flight work.

    A PP loop can stop with microbatches still travelling between ranks, and the
    overlap loop would skip processing its previous batch. TorchLlmArgs accepts the
    setting only without them; a value that reaches the executor anyway is ignored
    with a warning. llm_args without the field (stand-ins) keep the drain.
    """
    llm_args = SimpleNamespace()
    if setting is not _UNSET:
        llm_args.drop_inflight_requests_on_shutdown = setting

    resolved = _resolve_drop_inflight_on_shutdown(llm_args, pp_size, disable_overlap_scheduler)

    assert resolved is enabled
    warnings = _messages(log_calls, logger.WARNING)
    if warned:
        (warning,) = warnings
        assert warning.startswith("drop_inflight_requests_on_shutdown is ignored: it needs")
    else:
        assert warnings == []


@pytest.mark.parametrize("connector", [None, "idle", "load", "transfer"])
@pytest.mark.parametrize(("active", "queued"), [(0, 0), (2, 0), (0, 3)])
@pytest.mark.parametrize("is_shutdown", [False, True])
@pytest.mark.parametrize("drop_inflight", [_UNSET, False, True])
def test_a_requested_shutdown_drops_in_flight_work_only_when_set(
    log_calls: Mock,
    drop_inflight: object,
    is_shutdown: bool,
    active: int,
    queued: int,
    connector: str | None,
) -> None:
    """Set, work still active or queued no longer holds the loop; unset, it is drained.

    A connector load or save reads the KV pools that shutdown frees next, so it is
    waited for either way. Dropping is warned once, with the counts.
    """
    executor = _executor(
        drop_inflight=drop_inflight, active=active, queued=queued, connector=connector
    )
    executor.is_shutdown = is_shutdown
    dropping = drop_inflight is True and active + queued > 0
    stops = is_shutdown and connector in (None, "idle") and (dropping or active + queued == 0)

    assert executor.should_stop_processing is stops
    assert executor.should_stop_processing is stops
    warnings = _messages(log_calls, logger.WARNING)
    if stops and dropping:
        (warning,) = warnings
        assert warning.startswith(
            f"shutdown requested: dropping {active} active and {queued} queued"
        )
        assert "(drop_inflight_requests_on_shutdown)" in warning
    else:
        assert warnings == []


@pytest.mark.parametrize("drop_inflight", [_UNSET, False, True])
def test_dropped_requests_release_through_the_termination_path(drop_inflight: object) -> None:
    """Each dropped request hands its KV cache and slots back before shutdown frees the pools.

    A request whose KV cache the transfer fabric may still read keeps it, as on
    cancellation; the KV cache manager closes that cache at its own shutdown.
    Unset, the release leaves everything in place, e.g. after the insufficient-KV
    benchmark exit.
    """
    executor = _executor(drop_inflight=drop_inflight, active=3, queued=2)
    requests = list(executor.active_requests)
    released, in_transfer_state, pipelined = requests
    in_transfer_state.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.has_inflight_transfer.side_effect = lambda req: req is pipelined
    executor.result_wait_queues = {req.py_request_id: Mock() for req in requests}

    executor._release_requests_dropped_at_shutdown()

    if drop_inflight is not True:
        assert executor.active_requests == requests and len(executor.waiting_queue) == 2
        executor.resource_manager.free_resources.assert_not_called()
        return
    executor.resource_manager.free_resources.assert_called_once_with(released)
    executor._disagg_coordinator.forget_request.assert_called_once_with(released.py_request_id)
    assert set(executor.result_wait_queues) == {
        in_transfer_state.py_request_id,
        pipelined.py_request_id,
    }
    assert executor.active_requests == [] and len(executor.waiting_queue) == 0


@pytest.mark.parametrize(("active", "queued"), [(0, 0), (2, 3)])
@pytest.mark.parametrize("drop_inflight", [_UNSET, False, True])
def test_the_loop_releases_dropped_requests_after_its_last_collective(
    monkeypatch: pytest.MonkeyPatch, drop_inflight: object, active: int, queued: int
) -> None:
    """The release follows the loop's final response flush and its completion mark.

    The flush is the loop's last collective. Releasing after it keeps the release
    rank-local, and an error raised by it is reported as teardown rather than as a
    crash that strands peer ranks. Unset, pending work keeps the loop going.
    """
    for name in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{name}", Mock())
    executor = _executor(drop_inflight=drop_inflight, active=active, queued=queued)
    executor.is_shutdown = False

    def fetch_the_shutdown_request() -> list:
        executor.is_shutdown = True
        return []

    executor._fetch_and_activate_new_requests = Mock(side_effect=fetch_the_shutdown_request)
    executor._handle_control_request = Mock(side_effect=_KeptDraining)
    # No disaggregated transfer failed, so the stop's flush is the loop's only one.
    executor._disagg_coordinator.handle_errors_synced.return_value = False
    order = []
    executor._flush_pending_transfer_responses = Mock(side_effect=lambda: order.append("flush"))
    executor.resource_manager.free_resources.side_effect = lambda req: order.append(
        (req.py_request_id, executor._event_loop_completed)
    )
    executor.device_id = 0
    executor._profiler = lambda: nullcontext(lambda: None)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor.is_benchmark_disagg = False
    executor._event_loop_completed = False
    keeps_draining = active + queued > 0 and drop_inflight is not True

    with pytest.raises(_KeptDraining) if keeps_draining else nullcontext():
        executor._executor_loop()

    assert executor._event_loop_completed is not keeps_draining
    assert order == ([] if keeps_draining else ["flush", *[(i, True) for i in range(active)]])
    left = len(executor.active_requests) + len(executor.waiting_queue)
    assert left == (active + queued if keeps_draining else 0)
