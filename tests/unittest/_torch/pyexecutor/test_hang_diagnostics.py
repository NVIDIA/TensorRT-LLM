# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import hang_detector as hang_detector_module
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.hang_diagnostics import (
    _INITIALIZATION_HANG_TIMEOUT_SECONDS,
    _INITIALIZATION_REPORT_CONTEXT,
    HANG_DIAGNOSTICS_ENV,
    ExecutorHangDiagnostics,
    create_executor_hang_diagnostics,
    monitor_executor_initialization,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def test_executor_initialization_monitor_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(HANG_DIAGNOSTICS_ENV, raising=False)
    detector = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.hang_diagnostics.HangDetector",
        detector,
    )

    with monitor_executor_initialization():
        pass

    detector.assert_not_called()


def test_executor_initialization_monitor_arms_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HANG_DIAGNOSTICS_ENV, "1")
    detector = MagicMock()
    detector_factory = Mock(return_value=detector)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.hang_diagnostics.HangDetector",
        detector_factory,
    )

    with monitor_executor_initialization():
        detector.checkpoint.assert_called_once_with()

    detector_factory.assert_called_once_with(
        timeout=_INITIALIZATION_HANG_TIMEOUT_SECONDS,
        report_context=_INITIALIZATION_REPORT_CONTEXT,
    )
    detector.register_status_provider.assert_called_once()
    detector.__enter__.assert_called_once_with()
    detector.__exit__.assert_called_once_with(None, None, None)


def test_hang_diagnostics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HANG_DIAGNOSTICS_ENV, raising=False)
    detector = HangDetector()
    event_factory = Mock()

    assert ExecutorHangDiagnostics.from_environment(rank=0) is None
    diagnostics, forward_event = create_executor_hang_diagnostics(
        rank=0,
        status_provider_registry=detector,
        event_factory=event_factory,
    )
    assert diagnostics is None
    assert forward_event is None
    event_factory.assert_not_called()
    assert detector._status_providers == []


def test_hang_diagnostic_hook_tolerates_partially_initialized_executor() -> None:
    executor = object.__new__(PyExecutor)
    PyExecutor._maybe_record_hang_diagnostic_phase(executor, "scheduling")

    executor_mock = MagicMock(spec=PyExecutor)
    PyExecutor._maybe_record_hang_diagnostic_phase(executor_mock, "scheduling")


def test_hang_diagnostic_hook_tolerates_missing_forward_event() -> None:
    executor = object.__new__(PyExecutor)
    executor._hang_diagnostics = Mock()
    executor.active_requests = []
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue_size.return_value = 0
    executor.iter_counter = 3

    executor._maybe_record_hang_diagnostic_phase("forward_returned", forward_completion="record")

    executor._hang_diagnostics.record.assert_called_once_with(
        "forward_returned",
        3,
        "active_request_ids=[], request_queue_size=0",
        forward_completion_event=None,
    )


def test_enqueue_responses_records_skipped_phase_on_non_output_rank() -> None:
    executor = object.__new__(PyExecutor)
    executor.dist = Mock()
    executor.dist.mapping.tp_group = [1]
    executor.gather_all_responses = False
    executor._maybe_record_hang_diagnostic_phase = Mock()

    executor._enqueue_responses([])

    assert executor._maybe_record_hang_diagnostic_phase.call_args_list == [
        call("enqueueing_responses"),
        call("response_enqueue_skipped"),
    ]


def test_create_hang_diagnostics_registers_status_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HANG_DIAGNOSTICS_ENV, "1")
    detector = HangDetector()
    event = Mock()
    event_factory = Mock(return_value=event)

    diagnostics, forward_event = create_executor_hang_diagnostics(
        rank=3,
        status_provider_registry=detector,
        event_factory=event_factory,
    )

    assert diagnostics is not None
    assert forward_event is event
    event_factory.assert_called_once_with(enable_timing=False)
    assert diagnostics.get_status_dump in detector._status_providers
    assert detector._status_providers.index(diagnostics.get_snapshot_dump) < (
        detector._status_providers.index(diagnostics.get_status_dump)
    )


def test_create_hang_diagnostics_tolerates_event_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HANG_DIAGNOSTICS_ENV, "1")
    detector = HangDetector()
    event_factory = Mock(side_effect=RuntimeError("CUDA context unavailable"))

    diagnostics, forward_event = create_executor_hang_diagnostics(
        rank=3,
        status_provider_registry=detector,
        event_factory=event_factory,
    )

    assert diagnostics is not None
    assert forward_event is None
    event_factory.assert_called_once_with(enable_timing=False)
    assert diagnostics.get_status_dump in detector._status_providers


@pytest.mark.parametrize(
    ("event_complete", "expected"),
    [(False, "pending"), (True, "complete")],
)
def test_hang_diagnostics_reports_latest_phase_and_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
    event_complete: bool,
    expected: str,
) -> None:
    monkeypatch.setenv(HANG_DIAGNOSTICS_ENV, "1")
    diagnostics = ExecutorHangDiagnostics.from_environment(rank=3)
    assert diagnostics is not None
    event = Mock()
    event.query.return_value = event_complete

    diagnostics.record(
        "forward_returned",
        iteration=17,
        details="active_request_ids=[42]",
        forward_completion_event=event,
    )
    # The two registered providers are complementary, not redundant: the
    # snapshot carries the phase metadata without touching CUDA, and the status
    # dump adds only the completion state its query can stall on.
    snapshot = diagnostics.get_snapshot_dump()
    status = diagnostics.get_status_dump()

    assert "rank=3" in snapshot
    assert "iteration=17" in snapshot
    assert "phase=forward_returned" in snapshot
    assert "active_request_ids=[42]" in snapshot

    assert "rank=3" in status
    assert f"forward_cuda_completion={expected}" in status
    assert "phase=" not in status
    assert "iteration=" not in status


def test_hang_diagnostics_tolerates_cuda_query_failure() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=0)
    event = Mock()
    event.query.side_effect = RuntimeError("CUDA context unavailable")
    diagnostics.record("sampling", iteration=2, forward_completion_event=event)

    status = diagnostics.get_status_dump()

    assert "forward_cuda_completion=query-failed(RuntimeError: CUDA context unavailable)" in status


def test_phase_snapshot_does_not_query_cuda() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=2)
    event = Mock()
    diagnostics.record(
        "sampling", iteration=17, details="active_request_ids=[42]", forward_completion_event=event
    )
    status = diagnostics.get_snapshot_dump()
    assert "rank=2" in status
    assert "phase=sampling" in status
    assert "iteration=17" in status
    assert "active_request_ids=[42]" in status
    event.query.assert_not_called()


def test_hang_diagnostics_tolerates_cuda_record_failure() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=0)
    event = Mock()
    event.record.side_effect = RuntimeError("CUDA context unavailable")
    stream = Mock()

    recorded_event = diagnostics.record_forward_completion_event(event, stream)

    assert recorded_event is None
    event.record.assert_called_once_with(stream)


def test_hang_diagnostics_retains_event_until_explicitly_cleared() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=0)
    event = Mock()
    event.query.return_value = False
    diagnostics.record("forward_returned", iteration=2, forward_completion_event=event)

    diagnostics.record("sampling", iteration=2)
    assert "forward_cuda_completion=pending" in diagnostics.get_status_dump()

    diagnostics.record(
        "scheduling",
        iteration=3,
        forward_completion_event=None,
    )
    assert "forward_cuda_completion=not-recorded" in diagnostics.get_status_dump()


def test_report_file_failure_does_not_prevent_hang_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("TLLM_HANG_DIAGNOSTICS_DIR", str(blocker / "reports"))
    callback = Mock()
    detector = HangDetector(on_detected=callback)
    asyncio.run(detector._report_hang())
    callback.assert_called_once_with()


def test_failed_file_stack_dump_keeps_logging_before_status_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TLLM_HANG_DIAGNOSTICS_DIR", str(tmp_path))
    events = []

    def failed_dump(**_kwargs):
        raise OSError("diagnostic file unavailable")

    def status():
        events.append("status")
        return "executor state"

    monkeypatch.setattr(hang_detector_module.faulthandler, "dump_traceback", failed_dump)
    monkeypatch.setattr(
        hang_detector_module, "_best_effort_log_error", lambda _: events.append("log")
    )
    monkeypatch.setattr(hang_detector_module, "print_all_stacks", lambda: events.append("stacks"))
    callback = Mock()
    detector = HangDetector(on_detected=callback)
    detector.register_status_provider(status)
    asyncio.run(detector._report_hang())
    assert events[:3] == ["log", "stacks", "status"]
    callback.assert_called_once_with()


def test_blocked_status_provider_still_reports_to_both_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that never returns must suppress neither the file nor stderr."""
    monkeypatch.setenv("TLLM_HANG_DIAGNOSTICS_DIR", str(tmp_path))
    events = []
    monkeypatch.setattr(
        hang_detector_module, "_best_effort_log_error", lambda _: events.append("log")
    )
    monkeypatch.setattr(hang_detector_module, "print_all_stacks", lambda: events.append("stacks"))

    blocked = threading.Event()
    reported = threading.Event()

    def blocked_provider():
        reported.set()
        blocked.wait(30)
        return "never reached"

    detector = HangDetector(on_detected=Mock())
    detector.register_status_provider(blocked_provider)
    worker = threading.Thread(target=lambda: asyncio.run(detector._report_hang()), daemon=True)
    worker.start()
    try:
        assert reported.wait(10), "status provider was never queried"
        # Logging already happened, so the stall cannot cost the stderr report.
        assert events == ["log", "stacks"]
        report = next(iter(tmp_path.glob("worker-*.txt"))).read_text()
        assert "Hang detected" in report
        assert "_report_hang" in report
    finally:
        blocked.set()
        worker.join(timeout=10)
