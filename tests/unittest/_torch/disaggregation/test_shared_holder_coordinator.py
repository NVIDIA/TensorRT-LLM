# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-loop integration of the shared holder, independent of native transfer."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import NoopDisaggCoordinator
from tensorrt_llm._torch.disaggregation.transfer_manager import TransferManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _holder() -> Mock:
    """Make the holder boundary double; ledger behavior has its own contract tests.

    Returns:
        A holder with explicit progress and cleanup observations.
    """
    holder = Mock(spec=TransferManager)
    holder.advance.return_value = ()
    holder.take_domain_failures.return_value = ()
    holder.pending.return_value = True
    holder.shutdown_timeout_s = 5.0
    holder.shutdown_status.return_value = SimpleNamespace(complete=False)
    return holder


@pytest.mark.parametrize("native", [False, True])
def test_loop_head_progresses_shared_holder_and_defers_request_updates(native: bool) -> None:
    """Both coordinator variants observe shared work without applying scheduler state."""
    coordinator = CoordinatorHarness().coordinator if native else NoopDisaggCoordinator()
    holder = _holder()
    update = object()
    holder.advance.return_value = (update,)
    coordinator.attach_shared_transfer_manager(holder)

    coordinator.handle_errors_synced()

    holder.advance.assert_called_once()
    assert coordinator.take_shared_updates() == (update,)
    assert coordinator.take_shared_updates() == ()
    assert coordinator.has_pending_shared_transfers()


def test_holder_attachment_cannot_replace_retained_resources() -> None:
    """Repeated attachment is harmless; replacing a rooted holder is refused."""
    coordinator = NoopDisaggCoordinator()
    holder = _holder()
    coordinator.attach_shared_transfer_manager(holder)
    coordinator.attach_shared_transfer_manager(holder)
    with pytest.raises(RuntimeError, match="already attached"):
        coordinator.attach_shared_transfer_manager(_holder())


@pytest.mark.parametrize("native", [False, True])
def test_lender_domain_faults_reach_orchestration_without_request_actions(native: bool) -> None:
    """Recovery receives the opaque failed domain; the holder does not own requests."""
    coordinator = CoordinatorHarness().coordinator if native else NoopDisaggCoordinator()
    holder = _holder()
    failure = object()
    holder.take_domain_failures.return_value = (failure,)
    coordinator.attach_shared_transfer_manager(holder)
    assert coordinator.take_shared_domain_failures() == (failure,)
    holder.take_domain_failures.assert_called_once_with()


def test_shared_idle_uses_wake_and_bounded_heartbeat() -> None:
    """Shared-only idle progress delegates to the holder's notification wait."""
    coordinator = NoopDisaggCoordinator()
    holder = _holder()
    coordinator.attach_shared_transfer_manager(holder)
    coordinator.pace_idle()
    holder.wait_for_progress.assert_called_once_with()


def test_cleanup_initiation_exception_still_wakes_waiters_and_runs_safe_cleanup() -> None:
    """The actual executor cleanup wakes response waiters before a fallible holder hook."""
    executor = object.__new__(PyExecutor)
    executor.response_cv = threading.Condition()
    executor.is_shutdown = False
    executor.shutdown_event = threading.Event()
    executor.num_micro_batches = 0
    executor._gpu_keepalive = Mock()
    coordinator = Mock()

    def fail_after_notification() -> None:
        """Verify ordering at the throwing integration boundary."""
        assert executor.is_shutdown and executor.shutdown_event.is_set()
        raise RuntimeError("cleanup initiation failed")

    coordinator.begin_shared_shutdown.side_effect = fail_after_notification
    executor._disagg_coordinator = coordinator
    executor._executor_loop_cleanup()
    executor._gpu_keepalive.close.assert_called_once_with()
    assert executor.shutdown_event.is_set()


def test_shutdown_keeps_shared_only_executor_running_until_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-native executor must not stop or wait indefinitely while a holder drains."""
    executor = object.__new__(PyExecutor)
    executor.is_shutdown = True
    executor.active_requests = []
    executor.waiting_queue = []
    executor.kv_connector_manager = None
    coordinator = NoopDisaggCoordinator()
    holder = _holder()
    coordinator.attach_shared_transfer_manager(holder)
    executor._disagg_coordinator = coordinator
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.orchestration.coordinator.time.monotonic",
        lambda: 10.0,
    )

    coordinator.begin_shared_shutdown()
    holder.begin_shutdown.assert_called_once_with(15.0)
    assert executor._has_pending_connector_transfers()
    assert not executor.should_stop_processing
    assert not coordinator.shared_shutdown_complete()

    holder.pending.return_value = False
    holder.shutdown_status.return_value = SimpleNamespace(complete=True)
    assert not executor._has_pending_connector_transfers()
    assert executor.should_stop_processing
    assert coordinator.shared_shutdown_complete()


def test_unattached_shared_hooks_preserve_no_native_behavior() -> None:
    """Existing no-native paths need no holder or additional teardown action."""
    coordinator = NoopDisaggCoordinator()
    coordinator.begin_shared_shutdown()
    coordinator.handle_errors_synced()
    coordinator.pace_idle()
    assert not coordinator.has_pending_shared_transfers()
    assert coordinator.shared_shutdown_complete()
    assert coordinator.take_shared_updates() == ()
    assert coordinator.take_shared_domain_failures() == ()


@pytest.mark.parametrize(
    "connector_state,shared_pending,expected",
    [
        ("none", True, True),
        ("loads", False, True),
        ("inflight", False, True),
        ("empty", True, True),
        ("empty", False, False),
        ("none", False, False),
    ],
)
def test_pending_transfer_predicate_combines_connector_and_shared_work(
    connector_state: str, shared_pending: bool, expected: bool
) -> None:
    """Stop, idle and control-drain checks include either source of retained work."""
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = None
    executor.async_transfer_manager = None
    if connector_state != "none":
        executor.kv_connector_manager = Mock()
        executor.kv_connector_manager.has_pending_loads.return_value = connector_state == "loads"
        executor.async_transfer_manager = Mock()
        executor.async_transfer_manager.has_any_inflight_requests.return_value = (
            connector_state == "inflight"
        )
    coordinator = NoopDisaggCoordinator()
    holder = _holder()
    holder.pending.return_value = shared_pending
    coordinator.attach_shared_transfer_manager(holder)
    executor._disagg_coordinator = coordinator

    assert executor._has_pending_connector_transfers() is expected
