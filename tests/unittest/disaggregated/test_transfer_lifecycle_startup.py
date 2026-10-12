# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the GPU qualification's shared startup deadline."""

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_transfer_lifecycle_gpu as qualification

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Advance the fixture's monotonic clock without real sleeps."""
    now = [100.0]

    def advance(delay: float) -> None:
        now[0] += delay

    monkeypatch.setattr(qualification.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(qualification.time, "sleep", advance)
    return now


def test_startup_gates_share_absolute_deadline(clock: list[float]) -> None:
    """Slow startup succeeds, but successive gates cannot renew its budget."""
    deadline = clock[0] + 120.0
    for ready_at in (145.0, 190.0):
        qualification._wait(
            lambda ready_at=ready_at: clock[0] >= ready_at, "startup gate", deadline=deadline
        )
        assert clock[0] == pytest.approx(ready_at, abs=0.01)

    with pytest.raises(AssertionError, match="Timed out: missing startup gate"):
        qualification._wait(lambda: False, "missing startup gate", deadline=deadline)
    assert clock[0] == pytest.approx(deadline, abs=0.01)


def test_post_start_wait_keeps_relative_timeout(clock: list[float]) -> None:
    """Ordinary phase waits retain their independent 30-second budget."""
    clock[0] = 250.0
    with pytest.raises(AssertionError, match="Timed out: transfer phase"):
        qualification._wait(lambda: False, "transfer phase")
    assert clock[0] == pytest.approx(280.0, abs=0.01)


@pytest.mark.parametrize("ready_at", [109.999, 110.0, 110.001], ids=["before", "at", "after"])
def test_readiness_must_be_observed_before_deadline(clock: list[float], ready_at: float) -> None:
    """A predicate that crosses the deadline cannot turn expiry into success."""
    deadline = 110.0

    def ready() -> bool:
        clock[0] = ready_at
        return True

    if ready_at < deadline:
        qualification._wait(ready, "startup readiness", deadline=deadline)
    else:
        with pytest.raises(AssertionError, match="Timed out: startup readiness"):
            qualification._wait(ready, "startup readiness", deadline=deadline)


def test_readiness_after_poll_sleep_cannot_bypass_deadline(clock: list[float]) -> None:
    """The next poll must reject readiness if sleeping consumed the budget."""
    deadline = clock[0] + 0.005
    with pytest.raises(AssertionError, match="Timed out: late startup readiness"):
        qualification._wait(
            lambda: clock[0] > deadline, "late startup readiness", deadline=deadline
        )


def test_supervisor_propagates_one_startup_deadline(
    clock: list[float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Both worlds and supervisor gates share a deadline set before launch."""
    expected_deadline = clock[0] + qualification._WORKER_TIMEOUT_S
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4))
    )
    monkeypatch.setattr(qualification.sys, "platform", "linux")
    monkeypatch.setattr(qualification.shutil, "which", lambda name: "/test/mpirun")

    def launch(command: list[str], **kwargs: object) -> Mock:
        assert float(command[command.index("--startup-deadline") + 1]) == expected_deadline
        role = command[command.index("--role") + 1]
        (tmp_path / f"{role}.log").write_text(
            "discarded prefix\n" + "x" * 9000 + f"\n{role} startup stalled\n"
        )
        clock[0] += 7.0
        job = Mock()
        job.poll.return_value = None
        return job

    launcher = Mock(side_effect=launch)
    cleanup = Mock()
    monkeypatch.setattr(qualification.subprocess, "Popen", launcher)
    monkeypatch.setattr(qualification, "_stop_owned_jobs", cleanup)
    waits: list[str] = []

    class StartupObserved(RuntimeError):
        """Stop the supervisor before any transfer or native runtime activity."""

    def observe_wait(
        predicate: Callable[[], bool],
        description: str,
        timeout: float = 30.0,
        *,
        deadline: float | None = None,
    ) -> None:
        assert deadline == expected_deadline
        waits.append(description)
        if len(waits) == 2:
            raise StartupObserved
        clock[0] += 40.0

    monkeypatch.setattr(qualification, "_wait", observe_wait)
    with pytest.raises(StartupObserved):
        qualification.test_transfer_lifecycle_gpu("baseline", tmp_path)

    assert launcher.call_count == 2
    assert len(waits) == 2
    cleanup.assert_called_once()
    diagnostics = capfd.readouterr().out
    assert "ctx.0.ready" in diagnostics and "gen.1.ready" in diagnostics
    assert "ctx startup stalled" in diagnostics and "gen startup stalled" in diagnostics
    assert "discarded prefix" not in diagnostics
