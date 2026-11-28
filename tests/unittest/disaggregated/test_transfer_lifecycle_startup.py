# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for GPU qualification clock margins and independent wait limits."""

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_transfer_lifecycle_gpu as qualification

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("case", "ctx_timeout", "gen_timeout"),
    [
        ("baseline", 120.0, 120.0),
        ("cancel_before_publication", 120.0, 120.0),
        ("cancel_late_aux", 120.0, 120.0),
        ("cancel_late_kv", 120.0, 120.0),
        ("timeout_late_aux", 30.0, 30.0),
        ("source_fatal", 30.0, 90.0),
        ("destination_fatal", 90.0, 30.0),
    ],
)
def test_only_expiry_cases_use_short_clocks(
    case: str, ctx_timeout: float, gen_timeout: float
) -> None:
    """Short real clocks preserve cancellation margins and fatal-peer isolation.

    Args:
        case: Distinct lifecycle path selected by the GPU supervisor.
        ctx_timeout: Expected CTX request/grace duration in seconds.
        gen_timeout: Expected GEN request/grace duration in seconds.
    """
    assert qualification._case_timeout_s(case) == min(ctx_timeout, gen_timeout)
    assert qualification._endpoint_timeout_s(case, "ctx") == ctx_timeout
    assert qualification._endpoint_timeout_s(case, "gen") == gen_timeout


def test_short_clocks_do_not_reduce_protective_wait_limits() -> None:
    """Expiry acceleration must not shorten startup, observation or outer rescue."""
    assert qualification._PHASE_TIMEOUT_S == 30.0
    assert qualification._TIMEOUT_S == 120.0
    assert qualification._WORKER_TIMEOUT_S == 600.0
    for test in (
        qualification.test_transfer_lifecycle_gpu,
        qualification.test_context_first_bf16_lifecycle_gpu,
    ):
        timeout = next(mark for mark in test.pytestmark if mark.name == "timeout")
        assert timeout.args == (1080.0,)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Advance the fixture's monotonic clock without real sleeps."""
    now = [100.0]

    def advance(delay: float) -> None:
        now[0] += delay

    monkeypatch.setattr(qualification.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(qualification.time, "sleep", advance)
    return now


@pytest.mark.parametrize("case", ["cancel_late_kv", "source_fatal", "timeout_late_aux"])
@pytest.mark.parametrize("offset", [-0.001, 0.0, 0.001], ids=["below", "at", "above"])
def test_transition_requires_strict_preparation_margin(
    clock: list[float], case: str, offset: float
) -> None:
    """Reject preparation that consumes either case's strict pre-expiry reserve.

    Args:
        clock: Controlled supervisor monotonic clock.
        case: Lifecycle path selecting the real transfer clock.
        offset: Seconds above or below the strict preparation boundary.
    """
    timeout = 120.0 if case == "cancel_late_kv" else 30.0
    clocks = [{"request_deadline": clock[0] + timeout / 2 + offset, "timeout": timeout}] * 2
    if case == "source_fatal":
        clocks[1] = {"request_deadline": clock[0] + 75.0, "timeout": 90.0}
    if offset > 0:
        qualification._assert_transition_budget(case, clocks)
    else:
        with pytest.raises(AssertionError):
            qualification._assert_transition_budget(case, clocks)


@pytest.mark.parametrize(("timeout", "skew_limit"), [(30.0, 15.0), (120.0, 60.0)])
@pytest.mark.parametrize("offset", [-0.001, 0.0, 0.001], ids=["below", "at", "above"])
def test_both_request_timeouts_precede_earliest_fatal(
    clock: list[float],
    monkeypatch: pytest.MonkeyPatch,
    timeout: float,
    skew_limit: float,
    offset: float,
) -> None:
    """Keep timeout ordering strict with both accelerated and original clocks.

    Args:
        clock: Controlled supervisor monotonic clock.
        monkeypatch: Scope the fixture's expiry-clock calibration.
        timeout: Real request/grace duration in seconds.
        skew_limit: Largest excluded difference between endpoint deadlines.
        offset: Seconds above or below the strict skew boundary.
    """
    monkeypatch.setattr(qualification, "_EXPIRY_TIMEOUT_S", timeout)
    clocks = [
        {"request_deadline": clock[0] + timeout, "timeout": timeout},
        {"request_deadline": clock[0] + timeout + skew_limit + offset, "timeout": timeout},
    ]
    if offset < 0:
        qualification._assert_transition_budget("timeout_late_aux", clocks)
    else:
        with pytest.raises(AssertionError):
            qualification._assert_transition_budget("timeout_late_aux", clocks)


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


@pytest.mark.parametrize("survivor_exits", [False, True], ids=["alive", "exits_during_observation"])
def test_survivor_must_remain_alive_until_target_exit_is_observed(
    clock: list[float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    survivor_exits: bool,
) -> None:
    """Exercise the real supervisor across a mocked passive process-exit wait.

    Args:
        clock: Controlled supervisor monotonic clock and polling sleeps.
        tmp_path: Native-process evidence and supervisor gate directory.
        monkeypatch: Replace inaccessible runtime and process boundaries.
        survivor_exits: Whether the independent peer dies during target observation.
    """
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4))
    )
    monkeypatch.setattr(qualification.sys, "platform", "linux")
    monkeypatch.setattr(qualification.shutil, "which", lambda name: "/test/mpirun")
    jobs = {role: Mock(returncode=None) for role in ("ctx", "gen")}
    for role, job in jobs.items():
        job.poll.side_effect = lambda job=job: job.returncode
        for rank in range(2):
            qualification._record(tmp_path, f"{role}.{rank}.ready")
        qualification._record(tmp_path, f"{role}.1.blocked")
        for phase in ("allocated", "retained", "in_doubt"):
            qualification._record(tmp_path, f"{role}.{phase}")
        timeout = qualification._endpoint_timeout_s("source_fatal", role)
        qualification._record(
            tmp_path, f"{role}.prepared", request_deadline=clock[0] + timeout, timeout=timeout
        )
    qualification._record(tmp_path, "ctx.fatal")
    qualification._record(tmp_path, "gen.survived", resources_retained=True)

    def launch(command: list[str], **kwargs: object) -> Mock:
        """Return the endpoint launcher selected by the real supervisor.

        Args:
            command: Actual MPI launcher command.
            **kwargs: Native subprocess options, unused by this process double.

        Returns:
            Endpoint process whose liveness remains observable.
        """
        return jobs[command[command.index("--role") + 1]]

    def target_exited(timeout: float) -> int:
        """Complete the target launcher wait without advancing the native ranks.

        Args:
            timeout: Supervisor's protective launcher wait limit.

        Returns:
            Nonzero fatal exit code.
        """
        assert timeout == 15
        jobs["ctx"].returncode = 1
        return 1

    cleanup = Mock()

    def observe_target_exit(directory: Path, role: str) -> None:
        """Let the independent peer exit while both target ranks are observed.

        Args:
            directory: Supervisor evidence directory.
            role: Fatal target, not the independent surviving endpoint.
        """
        assert directory == tmp_path and role == "ctx"
        cleanup.assert_not_called()
        assert jobs["gen"].returncode is None
        clock[0] += 15.0
        if survivor_exits:
            jobs["gen"].returncode = 1

    observer = Mock(side_effect=observe_target_exit)
    jobs["ctx"].wait.side_effect = target_exited
    monkeypatch.setattr(qualification.subprocess, "Popen", launch)
    monkeypatch.setattr(qualification, "_wait_for_rank_exit", observer)
    monkeypatch.setattr(qualification, "_stop_owned_jobs", cleanup)
    try:
        if survivor_exits:
            with pytest.raises(
                AssertionError,
                match="independent peer exited before target exit observation completed",
            ):
                qualification._run_transfer_lifecycle_gpu("source_fatal", tmp_path, False)
        else:
            qualification._run_transfer_lifecycle_gpu("source_fatal", tmp_path, False)
    finally:
        observer.assert_called_once_with(tmp_path, "ctx")
        cleanup.assert_called_once_with(jobs, tmp_path)


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
