# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, single-host disaggregated test budgets separated at weight loading.

The first budget includes process startup and pre-load initialization. The
second starts when the last expected worker rank finishes loading its main
model and includes all remaining initialization, test execution and teardown.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Mapping
from pathlib import Path

import pytest
from pytest_timeout import Settings

_ATTEMPT_ENV = "TRTLLM_TEST_DISAGG_TIMEOUT_DIR"
_WORKER_ENV = "TRTLLM_STARTUP_EVENT_DIR"
_WEIGHT_LOADING_PHASE = "model_weight_loading"
_LOADING_OVERRIDE = "TRTLLM_TEST_WEIGHT_LOADING_TIMEOUT"
_MARKER = "disagg_weight_loading"
PLUGIN_NAME = "disagg-weight-loading-timeout"
_LOADING_BUDGET = pytest.StashKey[float]()
_WATCHDOG = pytest.StashKey["_Watchdog"]()
_SETTINGS = pytest.StashKey[Settings]()
_ReportTimeout = Callable[[pytest.Item, float], None]

# Find the submodule without importing the CUDA runtime.
_STARTUP_EVENTS_PROBE = """\
from importlib.machinery import PathFinder
from importlib.util import find_spec
import sys

# Match the installed console script's import behavior outside its bin dir.
if sys.path and sys.path[0] == "":
    sys.path.pop(0)

package = find_spec("tensorrt_llm")
sys.exit(package is None or not package.submodule_search_locations
         or PathFinder.find_spec("tensorrt_llm._startup_events",
                                 package.submodule_search_locations) is None)
"""


def require_worker_startup_events(env: Mapping[str, str]) -> None:
    """Check an opted-in worker's runtime before its console script starts.

    The worker's PATH must select Python 3 and trtllm-serve from the same
    environment. The console script remains the worker's main module so MPI
    workers can use its directory to resolve imports.

    Args:
        env: The same environment passed to the worker subprocess.

    Raises:
        RuntimeError: The worker's Python environment lacks startup phase events.
    """
    if not os.environ.get(_ATTEMPT_ENV) or not env.get(_WORKER_ENV):
        return
    probe = subprocess.run(
        ["python3", "-c", _STARTUP_EVENTS_PROBE], env=env, timeout=120, check=False
    )
    if probe.returncode:
        raise RuntimeError(
            "Disaggregated weight-loading supervision requires startup phase events. "
            "Install a TensorRT-LLM wheel built from this checkout."
        )


def configure_weight_loading(workers: list[tuple[dict[str, str], int]]) -> ReadinessBudget | None:
    """Wire up every worker before launching any process.

    Worker environments may be snapshots made before pytest setup, so the
    attempt directory is read from the parent, rather than from those snapshots.
    The directory must be visible to every rank on the same host.

    Args:
        workers: A pair of environment and world size per worker. Each
            environment is updated in place with its rank-event directory.

    Returns:
        The readiness budget for this launch, or None without supervision.

    Raises:
        RuntimeError: A cluster launch has already been configured in this attempt.
    """
    attempt = os.environ.get(_ATTEMPT_ENV)
    if not attempt:
        return None
    root = Path(attempt)
    if (root / "expected.json").exists():
        raise RuntimeError("Two-stage timeout supports one cluster launch per test attempt")
    expected = []
    for index, (env, world_size) in enumerate(workers):
        if world_size < 1:
            raise ValueError("Worker world size must be positive")
        directory = root / f"worker-{index}"
        directory.mkdir()
        env[_WORKER_ENV] = str(directory)
        expected.extend(
            f"{directory.name}/{_WEIGHT_LOADING_PHASE}.rank-{rank}.done"
            for rank in range(world_size)
        )
    if not expected:
        raise ValueError("At least one worker rank is required")
    temporary = root / "expected.tmp"
    temporary.write_text(json.dumps(expected), encoding="utf-8")
    temporary.replace(root / "expected.json")
    budgets = json.loads((root / "budgets.json").read_text(encoding="utf-8"))
    return ReadinessBudget(root, budgets["loading"], budgets["execution"], budgets["started"])


class _RankCompletions:
    def __init__(self, root: Path, not_before: float = 0) -> None:
        self._root = root
        self._not_before = not_before
        self._paths: list[Path] | None = None
        self._completed_at: float | None = None

    def _expected_paths(self) -> list[Path]:
        if self._paths is None:
            try:
                expected = json.loads((self._root / "expected.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                return []  # Worker launch has not been configured yet.
            self._paths = [self._root / name for name in expected]
        return self._paths

    def completed_at(self) -> float | None:
        if self._completed_at is None:
            paths = self._expected_paths()
            if not paths:
                return None
            try:
                timestamps = [float(path.read_text(encoding="utf-8")) for path in paths]
            except FileNotFoundError:
                return None
            if not all(math.isfinite(stamp) and stamp >= self._not_before for stamp in timestamps):
                raise ValueError(
                    "Weight-loading events must have valid timestamps from this attempt"
                )
            self._completed_at = max(timestamps)
        return self._completed_at

    def missing(self) -> str:
        paths = self._expected_paths()
        if not paths:
            return "worker launch has not been configured"
        return ", ".join(str(path.relative_to(self._root)) for path in paths if not path.exists())


class ReadinessBudget:
    """Bound launcher waits while leaving time for normal worker cleanup."""

    def __init__(self, root: Path, loading: float, execution: float, started: float) -> None:
        """Initialize from the test's watchdog budgets.

        Args:
            root: Directory containing the expected rank events.
            loading: Hard loading budget in seconds.
            execution: Hard execution budget in seconds.
            started: Monotonic start of the loading budget.
        """
        self._completions = _RankCompletions(root, started)
        self.safety_timeout = math.ceil(loading + execution)
        self._loading_wait = loading - min(60.0, loading / 4)
        self._loading_deadline = started + self._loading_wait

    def _loading_remaining(self, now: float) -> float:
        remaining = self._loading_deadline - now
        if remaining <= 0:
            raise TimeoutError(
                f"Weight loading did not finish within {self._loading_wait:g}s "
                f"before the cleanup window. Missing events: {self._completions.missing()}"
            )
        return remaining

    def limit_address_timeout(self, timeout: float, now: float) -> float:
        """Cap the coordinator's address wait by the remaining loading time.

        Args:
            timeout: The coordinator's existing address timeout in seconds.
            now: Current host monotonic timestamp before starting any workers.
        """
        return min(timeout, self._loading_remaining(now))

    def remaining(self, wait_start: float, timeout: float, now: float) -> float | None:
        """Check the deadline and return seconds remaining, or None while loading.

        Args:
            wait_start: Host monotonic timestamp when readiness polling started.
            timeout: Readiness budget in seconds.
            now: Current host monotonic timestamp.

        Raises:
            TimeoutError: The launcher exceeded its loading or readiness wait.
        """
        completed_at = self._completions.completed_at()
        if completed_at is None:
            self._loading_remaining(now)
            return None
        remaining = max(wait_start, completed_at) + timeout - now
        if remaining <= 0:
            raise TimeoutError(
                f"Cluster did not become ready within {timeout}s after weight loading"
            )
        return remaining

    def require_complete(self) -> None:
        """Fail a ready cluster that did not publish all loading events."""
        if self._completions.completed_at() is None:
            raise RuntimeError(
                f"Cluster became ready without weight-loading events: {self._completions.missing()}. "
                "Check worker ranks and use a TensorRT-LLM wheel built from this checkout."
            )


class PhaseBudget:
    """Track two wall-clock budgets without adding parallel rank durations."""

    def __init__(self, root: Path, loading: float, execution: float, start: float) -> None:
        if not all(math.isfinite(value) and value > 0 for value in (loading, execution)):
            raise ValueError("Both disagg timeout budgets must be finite and positive")
        self.root = root
        self.loading = loading
        self.execution = execution
        self.start = start
        self.loaded_at: float | None = None
        self.completions = _RankCompletions(root, start)

    def remaining(self, now: float) -> tuple[str, float]:
        """Return the phase and remaining seconds, using the last rank's event."""
        if self.loaded_at is None:
            completed = self.completions.completed_at()
            # Late events must not rescue an already expired loading phase.
            if completed is not None and self.start <= completed <= self.start + self.loading:
                self.loaded_at = completed
        if self.loaded_at is None:
            return "weight_loading", self.start + self.loading - now
        return "post_loading", self.loaded_at + self.execution - now


class _Watchdog:
    def __init__(
        self, item: pytest.Item, settings: Settings, loading: float, report: _ReportTimeout | None
    ) -> None:
        self.item = item
        self.settings = settings
        self.report = report
        output = item.config.getoption("output_dir", default=None)
        if output:
            output = str(Path(output).resolve())
            Path(output).mkdir(parents=True, exist_ok=True)
        # A rank that is still exiting can publish while the directory is removed.
        self.directory = tempfile.TemporaryDirectory(
            prefix="disagg-timeout-", dir=output, ignore_cleanup_errors=True
        )
        root = Path(self.directory.name).resolve()
        self.budget = PhaseBudget(root, loading, settings.timeout, time.monotonic())
        (root / "budgets.json").write_text(
            json.dumps(
                {"loading": loading, "execution": settings.timeout, "started": self.budget.start}
            ),
            encoding="utf-8",
        )
        self.previous_env = os.environ.get(_ATTEMPT_ENV)
        os.environ[_ATTEMPT_ENV] = str(root)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="disagg-timeout", daemon=True)

    def _expire(self, phase: str) -> None:
        import pytest_timeout

        if not self.settings.disable_debugger_detection and pytest_timeout.is_debugging():
            return
        loading = phase == "weight_loading"
        timeout = self.budget.loading if loading else self.budget.execution
        try:
            print(f"[disagg-timeout] {phase} timeout: {self.item.nodeid}", flush=True)
            if self.report is not None:
                self.report(self.item, timeout)
            if loading and (missing := self.budget.completions.missing()):
                print(f"[disagg-timeout] Missing weight-loading events: {missing}", flush=True)
        finally:
            pytest_timeout.timeout_timer(self.item, self.settings._replace(timeout=timeout))

    def _log_phase(self, phase: str, remaining: float, now: float) -> None:
        loading_elapsed = (self.budget.loaded_at or now) - self.budget.start
        print(
            f"[disagg-timeout] phase={phase}, remaining={remaining:.1f}s, "
            f"loading_elapsed={loading_elapsed:.1f}s, "
            f"loading_budget={self.budget.loading}s, execution_budget={self.budget.execution}s",
            flush=True,
        )

    def _run(self) -> None:
        previous_phase = None
        next_log = 0.0
        while True:
            now = time.monotonic()
            try:
                phase, remaining = self.budget.remaining(now)
            except (OSError, ValueError, TypeError) as error:
                print(f"[disagg-timeout] Invalid loading state: {error}", flush=True)
                if not self.stop.is_set():
                    self._expire("weight_loading")
                return
            if phase != previous_phase or (not self.stop.is_set() and now >= next_log):
                self._log_phase(phase, remaining, now)
                previous_phase = phase
                next_log = now + 30
            if self.stop.is_set():
                return
            if remaining <= 0:
                self._expire(phase)
                return
            self.stop.wait(min(0.1, remaining))

    def cancel(self) -> None:
        self.stop.set()
        self.thread.join()
        if self.previous_env is None:
            os.environ.pop(_ATTEMPT_ENV, None)
        else:
            os.environ[_ATTEMPT_ENV] = self.previous_env
        self.directory.cleanup()


def can_supervise(item: pytest.Item, settings: Settings) -> bool:
    """Whether the marked item uses pytest-timeout's whole-item thread mode.

    Args:
        item: The pytest test item.
        settings: Its resolved pytest-timeout settings.
    """
    uses_thread = settings.method == "thread" or (
        settings.method == "signal" and threading.current_thread() is not threading.main_thread()
    )
    return (
        item.get_closest_marker(_MARKER) is not None
        and item.config.pluginmanager.hasplugin(PLUGIN_NAME)
        and uses_thread
        and not settings.func_only
    )


def is_supervised(item: pytest.Item) -> bool:
    """Whether the disagg watchdog owns the pytest test item's timeout."""
    return item.get_closest_marker(_MARKER) is not None and _SETTINGS in item.stash


class DisaggTimeoutPlugin:
    """Replace pytest-timeout's timer only on explicitly marked tests."""

    def __init__(self, report: _ReportTimeout | None) -> None:
        self._report = report

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        if _SETTINGS not in item.stash:
            return
        # pytest-rerunfailures invokes setup once per attempt.
        self._cancel(item)
        watchdog = _Watchdog(item, item.stash[_SETTINGS], item.stash[_LOADING_BUDGET], self._report)
        item.stash[_WATCHDOG] = watchdog
        watchdog.thread.start()

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_teardown(self, item: pytest.Item) -> Generator[None, None, None]:
        try:
            return (yield)
        finally:
            self._cancel(item)

    def _cancel(self, item: pytest.Item) -> None:
        watchdog = item.stash.get(_WATCHDOG, None)
        if watchdog is not None:
            del item.stash[_WATCHDOG]
            watchdog.cancel()

    @pytest.hookimpl(tryfirst=True, optionalhook=True)
    def pytest_timeout_set_timer(self, item: pytest.Item, settings: Settings) -> bool | None:
        if not can_supervise(item, settings):
            return None
        item.stash[_SETTINGS] = settings
        return True

    @pytest.hookimpl(tryfirst=True, optionalhook=True)
    def pytest_timeout_cancel_timer(self, item: pytest.Item) -> bool | None:
        if _SETTINGS not in item.stash:
            return None
        self._cancel(item)
        return True

    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        override = os.environ.get(_LOADING_OVERRIDE)
        for item in items:
            marker = item.get_closest_marker(_MARKER)
            if marker is None:
                continue
            if not config.pluginmanager.hasplugin("timeout"):
                raise pytest.UsageError(f"{item.nodeid}: {_MARKER} requires pytest-timeout")
            if len(marker.args) != 1 or marker.kwargs:
                raise pytest.UsageError(f"{item.nodeid}: use {_MARKER}(seconds)")
            seconds = _positive_seconds(item, _MARKER, marker.args[0])
            if override is not None:
                seconds = _positive_seconds(item, _LOADING_OVERRIDE, override)
            item.stash[_LOADING_BUDGET] = seconds


def _positive_seconds(item: pytest.Item, setting: str, value: object) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError
    except (OverflowError, ValueError) as error:
        raise pytest.UsageError(
            f"{item.nodeid}: {setting} expects positive seconds, got {value!r}"
        ) from error
    return seconds


def register(config: pytest.Config, report_timeout: _ReportTimeout | None = None) -> None:
    """Register once from the integration conftest.

    Args:
        config: Pytest session configuration.
        report_timeout: Optional callback for evidence and traceback reporting.
    """
    config.addinivalue_line(
        "markers", "disagg_weight_loading(seconds): separate loading and post-loading budgets"
    )
    if not config.pluginmanager.hasplugin(PLUGIN_NAME):
        config.pluginmanager.register(DisaggTimeoutPlugin(report_timeout), PLUGIN_NAME)
