# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, single-host disaggregated test budgets separated at weight loading.

The first budget includes process startup and pre-load initialization. The
second starts when the last expected worker rank finishes loading its main
model and includes all remaining initialization, test execution and teardown.
"""

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
from pytest_timeout import Settings

_ATTEMPT_ENV = "TRTLLM_TEST_DISAGG_TIMEOUT_DIR"
_WORKER_ENV = "TRTLLM_TEST_WEIGHT_LOADING_DIR"
_WATCHDOG = pytest.StashKey["_Watchdog"]()
_BUDGETS = pytest.StashKey[tuple[Settings, float]]()


def configure_weight_loading(workers: list[tuple[dict[str, str], int]]) -> float | None:
    """Configure every worker before launching any process; return a safety limit.

    Worker environments may be snapshots made before pytest setup, so the
    attempt directory is read from the parent, rather than from those snapshots.
    The directory must be visible to every rank on the same host.
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
        expected.extend(f"worker-{index}/rank-{rank}.done" for rank in range(world_size))
    if not expected:
        raise ValueError("At least one worker rank is required")
    temporary = root / "expected.tmp"
    temporary.write_text(json.dumps(expected), encoding="utf-8")
    temporary.replace(root / "expected.json")
    budgets = json.loads((root / "budgets.json").read_text(encoding="utf-8"))
    return budgets["loading"] + budgets["execution"]


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

    def remaining(self, now: float) -> tuple[str, float]:
        """Return the phase and remaining seconds, using the last rank's event."""
        if self.loaded_at is None:
            manifest = self.root / "expected.json"
            if manifest.exists():
                expected = json.loads(manifest.read_text(encoding="utf-8"))
                paths = [self.root / name for name in expected]
                if paths and all(path.exists() for path in paths):
                    timestamps = [float(path.read_text(encoding="utf-8")) for path in paths]
                    if not all(
                        math.isfinite(stamp) and self.start <= stamp for stamp in timestamps
                    ):
                        raise ValueError("Weight-loading events must belong to this attempt")
                    completed = max(timestamps)
                    # Late events must not rescue an already expired loading phase.
                    if completed <= self.start + self.loading:
                        self.loaded_at = completed
        if self.loaded_at is None:
            return "weight_loading", self.start + self.loading - now
        return "post_loading", self.loaded_at + self.execution - now


class _Watchdog:
    def __init__(self, item: pytest.Item, settings: Settings, loading: float) -> None:
        self.item = item
        self.settings = settings
        output = item.config.getoption("output_dir", default=None)
        if output:
            output = str(Path(output).resolve())
            Path(output).mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="disagg-timeout-", dir=output)
        root = Path(self.directory.name).resolve()
        self.budget = PhaseBudget(root, loading, settings.timeout, time.monotonic())
        (root / "budgets.json").write_text(
            json.dumps({"loading": loading, "execution": settings.timeout}), encoding="utf-8"
        )
        self.previous_env = os.environ.get(_ATTEMPT_ENV)
        os.environ[_ATTEMPT_ENV] = str(root)
        self.stop = threading.Event()
        self.setup_seen = False
        self.thread = threading.Thread(target=self._run, name="disagg-timeout", daemon=True)

    def _expire(self, phase: str) -> None:
        import pytest_timeout

        timeout = self.budget.loading if phase == "weight_loading" else self.budget.execution
        reporter = self.item.config.pluginmanager.getplugin("periodic_junit")
        if reporter is not None:
            reporter._dump_hang(self.item.nodeid)
        evidence = self.item.config.pluginmanager.getplugin("trtllm_pytest_timeout_evidence")
        if evidence is not None:
            evidence.emit_timeout(self.item, timeout)
        print(f"[disagg-timeout] {phase} timeout: {self.item.nodeid}", flush=True)
        pytest_timeout.timeout_timer(self.item, self.settings._replace(timeout=timeout))

    def _run(self) -> None:
        import pytest_timeout

        previous_phase = None
        next_log = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            try:
                phase, remaining = self.budget.remaining(now)
            except (OSError, ValueError, TypeError) as error:
                print(f"[disagg-timeout] Invalid loading state: {error}", flush=True)
                pytest_timeout.timeout_timer(self.item, self.settings)
                return
            if phase != previous_phase or now >= next_log:
                loading_elapsed = (self.budget.loaded_at or now) - self.budget.start
                print(
                    f"[disagg-timeout] phase={phase}, remaining={remaining:.1f}s, "
                    f"loading_elapsed={loading_elapsed:.1f}s, "
                    f"loading_budget={self.budget.loading}s, "
                    f"execution_budget={self.budget.execution}s",
                    flush=True,
                )
                previous_phase = phase
                next_log = now + 30
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


class DisaggTimeoutPlugin:
    """Replace pytest-timeout's timer only on explicitly marked tests."""

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        if _BUDGETS not in item.stash:
            return
        watchdog = item.stash.get(_WATCHDOG, None)
        if watchdog is None or watchdog.setup_seen:
            # pytest-rerunfailures repeats setup inside one runtest protocol.
            # pytest-timeout also cancels on exception_interact before a rerun.
            settings, loading = item.stash[_BUDGETS]
            if watchdog is not None:
                watchdog.cancel()
            watchdog = _Watchdog(item, settings, loading)
            item.stash[_WATCHDOG] = watchdog
            watchdog.thread.start()
        watchdog.setup_seen = True

    @pytest.hookimpl(tryfirst=True, optionalhook=True)
    def pytest_timeout_set_timer(self, item: pytest.Item, settings: Settings) -> bool | None:
        marker = item.get_closest_marker("disagg_weight_loading")
        if marker is None:
            return None
        if settings.func_only:
            raise pytest.UsageError("disagg_weight_loading requires timeout func_only=False")
        loading = float(os.environ.get("TRTLLM_TEST_WEIGHT_LOADING_TIMEOUT", marker.args[0]))
        item.stash[_BUDGETS] = settings, loading
        watchdog = _Watchdog(item, settings, loading)
        item.stash[_WATCHDOG] = watchdog
        watchdog.thread.start()
        return True

    @pytest.hookimpl(tryfirst=True, optionalhook=True)
    def pytest_timeout_cancel_timer(self, item: pytest.Item) -> bool | None:
        if _WATCHDOG not in item.stash:
            return None
        item.stash[_WATCHDOG].cancel()
        del item.stash[_WATCHDOG]
        return True

    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        marked = [item for item in items if item.get_closest_marker("disagg_weight_loading")]
        if marked and not config.pluginmanager.hasplugin("timeout"):
            raise pytest.UsageError("disagg_weight_loading requires pytest-timeout")


def register(config: pytest.Config) -> None:
    """Register once from the integration conftest."""
    config.addinivalue_line(
        "markers", "disagg_weight_loading(seconds): separate loading and post-loading budgets"
    )
    if not config.pluginmanager.hasplugin("disagg-weight-loading-timeout"):
        config.pluginmanager.register(DisaggTimeoutPlugin(), "disagg-weight-loading-timeout")
