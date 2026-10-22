# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the hang-traceback dumping in PeriodicJUnitXML.

These exercise the pure-Python wiring only (timeout resolution and the watchdog
that writes output-dir/hang_traceback.txt); they need no GPU or model access.

The module under test lives in the integration-test helper package, which is not
importable as a package here (its ``__init__`` pulls in torch), so it is imported
by directory the same way ``test_test_to_stage_mapping.py`` imports from
``scripts/``.
"""

import faulthandler
import os
import time
from pathlib import Path
from typing import Callable

import pytest

__extra_import_path__ = ["~/tests/integration/defs/utils"]
from periodic_junit import PeriodicJUnitXML

pytestmark = pytest.mark.cpu_only


# Bounds a stalled CI worker, not the expected latency: every wait below polls
# for a condition that normally becomes true in milliseconds.
_POLL_TIMEOUT = 20.0
_POLL_INTERVAL = 0.02


def _wait_until(predicate: Callable[[], bool], timeout: float = _POLL_TIMEOUT) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result or time.monotonic() >= deadline:
            return bool(result)
        time.sleep(_POLL_INTERVAL)


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class _Marker:
    def __init__(
        self, args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None
    ) -> None:
        self.args = args
        self.kwargs = kwargs or {}


class _Config:
    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout

    def getoption(self, name: str, default: object | None = None) -> object | None:
        return self._timeout if name == "timeout" else default


class _Item:
    def __init__(
        self,
        nodeid: str = "pkg/test_x.py::test_y",
        timeout_opt: float | None = None,
        marker: _Marker | None = None,
    ) -> None:
        self.nodeid = nodeid
        self.config = _Config(timeout_opt)
        self._marker = marker

    def get_closest_marker(self, name: str) -> _Marker | None:
        return self._marker


class _Report:
    """Minimal stand-in for a pytest TestReport."""

    def __init__(self, when: str, nodeid: str = "pkg/test_x.py::test_y") -> None:
        self.when = when
        self.nodeid = nodeid


def _close(reporter: PeriodicJUnitXML) -> None:
    """Stop the watchdog and close the sidecar file."""
    # Timer.cancel() only stops a callback that has not started yet, so capture the
    # timer before _cancel_hang_timer() drops the reference and join it -- otherwise
    # an in-flight _dump_hang() would write into the file closed below.
    timer = reporter._hang_timer
    reporter._cancel_hang_timer()
    if timer is not None:
        timer.join(_POLL_TIMEOUT)
    if reporter._hang_file is not None:
        reporter._hang_file.close()
        reporter._hang_file = None


ReporterFactory = Callable[..., PeriodicJUnitXML]


@pytest.fixture(autouse=True)
def _no_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep _setup_hang_dump() from installing process-wide signal handlers.

    These tests only need the sidecar file. Real SIGINT/SIGTERM handlers would
    outlive the test that installed them and point at a file closed on teardown.
    """
    monkeypatch.setattr(faulthandler, "register", lambda *args, **kwargs: None)


@pytest.fixture
def make_reporter(tmp_path: Path, request: pytest.FixtureRequest) -> ReporterFactory:
    """Build a reporter whose watchdog and sidecar file are cleaned up."""

    def _make(dump_hang_traceback: bool = True, **kwargs: float) -> PeriodicJUnitXML:
        reporter = PeriodicJUnitXML(
            xmlpath=os.path.join(str(tmp_path), "results.xml"),
            dump_hang_traceback=dump_hang_traceback,
            **kwargs,
        )
        if dump_hang_traceback:
            reporter._setup_hang_dump()
            request.addfinalizer(lambda: _close(reporter))
        return reporter

    return _make


def test_effective_timeout_resolution(make_reporter: ReporterFactory) -> None:
    reporter = make_reporter()
    # positional marker, keyword marker, and the global --timeout all resolve.
    assert reporter._effective_timeout(_Item(marker=_Marker(args=(90,)))) == 90.0
    assert reporter._effective_timeout(_Item(marker=_Marker(kwargs={"timeout": 45}))) == 45.0
    assert reporter._effective_timeout(_Item(timeout_opt=120)) == 120.0
    # no marker and no --timeout -> no watchdog.
    assert reporter._effective_timeout(_Item()) is None


def test_hang_dumps_traceback(make_reporter: ReporterFactory) -> None:
    reporter = make_reporter(hang_dump_fraction=0.1)
    # timeout 2s * 0.1 -> dump after ~0.2s.
    reporter.pytest_runtest_setup(_Item(timeout_opt=2.0))
    path = reporter._hang_traceback_path()
    assert _wait_until(lambda: "hang watchdog fired for pkg/test_x.py::test_y" in _read(path))
    content = _read(path)
    assert "Thread" in content or 'File "' in content


def test_subsecond_timeout_arms_before_the_kill(make_reporter: ReporterFactory) -> None:
    # pytest-timeout accepts fractional seconds, so the delay must not be floored
    # at 1s -- that would arm the watchdog only after such a test was killed.
    # Asserting on the armed interval keeps this independent of worker load.
    reporter = make_reporter(hang_dump_fraction=0.5)
    reporter.pytest_runtest_setup(_Item(timeout_opt=0.5))
    assert reporter._hang_timer.interval == pytest.approx(0.25)
    assert reporter._hang_timer.interval < 0.5
    assert _wait_until(lambda: "hang watchdog fired" in _read(reporter._hang_traceback_path()))


def test_completed_test_is_not_dumped(make_reporter: ReporterFactory) -> None:
    # The teardown report arrives after every fixture finalizer, so it must disarm
    # the watchdog; nothing may be dumped for an item that already finished. The
    # timeout is far longer than this test body, so the timer cannot fire while the
    # reports are fed in even on a heavily loaded worker.
    reporter = make_reporter(hang_dump_fraction=0.5)
    reporter.pytest_runtest_setup(_Item(timeout_opt=600.0))
    timer = reporter._hang_timer
    assert timer is not None

    for when in ("setup", "call"):
        reporter.pytest_runtest_logreport(_Report(when))
        assert reporter._hang_timer is timer  # armed through setup, call, teardown

    reporter.pytest_runtest_logreport(_Report("teardown"))
    assert reporter._hang_timer is None
    # cancel() is what actually prevents the dump, so assert the timer thread died
    # rather than waiting out the (deliberately long) interval.
    assert _wait_until(lambda: not timer.is_alive())
    assert os.path.getsize(reporter._hang_traceback_path()) == 0


def test_no_watchdog_without_timeout(make_reporter: ReporterFactory) -> None:
    reporter = make_reporter()
    reporter.pytest_runtest_setup(_Item())  # no timeout -> nothing armed
    assert reporter._hang_timer is None


def test_hang_traceback_off_by_default(make_reporter: ReporterFactory) -> None:
    # --periodic-hang-traceback defaults to False; the reporter must then arm no
    # watchdog and leave no sidecar file behind.
    reporter = make_reporter(dump_hang_traceback=False)
    assert reporter.dump_hang_traceback is False
    reporter.pytest_runtest_setup(_Item(timeout_opt=1.0))
    assert reporter._hang_timer is None
    assert reporter._hang_file is None
    assert not os.path.exists(reporter._hang_traceback_path())
