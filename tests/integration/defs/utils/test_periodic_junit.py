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
"""

import os
import time

from .periodic_junit import PeriodicJUnitXML


class _Marker:
    def __init__(self, args=(), kwargs=None):
        self.args = args
        self.kwargs = kwargs or {}


class _Config:
    def __init__(self, timeout=None):
        self._timeout = timeout

    def getoption(self, name, default=None):
        return self._timeout if name == "timeout" else default


class _Item:
    def __init__(self, nodeid="pkg/test_x.py::test_y", timeout_opt=None, marker=None):
        self.nodeid = nodeid
        self.config = _Config(timeout_opt)
        self._marker = marker

    def get_closest_marker(self, name):
        return self._marker


def _make(tmp_path, **kwargs):
    reporter = PeriodicJUnitXML(
        xmlpath=os.path.join(tmp_path, "results.xml"), dump_hang_traceback=True, **kwargs
    )
    reporter._setup_hang_dump()
    return reporter


def test_effective_timeout_resolution(tmp_path):
    reporter = _make(str(tmp_path))
    # positional marker, keyword marker, and the global --timeout all resolve.
    assert reporter._effective_timeout(_Item(marker=_Marker(args=(90,)))) == 90.0
    assert reporter._effective_timeout(_Item(marker=_Marker(kwargs={"timeout": 45}))) == 45.0
    assert reporter._effective_timeout(_Item(timeout_opt=120)) == 120.0
    # no marker and no --timeout -> no watchdog.
    assert reporter._effective_timeout(_Item()) is None


def test_hang_dumps_traceback(tmp_path):
    reporter = _make(str(tmp_path), hang_dump_fraction=0.5)
    # timeout 2s * 0.5 -> dump after ~1s.
    reporter.pytest_runtest_setup(_Item(timeout_opt=2.0))
    time.sleep(1.4)
    content = open(reporter._hang_traceback_path(), encoding="utf-8").read()
    assert "hang watchdog fired for pkg/test_x.py::test_y" in content
    assert "Thread" in content or 'File "' in content


def test_completed_test_is_not_dumped(tmp_path):
    reporter = _make(str(tmp_path), hang_dump_fraction=0.5)
    reporter.pytest_runtest_setup(_Item(timeout_opt=2.0))
    time.sleep(0.2)
    reporter._cancel_hang_timer()  # what the next setup / sessionfinish does
    time.sleep(1.2)  # past the would-be dump point
    assert os.path.getsize(reporter._hang_traceback_path()) == 0


def test_no_watchdog_without_timeout(tmp_path):
    reporter = _make(str(tmp_path))
    reporter.pytest_runtest_setup(_Item())  # no timeout -> nothing armed
    assert reporter._hang_timer is None
