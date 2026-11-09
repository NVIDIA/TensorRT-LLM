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
"""TLLM_MEM_MONITOR modes: 0 (off), oom (the default), 1 (full)."""

import threading

import pytest
import torch

import tensorrt_llm._torch.memory.monitor as mon
from tensorrt_llm._torch.memory.monitor import startup, tagging, triggers

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(triggers, "_attached", True)  # leave torch's observer alone
    triggers.uninstall_oom_reporter()
    yield
    triggers.uninstall_oom_reporter()


def test_values(monkeypatch):
    for value, expected in [
        (None, "oom"),
        ("", "oom"),
        ("oom", "oom"),
        ("OOM", "oom"),
        ("0", "off"),
        ("1", "full"),
    ]:
        if value is None:
            monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        else:
            monkeypatch.setenv("TLLM_MEM_MONITOR", value)
        assert mon.mode() == expected, f"TLLM_MEM_MONITOR={value!r}"

    # An unknown value falls back to the default, with a warning.
    warnings = []
    monkeypatch.setenv("TLLM_MEM_MONITOR", "yes-please")
    from tensorrt_llm.logger import logger

    monkeypatch.setattr(logger, "warning", warnings.append)
    mon._warned_values.discard("yes-please")
    assert mon.mode() == "oom"
    assert warnings and "expected 0, oom or 1" in warnings[0]


class TestDefaultReportsOnlyOutOfMemory:
    def test_reports_only_out_of_memory(self, monkeypatch):
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        monkeypatch.setenv(triggers.INTERVAL_VAR, "0.05")
        seen = []
        monkeypatch.setenv("TLLM_MEM_MONITOR_OOM_INTERVAL", "0")
        monkeypatch.setattr(
            triggers, "report_now", lambda tag, **kw: seen.append((tag, kw.get("force"))) or "r"
        )
        assert triggers.install_oom_reporter(rank=0) is True
        triggers.notify_error("CUDA error: out of memory")
        # The full monitor is off, so the report has to be forced.
        assert seen == [("out_of_memory", True)]

        # No periodic thread even though an interval is set ...
        assert not any(t.name == "memory-monitor-periodic" for t in threading.enumerate())
        # ... no startup recording or regions ...
        assert not startup._on()
        with tagging.startup_window() as recording:
            assert recording is False
        # ... and an unforced report (startup, periodic, warmup logging) is not built.
        if torch.cuda.is_available():
            assert mon.capture(tag="unit") is None

        triggers.uninstall_oom_reporter()
        monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", "3")
        assert triggers.install_oom_reporter(rank=0) is False, "the rank filter applies"


def test_off_reports_nothing(monkeypatch):
    seen = []
    monkeypatch.setenv("TLLM_MEM_MONITOR", "0")
    monkeypatch.setattr(triggers, "report_now", lambda tag, **kw: seen.append(tag) or "r")
    assert triggers.install_oom_reporter(rank=0) is False
    triggers.notify_error("CUDA error: out of memory")
    assert seen == []
