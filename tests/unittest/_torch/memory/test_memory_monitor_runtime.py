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
"""Reporting while serving: periodic, on request, and from other threads,
plus startup attribution of memory that worker threads allocate."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import tagging, triggers
from tensorrt_llm.logger import logger

MB = 1 << 20
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _armed_cleanly(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv("TLLM_MEM_MONITOR_OOM_INTERVAL", "0")
    triggers.uninstall_oom_reporter()
    yield
    triggers.uninstall_oom_reporter()


@pytest.fixture
def reports(monkeypatch):
    """Record OOM reports instead of building them."""
    seen = []

    def fake_report_now(tag, rank=None, **kw):
        seen.append((tag, kw.get("headline", "")))
        # A real report logs; that must not loop back through the filter.
        logger.error("memory monitor report mentioning out of memory")
        return "report"

    monkeypatch.setattr(triggers, "report_now", fake_report_now)
    monkeypatch.setattr(triggers, "_attached", True)  # leave torch's observer alone
    return seen


class TestPeriodic:
    def test_reports_on_its_interval_and_stops(self, monkeypatch, tmp_path):
        monkeypatch.setenv(triggers.INTERVAL_VAR, "0.05")
        monkeypatch.setattr(triggers, "_attached", True)
        monkeypatch.setenv(triggers.DIR_VAR, str(tmp_path))
        assert triggers.install_oom_reporter()
        deadline = time.monotonic() + 30
        while not list(tmp_path.glob("*_periodic.txt")) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert list(tmp_path.glob("*_periodic.txt"))
        triggers.uninstall_oom_reporter()
        assert not any(t.name == "memory-monitor-periodic" for t in threading.enumerate())

    def test_off_unless_asked(self, monkeypatch):
        monkeypatch.delenv(triggers.INTERVAL_VAR, raising=False)
        monkeypatch.setattr(triggers, "_attached", True)
        triggers.install_oom_reporter()
        assert not any(t.name == "memory-monitor-periodic" for t in threading.enumerate())


class TestErrorsFromOtherThreads:
    def test_logged_oom_is_reported_once(self, reports):
        triggers.install_oom_reporter()
        # How the KV transfer workers report a failed delivery.
        logger.error("_process_task_queue[0]: unhandled exception: CUDA error: out of memory")
        assert len(reports) == 1 and reports[0][0] == "out_of_memory"

    def test_other_errors_and_warnings_are_ignored(self, reports):
        triggers.install_oom_reporter()
        level = logger.level
        logger.set_level("info")  # let the warning reach the filter
        try:
            logger.error("request 7 failed: invalid token id")
            # Runtime autotuning skips tactics this way; not a failure.
            logger.warning("autotuner: tactic 3 ran out of memory, skipping")
        finally:
            logger.set_level(level)
        assert reports == []

    def test_thread_dying_of_oom_is_reported(self, reports):
        triggers.install_oom_reporter()

        def run():
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        assert len(reports) == 1

    def test_hooks_are_removed_with_the_reporter(self, reports):
        previous = threading.excepthook
        triggers.install_oom_reporter()
        triggers.uninstall_oom_reporter()
        assert threading.excepthook is previous
        logger.error("CUDA out of memory")
        assert reports == []

    def test_nothing_before_arming(self, reports):
        logger.error("CUDA out of memory")
        assert reports == []


class TestOnRequest:
    @needs_gpu
    def test_worker_reports_even_with_the_monitor_off(self, monkeypatch):
        from tensorrt_llm.executor.base_worker import BaseWorker

        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        worker = object.__new__(BaseWorker)
        worker.engine = SimpleNamespace(device_id=torch.cuda.current_device())
        text = worker.get_memory_report()
        assert text and "This process uses" in text

    def test_llm_returns_each_workers_report(self):
        from tensorrt_llm import LLM

        llm = object.__new__(LLM)
        llm._collective_rpc = MagicMock(return_value=["rank 0 report", None])
        assert llm.get_memory_report() == ["rank 0 report"]
        llm._collective_rpc.assert_called_once_with("get_memory_report")

    def test_allowed_on_a_multi_rank_mpi_executor(self):
        from tensorrt_llm.executor.proxy import _check_collective_rpc_guard

        _check_collective_rpc_guard(2, None, None, method="get_memory_report")


@needs_gpu
def test_worker_thread_allocation_is_named_after_the_running_step(monkeypatch):
    if not tagging.start_tagging_window():
        pytest.skip("tagging unsupported here")
    held = {}
    try:
        with tagging.tag("model"):
            worker = threading.Thread(
                target=lambda: held.setdefault(
                    "x", torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
                )
            )
            worker.start()
            worker.join()
        after = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        frozen = tagging.freeze_tag_map()
    finally:
        tagging.stop_tagging_window()
    by_tag = tagging.live_bytes_by_tag(frozen)
    # The thread carried no tag of its own; the step's window names it.
    assert by_tag.get("model", 0) >= 64 * MB
    # Outside every window stays untagged.
    assert by_tag.get(tagging.UNTAGGED, 0) >= 16 * MB
    held.clear()
    del after


def test_nested_windows_resolve_to_the_innermost():
    lookup = tagging._window_lookup([(0, 100, "outer"), (40, 60, "inner")])
    assert [lookup(t) for t in (10, 50, 70, 150)] == ["outer", "inner", "outer", None]
