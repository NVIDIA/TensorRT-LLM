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
"""Report cost, exact sums, and state that must not leak between startups."""

import contextlib
import os
import sys
import types
import weakref

import pytest
import torch

import tensorrt_llm._torch.memory.monitor as mon
from tensorrt_llm._torch.memory.monitor import pools, startup, tagging
from tensorrt_llm._torch.memory.monitor.schema import Category
from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals

MB = 1 << 20
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _monitor_on(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    tagging.clear_claims()
    startup.clear()
    yield
    tagging.clear_claims()
    startup.clear()
    mon.clear_baseline()


def _capture_graph(size):
    graph = torch.cuda.CUDAGraph()
    warm = torch.ones(1024, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm * 2
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        kept = torch.empty(size, dtype=torch.uint8, device="cuda")
    return graph, kept


@needs_gpu
class TestReportCost:
    def test_one_allocator_snapshot_per_report(self, monkeypatch):
        real = torch.cuda.memory._snapshot
        calls = []
        monkeypatch.setattr(
            torch.cuda.memory, "_snapshot", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        mon.capture(tag="unit")
        assert len(calls) == 1

    def test_heap_is_walked_once_per_new_pool_not_per_report(self, monkeypatch):
        graph, kept = _capture_graph(8 * MB)
        walks = []
        real = pools._live_mempool_ids
        monkeypatch.setattr(pools, "_live_mempool_ids", lambda: walks.append(1) or real())
        try:
            for _ in range(3):
                mon.capture(tag="unit")
            # The graph pool is classified when first seen and remembered.
            assert len(walks) <= 1
        finally:
            del graph, kept

    def test_a_mempool_collected_while_its_blocks_live_stays_a_mempool(self):
        pools.clear_registered_pools()
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            held = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
        key = tuple(pool.id)
        assert pools.classify_all([key])[key][0] == "mempool"
        del pool
        assert pools.classify_all([key])[key][0] == "mempool"
        del held


@needs_gpu
def test_groups_add_up_when_allocator_counters_lag_the_snapshot(monkeypatch):
    """Tags come from the snapshot; a live figure read at another moment
    could fall below them, and clamping the difference broke the sum."""
    module = torch.nn.Linear(4096, 4096, device="cuda")
    tagging.claim_module_memory(module, "WEIGHTS")
    untagged = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")  # noqa: F841
    monkeypatch.setattr(
        torch.cuda, "memory_stats", lambda *a, **k: {"allocated_bytes.all.current": 0}
    )
    frame = mon.capture(tag="unit").frame
    totals = group_totals(frame)
    assert totals.sizes[Group.WEIGHTS] >= 4096 * 4096 * 4
    assert totals.sizes[Group.TORCH_UNATTRIBUTED] >= 32 * MB
    assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total


class TestSleepReleasedKvCache:
    class _V1Manager:
        def get_kv_cache_stats(self):
            return types.SimpleNamespace(allocated_bytes=512 * MB)

    def test_released_pool_is_taken_back_out(self):
        from tensorrt_llm._torch.memory.monitor.providers.kv_cache import KvCacheProvider

        manager = self._V1Manager()
        provider = KvCacheProvider(manager)
        pools.clear_registered_pools()
        try:
            pools.mark_released(["kv_cache"])
            entries = provider.collect()
            counted = sum(e.size_bytes for e in entries if e.counts_toward_total)
            assert counted == 0
            assert any(e.category is Category.SLEEP_RELEASED for e in entries)
            pools.mark_materialized(["kv_cache"])
            assert sum(e.size_bytes for e in provider.collect()) == 512 * MB
        finally:
            pools.clear_registered_pools()
        assert weakref.ref(manager)() is manager

    @needs_gpu
    def test_detail_section_keeps_the_residual_row_while_asleep(self):
        from tensorrt_llm._torch.virtual_memory import RestoreMode, release_with_tag
        from tensorrt_llm._torch.virtual_memory import scope as vm_scope

        pools.clear_registered_pools()
        with vm_scope("unit_detail", RestoreMode.NONE) as proxy:
            held = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
        try:
            release_with_tag("unit_detail")
            text = mon.render_capture(mon.capture(tag="unit"))
            assert "nontorch_residual" in text and "sleep_released" in text
            assert "unattributed_live" in text
        finally:
            from tensorrt_llm._torch.virtual_memory import materialize_with_tag

            materialize_with_tag("unit_detail")
            pools.clear_registered_pools()
            del held, proxy


class TestStartupState:
    @pytest.fixture
    def window(self, monkeypatch):
        logged = []
        monkeypatch.setattr(tagging, "start_tagging_window", lambda: True)
        monkeypatch.setattr(tagging, "stop_tagging_window", lambda: None)
        monkeypatch.setattr(
            tagging, "_log_window_result", lambda *a, **k: logged.append(k.get("failed"))
        )
        return logged

    def test_a_second_executor_starts_from_scratch(self, window, monkeypatch):
        monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(True))
        startup._stages.append(startup.StageSample("model", "Model", 0, 0, 1, 1))
        with tagging.startup_window():
            assert not startup.stages()

    def test_a_failing_freeze_does_not_replace_the_startup_error(self, window, monkeypatch):
        def broken():
            raise MemoryError("trace too large")

        monkeypatch.setattr(tagging, "freeze_tag_map", broken)
        with pytest.raises(ValueError, match="startup failed"):
            with tagging.startup_window():
                raise ValueError("startup failed")
        assert window == [True]

    def test_no_report_when_interrupted(self, window, monkeypatch):
        monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(True))
        with pytest.raises(KeyboardInterrupt):
            with tagging.startup_window():
                raise KeyboardInterrupt
        assert window == []

    def test_region_before_the_first_step_is_not_counted_twice(self, monkeypatch):
        state = types.SimpleNamespace(process=1000 * MB)
        monkeypatch.setattr(startup, "_reading", lambda: startup.Reading(state.process, 0, 0))
        with startup.region("NCCL communicators (torch.distributed)"):
            state.process += 300 * MB
        startup.begin_stage("model")
        parts = dict(startup.residual_breakdown(1300 * MB))
        assert parts["NCCL communicators (torch.distributed)"] == 300 * MB
        assert parts["before the first startup step"] == 1000 * MB
        assert sum(parts.values()) == 1300 * MB


class TestNvmlReason:
    def _fake_nvml(self, monkeypatch, pid):
        from tensorrt_llm import profiler

        proc = types.SimpleNamespace(pid=pid, usedGpuMemory=7 * MB)
        fake = types.SimpleNamespace(
            nvmlDeviceGetHandleByUUID=lambda u: "handle",
            nvmlDeviceGetComputeRunningProcesses_v3=lambda h: [proc],
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        monkeypatch.setattr(profiler, "pynvml_context", contextlib.nullcontext)
        monkeypatch.setattr(
            mon.torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(uuid="x")
        )

    def test_process_missing_from_nvml_says_so(self, monkeypatch):
        # Another PID namespace: NVML lists the host PID, never ours.
        self._fake_nvml(monkeypatch, os.getpid() + 100000)
        used, reason = mon._nvml_process_lookup(0)
        assert used is None and "PID namespace" in reason

    def test_found_process_has_no_reason(self, monkeypatch):
        self._fake_nvml(monkeypatch, os.getpid())
        assert mon._nvml_process_lookup(0) == (7 * MB, "")


def test_warmup_report_never_raises(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import trace_log_utils

    def broken(**_):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(mon, "capture", broken)
    trace_log_utils._maybe_log_monitor_report("warmup")


def test_wrapped_startup_history_is_reported(monkeypatch):
    warnings = []
    monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
    monkeypatch.setattr(tagging, "_DEFAULT_MAX_ENTRIES", 2)
    events = [{"action": "alloc", "addr": i, "user_metadata": "model"} for i in range(2)]
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda *a, **k: {"device_traces": [events]})
    monkeypatch.setattr(tagging.logger, "warning", warnings.append)
    tagging.freeze_tag_map()
    assert warnings and "not attributed" in warnings[0]


@needs_gpu
def test_reports_are_written_to_files_when_asked(monkeypatch, tmp_path):
    from tensorrt_llm._torch.memory.monitor import triggers

    monkeypatch.setenv(triggers.DIR_VAR, str(tmp_path / "reports"))
    text = triggers.report_now(tag="unit", rank=3)
    (path,) = list((tmp_path / "reports").iterdir())
    assert path.name.startswith(f"memory_rank3_pid{os.getpid()}_") and path.name.endswith(
        "_unit.txt"
    )
    assert path.read_text().strip() == text.strip()


@needs_gpu
def test_an_unwritable_directory_does_not_lose_the_report(monkeypatch, tmp_path):
    from tensorrt_llm._torch.memory.monitor import triggers

    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(triggers.DIR_VAR, str(blocker / "sub"))
    assert triggers.report_now(tag="unit") is not None
