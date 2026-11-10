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
from memory_monitor_test_utils import MB, capture_graph, needs_gpu

import tensorrt_llm._torch.memory.monitor as mon
from tensorrt_llm._torch.memory.monitor import pools, startup, tagging
from tensorrt_llm._torch.memory.monitor import triggers as triggers_module
from tensorrt_llm._torch.memory.monitor.schema import Category
from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals


@pytest.fixture(autouse=True)
def _monitor_on(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")


@needs_gpu
def test_report_cost(monkeypatch):
    # One allocator snapshot per report.
    real_snapshot = torch.cuda.memory._snapshot
    calls = []
    with monkeypatch.context() as m:
        m.setattr(
            torch.cuda.memory,
            "_snapshot",
            lambda *a, **k: calls.append(1) or real_snapshot(*a, **k),
        )
        mon.capture(tag="unit")
    assert len(calls) == 1, "one allocator snapshot per report"

    # The heap is walked once per new pool, not per report: the graph pool
    # is classified when first seen and remembered.
    graph, _, kept = capture_graph(8 * MB)
    walks = []
    real_walk = pools._live_mempool_ids
    try:
        with monkeypatch.context() as m:
            m.setattr(pools, "_live_mempool_ids", lambda: walks.append(1) or real_walk())
            for _ in range(3):
                mon.capture(tag="unit")
        assert len(walks) <= 1, "heap walked on every report"
    finally:
        del graph, kept

    # A MemPool collected while its blocks live stays a mempool.
    pools.clear_registered_pools()
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool):
        held = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
    key = tuple(pool.id)
    assert pools.classify_all([key])[key][0] == "mempool"
    del pool
    assert pools.classify_all([key])[key][0] == "mempool", "collected MemPool reclassified"
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


class _V1Manager:
    def get_kv_cache_stats(self):
        return types.SimpleNamespace(allocated_bytes=512 * MB)


@pytest.mark.cpu_only
def test_released_kv_cache_is_listed_not_counted():
    from tensorrt_llm._torch.memory.monitor.providers.kv_cache import KvCacheProvider

    manager = _V1Manager()
    provider = KvCacheProvider(manager)
    pools.clear_registered_pools()
    try:
        pools.mark_released(["kv_cache"])
        entries = provider.collect()
        counted = sum(e.size_bytes for e in entries if e.counts_toward_total)
        assert counted == 0
        assert all(e.released for e in entries if e.category is Category.KV_CACHE_POOL)
        assert all(e.size_bytes >= 0 for e in entries if e.size_bytes is not None)
        pools.mark_materialized(["kv_cache"])
        assert sum(e.size_bytes for e in provider.collect()) == 512 * MB
    finally:
        pools.clear_registered_pools()
    assert weakref.ref(manager)() is manager


@needs_gpu
def test_detail_section_keeps_the_residual_row_while_asleep():
    from tensorrt_llm._torch.virtual_memory import RestoreMode, release_with_tag
    from tensorrt_llm._torch.virtual_memory import scope as vm_scope

    pools.clear_registered_pools()
    with vm_scope("unit_detail", RestoreMode.NONE) as proxy:
        held = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
    try:
        release_with_tag("unit_detail")
        text = mon.render_capture(mon.capture(tag="unit"))
        assert "nontorch_residual" in text and "Released by sleep mode" in text
        assert "unattributed_live" in text
    finally:
        from tensorrt_llm._torch.virtual_memory import materialize_with_tag

        materialize_with_tag("unit_detail")
        pools.clear_registered_pools()
        del held, proxy


@pytest.mark.cpu_only
def test_startup_state_does_not_leak(monkeypatch):
    logged = []
    monkeypatch.setattr(tagging, "start_tagging_window", lambda: True)
    monkeypatch.setattr(tagging, "stop_tagging_window", lambda: None)
    monkeypatch.setattr(
        tagging, "_log_window_result", lambda *a, **k: logged.append(k.get("failed"))
    )

    def frozen():
        return tagging.TagSnapshot(True)

    # A second executor starts from scratch.
    monkeypatch.setattr(tagging, "freeze_tag_map", frozen)
    startup._stages.append(startup.StageSample("model", "Model", 0, 0, 1, 1))
    with tagging.startup_window():
        assert not startup.stages(), "a second executor inherited the first one's steps"
    logged.clear()

    # A failing freeze does not replace the startup error.
    def broken():
        raise MemoryError("trace too large")

    monkeypatch.setattr(tagging, "freeze_tag_map", broken)
    with pytest.raises(ValueError, match="startup failed"):
        with tagging.startup_window():
            raise ValueError("startup failed")
    assert logged == [True]
    logged.clear()

    # No report when interrupted.
    monkeypatch.setattr(tagging, "freeze_tag_map", frozen)
    with pytest.raises(KeyboardInterrupt):
        with tagging.startup_window():
            raise KeyboardInterrupt
    assert logged == [], "reported after an interrupt"

    # A region before the first step is not counted twice.
    startup.clear()
    state = types.SimpleNamespace(process=1000 * MB)
    monkeypatch.setattr(startup, "_reading", lambda: startup.Reading(state.process, 0, 0))
    with startup.region("NCCL communicators (torch.distributed)"):
        state.process += 300 * MB
    startup.begin_stage("model")
    parts = dict(startup.residual_breakdown(1300 * MB))
    assert parts["NCCL communicators (torch.distributed)"] == 300 * MB
    assert parts["before the first startup step"] == 1000 * MB
    assert sum(parts.values()) == 1300 * MB


@pytest.mark.cpu_only
def test_nvml_lookup_says_why_the_process_is_missing(monkeypatch):
    from tensorrt_llm import profiler
    from tensorrt_llm._torch.memory.monitor.capability import nvml_process_usage

    proc = types.SimpleNamespace(pid=os.getpid(), usedGpuMemory=7 * MB)
    fake = types.SimpleNamespace(
        nvmlDeviceGetHandleByUUID=lambda u: "handle",
        nvmlDeviceGetComputeRunningProcesses_v3=lambda h: [proc],
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(profiler, "pynvml_context", contextlib.nullcontext)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(uuid="x")
    )

    assert nvml_process_usage(0) == (7 * MB, ""), "found process has no reason"
    # Another PID namespace: NVML lists the host PID, never ours.
    proc.pid = os.getpid() + 100000
    used, reason = nvml_process_usage(0)
    assert used is None and "PID namespace" in reason


@pytest.mark.cpu_only
def test_startup_problems_are_logged_not_raised(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import trace_log_utils

    def broken(**_):
        raise RuntimeError("snapshot failed")

    # The warmup report never raises.
    with monkeypatch.context() as m:
        m.setattr(triggers_module, "capture", broken)
        trace_log_utils._maybe_log_monitor_report("warmup")

    # Startup history that wrapped the ring buffer is reported.
    warnings = []
    monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
    monkeypatch.setattr(tagging, "_max_entries", 2)
    events = [{"action": "alloc", "addr": i, "user_metadata": "model"} for i in range(2)]
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda *a, **k: {"device_traces": [events]})
    monkeypatch.setattr(tagging.logger, "warning", warnings.append)
    tagging.freeze_tag_map()
    assert warnings and "not attributed" in warnings[0]


@pytest.mark.cpu_only
def test_tagging_failures_stay_out_of_the_tagged_code(monkeypatch):
    # A frozen owner labels a live block only while it is still that block.
    frozen = tagging.TagSnapshot(True, owners={1: "A", 2: "B"}, sizes={1: 100})
    assert frozen.owner_of(1, 100) == "A"
    assert frozen.owner_of(1, 50) is None, "same address, different allocation"
    assert frozen.owner_of(2, 7) == "B", "no recorded size matches any size"
    assert frozen.owner_of(3, 100) is None

    # A tag that could not be set does not become the context of the tagged
    # code's own exception.
    def refuse(_name):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "_get_memory_metadata", lambda: "", raising=False)
    monkeypatch.setattr(torch.cuda.memory, "_set_memory_metadata", refuse, raising=False)
    with pytest.raises(ValueError) as raised:
        with tagging.tag("STEP"):
            raise ValueError("the step's own failure")
    assert raised.value.__context__ is None


@pytest.mark.cpu_only
def test_sleep_tags_are_marked_as_each_call_succeeds(monkeypatch):
    from tensorrt_llm._torch import virtual_memory

    class Manager:
        def __init__(self, failing):
            self.failing = failing

        def release_with_tag(self, tag):
            if tag == self.failing:
                raise RuntimeError("release failed")
            return 1

        def materialize_with_tag(self, tag):
            if tag == self.failing:
                raise RuntimeError("out of memory")
            return 1

    pools.forget_released_tags()
    try:
        # Waking sorted tags: kv_cache comes back, then the weights fail and
        # roll back. Only the weights are still released.
        pools.mark_released(("kv_cache", "model_weights"))
        monkeypatch.setattr(
            virtual_memory, "get_virtual_memory_manager", lambda: Manager("model_weights")
        )
        with pytest.raises(RuntimeError):
            virtual_memory.materialize_with_tag("kv_cache", "model_weights")
        assert not pools.is_tag_released("kv_cache"), "materialized before the failure"
        assert pools.is_tag_released("model_weights"), "rolled back"

        # A release that fails part way has still released the failing tag.
        pools.forget_released_tags()
        monkeypatch.setattr(virtual_memory, "get_virtual_memory_manager", lambda: Manager("a"))
        with pytest.raises(RuntimeError):
            virtual_memory.release_with_tag("a", "b")
        assert pools.is_tag_released("a") and not pools.is_tag_released("b")
    finally:
        pools.forget_released_tags()


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

    # An unwritable directory does not lose the report.
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(triggers.DIR_VAR, str(blocker / "sub"))
    assert triggers.report_now(tag="unit") is not None


@pytest.mark.cpu_only
def test_summary_edge_cases():
    import re

    from tensorrt_llm._torch.memory.monitor.frame import DeviceReading, TorchReading, build_frame
    from tensorrt_llm._torch.memory.monitor.pools import PoolUsage
    from tensorrt_llm._torch.memory.monitor.summary import render_summary

    gib = 1 << 30

    # Another process holding the GPU is named.
    device = DeviceReading(total=178 * gib, free=gib // 2, process_used=2 * gib)
    frame = build_frame(device, TorchReading(reserved=gib, live=gib), [])
    text = "\n".join(render_summary(frame, group_totals(frame), headline="STARTUP FAILED."))
    assert "other processes on this GPU hold 175.50 GiB" in text

    # The share column never runs into the size, even with a residual far
    # larger than the process, as when readings disagree.
    device = DeviceReading(total=178 * gib, free=100 * gib, process_used=gib)
    frame = build_frame(device, TorchReading(reserved=40 * gib, live=40 * gib), [])
    for line in render_summary(frame, group_totals(frame)):
        assert "GiB-" not in line and not re.search(r"GiB\d", line), line

    # The released note uses readable, merged names.
    pools_ = (
        PoolUsage((0, 1), "mempool", "sleep: model_weights", gib, gib, True, "model_weights"),
        PoolUsage((0, 2), "mempool", "sleep: model", 20 * MB, 10 * MB, True, "model"),
        PoolUsage((0, 3), "mempool", "sleep: model", 1 * MB, 0, True, "model"),
    )
    frame = build_frame(
        DeviceReading(total=178 * gib, free=170 * gib, process_used=2 * gib),
        TorchReading(reserved=0, live=0, pools=pools_),
        [],
    )
    startup.record_memory_plan(non_kv_peak=3 * gib, activation=gib)
    text = "\n".join(render_summary(frame, group_totals(frame)))
    note = [line for line in text.splitlines() if "Released by sleep mode" in line][0]
    assert "Model weights 1.00 GiB" in note
    assert note.count("Model ") == 1 + note.count("Model weights")  # one merged "Model" part
    # The startup plan describes the awake process; not compared while asleep.
    assert "Startup plan" not in text
