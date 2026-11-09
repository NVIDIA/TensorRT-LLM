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
"""Reuse of what executor startup already measures.

``_ExecutorMemoryMonitor`` and KV cache sizing keep their own accounts. The
report has to show *their* numbers and *their* advice, not a second opinion
that could drift from them.
"""

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import startup
from tensorrt_llm._torch.memory.monitor.capability import Capability
from tensorrt_llm._torch.memory.monitor.frame import DeviceReading, TorchReading, build_frame
from tensorrt_llm._torch.memory.monitor.report import render
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    startup.clear()
    yield
    startup.clear()


def _capability():
    return Capability(
        allocator_backend="native",
        alloc_conf="",
        alloc_conf_var=None,
        expandable_segments=False,
        metadata_supported=True,
        history_api_present=True,
        snapshot_api_present=True,
        oom_observer_present=True,
        nvml_available=True,
    )


def _frame(free=150 * GIB, kv=16 * GIB):
    return build_frame(
        DeviceReading(total=180 * GIB, free=free, process_used=180 * GIB - free),
        TorchReading(reserved=2 * GIB, live=1 * GIB),
        [
            Entry(
                category=Category.KV_CACHE_POOL,
                size_bytes=kv,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.PHYSICAL_BACKING,
            )
        ],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the executor monitor reads the device")
def test_executor_memory_monitor_feeds_the_report(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import _advice, tagging, triggers
    from tensorrt_llm._torch.memory.monitor.summary import Group
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

    monitor = _ExecutorMemoryMonitor()

    # Advice is reused, not restated: a change to the executor's advice
    # reaches the report without a second edit.
    assert (
        _advice()[Group.KV_CACHE]
        == monitor.memory_type_tuning_suggestion[ExecutorMemoryType.KV_CACHE]
    )

    # Its steps are recorded, under its own names.
    with monitor.observe_creation_stage(ExecutorMemoryType.SAMPLER):
        held = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    try:
        steps = startup.stages()
        assert [s.stage for s in steps] == ["sampler"]
        assert steps[0].name == monitor.memory_type_friendly_names[ExecutorMemoryType.SAMPLER]
    finally:
        del held

    # A failed step is recorded.
    startup.clear()
    with pytest.raises(RuntimeError):
        with monitor.observe_creation_stage(ExecutorMemoryType.KV_CACHE):
            raise torch.OutOfMemoryError("CUDA out of memory")
    assert startup.failed_stage().stage == "kv_cache"

    # A failed startup's headline names the step and its advice.
    startup.clear()
    monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
    texts = []
    real = triggers.report_now
    monkeypatch.setattr(triggers, "report_now", lambda **kw: texts.append(real(**kw)) or texts[-1])
    with pytest.raises(RuntimeError):
        with tagging.startup_window():
            # As in the executor: its memory monitor is built, and the
            # failing step recorded, inside the window.
            startup.set_stage_advice({"kv_cache": "KV cache"}, {"kv_cache": "reduce the fraction"})
            startup.record_failed_stage("kv_cache", 3 * GIB)
            raise RuntimeError("Executor creation failed due to insufficient GPU memory.")
    text = texts[0]
    assert "STARTUP FAILED while creating: KV cache" in text
    assert "Suggestion: reduce the fraction." in text
    assert "FAILED      KV cache (free device memory before it: 3.00 GiB)" in text


@pytest.mark.cpu_only
def test_memory_plan(monkeypatch):
    def rendered(non_kv_peak, activation, free, kv):
        startup.clear()
        startup.record_memory_plan(non_kv_peak=non_kv_peak, activation=activation)
        return render(_frame(free=free, kv=kv), _capability())

    # Device in use: 180 - 165 = 15 GiB, against 10 + 6 planned.
    text = rendered(10 * GIB, 2 * GIB, free=165 * GIB, kv=6 * GIB)
    assert "Startup plan: peak outside the KV cache 10.00 GiB + KV cache 6.00 GiB" in text
    assert "profiled forward pass allocated 2.00 GiB" in text
    assert "exceeds the startup plan" not in text, "within plan"

    # Device in use: 180 - 150 = 30 GiB, less 1 GiB of free PyTorch cache,
    # against 10 + 16 planned.
    text = rendered(10 * GIB, None, free=150 * GIB, kv=16 * GIB)
    assert "excluding 1.00 GiB of free PyTorch cache" in text
    assert "Usage exceeds the startup plan by 3.00 GiB" in text

    # Sizing measured after emptying the cache, and the workload can
    # reuse it. Counting it flagged a healthy startup as over plan.
    # In use 180 - 153.5 = 26.5 GiB; 1 GiB of it is free cache; plan 26.
    text = rendered(10 * GIB, None, free=int(153.5 * GIB), kv=16 * GIB)
    assert "exceeds the startup plan" not in text, "free PyTorch cache counted against the plan"

    # 100 MiB over after excluding the cache: below the tolerance.
    text = rendered(10 * GIB, None, free=int(153 * GIB) - (100 << 20), kv=16 * GIB)
    assert "exceeds the startup plan" not in text, "sampling noise called out"

    startup.clear()
    assert "Startup plan" not in render(_frame(), _capability()), "no plan, no line"

    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    startup.record_memory_plan(non_kv_peak=10 * GIB, activation=None)
    startup.record_stage("kv_cache", 100, 50)
    assert startup.memory_plan() is None and startup.stages() == (), "recorded while off"


@pytest.mark.cpu_only
def test_steps_only_in_startup_reports():
    startup.set_stage_advice({"model": "Model"}, {})
    startup.record_stage("model", 100 * GIB, 99 * GIB)
    assert "Memory taken by each startup step" not in render(_frame(), _capability())
    text = render(_frame(), _capability(), startup_steps=True)
    assert "Memory taken by each startup step" in text
    assert "+1.00 GiB  Model" in text

    # Device-wide deltas let another process's frees hide a step's cost;
    # two runs once showed the weight-loading step as 1.08 and 0.06 GiB.
    startup.clear()
    startup.set_stage_advice({"model": "Model"}, {})
    startup._stages.append(startup.StageSample("model", "Model", 100 * GIB, 100 * GIB, GIB))
    text = render(_frame(), _capability(), startup_steps=True)
    assert "(this process)" in text and "+1.00 GiB  Model" in text, "per-process preferred"

    startup.clear()
    startup._stages.append(startup.StageSample("model", "Model", 100 * GIB, 99 * GIB, None))
    text = render(_frame(), _capability(), startup_steps=True)
    assert "device-wide" in text


@pytest.mark.cpu_only
def test_every_scoped_step_has_a_category():
    # allocation_scope accepts exactly the steps that have a metric name;
    # one without a category would report under its raw value.
    from tensorrt_llm._torch.memory.monitor import _TAG_CATEGORIES
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor

    missing = [
        step.value
        for step in _ExecutorMemoryMonitor.creation_stage_metric_names
        if step.value not in _TAG_CATEGORIES
    ]
    assert not missing, f"creation steps without a category: {missing}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads a real allocator snapshot")
def test_creation_steps_tag_their_allocations():
    from tensorrt_llm._torch.memory.monitor import _tag_entries, tagging
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

    monitor = _ExecutorMemoryMonitor()
    startup.set_stage_advice({"sampler": "Sampler"}, {})
    if not tagging.start_tagging_window():
        pytest.skip("tagging unsupported here")
    try:
        with monitor.observe_creation_stage(ExecutorMemoryType.GUIDED_DECODER):
            decoder = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
        with tagging.tag("sampler"):
            sampler = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
        frozen = tagging.freeze_tag_map()
        tagging.set_frozen_map(frozen)
    finally:
        tagging.stop_tagging_window()
    try:
        assert tagging.live_bytes_by_tag(frozen).get("guided_decoder", 0) >= 16 << 20
        # Reported under the executor's own name for the step.
        entries = [e for e in _tag_entries() if e.category is Category.SAMPLER]
        assert entries and entries[0].owner == "Sampler"
        assert entries[0].size_bytes >= 32 << 20
    finally:
        tagging.set_frozen_map(tagging.TagSnapshot(supported=False))
        del decoder, sampler


@pytest.mark.cpu_only
def test_residual_breakdown(monkeypatch):
    """Splitting the overhead row by when, and where, it appeared."""
    assert startup.residual_breakdown(GIB) == [], "nothing recorded, nothing shown"

    startup._first = startup.Reading(process=10 * GIB, torch=3 * GIB, known=2 * GIB)
    startup._stages.append(
        startup.StageSample("model", "Model", 0, 0, process_taken=GIB, residual_taken=GIB)
    )
    startup._regions.append(startup.RegionSample("custom allreduce workspace", "model", GIB // 2))
    parts = dict(startup.residual_breakdown(8 * GIB))
    assert sum(parts.values()) == 8 * GIB
    assert parts["before the first startup step"] == 5 * GIB
    # The region's share is taken out of the step it ran in.
    assert parts["custom allreduce workspace"] == GIB // 2
    assert parts["Model (other)"] == GIB // 2
    assert parts["since startup"] == 2 * GIB

    startup.clear()
    startup._first = startup.Reading(process=10 * GIB, torch=0, known=0)
    parts = dict(startup.residual_breakdown(9 * GIB))
    assert parts["released since startup"] == -GIB, "freed parts are a negative remainder"

    startup.clear()
    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    with startup.region("anything"):
        pass
    assert startup._regions == [], "a region is a no-op while off"


@pytest.mark.cpu_only
def test_rows_list_their_parts():
    def tagged(category, size, owner):
        return Entry(
            category=category,
            size_bytes=size,
            source=Source.TAG,
            placement=Placement.TORCH_DEFAULT,
            basis=Basis.LOGICAL,
            owner=owner,
        )

    frame = build_frame(
        DeviceReading(total=180 * GIB, free=150 * GIB, process_used=30 * GIB),
        TorchReading(reserved=2 * GIB, live=2 * GIB),
        [
            tagged(Category.EXECUTOR_RUNTIME, GIB, "Additional executor resources"),
            tagged(Category.SAMPLER, GIB // 2, "Sampler"),
        ],
    )
    text = render(frame, _capability())
    assert "of which (GiB): Additional executor resources 1.00, Sampler 0.50" in text

    startup._first = startup.Reading(process=10 * GIB, torch=0, known=0)
    startup._regions.append(startup.RegionSample("custom allreduce workspace", None, 2 * GIB))
    frame = build_frame(
        DeviceReading(total=180 * GIB, free=150 * GIB, process_used=30 * GIB),
        TorchReading(reserved=0, live=0),
        [],
    )
    text = render(frame, _capability())
    assert "before the first startup step 10.00" in text
    assert "custom allreduce workspace 2.00" in text

    # Startup parts are per process; they would not reconcile with a
    # device-wide upper bound.
    frame = build_frame(
        DeviceReading(total=180 * GIB, free=150 * GIB, process_used=None),
        TorchReading(reserved=0, live=0),
        [],
    )
    assert "before the first startup step" not in render(frame, _capability())


def _cuda_runtime():
    import ctypes
    import glob

    for path in sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*")):
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures a real allocation")
def test_region_names_only_memory_allocated_outside_torch():
    import ctypes

    # The context must exist first: NVML lists only processes that have
    # one, so checking before creating it skips the test spuriously.
    torch.empty(1, device="cuda")
    if startup._process_used() is None:
        pytest.skip("NVML cannot report this process here")

    with startup.region("torch only"):
        held = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    try:
        # Already counted under PyTorch; claiming it here would double count.
        assert abs(startup._regions[-1].residual_taken) < 64 << 20
    finally:
        del held

    # A library calling cudaMalloc itself: invisible to torch, visible to
    # NVML, and therefore exactly what a region should catch.
    cudart = _cuda_runtime()
    if cudart is None:
        pytest.skip("no CUDA runtime library to call directly")
    pointer = ctypes.c_void_p()
    with startup.region("fake library"):
        assert cudart.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(256 << 20)) == 0
    try:
        sample = startup._regions[-1]
        assert sample.name == "fake library"
        assert sample.residual_taken >= 256 << 20
    finally:
        cudart.cudaFree(pointer)
