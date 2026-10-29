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
class TestExecutorMemoryMonitorFeedsTheReport:
    def _monitor(self):
        from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor

        return _ExecutorMemoryMonitor()

    def test_its_steps_are_recorded(self):
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        monitor = self._monitor()
        with monitor.observe_creation_stage(ExecutorMemoryType.SAMPLER):
            held = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        try:
            steps = startup.stages()
            assert [s.stage for s in steps] == ["sampler"]
            assert steps[0].name == monitor.memory_type_friendly_names[ExecutorMemoryType.SAMPLER]
        finally:
            del held

    def test_failed_step_is_recorded(self):
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        monitor = self._monitor()
        with pytest.raises(RuntimeError):
            with monitor.observe_creation_stage(ExecutorMemoryType.KV_CACHE):
                raise torch.OutOfMemoryError("CUDA out of memory")
        assert startup.failed_stage().stage == "kv_cache"

    def test_advice_is_the_executors_own_text(self):
        from tensorrt_llm._torch.memory.monitor import _advice
        from tensorrt_llm._torch.memory.monitor.summary import Group
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        monitor = self._monitor()
        # Reused, not restated: a change to the executor's advice reaches
        # the report without a second edit.
        assert (
            _advice()[Group.KV_CACHE]
            == monitor.memory_type_tuning_suggestion[ExecutorMemoryType.KV_CACHE]
        )


class TestFailedStartup:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="capture reads the device")
    def test_headline_names_the_step_and_its_advice(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import tagging, triggers

        startup.set_stage_advice({"kv_cache": "KV cache"}, {"kv_cache": "reduce the fraction"})
        startup.record_failed_stage("kv_cache", 3 * GIB)
        monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
        texts = []
        real = triggers.report_now
        monkeypatch.setattr(
            triggers, "report_now", lambda **kw: texts.append(real(**kw)) or texts[-1]
        )

        with pytest.raises(RuntimeError):
            with tagging.startup_window():
                raise RuntimeError("Executor creation failed due to insufficient GPU memory.")

        text = texts[0]
        assert "STARTUP FAILED while creating: KV cache" in text
        assert "Suggestion: reduce the fraction." in text
        assert "FAILED      KV cache (free device memory before it: 3.00 GiB)" in text


class TestMemoryPlan:
    def test_within_plan(self):
        startup.record_memory_plan(non_kv_peak=10 * GIB, activation=2 * GIB)
        # Device in use: 180 - 165 = 15 GiB, against 10 + 6 planned.
        text = render(_frame(free=165 * GIB, kv=6 * GIB), _capability())
        assert "Startup plan: peak outside the KV cache 10.00 GiB + KV cache 6.00 GiB" in text
        assert "profiled forward pass allocated 2.00 GiB" in text
        assert "exceeds the startup plan" not in text

    def test_over_plan_is_called_out(self):
        startup.record_memory_plan(non_kv_peak=10 * GIB, activation=None)
        # Device in use: 180 - 150 = 30 GiB, less 1 GiB of free PyTorch cache,
        # against 10 + 16 planned.
        text = render(_frame(free=150 * GIB, kv=16 * GIB), _capability())
        assert "excluding 1.00 GiB of free PyTorch cache" in text
        assert "Usage exceeds the startup plan by 3.00 GiB" in text

    def test_free_pytorch_cache_is_not_counted_against_the_plan(self):
        # Sizing measured after emptying the cache, and the workload can
        # reuse it. Counting it flagged a healthy startup as over plan.
        startup.record_memory_plan(non_kv_peak=10 * GIB, activation=None)
        # In use 180 - 153.5 = 26.5 GiB; 1 GiB of it is free cache; plan 26.
        text = render(_frame(free=int(153.5 * GIB), kv=16 * GIB), _capability())
        assert "exceeds the startup plan" not in text

    def test_sampling_noise_is_not_called_out(self):
        startup.record_memory_plan(non_kv_peak=10 * GIB, activation=None)
        # 100 MiB over after excluding the cache: below the tolerance.
        text = render(_frame(free=int(153 * GIB) - (100 << 20), kv=16 * GIB), _capability())
        assert "exceeds the startup plan" not in text

    def test_nothing_recorded_while_off(self, monkeypatch):
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        startup.record_memory_plan(non_kv_peak=10 * GIB, activation=None)
        startup.record_stage("kv_cache", 100, 50)
        assert startup.memory_plan() is None and startup.stages() == ()

    def test_no_plan_no_line(self):
        assert "Startup plan" not in render(_frame(), _capability())


class TestStepsOnlyInStartupReports:
    def test_listed_when_asked(self):
        startup.set_stage_advice({"model": "Model"}, {})
        startup.record_stage("model", 100 * GIB, 99 * GIB)
        text = render(_frame(), _capability(), startup_steps=True)
        assert "Memory taken by each startup step" in text
        assert "+1.00 GiB  Model" in text

    def test_per_process_figures_preferred(self):
        # Device-wide deltas let another process's frees hide a step's cost;
        # two runs once showed the weight-loading step as 1.08 and 0.06 GiB.
        startup.set_stage_advice({"model": "Model"}, {})
        startup._stages.append(startup.StageSample("model", "Model", 100 * GIB, 100 * GIB, GIB))
        text = render(_frame(), _capability(), startup_steps=True)
        assert "(this process)" in text and "+1.00 GiB  Model" in text

    def test_device_wide_figures_are_labelled_as_such(self):
        startup._stages.append(startup.StageSample("model", "Model", 100 * GIB, 99 * GIB, None))
        text = render(_frame(), _capability(), startup_steps=True)
        assert "device-wide" in text

    def test_absent_otherwise(self):
        startup.record_stage("model", 100 * GIB, 99 * GIB)
        assert "Memory taken by each startup step" not in render(_frame(), _capability())


class TestCreationStepsAreTagged:
    """Each executor creation step tags what it allocates."""

    def test_every_scoped_step_has_a_category(self):
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
    def test_observed_step_tags_its_allocations(self):
        from tensorrt_llm._torch.memory.monitor import tagging
        from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        monitor = _ExecutorMemoryMonitor()
        if not tagging.start_tagging_window():
            pytest.skip("tagging unsupported here")
        try:
            with monitor.observe_creation_stage(ExecutorMemoryType.GUIDED_DECODER):
                held = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
            by_tag = tagging.live_bytes_by_tag(tagging.freeze_tag_map())
        finally:
            tagging.stop_tagging_window()
        try:
            assert by_tag.get("guided_decoder", 0) >= 16 << 20
        finally:
            del held

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="reads a real allocator snapshot")
    def test_tagged_step_is_reported_under_the_executors_name(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import _tag_entries, tagging

        startup.set_stage_advice({"sampler": "Sampler"}, {})
        if not tagging.start_tagging_window():
            pytest.skip("tagging unsupported here")
        try:
            with tagging.tag("sampler"):
                held = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
            tagging.set_frozen_map(tagging.freeze_tag_map())
        finally:
            tagging.stop_tagging_window()
        try:
            entries = [e for e in _tag_entries() if e.category is Category.SAMPLER]
            assert entries and entries[0].owner == "Sampler"
            assert entries[0].size_bytes >= 32 << 20
        finally:
            tagging.set_frozen_map(tagging.TagSnapshot(supported=False))
            del held


class TestGroupMembers:
    def test_a_group_lists_what_it_is_made_of(self):
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=150 * GIB, process_used=30 * GIB),
            TorchReading(reserved=2 * GIB, live=2 * GIB),
            [
                Entry(
                    category=Category.EXECUTOR_RUNTIME,
                    size_bytes=GIB,
                    source=Source.TAG,
                    placement=Placement.TORCH_DEFAULT,
                    basis=Basis.LOGICAL,
                    owner="Additional executor resources",
                ),
                Entry(
                    category=Category.SAMPLER,
                    size_bytes=GIB // 2,
                    source=Source.TAG,
                    placement=Placement.TORCH_DEFAULT,
                    basis=Basis.LOGICAL,
                    owner="Sampler",
                ),
            ],
        )
        text = render(frame, _capability())
        assert "of which (GiB): Additional executor resources 1.00, Sampler 0.50" in text
