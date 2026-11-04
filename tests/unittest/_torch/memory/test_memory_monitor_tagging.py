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
"""Tag-based attribution of torch-visible memory.

The traps these pin down were all found by measurement, and each is silent:
the ring buffer drops the earliest tags (which are the weights), a non-native
allocator accepts a tag and discards it, and an unsupported setup produces a
breakdown of zeros that reads exactly like a process holding nothing.

Needs a GPU: the attribution is read back out of a real allocator snapshot.
"""

import pytest
import torch

from tensorrt_llm._torch.memory.monitor.checks import check_tag_coverage
from tensorrt_llm._torch.memory.monitor.tagging import (
    UNTAGGED,
    TagSnapshot,
    freeze_tag_map,
    live_bytes_by_tag,
    start_tagging_window,
    stop_tagging_window,
    tag,
    tagging_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reads a real allocator snapshot"
)

MB = 1 << 20


@pytest.fixture
def window(monkeypatch):
    """A tagging window, with the monitor on, that is always closed again."""
    from tensorrt_llm._torch.memory.monitor.tagging import clear_claims, set_frozen_map

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    clear_claims()
    started = start_tagging_window()
    try:
        yield started
    finally:
        stop_tagging_window()
        clear_claims()
        set_frozen_map(TagSnapshot(supported=False))


class TestAttribution:
    def test_tags_land_on_the_right_bytes(self, window):
        if not window:
            pytest.skip("tagging unsupported here")

        with tag("WEIGHTS"):
            weights = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
        with tag("SAMPLER"):
            sampler = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        plain = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")

        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            assert by_tag["WEIGHTS"] == 64 * MB
            assert by_tag["SAMPLER"] == 32 * MB
            assert by_tag[UNTAGGED] >= 16 * MB
        finally:
            del weights, sampler, plain

    def test_untagged_is_always_present(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("ONLY"):
            buf = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            # Reporting only the tags would read as a complete breakdown.
            assert UNTAGGED in by_tag
        finally:
            del buf

    def test_nesting_restores_the_outer_tag(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("OUTER"):
            a = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
            with tag("INNER"):
                b = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
            c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            assert by_tag.get("INNER", 0) == 16 * MB
            # Clearing instead of restoring would leave this at 16 MiB.
            assert by_tag.get("OUTER", 0) == 32 * MB
        finally:
            del a, b, c


class TestFrozenMapSurvivesTheRingBuffer:
    """Why the map is frozen rather than replayed on demand."""

    def test_attribution_holds_after_the_window_closes(self, window):
        if not window:
            pytest.skip("tagging unsupported here")

        with tag("WEIGHTS"):
            weights = torch.empty(128 * MB, dtype=torch.uint8, device="cuda")
        frozen = freeze_tag_map()
        stop_tagging_window()

        try:
            # Steady state: plenty of churn, no recording. A design that
            # replayed the trace at read time would have lost WEIGHTS by now.
            for _ in range(3000):
                scratch = torch.empty(4096, dtype=torch.uint8, device="cuda")
                del scratch
            activation = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")

            by_tag = live_bytes_by_tag(frozen)
            assert by_tag["WEIGHTS"] == 128 * MB, (
                "the frozen map must keep attributing weights once the trace has moved on"
            )
            assert by_tag[UNTAGGED] >= 64 * MB
            del activation
        finally:
            del weights


class TestUnsupportedIsNotZero:
    """An unsupported setup must not look like an empty one."""

    def test_unsupported_snapshot_yields_no_entries(self):
        empty = TagSnapshot(supported=False)
        assert live_bytes_by_tag(empty) == {}, "a dict of zeros would read as a measurement"
        assert not empty

    def test_tag_is_a_noop_when_unsupported(self, monkeypatch):
        import tensorrt_llm._torch.memory.monitor.tagging as mod

        monkeypatch.setattr(mod, "tagging_supported", lambda: False)
        # Call sites must not need a guard of their own.
        with mod.tag("WEIGHTS"):
            pass

    def test_freeze_reports_unsupported_rather_than_empty(self, monkeypatch):
        import tensorrt_llm._torch.memory.monitor.tagging as mod

        monkeypatch.setattr(mod, "tagging_supported", lambda: False)
        snapshot = mod.freeze_tag_map()
        assert snapshot.supported is False

    def test_non_native_backend_refuses(self, monkeypatch):
        # The torch docs are explicit that cudaMallocAsync and pluggable
        # allocators accept the call and drop the tag. Detecting that up
        # front is the difference between "unsupported" and a small,
        # plausible, wrong number.
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
        assert tagging_supported() is False

    def test_pluggable_allocator_raises_and_is_caught(self, monkeypatch):
        def boom():
            raise RuntimeError("pluggable allocator installed")

        monkeypatch.setattr(torch.cuda.memory, "_memory_metadata_supported", boom)
        assert tagging_supported() is False


class TestCheckB:
    def test_passes_when_tags_fit_inside_live(self):
        by_tag = {"WEIGHTS": 64 * MB, UNTAGGED: 16 * MB}
        results = check_tag_coverage(by_tag, torch_live_bytes=80 * MB)
        assert results[0].passed

    def test_fails_when_tags_exceed_live(self):
        # What a stale frozen tag looks like: an address reused by an
        # untagged allocation whose reuse event the ring buffer dropped.
        by_tag = {"WEIGHTS": 512 * MB, UNTAGGED: 0}
        results = check_tag_coverage(by_tag, torch_live_bytes=64 * MB)
        assert not results[0].passed
        assert "tagged=536870912" in results[0].detail

    def test_unsupported_is_not_a_failure(self):
        results = check_tag_coverage({}, torch_live_bytes=64 * MB)
        assert results[0].passed
        assert "not reported as zero" in results[0].detail

    def test_untagged_share_is_informational_not_a_failure(self):
        # Activations are never tagged, so a large untagged share is normal
        # and must not be raised as a problem.
        by_tag = {"WEIGHTS": 1 * MB, UNTAGGED: 99 * MB}
        results = check_tag_coverage(by_tag, torch_live_bytes=100 * MB)
        assert all(r.passed for r in results)
        assert "99.0%" in results[1].detail

    def test_real_reading_satisfies_check_b(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("WEIGHTS"):
            weights = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            results = check_tag_coverage(by_tag, torch.cuda.memory_allocated())
            assert all(r.passed for r in results)
        finally:
            del weights


class TestClaimModuleMemory:
    """Weights are attributed by walking the live module at report time."""

    def test_claims_parameters_and_buffers(self, window):
        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

        module = torch.nn.Linear(512, 512).cuda()
        module.register_buffer("scratch", torch.empty(1024, 1024, device="cuda"))
        try:
            assert claim_module_memory(module, "WEIGHTS") is True
            by_tag = live_bytes_by_tag(TagSnapshot(supported=False))
            # The buffer alone is 4 MiB; weight and bias add to that.
            assert by_tag.get("WEIGHTS", 0) >= 4 * MB
        finally:
            del module

    def test_shared_storage_counted_once(self, window):
        # Tied embeddings: two parameters, one storage. Summing their
        # logical sizes double counts; claiming by block does not. Measured
        # on Qwen2-0.5B, this is a 259.7 MiB difference.
        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

        shared = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")

        class Tied(torch.nn.Module):
            def __init__(self, storage):
                super().__init__()
                self.embed = torch.nn.Parameter(storage, requires_grad=False)
                self.head = torch.nn.Parameter(storage, requires_grad=False)

        module = Tied(shared)
        try:
            claim_module_memory(module, "WEIGHTS")
            by_tag = live_bytes_by_tag(TagSnapshot(supported=False))
            assert by_tag.get("WEIGHTS", 0) == 8 * MB
        finally:
            del module, shared

    def test_survives_the_window_freeze(self, window):
        # The closing freeze rebuilds the trace-derived map wholesale; claims
        # live elsewhere so they are not discarded by it.
        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory, set_frozen_map

        module = torch.nn.Linear(1024, 1024).cuda()
        try:
            claim_module_memory(module, "WEIGHTS")
            set_frozen_map(freeze_tag_map())
            assert live_bytes_by_tag(freeze_tag_map()).get("WEIGHTS", 0) > 0
        finally:
            del module

    def test_freed_weights_do_not_label_their_successor(self, window):
        # An address recorded at load time keeps its label after the block
        # is freed and reused. Walking the module sees only what it holds
        # now, so a replaced weight's old block, reused by an activation, is
        # not counted as weights.
        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(
            torch.empty(16 * MB, dtype=torch.uint8, device="cuda"), requires_grad=False
        )
        claim_module_memory(module, "WEIGHTS")
        before = live_bytes_by_tag(TagSnapshot(supported=False)).get("WEIGHTS", 0)
        assert before == 16 * MB

        # Replace the weight (a smaller one, as quantisation would), free the
        # old block, and let an unrelated allocation take its place.
        module.weight = torch.nn.Parameter(
            torch.empty(4 * MB, dtype=torch.uint8, device="cuda"), requires_grad=False
        )
        torch.cuda.synchronize()
        activation = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            after = live_bytes_by_tag(TagSnapshot(supported=False)).get("WEIGHTS", 0)
            assert after == 4 * MB, f"expected only the new weight, got {after / MB} MiB"
        finally:
            del activation, module

    def test_released_module_is_forgotten(self, window):
        import gc

        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

        module = torch.nn.Linear(256, 256).cuda()
        claim_module_memory(module, "WEIGHTS")
        del module
        gc.collect()
        # A claim must not keep the model alive, nor survive it.
        assert "WEIGHTS" not in live_bytes_by_tag(TagSnapshot(supported=False))

    def test_cpu_module_claims_nothing(self, window):
        from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

        module = torch.nn.Linear(8, 8)
        claim_module_memory(module, "WEIGHTS")
        assert "WEIGHTS" not in live_bytes_by_tag(TagSnapshot(supported=False))


class TestNothingHappensWhenOff:
    """The monitor is off by default; loading a model must cost nothing."""

    def test_claim_only_takes_a_weak_reference(self, monkeypatch):
        # Recorded even when off, so an on-demand report names the weights;
        # but it must cost nothing and must not keep the model alive.
        from tensorrt_llm._torch.memory.monitor import tagging

        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        tagging.clear_claims()
        snapshots = []
        monkeypatch.setattr(
            torch.cuda.memory, "_snapshot", lambda *a, **k: snapshots.append(1) or {"segments": []}
        )

        model = torch.nn.Linear(4, 4)
        assert tagging.claim_module_memory(model, "WEIGHTS") is True
        assert snapshots == [], "a disabled monitor must not take an allocator snapshot"
        assert tagging.claimed_modules() == [model]
        del model
        assert tagging.claimed_modules() == []
        tagging.clear_claims()

    def test_tag_does_not_touch_allocator_metadata(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import tagging

        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        calls = []
        monkeypatch.setattr(
            torch.cuda.memory, "_set_memory_metadata", lambda *a: calls.append(a), raising=False
        )
        with tagging.tag("SAMPLER"):
            pass
        assert calls == []


class TestCaptureBuildsTagEntries:
    """``capture()`` with tags present, which is the shipped path.

    Earlier tests all passed ``entries`` explicitly and so never ran
    ``_tag_entries``. That left a reference to a non-existent enum member
    unexercised until a real serving run hit it, so these go through
    ``capture()`` itself.
    """

    def test_capture_succeeds_with_tagged_memory(self, window, monkeypatch):
        if not window:
            pytest.skip("tagging unsupported here")
        monkeypatch.setenv("TLLM_MEM_MONITOR", "1")

        from tensorrt_llm._torch.memory.monitor import capture, render_capture
        from tensorrt_llm._torch.memory.monitor.tagging import set_frozen_map

        with tag("WEIGHTS"):
            weights = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        try:
            set_frozen_map(freeze_tag_map())
            result = capture(tag="test", rank=0)
            assert result is not None
            # Rendering exercises every field each entry declares.
            assert "weights" in render_capture(result)
        finally:
            set_frozen_map(TagSnapshot(supported=False))
            del weights

    def test_tag_entries_use_a_real_placement(self, window, monkeypatch):
        if not window:
            pytest.skip("tagging unsupported here")
        monkeypatch.setenv("TLLM_MEM_MONITOR", "1")

        from tensorrt_llm._torch.memory.monitor import _tag_entries
        from tensorrt_llm._torch.memory.monitor.schema import Placement
        from tensorrt_llm._torch.memory.monitor.tagging import set_frozen_map

        with tag("WEIGHTS"):
            weights = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            set_frozen_map(freeze_tag_map())
            entries = _tag_entries()
            assert entries, "a tagged allocation must produce an entry"
            for entry in entries:
                assert isinstance(entry.placement, Placement)
        finally:
            set_frozen_map(TagSnapshot(supported=False))
            del weights


class TestStartupWindow:
    def test_reports_even_when_tagging_is_unsupported(self, monkeypatch):
        # On cudaMallocAsync tags cannot work, but the KV cache and every
        # other component still belong in the startup breakdown.
        from tensorrt_llm._torch.memory.monitor import tagging, triggers

        monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
        monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
        reports = []
        monkeypatch.setattr(triggers, "report_now", lambda **kw: reports.append(kw) or "r")

        with tagging.startup_window() as recording:
            assert recording is False
        assert len(reports) == 1 and reports[0]["tag"] == "startup_complete"

    def test_records_without_stacks(self, monkeypatch):
        # Stacks make each snapshot cost seconds; the tag survives without them.
        from tensorrt_llm._torch.memory.monitor import tagging

        calls = []
        monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
        monkeypatch.setattr(
            torch.cuda.memory, "_record_memory_history", lambda *a, **k: calls.append((a, k))
        )
        assert tagging.start_tagging_window()
        assert calls[0][1].get("context", "all") is None

    def test_recording_stops_before_the_report_is_built(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import tagging, triggers

        monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
        order = []
        monkeypatch.setattr(tagging, "start_tagging_window", lambda: True)
        monkeypatch.setattr(tagging, "freeze_tag_map", lambda: TagSnapshot(supported=True))
        monkeypatch.setattr(tagging, "stop_tagging_window", lambda: order.append("stop"))
        monkeypatch.setattr(tagging, "live_bytes_by_tag", lambda s: order.append("read") or {})
        monkeypatch.setattr(triggers, "report_now", lambda **kw: order.append("report"))

        with tagging.startup_window():
            pass
        assert order.index("stop") < order.index("read") < order.index("report")
