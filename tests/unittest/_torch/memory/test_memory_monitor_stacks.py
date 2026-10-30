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
"""Naming unattributed PyTorch memory by the code that allocated it.

Off by default; TLLM_MEM_MONITOR_STACKS=1 turns it on. The end-to-end cases
use real allocator recording, since what matters is that torch keeps a stack
on each live block.
"""

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import stacks, tagging

MB = 1 << 20


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    tagging.clear_claims()
    yield
    torch.cuda.memory._record_memory_history(None) if torch.cuda.is_available() else None
    tagging.clear_claims()
    tagging.set_frozen_map(tagging.TagSnapshot(supported=False))


class TestSiteNames:
    def test_innermost_tensorrt_llm_frame_wins(self):
        frames = [
            {"filename": "/usr/lib/python3/site-packages/torch/_tensor.py", "name": "empty"},
            {"filename": "/repo/tensorrt_llm/_torch/modules/fused_moe/moe.py", "name": "forward"},
            {"filename": "/repo/tensorrt_llm/_torch/pyexecutor/model_engine.py", "name": "run"},
        ]
        assert stacks.site_of(frames) == "modules/fused_moe/moe.py:forward"

    def test_monitor_frames_are_skipped(self):
        frames = [
            {"filename": "/repo/tensorrt_llm/_torch/memory/monitor/tagging.py", "name": "x"},
            {
                "filename": "/repo/tensorrt_llm/_torch/attention/backends/trtllm.py",
                "name": "forward",
            },
        ]
        assert stacks.site_of(frames) == "attention/backends/trtllm.py:forward"

    def test_third_party_library_without_our_frames(self):
        frames = [
            {"filename": "/usr/lib/python3/site-packages/flashinfer/decode.py", "name": "plan"}
        ]
        assert stacks.site_of(frames) == "flashinfer/decode.py:plan"

    def test_no_frames(self):
        assert stacks.site_of([]) is None

    def test_users_own_code_beats_the_tensorrt_llm_frame_that_called_it(self):
        # A logits processor holding memory: the processor is what to fix.
        frames = [
            {"filename": "/usr/lib/python3/site-packages/torch/_tensor.py", "name": "empty"},
            {"filename": "/work/my_app/processors.py", "name": "__call__"},
            {
                "filename": "/x/site-packages/tensorrt_llm/_torch/pyexecutor/runner.py",
                "name": "_apply_logits_processors",
            },
        ]
        assert stacks.site_of(frames) == "my_app/processors.py:__call__"

    def test_standard_library_is_not_user_code(self):
        frames = [
            {"filename": "/usr/lib/python3.12/functools.py", "name": "wrapper"},
            {"filename": "/usr/lib/python3.12/threading.py", "name": "run"},
            {"filename": "/repo/tensorrt_llm/_torch/modules/linear.py", "name": "forward"},
        ]
        assert stacks.site_of(frames) == "modules/linear.py:forward"


class TestOffByDefault:
    def test_disabled_without_the_variable(self, monkeypatch):
        monkeypatch.delenv(stacks.ENV_VAR, raising=False)
        assert stacks.stacks_enabled() is False

    def test_startup_window_stops_recording_by_default(self, monkeypatch):
        monkeypatch.delenv(stacks.ENV_VAR, raising=False)
        calls = []
        monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
        monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(supported=True))
        monkeypatch.setattr(tagging, "_log_window_result", lambda *a, **k: None)
        monkeypatch.setattr(
            torch.cuda.memory, "_record_memory_history", lambda *a, **k: calls.append((a, k))
        )
        with tagging.startup_window():
            pass
        # The last call ends recording; serving pays nothing.
        assert calls[-1][0] == (None,)

    def test_startup_window_hands_over_when_asked(self, monkeypatch):
        monkeypatch.setenv(stacks.ENV_VAR, "1")
        calls = []
        monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
        monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(supported=True))
        monkeypatch.setattr(tagging, "_log_window_result", lambda *a, **k: None)
        monkeypatch.setattr(
            torch.cuda.memory, "_record_memory_history", lambda *a, **k: calls.append((a, k))
        )
        with tagging.startup_window():
            pass
        # State only: each live block keeps its stack, no event history.
        assert calls[-1][1].get("enabled") == "state"
        assert calls[-1][1].get("stacks") == "python"


def _allocate_in_moe_forward():
    return torch.empty(64 * MB, dtype=torch.uint8, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestEndToEnd:
    def test_unattributed_memory_is_named_by_its_allocating_code(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import capture, render_capture

        monkeypatch.setenv(stacks.ENV_VAR, "1")
        if not stacks.stacks_enabled():
            pytest.skip("stack recording unsupported here")
        torch.empty(1, device="cuda")
        assert stacks.start_stack_recording()
        held = _allocate_in_moe_forward()
        try:
            snapshot = capture(tag="unit")
            sites = snapshot.unattributed_sites or {}
            ours = [s for s in sites if s.endswith(":_allocate_in_moe_forward")]
            assert ours and sites[ours[0]] >= 64 * MB
            # Named even as the only part: that is the point of the mode.
            assert "_allocate_in_moe_forward" in render_capture(snapshot)
        finally:
            del held

    def test_memory_from_before_recording_is_called_out(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import capture, render_capture

        monkeypatch.setenv(stacks.ENV_VAR, "1")
        if not stacks.stacks_enabled():
            pytest.skip("stack recording unsupported here")
        torch.cuda.memory._record_memory_history(None)
        earlier = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        assert stacks.start_stack_recording()
        held = _allocate_in_moe_forward()
        try:
            text = render_capture(capture(tag="unit"))
            # Parts must add up to the row, so what predates recording is
            # listed rather than silently missing.
            assert "no stack recorded 0.03" in text
            assert "_allocate_in_moe_forward" in text
        finally:
            del held, earlier

    def test_parts_sum_to_the_row(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import capture
        from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals, render_summary

        monkeypatch.setenv(stacks.ENV_VAR, "1")
        if not stacks.stacks_enabled():
            pytest.skip("stack recording unsupported here")
        torch.empty(1, device="cuda")
        stacks.start_stack_recording()
        held = _allocate_in_moe_forward()
        try:
            snapshot = capture(tag="unit")
            totals = group_totals(snapshot.frame)
            lines = render_summary(
                snapshot.frame, totals, unattributed_sites=snapshot.unattributed_sites
            )
            row = totals.sizes[Group.TORCH_UNATTRIBUTED]
            parts = dict(snapshot.unattributed_sites)
            remainder = row - sum(parts.values())
            assert sum(parts.values()) + remainder == row
            assert any("of which" in line for line in lines)
        finally:
            del held

    def test_no_sites_without_recording(self, monkeypatch):
        from tensorrt_llm._torch.memory.monitor import capture

        monkeypatch.delenv(stacks.ENV_VAR, raising=False)
        torch.cuda.memory._record_memory_history(None)
        held = _allocate_in_moe_forward()
        try:
            assert capture(tag="unit").unattributed_sites is None
        finally:
            del held
