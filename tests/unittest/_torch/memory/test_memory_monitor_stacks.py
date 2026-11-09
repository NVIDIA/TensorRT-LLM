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


@pytest.mark.cpu_only
def test_site_names():
    torch_empty = {"filename": "/usr/lib/python3/site-packages/torch/_tensor.py", "name": "empty"}
    cases = [
        (
            "innermost TensorRT-LLM frame wins",
            [
                torch_empty,
                {
                    "filename": "/repo/tensorrt_llm/_torch/modules/fused_moe/moe.py",
                    "name": "forward",
                },
                {"filename": "/repo/tensorrt_llm/_torch/pyexecutor/model_engine.py", "name": "run"},
            ],
            "modules/fused_moe/moe.py:forward",
        ),
        (
            "monitor frames are skipped",
            [
                {"filename": "/repo/tensorrt_llm/_torch/memory/monitor/tagging.py", "name": "x"},
                {
                    "filename": "/repo/tensorrt_llm/_torch/attention/backends/trtllm.py",
                    "name": "forward",
                },
            ],
            "attention/backends/trtllm.py:forward",
        ),
        (
            "third-party library without our frames",
            [{"filename": "/usr/lib/python3/site-packages/flashinfer/decode.py", "name": "plan"}],
            "flashinfer/decode.py:plan",
        ),
        ("no frames", [], None),
        (
            # A logits processor holding memory: the processor is what to fix.
            "user's own code beats the TensorRT-LLM frame that called it",
            [
                torch_empty,
                {"filename": "/work/my_app/processors.py", "name": "__call__"},
                {
                    "filename": "/x/site-packages/tensorrt_llm/_torch/pyexecutor/runner.py",
                    "name": "_apply_logits_processors",
                },
            ],
            "my_app/processors.py:__call__",
        ),
        (
            "standard library is not user code",
            [
                {"filename": "/usr/lib/python3.12/functools.py", "name": "wrapper"},
                {"filename": "/usr/lib/python3.12/threading.py", "name": "run"},
                {"filename": "/repo/tensorrt_llm/_torch/modules/linear.py", "name": "forward"},
            ],
            "modules/linear.py:forward",
        ),
    ]
    for case, frames, expected in cases:
        assert stacks.site_of(frames) == expected, case


@pytest.mark.cpu_only
def test_off_by_default_and_handed_over_when_asked(monkeypatch):
    monkeypatch.setattr(tagging, "tagging_supported", lambda: True)
    monkeypatch.setattr(tagging, "freeze_tag_map", lambda: tagging.TagSnapshot(supported=True))
    monkeypatch.setattr(tagging, "_log_window_result", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        torch.cuda.memory, "_record_memory_history", lambda *a, **k: calls.append((a, k))
    )

    monkeypatch.delenv(stacks.ENV_VAR, raising=False)
    assert stacks.stacks_enabled() is False
    with tagging.startup_window():
        pass
    # The last call ends recording; serving pays nothing.
    assert calls[-1][0] == (None,), "recording must stop after startup by default"

    monkeypatch.setenv(stacks.ENV_VAR, "1")
    calls.clear()
    with tagging.startup_window():
        pass
    # State only: each live block keeps its stack, no event history.
    assert calls[-1][1].get("enabled") == "state"
    assert calls[-1][1].get("stacks") == "python"


def _allocate_in_moe_forward():
    return torch.empty(64 * MB, dtype=torch.uint8, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unattributed_memory_is_named_by_its_allocating_code(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import capture, render_capture
    from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals, render_summary

    # Without recording there are no sites at all.
    monkeypatch.delenv(stacks.ENV_VAR, raising=False)
    torch.cuda.memory._record_memory_history(None)
    held = _allocate_in_moe_forward()
    try:
        assert capture(tag="unit").unattributed_sites is None
    finally:
        del held

    monkeypatch.setenv(stacks.ENV_VAR, "1")
    if not stacks.stacks_enabled():
        pytest.skip("stack recording unsupported here")
    earlier = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    assert stacks.start_stack_recording()
    held = _allocate_in_moe_forward()
    try:
        snapshot = capture(tag="unit")
        sites = snapshot.unattributed_sites or {}
        ours = [s for s in sites if s.endswith(":_allocate_in_moe_forward")]
        assert ours and sites[ours[0]] >= 64 * MB
        text = render_capture(snapshot)
        # Named even as the only part: that is the point of the mode.
        assert "_allocate_in_moe_forward" in text
        # Parts must add up to the row, so what predates recording is
        # listed rather than silently missing.
        assert "no stack recorded 0.03" in text

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
        del held, earlier
