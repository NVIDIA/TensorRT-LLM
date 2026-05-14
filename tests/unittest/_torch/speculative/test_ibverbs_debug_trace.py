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
"""Tests for the ``_debug_trace`` and ``_tensor_row_preview`` helpers."""

import pytest

torch = pytest.importorskip("torch")

# E402: imports below importorskip are intentional.
from tensorrt_llm._torch.speculative.ibverbs_draft_offload import (  # noqa: E402
    _debug_trace as offload_debug_trace,
)
from tensorrt_llm._torch.speculative.ibverbs_draft_offload import _tensor_row_preview  # noqa: E402
from tensorrt_llm._torch.speculative.ibverbs_endpoint import (  # noqa: E402
    _debug_trace as endpoint_debug_trace,
)


def test_debug_trace_writes(tmp_path, monkeypatch):
    log = tmp_path / "trace.log"
    monkeypatch.setenv("TLLM_RDMA_DEBUG_TRACE_PATH", str(log))
    offload_debug_trace("hello %s", "world")
    endpoint_debug_trace("nic=%s qpn=%d", "mlx5_0", 102)
    text = log.read_text()
    assert "hello world" in text
    assert "nic=mlx5_0 qpn=102" in text
    # Each line starts with timestamp + pid + module tag.
    for line in text.strip().splitlines():
        assert "pid=" in line


def test_debug_trace_disabled(monkeypatch):
    monkeypatch.setenv("TLLM_RDMA_DEBUG_TRACE_PATH", "")
    # No exception even with a path that the trace can't write to.
    offload_debug_trace("ignored %s", "value")


def test_debug_trace_handles_format_error(tmp_path, monkeypatch):
    """Bad format strings shouldn't crash — they get logged literally."""
    log = tmp_path / "trace.log"
    monkeypatch.setenv("TLLM_RDMA_DEBUG_TRACE_PATH", str(log))
    offload_debug_trace("%s and %s", "only-one")
    text = log.read_text()
    # The fallback embeds args repr.
    assert "only-one" in text


def test_tensor_row_preview_basic():
    t = torch.tensor([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=torch.int32)
    assert _tensor_row_preview(t, row=0) == [1, 2, 3, 4, 5]
    assert _tensor_row_preview(t, row=1, count=3) == [10, 20, 30]
    assert _tensor_row_preview(t, row=0, limit=2) == [1, 2]
    assert _tensor_row_preview(None, row=0) == []


def test_tensor_row_preview_count_clamped():
    t = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    # count larger than tensor width
    assert _tensor_row_preview(t, row=0, count=99) == [1, 2, 3]
    # count negative
    assert _tensor_row_preview(t, row=0, count=-5) == []
