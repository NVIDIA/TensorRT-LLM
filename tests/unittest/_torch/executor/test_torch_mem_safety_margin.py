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
"""The device memory kept below the PyTorch allocator cap is configurable."""

import pytest

from tensorrt_llm._torch.pyexecutor import _util

_MIB = 1024**2
_GIB = 1024**3


@pytest.fixture
def cuda_memory(monkeypatch):
    fractions = []
    monkeypatch.setattr(_util.torch.cuda, "set_per_process_memory_fraction", fractions.append)
    monkeypatch.setattr(_util.torch.cuda, "memory_reserved", lambda: 10 * _GIB)
    monkeypatch.setattr(_util.torch.cuda, "memory_allocated", lambda: 5 * _GIB)
    monkeypatch.setattr(_util.torch.cuda, "mem_get_info", lambda: (30 * _GIB, 100 * _GIB))
    monkeypatch.setattr(_util.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("TLLM_TORCH_MEM_SAFETY_MARGIN_MB", raising=False)
    return fractions


@pytest.mark.parametrize("margin_mb", [None, "0", "2048"])
def test_margin_below_the_torch_cap(cuda_memory, monkeypatch, margin_mb):
    if margin_mb is not None:
        monkeypatch.setenv("TLLM_TORCH_MEM_SAFETY_MARGIN_MB", margin_mb)
    _util._adjust_torch_mem_fraction()
    # The first call lifts the cap to 1.0; the second sets free + reserved - margin.
    assert cuda_memory[0] == 1.0
    margin = (4096 if margin_mb is None else int(margin_mb)) * _MIB
    assert cuda_memory[-1] == pytest.approx((40 * _GIB - margin) / (100 * _GIB), rel=0, abs=1e-12)
