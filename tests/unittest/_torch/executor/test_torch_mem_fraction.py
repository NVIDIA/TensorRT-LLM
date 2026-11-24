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
"""The PyTorch memory cap leaves a safety margin for driver allocations after setup."""

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor._util import (
    KIMI_K3_TORCH_MEM_SAFETY_MARGIN,
    _adjust_torch_mem_fraction,
)

pytestmark = pytest.mark.cpu_only

GiB = 1024**3


@pytest.mark.parametrize(
    "kwargs,margin",
    [
        ({}, 32 * 1024**2),
        ({"safety_margin": KIMI_K3_TORCH_MEM_SAFETY_MARGIN}, 4 * GiB),
    ],
    ids=["default", "kimi_k3"],
)
def test_torch_cap_is_free_memory_plus_reserve_minus_margin(monkeypatch, kwargs, margin):
    fractions = []
    cuda = _util.torch.cuda
    monkeypatch.setattr(cuda, "set_per_process_memory_fraction", fractions.append)
    monkeypatch.setattr(cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(cuda, "memory_reserved", lambda: 6 * GiB)
    monkeypatch.setattr(cuda, "mem_get_info", lambda: (40 * GiB, 80 * GiB))

    _adjust_torch_mem_fraction(**kwargs)

    assert fractions == [1.0, (40 * GiB + 6 * GiB - margin) / (80 * GiB)]


def test_kimi_k3_margin_is_four_gib():
    assert KIMI_K3_TORCH_MEM_SAFETY_MARGIN == 4 * GiB
