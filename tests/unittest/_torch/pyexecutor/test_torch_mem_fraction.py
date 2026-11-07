# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch memory cap leaves the requested margin to the driver."""

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

pytestmark = pytest.mark.cpu_only

GiB = 1024**3
MiB = 1024**2


@pytest.mark.parametrize(
    "margin_args, margin",
    [((), 32 * MiB), ((DwdpManager.DRIVER_MEMORY_RESERVE,), GiB)],
    ids=["default", "dwdp"],
)
@pytest.mark.parametrize("mem_reserved", [0, 3 * GiB])
def test_torch_cap_leaves_margin_outside(monkeypatch, mem_reserved, margin_args, margin):
    mem_total = 80 * GiB
    mem_free = 20 * GiB
    fractions = []
    monkeypatch.setattr(_util.torch.cuda, "set_per_process_memory_fraction", fractions.append)
    monkeypatch.setattr(_util.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(_util.torch.cuda, "memory_reserved", lambda: mem_reserved)
    monkeypatch.setattr(_util.torch.cuda, "mem_get_info", lambda: (mem_free, mem_total))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    _util._adjust_torch_mem_fraction(*margin_args)

    assert fractions[0] == 1.0
    assert fractions[-1] * mem_total == pytest.approx(mem_free + mem_reserved - margin)
