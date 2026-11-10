# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch memory cap leaves up to 1 GiB of device memory to the driver."""

import pytest

from tensorrt_llm._torch.pyexecutor import _util

pytestmark = pytest.mark.cpu_only

GiB = 1024**3
MiB = 1024**2


def _torch_cap(monkeypatch, *, mem_free, mem_reserved, mem_total=80 * GiB):
    fractions = []
    monkeypatch.setattr(_util.torch.cuda, "set_per_process_memory_fraction", fractions.append)
    monkeypatch.setattr(_util.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(_util.torch.cuda, "memory_reserved", lambda: mem_reserved)
    monkeypatch.setattr(_util.torch.cuda, "mem_get_info", lambda: (mem_free, mem_total))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    _util._adjust_torch_mem_fraction()

    assert fractions[0] == 1.0
    return fractions[-1] * mem_total


@pytest.mark.parametrize("mem_reserved", [0, 3 * GiB])
def test_torch_cap_leaves_one_gib_outside(monkeypatch, mem_reserved):
    cap = _torch_cap(monkeypatch, mem_free=20 * GiB, mem_reserved=mem_reserved)

    assert cap == pytest.approx(20 * GiB + mem_reserved - GiB)


@pytest.mark.parametrize("mem_free", [512 * MiB, 0])
@pytest.mark.parametrize("mem_reserved", [0, 3 * GiB])
def test_torch_cap_never_drops_below_reserved_memory(monkeypatch, mem_free, mem_reserved):
    # Less than 1 GiB is left after the final KV cache and CUDA graphs: the
    # margin takes what is free, and the cap stays at what PyTorch holds.
    cap = _torch_cap(monkeypatch, mem_free=mem_free, mem_reserved=mem_reserved)

    assert cap == pytest.approx(mem_reserved)
