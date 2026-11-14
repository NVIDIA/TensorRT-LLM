# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch memory cap leaves up to 1 GiB, or the model's declared margin, to the driver."""

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of

pytestmark = pytest.mark.cpu_only

GiB = 1024**3
MiB = 1024**2


def _torch_cap(monkeypatch, *, mem_free, mem_reserved, mem_total=80 * GiB, margin=None):
    fractions = []
    monkeypatch.setattr(_util.torch.cuda, "set_per_process_memory_fraction", fractions.append)
    monkeypatch.setattr(_util.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(_util.torch.cuda, "memory_reserved", lambda: mem_reserved)
    monkeypatch.setattr(_util.torch.cuda, "mem_get_info", lambda: (mem_free, mem_total))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    _util._adjust_torch_mem_fraction(margin)

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


class _DeclaresMargin:
    runtime_hooks = RuntimeHooks(torch_mem_safety_margin=4 * GiB)


class _DeclaresNothing:
    pass


def test_model_declares_its_margin_on_the_class():
    # The executor creator reads the margin from the model it holds.
    assert runtime_hooks_of(_DeclaresMargin()).torch_mem_safety_margin == 4 * GiB
    assert runtime_hooks_of(_DeclaresNothing()).torch_mem_safety_margin is None


@pytest.mark.parametrize("mem_reserved", [0, 3 * GiB])
def test_declared_margin_replaces_the_default(monkeypatch, mem_reserved):
    cap = _torch_cap(monkeypatch, mem_free=20 * GiB, mem_reserved=mem_reserved, margin=4 * GiB)

    assert cap == pytest.approx(20 * GiB + mem_reserved - 4 * GiB)


@pytest.mark.parametrize("mem_free", [3 * GiB, 0])
def test_declared_margin_is_clamped_to_free_memory(monkeypatch, mem_free):
    cap = _torch_cap(monkeypatch, mem_free=mem_free, mem_reserved=2 * GiB, margin=4 * GiB)

    assert cap == pytest.approx(2 * GiB)
