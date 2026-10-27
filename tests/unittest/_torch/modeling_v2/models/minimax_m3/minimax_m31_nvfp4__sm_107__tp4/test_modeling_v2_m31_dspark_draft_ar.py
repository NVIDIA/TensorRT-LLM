# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S9b-VR-a: the DSpark drafter's row-parallel all-reduces take the one-shot all-reduce up to
``DRAFT_AR_ONESHOT_MAX_ROWS`` rows and keep their AUTO all-reduce above (``dspark.install_draft_oneshot_allreduce``).

The all-reduce and linear classes are stand-ins (the routing is the unit under test; the one-shot kernel has its own
tests), so this runs on the CPU without a process group.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark,
)

Strategy = dspark.AllReduceStrategy


class FakeAllReduce(nn.Module):
    """Records each call; ``strategy`` and ``mapping`` as the real module exposes them."""

    def __init__(self, mapping, strategy=Strategy.AUTO, dtype=None):
        super().__init__()
        self.mapping = mapping
        self.strategy = strategy
        self.dtype = dtype
        self.calls = []

    def uses_nccl_symmetric_memory_window(self):
        return self.strategy == Strategy.AUTO

    def forward(self, input, *, all_reduce_params=None):
        self.calls.append((input.shape[0], all_reduce_params))
        return input + 1


class FakeLinear(nn.Module):
    def __init__(self, tp_mode, all_reduce, dtype=torch.bfloat16):
        super().__init__()
        self.tp_mode = tp_mode
        self.all_reduce = all_reduce
        self.dtype = dtype


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(dspark, "AllReduce", FakeAllReduce)
    monkeypatch.setattr(dspark, "Linear", FakeLinear)
    return SimpleNamespace(mapping=SimpleNamespace(tp_size=4, tp_rank=0))


def _drafter(mapping):
    ROW, COL = dspark.TensorParallelMode.ROW, dspark.TensorParallelMode.COLUMN
    d = nn.Module()
    d.o_proj = FakeLinear(ROW, FakeAllReduce(mapping))
    d.down_proj = FakeLinear(ROW, FakeAllReduce(mapping))
    d.qkv_proj = FakeLinear(COL, None)
    d.nccl_row = FakeLinear(
        ROW, FakeAllReduce(mapping, Strategy.NCCL)
    )  # an explicit strategy is left alone
    d.no_reduce = FakeLinear(ROW, None)
    return d


def test_wraps_only_row_parallel_auto_linears(fakes):
    d = _drafter(fakes.mapping)
    assert dspark.install_draft_oneshot_allreduce(d, 32) == 2
    assert isinstance(d.o_proj.all_reduce, dspark.DraftOneshotAllReduce)
    assert isinstance(d.down_proj.all_reduce, dspark.DraftOneshotAllReduce)
    assert isinstance(d.nccl_row.all_reduce, FakeAllReduce)
    assert d.qkv_proj.all_reduce is None and d.no_reduce.all_reduce is None
    # the one-shot module is the explicit ONESHOT strategy, never TWOSHOT
    assert d.o_proj.all_reduce.oneshot.strategy == Strategy.ONESHOT


@pytest.mark.parametrize(
    "rows, takes_oneshot", [(1, True), (4, True), (32, True), (33, False), (4096, False)]
)
def test_routes_by_rows(fakes, rows, takes_oneshot):
    d = _drafter(fakes.mapping)
    dspark.install_draft_oneshot_allreduce(d, 32)
    w = d.o_proj.all_reduce
    params = object()
    out = w(torch.zeros(rows, 8), all_reduce_params=params)
    assert torch.equal(out, torch.ones(rows, 8))
    taken, other = (w.oneshot, w.auto) if takes_oneshot else (w.auto, w.oneshot)
    assert taken.calls == [(rows, params)] and other.calls == []


def test_output_buffer_choice_stays_the_original(fakes):
    d = _drafter(fakes.mapping)
    dspark.install_draft_oneshot_allreduce(d, 32)
    assert (
        d.o_proj.all_reduce.uses_nccl_symmetric_memory_window() is True
    )  # the AUTO module's answer


def test_zero_rows_installs_nothing(fakes):
    d = _drafter(fakes.mapping)
    assert dspark.install_draft_oneshot_allreduce(d, 0) == 0
    assert isinstance(d.o_proj.all_reduce, FakeAllReduce)


@pytest.mark.parametrize("value, expected", [("1", 0), ("0", 32), ("", 32)])
def test_kill_switch(monkeypatch, value, expected):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_S9BVR", value)
    try:
        mod = importlib.reload(dspark)
        assert mod.DRAFT_AR_ONESHOT_MAX_ROWS == expected
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_S9BVR", raising=False)
        importlib.reload(dspark)
    assert os.environ.get("TRTLLM_M31_DISABLE_S9BVR") is None
