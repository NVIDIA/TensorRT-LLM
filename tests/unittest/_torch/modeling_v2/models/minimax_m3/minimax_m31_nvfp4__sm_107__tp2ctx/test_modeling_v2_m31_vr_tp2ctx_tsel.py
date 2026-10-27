# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_TSEL registration and its gate (tsel.py), driven by stand-in layers: ``register`` puts the tile select on
every MSA indexer whose hook is unset, leaves a set hook and an indexer-less backend alone, and registers nothing when
``TRTLLM_M31_DISABLE_TP2CTX_TSEL`` is set; ``select`` declines every step outside its rows / blocks range, so the
built-in select runs there. The table itself is gated on the GPU (TRTLLM_M31_CHECK_TSEL in-model). CPU only.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    tsel,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import MsaIndexer

pytestmark = pytest.mark.cpu_only


def _indexer(hook=None):
    indexer = object.__new__(MsaIndexer)  # the hook attribute only; no config is read here
    indexer.selector_hook = hook
    return indexer


def _model(*backends):
    layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=b)) for b in backends]
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def test_register_sets_every_unset_msa_hook_and_nothing_else(monkeypatch):
    monkeypatch.setattr(tsel, "TSEL", True)
    other = object()
    free, taken = _indexer(), _indexer(hook=other)
    model = _model(
        types.SimpleNamespace(indexer=free),
        types.SimpleNamespace(indexer=taken),
        types.SimpleNamespace(),  # a backend without an MSA indexer
    )
    assert tsel.register(model) == 1
    assert free.selector_hook is tsel.select
    assert taken.selector_hook is other


def test_the_kill_switch_registers_nothing(monkeypatch):
    monkeypatch.setattr(tsel, "TSEL", False)
    free = _indexer()
    assert tsel.register(_model(types.SimpleNamespace(indexer=free))) == 0
    assert free.selector_hook is None


@pytest.mark.parametrize(
    "shape, dtype, why",
    [
        ((2, 64, tsel.TSEL_MIN_ROWS - 1), torch.float32, "too few rows"),
        ((2, tsel.TSEL_MAX_BLOCKS + 1, 512), torch.float32, "too many blocks"),
        ((2, 64, 512), torch.bfloat16, "not float32"),
    ],
)
def test_select_declines_outside_its_range(monkeypatch, shape, dtype, why):
    monkeypatch.setattr(
        tsel, "_op", lambda: pytest.fail(f"the op was reached for a step with {why}")
    )
    scores = torch.zeros(shape, dtype=dtype)
    assert tsel.select(scores, torch.ones(shape[2], dtype=torch.int32), 1, 2) is None


def test_select_declines_without_the_op(monkeypatch):
    monkeypatch.setattr(tsel, "_op", lambda: None)
    scores = torch.zeros((2, 64, 512), dtype=torch.float32)
    assert tsel.select(scores, torch.ones(512, dtype=torch.int32), 1, 2) is None
