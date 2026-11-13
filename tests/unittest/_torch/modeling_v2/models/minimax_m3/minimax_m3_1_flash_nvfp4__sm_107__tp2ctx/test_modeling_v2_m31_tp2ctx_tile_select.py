# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_TSEL registration and its gate (tile_select.py), driven by stand-in layers: ``register`` puts the tile select
on every MSA indexer whose hook is unset, leaves a set hook and an indexer-less backend alone, and registers nothing
when ``TRTLLM_M31_DISABLE_TP2CTX_TSEL`` is set; ``select`` declines every step outside its rows / blocks range, so
the built-in select runs there. The table itself is the GPU test's. CPU only.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import MsaIndexer

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        tile_select,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def _indexer(hook=None):
    indexer = object.__new__(MsaIndexer)  # the hook attribute only; no config is read here
    indexer.selector_hook = hook
    return indexer


def _model(*backends):
    layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=b)) for b in backends]
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def test_register_sets_every_unset_msa_hook_and_nothing_else(monkeypatch):
    monkeypatch.setattr(tile_select, "TSEL", True)
    other = object()
    free, taken = _indexer(), _indexer(hook=other)
    model = _model(
        types.SimpleNamespace(indexer=free),
        types.SimpleNamespace(indexer=taken),
        types.SimpleNamespace(),  # a backend without an MSA indexer
    )
    assert tile_select.register(model) == 1
    assert free.selector_hook is tile_select.select
    assert taken.selector_hook is other


def test_the_kill_switch_registers_nothing(monkeypatch):
    monkeypatch.setattr(tile_select, "TSEL", False)
    free = _indexer()
    assert tile_select.register(_model(types.SimpleNamespace(indexer=free))) == 0
    assert free.selector_hook is None


@pytest.mark.parametrize(
    "shape, dtype, why",
    [
        ((2, 64, tile_select.TSEL_MIN_ROWS - 1), torch.float32, "too few rows"),
        ((2, tile_select.TSEL_MAX_BLOCKS + 1, 512), torch.float32, "too many blocks"),
        ((2, 64, 512), torch.bfloat16, "not float32"),
    ],
)
def test_select_declines_outside_its_range(monkeypatch, shape, dtype, why):
    monkeypatch.setattr(
        tile_select, "_op", lambda: pytest.fail(f"the op was reached for a step with {why}")
    )
    scores = torch.zeros(shape, dtype=dtype)
    assert tile_select.select(scores, torch.ones(shape[2], dtype=torch.int32), 1, 2) is None


def test_select_declines_without_the_op(monkeypatch):
    monkeypatch.setattr(tile_select, "_op", lambda: None)
    scores = torch.zeros((2, 64, 512), dtype=torch.float32)
    assert tile_select.select(scores, torch.ones(512, dtype=torch.int32), 1, 2) is None
