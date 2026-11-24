# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``FlatModel._capture_rows`` / ``FlatChainRuntime._tap``: which captured taps the AttnRes stores
straight into DFlash's capture buffer (no slice copy) and where.

The rows must be exactly the block ``DFlashSpecMetadata.maybe_capture_hidden_states`` copies into
(``captured_hidden_states[:T, i*H:(i+1)*H]`` for the i-th captured layer), keyed by the layer whose
AttnRes makes the tap (captured layer + 1; 93 for layer 92). Every other capture mode keeps the
callable: no metadata, a bound projection, a non-BF16 buffer, another metadata class.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
H = 7168
DEV = "cuda"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the slice copy needs CUDA")


def _modeling():
    return importlib.import_module(_PKG + ".modeling")


def _metadata(layers=(7, 23, 51, 67, 83, 91, 92), rows=16, dtype=torch.bfloat16):
    m = DFlashSpecMetadata.__new__(DFlashSpecMetadata)
    m.hidden_size = H
    m._capture_projection = None
    m._layer_to_idx = {layer: i for i, layer in enumerate(sorted(layers))}
    m._capture_layer_set = frozenset(layers)
    m.captured_hidden_states = torch.zeros((rows, H * len(layers)), dtype=dtype, device=DEV)
    return m


def test_rows_are_the_slice_copys_destination():
    m = _metadata()
    got = _modeling().FlatModel._capture_rows(m, 8)
    assert sorted(got) == [8, 24, 52, 68, 84, 92, 93]
    buffer = m.captured_hidden_states
    for layer, i in m._layer_to_idx.items():
        rows = got[layer + 1]
        assert rows.shape == (8, H)
        assert rows.data_ptr() == buffer[:, i * H :].data_ptr()
        assert rows.stride() == buffer.stride()
    # The copy's own semantics, for reference: the same block.
    value = torch.randn(8, H, device=DEV).to(torch.bfloat16)
    m.maybe_capture_hidden_states(51, value, None)
    assert torch.equal(got[52], value)


def test_other_capture_modes_keep_the_callable():
    capture_rows = _modeling().FlatModel._capture_rows
    assert capture_rows(None, 8) == {}
    assert (
        capture_rows(SimpleNamespace(captured_hidden_states=torch.zeros(8, H, device=DEV)), 8) == {}
    )
    projected = _metadata()
    projected._capture_projection = [torch.zeros(4, H, device=DEV)]
    assert capture_rows(projected, 8) == {}
    assert capture_rows(_metadata(dtype=torch.float32), 8) == {}
    empty = _metadata()
    empty.captured_hidden_states = None
    assert capture_rows(empty, 8) == {}

    class Other(DFlashSpecMetadata):
        def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None):
            pass

    other = _metadata()
    other.__class__ = Other
    assert capture_rows(other, 8) == {}


def test_tap_uses_the_bound_rows_else_the_scratch():
    runtime = _modeling().FlatChainRuntime.__new__(_modeling().FlatChainRuntime)
    runtime.tap = torch.zeros(64, H, dtype=torch.bfloat16, device=DEV)
    rows = _modeling().FlatModel._capture_rows(_metadata(), 8)
    runtime.bind_captures(rows)
    assert runtime._tap(52, 8).data_ptr() == rows[52].data_ptr()
    assert runtime._tap(93, 8).data_ptr() == rows[93].data_ptr()
    assert runtime._tap(53, 8).data_ptr() == runtime.tap.data_ptr()
    assert runtime._tap(53, 8).shape == (8, H)
    runtime.bind_captures({})
    assert runtime._tap(52, 8).data_ptr() == runtime.tap.data_ptr()
