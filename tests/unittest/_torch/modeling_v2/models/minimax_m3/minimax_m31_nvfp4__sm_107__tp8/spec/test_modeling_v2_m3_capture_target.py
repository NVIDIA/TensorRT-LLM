# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker's capture target (the capture norm in the boundary all-reduce): the capture buffer's column block of a
captured layer with its fc_norm weight and eps; None without an armed capture norm, for a layer that is not captured,
past the buffer's rows, or when the loaded all-reduce op has no capture epilogue (then the capture_norm kernel runs)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_mc_allreduce,
)

H = 64
LAYERS = (2, 30, 58)


def _setup(armed=True):
    hidden = torch.arange(8 * 3 * H, dtype=torch.float32).reshape(8, 3 * H).to(torch.bfloat16)
    weights = torch.randn(3, H).to(torch.bfloat16)
    meta = SimpleNamespace(hidden_states=hidden, hidden_size=H, layers_to_capture=LAYERS)
    worker = SimpleNamespace(_m3_capture_norm=(weights, 1e-5) if armed else None)
    return worker, meta, hidden, weights


def _target(worker, meta, layer_id, num_tokens):
    return modeling.M3Eagle3Worker.capture_target(worker, meta, layer_id, num_tokens)


@pytest.fixture
def supported(monkeypatch):
    monkeypatch.setattr(m3_mc_allreduce, "capture_supported", lambda: True)


def test_column_block_of_each_captured_layer(supported) -> None:
    worker, meta, hidden, weights = _setup()
    for i, layer_id in enumerate(LAYERS):
        out, w, eps = _target(worker, meta, layer_id, 4)
        assert out.shape == (4, H)
        assert out.data_ptr() == hidden[:4, i * H : (i + 1) * H].data_ptr()
        assert out.stride() == hidden.stride()
        assert torch.equal(w, weights[i])
        assert eps == 1e-5


def test_none_when_not_capturable(supported) -> None:
    worker, meta, _, _ = _setup()
    assert _target(worker, meta, 3, 4) is None  # not a captured layer
    assert _target(worker, meta, LAYERS[0], 9) is None  # past the buffer's rows
    unarmed, meta2, _, _ = _setup(armed=False)
    assert _target(unarmed, meta2, LAYERS[0], 4) is None


def test_none_without_the_capture_epilogue(monkeypatch) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "capture_supported", lambda: False)
    worker, meta, _, _ = _setup()
    assert _target(worker, meta, LAYERS[0], 4) is None


def test_capture_supported_reads_the_op_schema() -> None:
    # The schema of the op the boundaries call: this target's copy (m3_mc_allreduce.OWN_OP), else the image's.
    name = (
        "minimax_m3_sm107_tp8_mc_allreduce_norm"
        if m3_mc_allreduce.own_op()
        else "minimax_m3_sm107_tp2_mc_allreduce_norm"
    )
    op = getattr(torch.ops.trtllm, name, None)
    want = op is not None and any(a.name == "cap_out" for a in op.default._schema.arguments)
    assert m3_mc_allreduce.capture_supported() is want


def test_own_op_follows_the_kernel_constant(monkeypatch) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "OWN_OP", False)
    assert m3_mc_allreduce.own_op() is False
    monkeypatch.setattr(m3_mc_allreduce, "OWN_OP", True)
    loaded = getattr(torch.ops.trtllm, "minimax_m3_sm107_tp8_mc_allreduce_norm", None) is not None
    assert m3_mc_allreduce.own_op() is loaded
