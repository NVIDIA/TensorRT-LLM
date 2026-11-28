# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU fault-injection checks for weight loading; no checkpoint weights or collectives."""

__extra_import_path__ = ["."]

from collections import OrderedDict
from types import SimpleNamespace

import _beam_reference as modeling_beam
import pytest
import torch
from _components.dense_block import block as dense  # noqa: E501

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def tiny_dense(monkeypatch):
    # Exercise the loader with tiny CPU storage, without allocating a GB300 block.
    block = dense.BeamDenseBlockBf16Sm103Tp1.__new__(dense.BeamDenseBlockBf16Sm103Tp1)
    torch.nn.Module.__init__(block)
    block._shapes = OrderedDict([("input_layernorm.weight", (2,)), ("mlp.up_proj.weight", (2, 2))])
    for name, shape in block._shapes.items():
        block.register_parameter(
            dense._attribute(name),
            torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False),
        )
    block._loaded = False
    block.layer_idx, block.num_kv_heads, block.head_dim = 0, 1, 2
    block.window = dense.SUPPORTED_CONFIG["sliding_window"]
    monkeypatch.setattr(dense._attention, "check_step", lambda *args, **kwargs: None)
    return block


def _weights(block, value):
    return {
        name: torch.full(shape, value, dtype=torch.bfloat16)
        for name, shape in block._shapes.items()
    }


@pytest.mark.parametrize("bad_kind", ["shape", "dtype", "object", "meta"])
def test_rejected_dense_reload_preserves_weights(tiny_dense, bad_kind):
    block = tiny_dense
    good = _weights(block, 1)
    block.load_weights(good)
    invalid = _weights(block, 2)
    invalid["mlp.up_proj.weight"] = {
        "shape": torch.zeros(3, dtype=torch.bfloat16),
        "dtype": torch.zeros(2, 2, dtype=torch.float32),
        "object": None,
        "meta": torch.empty(2, 2, dtype=torch.bfloat16, device="meta"),
    }[bad_kind]
    with pytest.raises((TypeError, ValueError)):
        block.load_weights(invalid)
    assert block._loaded
    assert all(
        torch.equal(tensor, good[name]) for name, tensor in block.checkpoint_tensors().items()
    )


def test_dense_cannot_run_before_load_or_after_failed_copy(tiny_dense, monkeypatch):
    block = tiny_dense
    with pytest.raises(ValueError, match="load_weights"):
        block(None, None, None)
    block.load_weights(_weights(block, 1))
    original_copy = torch.Tensor.copy_

    def fail_second(dst, src, *args, **kwargs):
        if dst is block.mlp_up_proj:
            raise RuntimeError("injected copy failure")
        return original_copy(dst, src, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_second)
        with pytest.raises(RuntimeError, match="copy failure"):
            block.load_weights(_weights(block, 2))
    assert not block._loaded
    with pytest.raises(ValueError, match="load_weights"):
        block(None, None, None)
    block.load_weights(_weights(block, 3))
    assert block._loaded


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_builtin_loader_rejects_tp_before_copy(monkeypatch, tp_size):
    calls = []
    monkeypatch.setattr(modeling_beam, "load_beam_state_dict", lambda *args: calls.append(args))
    model = SimpleNamespace(model_config=SimpleNamespace(mapping=SimpleNamespace(tp_size=tp_size)))
    if tp_size > 1:
        with pytest.raises(ValueError, match="unsharded"):
            modeling_beam.BeamForCausalLM.load_weights(model, {})
        assert calls == []
    else:
        modeling_beam.BeamForCausalLM.load_weights(model, {})
        assert calls == [(model, {})]
