# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU fault-injection checks for weight loading; no checkpoint weights or collectives."""

import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.dense_block_bf16__sm_103__tp1 import (  # noqa: E501
    block as dense,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    component as nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    component_bf16 as bf16,
)
from tensorrt_llm._torch.models import modeling_beam

pytestmark = pytest.mark.cpu_only


@pytest.fixture(params=[bf16, nvfp4], ids=["bf16", "nvfp4"])
def layout_loader(request, monkeypatch, tmp_path):
    module = request.param
    cls = (
        bf16.BeamCheckpointLayoutBf16Sm103Tp4
        if module is bf16
        else nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "hf_quant_config.json").write_text("{}")
    spec = SimpleNamespace(
        name="weight", bank=None, expert=None, interval=lambda rank: (0, 2), local_shape=(2,)
    )
    dst = torch.zeros(2, dtype=torch.bfloat16)
    component = SimpleNamespace(
        config={},
        rank=0,
        layers=(),
        expert_names=(),
        specs=OrderedDict(weight=spec),
        usable=True,
        load_record={"old": True},
        tensor=lambda name: dst,
    )
    component.schedule = SimpleNamespace(is_sparse=lambda layer: False)
    agreements = []

    def agree(ok):
        agreements.append(ok)
        return ok

    component._agree = agree
    component._check_diagnostics_agree = lambda: None
    catalog = SimpleNamespace(
        entries={"weight": SimpleNamespace(path="part", begin=0)},
        validate=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(module, "SafetensorsCatalog", lambda path: catalog)
    monkeypatch.setattr(module, "tensor_specs", lambda config: {})
    if module is nvfp4:
        monkeypatch.setattr(module, "check_recipe", lambda *args: {})
        monkeypatch.setattr(module, "stored_entries", lambda *args: {})
        monkeypatch.setattr(module, "validate", lambda *args, **kwargs: None)
    reader = SimpleNamespace(bytes_read=4, reads=1, closed=False)

    def close():
        reader.closed = True

    reader.close = close
    monkeypatch.setattr(module, "TileReader", lambda: reader)
    monkeypatch.setattr(module, "read_slice_into", lambda *args, **kwargs: dst.fill_(1))
    return module, cls.load, component, agreements, catalog, reader, tmp_path


@pytest.mark.parametrize("stored", [None, ["bad"]])
def test_malformed_config_reaches_validation_agreement(layout_loader, stored):
    _, load, component, agreements, _, _, path = layout_loader
    (path / "config.json").write_text(json.dumps(stored))
    with pytest.raises(bf16.LoadError) as exc:
        load(component, str(path))
    assert isinstance(exc.value.__cause__, (TypeError, AttributeError))
    assert agreements == [False]
    assert component.usable  # Validation rejected the reload before touching storage.
    assert component.load_record == {"old": True}


@pytest.mark.parametrize("stage", ["catalog", "reader", "ordering", "read", "close"])
def test_rank_local_exception_reaches_agreement(layout_loader, monkeypatch, stage):
    module, load, component, agreements, catalog, reader, path = layout_loader
    failure = TypeError("injected rank-local failure")

    def fail(*args, **kwargs):
        raise failure

    if stage == "catalog":
        monkeypatch.setattr(module, "SafetensorsCatalog", fail)
    elif stage == "reader":
        monkeypatch.setattr(module, "TileReader", fail)
    elif stage == "ordering":
        catalog.entries.clear()
    elif stage == "read":
        monkeypatch.setattr(module, "read_slice_into", fail)
    else:
        reader.close = fail
    with pytest.raises(bf16.LoadError) as exc:
        load(component, str(path))
    assert agreements == ([False] if stage == "catalog" else [True, False])
    if stage == "ordering":
        assert isinstance(exc.value.__cause__, KeyError)
    else:
        assert exc.value.__cause__ is failure
    if stage != "catalog":
        assert not component.usable and component.load_record is None
    if stage in ("ordering", "read"):
        assert reader.closed


def test_successful_load_and_peer_failure(layout_loader):
    _, load, component, agreements, _, reader, path = layout_loader
    record = load(component, str(path))
    assert agreements == [True, True]
    assert component.usable and record["tensors"] == 1
    assert reader.closed and torch.equal(component.tensor("weight"), torch.ones(2))
    agreements.clear()

    def peer_fails_completion(ok):
        agreements.append(ok)
        return len(agreements) == 1

    component._agree = peer_fails_completion
    with pytest.raises(bf16.LoadError, match="every rank"):
        load(component, str(path))
    assert agreements == [True, True]
    assert not component.usable and component.load_record is None


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
