# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PROJ_PIN on MXFP8 Linear modules, CPU only.

The install wraps each layer's qkv_proj and o_proj whose shape the pinned-tactic table covers, so that their calls,
and no other, run with the table engaged. The pinned GEMMs' bitwise checks are the table's GPU tests.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        proj_pin,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

QKV, O_PROJ, SHARED = (9856, 6144), (6144, 8192), (6144, 6144)
IMPORTED_DEP_PROJ_PIN = proj_pin.DEP_PROJ_PIN


@pytest.fixture(autouse=True)
def _cpu_mxfp8(monkeypatch):
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: False)
    monkeypatch.setattr(proj_pin, "DEP_PROJ_PIN", True)


def _linear(shape, algo=QuantAlgo.MXFP8):
    n, k = shape
    quant_config = QuantConfig(quant_algo=algo, group_size=32) if algo is not None else None
    return Linear(
        in_features=k, out_features=n, bias=False, dtype=torch.bfloat16, quant_config=quant_config
    )


def _layer(qkv=QKV, o_proj=O_PROJ):
    attn = torch.nn.Module()
    attn.qkv_proj = _linear(qkv) if qkv is not None else None
    attn.o_proj = _linear(o_proj) if o_proj is not None else None
    layer = torch.nn.Module()
    layer.self_attn = attn
    return layer


def _config(**extra_attrs):
    return types.SimpleNamespace(extra_attrs=extra_attrs)


def _recorded_forward(monkeypatch):
    """Linear.forward replaced by a recorder of the table's caller during the call."""
    calls = []

    def forward(self, x, **kwargs):
        calls.append((pin._CALLER.get(), kwargs))
        return x

    monkeypatch.setattr(Linear, "forward", forward)
    return calls


def test_install_wraps_both_projections_of_every_layer():
    layers = [_layer(), _layer()]
    assert proj_pin.install(layers, _config()) == 4
    for layer in layers:
        for name in proj_pin.PROJECTIONS:
            proj = getattr(layer.self_attn, name)
            assert proj.forward.__func__ is proj_pin._pinned_forward and proj._dep_proj_pin == name
    assert proj_pin.install(layers, _config()) == 4


def test_a_wrapped_projection_runs_its_call_with_the_table_engaged(monkeypatch):
    calls = _recorded_forward(monkeypatch)
    layer = _layer()
    proj_pin.install([layer], _config())
    x = torch.zeros(3, 8)
    assert layer.self_attn.o_proj(x, all_reduce_params="p") is x
    layer.self_attn.qkv_proj(x)
    assert calls == [
        ("DEP_PROJ_PIN o_proj", {"all_reduce_params": "p"}),
        ("DEP_PROJ_PIN qkv_proj", {}),
    ]
    assert pin._CALLER.get() is None


def test_the_engagement_ends_when_the_call_raises(monkeypatch):
    def forward(self, x, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(Linear, "forward", forward)
    layer = _layer()
    proj_pin.install([layer], _config())
    with pytest.raises(ValueError, match="boom"):
        layer.self_attn.qkv_proj(torch.zeros(1, 8))
    assert pin._CALLER.get() is None


def test_other_linears_stay_unengaged(monkeypatch):
    """The shared expert's [6144, 6144] entry stays unarmed: only the attention projections are wrapped."""
    calls = _recorded_forward(monkeypatch)
    layer = _layer()
    layer.shared = _linear(SHARED)
    proj_pin.install([layer], _config())
    assert "forward" not in layer.shared.__dict__
    layer.shared(torch.zeros(1, 8))
    assert calls == [(None, {})]


@pytest.mark.parametrize(
    "make, reason",
    [
        pytest.param(lambda: torch.nn.Identity(), "Identity is not a Linear", id="not-linear"),
        pytest.param(
            lambda: _linear(QKV, algo=None), "quant method UnquantizedLinearMethod", id="bf16"
        ),
        pytest.param(lambda: _linear((4992, 6144)), None, id="tp2-shape"),
        pytest.param(
            lambda: _linear((256, 6144)), "no pinned tactic for [256, 6144]", id="no-entry"
        ),
    ],
)
def test_proj_declines(make, reason):
    assert proj_pin.proj_decline_reason(make()) == reason


def test_install_leaves_a_forward_something_else_replaced():
    layer = _layer()
    other = types.MethodType(lambda self, x: x, layer.self_attn.o_proj)
    layer.self_attn.o_proj.forward = other
    assert proj_pin.install([layer], _config()) == 1
    assert layer.self_attn.o_proj.forward is other
    assert ("not installed", "o_proj: its forward is already replaced") in proj_pin._LOGGED


def test_install_skips_layers_without_attention_projections():
    dense = types.SimpleNamespace(self_attn=None)
    assert proj_pin.install([dense, _layer(qkv=None)], _config()) == 1


def test_install_sets_nothing_when_off(monkeypatch):
    monkeypatch.setattr(proj_pin, "DEP_PROJ_PIN", False)
    layer = _layer()
    assert proj_pin.install([layer], _config()) == 0
    assert "forward" not in layer.self_attn.qkv_proj.__dict__


def test_install_sets_nothing_under_torch_compile():
    layer = _layer()
    assert proj_pin.install([layer], _config(torch_compile_enabled=True)) == 0
    assert "forward" not in layer.self_attn.qkv_proj.__dict__
    assert ("not installed", "torch.compile is on") in proj_pin._LOGGED


def test_the_switch_is_the_dep_proj_pin_item():
    assert IMPORTED_DEP_PROJ_PIN == (not m31_switches.m31_disabled("DEP_PROJ_PIN"))
