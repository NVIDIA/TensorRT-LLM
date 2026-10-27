# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the MiniMax-M3.1 context worker's deployment lands (routing.route), driven by synthetic configs.

The tp2ctx target takes exactly the disaggregated context role's topology (tp 2, moe_ep 2, attention DP off), on by
default; ``TRTLLM_M31_DISABLE_TP2CTX`` keeps the built-in model (NoTarget). Attention DP, the aggregated tp2 layout and
the MiniMax-M3 checkpoint never land on it, and the tp2 / tp4 / tp8 targets route as before. CPU only.
"""

import sys
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
_TARGETS = {
    "minimax_m31_nvfp4__sm_107__tp2": "ModelingV2MiniMaxM31Nvfp4Sm107Tp2",
    "minimax_m31_nvfp4__sm_107__tp4": "ModelingV2MiniMaxM31Nvfp4Sm107Tp4",
    "minimax_m31_nvfp4__sm_107__tp8": "ModelingV2MiniMaxM31Nvfp4Sm107Tp8",
    "minimax_m31_nvfp4__sm_107__tp2ctx": "ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx",
}
_CTX = dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2)
_M31_TEXT = dict(
    num_hidden_layers=60,
    hidden_size=6144,
    num_local_experts=128,
    intermediate_size=3072,
    num_attention_heads=64,
    num_key_value_heads=4,
    rope_theta=10_000_000,
    sparse_attention_config=dict(sparse_attention_freq=[1] * 60),
)
_M31_QUANT = dict(
    quant_method="mxfp8", moe_quant_algo="NVFP4", moe_quant_format="nvfp4-pack-quantized"
)


@pytest.fixture
def switches(monkeypatch):
    """Set the tp2ctx kill switch for one case: ``switches(disable=...)``. Patches the switch set read at import
    rather than reloading the module, so no other case sees the setting."""

    def set_(disable=False):
        monkeypatch.setattr(m31_switches, "_DISABLED", frozenset({"TP2CTX"} if disable else ()))

    set_()
    return set_


def _route(monkeypatch, mapping_kwargs, text=None, quant=QuantAlgo.MXFP8):
    """route() with each target's modeling module stubbed by a class of the target's name (no modeling import)."""
    for package, name in _TARGETS.items():
        stub = types.ModuleType(f"{_FAMILY}.{package}.modeling")
        setattr(stub, name, type(name, (), {"__module__": stub.__name__}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(routing, "_quant_algo", lambda _config: quant)
    pretrained = types.SimpleNamespace(
        text_config=dict(text or _M31_TEXT), quantization_config=dict(_M31_QUANT)
    )
    return routing.route(
        pretrained, Mapping(**mapping_kwargs), routing._SM_RUBIN, None, QuantAlgo.NVFP4
    )


def test_the_context_role_routes_to_tp2ctx_by_default(monkeypatch, switches):
    assert _route(monkeypatch, _CTX).__name__ == "ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx"


def test_the_kill_switch_keeps_the_builtin_model(monkeypatch, switches):
    switches(disable=True)
    with pytest.raises(NoTarget, match="MiniMax-M3.1 requires"):
        _route(monkeypatch, _CTX)


def test_the_kill_switch_reads_the_tp2ctx_targets_own_switches():
    assert routing._tp2ctx_routed() is (not m31_switches.m31_disabled("TP2CTX"))


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(_CTX, enable_attention_dp=True),  # attention DP: the built-in model's fallback layout
        dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),  # wider expert parallelism
        dict(world_size=4, tp_size=2, pp_size=2, moe_tp_size=1, moe_ep_size=2),  # pipeline parallel
    ],
    ids=["attention_dp", "tp4_ep4", "tp2_ep2_pp2"],
)
def test_other_topologies_never_land_on_tp2ctx(monkeypatch, switches, mapping_kwargs):
    with pytest.raises(NoTarget):
        _route(monkeypatch, mapping_kwargs)


def test_the_aggregated_tp2_layout_lands_on_the_tp2_target(monkeypatch, switches):
    mapping_kwargs = dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1)
    assert _route(monkeypatch, mapping_kwargs).__name__ == "ModelingV2MiniMaxM31Nvfp4Sm107Tp2"


def test_the_minimax_m3_checkpoint_never_lands_on_tp2ctx(monkeypatch, switches):
    m3_text = dict(
        _M31_TEXT,
        rope_theta=5_000_000,
        sparse_attention_config=dict(sparse_attention_freq=[0] * 3 + [1] * 57),
    )
    stub = types.ModuleType(f"{_FAMILY}.minimax_m3_nvfp4__sm_107__tp2.modeling")
    stub.ModelingV2MiniMaxM3Nvfp4Sm107Tp2 = type("ModelingV2MiniMaxM3Nvfp4Sm107Tp2", (), {})
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    try:
        target = _route(monkeypatch, _CTX, text=m3_text, quant=QuantAlgo.MIXED_PRECISION)
    except NoTarget:
        return
    assert target.__name__ != "ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx"


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("disable", [False, True], ids=["tp2ctx_on", "tp2ctx_killed"])
def test_tp4_and_tp8_route_as_before(monkeypatch, switches, tp, disable):
    switches(disable=disable)
    mapping_kwargs = dict(world_size=tp, tp_size=tp, moe_tp_size=tp, moe_ep_size=1)
    assert _route(monkeypatch, mapping_kwargs).__name__ == f"ModelingV2MiniMaxM31Nvfp4Sm107Tp{tp}"
