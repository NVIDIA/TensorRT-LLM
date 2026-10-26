# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MiniMax-M3.1 deployment on 2 Rubin GPUs lands (routing.route), driven by synthetic configs.

The tp2 decode target takes tp 2 with the experts tensor-parallel (moe_tp 2, moe_ep 1); the tp2ctx context worker takes
tp 2 with expert parallelism (moe_tp 1, moe_ep 2), so the two never meet. ``TRTLLM_M31_DISABLE_TP2`` /
``TRTLLM_M31_DISABLE_TP2CTX`` keep the built-in model at their topologies. tp4 / tp8 route as before; GB300 (sm103)
has no tp2 target; attention DP, other widths and the MiniMax-M3 checkpoint never land on the tp2 target. CPU only.
"""

import sys
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches as tp2_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (  # noqa: E501
    m31_switches as tp2ctx_switches,
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
    "minimax_m31_nvfp4__sm_103__tp4": "ModelingV2MiniMaxM31Nvfp4Sm103Tp4",
    "minimax_m31_nvfp4__sm_103__tp8": "ModelingV2MiniMaxM31Nvfp4Sm103Tp8",
}
_DECODE = dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1)
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
def gates(monkeypatch):
    """``gates(tp2_off=..., ctx_off=...)``: TRTLLM_M31_DISABLE_TP2 on the tp2 target's switches and
    TRTLLM_M31_DISABLE_TP2CTX on the tp2ctx target's, set by patching the sets read at import (no reload)."""

    def set_(tp2_off=False, ctx_off=False):
        monkeypatch.setattr(tp2_switches, "_DISABLED", frozenset({"TP2"} if tp2_off else ()))
        monkeypatch.setattr(tp2ctx_switches, "_DISABLED", frozenset({"TP2CTX"} if ctx_off else ()))

    set_()
    return set_


def _route(monkeypatch, mapping_kwargs, capability=(10, 7), text=None, quant=QuantAlgo.MXFP8):
    """route() with each target's modeling module stubbed by a class of the target's name (no modeling import)."""
    for package, name in _TARGETS.items():
        stub = types.ModuleType(f"{_FAMILY}.{package}.modeling")
        setattr(stub, name, type(name, (), {"__module__": stub.__name__}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(routing, "_quant_algo", lambda _config: quant)
    pretrained = types.SimpleNamespace(
        text_config=dict(text or _M31_TEXT), quantization_config=dict(_M31_QUANT)
    )
    return routing.route(pretrained, Mapping(**mapping_kwargs), capability, None, QuantAlgo.NVFP4)


def test_parallel_names_the_decode_topology_tp2_and_not_the_context_one():
    assert routing._parallel(Mapping(**_DECODE)) == "tp2"
    assert routing._parallel(Mapping(**_CTX)) is None
    assert routing._tp2ctx(Mapping(**_CTX)) and not routing._tp2ctx(Mapping(**_DECODE))


@pytest.mark.parametrize("ctx_off", [False, True], ids=["ctx_default", "ctx_killed"])
def test_the_decode_topology_lands_on_the_tp2_target(monkeypatch, gates, ctx_off):
    gates(ctx_off=ctx_off)
    assert _route(monkeypatch, _DECODE).__name__ == "ModelingV2MiniMaxM31Nvfp4Sm107Tp2"


def test_the_context_topology_lands_on_tp2ctx_unless_killed(monkeypatch, gates):
    assert _route(monkeypatch, _CTX).__name__ == "ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx"
    gates(ctx_off=True)
    with pytest.raises(NoTarget, match="MiniMax-M3.1 requires"):
        _route(monkeypatch, _CTX)


def test_the_kill_switch_keeps_the_builtin_model(monkeypatch, gates):
    gates(tp2_off=True)
    with pytest.raises(NoTarget, match="MiniMax-M3.1 requires"):
        _route(monkeypatch, _DECODE)


def test_the_kill_switch_reads_the_tp2_targets_own_switches():
    assert routing._m31_tp2_routed() is (not tp2_switches.m31_disabled("TP2"))


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("tp2_off", [False, True], ids=["tp2_on", "tp2_killed"])
def test_tp4_and_tp8_route_as_before(monkeypatch, gates, tp, tp2_off):
    gates(tp2_off=tp2_off)
    mapping_kwargs = dict(world_size=tp, tp_size=tp, moe_tp_size=tp, moe_ep_size=1)
    assert _route(monkeypatch, mapping_kwargs).__name__ == f"ModelingV2MiniMaxM31Nvfp4Sm107Tp{tp}"


def test_gb300_has_no_tp2_target(monkeypatch, gates):
    with pytest.raises(NoTarget, match="MiniMax-M3.1 requires"):
        _route(monkeypatch, _DECODE, capability=(10, 3))


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(_DECODE, enable_attention_dp=True),
        dict(world_size=4, tp_size=2, pp_size=2, moe_tp_size=2, moe_ep_size=1),
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
    ],
    ids=["attention_dp", "tp2_pp2", "tp4_moetp2_ep2"],
)
def test_other_layouts_never_land_on_the_tp2_target(monkeypatch, gates, mapping_kwargs):
    with pytest.raises(NoTarget):
        _route(monkeypatch, mapping_kwargs)


def test_the_minimax_m3_checkpoint_never_lands_on_the_m31_tp2_target(monkeypatch, gates):
    m3_text = dict(
        _M31_TEXT,
        rope_theta=5_000_000,
        sparse_attention_config=dict(sparse_attention_freq=[0] * 3 + [1] * 57),
    )
    stub = types.ModuleType(f"{_FAMILY}.minimax_m3_nvfp4__sm_107__tp2.modeling")
    stub.ModelingV2MiniMaxM3Nvfp4Sm107Tp2 = type("ModelingV2MiniMaxM3Nvfp4Sm107Tp2", (), {})
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    try:
        target = _route(monkeypatch, _DECODE, text=m3_text, quant=QuantAlgo.MIXED_PRECISION)
    except NoTarget:
        return
    assert target.__name__ != "ModelingV2MiniMaxM31Nvfp4Sm107Tp2"
