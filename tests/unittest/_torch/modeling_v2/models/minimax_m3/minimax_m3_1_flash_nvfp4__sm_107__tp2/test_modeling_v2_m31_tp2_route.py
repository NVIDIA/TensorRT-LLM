# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MiniMax-M3.1-Flash deployment on 2 Rubin GPUs lands (routing.route), driven by synthetic configs.

The tp2 decode target takes tp 2 with the experts tensor-parallel (moe_tp 2, moe_ep 1); the tp2ctx context worker takes
tp 2 with expert parallelism (moe_tp 1, moe_ep 2), so the two never meet. tp8 routes to its own target; GB300 (sm103)
has no M3.1 target; a bf16 KV cache, other layouts and the MiniMax-M3 checkpoint never land on the tp2 target. The
target modules are stubbed: this checks the decision, not that the modeling code imports on this host. CPU only.
"""

import sys
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
_TARGETS = {
    "minimax_m3_1_flash_nvfp4__sm_107__tp2": "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2",
    "minimax_m3_1_flash_nvfp4__sm_107__tp8": "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8",
    "minimax_m3_1_flash_nvfp4__sm_107__tp2ctx": "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx",
    "minimax_m3_nvfp4__sm_107__tp2": "ModelingV2MiniMaxM3Nvfp4Sm107Tp2",
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


def _route(
    monkeypatch, mapping_kwargs, capability=(10, 7), text=None, quant=QuantAlgo.MXFP8, kv="nvfp4"
):
    """route() with each target's modeling module stubbed by a class of the target's name (no modeling import)."""
    for package, name in _TARGETS.items():
        stub = types.ModuleType(f"{_FAMILY}.{package}.modeling")
        setattr(stub, name, type(name, (), {"__module__": stub.__name__}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(routing, "_quant_algo", lambda _config: quant)
    model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(
            text_config=dict(text or _M31_TEXT), quantization_config=dict(_M31_QUANT)
        ),
        mapping=Mapping(**mapping_kwargs),
        spec_config=None,
        quant_config=types.SimpleNamespace(kv_cache_quant_algo=None),
    )
    llm_args = TorchLlmArgs(model="/tmp/dummy_model", kv_cache_config=dict(dtype=kv))
    return routing.route(model_config, llm_args, capability)


def test_the_topology_names_the_decode_and_the_context_worker_apart():
    assert routing._m31_topology(Mapping(**_DECODE)) == "tp2"
    assert routing._m31_topology(Mapping(**_CTX)) == "tp2ctx"
    assert routing._tp2ctx(Mapping(**_CTX)) and not routing._tp2ctx(Mapping(**_DECODE))


def test_the_decode_topology_lands_on_the_tp2_target(monkeypatch):
    assert _route(monkeypatch, _DECODE).__name__ == "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2"


def test_the_context_topology_lands_on_tp2ctx(monkeypatch):
    assert _route(monkeypatch, _CTX).__name__ == "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx"


def test_tp8_lands_on_its_own_target(monkeypatch):
    mapping_kwargs = dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert _route(monkeypatch, mapping_kwargs).__name__ == "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8"


def test_gb300_has_no_m31_target(monkeypatch):
    with pytest.raises(NoTarget, match="MiniMax-M3.1-Flash: not sm107"):
        _route(monkeypatch, _DECODE, capability=(10, 3))


def test_a_bf16_kv_cache_has_no_m31_target(monkeypatch):
    with pytest.raises(NoTarget, match="a KV cache other than FP8 or NVFP4"):
        _route(monkeypatch, _DECODE, kv="auto")


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(world_size=4, tp_size=2, pp_size=2, moe_tp_size=2, moe_ep_size=1),
        dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
    ],
    ids=["tp2_pp2", "tp4", "tp4_moetp2_ep2"],
)
def test_other_layouts_never_land_on_the_tp2_target(monkeypatch, mapping_kwargs):
    with pytest.raises(NoTarget, match="a parallel layout no target runs"):
        _route(monkeypatch, mapping_kwargs)


def test_the_minimax_m3_checkpoint_never_lands_on_the_m31_tp2_target(monkeypatch):
    m3_text = dict(
        _M31_TEXT,
        rope_theta=5_000_000,
        sparse_attention_config=dict(sparse_attention_freq=[0] * 3 + [1] * 57),
    )
    try:
        target = _route(monkeypatch, _DECODE, text=m3_text, quant=QuantAlgo.MIXED_PRECISION)
    except NoTarget:
        return
    assert target.__name__ != "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2"
