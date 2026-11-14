# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash tp8 routing contract, CPU only: the family ``route`` with the target modules stubbed.

The M3.1 checkpoint on 8 tensor-parallel ranks (experts tensor-parallel, no attention DP) goes to the sm_107 tp8
target on Rubin (the GB300 tp8 target on sm103) and nowhere else; every other chip, parallel layout or KV cache, and
every checkpoint revision that is not the M3.1 export, is declined with ``NoTarget``. The full routing matrix lives in
test_modeling_v2_routing.py.
"""

import sys
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
_TP8 = ("minimax_m3_1_flash_nvfp4__sm_107__tp8", "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8")
_OTHERS = (
    ("minimax_m3_1_flash_nvfp4__sm_107__tp2", "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2"),
    ("minimax_m3_1_flash_nvfp4__sm_107__tp2ctx", "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx"),
    ("minimax_m3_1_flash_nvfp4__sm_107__dep2ctx", "ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx"),
    ("minimax_m3_1_flash_nvfp4__sm_107__dep", "ModelingV2MiniMaxM31FlashNvfp4Sm107Dep"),
    ("minimax_m3_1_flash_nvfp4__sm_107__tp4", "ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4"),
    ("minimax_m3_1_flash_nvfp4__sm_103__tp4", "ModelingV2MiniMaxM31FlashNvfp4Sm103Tp4"),
    ("minimax_m3_1_flash_nvfp4__sm_103__tp8", "ModelingV2MiniMaxM31FlashNvfp4Sm103Tp8"),
)


@pytest.fixture
def route(monkeypatch):
    """``route(config, mapping, capability, kv)`` with every M3.1 target module stubbed (records its class)."""
    for directory, name in (_TP8,) + _OTHERS:
        stub = types.ModuleType(f"{_FAMILY}.{directory}.modeling")
        setattr(stub, name, type(name, (), {"__module__": stub.__name__}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(routing, "_quant_algo", lambda _config: QuantAlgo.MXFP8)
    monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    monkeypatch.delenv("TRTLLM_M3_KV_POOL_ZERO_INIT", raising=False)

    def call(config, mapping, capability, kv):
        monkeypatch.setattr(routing, "engine_kv_cache_quant_algo", lambda *_args: kv)
        model_config = types.SimpleNamespace(
            pretrained_config=config, mapping=mapping, spec_config=None, quant_config=None
        )
        llm_args = types.SimpleNamespace(cache_transceiver_config=None)
        return routing.route(model_config, llm_args, capability)

    return call


def inputs():
    config = types.SimpleNamespace(
        text_config=dict(
            num_hidden_layers=60,
            hidden_size=6144,
            num_local_experts=128,
            intermediate_size=3072,
            num_attention_heads=64,
            num_key_value_heads=4,
            rope_theta=10_000_000,
            sparse_attention_config=dict(use_sparse_attention=True, sparse_attention_freq=[1] * 60),
        ),
        quantization_config=dict(moe_quant_algo="NVFP4", moe_quant_format="nvfp4-pack-quantized"),
    )
    mapping = types.SimpleNamespace(
        world_size=8,
        tp_size=8,
        pp_size=1,
        cp_size=1,
        moe_tp_size=8,
        moe_ep_size=1,
        enable_attention_dp=False,
    )
    return config, mapping


@pytest.mark.parametrize("kv", [QuantAlgo.FP8, QuantAlgo.NVFP4])
def test_tp8_routes_only_to_tp8(route, kv):
    config, mapping = inputs()
    target = route(config, mapping, (10, 7), kv)
    assert target.__name__ == _TP8[1] and target.__module__ == f"{_FAMILY}.{_TP8[0]}.modeling"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chip", (10, 0)),
        ("chip", (12, 0)),
        ("world_size", 4),
        ("tp_size", 4),
        ("pp_size", 2),
        ("cp_size", 2),
        ("moe_tp_size", 1),
        ("moe_ep_size", 2),
        ("m3_rope", None),
        ("m3_layers", None),
        ("wrong_experts", None),
        ("bf16_kv", None),
    ],
)
def test_declines_outside_scope(route, name, value):
    config, mapping = inputs()
    chip, kv = (10, 7), QuantAlgo.NVFP4
    if name == "chip":
        chip = value
    elif hasattr(mapping, name):
        setattr(mapping, name, value)
    elif name == "m3_rope":
        config.text_config["rope_theta"] = 5_000_000
    elif name == "m3_layers":
        config.text_config["sparse_attention_config"]["sparse_attention_freq"] = [0] * 3 + [1] * 57
    elif name == "wrong_experts":
        config.quantization_config["moe_quant_algo"] = "FP8"
    else:
        kv = None
    with pytest.raises(NoTarget):
        route(config, mapping, chip, kv)


def test_tp8_on_sm103_takes_the_gb300_target(route):
    config, mapping = inputs()
    target = route(config, mapping, (10, 3), QuantAlgo.NVFP4)
    assert target.__name__ == "ModelingV2MiniMaxM31FlashNvfp4Sm103Tp8"


def test_a_wrong_shape_is_declined(route):
    config, mapping = inputs()
    config.text_config["num_key_value_heads"] = 8
    with pytest.raises(NoTarget):
        route(config, mapping, (10, 7), QuantAlgo.NVFP4)
