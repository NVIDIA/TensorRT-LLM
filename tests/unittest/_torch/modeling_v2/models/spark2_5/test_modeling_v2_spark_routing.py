# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family routing for Spark2_5ForCausalLM: what the target admits and what
falls back to the built-in Modeling V1 implementation."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.configs import Spark2_5Config
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.models.spark2_5.routing import route
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_1_7B = dict(
    architectures=["Spark2_5ForCausalLM"],
    model_type="spark2_5",
    vocab_size=131072,
    hidden_size=2048,
    intermediate_size=6656,
    num_hidden_layers=28,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=256,
    hidden_act="gelu",
    rms_norm_eps=1e-6,
    max_position_embeddings=1048576,
    sliding_window=512,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    * 7,
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
    },
    headwise_attn_output_gate=True,
    gate_attn_act_mode="sigmoid",
    attention_bias=False,
    mlp_bias=False,
    tie_word_embeddings=True,
)

_4B_SHAPE = dict(
    hidden_size=2560,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=4,
)

_SM100 = (10, 0)


def _deployment(cfg_overrides=None, *, tp=1, spec=False, kv_dtype="auto", quant=False):
    fields = dict(_1_7B)
    fields.update(cfg_overrides or {})
    cfg = Spark2_5Config(**fields)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=0)
    model_config = SimpleNamespace(
        pretrained_config=cfg,
        mapping=mapping,
        spec_config=SimpleNamespace() if spec else None,
        quant_config=SimpleNamespace(
            kv_cache_quant_algo=None,
            quant_algo="FP8" if quant else None,
        ),
    )
    llm_args = SimpleNamespace(kv_cache_config=SimpleNamespace(dtype=kv_dtype))
    return model_config, llm_args


def _route(**kwargs):
    model_config, llm_args = _deployment(**kwargs)
    return route(model_config, llm_args, _SM100)


def test_routes_1_7b_tp1():
    cls = _route()
    assert cls.__name__ == "ModelingV2Spark25DenseSm10xDefault"


def test_routes_4b_shape():
    cls = _route(cfg_overrides=_4B_SHAPE)
    assert cls.__name__ == "ModelingV2Spark25DenseSm10xDefault"


def test_routes_sm103():
    model_config, llm_args = _deployment()
    cls = route(model_config, llm_args, (10, 3))
    assert cls.__name__ == "ModelingV2Spark25DenseSm10xDefault"


@pytest.mark.parametrize(
    "kwargs, why",
    [
        (dict(tp=2), "the target computes no collective"),
        (dict(spec=True), "no draft path"),
        (dict(kv_dtype="fp8"), "bf16 KV pool only"),
        (dict(quant=True), "BF16 checkpoints only"),
        (dict(cfg_overrides=dict(hidden_size=4096)), "unpublished shape"),
        (dict(cfg_overrides=dict(gate_attn_act_mode="silu")), "sigmoid gate only"),
        (
            dict(cfg_overrides=dict(sliding_window=None, layer_types=["full_attention"] * 28)),
            "no window for sliding",
        ),
        (dict(cfg_overrides=dict(attention_bias=True)), "biased projections"),
    ],
    ids=lambda v: v if isinstance(v, str) else "case",
)
def test_declines(kwargs, why):
    with pytest.raises(NoTarget):
        _route(**kwargs)


def test_declines_off_sm():
    model_config, llm_args = _deployment()
    with pytest.raises(NoTarget):
        route(model_config, llm_args, (9, 0))
