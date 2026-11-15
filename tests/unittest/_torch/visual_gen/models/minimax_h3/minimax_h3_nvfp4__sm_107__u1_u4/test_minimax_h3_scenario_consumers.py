# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The modules of the MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4`` hold the process's one
configuration object (SM 107 only: the modules build their kernels at import)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    config as scenario_config,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (10, 7),
    reason="the scenario's kernels are SM 107 only",
)

_PACKAGE = scenario_config.__name__.rsplit(".", 1)[0]
_CONSUMERS = (
    "kernels.fused_fp8_qk",
    "kernels.fused_out_a2a",
    "kernels.fused_out_quant",
    "kernels.fused_quant",
    "kernels.vae_decoder_fp8",
    "modeling",
    "pipeline",
)


@pytest.mark.parametrize("name", _CONSUMERS)
def test_consumer_holds_the_selected_config(name):
    module = importlib.import_module(f"{_PACKAGE}.{name}")
    assert module.SCENARIO is scenario_config.SCENARIO


def test_differing_selection_is_refused_once_consumers_are_loaded():
    for name in _CONSUMERS:
        importlib.import_module(f"{_PACKAGE}.{name}")
    in_use = scenario_config.SCENARIO
    with pytest.raises(RuntimeError, match="one scenario configuration per process"):
        scenario_config.apply({"out_quant": not in_use.out_quant})
    assert scenario_config.SCENARIO is in_use


def test_scenario_transformer_declares_per_layer_quant_support():
    modeling = importlib.import_module(f"{_PACKAGE}.modeling")
    assert modeling.ScenarioTransformer._supports_per_layer_quant


_ARCHITECTURE: dict[str, object] = {
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 3,
    "num_refiner_layers": 2,
    "ffn_dim": 64,
    "in_channels": 16,
    "audio_in_channels": 16,
    "patch_size": (1, 1, 1),
    "text_dim": 64,
    "freq_dim": 8,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 64,
    "rope_freq_dim": 4,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}


@pytest.mark.parametrize(
    "layers, reason",
    [
        ({"transformer_blocks.1.attn": "fp8"}, "only 'bf16'"),
        ({"transformer_blocks.9.attn": "bf16"}, "not attention modules"),
    ],
)
def test_scenario_transformer_refuses_a_bad_attention_precision_before_any_attention(
    monkeypatch, layers, reason
):
    """The route does not parse the entry; the scenario's transformer must, before it builds."""
    modeling = importlib.import_module(f"{_PACKAGE}.modeling")

    def no_attention(*args, **kwargs):
        raise AssertionError("an attention was built before the entry was validated")

    monkeypatch.setattr(h3.MiniMaxH3Attention, "__init__", no_attention)
    monkeypatch.setattr(modeling.ScenarioAttention, "__init__", no_attention)
    entry = {"schema_version": 1, "layers": layers}
    pretrained = SimpleNamespace(
        **_ARCHITECTURE, quantization_config={"quant_algo": "NVFP4", "attention_precision": entry}
    )
    model_config = DiffusionModelConfig(
        pretrained_config=pretrained,
        quant_config=QuantConfig(),
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
        mapping=Mapping(),
        attention=AttentionConfig(backend="CUDNN"),
    )
    with pytest.raises(ValueError, match=reason):
        modeling.ScenarioTransformer(model_config)
