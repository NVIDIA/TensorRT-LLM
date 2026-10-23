# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
import transformers

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import (
    PplxDeciderHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_pplx_decider import (
    PPLX_DECIDER_ARCHITECTURE,
    PPLX_DECIDER_NUM_LABELS,
    PplxDeciderForDecision,
    PplxDeciderFullAttentionDecoderLayer,
    PplxDeciderGatedDeltaNet,
    PplxDeciderLinearDecoderLayer,
    PplxDeciderModel,
)
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextFullAttentionDecoderLayer,
    Qwen3NextLinearDecoderLayer,
)
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

pytestmark = pytest.mark.cpu_only


def _write_composite_config(tmp_path: Path) -> Path:
    config = {
        "architectures": ["Qwen3_5Model"],
        "dtype": "bfloat16",
        "model_type": "qwen3_5",
        "text_config": {
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "vocab_size": 248320,
        },
        "vision_config": {"model_type": "qwen3_5_vision"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_config_override_flattens_composite_checkpoint(tmp_path: Path) -> None:
    config = load_pretrained_config(
        str(_write_composite_config(tmp_path)),
        model_kwargs={"architectures": [PPLX_DECIDER_ARCHITECTURE]},
    )

    assert isinstance(config, transformers.Qwen3NextConfig)
    assert config.architectures == [PPLX_DECIDER_ARCHITECTURE]
    assert config.num_labels == PPLX_DECIDER_NUM_LABELS
    assert config.hidden_size == 5120
    assert not hasattr(config, "vision_config")


def test_model_registration_and_routing() -> None:
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class

    assert get_registered_model_class(PPLX_DECIDER_ARCHITECTURE) is PplxDeciderForDecision
    assert ModelConfig.is_generation_model([PPLX_DECIDER_ARCHITECTURE]) is False
    assert PplxDeciderForDecision.get_model_defaults(None) == {"attn_backend": "FLASHINFER"}


def test_decider_layer_overrides_are_isolated() -> None:
    assert PplxDeciderModel.decoder_layer_types == {
        "full_attention": PplxDeciderFullAttentionDecoderLayer,
        "linear_attention": PplxDeciderLinearDecoderLayer,
    }
    assert PplxDeciderLinearDecoderLayer.linear_attn_cls is PplxDeciderGatedDeltaNet
    assert Qwen3NextLinearDecoderLayer.linear_attn_cls is not PplxDeciderGatedDeltaNet
    assert PplxDeciderFullAttentionDecoderLayer.attention_mask.name == "FULL"
    assert Qwen3NextFullAttentionDecoderLayer.attention_mask.name == "CAUSAL"


def test_native_checkpoint_weight_names_are_mapped() -> None:
    mapper = AutoCheckpointMapper.get("HF", PPLX_DECIDER_ARCHITECTURE)
    backbone = torch.empty(1)
    readout = torch.empty(1)

    assert isinstance(mapper, PplxDeciderHfWeightMapper)
    normalized = mapper._normalize_weight_names(
        {
            "language_model.layers.0.input_layernorm.weight": backbone,
            "weight": readout,
            "visual.blocks.0.attn.qkv.weight": torch.empty(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.empty(1),
        }
    )
    assert set(normalized) == {
        "model.layers.0.input_layernorm.weight",
        "readout.weight",
    }
    assert normalized["model.layers.0.input_layernorm.weight"] is backbone
    assert normalized["readout.weight"] is readout
