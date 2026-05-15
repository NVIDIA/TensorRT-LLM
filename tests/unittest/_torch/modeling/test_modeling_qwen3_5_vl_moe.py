# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch
import transformers

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import Qwen3_5MoeVLModel
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    load_pretrained_config,
)
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype
from tensorrt_llm.inputs import ContentFormat
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY


def _write_qwen35_moe_vl_config(tmp_path: Path) -> Path:
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "full_attention_interval": 4,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "shared_expert_intermediate_size": 512,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "default",
            },
            "use_cache": True,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_qwen35_moe_vl_config_preserves_vlm_architecture(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))

    assert isinstance(config, transformers.Qwen3_5MoeConfig)
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen3_5MoeForCausalLM"]
    assert config.text_config.num_experts == 128
    assert config.text_config.intermediate_size == 4608
    assert config.text_config.rope_theta == 1000000.0
    assert config.text_config.partial_rotary_factor == 0.25
    assert config.text_config.rope_scaling["type"] == "mrope"
    assert config.text_config.rope_scaling["mrope_section"] == [11, 11, 10]
    assert config.text_config.mamba_ssm_dtype == "float32"
    assert config.get_text_config() is config.text_config


def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(
    tmp_path: Path,
) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    validate_and_set_mamba_ssm_cache_dtype(model_config, "auto")
    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32

    mamba_params = extract_mamba_kv_cache_params(
        config.text_config,
        quant_config=model_config.quant_config,
    )
    assert mamba_params.dtype is torch.bfloat16
    assert mamba_params.mamba_ssm_cache_dtype is torch.float32


def test_qwen35_moe_vl_resolves_model_and_mapper(tmp_path: Path) -> None:
    config = load_pretrained_config(str(_write_qwen35_moe_vl_config(tmp_path)))
    model_config = ModelConfig(pretrained_config=config)

    assert AutoModelForCausalLM._resolve_class(model_config) is Qwen3_5MoeVLModel
    assert isinstance(
        AutoCheckpointMapper.get("HF", "Qwen3_5MoeForConditionalGeneration"),
        Qwen3_5MoeHfWeightMapper,
    )


def test_qwen35_moe_vl_placeholder_metadata_registered() -> None:
    metadata = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata("qwen3_5_moe")

    assert metadata.placeholder_map == {
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    }
    assert metadata.placeholders_separator == ""
    assert metadata.content_format is ContentFormat.STRING
