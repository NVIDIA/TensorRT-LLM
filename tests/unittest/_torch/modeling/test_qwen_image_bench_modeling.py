# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from types import SimpleNamespace

import pytest
import torch
import transformers

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5VLModel
from tensorrt_llm._torch.models.modeling_qwen_image_bench import QwenImageBenchModel
from tensorrt_llm._torch.models.modeling_utils import (
    MODEL_CLASS_MAPPER_MAPPING,
    MODEL_CLASS_MAPPING,
    MODEL_CLASS_VISION_ENCODER_MAPPING,
)
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    get_qwen3_hybrid_layer_types,
    is_qwen_image_bench_config,
    load_pretrained_config,
)


def _write_qwen_image_bench_config(model_dir):
    text_config = _qwen3_5_text_config()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "dtype": "bfloat16",
        "eos_token_id": 248046,
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen3_5",
        "pad_token_id": 248044,
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
        "text_config": text_config,
        "vision_config": _qwen3_5_vision_config(),
    }
    (model_dir / "config.json").write_text(json.dumps(config))


def _qwen3_8_dense_vlm_config():
    """A representative Qwen3.8 dense VLM config, scaled down.

    Mirrors the top-level shape published by Qwen/Qwen3.8-27B (and by
    Qwen/Qwen3.6-27B, and their quantized re-exports): the generic
    `Qwen3_5ForConditionalGeneration` architecture, a composite
    text_config/vision_config, all four vision token ids, and the generic
    transformers `language_model_only: false` field. Its vision sub-config uses
    the newer `qwen3_5_vision` model type; Qwen-Image-Bench retains the legacy
    `qwen3_5` spelling.
    """
    vision_config = _qwen3_5_vision_config()
    vision_config["model_type"] = "qwen3_5_vision"
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen3_5",
        "text_config": _qwen3_5_text_config(),
        "tie_word_embeddings": False,
        "video_token_id": 248057,
        "vision_config": vision_config,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


def _qwen3_5_vision_config():
    return {
        "deepstack_visual_indexes": [],
        "depth": 1,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "intermediate_size": 128,
        "model_type": "qwen3_5",
        "num_heads": 4,
        "num_position_embeddings": 16,
        "out_hidden_size": 256,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }


def _qwen3_5_text_config():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "full_attention_interval": 4,
        "head_dim": 64,
        "hidden_size": 256,
        "intermediate_size": 512,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 32,
        "max_position_embeddings": 4096,
        "mamba_ssm_dtype": "float32",
        "model_type": "qwen3_5_text",
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "output_gate_type": "swish",
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": False,
        "vocab_size": 1024,
    }


def test_qwen_image_bench_config_uses_internal_arch_and_normalizes_text(tmp_path):
    model_dir = tmp_path / "Qwen-Image-Bench"
    model_dir.mkdir()
    _write_qwen_image_bench_config(model_dir)

    config = load_pretrained_config(str(model_dir))

    assert config.architectures == ["QwenImageBenchForConditionalGeneration"]
    assert config.model_type == "qwen3_5"
    assert config.vision_config.depth == 1
    assert isinstance(config.text_config, transformers.Qwen3NextConfig)
    assert config.text_config.architectures == ["Qwen3_5ForCausalLM"]
    assert config.text_config.rope_scaling["type"] == "mrope"
    assert config.text_config.rope_scaling["mrope_section"] == [1, 1, 2]
    assert get_qwen3_hybrid_layer_types(config.text_config) == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    # The declared mamba_ssm_dtype=float32 (SSM compute intent) is deliberately
    # not used for cache allocation; the cache stays in the weights dtype.
    assert extract_mamba_kv_cache_params(config.text_config).mamba_ssm_cache_dtype is torch.bfloat16


def test_qwen_image_bench_config_normalizes_top_level_quantization(tmp_path):
    model_dir = tmp_path / "Qwen-Image-Bench-FP8"
    model_dir.mkdir()
    _write_qwen_image_bench_config(model_dir)

    config_path = model_dir / "config.json"
    raw_config = json.loads(config_path.read_text())
    raw_config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "modules_to_not_convert": [
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.visual.blocks.0.attn.qkv",
            "mtp.layers.0.self_attn.q_proj",
        ],
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    config_path.write_text(json.dumps(raw_config))

    config = load_pretrained_config(str(model_dir))

    excluded = config.quantization_config["modules_to_not_convert"]
    assert "model.layers.0.linear_attn.in_proj_ba" in excluded
    assert "model.layers.0.linear_attn.in_proj_qkvz" in excluded
    assert "model.layers.1.linear_attn.in_proj_qkvz" in excluded
    assert "model.layers.2.linear_attn.in_proj_qkvz" in excluded
    assert "model.layers.3.linear_attn.in_proj_qkvz" not in excluded
    assert not any(name.startswith("model.language_model.") for name in excluded)
    assert not any(name.startswith("model.visual.") for name in excluded)
    assert not any(name.startswith("mtp.") for name in excluded)


def test_qwen3_5_conditional_text_config_does_not_use_image_bench_arch(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen3_5_text_config()))

    config = load_pretrained_config(str(tmp_path))

    assert isinstance(config, transformers.Qwen3NextConfig)
    assert config.architectures == ["Qwen3_5ForCausalLM"]


def test_qwen3_5_generic_composite_config_does_not_use_image_bench_arch(tmp_path):
    model_dir = tmp_path / "Qwen3.5-4B"
    model_dir.mkdir()
    _write_qwen_image_bench_config(model_dir)

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    del config["language_model_only"]
    config_path.write_text(json.dumps(config))

    config = load_pretrained_config(str(model_dir))

    assert isinstance(config, transformers.Qwen3_5Config)
    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen3_5ForCausalLM"]


def test_qwen_image_bench_config_detection_does_not_depend_on_model_name(tmp_path):
    model_dir = tmp_path / "local-checkpoint"
    model_dir.mkdir()
    _write_qwen_image_bench_config(model_dir)

    config = load_pretrained_config(str(model_dir))

    assert config.architectures == ["QwenImageBenchForConditionalGeneration"]
    assert isinstance(config.text_config, transformers.Qwen3NextConfig)


def test_qwen3_8_dense_vlm_config_keeps_generic_arch(tmp_path):
    """A Qwen3.8 dense VLM must not be rewritten to the Qwen-Image-Bench arch.

    Qwen3.6-27B, Qwen3.8-27B and their quantized re-exports all publish
    `language_model_only: false` alongside the generic composite VLM metadata,
    so keying detection on it swallowed the whole dense family.
    """
    model_dir = tmp_path / "Qwen3.8-27B-NVFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_qwen3_8_dense_vlm_config()))

    config = load_pretrained_config(str(model_dir))

    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen3_5ForCausalLM"]


def test_qwen3_8_dense_vlm_config_is_not_image_bench():
    assert not is_qwen_image_bench_config(_qwen3_8_dense_vlm_config())


def test_qwen3_5_composite_config_requires_text_config(tmp_path):
    model_dir = tmp_path / "malformed-composite"
    model_dir.mkdir()
    _write_qwen_image_bench_config(model_dir)

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["QwenImageBenchForConditionalGeneration"]
    del config["text_config"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="missing a usable text_config"):
        load_pretrained_config(str(model_dir))


def test_qwen_image_bench_model_and_mapper_registration():
    assert MODEL_CLASS_MAPPING["QwenImageBenchForConditionalGeneration"] is QwenImageBenchModel
    assert MODEL_CLASS_MAPPING["Qwen3_5ForConditionalGeneration"] is Qwen3_5VLModel
    assert "QwenImageBenchForConditionalGeneration" in MODEL_CLASS_VISION_ENCODER_MAPPING
    assert (
        MODEL_CLASS_MAPPER_MAPPING["QwenImageBenchForConditionalGeneration_HF"]
        is Qwen3_5MoeHfWeightMapper
    )


def test_qwen_image_bench_forwards_speculative_interface():
    draft_config = object()
    draft_model = object()
    text_model = object()
    lm_head = object()
    sentinel = object()

    llm = SimpleNamespace(
        draft_config=draft_config,
        draft_model=draft_model,
        model=text_model,
        lm_head=lm_head,
        load_draft_weights=lambda *args, **kwargs: sentinel,
    )
    model = QwenImageBenchModel.__new__(QwenImageBenchModel)
    object.__setattr__(model, "llm", llm)

    assert model.draft_config is draft_config
    assert model.draft_model is draft_model
    assert model.model is text_model
    assert model.lm_head is lm_head
    assert model.load_draft_weights("weights") is sentinel
