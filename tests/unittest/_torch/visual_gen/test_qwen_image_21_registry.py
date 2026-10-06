# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Registry/config smoke tests for Qwen-Image 2.1 VisualGen."""

import json
from pathlib import Path

# Importing models applies the @register_pipeline side effect.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image_21 import QwenImage21Pipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs


def _write_minimal_qwen_image_21_checkpoint(tmp_path: Path) -> Path:
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "processor": ["transformers", "Qwen3VLProcessor"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage21"],
            }
        )
    )
    for name, payload in {
        "transformer": {
            "_class_name": "QwenImage21Transformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "context_in_dim": 4096,
            "in_channels": 64,
            "num_attention_heads": 32,
            "num_layers": 32,
            "out_channels": 64,
            "patch_size": 1,
            "mlp_ratio": 3,
            "eps": 1e-6,
            "causal_condition": True,
        },
        "vae": {"_class_name": "AutoencoderKLQwenImage21", "z_dim": 64, "scale_factor_spatial": 16},
        "scheduler": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
    }.items():
        subdir = tmp_path / name
        subdir.mkdir()
        (subdir / "config.json").write_text(json.dumps(payload))
    return tmp_path


def test_qwen_image_21_pipeline_is_registered():
    assert "QwenImage21Pipeline" in PIPELINE_REGISTRY
    entry = PIPELINE_REGISTRY["QwenImage21Pipeline"]
    assert entry.pipeline_cls is QwenImage21Pipeline
    assert entry.hf_ids == ["Qwen/Qwen-Image-2.1"]
    assert "transformer/*" in entry.download_patterns
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["num_inference_steps"] == 40
    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["guidance_scale"] == 1.0


def test_auto_pipeline_detects_qwen_image_21_class_name(tmp_path):
    checkpoint_dir = _write_minimal_qwen_image_21_checkpoint(tmp_path)
    assert AutoPipeline._detect_from_checkpoint(str(checkpoint_dir)) == "QwenImage21Pipeline"


def test_qwen_image_21_pipeline_config_loads_component_metadata(tmp_path):
    checkpoint_dir = _write_minimal_qwen_image_21_checkpoint(tmp_path)
    args = VisualGenArgs(model=str(checkpoint_dir))

    config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_dir), args=args)

    transformer_config = config.model_configs["transformer"].pretrained_config
    assert transformer_config._class_name == "QwenImage21Transformer2DModel"
    assert transformer_config.patch_size == 1
    assert transformer_config.in_channels == 64
    assert transformer_config.out_channels == 64
    assert transformer_config.num_layers == 32
    assert config.model_configs["vae"].pretrained_config.z_dim == 64
