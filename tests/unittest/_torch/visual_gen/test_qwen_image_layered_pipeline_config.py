# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level configuration tests for Qwen-Image-Layered."""

import json

import pytest

# Importing the models package applies the Qwen-Image-Layered registration side effect.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenEmbedLayer3DRope,
    QwenImageLayeredPipeline,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs


def _tiny_qwen_layered_transformer_config(**overrides):
    config = {
        "_class_name": "QwenImageTransformer2DModel",
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 12,
        "axes_dims_rope": [2, 2, 4],
    }
    config.update(overrides)
    return config


def _write_minimal_qwen_layered_checkpoint(tmp_path, transformer_config=None):
    """Create the minimum diffusers layout needed by PipelineLoader config code."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageLayeredPipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            }
        )
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    transformer_config = transformer_config or {"_class_name": "QwenImageTransformer2DModel"}
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config))
    return tmp_path


def test_qwen_layered_pipeline_config_defaults_to_empty_dict(tmp_path):
    checkpoint_dir = _write_minimal_qwen_layered_checkpoint(tmp_path)

    args = VisualGenArgs(model=str(checkpoint_dir))
    resolved = PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))

    assert resolved == {}


def test_qwen_layered_pipeline_config_rejects_unknown_keys(tmp_path):
    checkpoint_dir = _write_minimal_qwen_layered_checkpoint(tmp_path)

    args = VisualGenArgs(
        model=str(checkpoint_dir),
        pipeline_config={"text_encoder_path": "/tmp/not-a-qwen-layered-knob"},
    )
    with pytest.raises(
        ValueError, match="Unknown pipeline_config keys for QwenImageLayeredPipeline"
    ):
        PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))


def test_qwen_layered_config_initializes_layered_transformer(tmp_path):
    checkpoint_dir = _write_minimal_qwen_layered_checkpoint(
        tmp_path,
        transformer_config=_tiny_qwen_layered_transformer_config(
            use_additional_t_cond=True,
            use_layer3d_rope=True,
        ),
    )
    args = VisualGenArgs(model=str(checkpoint_dir))

    config = DiffusionModelConfig.from_pretrained(str(checkpoint_dir), args=args)
    pipeline = QwenImageLayeredPipeline(config)

    assert pipeline.guidance_embeds is False
    assert pipeline.zero_cond_t is False
    assert pipeline.transformer.time_text_embed.use_additional_t_cond is True
    assert isinstance(pipeline.transformer.pos_embed, QwenEmbedLayer3DRope)
