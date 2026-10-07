# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    _resolve_qwen_image_21_vae_fallback,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Transformer2DModel,
)


def test_qwen_image_21_pipeline_runtime_contract_is_native_t2i_default():
    """Static contract for the M7 text-to-image path.

    Qwen-Image 2.1 currently validates quality on the source-default
    guidance-free text-to-image path. The pipeline contract should not silently
    imply an upstream Diffusers pipeline/transformer path for runtime execution.
    """

    assert QwenImage21Pipeline.DEFAULT_GENERATION_PARAMS["guidance_scale"] == 1.0
    assert QwenImage21Pipeline.default_num_inference_steps == 40
    assert QwenImage21Pipeline.transformer_class is QwenImage21Transformer2DModel
    assert QwenImage21Pipeline.scheduler_class.__name__ == "QwenImage21FlowMatchEulerScheduler"

    pipeline = object.__new__(QwenImage21Pipeline)
    pipeline.vae_scale_factor = 16
    pipeline.latent_channels = 64
    runtime_config = QwenImage21Pipeline._qwen_image_21_runtime_config(pipeline)

    assert runtime_config["pipeline_class_name"] == "QwenImage21Pipeline"
    assert runtime_config["scheduler"] == "QwenImage21FlowMatchEulerScheduler"
    assert runtime_config["image_vae"] == "diffusers.AutoencoderKLQwenImage21 fallback"
    assert "DiffusionPipeline" not in repr(runtime_config)


def test_qwen_image_21_vae_fallback_requires_model_index_symbol(monkeypatch):
    """The declared Qwen-Image-2.1 VAE symbol is the only accepted fallback.

    Qwen-Image 2.1 checkpoints declare ``AutoencoderKLQwenImage21`` and use a
    64-channel 2.1 VAE.  The older ``AutoencoderKLQwenImage`` class is a
    different architecture and must not be silently accepted for runtime decode.
    """

    fake_diffusers = types.ModuleType("diffusers")

    class AutoencoderKLQwenImage21:
        pass

    class AutoencoderKLQwenImage:
        pass

    fake_diffusers.AutoencoderKLQwenImage21 = AutoencoderKLQwenImage21
    fake_diffusers.AutoencoderKLQwenImage = AutoencoderKLQwenImage
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    vae_cls, vae_name = _resolve_qwen_image_21_vae_fallback()

    assert vae_cls is AutoencoderKLQwenImage21
    assert vae_name == "diffusers.AutoencoderKLQwenImage21"


def test_qwen_image_21_vae_fallback_rejects_qwen_image_1x_alias(monkeypatch):
    """Diffusers 0.40.x ``AutoencoderKLQwenImage`` is not compatible with 2.1 weights."""

    fake_diffusers = types.ModuleType("diffusers")

    class AutoencoderKLQwenImage:
        pass

    fake_diffusers.AutoencoderKLQwenImage = AutoencoderKLQwenImage
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    with pytest.raises(ImportError, match="AutoencoderKLQwenImage21"):
        _resolve_qwen_image_21_vae_fallback()


def test_qwen_image_21_transformer_checkpoint_norm_keys_keep_distinct_semantics():
    """Checkpoint-facing keys distinguish regular attention RMSNorm and zero-centered text RMSNorm."""

    transformer = QwenImage21Transformer2DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        context_in_dim=16,
        in_channels=4,
        out_channels=4,
        axes_dims_rope=(2, 6, 8),
    )
    state = transformer.state_dict()

    assert "transformer_blocks.0.attn.norm_q.weight" in state
    assert "transformer_blocks.0.attn.norm_k.weight" in state
    assert "txt_in.text_norm.weight" in state
    torch.testing.assert_close(state["transformer_blocks.0.attn.norm_q.weight"], torch.ones(8))
    torch.testing.assert_close(state["transformer_blocks.0.attn.norm_k.weight"], torch.ones(8))
    torch.testing.assert_close(state["txt_in.text_norm.weight"], torch.zeros(16))
