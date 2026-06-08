# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry smoke tests for Qwen-Image.

These tests make sure Qwen-Image checkpoints route through VisualGen's
``AutoPipeline`` instead of failing with ``Unknown pipeline: ''`` from
registry detection.
"""

import json
from types import SimpleNamespace

import pytest
import torch

# Importing the models package side-effects the ``@register_pipeline``
# decorator on ``QwenImagePipeline`` being applied, which is what we are
# testing here.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import QuantAttentionConfig


def test_qwen_image_pipeline_is_registered():
    """@register_pipeline("QwenImagePipeline") must have been applied."""
    assert "QwenImagePipeline" in PIPELINE_REGISTRY
    assert PIPELINE_REGISTRY["QwenImagePipeline"].pipeline_cls is QwenImagePipeline


def test_auto_pipeline_detects_qwen_image_class_name(tmp_path):
    """model_index.json with _class_name=QwenImagePipeline resolves."""
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImagePipeline"


def test_qwen_image_cache_dit_enabler_is_registered():
    """Cache-DiT configs should dispatch to Qwen-specific block wrapping."""
    pytest.importorskip("cache_dit")
    from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import (
        CUSTOM_CACHE_DIT_ENABLERS,
        enable_cache_dit_for_qwen_image,
    )

    assert CUSTOM_CACHE_DIT_ENABLERS["QwenImagePipeline"] is enable_cache_dit_for_qwen_image


def test_qwen_image_post_load_enables_cache_dit(monkeypatch):
    """cache_backend=cache_dit should be activated after transformer load."""
    pipeline = object.__new__(QwenImagePipeline)
    transformer = SimpleNamespace()
    pipeline.transformer = transformer
    pipeline.model_config = SimpleNamespace(cache_backend="cache_dit")
    calls = {}

    def fake_setup(model, coefficients=None):
        calls["model"] = model
        calls["coefficients"] = coefficients

    monkeypatch.setattr(pipeline, "_setup_cache_acceleration", fake_setup)

    QwenImagePipeline.post_load_weights(pipeline)

    assert calls == {"model": transformer, "coefficients": None}


def test_qwen_image_refresh_cache_acceleration():
    """Qwen's custom denoise loop must refresh Cache-DiT before stepping."""
    pipeline = object.__new__(QwenImagePipeline)
    calls = []
    pipeline.cache_accelerator = SimpleNamespace(
        is_enabled=lambda: True,
        refresh=lambda steps: calls.append(steps),
    )

    pipeline._refresh_cache_acceleration(50)

    assert calls == [50]


@pytest.mark.parametrize(
    "variant_class_name",
    [
        "QwenImageImg2ImgPipeline",
        "QwenImageEditPipeline",
        "QwenImageControlNetPipeline",
    ],
)
def test_auto_pipeline_routes_qwen_image_variants(tmp_path, variant_class_name):
    """Unknown Qwen-Image variants route to the base QwenImagePipeline.

    Matches the behaviour of the Wan and Flux branches in
    ``pipeline_registry.py``: anything containing ``Qwen`` + ``Image`` in
    its class name falls through to the base pipeline.
    """
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": variant_class_name}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImagePipeline"


def test_transformer_constructs_with_defaults():
    """The full transformer instantiates with the documented defaults.

    Doesn't touch real weights -- just verifies that the class graph is
    buildable and exposes the expected public attributes.
    """
    model = QwenImageTransformer2DModel(
        model_config=None,
        num_layers=2,  # tiny to keep CPU instantiation fast
    )
    assert len(model.transformer_blocks) == 2
    assert model.inner_dim == 24 * 128
    assert hasattr(model, "img_in")
    assert hasattr(model, "txt_in")
    assert hasattr(model, "txt_norm")
    assert hasattr(model, "norm_out")
    assert hasattr(model, "proj_out")
    assert hasattr(model, "pos_embed")


def test_qwen_separate_qkv_trtllm_without_sage_falls_back_to_vanilla():
    """Qwen separate-QKV attention keeps legacy TRTLLM fallback by default."""
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(backend="TRTLLM"),
        attention_metadata_state=create_attention_metadata_state(),
        skip_create_weights_in_init=True,
    )
    model = QwenImageTransformer2DModel(model_config=model_config, num_layers=1)

    assert model.transformer_blocks[0].attn.attn_backend == "VANILLA"


@pytest.mark.parametrize("qk_int8", [True, False])
def test_qwen_separate_qkv_trtllm_uses_sage_attention(qk_int8):
    """Qwen joint attention can reach TRTLLM SageAttention with separate Q/K/V."""
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="int8" if qk_int8 else "fp8",
                v_dtype="fp8",
                q_block_size=1,
                k_block_size=16 if qk_int8 else 1,
                v_block_size=1,
            ),
        ),
        attention_metadata_state=create_attention_metadata_state(),
        skip_create_weights_in_init=True,
    )
    model = QwenImageTransformer2DModel(model_config=model_config, num_layers=1)
    attn = model.transformer_blocks[0].attn

    assert attn.attn_backend == "TRTLLM"
    assert attn.attn.quant_attention_config is model_config.attention.quant_attention_config


def test_transformer_applies_quant_exclude_modules_before_weight_creation():
    """Qwen-Image should honor QuantConfig.exclude_modules during load-time materialization."""
    model_config = DiffusionModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8,
            exclude_modules=[
                "txt_in",
                "transformer_blocks.0.attn.to_q",
            ],
        ),
        skip_create_weights_in_init=True,
    )
    model = QwenImageTransformer2DModel(
        model_config=model_config,
        num_layers=1,
    )

    model.apply_quant_config_exclude_modules()

    assert model.txt_in.quant_config.quant_algo is None
    assert model.transformer_blocks[0].attn.to_q.quant_config.quant_algo is None
    assert model.transformer_blocks[0].attn.to_k.quant_config.quant_algo == QuantAlgo.FP8


def test_transformer_to_inference_dtype_preserves_quantized_linear_params():
    """Casting Qwen-Image to bf16 must not dequantize FP8 Linear weights/scales."""
    model_config = DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
    )
    model = QwenImageTransformer2DModel(
        model_config=model_config,
        num_layers=1,
    )

    assert model.txt_in.weight.dtype == torch.float8_e4m3fn
    assert model.time_text_embed.timestep_embedder.linear_1.weight.dtype == torch.float32

    model.to_inference_dtype()

    assert model.txt_in.weight.dtype == torch.float8_e4m3fn
    assert model.txt_in.weight_scale.dtype == torch.float32
    assert model.img_in.weight.dtype == torch.bfloat16
    assert model.norm_out.linear.weight.dtype == torch.float8_e4m3fn
    assert model.norm_out.linear.weight_scale.dtype == torch.float32
    assert model.time_text_embed.timestep_embedder.linear_1.weight.dtype == torch.bfloat16


def test_transformer_load_weights_detects_mismatch():
    """load_weights should surface a clear RuntimeError on a bad dict."""
    model = QwenImageTransformer2DModel(model_config=None, num_layers=2)
    with pytest.raises(RuntimeError, match=r"Missing keys|Unexpected keys"):
        model.load_weights({})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_text_mask", [False, True])
def test_transformer_forward_sanity(with_text_mask):
    """A tiny Qwen-Image transformer runs both unmasked and text-masked paths."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )
    model = (
        QwenImageTransformer2DModel(
            model_config=model_config,
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=12,
            axes_dims_rope=(2, 2, 4),
        )
        .to(device, dtype=dtype)
        .eval()
    )

    hidden_states = torch.randn(1, 4, 4, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 5, 12, device=device, dtype=dtype)
    encoder_hidden_states_mask = None
    if with_text_mask:
        encoder_hidden_states_mask = torch.tensor([[True, True, True, False, False]], device=device)

    with torch.inference_mode():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=torch.tensor([0.5], device=device, dtype=dtype),
            img_shapes=[[[1, 2, 2]]],
        )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 4, 4)
