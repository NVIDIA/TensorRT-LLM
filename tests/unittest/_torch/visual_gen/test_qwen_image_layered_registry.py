# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry and transformer smoke tests for Qwen-Image-Layered."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

# Importing the models package side-effects the ``@register_pipeline``
# decorator on ``QwenImageLayeredPipeline`` being applied.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenEmbedLayer3DRope,
    QwenImageLayeredPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    QwenJointAttention,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import (
    PIPELINE_REGISTRY,
    AutoPipeline,
    PipelineComponent,
)
from tensorrt_llm.visual_gen.args import QuantAttentionConfig


def test_qwen_image_layered_pipeline_is_registered():
    """@register_pipeline("QwenImageLayeredPipeline") must have been applied."""
    assert "QwenImageLayeredPipeline" in PIPELINE_REGISTRY
    assert PIPELINE_REGISTRY["QwenImageLayeredPipeline"].pipeline_cls is QwenImageLayeredPipeline


def test_auto_pipeline_detects_qwen_image_layered_class_name(tmp_path):
    """Qwen-Image-Layered checkpoints must resolve to their layered pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImageLayeredPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImageLayeredPipeline"


def test_auto_pipeline_routes_qwen_image_layered_variants(tmp_path):
    """Unknown layered variants stay on the layered pipeline, not the base Qwen pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImageLayeredEditPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImageLayeredPipeline"


def test_qwen_image_layered_processor_loads_from_subfolder(monkeypatch, tmp_path):
    """Layered checkpoints store the VL processor in the processor/ subfolder."""
    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, checkpoint_dir, **kwargs):
            calls.append((checkpoint_dir, kwargs))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Qwen2VLProcessor=FakeProcessor),
    )
    monkeypatch.setattr(
        QwenImagePipeline,
        "load_standard_components",
        lambda self, checkpoint_dir, device, skip_components=None: None,
    )

    pipeline = QwenImageLayeredPipeline(DiffusionModelConfig())
    pipeline.load_standard_components(
        str(tmp_path),
        torch.device("cpu"),
        skip_components=[PipelineComponent.IMAGE_PROCESSOR],
    )

    assert calls == [
        (str(tmp_path), {"subfolder": PipelineComponent.PROCESSOR}),
    ]
    assert isinstance(pipeline.processor, FakeProcessor)


def test_qwen_image_layered_cache_dit_enabler_is_registered():
    """Cache-DiT configs should dispatch Qwen-Image-Layered to Qwen-specific wrapping."""
    pytest.importorskip("cache_dit")
    from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import (
        CUSTOM_CACHE_DIT_ENABLERS,
        enable_cache_dit_for_qwen_image,
    )

    assert CUSTOM_CACHE_DIT_ENABLERS["QwenImageLayeredPipeline"] is enable_cache_dit_for_qwen_image


def test_qwen_image_layered_post_load_enables_cache_dit(monkeypatch):
    """cache_backend=cache_dit should be activated after layered transformer load."""
    pipeline = object.__new__(QwenImageLayeredPipeline)
    transformer = SimpleNamespace()
    pipeline.transformer = transformer
    pipeline.model_config = SimpleNamespace(cache_backend="cache_dit")
    calls = {}

    def fake_setup(model, coefficients=None):
        calls["model"] = model
        calls["coefficients"] = coefficients

    monkeypatch.setattr(pipeline, "_setup_cache_acceleration", fake_setup)

    QwenImageLayeredPipeline.post_load_weights(pipeline)

    assert calls == {"model": transformer, "coefficients": None}


def test_qwen_image_layered_refresh_cache_acceleration():
    """Layered's custom denoise loop must refresh Cache-DiT before stepping."""
    pipeline = object.__new__(QwenImageLayeredPipeline)
    calls = []
    pipeline.cache_accelerator = SimpleNamespace(
        is_enabled=lambda: True,
        refresh=lambda steps: calls.append(steps),
    )

    pipeline._refresh_cache_acceleration(50)

    assert calls == [50]


def test_qwen_image_layered_default_params_match_runtime_inputs():
    """Layered derives size from image+resolution unless height/width are explicitly set."""
    pipeline = QwenImageLayeredPipeline(DiffusionModelConfig())
    assert pipeline.default_generation_params["height"] is None
    assert pipeline.default_generation_params["width"] is None
    assert pipeline.extra_param_specs["resolution"].range is None
    assert pipeline.warmup_cache_key(None, None, num_frames=1) == (640, 640)
    assert pipeline.warmup_cache_key(512, 768, num_frames=1) == (512, 768)


def test_qwen_image_layered_rejects_multi_frame_latent_input():
    """Layered latent input accepts one conditioning frame before packing."""
    pipeline = QwenImageLayeredPipeline(DiffusionModelConfig())
    image = torch.zeros(1, pipeline.latent_channels, 2, 4, 4)

    assert pipeline._is_layered_latent_image(image)
    with pytest.raises(ValueError, match="exactly one conditioning frame"):
        pipeline._validate_single_conditioning_frame(image)


def test_transformer_constructs_with_layered_config():
    """Layered config fields select layer-aware RoPE and additional time conditioning."""
    model = QwenImageTransformer2DModel(
        model_config=None,
        num_layers=2,
        use_additional_t_cond=True,
        use_layer3d_rope=True,
    )
    assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)
    assert model.time_text_embed.use_additional_t_cond is True
    assert hasattr(model.time_text_embed, "addition_t_embedding")


def test_transformer_from_config_dict_preserves_layered_fields():
    """from_config_dict should not drop Qwen-Image-Layered transformer fields."""
    model = QwenImageTransformer2DModel.from_config_dict(
        {
            "num_layers": 2,
            "use_additional_t_cond": True,
            "use_layer3d_rope": True,
        }
    )
    assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)
    assert model.time_text_embed.use_additional_t_cond is True


def test_qwen_layered_sage_attention_compacts_batched_mask(monkeypatch):
    """Masked layered attention should still reach SageAttention for padded text batches."""
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="fp8",
                v_dtype="fp8",
                q_block_size=1,
                k_block_size=1,
                v_block_size=1,
            ),
        ),
        attention_metadata_state=create_attention_metadata_state(),
    )
    attn = QwenJointAttention(
        dim=8,
        num_attention_heads=2,
        attention_head_dim=4,
        dtype=torch.bfloat16,
        config=model_config,
    )
    captured = {"q_shapes": [], "k_shapes": []}

    def fake_attn_impl(q, k, v, **kwargs):
        captured["q_shapes"].append(tuple(q.shape))
        captured["k_shapes"].append(tuple(k.shape))
        return torch.zeros_like(q)

    monkeypatch.setattr(attn, "_attn_impl", fake_attn_impl)

    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    attention_mask = torch.tensor(
        [
            [True, True, False, False, False, True, True, True],
            [True, True, True, False, False, True, True, True],
        ]
    )

    img_out, txt_out = attn(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
    )

    assert captured["q_shapes"] == [(1, 5, 8), (1, 6, 8)]
    assert captured["k_shapes"] == [(1, 5, 8), (1, 6, 8)]
    assert img_out.shape == hidden_states.shape
    assert txt_out.shape == encoder_hidden_states.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_text_mask", [False, True])
def test_layered_transformer_forward_sanity(with_text_mask):
    """A tiny Qwen-Image-Layered transformer runs layer-aware RoPE and t-cond paths."""
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
            use_additional_t_cond=True,
            use_layer3d_rope=True,
        )
        .to(device, dtype=dtype)
        .eval()
    )

    hidden_states = torch.randn(1, 12, 4, device=device, dtype=dtype)
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
            additional_t_cond=torch.zeros(1, device=device, dtype=torch.long),
            img_shapes=[[(1, 2, 2), (1, 2, 2), (1, 2, 2)]],
        )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 12, 4)
