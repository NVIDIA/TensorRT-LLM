# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native LTX-2.5 architecture selection and model deltas."""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

_VIDEO_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=32,
    in_channels=16,
    out_channels=16,
    num_layers=1,
    cross_attention_dim=128,
    caption_channels=64,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 32, 32],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model_config() -> DiffusionModelConfig:
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )


def _build_model():
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import LTX25Model

    model = (
        LTX25Model(
            model_type=LTXModelType.VideoOnly,
            model_config=_model_config(),
            **_VIDEO_CONFIG,
        )
        .to(_DEVICE)
        .float()
        .eval()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() > 0:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    return model


def test_ltx25_model_layout() -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import (
        LTX25TransformerBlock,
    )

    model = _build_model()
    assert tuple(model.keyframes_abs_pos_embedding.shape) == (1, 128)
    assert isinstance(model.transformer_blocks[0], LTX25TransformerBlock)
    assert not model.transformer_blocks[0]._fuse_adaln
    assert not model.transformer_blocks[0].attn1.fuse_qk_norm_rope
    assert model.transformer_blocks[0].ff.up_proj.bias is None
    assert model.transformer_blocks[0].ff.down_proj.bias is None


def test_keyframe_embedding_is_applied_only_to_marked_tokens() -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.modality import LTX25Modality

    model = _build_model()
    with torch.no_grad():
        model.keyframes_abs_pos_embedding.fill_(0.25)

    latent = torch.randn(1, 4, 16, device=_DEVICE)
    context = torch.randn(1, 2, 128, device=_DEVICE)
    common = {
        "latent": latent,
        "timesteps": torch.full((1, 4), 0.5, device=_DEVICE),
        "sigma": torch.tensor([0.5], device=_DEVICE),
        "positions": torch.zeros(1, 3, 4, 2, device=_DEVICE),
        "context": context,
    }
    unmarked = LTX25Modality(**common)
    keyframes_mask = torch.tensor([[[0], [1], [0], [1]]], device=_DEVICE)
    marked = LTX25Modality(**common, keyframes_mask=keyframes_mask)
    static_pe = (torch.empty(0, device=_DEVICE), torch.empty(0, device=_DEVICE))
    preprocessor = model.video_args_preprocessor

    without_marker = preprocessor.prepare(unmarked, context, None, static_pe).x
    with_marker = preprocessor.prepare(marked, context, None, static_pe).x

    expected = keyframes_mask * model.keyframes_abs_pos_embedding
    torch.testing.assert_close(with_marker - without_marker, expected.expand_as(with_marker))


def test_keyframe_marker_preserves_bf16_projection_dtype():
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.modality import LTX25Modality

    model = _build_model().bfloat16()
    # Materialization may retain a floating-point marker parameter. The official
    # implementation casts it at the addition boundary, never promoting latents.
    model.keyframes_abs_pos_embedding = torch.nn.Parameter(
        torch.full((1, 128), 0.25, device=_DEVICE, dtype=torch.float32)
    )
    latent = torch.randn(1, 4, 16, device=_DEVICE, dtype=torch.bfloat16)
    context = torch.randn(1, 2, 128, device=_DEVICE, dtype=torch.bfloat16)
    modality = LTX25Modality(
        latent=latent,
        timesteps=torch.full((1, 4), 0.5, device=_DEVICE),
        sigma=torch.tensor([0.5], device=_DEVICE),
        positions=torch.zeros(1, 3, 4, 2, device=_DEVICE),
        context=context,
        keyframes_mask=torch.tensor([[[0], [2], [0], [1]]], device=_DEVICE),
    )
    preprocessor = model.video_args_preprocessor
    result = preprocessor.prepare(
        modality, context, None, (torch.empty(0, device=_DEVICE), torch.empty(0, device=_DEVICE))
    )
    expected = preprocessor.patchify_proj(latent)
    expected = expected + (modality.keyframes_mask > 0).to(expected.dtype) * 0.25
    assert result.x.dtype == torch.bfloat16
    torch.testing.assert_close(result.x, expected, rtol=0, atol=0)


def test_reference_norm_supports_meta_initialization():
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import LTX25Model

    with MetaInitMode():
        model = LTX25Model(
            model_type=LTXModelType.VideoOnly, model_config=_model_config(), **_VIDEO_CONFIG
        )
    norm = model.transformer_blocks[0].attn1.norm_q
    assert isinstance(norm, torch.nn.RMSNorm)
    assert norm.weight.is_meta


def test_audio_mlp_preserves_bf16_projection_before_gelu():
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx2_5.transformer_ltx2_5 import LTX25Model

    model = LTX25Model(
        model_type=LTXModelType.AudioVideo,
        model_config=_model_config(),
        audio_num_attention_heads=4,
        audio_attention_head_dim=32,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_cross_attention_dim=128,
        **_VIDEO_CONFIG,
    ).to(device=_DEVICE, dtype=torch.bfloat16)
    mlp = model.transformer_blocks[0].audio_ff
    with torch.no_grad():
        for parameter in mlp.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        x = torch.randn(1, 9, 128, device=_DEVICE, dtype=torch.bfloat16)
        projected = torch.nn.functional.linear(x, mlp.up_proj.weight, mlp.up_proj.bias)
        activated = torch.nn.functional.gelu(projected, approximate="tanh")
        expected = torch.nn.functional.linear(activated, mlp.down_proj.weight, mlp.down_proj.bias)
        actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
