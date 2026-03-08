"""Unit tests for LTX2 transformer models.

Tests cover:
- Model structure and instantiation (VideoOnly and AudioVideo)
- Forward pass sanity checks with correct output shapes

No checkpoint or LTX-2 reference code required — all tests use random weights.
"""

import unittest

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

# Reduced VideoOnly config for fast CI testing.
# Real model: 32 heads × 128 dim_head = 4096 inner, 48 layers.
VIDEO_ONLY_CONFIG = dict(
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

# Reduced AudioVideo config (adds audio stream to video config).
# Real model: audio 32 heads × 64 dim_head = 2048 inner.
AUDIO_VIDEO_CONFIG = dict(
    **VIDEO_ONLY_CONFIG,
    audio_num_attention_heads=4,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=64,
    audio_positional_embedding_max_pos=[4],
    av_ca_timestep_scale_multiplier=1,
)


def _create_model_config(backend: str = "VANILLA") -> DiffusionModelConfig:
    """Create a minimal DiffusionModelConfig for unit tests."""
    from types import SimpleNamespace

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _init_all_weights(model: torch.nn.Module, std: float = 0.02):
    """Initialize all parameters with small random values.

    TRT-LLM Linear uses torch.empty() (uninitialized memory); explicit
    initialization prevents NaN from recycled GPU memory.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=std)


def _make_video_positions(batch: int, n_patches: int, n_frames: int,
                          grid_h: int, grid_w: int,
                          device: torch.device) -> torch.Tensor:
    """Construct video position tensor (B, 3, T, 2) for RoPE.

    Each patch has 3D start/end coordinates: (frame, height, width).
    """
    positions = torch.zeros(batch, 3, n_patches, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                positions[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                positions[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                positions[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1
    return positions


def _make_audio_positions(batch: int, n_patches: int,
                          device: torch.device) -> torch.Tensor:
    """Construct audio position tensor (B, 1, T, 2) for RoPE.

    Audio positions are 1D (time axis only).
    """
    positions = torch.zeros(batch, 1, n_patches, 2, device=device)
    for i in range(n_patches):
        positions[:, 0, i, :] = torch.tensor([i, i + 1], dtype=torch.float32)
    return positions


class TestLTX2VideoOnlyModel(unittest.TestCase):
    """Unit tests for LTXModel with VideoOnly type."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_video_only_model_structure(self):
        """Test VideoOnly model can be instantiated with correct structure."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            LTXModel,
            LTXModelType,
        )

        model_config = _create_model_config()
        model = LTXModel(
            model_type=LTXModelType.VideoOnly,
            model_config=model_config,
            **VIDEO_ONLY_CONFIG,
        ).to(self.DEVICE)

        # Check video components
        self.assertTrue(hasattr(model, "transformer_blocks"))
        self.assertEqual(len(model.transformer_blocks), 1)
        self.assertTrue(hasattr(model, "patchify_proj"))
        self.assertTrue(hasattr(model, "adaln_single"))
        self.assertTrue(hasattr(model, "caption_projection"))
        self.assertTrue(hasattr(model, "scale_shift_table"))
        self.assertTrue(hasattr(model, "norm_out"))
        self.assertTrue(hasattr(model, "proj_out"))

        # Check audio components are absent
        self.assertFalse(hasattr(model, "audio_patchify_proj"))
        self.assertFalse(hasattr(model, "audio_adaln_single"))

        # Check block structure
        block = model.transformer_blocks[0]
        self.assertTrue(hasattr(block, "attn1"))
        self.assertTrue(hasattr(block, "attn2"))
        self.assertTrue(hasattr(block, "ff"))
        self.assertFalse(hasattr(block, "audio_attn1"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_video_only_forward_sanity(self):
        """Test VideoOnly forward pass produces valid output shapes."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            LTXModel,
            LTXModelType,
        )

        torch.manual_seed(42)
        dtype = torch.bfloat16
        model_config = _create_model_config()

        model = LTXModel(
            model_type=LTXModelType.VideoOnly,
            model_config=model_config,
            **VIDEO_ONLY_CONFIG,
        ).to(self.DEVICE, dtype=dtype).eval()
        _init_all_weights(model)

        batch = 1
        n_frames, grid_h, grid_w = 1, 4, 4
        n_patches = n_frames * grid_h * grid_w
        in_channels = VIDEO_ONLY_CONFIG["in_channels"]
        caption_channels = VIDEO_ONLY_CONFIG["caption_channels"]
        text_len = 8

        video_modality = Modality(
            latent=torch.randn(batch, n_patches, in_channels, device=self.DEVICE, dtype=dtype) * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=_make_video_positions(batch, n_patches, n_frames, grid_h, grid_w, self.DEVICE),
            context=torch.randn(batch, text_len, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02,
        )

        with torch.no_grad():
            video_out, audio_out = model(video=video_modality, audio=None)

        self.assertIsNotNone(video_out)
        self.assertIsNone(audio_out)
        self.assertEqual(
            video_out.shape,
            (batch, n_patches, VIDEO_ONLY_CONFIG["out_channels"]),
        )


class TestLTX2AudioVideoModel(unittest.TestCase):
    """Unit tests for LTXModel with AudioVideo type."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_audio_video_model_structure(self):
        """Test AudioVideo model has both video and audio components."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            LTXModel,
            LTXModelType,
        )

        model_config = _create_model_config()
        model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            model_config=model_config,
            **AUDIO_VIDEO_CONFIG,
        ).to(self.DEVICE)

        # Check video components
        self.assertTrue(hasattr(model, "patchify_proj"))
        self.assertTrue(hasattr(model, "adaln_single"))
        self.assertTrue(hasattr(model, "scale_shift_table"))

        # Check audio components
        self.assertTrue(hasattr(model, "audio_patchify_proj"))
        self.assertTrue(hasattr(model, "audio_adaln_single"))
        self.assertTrue(hasattr(model, "audio_scale_shift_table"))

        # Check AV cross-attention components
        self.assertTrue(hasattr(model, "av_ca_video_scale_shift_adaln_single"))
        self.assertTrue(hasattr(model, "av_ca_audio_scale_shift_adaln_single"))

        # Check block structure has both modalities + AV cross-attn
        block = model.transformer_blocks[0]
        self.assertTrue(hasattr(block, "attn1"))
        self.assertTrue(hasattr(block, "audio_attn1"))
        self.assertTrue(hasattr(block, "audio_to_video_attn"))
        self.assertTrue(hasattr(block, "video_to_audio_attn"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_audio_video_forward_sanity(self):
        """Test AudioVideo forward pass produces valid output shapes for both modalities."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            LTXModel,
            LTXModelType,
        )

        torch.manual_seed(42)
        dtype = torch.bfloat16
        model_config = _create_model_config()

        model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            model_config=model_config,
            **AUDIO_VIDEO_CONFIG,
        ).to(self.DEVICE, dtype=dtype).eval()
        _init_all_weights(model)

        batch = 1
        v_frames, v_h, v_w = 1, 4, 4
        v_patches = v_frames * v_h * v_w
        a_patches = 8
        in_channels = AUDIO_VIDEO_CONFIG["in_channels"]
        caption_channels = AUDIO_VIDEO_CONFIG["caption_channels"]
        text_len = 8

        video_modality = Modality(
            latent=torch.randn(batch, v_patches, in_channels, device=self.DEVICE, dtype=dtype) * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=_make_video_positions(batch, v_patches, v_frames, v_h, v_w, self.DEVICE),
            context=torch.randn(batch, text_len, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02,
        )

        audio_in_channels = AUDIO_VIDEO_CONFIG["audio_in_channels"]
        audio_modality = Modality(
            latent=torch.randn(batch, a_patches, audio_in_channels, device=self.DEVICE, dtype=dtype) * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=_make_audio_positions(batch, a_patches, self.DEVICE),
            context=torch.randn(batch, text_len, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02,
        )

        with torch.no_grad():
            video_out, audio_out = model(video=video_modality, audio=audio_modality)

        self.assertIsNotNone(video_out)
        self.assertIsNotNone(audio_out)
        self.assertEqual(
            video_out.shape,
            (batch, v_patches, AUDIO_VIDEO_CONFIG["out_channels"]),
        )
        self.assertEqual(
            audio_out.shape,
            (batch, a_patches, AUDIO_VIDEO_CONFIG["audio_out_channels"]),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_video_only_input_to_audio_video_model(self):
        """Test AudioVideo model works when only video modality is provided."""
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            LTXModel,
            LTXModelType,
        )

        torch.manual_seed(42)
        dtype = torch.bfloat16
        model_config = _create_model_config()

        model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            model_config=model_config,
            **AUDIO_VIDEO_CONFIG,
        ).to(self.DEVICE, dtype=dtype).eval()
        _init_all_weights(model)

        batch = 1
        v_frames, v_h, v_w = 1, 4, 4
        v_patches = v_frames * v_h * v_w
        in_channels = AUDIO_VIDEO_CONFIG["in_channels"]
        caption_channels = AUDIO_VIDEO_CONFIG["caption_channels"]
        text_len = 8

        video_modality = Modality(
            latent=torch.randn(batch, v_patches, in_channels, device=self.DEVICE, dtype=dtype) * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=_make_video_positions(batch, v_patches, v_frames, v_h, v_w, self.DEVICE),
            context=torch.randn(batch, text_len, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02,
        )

        with torch.no_grad():
            video_out, audio_out = model(video=video_modality, audio=None)

        self.assertIsNotNone(video_out)
        self.assertIsNone(audio_out)
        self.assertEqual(
            video_out.shape,
            (batch, v_patches, AUDIO_VIDEO_CONFIG["out_channels"]),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
