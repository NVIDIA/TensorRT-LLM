# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unit tests for modeling_parakeet.py."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models.modeling_parakeet import ParakeetExtractor, ProjectedParakeet


def _make_extractor_config(**overrides):
    """Return a mock PretrainedConfig with small, realistic extractor values."""
    defaults = dict(
        num_mel_bins=80,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_extractor(**overrides):
    return ParakeetExtractor(_make_extractor_config(**overrides))


def _make_sound_config(**overrides):
    """Return a minimal sound_config suitable for ProjectedParakeet."""
    defaults = dict(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        conv_kernel_size=9,
        convolution_bias=False,
        subsampling_factor=8,
        subsampling_conv_channels=64,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=2,
        num_mel_bins=80,
        projection_hidden_size=128,
        projection_bias=False,
        scale_input=False,
        attention_bias=False,
        max_position_embeddings=5000,
        sampling_rate=16000,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestParakeetExtractor:
    def test_clip_sizes_short_audio(self):
        ext = _make_extractor()
        short_len = 100  # well below clip_target_samples
        sizes = ext._clip_sizes(short_len)
        # Should be at least the tail minimum
        assert len(sizes) >= 1
        assert sizes[0] >= ext._tail_min_samples

    def test_clip_sizes_exact_clip(self):
        ext = _make_extractor()
        exact_len = ext._clip_target_samples
        sizes = ext._clip_sizes(exact_len)
        assert sizes == [ext._clip_target_samples]

    def test_clip_sizes_with_remainder(self):
        ext = _make_extractor()
        audio_len = ext._clip_target_samples + ext._tail_min_samples + 500
        sizes = ext._clip_sizes(audio_len)
        assert len(sizes) >= 2
        # First clip is full.
        assert sizes[0] == ext._clip_target_samples
        # Tail is at least the minimum.
        assert sizes[-1] >= ext._tail_min_samples

    @pytest.mark.parametrize("length", [100, 8000, 16000, 48000, 320000])
    def test_audio_token_count_positive(self, length):
        ext = _make_extractor()
        assert ext.audio_token_count(length) >= 1

    def test_split_audio_into_clips_preserves_total_samples(self):
        ext = _make_extractor()
        audio = np.random.randn(50000).astype(np.float32)
        clips = ext._split_audio_into_clips(audio)
        total = sum(c.shape[0] for c in clips)
        assert total >= audio.shape[0]

    def test_call_returns_expected_keys(self):
        ext = _make_extractor()
        audios = [np.random.randn(16000).astype(np.float32)]
        result = ext(audios, sampling_rate=16000, return_tensors="pt")
        assert "input_features" in result
        assert "attention_mask" in result
        assert "audio_num_clips" in result
        assert result["audio_num_clips"].shape == (1,)
        assert result["audio_num_clips"].item() >= 1


class TestProjectedParakeet:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @torch.no_grad()
    def test_projected_parakeet_forward_contract(self):
        device = torch.device("cuda")
        sound_cfg = _make_sound_config()
        llm_hidden = 128
        dtype = torch.bfloat16
        model = ProjectedParakeet(sound_cfg, llm_hidden_size=llm_hidden, dtype=dtype)
        model.to(device).eval()

        batch, time_steps = 2, 200
        features = torch.randn(
            batch, time_steps, sound_cfg.num_mel_bins, device=device, dtype=dtype
        )
        mask = torch.ones(batch, time_steps, dtype=torch.long, device=device)
        out = model(features, attention_mask=mask)

        assert out.ndim == 3
        assert out.shape[0] == batch
        assert out.shape[2] == llm_hidden
        assert out.dtype == torch.bfloat16
