# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for LTX-2 two-stage audio handoff."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2 import (
    pipeline_ltx2_two_stages as ltx2_two_stages,
)


def _make_audio_preservation_pipeline(monkeypatch, stage1_video, stage1_audio):
    """Build a minimal two-stage pipeline that exercises forward() control flow."""

    class FakeLoRACache:
        applied_count = 0

        def bind_merged(self):
            pass

        def bind_original(self):
            pass

    class FakeTransformer:
        def set_ulysses_enabled(self, enabled):
            self.ulysses_enabled = enabled

    class FakeVideoDecoder:
        def tiled_decode(self, video_latents, tiling_config, generator=None):
            yield video_latents + 1000.0

    def fake_stage1_forward(self, *args, **kwargs):
        return ltx2_two_stages.PipelineOutput(
            video=stage1_video.clone(),
            audio=stage1_audio.clone(),
            frame_rate=float(kwargs["frame_rate"]),
            audio_sample_rate=16000,
        )

    def fake_upsample_video(video_latents, per_ch_stats, spatial_upsampler):
        return video_latents + 1.0

    def fake_refinement_denoise(**kwargs):
        audio_latents = kwargs["audio_latents"]
        assert audio_latents is not None
        audio_latents.add_(100.0)
        return kwargs["video_latents"] + 10.0, audio_latents

    pipeline = object.__new__(ltx2_two_stages.LTX2TwoStagesPipeline)
    pipeline.pipeline_config = type("FakePipelineConfig", (), {"torch_dtype": torch.float32})()
    pipeline._parallel_vae_enabled = False
    pipeline._distilled_lora_weight_cache = FakeLoRACache()
    pipeline._lora_cuda_graph_state = "original"
    pipeline.transformer = FakeTransformer()
    pipeline.spatial_upsampler = object()
    pipeline.video_decoder = FakeVideoDecoder()
    pipeline.audio_decoder = object()
    pipeline.vocoder = object()
    pipeline.audio_sampling_rate = 16000
    pipeline._assert_cuda_graph_safe_lora_bindings = lambda: None
    pipeline._get_per_channel_statistics = lambda: object()
    pipeline._refinement_denoise = fake_refinement_denoise

    monkeypatch.setattr(
        ltx2_two_stages.LTX2Pipeline,
        "forward",
        fake_stage1_forward,
    )
    monkeypatch.setattr(ltx2_two_stages, "upsample_video", fake_upsample_video)
    monkeypatch.setattr(ltx2_two_stages, "postprocess_video_tensor", lambda video: video)
    return pipeline


def test_two_stage_latent_output_preserves_stage1_audio(monkeypatch):
    """Stage 2 audio must not replace Stage 1 audio for latent outputs."""
    stage1_video = torch.zeros(1, 1, 1, 1, 1)
    stage1_audio = torch.full((1, 1, 2, 2), 2.0)
    pipeline = _make_audio_preservation_pipeline(monkeypatch, stage1_video, stage1_audio)

    output = pipeline.forward(prompt="prompt", seed=7, output_type="latent")

    assert torch.equal(output.video, stage1_video + 11.0)
    assert torch.equal(output.audio, stage1_audio)


def test_two_stage_decode_uses_stage1_audio(monkeypatch):
    """Audio decode must consume Stage 1 audio, not Stage 2 returned audio."""
    stage1_video = torch.zeros(1, 1, 1, 1, 1)
    stage1_audio = torch.full((1, 1, 2, 2), 3.0)
    pipeline = _make_audio_preservation_pipeline(monkeypatch, stage1_video, stage1_audio)
    decoded_audio_inputs = []

    def fake_decode_audio(audio_latents, audio_decoder, vocoder):
        decoded_audio_inputs.append(audio_latents.clone())
        return audio_latents + 1000.0

    monkeypatch.setattr(ltx2_two_stages, "decode_audio", fake_decode_audio)

    output = pipeline.forward(prompt="prompt", seed=7, output_type="pt")

    assert len(decoded_audio_inputs) == 1
    assert torch.equal(decoded_audio_inputs[0], stage1_audio)
    assert torch.equal(output.audio, stage1_audio + 1000.0)
