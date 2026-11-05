# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for frozen-video retake initialization."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import (
    AudioPatchifier,
    VideoLatentPatchifier,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import (
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from tensorrt_llm._torch.visual_gen.models.ltx23 import pipeline_ltx23_retake as retake_module
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import LTX23RetakePipeline

pytestmark = pytest.mark.cpu_only


def test_ltx23_retake_frozen_video_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep video clean before denoising without changing the audio RNG sequence."""
    dtype = torch.bfloat16
    pixel_shape = VideoPixelShape(batch=1, frames=17, height=32, width=32, fps=8.0)
    video_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=2)
    audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    source_video = torch.full(video_shape.to_torch_shape(), 2.0, dtype=dtype)
    source_audio = torch.full(audio_shape.to_torch_shape(), 3.0, dtype=dtype)
    pipeline = LTX23RetakePipeline.__new__(LTX23RetakePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._device = torch.device("cpu")
    pipeline.pipeline_config = SimpleNamespace(torch_dtype=dtype)
    pipeline.transformer_in_channels = video_shape.channels
    pipeline.video_patchifier = VideoLatentPatchifier(patch_size=1)
    pipeline.audio_patchifier = AudioPatchifier(patch_size=1)
    pipeline.video_encoder = Mock()
    pipeline.video_encoder.tiled_encode.return_value = source_video
    pipeline._read_source_video = Mock()
    pipeline._encode_source_audio_latent = Mock(return_value=source_audio)
    pipeline.audio_decoder = Mock()
    pipeline.vocoder = Mock()
    pipeline.transformer = Mock()
    context = torch.zeros(1, 1, 2, dtype=dtype)
    pipeline.text_encoder = Mock()
    pipeline._encode_prompt = Mock(return_value=(context, None))
    pipeline._process_connectors = Mock(return_value=(context, context, None))
    monkeypatch.setattr(retake_module, "get_videostream_metadata", lambda _: pixel_shape)
    monkeypatch.setattr(retake_module, "decode_audio_from_file", lambda _: object())

    class InitializationComplete(Exception):
        """Stop at the denoising boundary without running or decoding the model."""

    pipeline.denoise = Mock(side_effect=InitializationComplete)
    req = SimpleNamespace(params=SimpleNamespace(num_inference_steps=8, seed=42, extra_params={}))
    with pytest.raises(InitializationComplete):
        pipeline._run_retake(
            req,
            b"source video bytes",
            0.5,
            1.5,
            "prompt",
            Mock(),
            regenerate_video=False,
            regenerate_audio=True,
        )

    inputs = pipeline.denoise.call_args.kwargs
    clean_video = pipeline.video_patchifier.patchify(source_video)
    torch.testing.assert_close(inputs["latents"], clean_video, rtol=0, atol=0)

    audio_state = inputs["extra_streams"]["audio"][0]
    pipeline._masked_transformer_step = Mock(return_value=(None, None))
    inputs["forward_fn"](
        inputs["latents"],
        {"audio": audio_state},
        0,
        inputs["timesteps"][:1],
        inputs["prompt_embeds"],
        {},
    )
    conditioning = pipeline._masked_transformer_step.call_args.kwargs
    assert conditioning["regenerate_video"] is False
    assert conditioning["regenerate_audio"] is True
    assert torch.count_nonzero(conditioning["denoise_mask"]) == 0
    audio_mask = conditioning["audio_denoise_mask"]
    assert audio_mask.any() and not audio_mask.all()

    clean_audio = pipeline.audio_patchifier.patchify(source_audio)
    generator = torch.Generator().manual_seed(req.params.seed)
    torch.randn(clean_video.shape, generator=generator, dtype=dtype)
    audio_noise = torch.randn(clean_audio.shape, generator=generator, dtype=dtype)
    expected_audio = torch.lerp(
        clean_audio.float(), audio_noise.float(), audio_mask.unsqueeze(-1)
    ).to(dtype)
    torch.testing.assert_close(audio_state, expected_audio, rtol=0, atol=0)
    pipeline.transformer.assert_not_called()
