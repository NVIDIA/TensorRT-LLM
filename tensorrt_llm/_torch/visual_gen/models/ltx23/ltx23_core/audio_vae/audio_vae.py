# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.audio_vae import (
    AudioEncoder as AudioEncoder,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.audio_vae import (
    encode_audio as upstream_encode_audio,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import Audio


def encode_audio(audio, audio_encoder):
    parameter = next(audio_encoder.parameters())
    samples = torch.as_tensor(audio.samples, device=parameter.device)
    if samples.ndim != 3:
        raise ValueError(f"Expected audio samples with shape (B, C, T); got {samples.shape}.")
    channels = audio_encoder.conv_in.conv.in_channels
    if samples.shape[1] == 1 and channels == 2:
        samples = samples.expand(-1, 2, -1).contiguous()
    elif samples.shape[1] != channels:
        raise ValueError(f"Audio encoder expects {channels} channels, got {samples.shape[1]}")
    return upstream_encode_audio(
        Audio(waveform=samples, sampling_rate=audio.sample_rate), audio_encoder
    )
