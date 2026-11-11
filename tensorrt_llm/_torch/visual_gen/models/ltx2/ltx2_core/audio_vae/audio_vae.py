# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports from the pinned November 4 LTX core."""

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import (
    LATENT_DOWNSAMPLE_FACTOR as LATENT_DOWNSAMPLE_FACTOR,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import (
    AudioDecoder as AudioDecoder,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import (
    build_mid_block as build_mid_block,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import (
    decode_audio as upstream_decode_audio,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.audio_vae import (
    run_mid_block as run_mid_block,
)


def decode_audio(latent, audio_decoder, vocoder):
    audio = upstream_decode_audio(latent, audio_decoder, vocoder)
    return audio.waveform.unsqueeze(0) if latent.shape[0] == 1 else audio.waveform
