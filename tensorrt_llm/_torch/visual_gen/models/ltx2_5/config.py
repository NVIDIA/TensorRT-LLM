# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-scoped configuration for BF16 IC-LoRA inference."""

from pydantic import Field

from tensorrt_llm.llmapi.utils import StrictBaseModel


class LTX25Config(StrictBaseModel):
    """Explicit split-checkpoint locations; no customer recipe defaults."""

    text_encoder_path: str = Field(default="", description="LTX-tuned Gemma4 checkpoint.")
    video_vae_path: str = Field(default="", description="LTX-2.5 convolutional video VAE.")
    audio_vae_path: str = Field(default="", description="LTX-2.5 audio VAE and vocoder.")
    spatial_upsampler_path: str = Field(default="", description="LTX-2.5 spatial latent upsampler.")
    iclora_path: str = Field(default="", description="Official IC-LoRA adapter for Stage 1.")
    iclora_scale: float = Field(
        default=1.0, allow_inf_nan=False, description="Stage-1 adapter scale."
    )
