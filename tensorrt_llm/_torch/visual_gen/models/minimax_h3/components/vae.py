# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ..packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from ..parallel.vae import TiledAutoencoderKLMiniMaxH3


def _prepare_vae_decoder(vae: TiledAutoencoderKLMiniMaxH3) -> None:
    """Prepare the FP16 video decoder and size its tile batches before compilation."""
    for module in vae.decoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.float16)
    vae.calibrate_tile_decode_memory()


def decode_video(
    vae,
    offloader,
    device,
    patch_size,
    rows: torch.Tensor,
    num_condition_rows: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    latents = unpatchify_video_tokens(
        rows[num_condition_rows:],
        num_latent_frames,
        latent_height,
        latent_width,
        vae.config.latent_channels,
        patch_size,
    )
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=device,
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=device,
    ).view(1, -1, 1, 1, 1)
    latents = latents * latents_std + latents_mean
    with (
        offloader.context_if_requested("vae"),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
    ):
        video = vae.decode(latents, return_dict=False)[0]
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    return (video.permute(0, 2, 3, 4, 1) * 255).round().to(torch.uint8)


def decode_audio(
    vae,
    offloader,
    device,
    rows: torch.Tensor,
    num_audio_latents: int,
) -> torch.Tensor:
    latents = unpack_audio_tokens(rows, num_audio_latents)
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=device,
    ).view(1, -1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=device,
    ).view(1, -1, 1)
    with offloader.context_if_requested("audio_vae"):
        audio = vae.decode(
            latents * latents_std + latents_mean,
            return_dict=False,
        )[0]
    return audio.float().permute(1, 0, 2)
