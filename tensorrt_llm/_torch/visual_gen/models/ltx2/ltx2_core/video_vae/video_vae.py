# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM call signatures for current convolutional video VAE implementations."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.conv_video_decoder import (
    ConvVideoDecoder,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.conv_video_decoder import (
    _make_decoder_block as _make_decoder_block,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import (
    VideoEncoder as UpstreamVideoEncoder,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import (
    _make_encoder_block as _make_encoder_block,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import (
    get_video_chunks_number as get_video_chunks_number,
)

from .tiling import convert_tiling


class VideoEncoder(UpstreamVideoEncoder):
    def __init__(self, *args, causal=True, timestep_conditioning=False, **kwargs):
        if not causal or timestep_conditioning:
            raise ValueError(
                "Current upstream video encoder requires causal, unconditioned encoding"
            )
        super().__init__(*args, **kwargs)

    def tiled_encode(self, video, tiling_config=None):
        config = convert_tiling(
            tiling_config, video.shape[-2], video.shape[-1], self.video_scale_factors, encoding=True
        )
        return super().tiled_encode(video, config)


class VideoDecoder(ConvVideoDecoder):
    def _prepare_tiles(self, latent, tiling_config=None):
        scales = self.video_downscale_factors
        config = convert_tiling(
            tiling_config, latent.shape[-2] * scales.height, latent.shape[-1] * scales.width, scales
        )
        return super()._prepare_tiles(latent, config)

    def tiled_decode(self, latent, tiling_config=None, timestep=None, generator=None):
        scales = self.video_downscale_factors
        config = convert_tiling(
            tiling_config, latent.shape[-2] * scales.height, latent.shape[-1] * scales.width, scales
        )
        return super().tiled_decode(latent, config, timestep, generator)


def decode_video(latent, video_decoder, tiling_config=None, generator=None):
    scales = video_decoder.video_downscale_factors
    config = convert_tiling(
        tiling_config, latent.shape[-2] * scales.height, latent.shape[-1] * scales.width, scales
    )
    for frames in video_decoder.decode_video(latent, config, generator):
        yield (frames * 255.0).to(torch.uint8)
