# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration and statistics-boundary adapters."""

from types import SimpleNamespace

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.blur_downsample import (
    BlurDownsample as BlurDownsample,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.model import (
    LatentUpsampler as LatentUpsampler,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.model import (
    upsample_video as _upsample_video,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.model_configurator import (
    LatentUpsamplerConfigurator as UpstreamConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.pixel_shuffle import (
    PixelShuffleND as PixelShuffleND,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.res_block import (
    ResBlock as ResBlock,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.spatial_rational_resampler import (
    SpatialRationalResampler as SpatialRationalResampler,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.upsampler.spatial_rational_resampler import (
    _rational_for_scale as _rational_for_scale,
)


class LatentUpsamplerConfigurator:
    @classmethod
    def from_config(cls, config):
        return UpstreamConfigurator.from_metadata({"config": config})


def upsample_video(latent, per_channel_statistics, upsampler):
    def run_upsampler(value):
        # Cast after unnormalizing, preserving TRTLLM's float32 latent boundary.
        return upsampler(value.to(next(upsampler.parameters()).dtype))

    return _upsample_video(
        latent, SimpleNamespace(per_channel_statistics=per_channel_statistics), run_upsampler
    )
