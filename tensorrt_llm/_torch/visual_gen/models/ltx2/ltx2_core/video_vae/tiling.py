# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate TRTLLM's legacy tiling options to current upstream size layouts."""

from dataclasses import dataclass

from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DEFAULT_MAPPING_OPERATION as DEFAULT_MAPPING_OPERATION,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DEFAULT_SPLIT_OPERATION as DEFAULT_SPLIT_OPERATION,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DimensionIntervals as DimensionIntervals,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DimensionSizeConfig,
    TileSizeConfig,
    _validate_size_axis,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import Tile as Tile
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    compute_rectangular_mask_1d as compute_rectangular_mask_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    compute_trapezoidal_mask_1d as compute_trapezoidal_mask_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import create_tiles as create_tiles
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import VIDEO_SCALE_FACTORS


@dataclass(frozen=True)
class SpatialTilingConfig:
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self):
        if self.tile_size_in_pixels == 0:
            raise ValueError("Use spatial_config=None to disable spatial tiling")
        _validate_size_axis(
            DimensionSizeConfig(self.tile_size_in_pixels, self.tile_overlap_in_pixels),
            32,
            "spatial",
        )


@dataclass(frozen=True)
class TemporalTilingConfig:
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self):
        if self.tile_size_in_frames == 0:
            raise ValueError("Use temporal_config=None to disable temporal tiling")
        _validate_size_axis(
            DimensionSizeConfig(self.tile_size_in_frames, self.tile_overlap_in_frames),
            8,
            "temporal",
        )


@dataclass(frozen=True)
class TilingConfig:
    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls):
        return cls(SpatialTilingConfig(512, 64), TemporalTilingConfig(64, 24))

    def to_upstream(self, height, width, scale_factors=VIDEO_SCALE_FACTORS, *, encoding=False):
        temporal = self.temporal_config
        frames = (
            DimensionSizeConfig()
            if temporal is None
            else DimensionSizeConfig(temporal.tile_size_in_frames, temporal.tile_overlap_in_frames)
        )
        spatial = self.spatial_config
        if spatial is None:
            return TileSizeConfig(frames=frames)
        size = DimensionSizeConfig(spatial.tile_size_in_pixels, spatial.tile_overlap_in_pixels)
        if encoding:
            return TileSizeConfig(frames=frames, height=size, width=size)
        return TileSizeConfig.from_long_side(
            long_side=size, height=height, width=width, scale_factors=scale_factors, frames=frames
        )

    def video_chunks_number(self, num_frames):
        return self.to_upstream(1, 1).video_chunks_number(num_frames)


def convert_tiling(config, height, width, scales=VIDEO_SCALE_FACTORS, *, encoding=False):
    if isinstance(config, TilingConfig):
        return config.to_upstream(height, width, scales, encoding=encoding)
    return config
