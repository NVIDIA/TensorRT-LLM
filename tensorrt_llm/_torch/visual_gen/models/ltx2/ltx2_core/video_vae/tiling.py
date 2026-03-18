# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import itertools
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Tuple

import torch


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))
    mask = torch.ones(length)
    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in
    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out
    return mask.clamp_(0, 1)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] = 0
    if right_ramp > 0:
        mask[-right_ramp:] = 0
    return mask


@dataclass(frozen=True)
class SpatialTilingConfig:
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(
                f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}"
            )
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}"
            )
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}"
            )
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(
                f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}"
            )
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(
                f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}"
            )
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(
                f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}"
            )
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


@dataclass(frozen=True)
class TensorTilingSpec:
    original_shape: torch.Size
    dimension_intervals: Tuple[DimensionIntervals, ...]


SplitOperation = Callable[[int], DimensionIntervals]
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


class Tile(NamedTuple):
    in_coords: Tuple[slice, ...]
    out_coords: Tuple[slice, ...]
    masks_1d: Tuple[Tuple[torch.Tensor, ...]]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: List[torch.Tensor] = []
        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                one = torch.ones(1)
                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask
        return combined_mask


def create_tiles_from_intervals_and_mappers(
    intervals: TensorTilingSpec,
    mappers: List[MappingOperation],
) -> List[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        starts = dimension_intervals.starts
        ends = dimension_intervals.ends
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)
    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(
        tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True
    ):
        tiles.append(Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d))
    return tiles


def create_tiles(
    tensor_shape: torch.Size,
    splitters: List[SplitOperation],
    mappers: List[MappingOperation],
) -> List[Tile]:
    if len(splitters) != len(tensor_shape):
        raise ValueError(
            f"Number of splitters must equal number of dimensions, got {len(splitters)} and {len(tensor_shape)}"
        )
    if len(mappers) != len(tensor_shape):
        raise ValueError(
            f"Number of mappers must equal number of dimensions, got {len(mappers)} and {len(tensor_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, tensor_shape, strict=True)]
    tiling_spec = TensorTilingSpec(
        original_shape=tensor_shape, dimension_intervals=tuple(intervals)
    )
    return create_tiles_from_intervals_and_mappers(tiling_spec, mappers)
