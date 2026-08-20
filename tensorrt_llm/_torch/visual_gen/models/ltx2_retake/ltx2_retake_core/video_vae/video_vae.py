# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 video VAE extensions used by the retake workflow."""

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from ....ltx2.ltx2_core.normalization import PixelNorm
from ....ltx2.ltx2_core.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ....ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from ....ltx2.ltx2_core.video_vae.convolution import make_conv_nd
from ....ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ....ltx2.ltx2_core.video_vae.ops import PerChannelStatistics
from ....ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from ....ltx2.ltx2_core.video_vae.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    Tile,
    TilingConfig,
    compute_rectangular_mask_1d,
    create_tiles,
)
from ....ltx2.ltx2_core.video_vae.video_vae import (
    VideoDecoder,
    VideoEncoder,
    _make_decoder_block,
    make_mapping_operation,
    split_with_symmetric_overlaps,
)

logger: logging.Logger = logging.getLogger(__name__)

_MIN_SPATIAL_OVERLAP_PIXELS = 64
_MIN_TEMPORAL_OVERLAP_FRAMES = 16


class _RetakePerChannelStatistics(PerChannelStatistics):
    """LTX-2.3 statistics with the two buffers present in its checkpoint."""

    def __init__(self, latent_channels: int = 128) -> None:
        nn.Module.__init__(self)
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))


def _split_temporal_frames(tile_size: int, overlap: int) -> Callable[[int], DimensionIntervals]:
    non_causal_split = split_with_symmetric_overlaps(tile_size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        ends = list(intervals.ends)
        ends[:-1] = [end + 1 for end in ends[:-1]]
        return replace(intervals, ends=ends, right_ramps=[0] * len(ends))

    return split


def _map_temporal_interval(
    begin: int, end: int, left_ramp: int, right_ramp: int, scale: int
) -> tuple[slice, torch.Tensor]:
    start = begin // scale
    stop = (end - 1) // scale + 1
    left = 0 if left_ramp == 0 else 1 + (left_ramp - 1) // scale
    right = right_ramp // scale
    if right:
        raise ValueError(f"Retake encode tiles require a zero right ramp; got {right_ramp}")
    return slice(start, stop), compute_rectangular_mask_1d(stop - start, left, right)


def _map_spatial_interval(
    begin: int, end: int, left_ramp: int, right_ramp: int, scale: int
) -> tuple[slice, torch.Tensor]:
    start = begin // scale
    stop = end // scale
    return slice(start, stop), compute_rectangular_mask_1d(
        stop - start,
        max(0, left_ramp // scale - 1),
        0 if right_ramp == 0 else 1,
    )


def _prepare_encode_tiles(
    video: torch.Tensor,
    tiling_config: TilingConfig,
    scales: SpatioTemporalScaleFactors,
) -> list[Tile]:
    splitters = [DEFAULT_SPLIT_OPERATION] * video.ndim
    mappers = [DEFAULT_MAPPING_OPERATION] * video.ndim
    if tiling_config.spatial_config is not None:
        config = tiling_config.spatial_config
        overlap = max(config.tile_overlap_in_pixels, _MIN_SPATIAL_OVERLAP_PIXELS)
        for axis, scale in ((3, scales.height), (4, scales.width)):
            splitters[axis] = split_with_symmetric_overlaps(config.tile_size_in_pixels, overlap)
            mappers[axis] = make_mapping_operation(_map_spatial_interval, scale=scale)
    if tiling_config.temporal_config is not None:
        config = tiling_config.temporal_config
        overlap = max(config.tile_overlap_in_frames, _MIN_TEMPORAL_OVERLAP_FRAMES)
        splitters[2] = _split_temporal_frames(config.tile_size_in_frames, overlap)
        mappers[2] = make_mapping_operation(_map_temporal_interval, scale=scales.time)
    return create_tiles(video.shape, splitters, mappers)


class RetakeVideoEncoder(VideoEncoder):
    """VideoEncoder with overlap-aware tiled encoding for full retake clips."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: list[tuple[str, int | dict[str, Any]]] | None = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = True,
        timestep_conditioning: bool = False,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_blocks=encoder_blocks or [],
            patch_size=patch_size,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            encoder_spatial_padding_mode=encoder_spatial_padding_mode,
        )
        self.per_channel_statistics = _RetakePerChannelStatistics(latent_channels=self.out_channels)

    def tiled_encode(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        if tiling_config is None:
            return self.forward(video)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        scales = VIDEO_SCALE_FACTORS
        batch, _, frames, height, width = video.shape
        remainder = (frames - 1) % scales.time
        if remainder:
            logger.warning("Cropping %d video frame(s) for causal VAE encode", remainder)
            video = video[:, :, :-remainder]
            frames = video.shape[2]

        latent_shape = VideoLatentShape(
            batch=batch,
            channels=self.out_channels,
            frames=(frames - 1) // scales.time + 1,
            height=height // scales.height,
            width=width // scales.width,
        )
        latents = torch.zeros(latent_shape.to_torch_shape(), device=device, dtype=dtype)
        weights = torch.zeros_like(latents)
        for tile in _prepare_encode_tiles(video, tiling_config, scales):
            video_tile = video[tile.in_coords].to(device=device, dtype=dtype)
            latent_tile = self.forward(video_tile)
            mask = tile.blend_mask(device, dtype)
            latents[tile.out_coords] += latent_tile * mask
            weights[tile.out_coords] += mask
        return latents / weights.clamp(min=1e-8)


def _make_retake_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> tuple[nn.Module, int]:
    if block_name in ("compress_time", "compress_space"):
        multiplier = block_config.get("multiplier", 1)
        stride = (2, 1, 1) if block_name == "compress_time" else (1, 2, 2)
        return (
            DepthToSpaceUpsample(
                dims=convolution_dimensions,
                in_channels=in_channels,
                stride=stride,
                out_channels_reduction_factor=multiplier,
                spatial_padding_mode=spatial_padding_mode,
            ),
            in_channels // multiplier,
        )
    return _make_decoder_block(
        block_name=block_name,
        block_config=block_config,
        in_channels=in_channels,
        convolution_dimensions=convolution_dimensions,
        norm_layer=norm_layer,
        timestep_conditioning=timestep_conditioning,
        norm_num_groups=norm_num_groups,
        spatial_padding_mode=spatial_padding_mode,
    )


class RetakeVideoDecoder(VideoDecoder):
    """Native decoder matching the LTX-2.3 retake checkpoint layout."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[tuple[str, int | dict[str, Any]]] | None = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        base_channels: int = 128,
    ) -> None:
        nn.Module.__init__(self)
        decoder_blocks = decoder_blocks or []
        self.video_downscale_factors = VIDEO_SCALE_FACTORS
        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = _RetakePerChannelStatistics(latent_channels=in_channels)
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        feature_channels = base_channels * 8
        # VideoDecoder.forward passes self.causal at runtime, so these layers must
        # be CausalConv3d instances even when symmetric temporal padding is selected.
        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )
        self.up_blocks = nn.ModuleList()
        for block_name, block_params in reversed(decoder_blocks):
            block_config = (
                {"num_layers": block_params} if isinstance(block_params, int) else block_params
            )
            block, feature_channels = _make_retake_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )
            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels,
                num_groups=self._norm_num_groups,
                eps=1e-6,
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
        else:
            raise ValueError(f"Unsupported decoder norm layer: {norm_layer}")
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2,
                size_emb_dim=0,
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))
