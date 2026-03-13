# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import logging
from dataclasses import replace
from typing import Any, Callable, Iterator, List, Tuple

import torch
from einops import rearrange
from torch import nn

from ..normalization import PixelNorm
from ..timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..types import SpatioTemporalScaleFactors, VideoLatentShape
from .convolution import make_conv_nd
from .enums import NormLayerType, PaddingModeType
from .ops import PerChannelStatistics, patchify, unpatchify
from .resnet import ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from .tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
)

logger: logging.Logger = logging.getLogger(__name__)


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Build a single encoder block and return (block, out_channels)."""
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels * block_config.get("multiplier", 1)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            residual=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            residual=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            residual=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown encoder layer: {block_name}")
    return block, out_channels


class VideoEncoder(nn.Module):
    """Causal 3D video encoder for encoding images/video into the VAE latent space.

    Used for image-to-video conditioning: encodes a reference image into
    latent tokens that seed the first frame of the denoising process.
    """

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int | dict]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = True,
        timestep_conditioning: bool = False,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        patched_in_channels = in_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        feature_channels = out_channels
        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=patched_in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])
        for block_name, block_params in encoder_blocks:
            block_config = (
                {"num_layers": block_params} if isinstance(block_params, int) else block_params
            )
            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )
            self.down_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels,
                num_groups=self._norm_num_groups,
                eps=1e-6,
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels + 1,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space input into normalized VAE latents.

        Args:
            sample: Pixel tensor ``(B, 3, F, H, W)`` in ``[-1, 1]``.

        Returns:
            Latent tensor ``(B, C_latent, F_latent, H_latent, W_latent)``,
            per-channel normalized.
        """
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample, causal=self.causal)

        for down_block in self.down_blocks:
            sample = down_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        mean = sample[:, : self.out_channels, ...]
        return self.per_channel_statistics.normalize(mean)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")
    return block, out_channels


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()
        self.video_downscale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)
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
        self.up_blocks = nn.ModuleList([])
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = (
                {"num_layers": block_params} if isinstance(block_params, int) else block_params
            )
            block, feature_channels = _make_decoder_block(
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
                num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
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
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = sample.shape[0]
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(), generator=generator, dtype=sample.dtype, device=sample.device
                )
                * self.decode_noise_scale
            )
            sample = noise + (1.0 - self.decode_noise_scale) * sample
        sample = self.per_channel_statistics.un_normalize(sample)
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full(
                (batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype
            )
        sample = self.conv_in(sample, causal=self.causal)
        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)
        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(
                    sample,
                    causal=self.causal,
                    timestep=scaled_timestep if self.timestep_conditioning else None,
                    generator=generator,
                )
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)
        sample = self.conv_norm_out(sample)
        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(), hidden_dtype=sample.dtype
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        return sample

    def _prepare_tiles(
        self, latent: torch.Tensor, tiling_config: TilingConfig | None = None
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            long_side = max(latent.shape[3], latent.shape[4])

            def enable_on_axis(axis_idx: int, factor: int) -> None:
                size = cfg.tile_size_in_pixels // factor
                overlap = cfg.tile_overlap_in_pixels // factor
                axis_length = latent.shape[axis_idx]
                lower_threshold = max(2, overlap + 1)
                tile_size = max(lower_threshold, round(size * axis_length / long_side))
                splitters[axis_idx] = split_with_symmetric_overlaps(tile_size, overlap)
                mappers[axis_idx] = make_mapping_operation(
                    map_spatial_interval_to_pixel, scale=factor
                )

            enable_on_axis(3, self.video_downscale_factors.height)
            enable_on_axis(4, self.video_downscale_factors.width)
        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = cfg.tile_size_in_frames // self.video_downscale_factors.time
            overlap = cfg.tile_overlap_in_frames // self.video_downscale_factors.time
            splitters[2] = split_temporal_latents(tile_size, overlap)
            mappers[2] = make_mapping_operation(
                map_temporal_interval_to_frame, scale=self.video_downscale_factors.time
            )
        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(
            self.video_downscale_factors
        )
        tiles = self._prepare_tiles(latent, tiling_config)
        temporal_groups = self._group_tiles_by_temporal_slice(tiles)
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None
        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]
            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )
            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )
            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
            )
            if previous_chunk is not None:
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(
                        curr_temporal_slice.start - previous_temporal_slice.start, None
                    )
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[
                        :, :, slice(0, overlap_len), :, :
                    ]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                        :, :, slice(0, overlap_len), :, :
                    ]
                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[
                        :, :, temporal_overlap_slice, :, :
                    ]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                        :, :, temporal_overlap_slice, :, :
                    ]
                previous_weights = previous_weights.clamp(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp(min=1e-8)
            yield previous_chunk / previous_weights

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        if not tiles:
            return []
        groups = []
        current_slice = tiles[0].out_coords[2]
        current_group = []
        for tile in tiles:
            tile_slice = tile.out_coords[2]
            if tile_slice == current_slice:
                current_group.append(tile)
            else:
                groups.append(current_group)
                current_slice = tile_slice
                current_group = [tile]
        if current_group:
            groups.append(current_group)
        return groups

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        temporal_slice = group_tiles[0].out_coords[2]
        weights = torch.zeros_like(buffer)
        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]
            actual_temporal_len = min(
                expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset
            )
            chunk_coords = (
                slice(None),
                slice(None),
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],
                tile.out_coords[4],
            )
            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask
            buffer[chunk_coords] += decoded_slice * mask_slice
            weights[chunk_coords] += mask_slice
        return weights


def decode_video(
    latent: torch.Tensor,
    video_decoder: VideoDecoder,
    tiling_config: TilingConfig | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
        frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames

    if tiling_config is not None:
        for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
            yield convert_to_uint8(frames)
    else:
        decoded_video = video_decoder(latent, generator=generator)
        yield convert_to_uint8(decoded_video)


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    frame_stride = cfg.tile_size_in_frames - cfg.tile_overlap_in_frames
    return (num_frames - 1 + frame_stride - 1) // frame_stride


def split_with_symmetric_overlaps(size: int, overlap: int) -> SplitOperation:
    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(
            starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps
        )

    return split


def split_temporal_latents(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_with_symmetric_overlaps(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def make_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor | None]],
    scale: int,
) -> MappingOperation:
    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
        for i in range(len(intervals.starts)):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_interval_to_frame(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp_frames = 0 if left_ramp == 0 else 1 + (left_ramp - 1) * scale
    right_ramp_frames = right_ramp * scale
    mask_1d = compute_trapezoidal_mask_1d(stop - start, left_ramp_frames, right_ramp_frames, True)
    return slice(start, stop), mask_1d


def map_spatial_interval_to_pixel(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    mask_1d = compute_trapezoidal_mask_1d(
        stop - start, left_ramp * scale, right_ramp * scale, False
    )
    return slice(start, stop), mask_1d
