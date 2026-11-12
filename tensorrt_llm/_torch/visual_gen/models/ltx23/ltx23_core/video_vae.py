# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 channel-reducing decoder blocks and overlap-aware tiled encoding."""

import itertools
import logging
from typing import Any

import torch
import torch.nn as nn

from ...ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from ...ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ...ltx2.ltx2_core.video_vae.ops import PerChannelStatistics
from ...ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from ...ltx2.ltx2_core.video_vae.tiling import (
    DEFAULT_SPLIT_OPERATION,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
)
from ...ltx2.ltx2_core.video_vae.video_vae import VideoDecoder as LTX2VideoDecoder
from ...ltx2.ltx2_core.video_vae.video_vae import VideoEncoder as LTX2VideoEncoder
from ...ltx2.ltx2_core.video_vae.video_vae import (
    _make_decoder_block,
    split_temporal_latents,
    split_with_symmetric_overlaps,
)

logger = logging.getLogger(__name__)

_MIN_SPATIAL_OVERLAP_PIXELS = 64
_MIN_TEMPORAL_OVERLAP_FRAMES = 16

_COMPRESS_STRIDES = {
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
    "compress_all": (2, 2, 2),
}


def _make_ltx23_decoder_block(
    block_name: str,
    block_config: dict,
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
    timestep_conditioning: bool = False,
) -> tuple[nn.Module, int]:
    if block_name in ("compress_time", "compress_space"):
        multiplier = block_config.get("multiplier", 1)
        out_channels = in_channels // multiplier
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=_COMPRESS_STRIDES[block_name],
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=multiplier,
            spatial_padding_mode=spatial_padding_mode,
        )
        return block, out_channels
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


class LTX23VideoDecoder(LTX2VideoDecoder):
    """LTX-2.3 video decoder: LTX-2 primitives, LTX-2.3 channel recipe."""

    _channel_reducing_blocks = tuple(_COMPRESS_STRIDES)
    _make_decoder_block = staticmethod(_make_ltx23_decoder_block)

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[tuple[str, int | dict[str, Any]]] | None = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ) -> None:
        super().__init__(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks or [],
            patch_size=patch_size,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=spatial_padding_mode,
        )
        self.video_downscale_factors = VIDEO_SCALE_FACTORS


class LTX23VideoDecoderConfigurator:
    """Create an LTX23VideoDecoder from the native LTX-2.3 config dict."""

    @classmethod
    def from_config(cls, config: dict) -> LTX23VideoDecoder:
        vae = config.get("vae", {})
        padding_mode = vae.get(
            "spatial_padding_mode", vae.get("decoder_spatial_padding_mode", "reflect")
        )
        return LTX23VideoDecoder(
            convolution_dimensions=vae.get("dims", 3),
            in_channels=vae.get("latent_channels", 128),
            out_channels=vae.get("out_channels", 3),
            decoder_blocks=vae.get("decoder_blocks", []),
            patch_size=vae.get("patch_size", 4),
            norm_layer=NormLayerType(vae.get("norm_layer", "pixel_norm")),
            causal=vae.get("causal_decoder", False),
            spatial_padding_mode=PaddingModeType(padding_mode),
        )


class _LTX23PerChannelStatistics(PerChannelStatistics):
    """Statistics buffers present in the LTX-2.3 encoder checkpoint."""

    def __init__(self, latent_channels: int = 128) -> None:
        nn.Module.__init__(self)
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))


def _prepare_encode_tiles(
    video: torch.Tensor,
    tiling_config: TilingConfig,
    scales: SpatioTemporalScaleFactors,
) -> list[Tile]:
    """Split on the latent grid and map causal temporal tiles back to pixels."""
    latent_shape = (
        (video.shape[2] - 1) // scales.time + 1,
        video.shape[3] // scales.height,
        video.shape[4] // scales.width,
    )
    splitters = [DEFAULT_SPLIT_OPERATION] * 3
    if tiling_config.spatial_config is not None:
        config = tiling_config.spatial_config
        overlap = max(config.tile_overlap_in_pixels, _MIN_SPATIAL_OVERLAP_PIXELS)
        for axis, scale in ((1, scales.height), (2, scales.width)):
            latent_overlap = overlap // scale
            size = max(2, latent_overlap + 1, config.tile_size_in_pixels // scale)
            splitters[axis] = split_with_symmetric_overlaps(size, latent_overlap)
    if tiling_config.temporal_config is not None:
        config = tiling_config.temporal_config
        overlap = max(config.tile_overlap_in_frames, _MIN_TEMPORAL_OVERLAP_FRAMES)
        latent_overlap = overlap // scales.time
        size = max(2, latent_overlap + 1, config.tile_size_in_frames // scales.time)
        splitters[0] = split_temporal_latents(size, latent_overlap)
    intervals = [
        list(zip(value.starts, value.ends, value.left_ramps, value.right_ramps, strict=True))
        for value in (split(length) for split, length in zip(splitters, latent_shape, strict=True))
    ]
    tiles = []
    for axes in itertools.product(*intervals):
        inputs = [slice(None), slice(None)]
        outputs = [slice(None), slice(None)]
        masks = [torch.ones(1), torch.ones(1)]
        for axis, ((start, stop, left, right), scale) in enumerate(zip(axes, scales, strict=True)):
            pixel_stop = 1 + (stop - 1) * scale if axis == 0 else stop * scale
            inputs.append(slice(start * scale, pixel_stop))
            outputs.append(slice(start, stop))
            masks.append(compute_trapezoidal_mask_1d(stop - start, left, right, axis == 0))
        tiles.append(Tile(tuple(inputs), tuple(outputs), tuple(masks)))
    return tiles


def _encode_masks_are_complementary(tiles: list[Tile], shape: torch.Size) -> bool:
    for axis in range(2, len(shape)):
        weights = torch.zeros(shape[axis], dtype=torch.float32, device="cpu")
        seen = set()
        for tile in tiles:
            region = tile.out_coords[axis]
            key = (region.start, region.stop)
            if key not in seen:
                weights[region] += tile.masks_1d[axis].float().cpu()
                seen.add(key)
        if not torch.allclose(weights, torch.ones_like(weights), atol=1e-5, rtol=0.0):
            return False
    return True


def _apply_encode_masks(latent: torch.Tensor, tile: Tile) -> torch.Tensor:
    result = latent.float()
    for axis, mask in enumerate(tile.masks_1d):
        view = [1] * latent.ndim
        view[axis] = -1
        result = result * mask.to(device=latent.device, dtype=torch.float32).view(view)
    return result


class LTX23VideoEncoder(LTX2VideoEncoder):
    """LTX-2.3 video encoder with overlap-aware tiled encoding."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: list[tuple[str, int | dict[str, Any]]] | None = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = True,
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
            timestep_conditioning=False,
            encoder_spatial_padding_mode=encoder_spatial_padding_mode,
        )
        self.per_channel_statistics = _LTX23PerChannelStatistics(out_channels)

    def tiled_encode(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        if tiling_config is None:
            return self(video)

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
        tiles = _prepare_encode_tiles(video, tiling_config, scales)
        weights = (
            None
            if _encode_masks_are_complementary(tiles, latents.shape)
            else torch.zeros_like(latents)
        )
        for tile in tiles:
            latent_tile = self(video[tile.in_coords].to(device=device, dtype=dtype))
            latents[tile.out_coords] += _apply_encode_masks(latent_tile, tile)
            if weights is not None:
                strength = torch.ones_like(latent_tile, dtype=torch.float32)
                weights[tile.out_coords] += _apply_encode_masks(strength, tile)
        return latents if weights is None else latents / weights.clamp(min=1e-8)


class LTX23VideoEncoderConfigurator:
    """Create an LTX-2.3 video encoder from the native config."""

    @classmethod
    def from_config(cls, config: dict) -> LTX23VideoEncoder:
        vae = config.get("vae", {})
        padding_mode = vae.get(
            "spatial_padding_mode", vae.get("encoder_spatial_padding_mode", "zeros")
        )
        return LTX23VideoEncoder(
            convolution_dimensions=vae.get("dims", 3),
            in_channels=vae.get("in_channels", vae.get("out_channels", 3)),
            out_channels=vae.get("latent_channels", 128),
            encoder_blocks=vae.get("encoder_blocks", []),
            patch_size=vae.get("patch_size", 4),
            norm_layer=NormLayerType(vae.get("norm_layer", "pixel_norm")),
            causal=vae.get("causal_encoder", True),
            encoder_spatial_padding_mode=PaddingModeType(padding_mode),
        )
