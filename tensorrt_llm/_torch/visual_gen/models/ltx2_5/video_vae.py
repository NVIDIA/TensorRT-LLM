# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""LTX-2.5 video VAE components.

LTX-2.5 reuses LTX-2's VAE primitives and forward/tiled-decode machinery, but
its channel recipe differs: compress_time and compress_space reduce channels by
their multiplier, where LTX-2 only did that for compress_all. conv_in is
therefore latent_channels times the product of every multiplier, not just the
compress_all ones -- 128 * (2*1*2*2) = 1024 for the checkpoint recipe.

The decoder subclasses LTX-2's implementation and rebuilds only the
channel-bearing modules. The encoder reuses LTX-2's architecture with the
LTX-2.5 checkpoint statistics.
"""

from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

from ..ltx2.ltx2_core.normalization import PixelNorm
from ..ltx2.ltx2_core.video_vae.convolution import make_conv_nd
from ..ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ..ltx2.ltx2_core.video_vae.ops import PerChannelStatistics
from ..ltx2.ltx2_core.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from ..ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from ..ltx2.ltx2_core.video_vae.video_vae import VideoDecoder as LTX2VideoDecoder
from ..ltx2.ltx2_core.video_vae.video_vae import VideoEncoder as LTX2VideoEncoder

_COMPRESS_STRIDES = {
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
    "compress_all": (2, 2, 2),
}


def _channel_multiplier(block_name: str, block_config: dict) -> int:
    """Factor by which this block reduces channels, used to pre-size conv_in."""
    if block_name == "res_x_y":
        return block_config.get("multiplier", 2)
    if block_name in _COMPRESS_STRIDES:
        return block_config.get("multiplier", 1)
    return 1


def _make_ltx2_5_decoder_block(
    block_name: str,
    block_config: dict,
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Like LTX-2's decoder factory, but compress_time/space reduce channels."""
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
            timestep_conditioning=False,
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
            timestep_conditioning=False,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        multiplier = block_config.get("multiplier", 2)
        out_channels = in_channels // multiplier
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
    elif block_name in _COMPRESS_STRIDES:
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
    else:
        raise ValueError(f"unknown decoder block: {block_name}")
    return block, out_channels


class LTX25VideoDecoder(LTX2VideoDecoder):
    """LTX-2.5 video decoder: LTX-2 primitives, LTX-2.5 channel recipe."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, Union[int, dict]]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=spatial_padding_mode,
        )

        patched_out_channels = out_channels * patch_size**2

        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            cfg = block_params if isinstance(block_params, dict) else {}
            feature_channels *= _channel_multiplier(block_name, cfg)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        up_blocks = nn.ModuleList([])
        fc = feature_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            cfg = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            block, fc = _make_ltx2_5_decoder_block(
                block_name=block_name,
                block_config=cfg,
                in_channels=fc,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=spatial_padding_mode,
            )
            up_blocks.append(block)
        self.up_blocks = up_blocks

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=fc, num_groups=self._norm_num_groups, eps=1e-6
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=fc,
            out_channels=patched_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )


class LTX25VideoDecoderConfigurator:
    """Create an LTX25VideoDecoder from the native LTX-2.5 config dict."""

    @classmethod
    def from_config(cls, config: dict) -> LTX25VideoDecoder:
        vae = config.get("vae", {})
        padding_mode = vae.get(
            "spatial_padding_mode", vae.get("decoder_spatial_padding_mode", "reflect")
        )
        return LTX25VideoDecoder(
            convolution_dimensions=vae.get("dims", 3),
            in_channels=vae.get("latent_channels", 128),
            out_channels=vae.get("out_channels", 3),
            decoder_blocks=vae.get("decoder_blocks", []),
            patch_size=vae.get("patch_size", 4),
            norm_layer=NormLayerType(vae.get("norm_layer", "pixel_norm")),
            causal=vae.get("causal_decoder", False),
            spatial_padding_mode=PaddingModeType(padding_mode),
        )


class _LTX25PerChannelStatistics(PerChannelStatistics):
    """Statistics buffers present in the LTX-2.5 encoder checkpoint."""

    def __init__(self, latent_channels: int = 128) -> None:
        nn.Module.__init__(self)
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))


class LTX25VideoEncoder(LTX2VideoEncoder):
    """LTX-2.5 video encoder with checkpoint-specific statistics."""

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
        self.per_channel_statistics = _LTX25PerChannelStatistics(out_channels)


class LTX25VideoEncoderConfigurator:
    """Create an LTX-2.5 video encoder from the native config."""

    @classmethod
    def from_config(cls, config: dict) -> LTX25VideoEncoder:
        vae = config.get("vae", {})
        padding_mode = vae.get(
            "spatial_padding_mode", vae.get("encoder_spatial_padding_mode", "zeros")
        )
        return LTX25VideoEncoder(
            convolution_dimensions=vae.get("dims", 3),
            in_channels=vae.get("in_channels", vae.get("out_channels", 3)),
            out_channels=vae.get("latent_channels", 128),
            encoder_blocks=vae.get("encoder_blocks", []),
            patch_size=vae.get("patch_size", 4),
            norm_layer=NormLayerType(vae.get("norm_layer", "pixel_norm")),
            causal=vae.get("causal_encoder", True),
            encoder_spatial_padding_mode=PaddingModeType(padding_mode),
        )
