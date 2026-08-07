# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 ("V2") video VAE decoder.

LTX-2.3 reuses LTX-2's VAE *primitives* (ResnetBlock3D / UNetMidBlock3D /
DepthToSpaceUpsample / convs) and forward/tiled-decode machinery, but its
decoder channel recipe differs in one structural way that LTX-2's ``VideoDecoder``
cannot express:

* In LTX-2, only ``res_x_y`` and ``compress_all`` change channels; the spatial
  upsamplers ``compress_time`` / ``compress_space`` keep channels unchanged.
* In LTX-2.3, **``compress_time`` and ``compress_space`` also reduce channels by
  their ``multiplier``** (exactly like ``compress_all``), i.e. their internal
  conv emits ``in * prod(stride) // multiplier`` channels and the depth-to-space
  rearrange yields ``in // multiplier``.

Consequently the ``conv_in`` feature width is ``latent_channels`` times the
product of **every** compress/res_x_y multiplier (reversed order), not just the
``compress_all`` ones. For the LTX-2.3 checkpoint recipe
(compress_all=2, compress_all=1, compress_time=2, compress_space=2) that is
``128 * (2*1*2*2) = 1024`` -- matching ``conv_in.conv.weight [1024, 128, 3,3,3]``.

We subclass LTX-2's ``VideoDecoder`` so all of its ``forward`` / ``tiled_decode``
/ per-channel-statistics / tiling logic is inherited unchanged, and rebuild only
the channel-bearing modules (``conv_in``, ``up_blocks``, ``conv_norm_out``,
``conv_out``) with the corrected channel flow. LTX-2.3 also uses a single shared
``spatial_padding_mode`` config key (vs LTX-2's per-role decoder/encoder keys).
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ...ltx2.ltx2_core.normalization import PixelNorm
from ...ltx2.ltx2_core.video_vae.convolution import make_conv_nd
from ...ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ...ltx2.ltx2_core.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from ...ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from ...ltx2.ltx2_core.video_vae.video_vae import VideoDecoder

# Spatial/temporal strides per compress block (LTX-2.3 == LTX-2 spatially; the
# only difference is the channel reduction below).
_COMPRESS_STRIDES = {
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
    "compress_all": (2, 2, 2),
}


def _channel_multiplier(block_name: str, block_config: dict) -> int:
    """Factor by which this block reduces channels in the forward pass.

    Used to pre-size ``conv_in`` (the reversed product of these factors). Blocks
    that do not change channels (``res_x`` / ``attn_res_x``) return 1.
    """
    if block_name == "res_x_y":
        return block_config.get("multiplier", 2)
    if block_name in _COMPRESS_STRIDES:
        return block_config.get("multiplier", 1)
    return 1


def _make_ltx23_decoder_block(
    block_name: str,
    block_config: dict,
    in_channels: int,
    dims: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Like LTX-2's ``_make_decoder_block`` but compress_time/space reduce channels."""
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=dims,
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
            dims=dims,
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
        multiplier = block_config.get("multiplier", 2)
        out_channels = in_channels // multiplier
        block = ResnetBlock3D(
            dims=dims,
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
        # LTX-2.3: compress_* reduces channels by `multiplier` (LTX-2 only did
        # this for compress_all). out = in // multiplier; the internal conv emits
        # in * prod(stride) // multiplier.
        multiplier = block_config.get("multiplier", 1)
        out_channels = in_channels // multiplier
        block = DepthToSpaceUpsample(
            dims=dims,
            in_channels=in_channels,
            stride=_COMPRESS_STRIDES[block_name],
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=multiplier,
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown decoder block: {block_name}")
    return block, out_channels


class LTX23VideoDecoder(VideoDecoder):
    """LTX-2.3 video decoder: LTX-2 primitives, LTX-2.3 channel recipe."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, Union[int, dict]]] = [],
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        # Build the LTX-2 decoder first so every non-block attribute
        # (per_channel_statistics, downscale factors, decode params) and all
        # inherited forward/tiled_decode methods are set up. Its conv_in/up_blocks
        # use LTX-2 channel math (wrong for LTX-2.3); we overwrite them below.
        super().__init__(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=spatial_padding_mode,
        )

        dims = convolution_dimensions
        patched_out_channels = out_channels * patch_size**2

        # conv_in width = latent_channels * product of all channel multipliers
        # (reversed order), so the whole decoder narrows back down to `in_channels`.
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            cfg = block_params if isinstance(block_params, dict) else {}
            feature_channels *= _channel_multiplier(block_name, cfg)

        self.conv_in = make_conv_nd(
            dims=dims,
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
            block, fc = _make_ltx23_decoder_block(
                block_name=block_name,
                block_config=cfg,
                in_channels=fc,
                dims=dims,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
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
            dims=dims,
            in_channels=fc,
            out_channels=patched_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        # LTX-2.3 checkpoint has timestep_conditioning=False; keep the branch
        # correct anyway (rebuild sized to the corrected final channels).
        if timestep_conditioning:
            from ...ltx2.ltx2_core.timestep_embedding import (
                PixArtAlphaCombinedTimestepSizeEmbeddings,
            )

            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=fc * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, fc))


class LTX23VideoDecoderConfigurator:
    """Create an ``LTX23VideoDecoder`` from the native LTX-2.3 config dict."""

    @classmethod
    def from_config(cls, config: dict) -> LTX23VideoDecoder:
        vae = config.get("vae", {})
        # LTX-2.3 uses a single shared `spatial_padding_mode` key (LTX-2 split it
        # into decoder_/encoder_ variants).
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
            timestep_conditioning=vae.get("timestep_conditioning", False),
            spatial_padding_mode=PaddingModeType(padding_mode),
        )
