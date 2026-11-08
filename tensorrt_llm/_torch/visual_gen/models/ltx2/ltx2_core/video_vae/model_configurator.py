# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from .enums import NormLayerType, PaddingModeType
from .video_vae import VideoDecoder, VideoEncoder


class VideoDecoderConfigurator:
    """Create a VideoDecoder from the LTX-2 native config dict."""

    @classmethod
    def from_config(cls, config: dict) -> VideoDecoder:
        config = config.get("vae", {})
        return VideoDecoder(
            convolution_dimensions=config.get("dims", 3),
            in_channels=config.get("latent_channels", 128),
            out_channels=config.get("out_channels", 3),
            decoder_blocks=config.get("decoder_blocks", []),
            patch_size=config.get("patch_size", 4),
            norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
            causal=config.get("causal_decoder", False),
            timestep_conditioning=config.get("timestep_conditioning", True),
            decoder_spatial_padding_mode=PaddingModeType(
                config.get("decoder_spatial_padding_mode", "reflect")
            ),
        )


class VideoEncoderConfigurator:
    """Create a VideoEncoder from the LTX-2 native config dict."""

    @classmethod
    def from_config(cls, config: dict) -> VideoEncoder:
        config = config.get("vae", {})
        return VideoEncoder(
            convolution_dimensions=config.get("dims", 3),
            in_channels=config.get("out_channels", 3),
            out_channels=config.get("latent_channels", 128),
            encoder_blocks=config.get("encoder_blocks", []),
            patch_size=config.get("patch_size", 4),
            norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
            causal=config.get("causal_encoder", True),
            timestep_conditioning=False,
            encoder_spatial_padding_mode=PaddingModeType(
                config.get("encoder_spatial_padding_mode", "zeros")
            ),
        )
