# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from ...ltx2.ltx2_core.video_vae.model_configurator import (
    VideoDecoderConfigurator as BaseDecoderConfigurator,
)
from ...ltx2.ltx2_core.video_vae.model_configurator import (
    VideoEncoderConfigurator as BaseEncoderConfigurator,
)


class LTX23VideoDecoderConfigurator:
    @classmethod
    def from_config(cls, config):
        vae = dict(config.get("vae", {}))
        vae.setdefault("timestep_conditioning", False)
        vae.setdefault("decoder_spatial_padding_mode", vae.get("spatial_padding_mode", "reflect"))
        return BaseDecoderConfigurator.from_config({"vae": vae})


class LTX23VideoEncoderConfigurator:
    @classmethod
    def from_config(cls, config):
        vae = dict(config.get("vae", {}))
        vae.setdefault("encoder_spatial_padding_mode", vae.get("spatial_padding_mode", "zeros"))
        return BaseEncoderConfigurator.from_config({"vae": vae})
