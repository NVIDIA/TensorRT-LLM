# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae import (
    model_configurator as upstream,
)


class AudioDecoderConfigurator:
    @classmethod
    def from_config(cls, config):
        return upstream.AudioDecoderConfigurator.from_metadata({"config": config})


class VocoderConfigurator:
    @classmethod
    def from_config(cls, config):
        return upstream.VocoderConfigurator.from_metadata({"config": config})
