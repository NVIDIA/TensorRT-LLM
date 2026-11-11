# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae import (
    model_configurator as upstream,
)


class AudioEncoderConfigurator:
    @classmethod
    def from_config(cls, config):
        return upstream.AudioEncoderConfigurator.from_metadata({"config": config})
