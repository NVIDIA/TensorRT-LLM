# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.model_configurator import (
    VocoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.model_configurator import (
    _vocoder_from_config as _vocoder_from_config,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.vocoder import (
    Vocoder as Vocoder,
)


class LTX23VocoderConfigurator:
    @classmethod
    def from_config(cls, config):
        return VocoderConfigurator.from_metadata({"config": config})
