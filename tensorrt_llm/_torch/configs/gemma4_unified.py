# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config classes for Gemma 4 12B Unified (encoder-free multimodal).

Registered with the transformers CONFIG_MAPPING (see `_torch/configs/__init__.py`)
so `AutoConfig.from_pretrained` can parse a Gemma 4 12B checkpoint whenever the
installed transformers does not ship the `gemma4_unified` model_types natively.

All fields are read directly from the checkpoint's config.json, matching the
attribute names used by `Gemma4UnifiedForConditionalGeneration`. The text
backbone of the 12B is a standard dense Gemma 4 text model, so its sub-config
reuses the native `Gemma4TextConfig`.
"""

from transformers import Gemma4TextConfig
from transformers.configuration_utils import PretrainedConfig


class Gemma4UnifiedTextConfig(Gemma4TextConfig):
    """Text sub-config for Gemma 4 12B Unified.

    The 12B text backbone is a standard dense Gemma 4 text model; only the
    model_type string differs, so this is a pure alias of the native
    `Gemma4TextConfig`.
    """

    model_type = "gemma4_unified_text"


class Gemma4UnifiedVisionConfig(PretrainedConfig):
    """Sub-config for the encoder-free vision projector."""

    model_type = "gemma4_unified_vision"

    def __init__(
        self,
        mm_embed_dim: int = 3840,
        mm_posemb_size: int = 1120,
        output_proj_dims: int = 3840,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mm_embed_dim = mm_embed_dim
        self.mm_posemb_size = mm_posemb_size
        self.output_proj_dims = output_proj_dims
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.rms_norm_eps = rms_norm_eps


class Gemma4UnifiedAudioConfig(PretrainedConfig):
    """Sub-config for the encoder-free audio projector.

    `output_proj_dims` is a property returning `audio_embed_dim` (the raw
    audio frame width), matching the HF implementation where both fields alias
    the same value.
    """

    model_type = "gemma4_unified_audio"

    def __init__(
        self,
        audio_embed_dim: int = 640,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_embed_dim = audio_embed_dim
        self.rms_norm_eps = rms_norm_eps

    @property
    def output_proj_dims(self) -> int:
        return self.audio_embed_dim

    @property
    def hidden_size(self) -> int:
        return self.audio_embed_dim


class Gemma4UnifiedConfig(PretrainedConfig):
    """Top-level config for Gemma 4 12B Unified (encoder-free multimodal).

    Parses `config.json` fields required by
    `Gemma4UnifiedForConditionalGeneration` without depending on any
    natively shipped transformers class. The `text_config`, `vision_config`, and
    `audio_config` sub-configs are reconstructed from nested dicts using the
    shim classes above.
    """

    model_type = "gemma4_unified"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id: int = 258880,
        audio_token_id: int = 258881,
        video_token_id: int = 258884,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id

        # Sub-configs arrive as dicts from AutoConfig.from_pretrained; rebuild
        # them with the classes above.
        if text_config is not None:
            if isinstance(text_config, dict):
                self.text_config = Gemma4UnifiedTextConfig(**text_config)
            else:
                self.text_config = text_config
        else:
            self.text_config = None

        if vision_config is not None:
            if isinstance(vision_config, dict):
                self.vision_config = Gemma4UnifiedVisionConfig(**vision_config)
            else:
                self.vision_config = vision_config
        else:
            self.vision_config = None

        if audio_config is not None:
            if isinstance(audio_config, dict):
                self.audio_config = Gemma4UnifiedAudioConfig(**audio_config)
            else:
                self.audio_config = audio_config
        else:
            self.audio_config = None
