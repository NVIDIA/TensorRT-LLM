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
"""Compatibility configs for Gemma 4 assistant and Unified checkpoints."""

from transformers import Gemma4TextConfig, PreTrainedConfig


class Gemma4AssistantConfig(PreTrainedConfig):
    """Compatibility config for Gemma4 assistant checkpoints.

    Gemma4 assistant support postdates the transformers version currently
    pinned by TensorRT-LLM. Keep the compatibility surface minimal and remove
    this class once the pinned transformers release provides it natively.
    """

    model_type = "gemma4_assistant"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(
        self,
        text_config=None,
        backbone_hidden_size=1536,
        use_ordered_embeddings=False,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma4TextConfig(
                num_hidden_layers=4,
                num_kv_shared_layers=4,
                hidden_size_per_layer_input=0,
                vocab_size_per_layer_input=0,
                enable_moe_block=False,
                use_double_wide_mlp=False,
            )
        elif isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)

        # Assistant layers are Q-only and all read the target model's KV cache.
        # Match the native Transformers config behavior when the field is
        # omitted, and reject partially shared variants that this architecture
        # cannot execute correctly.
        if not text_config.num_kv_shared_layers:
            text_config.num_kv_shared_layers = text_config.num_hidden_layers
        if text_config.num_kv_shared_layers != text_config.num_hidden_layers:
            raise ValueError(
                "All Gemma4 assistant layers must share the target KV cache: "
                f"expected {text_config.num_hidden_layers}, got "
                f"{text_config.num_kv_shared_layers}"
            )
        if text_config.hidden_size_per_layer_input != 0:
            raise ValueError(
                "Gemma4 assistant hidden_size_per_layer_input must be 0, "
                f"got {text_config.hidden_size_per_layer_input}"
            )
        if text_config.vocab_size_per_layer_input != 0:
            raise ValueError(
                "Gemma4 assistant vocab_size_per_layer_input must be 0, "
                f"got {text_config.vocab_size_per_layer_input}"
            )
        if text_config.enable_moe_block:
            raise ValueError("Gemma4 assistant does not support MoE blocks")
        if text_config.use_double_wide_mlp:
            raise ValueError("Gemma4 assistant does not support double-wide MLPs")

        self.text_config = text_config
        self.backbone_hidden_size = backbone_hidden_size
        self.use_ordered_embeddings = use_ordered_embeddings
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        super().__init__(**kwargs)

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers


class Gemma4UnifiedTextConfig(Gemma4TextConfig):
    """Text sub-config for Gemma 4 12B Unified.

    The 12B text backbone is a standard dense Gemma 4 text model; only the
    model_type string differs, so this is a pure alias of the native
    `Gemma4TextConfig`.
    """

    model_type = "gemma4_unified_text"


class Gemma4UnifiedVisionConfig(PreTrainedConfig):
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


class Gemma4UnifiedAudioConfig(PreTrainedConfig):
    """Sub-config for the encoder-free audio projector.

    `output_proj_dims` and `hidden_size` alias `audio_embed_dim` (the raw audio
    frame width) when not given, matching the HF implementation; they are plain
    attributes here so a checkpoint config.json that spells them out loads as-is.
    """

    model_type = "gemma4_unified_audio"

    def __init__(
        self,
        audio_embed_dim: int = 640,
        rms_norm_eps: float = 1e-6,
        output_proj_dims: int | None = None,
        hidden_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_embed_dim = audio_embed_dim
        self.rms_norm_eps = rms_norm_eps
        self.output_proj_dims = (
            output_proj_dims if output_proj_dims is not None else audio_embed_dim
        )
        self.hidden_size = hidden_size if hidden_size is not None else audio_embed_dim


class Gemma4UnifiedConfig(PreTrainedConfig):
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
