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

from transformers import Gemma4TextConfig, PreTrainedConfig


class Gemma4AssistantConfig(PreTrainedConfig):
    """Compatibility config for Gemma4 assistant checkpoints.

    Gemma4 assistant support postdates the transformers version currently
    pinned by TensorRT-LLM. Keep the compatibility surface minimal and remove
    this class once the pinned transformers release provides it natively.
    """

    model_type = "gemma4_assistant"
    sub_configs = {"text_config": Gemma4TextConfig}

    # Runtime capabilities consumed by the generic two-model MTP pipeline.
    shares_target_kv_cache = True
    preserve_checkpoint_layer_count = True
    freezes_draft_attention_state = True
    cuda_graph_external_draft_len = 0

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

    @property
    def speculative_hidden_size(self):
        """Hidden-state width captured from the target model."""
        return self.backbone_hidden_size
