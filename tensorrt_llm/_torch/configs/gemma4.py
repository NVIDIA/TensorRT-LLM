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

    def __init__(
        self,
        text_config=None,
        backbone_hidden_size=None,
        use_ordered_embeddings=False,
        num_centroids=0,
        centroid_intermediate_top_k=0,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)

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
