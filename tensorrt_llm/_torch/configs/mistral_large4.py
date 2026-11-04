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

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

# index_sharing names each layer's indexer role; the DSA top-k pattern spells
# the same roles with one letter per layer.
_INDEX_SHARING_LETTERS = {"full": "F", "shared": "S"}


# A Mistral-Large-4 language model's config, adapted from the checkpoint's
# Mistral-native params.json (models/checkpoints/mistral/config_loader.py) under
# model_type "mistral_large_4", which transformers does not know. Every key is
# kept as a plain attribute, exactly as adapted. DeepseekV3Config is not used
# because it fills in defaults the checkpoint does not state (an MTP layer
# count, special-token ids).
class MistralLarge4Config(PretrainedConfig):
    model_type = "mistral_large_4"
    keys_to_ignore_at_inference = ["past_key_values"]

    # The indexer rotates GPT-J interleaved pairs, as MLA does. The
    # architecture has no other layout, so the checkpoint does not state it.
    indexer_rope_interleave = True

    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        if vision_config is not None:
            self.vision_config = (
                PretrainedConfig.from_dict(vision_config)
                if isinstance(vision_config, dict)
                else vision_config
            )

    @property
    def index_topk_pattern(self) -> Optional[str]:
        """``index_sharing`` as the DSA top-k pattern, one letter per layer.

        "F" marks a layer that runs its own indexer and owns an indexer-K
        cache; "S" marks one that reuses the top-k of the most recent "F"
        layer. ``DeepSeekSparseAttentionConfig`` reads this pattern.
        """
        index_sharing = getattr(self, "index_sharing", None)
        if index_sharing is None:
            return None
        unknown = sorted(set(index_sharing) - _INDEX_SHARING_LETTERS.keys())
        if unknown:
            raise ValueError(f"index_sharing entries must be 'full' or 'shared', got {unknown}")
        return "".join(_INDEX_SHARING_LETTERS[role] for role in index_sharing)
