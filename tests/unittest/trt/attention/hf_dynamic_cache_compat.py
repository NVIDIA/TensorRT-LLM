# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DynamicCache legacy tuple format for tests (removed from Transformers v5+)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from transformers.cache_utils import DynamicCache

LegacyLayerKV = Tuple[torch.Tensor, torch.Tensor]
LegacyCache = Tuple[LegacyLayerKV, ...]


def dynamic_cache_from_legacy(
    past_key_values: Optional[Union[LegacyCache, Sequence[LegacyLayerKV]]],
) -> DynamicCache:
    """Build a ``DynamicCache`` from the pre-v5 tuple-of-tuples format."""
    cache = DynamicCache()
    if past_key_values is None:
        return cache
    for layer_idx in range(len(past_key_values)):
        key_states, value_states = past_key_values[layer_idx]
        cache.update(key_states, value_states, layer_idx)
    return cache


def dynamic_cache_to_legacy(cache: DynamicCache) -> LegacyCache:
    """Export a ``DynamicCache`` back to the pre-v5 tuple-of-tuples format."""
    layers: List[LegacyLayerKV] = []
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        keys = layer.keys
        values = layer.values
        if keys is None or values is None:
            continue
        layers.append((keys, values))
    return tuple(layers)
