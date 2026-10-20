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
"""Virtual layers: managers that allocate several internal layers per model layer.

DeepSeek-V4 and CSA2 key their internal layers by ``(model layer, attention type)`` in
``_layer_attn_to_layer_id``. Internal ids depend on the PP split, so peers match these layers by
``model_layer * num_attention_types + attention_type`` instead.
"""

from typing import List, Optional

from ._v2 import LifeCycle


def global_layer_ids(manager, cycle: LifeCycle) -> List[int]:
    mapping = manager._layer_attn_to_layer_id
    if not mapping:
        raise ValueError(f"{type(manager).__name__} declares no virtual layers")
    # `setdefault`, not assignment: the mapping may be many-to-one (DeepSeek-V4.1 long-range layers
    # sharing a compressed cache resolve to the allocating layer's id). The owner precedes its
    # dependents, so keeping the first entry names the owner on every PP rank.
    inverse = {}
    for (model_layer, attn_type), layer_id in mapping.items():
        inverse.setdefault(layer_id, (model_layer, attn_type.value))
    # The full enum range keeps the encoding identical on PP ranks that hold different subsets of
    # attention types.
    first_type = next(iter(mapping))[1]
    num_attn_types = max(e.value for e in type(first_type)) + 1
    return [inverse[lid][0] * num_attn_types + inverse[lid][1] for lid in cycle.internal_layer_ids]


def window_size(manager, internal_layer_id: int) -> Optional[int]:
    """The window of an internal layer, through the model layer it belongs to."""
    config = manager.impl.init_config
    if internal_layer_id < len(config.layers):
        return getattr(config.layers[internal_layer_id], "window_size", None)
    for (model_layer, _attn_type), layer_id in manager._layer_attn_to_layer_id.items():
        if layer_id != internal_layer_id:
            continue
        local_layer = manager.layer_offsets.get(model_layer)
        if local_layer is not None and local_layer < len(config.layers):
            return getattr(config.layers[local_layer], "window_size", None)
        if model_layer < len(config.layers):
            return getattr(config.layers[model_layer], "window_size", None)
    raise ValueError(f"Cannot resolve layer config for internal layer {internal_layer_id}")
