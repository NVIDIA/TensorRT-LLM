# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

from tensorrt_llm.mapping import Mapping

from .dist_config import DistConfig


def deserialize_mapping(mapping_config: str) -> Mapping:
    """Deserialize a Mapping from a serialized DistConfig JSON string.

    The canonical serialization format for ``mapping_config`` is
    ``DistConfig.serialize()``.  This helper round-trips through DistConfig
    and converts to Mapping for callers that still need a Mapping object.
    """
    return DistConfig.deserialize(mapping_config).to_mapping()


def serialize_mapping(mapping: Mapping) -> str:
    """Serialize a Mapping to JSON string (legacy interface for MoE ops)."""
    return json.dumps(mapping.to_dict())


def serialize_dist_config(dist_config: DistConfig) -> str:
    """Serialize a DistConfig to JSON string for MoE ops."""
    return dist_config.serialize()


def print_grid(dist_config: DistConfig) -> str:
    return (
        f"process grid: [TP, MoE_TP, MoE_EP] = "
        f"[{dist_config.tp_size}, {dist_config.moe_tp_size}, {dist_config.moe_ep_size}]"
    )


def print_rank(dist_config: DistConfig) -> str:
    return f"rank: [{dist_config.rank}, {dist_config.moe_tp_rank}, {dist_config.moe_ep_rank}]"
