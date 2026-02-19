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


def deserialize_mapping(mapping_config: str) -> Mapping:
    return Mapping.from_dict(json.loads(mapping_config))


def serialize_mapping(mapping: Mapping) -> str:
    return json.dumps(mapping.to_dict())


def print_grid(mapping: Mapping) -> str:
    return f"process grid: [TP, MoE_TP, MoE_EP] = [{mapping.tp_size}, {mapping.moe_tp_size}, {mapping.moe_ep_size}]"


def print_rank(mapping: Mapping) -> str:
    return f"rank: [{mapping.rank}, {mapping.moe_tp_rank}, {mapping.moe_ep_rank}]"
