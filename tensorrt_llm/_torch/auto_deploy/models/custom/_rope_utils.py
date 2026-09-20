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

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def get_rope_parameters(config, layer_type: Optional[str] = None) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        if (
            layer_type is not None
            and layer_type in rope_parameters
            and isinstance(rope_parameters[layer_type], dict)
        ):
            return rope_parameters[layer_type]
        if "rope_type" in rope_parameters or "type" in rope_parameters:
            return rope_parameters

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        return rope_scaling

    return {}


def get_rope_type(config, layer_type: Optional[str] = None) -> str:
    rope_parameters = get_rope_parameters(config, layer_type)
    return rope_parameters.get("rope_type", rope_parameters.get("type", "default"))


def get_rope_theta(config, layer_type: Optional[str] = None, default: float = 10000.0) -> float:
    rope_parameters = get_rope_parameters(config, layer_type)
    theta = rope_parameters.get("rope_theta")
    if theta is not None:
        return float(theta)

    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)

    return default


def init_rope_inv_freq(config, head_dim: Optional[int] = None, layer_type: Optional[str] = None):
    rope_type = get_rope_type(config, layer_type)
    if rope_type != "default":
        return ROPE_INIT_FUNCTIONS[rope_type](config, device=None)

    if head_dim is None:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    head_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    base = get_rope_theta(config, layer_type)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return inv_freq, 1.0
