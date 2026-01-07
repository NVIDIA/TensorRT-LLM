# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GLM4-MoE model patches for auto-deploy compatibility.

This module patches the GLM4-MoE model to make it compatible with torch.fx export
by replacing data-dependent operations (torch.where/nonzero) with traceable custom ops.
"""

import types
from typing import Dict

import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def glm4_moe_forward(self, hidden_states):
    """Glm4MoeMoE forward function rewritten to enable torch export.

    Replaces self.moe() call (which uses torch.where) with torch_moe custom op.
    """
    residuals = hidden_states
    orig_shape = hidden_states.shape

    # Gate directly returns (topk_indices, topk_weights)
    topk_indices, topk_weights = self.gate(hidden_states)

    # Flatten for MoE processing
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # Replace self.moe() with torch_moe custom op
    # self.experts is a ModuleList of Glm4MoeMLP, each with gate_proj, up_proj, down_proj
    # Collect weights from each expert
    w1_weight = [expert.gate_proj.weight for expert in self.experts]  # gate_proj
    w2_weight = [expert.down_proj.weight for expert in self.experts]  # down_proj
    w3_weight = [expert.up_proj.weight for expert in self.experts]  # up_proj

    hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
    )

    hidden_states = hidden_states.view(*orig_shape)

    # Add shared experts output
    hidden_states = hidden_states + self.shared_experts(residuals)

    return hidden_states


# Store original from_config
_from_config_original = AutoModelForCausalLM.from_config

# Module patches mapping
CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "Glm4MoeMoE": glm4_moe_forward,
}


def get_model_from_config_patched(config, **kwargs):
    """Patched from_config that applies GLM4-MoE module patches."""
    model = _from_config_original(config, **kwargs)

    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# Apply the patch
AutoModelForCausalLM.from_config = get_model_from_config_patched
