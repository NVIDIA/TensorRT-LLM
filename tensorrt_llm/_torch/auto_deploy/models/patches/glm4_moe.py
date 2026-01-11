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

"""GLM4-MoE model patches for auto-deploy compatibility."""

import types  # noqa: F401
from typing import Dict

import torch

from ...export.interface import BaseExportPatch, ExportPatchRegistry


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


# Module patches mapping
CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "Glm4MoeMoE": glm4_moe_forward,
}


@ExportPatchRegistry.register("hf_glm4_moe")
class Glm4MoePatch(BaseExportPatch):
    """Patch for HF GLM4-MoE to make it compatible with torch.export.

    This patch temporarily replaces `transformers`' `Glm4MoeMoE.forward` with
    `glm4_moe_forward` during export, then restores the original afterwards.
    """

    def _apply_patch(self):
        """Apply the GLM4-MoE export patch."""
        try:
            from transformers.models.glm4_moe import modeling_glm4_moe
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import GLM4-MoE from transformers. "
                "Ensure a transformers version with `transformers.models.glm4_moe` is installed."
            ) from e

        self.original_values["Glm4MoeMoE.forward"] = modeling_glm4_moe.Glm4MoeMoE.forward
        modeling_glm4_moe.Glm4MoeMoE.forward = glm4_moe_forward

    def _revert_patch(self):
        """Revert the GLM4-MoE export patch."""
        # Only revert if we successfully applied.
        if "Glm4MoeMoE.forward" not in self.original_values:
            return

        from transformers.models.glm4_moe import modeling_glm4_moe

        modeling_glm4_moe.Glm4MoeMoE.forward = self.original_values["Glm4MoeMoE.forward"]
