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
"""A patch for Mixtral MoE to make it compatible with torch.export."""

import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry

# Import SiLUActivation for compatibility check
try:
    from transformers.activations import SiLUActivation

    _SILU_TYPES = (nn.SiLU, SiLUActivation)
except ImportError:
    _SILU_TYPES = (nn.SiLU,)


def _is_silu_activation(act_fn) -> bool:
    """Check if activation function is SiLU or equivalent."""
    return isinstance(act_fn, _SILU_TYPES)


def _forward_moe(self: MixtralSparseMoeBlock, hidden_states: torch.Tensor):
    # check if we can apply the patch
    if any(getattr(mod, "bias", None) is not None for mod in self.experts.modules()):
        raise NotImplementedError(
            "MixtralSparseMoeBlock forward patch does not support this model configuration: "
            "expert modules have bias. "
            "The original transformers forward uses torch.nonzero() and tensor indexing "
            "which are not compatible with torch.export on meta tensors."
        )

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)

    # In transformers 5.x, gate returns (router_logits, routing_weights, selected_experts).
    # The routing logic (softmax, topk, normalization) is now inside the gate.
    _, routing_weights, selected_experts = self.gate(hidden_states)

    # In transformers 5.x, self.experts is a fused object with stacked weight tensors
    # (Parameters directly, not modules with .weight).
    # Use torch_moe_fused directly since weights are already stacked.
    gate_up_param = self.experts.gate_up_proj
    gate_up = gate_up_param.weight if hasattr(gate_up_param, "weight") else gate_up_param
    down_param = self.experts.down_proj
    down = down_param.weight if hasattr(down_param, "weight") else down_param

    # HF format: gate_up is [E, 2*I, H] with gate(w1) first, up(w3) second.
    # TRT-LLM format: w3_w1 is [E, 2*I, H] with up(w3) first, gate(w1) second.
    half = gate_up.shape[1] // 2
    w3_w1_stacked = torch.cat([gate_up[:, half:, :], gate_up[:, :half, :]], dim=1)

    final_hidden_states = torch.ops.auto_deploy.torch_moe_fused(
        hidden_states,
        selected_experts,
        routing_weights,
        w3_w1_stacked_weight=w3_w1_stacked,
        w2_stacked_weight=down,
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states


@ExportPatchRegistry.register("hf_mixtral_moe")
class MixtralMoePatch(BaseExportPatch):
    """Patch for Mixtral MoE to make it compatible with torch.export.

    This patch replaces the forward method of MixtralSparseMoeBlock with
    a version that uses the torch_moe custom operator for better export compatibility.
    """

    def _apply_patch(self):
        """Apply the Mixtral MoE patch."""
        # Store original forward method
        self.original_values["MixtralSparseMoeBlock.forward"] = MixtralSparseMoeBlock.forward

        # Apply patch by replacing the forward method
        MixtralSparseMoeBlock._original_forward = MixtralSparseMoeBlock.forward  # type: ignore
        MixtralSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Mixtral MoE patch."""
        # Restore original forward method
        MixtralSparseMoeBlock.forward = self.original_values["MixtralSparseMoeBlock.forward"]  # type: ignore

        # Clean up the temporary attribute
        if hasattr(MixtralSparseMoeBlock, "_original_forward"):
            delattr(MixtralSparseMoeBlock, "_original_forward")
