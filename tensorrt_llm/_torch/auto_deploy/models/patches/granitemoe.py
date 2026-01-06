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
"""A patch for GraniteMoe to make it compatible with torch.export.

The main issue is that GraniteMoeTopKGating calls `.tolist()` on a tensor which is:
1. Incompatible with meta tensors (no data to access)
2. Incompatible with torch.compile/export (data-dependent operation)

This patch rewrites the MoE forward to use the torch_moe custom op instead.
"""

import torch
import torch.nn.functional as F
from transformers.models.granitemoe import modeling_granitemoe

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_moe(self, layer_input: torch.Tensor):
    """GraniteMoe forward rewritten for torch.export compatibility.

    This replaces the original forward which uses data-dependent operations
    like `.tolist()` that are incompatible with torch.export and meta tensors.

    Uses the torch_moe custom op for export compatibility.
    """
    bsz, length, emb_size = layer_input.size()
    layer_input_flat = layer_input.reshape(-1, emb_size)

    # Compute router logits directly - avoid the problematic router.forward() which calls .tolist()
    router_logits = self.router.layer(layer_input_flat).float()

    # Get top-k routing (matches original router: topk first, then softmax on top-k only)
    top_k_logits, selected_experts = torch.topk(router_logits, self.router.top_k, dim=-1)
    routing_weights = F.softmax(top_k_logits, dim=1, dtype=torch.float)
    routing_weights = routing_weights.to(layer_input_flat.dtype)

    input_weight = self.input_linear.weight  # [E, 2*I, H]
    intermediate_size = input_weight.shape[1] // 2  # I = intermediate_size

    # GraniteMoe stores weights as [gate, up] = [w1, w3]
    # torch_moe expects [w3, w1] order for stacked format
    gate_proj = input_weight[:, :intermediate_size, :]  # [E, I, H] - this is w1 (gate)
    up_proj = input_weight[:, intermediate_size:, :]  # [E, I, H] - this is w3 (up)
    w3_w1_stacked = torch.cat([up_proj, gate_proj], dim=1)  # [E, 2*I, H] - [w3, w1] order

    w2_stacked = self.output_linear.weight  # [E, H, I]

    # Use torch_moe with stacked tensor format (single-element list)
    layer_output = torch.ops.auto_deploy.torch_moe(
        layer_input_flat,
        selected_experts,
        routing_weights,
        w1_weight=[w3_w1_stacked],  # Stacked format: single tensor [E, 2*I, H]
        w2_weight=[w2_stacked],  # Stacked format: single tensor [E, H, I]
        w3_weight=[],  # Empty for stacked format
        mlp_style="gated_mlp",
        act_fn="silu",
    )

    layer_output = layer_output.view(bsz, length, self.input_size)
    return layer_output, router_logits


@ExportPatchRegistry.register("hf_granitemoe_moe")
class GraniteMoePatch(BaseExportPatch):
    """Patch for GraniteMoe to make it compatible with torch.export.

    GraniteMoe has a critical issue: GraniteMoeTopKGating calls `.tolist()` on a tensor,
    which is:
    1. Incompatible with meta tensors (no data to access - causes "Cannot copy out of meta tensor")
    2. Incompatible with torch.compile/export (data-dependent operation)

    This patch rewrites the MoE forward to use the torch_moe custom op instead,
    which handles routing in an export-compatible way.
    """

    def _apply_patch(self):
        """Apply the GraniteMoe patch."""
        # Store original forward method
        self.original_values["GraniteMoeMoE.forward"] = modeling_granitemoe.GraniteMoeMoE.forward

        # Apply patch by replacing the forward method
        modeling_granitemoe.GraniteMoeMoE.forward = _forward_moe

    def _revert_patch(self):
        """Revert the GraniteMoe patch."""
        # Restore original forward method
        modeling_granitemoe.GraniteMoeMoE.forward = self.original_values["GraniteMoeMoE.forward"]
