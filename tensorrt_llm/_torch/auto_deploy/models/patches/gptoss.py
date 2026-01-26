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
"""A patch for GPT-OSS MoE to make it compatible with torch.export.

GPT-OSS uses a dense MoE pattern where all experts are computed for all tokens,
with soft routing weights. This patch replaces the BMM-based forward with torch_moe
using SwigluBias activation.
"""

import torch

from tensorrt_llm._torch.utils import ActivationType

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_gptoss_mlp(self, hidden_states: torch.Tensor):
    """GPT-OSS MoE forward rewritten for torch.export compatibility.

    Uses torch_moe with SwigluBias activation to match the original GPT-OSS
    dense MoE computation: (up + 1) * (gate * sigmoid(alpha * gate)) with clamping.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_size)
    num_tokens = hidden_states_flat.shape[0]

    router_scores, _ = self.router(hidden_states)

    experts = self.experts
    num_experts = experts.num_experts

    selected_experts = (
        torch.arange(num_experts, device=hidden_states.device).unsqueeze(0).expand(num_tokens, -1)
    )

    # gate_up_proj: [E, H, 2*I] with interleaved gate/up
    # Split into gate [E, H, I] and up [E, H, I], then transpose to [I, H] for F.linear
    gate_proj = experts.gate_up_proj[..., ::2]  # [E, H, I] - gate
    up_proj = experts.gate_up_proj[..., 1::2]  # [E, H, I] - up

    # Create per-expert weight lists with shape [I, H] for F.linear
    w1_weight = [gate_proj[i].T for i in range(num_experts)]  # gate: [I, H]
    w3_weight = [up_proj[i].T for i in range(num_experts)]  # up: [I, H]
    # down_proj: [E, I, H] -> transpose to [H, I] for F.linear
    w2_weight = [experts.down_proj[i].T for i in range(num_experts)]  # down: [H, I]

    # Biases: gate_up_proj_bias [E, 2*I] -> split into gate [E, I] and up [E, I]
    w1_bias_stacked = experts.gate_up_proj_bias[..., ::2]  # [E, I] - gate bias
    w3_bias_stacked = experts.gate_up_proj_bias[..., 1::2]  # [E, I] - up bias
    w2_bias_stacked = experts.down_proj_bias  # [E, H] - down bias

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states_flat,
        selected_experts,
        router_scores,  # [num_tokens, num_experts] - soft routing weights
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        is_gated_mlp=True,
        act_fn=int(ActivationType.SwigluBias),
        w1_bias_stacked=w1_bias_stacked,
        w2_bias_stacked=w2_bias_stacked,
        w3_bias_stacked=w3_bias_stacked,
    )

    final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
    return final_hidden_states, router_scores


@ExportPatchRegistry.register("hf_gptoss_moe")
class GptOssMoePatch(BaseExportPatch):
    """Patch for GPT-OSS MoE to make it compatible with torch.export.

    GPT-OSS uses a dense MoE pattern with:
    - Soft routing over ALL experts (not top-k sparse)
    - SwigluBias activation: (up + 1) * (gate * sigmoid(alpha * gate)) with clamping
    - Biases on all projections

    The original BMM-based forward is replaced with torch_moe custom op
    which handles the computation in an export-compatible way.
    """

    def _apply_patch(self):
        """Apply the GPT-OSS MoE patch."""
        try:
            from transformers.models.gpt_oss import modeling_gpt_oss

            self.modeling_module = modeling_gpt_oss
            self.original_values["GptOssMLP.forward"] = modeling_gpt_oss.GptOssMLP.forward
            modeling_gpt_oss.GptOssMLP.forward = _forward_gptoss_mlp
        except (ImportError, AttributeError):
            pass

    def _revert_patch(self):
        """Revert the GPT-OSS MoE patch."""
        if hasattr(self, "modeling_module") and "GptOssMLP.forward" in self.original_values:
            self.modeling_module.GptOssMLP.forward = self.original_values["GptOssMLP.forward"]
