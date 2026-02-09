"""A patch for Qwen3Next MoE to make it compatible with torch.export and reduce export time."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_moe(self: Qwen3NextSparseMoeBlock, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Routed experts via torch_moe
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )

    # Shared expert path (unique to Qwen3Next vs Qwen3MoE)
    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@ExportPatchRegistry.register("hf_qwen3_next_moe")
class Qwen3NextMoePatch(BaseExportPatch):
    """Patch for Qwen3Next MoE for torch.export compatibility."""

    def _apply_patch(self):
        """Apply the Qwen3Next MoE patch."""
        self.original_values["Qwen3NextSparseMoeBlock.forward"] = Qwen3NextSparseMoeBlock.forward
        Qwen3NextSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Qwen3Next MoE patch."""
        Qwen3NextSparseMoeBlock.forward = self.original_values[  # type: ignore
            "Qwen3NextSparseMoeBlock.forward"
        ]
