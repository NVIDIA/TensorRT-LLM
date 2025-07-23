"""A patch for Qwen3 MoE to make it compatible with torch.export and reduce export time."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_moe(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    # check if we can apply the patch
    use_original_forward = False
    if not all(isinstance(expert.act_fn, nn.SiLU) for expert in self.experts):
        use_original_forward = True

    if any(getattr(mod, "bias", None) is not None for mod in self.experts.modules()):
        use_original_forward = True

    # rely on original forward instead
    if use_original_forward:
        return self._original_forward(hidden_states)

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@ExportPatchRegistry.register("hf_qwen3_moe")
class Qwen3MoePatch(BaseExportPatch):
    """Patch for Qwen3 MoE to make it compatible with torch.export and reduce export time.

    This patch replaces the forward method of Qwen3MoeSparseMoeBlock with
    a version that uses the torch_moe custom operator for better export compatibility.
    """

    def _apply_patch(self):
        """Apply the Qwen3 MoE patch."""
        # Store original forward method
        self.original_values["Qwen3MoeSparseMoeBlock.forward"] = Qwen3MoeSparseMoeBlock.forward

        # Apply patch by replacing the forward method
        Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward  # type: ignore
        Qwen3MoeSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Qwen3 MoE patch."""
        # Restore original forward method
        Qwen3MoeSparseMoeBlock.forward = self.original_values["Qwen3MoeSparseMoeBlock.forward"]  # type: ignore

        # Clean up the temporary attribute
        if hasattr(Qwen3MoeSparseMoeBlock, "_original_forward"):
            delattr(Qwen3MoeSparseMoeBlock, "_original_forward")
