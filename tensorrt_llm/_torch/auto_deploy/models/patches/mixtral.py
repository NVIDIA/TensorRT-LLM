"""A patch for Mixtral MoE to make it compatible with torch.export."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_moe(self: MixtralSparseMoeBlock, hidden_states: torch.Tensor):
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
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.w1.weight for expert in self.experts],  # gate projection
        w2_weight=[expert.w2.weight for expert in self.experts],  # down projection
        w3_weight=[expert.w3.weight for expert in self.experts],  # up projection
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


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
