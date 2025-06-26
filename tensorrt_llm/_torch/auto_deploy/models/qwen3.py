import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


# patch for MoE to reduce torch.export time
def _forward_moe(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
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


Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward
Qwen3MoeSparseMoeBlock.forward = _forward_moe
