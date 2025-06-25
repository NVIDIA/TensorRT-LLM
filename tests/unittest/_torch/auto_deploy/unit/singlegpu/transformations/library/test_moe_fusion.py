import torch
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import run_test
from _model_test_utils import MoEOpModel

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transformations.library.fused_moe import (
    fuse_moe,
    match_moe_pattern,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class BlockSparseTop2MLP(nn.Module):
    def __init__(self, ffn_dim, hidden_dim):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class BlockSparseMoE(nn.Module):
    def __init__(self, hidden_size=32, num_experts=4, intermediate_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.top_k = 2
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [BlockSparseTop2MLP(intermediate_size, hidden_size) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class MoEPatternModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.block_sparse_moe = BlockSparseMoE(hidden_size=32, num_experts=2, intermediate_size=16)

    def forward(self, x):
        embedded = F.embedding(x, self.embedding.weight)
        residual = embedded
        hidden_states = self.block_sparse_moe(embedded)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_input(self, device):
        return torch.randint(0, 100, (2, 10), device=device)


def test_moe_matching():
    device = "cuda"
    model = MoEPatternModel().to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device)

    _ = run_test(
        model,
        x,
        match_moe_pattern,
        lambda gm: any(is_op(n, torch.ops.auto_deploy.torch_moe) for n in gm.graph.nodes),
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
    )


def test_moe_fusion():
    device = "cuda"
    model = MoEOpModel().to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    fused_gm_transformed = run_test(
        model,
        x,
        fuse_moe,
        lambda gm: any(
            is_op(
                n, {torch.ops.auto_deploy.torch_moe_fused, torch.ops.auto_deploy.trtllm_moe_fused}
            )
            for n in gm.graph.nodes
        ),
        lambda num_p_og: num_p_og,
        atol=0.2,
        rtol=0.5,
        test_load_hook=False,  # state_dict changed after loading hook
        strict_loading=True,
    )

    # expert weights are fused and stacked in fusion
    num_param_nodes = len(list(model.named_parameters()))
    num_param_nodes_fused = len(list(fused_gm_transformed.named_parameters()))
    assert (
        num_param_nodes_fused < num_param_nodes
    ), f"""number of parameter nodes after fusion {num_param_nodes_fused} <
        number of parameter nodes before fusion {num_param_nodes}"""
