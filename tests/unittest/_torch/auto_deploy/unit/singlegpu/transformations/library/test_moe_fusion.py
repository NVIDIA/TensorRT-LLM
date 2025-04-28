import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import run_test
from _model_test_utils import MoEOpModel
from _torch_test_utils import fp8_compatible

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


class BlockSparseTop2MLPFP8(nn.Module):
    def __init__(self, ffn_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim
        # Input scale fixed to 1.0
        self.register_buffer("inp_scale", torch.tensor(1.0, dtype=torch.float, device=device))
        # FP8 weight scale factor depends on dtype
        wt_factor = 448 if dtype == torch.bfloat16 else 432

        w1_fp32 = torch.randn(ffn_dim, hidden_dim, device=device)
        w3_fp32 = torch.randn(ffn_dim, hidden_dim, device=device)
        w2_fp32 = torch.randn(hidden_dim, ffn_dim, device=device)
        w1_scale = (w1_fp32.abs().max() / wt_factor).float().to(device)
        w3_scale = (w3_fp32.abs().max() / wt_factor).float().to(device)
        w2_scale = (w2_fp32.abs().max() / wt_factor).float().to(device)

        self.register_buffer("w1_scale", w1_scale)
        self.register_buffer("w3_scale", w3_scale)
        self.register_buffer("w2_scale", w2_scale)

        w1_fp8 = (w1_fp32 / w1_scale).to(torch.float8_e4m3fn)
        w3_fp8 = (w3_fp32 / w3_scale).to(torch.float8_e4m3fn)
        w2_fp8 = (w2_fp32 / w2_scale).to(torch.float8_e4m3fn)
        self.register_parameter("w1_fp8", nn.Parameter(w1_fp8))
        self.register_parameter("w3_fp8", nn.Parameter(w3_fp8))
        self.register_parameter("w2_fp8", nn.Parameter(w2_fp8))
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states
        w1_out = torch.ops.quant.fp8_linear(
            x,
            self.w1_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w1_scale,
        )
        w3_out = torch.ops.quant.fp8_linear(
            x,
            self.w3_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w3_scale,
        )
        fused = self.act_fn(w1_out) * w3_out
        out = torch.ops.quant.fp8_linear(
            fused,
            self.w2_fp8,
            bias=None,
            input_scale=self.inp_scale,
            weight_scale=self.w2_scale,
        )
        return out


class BlockSparseMoE(nn.Module):
    def __init__(
        self,
        hidden_size=32,
        num_experts=4,
        intermediate_size=16,
        use_fp8: bool = False,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = 2
        self.gate = nn.Linear(hidden_size, num_experts, bias=False).to(device=device, dtype=dtype)
        expert_cls = BlockSparseTop2MLPFP8 if use_fp8 else BlockSparseTop2MLP
        if use_fp8:
            self.experts = nn.ModuleList(
                [
                    expert_cls(intermediate_size, hidden_size, dtype, device)
                    for _ in range(num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [expert_cls(intermediate_size, hidden_size) for _ in range(num_experts)]
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
    def __init__(self, use_fp8: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.block_sparse_moe = BlockSparseMoE(
            hidden_size=32,
            num_experts=2,
            intermediate_size=16,
            use_fp8=use_fp8,
        )

    def forward(self, x):
        embedded = F.embedding(x, self.embedding.weight)
        residual = embedded
        hidden_states = self.block_sparse_moe(embedded)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_input(self, device):
        return torch.randint(0, 100, (2, 10), device=device)


@pytest.mark.parametrize(
    "use_fp8,expected_op,skip",
    [
        pytest.param(False, torch.ops.moe.torch_moe, False, id="simple"),
        pytest.param(True, torch.ops.moe.torch_fp8_moe, not fp8_compatible(), id="fp8"),
    ],
)
def test_moe_matching(use_fp8, expected_op, skip):
    if skip:
        pytest.skip("FP8 not supported on this hardware")
    device = "cuda"

    model = MoEPatternModel(use_fp8=use_fp8).to(device=device)
    if not use_fp8:
        model = model.to(dtype=torch.bfloat16)
    else:
        model.embedding = model.embedding.to(dtype=torch.bfloat16)
        model.block_sparse_moe.gate = model.block_sparse_moe.gate.to(dtype=torch.bfloat16)
    x = model.get_input(device=device)

    _ = run_test(
        model,
        x,
        match_moe_pattern,
        lambda gm: any(is_op(n, expected_op) for n in gm.graph.nodes),
        lambda num: num,
        atol=0.05 if use_fp8 else 1e-3,
        rtol=0.01 if use_fp8 else 1e-3,
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
            is_op(n, {torch.ops.moe.torch_fused_moe, torch.ops.moe.trtllm_fused_moe})
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
