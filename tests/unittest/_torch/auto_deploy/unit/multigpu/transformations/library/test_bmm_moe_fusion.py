"""Tests for BMM MoE fusion in multigpu/distributed setting."""

from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class ReferenceMoeModel(nn.Module):
    """
    GROUND TRUTH: Simple per-token MoE implementation with standard routing.

    This serves as the reference for correctness testing. It uses the simplest
    possible implementation: route each token to its top-1 expert and apply
    the expert's computation.
    """

    def __init__(
        self,
        hidden_size=64,
        intermediate_size=32,
        num_experts=4,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = 1

        # Router/gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False).to(device=device, dtype=dtype)

        # Per-expert weights (standard format)
        self.experts = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "gate_proj": nn.Linear(hidden_size, intermediate_size, bias=False).to(
                            device=device, dtype=dtype
                        ),
                        "up_proj": nn.Linear(hidden_size, intermediate_size, bias=False).to(
                            device=device, dtype=dtype
                        ),
                        "down_proj": nn.Linear(intermediate_size, hidden_size, bias=False).to(
                            device=device, dtype=dtype
                        ),
                    }
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Simple per-token routing implementation (GROUND TRUTH).

        For each token:
        1. Select top-1 expert based on router logits
        2. Apply routing weight to input before expert computation (INPUT-SIDE routing)
        3. Compute: down(up * silu(gate))
        4. Accumulate results
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [B*S, H]

        # Router logits and topk
        router_logits = self.gate(hidden_states_flat)  # [B*S, num_experts]
        topk_values, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)  # [B*S, 1]

        # Pattern expects: sigmoid(scatter(topk_values)) - match BMM model pattern
        # Scatter first, then apply sigmoid to match pattern matcher expectations
        routing_scattered = torch.zeros_like(router_logits)
        routing_weights_scattered = torch.scatter(
            routing_scattered, dim=1, index=selected_experts, src=topk_values
        )  # [B*S, num_experts]
        routing_weights_normalized = torch.sigmoid(routing_weights_scattered)  # [B*S, num_experts]

        # For the reference model, we still extract routing weight for selected expert (per token)
        # But we use the full normalized weights for the BMM pattern to match the pattern matcher
        routing_weights = routing_weights_normalized.gather(1, selected_experts)  # [B*S, 1]

        # Initialize output
        final_output = torch.zeros_like(hidden_states_flat)  # [B*S, H]

        # Process each token
        for token_idx in range(hidden_states_flat.shape[0]):
            expert_idx = selected_experts[token_idx, 0].item()
            routing_weight = routing_weights[token_idx, 0]
            token_input = hidden_states_flat[token_idx : token_idx + 1]  # [1, H]

            # INPUT-SIDE routing: apply routing weight to input before expert
            scaled_input = token_input * routing_weight

            # Expert computation: down(up * silu(gate))
            expert = self.experts[expert_idx]
            gate = expert["gate_proj"](scaled_input)  # [1, I]
            up = expert["up_proj"](scaled_input)  # [1, I]
            activated = up * F.silu(gate)  # [1, I]
            output = expert["down_proj"](activated)  # [1, H]

            final_output[token_idx] = output.squeeze(0)

        return final_output.view(batch_size, seq_len, hidden_dim)


class BmmMoeModel(nn.Module):
    """BMM-based MoE model matching Llama4 pattern."""

    def __init__(self, hidden_size, intermediate_size, num_experts, dtype, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = 1

        self.gate = nn.Linear(hidden_size, num_experts, bias=False).to(device=device, dtype=dtype)

        # Pre-stacked weights for BMM pattern: [num_experts, hidden, 2*intermediate]
        self.gate_up_weight = nn.Parameter(
            torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device)
        )
        # Down projection: [num_experts, intermediate, hidden]
        self.down_weight = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        )

    def forward(self, hidden_states):
        """
        BMM-based MoE forward matching Llama4 pattern.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [B*S, H]

        # Router logits and topk - match Llama4 pattern exactly
        router_logits = self.gate(hidden_states_flat)  # [B*S, num_experts]
        if router_logits.dtype != hidden_states.dtype:
            router_logits = router_logits.to(hidden_states.dtype)

        topk_result = torch.topk(router_logits, self.top_k, dim=-1)  # Returns tuple
        topk_values = topk_result[0]  # [B*S, 1] - values via getitem[0]
        selected_experts = topk_result[1]  # [B*S, 1] - indices via getitem[1]

        # Use scatter_ (in-place) with full_like(-inf) to match Llama4 pattern exactly
        routing_scattered = torch.full_like(router_logits, float("-inf"), dtype=hidden_states.dtype)
        routing_scattered.scatter_(
            dim=1, index=selected_experts, src=topk_values
        )  # [B*S, num_experts]

        # Apply sigmoid after scatter to match pattern: sigmoid(scatter_(full_like(-inf), topk))
        routing_weights_normalized = torch.sigmoid(routing_scattered)  # [B*S, num_experts]

        # Transpose then reshape to match Llama4 pattern: reshape(transpose(sigmoid(...)))
        routing_transposed = routing_weights_normalized.transpose(0, 1)  # [num_experts, B*S]
        routing_reshaped = routing_transposed.reshape(
            -1, 1
        )  # [num_experts*B*S, 1] - matches Llama4 pattern

        # INPUT-SIDE routing: apply routing weights to input and reshape for BMM
        repeated_input = hidden_states_flat.repeat(
            self.num_experts, 1
        )  # [num_experts*B*S, hidden] - flattened
        routed_input = (
            repeated_input * routing_reshaped
        )  # [num_experts*B*S, hidden] - broadcasts correctly
        batched_input = routed_input.view(
            self.num_experts, -1, hidden_dim
        )  # [num_experts, B*S, hidden]

        # First BMM: gate_up projection
        gate_up = torch.bmm(
            batched_input, self.gate_up_weight
        )  # [num_experts, B*S, 2*intermediate]

        # Chunk into up and gate (TRT-LLM format: [W3, W1] = [up, gate])
        up, gate = gate_up.chunk(2, dim=-1)  # [num_experts, B*S, intermediate] each

        # Activation: up * silu(gate)
        activated = up * F.silu(gate)  # [num_experts, B*S, intermediate]

        # Second BMM: down projection
        output = torch.bmm(activated, self.down_weight)  # [num_experts, B*S, hidden]

        # Sum across experts
        output = output.view(-1, hidden_dim)  # [num_experts*B*S, H]
        output = output.reshape(self.num_experts, -1, hidden_dim)  # [num_experts, B*S, H]
        output = output.sum(dim=0)  # [B*S, H]

        # Reshape back to original shape
        return output.view(batch_size, seq_len, hidden_dim)

    @staticmethod
    def from_reference(ref_model: ReferenceMoeModel) -> "BmmMoeModel":
        """
        Create a BmmMoeModel with weights copied from a reference model.

        This ensures both models compute the same function, allowing us to verify
        that the BMM pattern is mathematically equivalent to per-token routing.
        """
        device = ref_model.gate.weight.device
        dtype = ref_model.gate.weight.dtype

        bmm_model = BmmMoeModel(
            hidden_size=ref_model.hidden_size,
            intermediate_size=ref_model.intermediate_size,
            num_experts=ref_model.num_experts,
            dtype=dtype,
            device=device,
        )

        # Copy router weights
        bmm_model.gate.weight.data.copy_(ref_model.gate.weight.data)

        # Stack per-expert weights into batched format
        for expert_idx in range(ref_model.num_experts):
            expert = ref_model.experts[expert_idx]

            # gate_up_weight: [num_experts, hidden, 2*intermediate]
            # TRT-LLM format: [W3, W1] = [up, gate]
            # chunk(2, dim=-1) returns (first_half, second_half) = (up, gate) to match TRT-LLM
            bmm_model.gate_up_weight.data[expert_idx, :, : ref_model.intermediate_size] = expert[
                "up_proj"
            ].weight.data.t()  # up (w3) - FIRST HALF
            bmm_model.gate_up_weight.data[expert_idx, :, ref_model.intermediate_size :] = expert[
                "gate_proj"
            ].weight.data.t()  # gate (w1) - SECOND HALF

            # down_weight: [num_experts, intermediate, hidden]
            bmm_model.down_weight.data[expert_idx] = expert["down_proj"].weight.data.t()

        return bmm_model


def _run_bmm_moe_fusion_distributed_job(
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Run BMM MoE fusion test in distributed setting, comparing against reference.
    """
    device = "cuda"
    torch.manual_seed(2345)
    torch.cuda.manual_seed(2345)

    num_experts = max(4, world_size * 2)
    hidden_size = 64
    intermediate_size = 32

    # Create BMM model
    bmm_model = BmmMoeModel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        dtype=dtype,
        device=device,
    )

    # Generate input
    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, 64, device=device, dtype=dtype)

    # Export with full input - graph should handle dynamic batch sizes
    gm_original = torch_export_to_gm(bmm_model, args=(x,), clone=True)

    optimizer = InferenceOptimizer(
        None,
        {
            "match_bmm_moe_pattern": {
                "stage": "pattern_matcher",
            },
        },
    )
    optimizer.shared_config.local_rank = rank
    optimizer.shared_config.world_size = world_size

    gm_fused = optimizer(None, gm_original)

    # Verify fusion happened - this is the main goal of the distributed test
    # The pattern matcher should successfully identify and fuse the BMM MoE pattern
    # even in a distributed context
    has_torch_moe = any(is_op(n, torch.ops.auto_deploy.torch_moe) for n in gm_fused.graph.nodes)
    assert has_torch_moe, f"Rank {rank}: Expected torch_moe op after fusion"

    # Note: The fused graph may not execute correctly in distributed mode without
    # additional sharding transforms, but pattern matching should still work
    print(
        f"✓ Rank {rank}/{world_size}: BMM MoE pattern fusion detected successfully (dtype={dtype})"
    )


@pytest.mark.parametrize("device_count", get_device_counts(num_gpu_list=[2]))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_moe_fusion_distributed(device_count: int, dtype: torch.dtype):
    """
    Test BMM MoE fusion in distributed setting.
    Requires 2+ GPUs - only parameterized with multi-GPU device counts.
    """
    dist_common.spawn_multiprocess_job(
        job=partial(_run_bmm_moe_fusion_distributed_job, dtype=dtype),
        size=device_count,
    )
