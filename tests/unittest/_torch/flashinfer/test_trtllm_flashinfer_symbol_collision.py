"""Unit tests for FlashInfer fused MOE custom op."""

import flashinfer.fused_moe
import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.torch_moe  # noqa: F401
import tensorrt_llm._torch.custom_ops.torch_custom_ops as trt_ops  # noqa: F401
from tensorrt_llm._torch.utils import ActivationType


def test_flashinfer_fused_moe_matches_torch_moe():
    """Test that flashinfer_fused_moe matches torch_moe reference."""
    torch.manual_seed(0)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flashinfer_fused_moe test")

    device = "cuda"
    dtype = torch.bfloat16

    # Small test case
    M = 8  # tokens
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 128
    E = 4  # experts
    top_k = 2

    # Input
    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype)

    # Expert weights for gated MLP (SwiGLU)
    # w1 = gate projection, w3 = up projection, w2 = down projection
    w1_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]
    w2_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]
    w3_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]

    # FlashInfer expects fc1 (gate + up concatenated) and fc2 (down)
    # fc1_expert_weights: [E, 2*INTERMEDIATE_SIZE, HIDDEN_SIZE]
    w1_w3_stacked = torch.stack(
        [torch.cat([w3, w1], dim=0) for w1, w3 in zip(w1_list, w3_list)], dim=0
    ).contiguous()

    # fc2_expert_weights: [E, HIDDEN_SIZE, INTERMEDIATE_SIZE]
    w2_stacked = torch.stack(w2_list, dim=0).contiguous()

    # Random routing with top-k normalization
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_full = torch.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_full, k=top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float32)

    # FlashInfer fused MOE - call directly
    out_flashinfer = flashinfer.fused_moe.cutlass_fused_moe(
        input=x,
        token_selected_experts=selected_experts.to(torch.int32),
        token_final_scales=routing_weights,
        fc1_expert_weights=w1_w3_stacked,
        fc2_expert_weights=w2_stacked,
        output_dtype=dtype,
        quant_scales=[],
    )

    # Reference Torch MoE (gated_mlp with SwiGLU)
    out_torch = torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w1_list,  # gate projection
        w2_weight=w2_list,  # down projection
        w3_weight=w3_list,  # up projection
        is_gated_mlp=True,
        act_fn=int(ActivationType.Silu),
    )

    # Compare outputs
    torch.testing.assert_close(out_flashinfer[0], out_torch, rtol=5e-1, atol=5e-1)
