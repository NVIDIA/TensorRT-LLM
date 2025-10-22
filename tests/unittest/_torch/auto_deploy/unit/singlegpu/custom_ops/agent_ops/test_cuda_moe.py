import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


def test_cuda_moe_matches_torch_moe_mlp_relu2():
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16

    M = 8  # tokens
    HIDDEN_SIZE = 8
    INTERMEDIATE_SIZE = 16
    E = 8  # experts
    top_k = 2

    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype)

    # Per-expert weights (Torch MoE API uses per-expert lists for mlp style)
    w_up_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]
    w_down_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]

    # CUDA kernel expects stacked weights
    w_up_stacked = torch.stack(w_up_list, dim=0).contiguous()  # [E, I, H]
    w_down_stacked = torch.stack(w_down_list, dim=0).contiguous()  # [E, H, I]

    # Create routing with top-k normalization
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_full = torch.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_full, k=top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float32)

    # CUDA MoE (mlp with relu^2 activation between two GEMMs)
    out_cuda = torch.ops.auto_deploy.cuda_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w_up_stacked,
        w_down_stacked,
        mlp_style="mlp",
        act_fn="relu2",
    )

    # Reference Torch MoE in mlp mode with relu2 activation
    out_torch = torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w_up_list,
        w2_weight=w_down_list,
        w3_weight=[],
        mlp_style="mlp",
        act_fn="relu2",
    )

    torch.testing.assert_close(out_cuda, out_torch, rtol=1e-2, atol=1e-2)
