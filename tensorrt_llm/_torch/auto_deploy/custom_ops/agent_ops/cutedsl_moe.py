"""
Cute-DSL implementation of the fused MoE MLP used throughout the benchmarks.

The previous version of this module relied on Triton kernels for both GEMMs.
We now dispatch through the CuTeDSL GEMM wrappers so that the first GEMM fuses
ReLU\u00b2 in its epilogue (`gemm_act_tuned`) and the second GEMM reuses the
standard `gemm_tuned` path.
"""

from __future__ import annotations

import torch

from .cutedsl_gemm import gemm_act_tuned, gemm_tuned


def _pack_routed_tokens(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Group token/expert assignments by expert id and build CuTeDSL offsets.

    Uses fast CUDA kernel (moe_align_cutedsl) for efficient sorting and direct
    output of token IDs, sorted weights, and cu_seqlens.

    Returns the maximum buffer size (not actual count) for CUDA graph compatibility.
    The actual valid range is encoded in cu_seqlens.
    """
    from ..fused_moe.load_moe_align import moe_align_cutedsl

    device = topk_ids.device

    if topk_ids.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=topk_weights.dtype, device=device),
            torch.zeros(num_experts + 1, dtype=torch.int32, device=device),
            0,
        )

    tokens, top_k = topk_ids.shape
    T = tokens * top_k

    # Allocate output buffers (full size, no dynamic slicing)
    sorted_token_ids = torch.empty((T,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((T,), dtype=topk_weights.dtype, device=device)
    cu_seqlens = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    # Flatten inputs
    topk_ids_flat = topk_ids.reshape(-1).contiguous()
    topk_weights_flat = topk_weights.reshape(-1).contiguous()

    # Call optimized CUDA kernel - outputs everything we need!
    moe_align_cutedsl(
        topk_ids_flat,
        topk_weights_flat,
        num_experts,
        top_k,
        sorted_token_ids,
        sorted_weights,
        cu_seqlens,
        num_tokens_post_pad,
    )

    # Convert sorted_token_ids from int32 to int64 for indexing
    sorted_token_ids = sorted_token_ids.to(torch.int64)

    # Return T (full buffer size) - cu_seqlens defines actual valid ranges
    # This avoids any dynamic operations for CUDA graph compatibility
    return sorted_token_ids, sorted_weights, cu_seqlens, T


def fused_mlp_relu2_unquantized(
    hidden_states: torch.Tensor,  # [M, H]
    w_up: torch.Tensor,  # [E, H, I]
    w_down: torch.Tensor,  # [E, H, I] -> [E, I, H]
    topk_ids: torch.Tensor,  # [M, top_k]
    topk_weights: torch.Tensor,  # [M, top_k]
    *,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Two-GEMM MoE block where the first GEMM fuses ReLU\u00b2 via CuTeDSL."""

    assert hidden_states.dim() == 2, "hidden_states must be 2D"
    assert w_up.dim() == 3 and w_down.dim() == 3, "Weights must be stacked per expert"
    assert topk_ids.shape == topk_weights.shape, "Routing ids and weights must align"
    assert hidden_states.device.type == "cuda", "MoE kernel requires CUDA"  # type: ignore[attr-defined]

    device = hidden_states.device
    dtype = hidden_states.dtype
    tokens, hidden_size = hidden_states.shape
    num_experts, hidden_size_up, intermediate_size = w_up.shape
    num_experts_down, intermediate_size_down, hidden_size_down = w_down.shape

    assert num_experts == num_experts_down, "Mismatch in expert count"
    assert hidden_size == hidden_size_up == hidden_size_down, "Hidden size mismatch"
    assert intermediate_size == intermediate_size_down, "Intermediate size mismatch"

    tokens_sorted, weights_sorted, cu_seqlens, total_pairs = _pack_routed_tokens(
        topk_ids, topk_weights, num_experts=num_experts
    )

    # CUDA graph safe: use full buffers (no dynamic slicing)
    # Padding entries have token_id=0 and weight=0 (set by kernel)
    # cu_seqlens defines valid data ranges for CuTeDSL GEMMs
    if total_pairs == 0:
        return torch.zeros_like(hidden_states)

    gathered = hidden_states[tokens_sorted].contiguous()
    weights_sorted = weights_sorted.to(dtype)

    if apply_router_weight_on_input:
        gathered = gathered * weights_sorted.unsqueeze(1)

    preact = torch.empty((total_pairs, intermediate_size), device=device, dtype=dtype)
    postact = torch.empty((total_pairs, intermediate_size), device=device, dtype=dtype)
    gemm_act_tuned(
        A=gathered,
        B=w_up,
        preact_out=preact,
        postact_out=postact,
        activation="relu_sq",
        cu_seqlens_m=cu_seqlens,
    )

    expert_outputs = torch.empty((total_pairs, hidden_size), device=device, dtype=dtype)
    gemm_tuned(
        A=postact,
        B=w_down,
        out=expert_outputs,
        cu_seqlens_m=cu_seqlens,
    )

    if not apply_router_weight_on_input:
        expert_outputs.mul_(weights_sorted.unsqueeze(1))

    final = torch.zeros((tokens, hidden_size), device=device, dtype=dtype)
    # Padding entries (if any) have weight=0, so adding them has no effect
    final.index_add_(0, tokens_sorted, expert_outputs)
    return final.view_as(hidden_states)


@torch.library.custom_op("auto_deploy::cutedsl_moe_fused", mutates_args=())
def cutedsl_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    """CuTeDSL implementation of the fused MoE ops for Nemotron-6 models."""

    x_shape = x.shape
    hidden_size = x_shape[-1]
    x2d = x.view(-1, hidden_size)

    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int64)

    topk_ids = selected_experts.contiguous()
    topk_weights = routing_weights.to(x.dtype).contiguous()

    out2d = fused_mlp_relu2_unquantized(
        x2d,
        w1_stacked_weight,
        w2_stacked_weight,
        topk_ids,
        topk_weights,
        apply_router_weight_on_input=False,
    )
    return out2d.view(x_shape)


@cutedsl_fused_moe.register_fake
def cutedsl_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)
