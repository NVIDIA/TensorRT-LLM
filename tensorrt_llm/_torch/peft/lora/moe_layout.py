# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for assembling routed-expert MoE LoRA adapters.

The MVP supports shared-outer LoRA via **load-time replication**: the shared
side of the adapter is duplicated `num_experts` times so that the underlying
MoE kernel (which symmetrically offsets both A and B pointers by
`expert_index * dim * rank`) sees the same `[E, ...]` layout as a fully
per-expert adapter.

These utilities are intentionally NumPy/torch-free at the top level so that
generators in unit tests can reuse them without paying for a torch import.
"""

from typing import Dict, List, Literal, Optional

import torch

SharedSide = Literal["A", "B", None]

# Routed-expert MoE LoRA modules supported by the MVP.
MOE_LORA_MODULES: List[str] = ["moe_h_to_4h", "moe_4h_to_h", "moe_gate"]

# Canonical mapping of modules to their "outer" side (shared across experts):
#   moe_h_to_4h / moe_gate are upward projections (hidden -> intermediate);
#     the "outer" (residual-stream-side) matrix is A (shape [rank, hidden]).
#   moe_4h_to_h is the down-projection (intermediate -> hidden); the outer
#     matrix is B (shape [hidden, rank]).
DEFAULT_SHARED_SIDE: Dict[str, SharedSide] = {
    "moe_h_to_4h": "A",
    "moe_gate": "A",
    "moe_4h_to_h": "B",
}


def make_per_expert_lora(
    num_experts: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    shared_side: SharedSide = None,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a (A, B) LoRA tensor pair for an MoE module.

    Always returns shapes `A: [E, rank, in_dim]` and `B: [E, out_dim, rank]`,
    suitable for `torch.stack(...)` in `tensorrt_llm/lora_manager.py`.

    When `shared_side == "A"`, the same `[rank, in_dim]` matrix is replicated
    across all `E` experts (load-time replication for shared-outer). Likewise
    for `shared_side == "B"`. When `shared_side is None`, each expert gets
    independent A and B matrices.

    Args:
        num_experts: number of experts in this MoE layer.
        rank: LoRA rank for this module.
        in_dim: input hidden size (e.g. `hidden_size` for moe_h_to_4h).
        out_dim: output hidden size (e.g. `intermediate_size` for moe_h_to_4h).
        dtype: tensor dtype.
        device: tensor device.
        shared_side: which side to replicate ("A", "B", or None).
        seed: optional torch RNG seed for deterministic generation.

    Returns:
        Dict with keys "A" (shape [E, rank, in_dim]) and "B" (shape [E, out_dim, rank]).
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    def _randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    if shared_side == "A":
        shared_a = _randn(rank, in_dim)
        a = shared_a.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
        b = _randn(num_experts, out_dim, rank)
    elif shared_side == "B":
        a = _randn(num_experts, rank, in_dim)
        shared_b = _randn(out_dim, rank)
        b = shared_b.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
    elif shared_side is None:
        a = _randn(num_experts, rank, in_dim)
        b = _randn(num_experts, out_dim, rank)
    else:
        raise ValueError(
            f"shared_side must be 'A', 'B', or None; got {shared_side!r}")

    return {"A": a, "B": b}


def reference_moe_lora_delta(
    a_stacked: torch.Tensor,
    b_stacked: torch.Tensor,
    x: torch.Tensor,
    token_to_expert: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute the reference LoRA delta `delta[t] = scale * (B[e_t] @ A[e_t] @ x[t])`
    for a routed MoE layer, in eager PyTorch.

    This is used by unit tests to check the fused MoE LoRA op against a
    straightforward per-expert reference. Works for both per-expert and
    shared-outer adapters as long as `a_stacked`/`b_stacked` are already
    expanded to `[E, ...]`.

    Args:
        a_stacked: `[E, rank, in_dim]`
        b_stacked: `[E, out_dim, rank]`
        x:         `[num_tokens, in_dim]`
        token_to_expert: `[num_tokens]` int tensor of expert indices.
        scale: LoRA scale factor (`alpha / rank` for vanilla LoRA, `alpha / sqrt(rank)`
            for rs-LoRA).

    Returns:
        Tensor of shape `[num_tokens, out_dim]`.
    """
    num_tokens = x.shape[0]
    out_dim = b_stacked.shape[1]
    output = torch.zeros(num_tokens,
                         out_dim,
                         dtype=x.dtype,
                         device=x.device)
    for t in range(num_tokens):
        e = int(token_to_expert[t].item())
        # (rank, in) @ (in,) -> (rank,)
        mid = a_stacked[e] @ x[t]
        # (out, rank) @ (rank,) -> (out,)
        output[t] = scale * (b_stacked[e] @ mid)
    return output
