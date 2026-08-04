# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for assembling routed-expert (per-expert) MoE LoRA adapters.

Each expert gets its own (A, B) low-rank pair; the underlying MoE kernel
offsets both A and B pointers by `expert_index * dim * rank` to read the
per-expert slice. These utilities are used by the MoE LoRA unit tests to
generate synthetic adapters and a reference forward pass.
"""

from typing import Dict, List, Optional

import torch

# Routed-expert MoE LoRA modules.
MOE_LORA_MODULES: List[str] = ["moe_h_to_4h", "moe_4h_to_h", "moe_gate"]


def make_per_expert_lora(
    num_experts: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a per-expert (A, B) LoRA tensor pair for an MoE module.

    Returns shapes `A: [E, rank, in_dim]` and `B: [E, out_dim, rank]`, with an
    independent low-rank pair per expert, suitable for `torch.stack(...)` in
    `tensorrt_llm/lora_manager.py`.

    Args:
        num_experts: number of experts in this MoE layer.
        rank: LoRA rank for this module.
        in_dim: input hidden size (e.g. `hidden_size` for moe_h_to_4h).
        out_dim: output hidden size (e.g. `intermediate_size` for moe_h_to_4h).
        dtype: tensor dtype.
        device: tensor device.
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

    a = _randn(num_experts, rank, in_dim)
    b = _randn(num_experts, out_dim, rank)
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
    straightforward per-expert reference, with `a_stacked`/`b_stacked` given
    in the `[E, ...]` stacked layout.

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
    output = torch.zeros(num_tokens, out_dim, dtype=x.dtype, device=x.device)
    for t in range(num_tokens):
        e = int(token_to_expert[t].item())
        # (rank, in) @ (in,) -> (rank,)
        mid = a_stacked[e] @ x[t]
        # (out, rank) @ (rank,) -> (out,)
        output[t] = scale * (b_stacked[e] @ mid)
    return output


def reference_swiglu_moe_lora(
    x: torch.Tensor,
    w3_w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    fc1_a: torch.Tensor,
    fc1_b: torch.Tensor,
    gated_a: torch.Tensor,
    gated_b: torch.Tensor,
    fc2_a: torch.Tensor,
    fc2_b: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Hand-written eager PyTorch reference for a routed-expert SwiGLU MoE layer
    with LoRA on all three projections.

    Mirrors the math performed by `torch.ops.trtllm.fused_moe` when called
    with the per-expert weight layout used by the unit tests:

      * `w3_w1` packs the two FC1 GEMMs as `[E, 2 * inter_size, hidden]`
        with the `up` (linear / non-silu) half at offset `[0, inter)` and
        the `gate` (silu) half at offset `[inter, 2 * inter)`. This
        matches the layout the kernel reads in `doActivationKernel`
        (`moe_kernels.cu`: `linear_value = gemm_result[..0, inter)`,
        `gate_value = gemm_result[..inter, 2 * inter)`,
        `out = silu(gate_value) * linear_value`).
      * `moe_h_to_4h` LoRA (passed via `fc1_*`) is added to the **gate**
        half (silu side); `moe_gate` LoRA (`gated_*`) is added to the
        **up** half (linear side). This matches the fan-out in
        `CutlassMoeFCRunner::loraFC1`: `lora_gated_out` writes at offset
        0 (linear/up) and `lora_fc1_result` writes at offset
        `inter_size` (gate/silu).
      * `moe_4h_to_h` LoRA (`fc2_*`) is added to the FC2 output before
        the top-k weighted finalize sum.

    The whole computation is run in fp32 internally to avoid the reference
    itself being subject to bf16 reduction noise; the result is cast back to
    `x.dtype` at the end.

    Args:
        x: `[num_tokens, hidden_size]` activations.
        w3_w1: `[num_experts, 2 * inter_size, hidden_size]` packed FC1
            weights with up half at `[0, inter)` and gate half at
            `[inter, 2 * inter)`.
        w2: `[num_experts, hidden_size, inter_size]` FC2 weights.
        topk_ids: `[num_tokens, top_k]` int tensor of expert indices.
        topk_scores: `[num_tokens, top_k]` float32 weights (already
            normalized; the kernel multiplies the per-expert FC2 output by
            these in the finalize step).
        fc1_a / fc1_b: `moe_h_to_4h` (gate-side, silu) LoRA matrices,
            `[E, rank, hidden]` and `[E, inter, rank]`.
        gated_a / gated_b: `moe_gate` (up-side, linear) LoRA matrices,
            `[E, rank, hidden]` and `[E, inter, rank]`.
        fc2_a / fc2_b: `moe_4h_to_h` (down) LoRA matrices,
            `[E, rank, inter]` and `[E, hidden, rank]`.
        scale: Scalar applied uniformly to all three LoRA branches. Matches
            the kernel's behavior of forwarding adapter A/B as-is (the
            kernel itself does not multiply by an alpha factor; any alpha
            scaling is expected to have been folded into the adapter at load
            time).

    Returns:
        `[num_tokens, hidden_size]` tensor in `x.dtype`.
    """
    num_tokens, hidden_size = x.shape
    num_experts, two_inter, hidden_check = w3_w1.shape
    assert hidden_check == hidden_size, (
        f"w3_w1 hidden dim mismatch: {hidden_check} vs {hidden_size}"
    )
    assert two_inter % 2 == 0, f"w3_w1 second dim must be 2 * inter_size, got {two_inter}"
    inter_size = two_inter // 2
    top_k = topk_ids.shape[-1]
    out_dtype = x.dtype

    f32 = torch.float32
    x32 = x.to(f32)
    # Up (linear, no silu) is the FIRST half; gate (silu) is the SECOND half.
    w_up = w3_w1[:, :inter_size, :].to(f32)
    w_gate = w3_w1[:, inter_size:, :].to(f32)
    w_down = w2.to(f32)

    fc1_a32 = fc1_a.to(f32)
    fc1_b32 = fc1_b.to(f32)
    gated_a32 = gated_a.to(f32)
    gated_b32 = gated_b.to(f32)
    fc2_a32 = fc2_a.to(f32)
    fc2_b32 = fc2_b.to(f32)
    scores32 = topk_scores.to(f32)

    silu = torch.nn.functional.silu
    out32 = torch.zeros(num_tokens, hidden_size, dtype=f32, device=x.device)

    for t in range(num_tokens):
        xt = x32[t]
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            w = float(scores32[t, k].item())

            gate_lin = w_gate[e] @ xt
            gate_lora = scale * (fc1_b32[e] @ (fc1_a32[e] @ xt))
            gate_v = gate_lin + gate_lora

            up_lin = w_up[e] @ xt
            up_lora = scale * (gated_b32[e] @ (gated_a32[e] @ xt))
            up_v = up_lin + up_lora

            inter_v = silu(gate_v) * up_v

            down_lin = w_down[e] @ inter_v
            down_lora = scale * (fc2_b32[e] @ (fc2_a32[e] @ inter_v))
            down_v = down_lin + down_lora

            out32[t] += w * down_v

    return out32.to(out_dtype)
