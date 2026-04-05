# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernels and custom op registration for dense MoE with fused GEMM + activation.

The dense MoE computation is:
    1. gate_up = bmm(hidden, gate_up_w) + gate_up_b
    2. Interleaved split: gate = gate_up[..., ::2], up = gate_up[..., 1::2]
    3. Clamp: gate = clamp(gate, max=limit), up = clamp(up, min=-limit, max=limit)
    4. GLU: glu = gate * sigmoid(gate * alpha)
    5. Fused multiply: act_out = (up + 1) * glu
    6. down_out = bmm(act_out, down_w) + down_b
    7. Weighted sum over experts

The Triton kernel fuses steps 2-5 (interleaved split, clamp, GLU, multiply) into a
single kernel to avoid multiple passes over intermediate memory.

A second Triton kernel handles the routing-weighted summation (step 7).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_glu_activation_kernel(
    gate_up_ptr,
    output_ptr,
    stride_gate_up_row,
    stride_out_row,
    alpha_val,
    limit_val,
    I_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Fused interleaved-split + clamp + GLU activation kernel.

    Reads gate_up tensor with interleaved gate/up values of shape [..., 2*I],
    computes the fused activation, and writes output of shape [..., I].

    For each element i in [0, I):
        gate = clamp(gate_up[..., 2*i], max=limit)
        up   = clamp(gate_up[..., 2*i+1], min=-limit, max=limit)
        glu  = gate * sigmoid(gate * alpha)
        out  = (up + 1) * glu

    Grid: (num_rows,) - one program per row (expert x token combination).
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_I)
    mask = col_offsets < I_SIZE

    # Compute interleaved indices: gate at even positions, up at odd positions
    gate_offsets = col_offsets * 2
    up_offsets = col_offsets * 2 + 1

    # Load from interleaved gate_up
    gate_up_row_ptr = gate_up_ptr + row_idx * stride_gate_up_row
    gate_vals = tl.load(gate_up_row_ptr + gate_offsets, mask=mask, other=0.0)
    up_vals = tl.load(gate_up_row_ptr + up_offsets, mask=mask, other=0.0)

    # Upcast to float32 for numerical stability
    gate_f = gate_vals.to(tl.float32)
    up_f = up_vals.to(tl.float32)

    # Clamp: gate max=limit, up min=-limit max=limit
    gate_f = tl.minimum(gate_f, limit_val)
    up_f = tl.maximum(tl.minimum(up_f, limit_val), -limit_val)

    # GLU: glu = gate * sigmoid(gate * alpha)
    glu = gate_f * tl.sigmoid(gate_f * alpha_val)

    # Fused multiply: (up + 1) * glu
    result = (up_f + 1.0) * glu

    # Store result
    out_row_ptr = output_ptr + row_idx * stride_out_row
    tl.store(out_row_ptr + col_offsets, result.to(gate_vals.dtype), mask=mask)


@triton.jit
def _weighted_expert_sum_kernel(
    expert_out_ptr,
    routing_weights_ptr,
    output_ptr,
    stride_expert_out_e,
    stride_expert_out_t,
    stride_expert_out_h,
    stride_routing_t,
    stride_routing_e,
    stride_out_t,
    stride_out_h,
    num_experts: tl.constexpr,
    H_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Weighted summation over experts for each token.

    For each token t and hidden dim h:
        output[t, h] = sum_e(routing_weights[t, e] * expert_out[e, t, h])

    Grid: (num_tokens,) - one program per token.
    """
    token_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_H)
    mask = col_offsets < H_SIZE

    # Accumulate weighted expert outputs in float32
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Probe input dtype before the loop (Triton can't see loop-scoped vars after the loop)
    _dtype_probe = tl.load(expert_out_ptr, eviction_policy="evict_first")

    for e in range(num_experts):
        # Load routing weight for this token-expert pair
        w = tl.load(routing_weights_ptr + token_idx * stride_routing_t + e * stride_routing_e)
        w_f = w.to(tl.float32)

        # Load expert output for this token
        expert_row_ptr = expert_out_ptr + e * stride_expert_out_e + token_idx * stride_expert_out_t
        expert_vals = tl.load(
            expert_row_ptr + col_offsets * stride_expert_out_h, mask=mask, other=0.0
        )

        acc += w_f * expert_vals.to(tl.float32)

    # Store result
    out_ptr = output_ptr + token_idx * stride_out_t
    tl.store(out_ptr + col_offsets * stride_out_h, acc.to(_dtype_probe.dtype), mask=mask)


def _moe_dense_mlp_triton(
    hidden_states: Tensor,
    routing_weights: Tensor,
    gate_up_w: Tensor,
    gate_up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> Tensor:
    """Python launcher for the Triton-accelerated dense MoE.

    Uses torch.bmm for the GEMM components and Triton kernels for:
      - Fused interleaved-split + clamp + GLU activation
      - Routing-weighted expert summation

    Args:
        hidden_states: Input tensor [B, S, H] or [B*S, H].
        routing_weights: Dense routing weights [B*S, E].
        gate_up_w: Fused gate+up weight [E, H, 2I].
        gate_up_b: Fused gate+up bias [E, 2I].
        down_w: Down projection weight [E, I, H].
        down_b: Down projection bias [E, H].
        alpha: Scaling factor for sigmoid in GLU.
        limit: Clamp limit for gate and up projections.

    Returns:
        Output tensor with the same shape as hidden_states.
    """
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    hidden_flat = hidden_states.reshape(-1, hidden_size)  # (T, H)
    num_tokens = hidden_flat.shape[0]
    num_experts = routing_weights.shape[1]
    intermediate_size = gate_up_w.shape[2] // 2  # 2I -> I

    # Step 1: Replicate tokens across experts and compute gate_up projection (BMM)
    # hidden_rep: [E, T, H]
    hidden_rep = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)
    # gate_up: [E, T, 2I]
    gate_up = torch.bmm(hidden_rep, gate_up_w) + gate_up_b[:, None, :]

    # Step 2-5: Fused interleaved-split + clamp + GLU activation (Triton kernel)
    # Output: [E, T, I]
    act_out = torch.empty(
        num_experts,
        num_tokens,
        intermediate_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    total_rows = num_experts * num_tokens
    BLOCK_I = triton.next_power_of_2(intermediate_size)

    gate_up_contig = gate_up.contiguous()
    grid = (total_rows,)
    _fused_glu_activation_kernel[grid](
        gate_up_contig,
        act_out,
        gate_up_contig.stride(-2),
        act_out.stride(-2),
        float(alpha),
        float(limit),
        I_SIZE=intermediate_size,
        BLOCK_I=BLOCK_I,
        num_warps=4,
        num_stages=3,
    )

    # Step 6: Down projection (BMM)
    # next_states: [E, T, H]
    next_states = torch.bmm(act_out, down_w) + down_b[:, None, :]

    # Step 7: Routing-weighted summation over experts (Triton kernel)
    # Ensure next_states is contiguous for the kernel
    next_states = next_states.contiguous()
    output = torch.empty(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    BLOCK_H = triton.next_power_of_2(hidden_size)
    grid_sum = (num_tokens,)
    _weighted_expert_sum_kernel[grid_sum](
        next_states,
        routing_weights,
        output,
        stride_expert_out_e=next_states.stride(0),
        stride_expert_out_t=next_states.stride(1),
        stride_expert_out_h=next_states.stride(2),
        stride_routing_t=routing_weights.stride(0),
        stride_routing_e=routing_weights.stride(1),
        stride_out_t=output.stride(0),
        stride_out_h=output.stride(1),
        num_experts=num_experts,
        H_SIZE=hidden_size,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )

    return output.reshape(*leading_shape, hidden_size)


@torch.library.custom_op("auto_deploy::triton_moe_dense_mlp", mutates_args=())
def triton_moe_dense_mlp(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    """Triton-accelerated dense MoE custom op (GPT-OSS style).

    Matches the signature and semantics of auto_deploy::torch_moe_dense_mlp but
    uses Triton kernels for the fused GLU activation and weighted expert summation.
    """
    return _moe_dense_mlp_triton(
        hidden_states, routing_weights, gate_up_w, gate_up_b, down_w, down_b, alpha, limit
    )


@triton_moe_dense_mlp.register_fake
def _triton_moe_dense_mlp_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
