# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused DeepSeek-V4.1 Engram gating, after the separate WKV GEMM."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _engram_gate_kernel(
    hidden_ptr,
    kv_ptr,
    query_weight_ptr,
    key_weight_ptr,
    output_ptr,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    query_stride_h,
    query_stride_d,
    key_stride_h,
    key_stride_d,
    EPS: tl.constexpr,
    HC: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    PRECOMPUTED_WEIGHT: tl.constexpr = False,
):
    if PRECOMPUTED_WEIGHT:
        row = tl.program_id(0).to(tl.int64)
        token, head = row // HC, row % HC
    else:
        token = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
    dim = tl.arange(0, BLOCK)
    valid = dim < DIM
    hidden = tl.load(
        hidden_ptr + token * hidden_stride_t + head * hidden_stride_h + dim * hidden_stride_d,
        valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv_ptr + token * kv_stride_t + (head * DIM + dim) * kv_stride_d,
        valid,
        other=0.0,
    ).to(tl.float32)
    query_weight = tl.load(
        query_weight_ptr + head * query_stride_h + dim * query_stride_d, valid, other=0.0
    ).to(tl.float32)
    if PRECOMPUTED_WEIGHT:
        weight = query_weight
    else:
        key_weight = tl.load(
            key_weight_ptr + head * key_stride_h + dim * key_stride_d, valid, other=0.0
        ).to(tl.float32)
        weight = query_weight * key_weight
    hidden_rstd = tl.rsqrt(tl.sum(hidden * hidden, 0) / DIM + EPS)
    key_rstd = tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
    dot = tl.sum(hidden * weight * key, 0) * (hidden_rstd * key_rstd) * DIM**-0.5
    # copysign retains the clamp floor at zero, including negative zero.
    signed_root = libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6)), dot)
    gate = tl.sigmoid(signed_root)
    value = tl.load(
        kv_ptr + token * kv_stride_t + (HC * DIM + dim) * kv_stride_d,
        valid,
        other=0.0,
    ).to(tl.float32)
    output = gate * value
    if ADD_RESIDUAL:
        output = hidden + output
    tl.store(output_ptr + (token * HC + head) * DIM + dim, output, valid)


def engram_gate(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    *,
    add_residual: bool = False,
    norm_weight_product: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the keys-first Engram gate in one GPU kernel.

    Args:
        hidden_states: BF16, FP16 or FP32 residual streams, shape ``[T, HC, D]``.
        kv: WKV output, shape ``[T, (HC + 1) * D]``; keys precede the shared value.
        query_weight: Per-stream query RMSNorm weights, shape ``[HC, D]``.
        key_weight: Per-stream key RMSNorm weights, shape ``[HC, D]``.
        eps: The checkpoint's RMSNorm epsilon.
        add_residual: Add the residual in FP32 before casting; otherwise return the delta.
        norm_weight_product: Optional cached FP32 query/key product, shape ``[HC, D]``.
            Refresh after weight changes. Used only for contiguous BF16 GB300 residuals.

    Returns:
        Contiguous ``[T, HC, D]`` in the residual dtype, with a PyTorch CPU fallback.
    """
    if hidden_states.ndim != 3:
        raise ValueError("Engram hidden states must have shape [T, HC, D]")
    tokens, hc, dim = hidden_states.shape
    if hc <= 0 or dim <= 0 or dim > 65536:
        raise ValueError("Engram requires positive HC and D, with D <= 65536")
    if kv.shape != (tokens, (hc + 1) * dim):
        raise ValueError("Engram WKV output must have shape [T, (HC + 1) * D]")
    if query_weight.shape != (hc, dim) or key_weight.shape != (hc, dim):
        raise ValueError("Engram norm weights must have shape [HC, D]")
    tensors = (hidden_states, kv, query_weight, key_weight)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("Engram gate tensors must be on the same device")
    if any(
        tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32) for tensor in tensors
    ):
        raise ValueError("Engram gate tensors must be BF16, FP16 or FP32")
    if eps <= 0:
        raise ValueError("Engram RMSNorm epsilon must be positive")
    if norm_weight_product is not None and (
        norm_weight_product.shape != (hc, dim)
        or norm_weight_product.dtype != torch.float32
        or norm_weight_product.device != hidden_states.device
    ):
        raise ValueError("Engram norm weight product must be FP32 [HC, D] on the input device")

    if not hidden_states.is_cuda:
        hidden = hidden_states.float()
        keys = kv[:, : hc * dim].float().unflatten(-1, (hc, dim))
        weight = query_weight.float() * key_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(
            keys.square().mean(-1) + eps
        )
        dot = (hidden * weight * keys).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        output = gate.unsqueeze(-1) * kv[:, hc * dim :].float().unsqueeze(-2)
        if add_residual:
            output = hidden + output
        return output.to(hidden_states.dtype)

    output = torch.empty((tokens, hc, dim), dtype=hidden_states.dtype, device=hidden_states.device)
    if tokens == 0:
        return output
    use_precomputed = (
        norm_weight_product is not None
        and add_residual
        and (hc, dim) == (4, 5120)
        and hidden_states.dtype == kv.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and kv.is_contiguous()
        and norm_weight_product.is_contiguous()
        and torch.cuda.get_device_capability(hidden_states.device) == (10, 3)
    )
    gate_weight = norm_weight_product if use_precomputed else query_weight
    launch_options = {"maxnreg": 96} if use_precomputed else {}
    _engram_gate_kernel[(tokens * hc,) if use_precomputed else (tokens, hc)](
        hidden_states,
        kv,
        gate_weight,
        key_weight,
        output,
        *hidden_states.stride(),
        *kv.stride(),
        *gate_weight.stride(),
        *key_weight.stride(),
        EPS=eps,
        HC=hc,
        DIM=dim,
        BLOCK=triton.next_power_of_2(dim),
        ADD_RESIDUAL=add_residual,
        PRECOMPUTED_WEIGHT=use_precomputed,
        num_warps=4 if use_precomputed or dim < 4096 else 8,
        enable_fp_fusion=False,
        **launch_options,
    )
    return output
