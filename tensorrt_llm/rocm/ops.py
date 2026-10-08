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
"""RDNA4 inference primitives with explicit PyTorch reference implementations.

Storage is FP32/FP16/BF16 and reductions accumulate in FP32. Matrix projections
use PyTorch's ROCm BLAS dispatch rather than incompatible NVIDIA MMA/PTX or
CDNA MFMA fragments. Native attention is a correctness-first online-softmax
implementation; SDPA remains the optimized default for Hugging Face inference.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F

from .kernels import load_kernels
from .runtime import KernelBackend


def _check_input(input: torch.Tensor, backend: KernelBackend) -> None:
    if backend not in ("torch", "hip"):
        raise ValueError("kernel backend must be torch or hip")
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Only FP32, FP16 and BF16 inputs are supported")
    if input.ndim == 0 or input.shape[-1] == 0:
        raise ValueError("Inputs must have a nonempty final dimension")
    if backend == "hip":
        if torch.is_grad_enabled() and input.requires_grad:
            raise ValueError("Native kernels are inference-only; use torch.inference_mode()")
        load_kernels(input.device)


def _check_weight(input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    if (
        weight.shape != (input.shape[-1],)
        or weight.device != input.device
        or weight.dtype != input.dtype
    ):
        raise ValueError("weight must match the input's final dimension, device and dtype")
    if not math.isfinite(epsilon) or not 2**-149 <= epsilon <= torch.finfo(torch.float32).max:
        raise ValueError("epsilon must be positive and finite in FP32")


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    backend: KernelBackend = "torch",
) -> torch.Tensor:
    """Normalize [..., hidden], rounding normalized values before the weight multiply."""
    _check_input(input, backend)
    _check_weight(input, weight, epsilon)
    if backend == "hip":
        return torch.ops.trtllm_rdna4.rms_norm(input.contiguous(), weight.contiguous(), epsilon)
    values = input.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(input.dtype) * weight


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    backend: KernelBackend = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normalized, updated_residual); inputs are never mutated.

    Residual addition rounds to storage dtype before the FP32 reduction, matching
    an unfused ``rms_norm(input + residual, weight)`` computation.
    """
    _check_input(input, backend)
    _check_weight(input, weight, epsilon)
    if (
        residual.shape != input.shape
        or residual.device != input.device
        or residual.dtype != input.dtype
    ):
        raise ValueError("residual must have the same shape, device and dtype as input")
    if backend == "hip":
        return torch.ops.trtllm_rdna4.add_rms_norm(
            input.contiguous(),
            residual.contiguous(),
            weight.contiguous(),
            epsilon,
        )
    updated = input + residual
    return rms_norm(updated, weight, epsilon), updated


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    epsilon: float = 1e-5,
    backend: KernelBackend = "torch",
) -> torch.Tensor:
    """Normalize [..., hidden] using FP32 centered variance and affine parameters."""
    _check_input(input, backend)
    _check_weight(input, weight, epsilon)
    if bias is not None and (
        bias.shape != weight.shape or bias.dtype != weight.dtype or bias.device != weight.device
    ):
        raise ValueError("bias must match weight")
    if backend == "hip":
        return torch.ops.trtllm_rdna4.layer_norm(
            input.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            epsilon,
        )
    return F.layer_norm(
        input.float(),
        weight.shape,
        weight.float(),
        bias.float() if bias is not None else None,
        epsilon,
    ).to(input.dtype)


def gated_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str = "silu",
    backend: KernelBackend = "torch",
) -> torch.Tensor:
    """Apply SiLU or tanh-approximate GELU to gate, multiply up, then round once."""
    _check_input(gate, backend)
    if up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device:
        raise ValueError("gate and up must have identical shape, dtype and device")
    if activation not in ("silu", "gelu_tanh"):
        raise ValueError("activation must be silu or gelu_tanh")
    if backend == "hip":
        return torch.ops.trtllm_rdna4.gated_activation(
            gate.contiguous(),
            up.contiguous(),
            0 if activation == "silu" else 1,
        )
    activated = (
        F.silu(gate.float()) if activation == "silu" else F.gelu(gate.float(), approximate="tanh")
    )
    return (activated * up.float()).to(gate.dtype)


def rotary_embedding(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    backend: KernelBackend = "torch",
) -> torch.Tensor:
    """Rotate [tokens, heads, dim] with cos/sin [tokens, rotary_dim / 2].

    Supports partial rotary dimensions and either split-half or interleaved pairs.
    Angles are FP32 on the same device. Unrotated tail elements are copied exactly.
    """
    _check_input(input, backend)
    if input.ndim != 3 or cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != input.shape[0]:
        raise ValueError("Expected input [tokens, heads, dim] and cos/sin [tokens, pairs]")
    if (
        cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or cos.device != input.device
        or sin.device != input.device
    ):
        raise ValueError("cos/sin must be FP32 on the input device")
    pairs = cos.shape[-1]
    if pairs < 1 or 2 * pairs > input.shape[-1]:
        raise ValueError("rotary_dim must be positive, even and no larger than head dim")
    if backend == "hip":
        return torch.ops.trtllm_rdna4.rotary(
            input.contiguous(),
            cos.contiguous(),
            sin.contiguous(),
            interleaved,
        )
    values = input.float()
    c, s = cos[:, None, :], sin[:, None, :]
    rotated = values[..., : 2 * pairs]
    if interleaved:
        first, second = rotated[..., 0::2], rotated[..., 1::2]
        result = torch.stack((first * c - second * s, second * c + first * s), dim=-1).flatten(-2)
    else:
        first, second = rotated[..., :pairs], rotated[..., pairs:]
        result = torch.cat((first * c - second * s, second * c + first * s), dim=-1)
    return torch.cat((result.to(input.dtype), input[..., 2 * pairs :]), dim=-1)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    query_start: int | None = None,
    scale: float | None = None,
    backend: KernelBackend = "torch",
) -> torch.Tensor:
    """SDPA/GQA on [batch, heads, sequence, dim], with explicit cached-decode alignment.

    Boolean masks use True=allowed; floating masks are additive logits biases.
    Masks must broadcast to [batch, query_heads, query_sequence, key_sequence].
    Causal query_start defaults to key_sequence-query_sequence (right-aligned KV
    cache). Fully masked rows return zeros. Native HIP supports head dim <= 256.
    """
    _check_input(query, backend)
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Expected 4D query/key and identically shaped key/value")
    batch, heads, query_len, dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != dim or key.shape[1] < 1 or heads % key.shape[1]:
        raise ValueError("Invalid batch, head dimension or GQA head ratio")
    if key.shape[2] < 1 or heads < 1:
        raise ValueError("Attention requires nonempty key sequence and heads")
    if (
        key.dtype != query.dtype
        or value.dtype != query.dtype
        or key.device != query.device
        or value.device != query.device
    ):
        raise ValueError("query/key/value must share dtype and device")
    start = (key.shape[2] - query_len if causal else 0) if query_start is None else query_start
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or not 0 <= start <= torch.iinfo(torch.int64).max - query_len
    ):
        raise ValueError("query_start must be a non-negative int64 position without overflow")
    factor = 1 / math.sqrt(dim) if scale is None else scale
    if not math.isfinite(factor) or abs(factor) > torch.finfo(torch.float32).max:
        raise ValueError("scale must be finite in FP32")
    shape = (batch, heads, query_len, key.shape[2])
    bias = None
    if mask is not None:
        if mask.device != query.device or (
            mask.dtype != torch.bool and not mask.is_floating_point()
        ):
            raise ValueError("mask must be boolean or floating point on the query device")
        broadcast = torch.broadcast_to(mask, shape)
        bias = (
            torch.where(broadcast, 0.0, float("-inf"))
            if mask.dtype == torch.bool
            else broadcast.float()
        )
    if backend == "hip":
        if dim > 256:
            raise ValueError(
                "Native reference attention supports head dim <= 256; use SDPA for larger heads"
            )
        return torch.ops.trtllm_rdna4.attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bias.contiguous() if bias is not None else None,
            causal,
            start,
            factor,
        )
    if causal:
        allowed = torch.arange(key.shape[2], device=query.device)[None, :] <= (
            torch.arange(query_len, device=query.device)[:, None] + start
        )
        causal_bias = torch.where(allowed, 0.0, float("-inf"))
        bias = causal_bias if bias is None else bias + causal_bias
    ratio = heads // key.shape[1]
    if ratio != 1:
        key, value = key.repeat_interleave(ratio, dim=1), value.repeat_interleave(ratio, dim=1)
    # Use explicit FP32 math for the reference path and deterministic all-masked rows.
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * factor
    if bias is not None:
        scores = scores + bias
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    probabilities = torch.where(fully_masked, 0.0, torch.softmax(scores, dim=-1))
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def _native_rms_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return rms_norm(hidden_states, weight, epsilon, backend="hip")


def apply_native_norms(model: torch.nn.Module) -> int:
    """Replace known, compatible normalization forwards for HIP inference only."""
    replacements = 0
    compatible_rms = {
        "LlamaRMSNorm",
        "MistralRMSNorm",
        "Qwen2RMSNorm",
        "Qwen3RMSNorm",
        "Phi3RMSNorm",
    }
    for module in model.modules():
        if type(module).__name__ in compatible_rms:
            module.forward = partial(
                _native_rms_forward,
                weight=module.weight,
                epsilon=module.variance_epsilon,
            )
            replacements += 1
        elif (
            isinstance(module, torch.nn.LayerNorm)
            and len(module.normalized_shape) == 1
            and module.weight is not None
        ):
            module.forward = partial(
                layer_norm,
                weight=module.weight,
                bias=module.bias,
                epsilon=module.eps,
                backend="hip",
            )
            replacements += 1
    return replacements


__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "layer_norm",
    "gated_activation",
    "rotary_embedding",
    "attention",
    "apply_native_norms",
]
