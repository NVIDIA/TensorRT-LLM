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
"""FlashInfer Prims-TS FP8-rowwise GEMMs with fused epilogues, as custom ops for model adapters."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, Linear

from .common import config_for, device_supported, token_positions


@torch.library.custom_op("trtllm::prims_ts_fp8_linear_swiglu", mutates_args=(), device_types="cuda")
def _fp8_linear_swiglu(
    data: Tensor, weight: Tensor, scales: Tensor, weight_scale: Tensor
) -> Tensor:
    """[M, K] e4m3 x [N, K] e4m3 [up, gate] row pairs with per-token and per-channel FP32
    scales, then silu(gate) * up: BF16 [M, N / 2]."""
    from flashinfer.prims_ts import gemm

    m, k, n = data.shape[0], data.shape[1], weight.shape[0]
    config = config_for(data.device, "fp8_e4m3", "swiglu", m, k, n)
    # Layer parameters require grad, which DLPack export rejects.
    return gemm.fp8_linear_swiglu(
        data, weight.detach(), scales, weight_scale.detach(), config=config
    )


@_fp8_linear_swiglu.register_fake
def _fp8_linear_swiglu_fake(data, weight, scales, weight_scale):
    return data.new_empty((data.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::prims_ts_fp8_qkv_qknorm_rope", mutates_args=(), device_types="cuda"
)
def _fp8_qkv_qknorm_rope(
    data: Tensor,
    weight: Tensor,
    scales: Tensor,
    weight_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
) -> Tensor:
    """Packed q|k|v GEMM with per-head RMSNorm (eps 1e-6) and interleaved RoPE.

    ``cos_sin`` [M, head_dim] holds half cos | half sin per token. The output is packed
    BF16 [M, N] (q | k | v along N).
    """
    from flashinfer.prims_ts import gemm

    m, k, n = data.shape[0], data.shape[1], weight.shape[0]
    head_dim = q_norm.numel()
    heads = n // (3 * head_dim)
    return gemm.fp8_qkv_qknorm_rope(
        data,
        weight.detach(),
        scales,
        weight_scale.detach(),
        q_norm.detach(),
        k_norm.detach(),
        cos_sin,
        token_positions(m, data.device),
        num_q_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        is_neox=False,
        config=config_for(data.device, "fp8_e4m3", "qkv_qknorm_rope", m, k, n, head_dim),
    )


@_fp8_qkv_qknorm_rope.register_fake
def _fp8_qkv_qknorm_rope_fake(data, weight, scales, weight_scale, q_norm, k_norm, cos_sin):
    return data.new_empty((data.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def quantize_per_token(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Per-token FP8 of ``x`` [M, K] with TRT-LLM's quantizer, as FP8-rowwise Linears use."""
    if FP8RowwiseLinearMethod.use_tunable_quantize:
        data, scales = torch.ops.trtllm.tunable_fp8_per_token_quant(x)
    else:
        data, scales = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)
    return data, scales.reshape(-1)


def fp8_gemm_eligible(layer: nn.Module) -> bool:
    """Whether a loaded FP8-rowwise Linear has a shape and layout the Prims-TS GEMM takes."""
    return (
        isinstance(layer, Linear)
        and isinstance(layer.quant_method, FP8RowwiseLinearMethod)
        and layer.bias is None
        and layer.weight.dtype == torch.float8_e4m3fn
        and layer.weight.shape[1] % 128 == 0
        and layer.weight.shape[0] % 512 == 0
        and tuple(layer.weight_scale.shape) == (layer.weight.shape[0],)
        and layer.dtype in (None, torch.bfloat16)
        and device_supported(layer.weight.device)
    )


__all__ = ["fp8_gemm_eligible", "quantize_per_token"]
