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

"""Fused (Add +) RMSNorm + NVFP4 quantization wrappers.

Wraps the C++ kernel ``trtllm.fused_add_rms_norm_quant`` (NVFP4 mode) with
two thin custom ops that:

* Always produce a high-precision (BF16/FP16) normed output so that
  non-quantised consumers can be rewired without an extra norm.
* Convert the int32-packed FP4 output to uint8 (``view(torch.uint8)``).
* Flatten arbitrary leading dims to 2-D before calling the C++ kernel
  (which requires ``[M, N]``) and reshape outputs back.

Two variants are provided:

``trtllm_rms_norm_quant_nvfp4``
    Direct norm + quant (no residual add).  A zero residual is synthesised
    internally so the same C++ kernel can be reused.

``trtllm_fused_add_rms_norm_quant_nvfp4``
    Fused add + norm + quant — the common case after ``fuse_add_rms_norm``
    has already run.
"""

from typing import Tuple

import torch


def _sf_size(m: int, n: int) -> int:
    """Compute the swizzled scale-factor buffer size (matches cpp_custom_ops fake)."""
    sf_vec_size = 16
    return ((m + 127) // 128) * 128 * ((n // sf_vec_size + 3) // 4) * 4


# ---------------------------------------------------------------------------
# Direct norm + quant (no add)
# ---------------------------------------------------------------------------
@torch.library.custom_op("auto_deploy::trtllm_rms_norm_quant_nvfp4", mutates_args=())
def trtllm_rms_norm_quant_nvfp4(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + NVFP4 quantization (no residual add).

    Args:
        input: [..., N] tensor (bf16/fp16).
        weight: [N] RMSNorm gamma.
        eps: Layernorm epsilon.
        sf_scale: Global FP4 scale (scalar, float32).

    Returns:
        (bf16_normed, fp4_u8, sf_out)
        - bf16_normed: [..., N] high-precision normed output.
        - fp4_u8: [M, N/2] uint8 packed FP4 quantised output (2-D).
        - sf_out: [sf_size] uint8 swizzled block scale factors.
    """
    orig_shape = input.shape
    n = weight.shape[0]
    input_2d = input.reshape(-1, n)
    zeros = torch.zeros_like(input_2d)

    normed_i32, _residual_out, sf_out, hp_normed = torch.ops.trtllm.fused_add_rms_norm_quant(
        input_2d, zeros, weight, sf_scale, True, eps=eps, output_hp_norm=True
    )
    fp4_u8 = normed_i32.view(torch.uint8)
    return hp_normed.reshape(orig_shape), fp4_u8, sf_out


@trtllm_rms_norm_quant_nvfp4.register_fake
def _rms_norm_quant_nvfp4_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    m = input.numel() // n
    bf16_out = torch.empty_like(input)
    fp4_u8 = input.new_empty((m, n // 2), dtype=torch.uint8)
    sf_out = input.new_empty((_sf_size(m, n),), dtype=torch.uint8)
    return bf16_out, fp4_u8, sf_out


# ---------------------------------------------------------------------------
# Fused add + norm + quant
# ---------------------------------------------------------------------------
@torch.library.custom_op("auto_deploy::trtllm_fused_add_rms_norm_quant_nvfp4", mutates_args=())
def trtllm_fused_add_rms_norm_quant_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Add + RMSNorm + NVFP4 quantization.

    Computes ``add_out = x + residual``, ``norm_out = rms_norm(add_out)``,
    then NVFP4 block-quantises ``norm_out``.

    Args:
        x: [..., N] first input (bf16/fp16).
        residual: [..., N] second input (residual).
        weight: [N] RMSNorm gamma.
        eps: Layernorm epsilon.
        sf_scale: Global FP4 scale (scalar, float32).

    Returns:
        (bf16_normed, fp4_u8, sf_out, add_out)
    """
    orig_shape = x.shape
    n = weight.shape[0]
    x_2d = x.reshape(-1, n)
    residual_2d = residual.reshape(-1, n)

    normed_i32, add_out_2d, sf_out, hp_normed = torch.ops.trtllm.fused_add_rms_norm_quant(
        x_2d, residual_2d, weight, sf_scale, True, eps=eps, output_hp_norm=True
    )
    fp4_u8 = normed_i32.view(torch.uint8)
    return hp_normed.reshape(orig_shape), fp4_u8, sf_out, add_out_2d.reshape(orig_shape)


@trtllm_fused_add_rms_norm_quant_nvfp4.register_fake
def _fused_add_rms_norm_quant_nvfp4_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    m = x.numel() // n
    bf16_out = torch.empty_like(x)
    fp4_u8 = x.new_empty((m, n // 2), dtype=torch.uint8)
    sf_out = x.new_empty((_sf_size(m, n),), dtype=torch.uint8)
    add_out = torch.empty_like(x)
    return bf16_out, fp4_u8, sf_out, add_out
