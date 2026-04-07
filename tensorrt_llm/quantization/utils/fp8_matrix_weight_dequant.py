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

# Pure PyTorch helpers to expand block FP8 weight scales and dequantize ``[N, K]``
# matrices. Used by AutoDeploy load hooks and FineGrained FP8 linear fallbacks.
# Intentionally avoids Triton so model import paths stay lightweight.

import math

import torch


def dequant_fp8_weight_two_dim_block_grid(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights using a 2-D block scale grid ``[ceil(N/bn), ceil(K/bk)]``.

    Matches FineGrained 128x128 checkpoints and the BF16 fallback in
    ``trtllm_finegrained_fp8_linear``.
    """
    N, K = weight_fp8.shape
    scale_n, scale_k = weight_scale.shape
    # Ceil division so the expanded scale covers non-divisible dims (see torch_quant).
    actual_block_n = math.ceil(N / scale_n) if scale_n > 0 else block_n
    actual_block_k = math.ceil(K / scale_k) if scale_k > 0 else block_k
    scale_expanded = weight_scale.repeat_interleave(actual_block_n, dim=0).repeat_interleave(
        actual_block_k, dim=1
    )
    scale_expanded = scale_expanded[:N, :K]
    return weight_fp8.to(dtype) * scale_expanded.to(dtype)


def dequant_fp8_nk_weight_auto_scale_layout(
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_k: int = 128,
) -> torch.Tensor:
    """Dequantize ``[N, K]`` FP8 weights from common checkpoint scale layouts.

    Supports:

    * FineGrained FP8 128x128 blocks: ``scale`` is ``[ceil(N/128), ceil(K/128)]``.
    * 1x128 along K (``triton_fp8_quantize_1x128`` on ``[N, K]``): ``[ceil(K/128), N]``.
    * Same 1x128 layout stored transposed: ``[N, ceil(K/128)]``.

    Raises:
        ValueError: If ``weight_scale_inv`` shape does not match any supported layout.
    """
    N, K = weight_fp8.shape
    num_k_chunks = (K + block_k - 1) // block_k
    num_n_chunks = (N + block_k - 1) // block_k
    sn, sk = weight_scale_inv.shape
    scale = weight_scale_inv.to(dtype)

    if sn == num_k_chunks and sk == N:
        scale_expanded = scale.transpose(0, 1).repeat_interleave(block_k, dim=1)[:, :K]
    elif sn == N and sk == num_k_chunks:
        scale_expanded = scale.repeat_interleave(block_k, dim=1)[:, :K]
    elif sn == num_n_chunks and sk == num_k_chunks:
        eff_bn = math.ceil(N / sn) if sn > 0 else block_k
        eff_bk = math.ceil(K / sk) if sk > 0 else block_k
        scale_expanded = scale.repeat_interleave(eff_bn, dim=0).repeat_interleave(eff_bk, dim=1)[
            :N, :K
        ]
    else:
        raise ValueError(
            f"weight_scale_inv shape {tuple(weight_scale_inv.shape)} does not match "
            f"any supported layout for weight {tuple(weight_fp8.shape)}: "
            f"1x128 [ceil(K/128), N] (= [{num_k_chunks}, {N}]), "
            f"transposed [N, ceil(K/128)] (= [{N}, {num_k_chunks}]), "
            f"or 128x128 [{num_n_chunks}, {num_k_chunks}]."
        )

    return weight_fp8.to(dtype) * scale_expanded
