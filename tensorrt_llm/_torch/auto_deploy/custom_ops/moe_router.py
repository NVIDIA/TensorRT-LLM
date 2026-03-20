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

"""Custom ops for MoE routing: softmax + top-k selection + optional normalization.

This module provides the canonical torch reference op and the Triton backend op for
the MoE router computation. The router takes raw logits from a gate linear layer and
produces top-k expert indices and their normalized routing weights.

The computation is:
    probs = softmax(logits, dim=-1)  # in float32
    topk_weights, topk_indices = topk(probs, k=top_k)
    if normalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from .triton_kernels.moe_router import moe_router as _triton_moe_router


@torch.library.custom_op("auto_deploy::torch_moe_router", mutates_args=())
def torch_moe_router(
    logits: torch.Tensor,
    top_k: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical PyTorch reference for MoE routing.

    Computes softmax over expert logits, selects top-k experts, and optionally
    normalizes the routing weights to sum to 1.

    Args:
        logits: Router logits of shape (M, E) where M is the number of tokens
            and E is the number of experts. Can be any float dtype.
        top_k: Number of top experts to select per token.
        normalize: Whether to normalize the top-k weights to sum to 1.

    Returns:
        A tuple (topk_weights, topk_indices) where:
        - topk_weights: float32 tensor of shape (M, top_k) with selected routing weights
        - topk_indices: int64 tensor of shape (M, top_k) with selected expert indices
    """
    routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    if normalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


@torch_moe_router.register_fake
def torch_moe_router_fake(
    logits: torch.Tensor,
    top_k: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.export tracing."""
    M = logits.shape[0]
    topk_weights = torch.empty((M, top_k), dtype=torch.float32, device=logits.device)
    topk_indices = torch.empty((M, top_k), dtype=torch.int64, device=logits.device)
    return topk_weights, topk_indices


@torch.library.custom_op("auto_deploy::triton_moe_router", mutates_args=())
def triton_moe_router(
    logits: torch.Tensor,
    top_k: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton backend for MoE routing.

    Fuses softmax + top-k selection + optional normalization into a single Triton kernel.

    Args:
        logits: Router logits of shape (M, E) where M is the number of tokens
            and E is the number of experts. Can be any float dtype.
        top_k: Number of top experts to select per token.
        normalize: Whether to normalize the top-k weights to sum to 1.

    Returns:
        A tuple (topk_weights, topk_indices) where:
        - topk_weights: float32 tensor of shape (M, top_k) with selected routing weights
        - topk_indices: int64 tensor of shape (M, top_k) with selected expert indices
    """
    return _triton_moe_router(logits, top_k, normalize)


@triton_moe_router.register_fake
def triton_moe_router_fake(
    logits: torch.Tensor,
    top_k: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.export tracing."""
    M = logits.shape[0]
    topk_weights = torch.empty((M, top_k), dtype=torch.float32, device=logits.device)
    topk_indices = torch.empty((M, top_k), dtype=torch.int64, device=logits.device)
    return topk_weights, topk_indices
