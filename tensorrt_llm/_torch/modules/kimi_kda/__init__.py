# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) in-tree module for TensorRT-LLM's PyTorch backend.

KDA is the linear-attention block used at ``linear_attn_config.kda_layers``
positions in the Kimi K3 text-core. It carries a short-convolution state and a
delta-rule recurrent state per layer, so it follows the hybrid-cache /
mamba ownership pattern rather than the paged-KV FMHA attention-backend
interface.
"""

from .kimi_kda_mixer import KimiKDAKernelPath, KimiKDALinearAttention

__all__ = [
    "KimiKDAKernelPath",
    "KimiKDALinearAttention",
]
