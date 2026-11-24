# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) for the Kimi K3 (k3_rubin) text model.

Local copy of ``tensorrt_llm/_torch/modules/kimi_kda`` with the K3 Rubin
features (Lamport all-reduce push, MXFP8 hand-off, fused gate projections,
verify-kernel variants, L2 prefetch) merged onto main's module. The
engine-owned cache manager stays main's ``modules/kimi_kda/cache_manager.py``.

KDA is the linear-attention block used at ``linear_attn_config.kda_layers``
positions in the Kimi K3 text-core. It carries a short-convolution state and a
delta-rule recurrent state per layer, so it follows the hybrid-cache /
mamba ownership pattern rather than the paged-KV FMHA attention-backend
interface.
"""

from .kimi_kda_mixer import KimiKDALinearAttention

__all__ = ["KimiKDALinearAttention"]
