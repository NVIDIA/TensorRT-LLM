# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi K3 latent-MoE tail (TP decode): AllReduce + RMSNorm + shared
ReduceScatter, sharded up-projection with a multicast all-gather epilogue."""

from .latent_moe_tail import (
    MAX_NUM_TOKENS,
    KimiK3LatentMoETail,
    fused_tail_finalize_requested,
    fused_tail_max_tokens,
    fused_tail_unsupported_reason,
    math_reference,
)
from .routed_output import UnfinalizedRoutedOutput

__all__ = [
    "KimiK3LatentMoETail",
    "MAX_NUM_TOKENS",
    "UnfinalizedRoutedOutput",
    "fused_tail_finalize_requested",
    "fused_tail_max_tokens",
    "fused_tail_unsupported_reason",
    "math_reference",
]
