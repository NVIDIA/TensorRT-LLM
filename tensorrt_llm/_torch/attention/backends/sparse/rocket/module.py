# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RocketKV module-layer integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.logger import logger

from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.attention import Attention


def initialize_sparse_attn(self) -> None:
    """Configure the Attention module for RocketKV."""
    logger.warning_once(
        "disable rope_fusion for RocketKV.",
        key="disable_rope_fusion_for_rocketkv",
    )
    self.rope_fusion = False


class RocketKVHooks(AttentionSparseHooks):
    """Typed RocketKV adapter for the shared Attention module."""

    def initialize(self, attention: Attention) -> None:
        initialize_sparse_attn(attention)


register_attention_sparse_hooks("rocket", RocketKVHooks)
