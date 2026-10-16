# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's decode-step embedding and its first layer's input RMSNorm in one CuTe DSL launch: the rows ``table[ids]``
written into a caller's buffer, and their RMSNorm."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_embed.op  # noqa: F401 — registers the op


def k3_embed_norm(
    ids: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    raw: torch.Tensor,
) -> torch.Tensor:
    """Write ``table[ids]`` into ``raw`` (zero rows for ids outside ``[0, V)``) and return
    ``raw * rsqrt(mean(raw^2) + eps) * weight`` as a new bf16 tensor, in one k3_embed_norm call."""
    return torch.ops.trtllm.k3_embed_norm(ids, table, weight, eps, raw)
