# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for FlashInfer sparse MLA."""

from collections.abc import Callable
from typing import Optional

import torch

# These values mirror the split-K contract of FlashInfer's private SM120 kernel.
# The import below is intentionally private and is coupled to the
# flashinfer-python version pinned in requirements.txt.
SPARSE_MLA_SPLIT_KV_TILE = 64
SPARSE_MLA_SPLIT_Q_THRESHOLD = 64


def get_sparse_mla_op() -> Callable[..., None]:
    """Return the pinned FlashInfer SM120 sparse-MLA operator."""
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention


def allocate_sparse_mla_split_workspace(
    *,
    num_tokens: int,
    num_heads: int,
    num_splits: int,
    value_dim: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Allocate split-K scratch required by FlashInfer's private SM120 kernel.

    For at most 64 query tokens, returns ``mid_out`` with shape
    ``[num_tokens, num_heads, num_splits, value_dim]`` and dtype bfloat16,
    plus ``mid_lse`` with shape ``[num_tokens, num_heads, num_splits]`` and
    dtype float32. For more than 64 query tokens, the kernel does not use the
    split-K path and this function returns ``(None, None)``.
    """
    if num_tokens > SPARSE_MLA_SPLIT_Q_THRESHOLD:
        return None, None

    mid_out = torch.empty(
        num_tokens,
        num_heads,
        num_splits,
        value_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, num_splits),
        dtype=torch.float32,
        device=device,
    )
    return mid_out, mid_lse
