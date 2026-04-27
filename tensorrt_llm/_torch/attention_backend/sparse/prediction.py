# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common sparse prediction orchestration for attention backends."""

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

from ..interface import AttentionForwardArgs, AttentionMetadata, SparsePrediction

if TYPE_CHECKING:
    from ..trtllm import TrtllmAttention


def prepare_sparse_prediction(
    backend: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    metadata: AttentionMetadata,
    forward_args: AttentionForwardArgs,
) -> SparsePrediction:
    """Run optional algorithm predictors and build the attention-op payload."""
    if backend.sparse_params is None:
        return forward_args.sparse_prediction

    kv_indices, kv_offsets = backend.sparse_kv_predict(q, k, metadata, forward_args)
    attn_indices, attn_offsets = backend.sparse_attn_predict(q, k, metadata, forward_args)
    block_size = (
        backend.sparse_params.indices_block_size
        if attn_indices is not None or attn_offsets is not None
        else forward_args.sparse_prediction.sparse_attn_indices_block_size
    )
    return replace(
        forward_args.sparse_prediction,
        sparse_kv_indices=kv_indices,
        sparse_kv_offsets=kv_offsets,
        sparse_attn_indices=attn_indices,
        sparse_attn_offsets=attn_offsets,
        sparse_attn_indices_block_size=block_size,
    )
