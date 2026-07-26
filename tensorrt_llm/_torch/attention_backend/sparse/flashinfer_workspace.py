# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph-safe workspace buffers for FlashInfer sparse MLA."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..interface import AttentionMetadata


def get_sparse_mla_workspace(
    metadata: "AttentionMetadata",
    device: torch.device,
    num_tokens: int,
    num_heads: int,
    num_splits: int,
    head_dim: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if metadata.is_cuda_graph:
        max_tokens = metadata.max_num_requests
        if num_tokens > max_tokens:
            raise RuntimeError(
                f"Sparse MLA CUDA graph has {num_tokens} tokens, but its "
                f"workspace holds at most {max_tokens}."
            )
        buffers = metadata.cuda_graph_buffers
        out_lse = metadata.get_empty(
            buffers,
            (max_tokens, num_heads),
            dtype=torch.float32,
            cache_name=f"flashinfer_sparse_mla_out_lse_{layer_idx}",
            capture_graph=True,
        )[:num_tokens]
        mid_out = metadata.get_empty(
            buffers,
            (max_tokens, num_heads, num_splits, head_dim),
            dtype=torch.bfloat16,
            cache_name=f"flashinfer_sparse_mla_mid_out_{layer_idx}",
            capture_graph=True,
        )[:num_tokens]
        mid_lse = metadata.get_empty(
            buffers,
            (max_tokens, num_heads, num_splits),
            dtype=torch.float32,
            cache_name=f"flashinfer_sparse_mla_mid_lse_{layer_idx}",
            capture_graph=True,
        )[:num_tokens]
    else:
        out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
        mid_out = torch.empty(
            (num_tokens, num_heads, num_splits, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        mid_lse = torch.empty(
            (num_tokens, num_heads, num_splits),
            dtype=torch.float32,
            device=device,
        )

    out_lse.zero_()
    mid_out.zero_()
    mid_lse.fill_(float("-inf"))
    return out_lse, mid_out, mid_lse
