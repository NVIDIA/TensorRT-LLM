# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Indexer planning helpers of the Rubin targets' DSA metadata: family copies of the engine DSA indexer's private
helpers (``attention/backends/sparse/dsa/indexer.py``), with the same arithmetic, so the family plans its steps
exactly as the engine's indexer would. Not an OpWrapper: host and device index arithmetic only."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping


def compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flat byte indices of the index keys' FP8/FP4 data and scales from global token positions.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (logical element count).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache (default: head_dim, one byte per FP8
            value; ``head_dim // 2`` for packed E2M1 codes). The scale layout is 4 bytes per token at head_dim 128
            (one float32 for FP8, four packed UE8M0 exponents for FP4).

    Returns:
        (data_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    if data_bytes_per_token is None:
        data_bytes_per_token = head_dim
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings during CUDA graph capture/replay.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size
    return fp8_indices, scale_indices


def select_indexer_compress_ratio(compress_ratios: List[int]) -> int:
    """The compress ratio the indexer keys use: 4 when a layer compresses by 4, else 1 when one does not, else 0."""
    if 4 in compress_ratios:
        return 4
    if 1 in compress_ratios:
        return 1
    return 0


def effective_compress_ratio_divisor(compress_ratio: int) -> int:
    """The divisor of key positions for ``compress_ratio`` (1 for no compression)."""
    return compress_ratio if compress_ratio > 1 else 1


def split_prefill_queries(
    count: int, mapping: Optional[Mapping], threshold: int
) -> Tuple[int, int, Optional[List[int]]]:
    """Local query bounds and, for a TP prefill gather, every rank's query count."""
    if (
        threshold < 0
        or count < threshold
        or mapping is None
        or mapping.enable_attention_dp
        or mapping.tp_size <= 1
    ):
        return 0, count, None
    size, rank = mapping.tp_size, mapping.tp_rank
    sizes = [(r + 1) * count // size - r * count // size for r in range(size)]
    return count * rank // size, count * (rank + 1) // size, sizes


__all__ = [
    "compute_slot_mappings",
    "effective_compress_ratio_divisor",
    "select_indexer_compress_ratio",
    "split_prefill_queries",
]
