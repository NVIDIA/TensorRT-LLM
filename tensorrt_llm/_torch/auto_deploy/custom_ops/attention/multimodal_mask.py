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

"""Model-owned multimodal semantic mask ops and cached mask preparation helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _blob_ids_from_spans(
    seq_len: int,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build contiguous blob ids from multimodal span positions and lengths."""
    blob_ids = torch.zeros(seq_len, dtype=torch.int64, device=device)
    next_blob_id = 1
    for position, length in zip(positions.tolist(), lengths.tolist(), strict=True):
        if length <= 0:
            continue
        start = max(int(position), 0)
        end = min(start + int(length), seq_len)
        if start >= end:
            continue
        blob_ids[start:end] = next_blob_id
        next_blob_id += 1
    return blob_ids


def _causal_or_bidirectional_mask(blob_ids: torch.Tensor) -> torch.Tensor:
    """Build a bool allow-mask from blob ids."""
    seq_len = blob_ids.shape[0]
    causal = torch.tril(torch.ones(seq_len, seq_len, device=blob_ids.device, dtype=torch.bool))
    q_blob = blob_ids.unsqueeze(1)
    kv_blob = blob_ids.unsqueeze(0)
    bidirectional = (q_blob == kv_blob) & (q_blob != 0)
    return causal | bidirectional


@torch.library.custom_op("auto_deploy::gemma4_multimodal_mask", mutates_args=())
def gemma4_multimodal_mask(
    input_ids: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a dense source-graph attention mask from multimodal spans."""
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    batch_size, seq_len = input_ids.shape[:2]
    masks = []
    for batch_idx in range(batch_size):
        span_start = int(mm_item_cu_seqlen[batch_idx].item())
        span_end = int(mm_item_cu_seqlen[batch_idx + 1].item())
        blob_ids = _blob_ids_from_spans(
            seq_len,
            mm_token_positions[span_start:span_end],
            mm_token_lengths[span_start:span_end],
            input_ids.device,
        )
        masks.append(_causal_or_bidirectional_mask(blob_ids))
    return torch.stack(masks, dim=0).unsqueeze(1)


@gemma4_multimodal_mask.register_fake
def gemma4_multimodal_mask_fake(
    input_ids: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del mm_token_positions, mm_token_lengths, mm_item_cu_seqlen
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    batch_size, seq_len = input_ids.shape[:2]
    return torch.empty(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=input_ids.device)


@torch.library.custom_op("auto_deploy::gemma4_prepare_multimodal_mask", mutates_args=())
def gemma4_prepare_multimodal_mask(
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    input_pos: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build backend-consumable prefill masks from multimodal spans and batch geometry."""
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    num_prefill = int(batch_info_host[0].item())
    if num_prefill == 0 or mm_token_positions.numel() == 0:
        return torch.empty(0, 1, 0, 0, dtype=torch.bool, device=input_pos.device)

    device = mm_token_positions.device
    masks = []
    max_q = 0
    max_kv = 0
    for batch_idx in range(num_prefill):
        q_start = int(input_pos[batch_idx].item())
        q_len = int(cu_seqlen[batch_idx + 1].item()) - int(cu_seqlen[batch_idx].item())
        kv_len = q_start + q_len

        span_start = int(mm_item_cu_seqlen[batch_idx].item())
        span_end = int(mm_item_cu_seqlen[batch_idx + 1].item())
        blob_ids = _blob_ids_from_spans(
            kv_len,
            mm_token_positions[span_start:span_end],
            mm_token_lengths[span_start:span_end],
            device,
        )

        q_blob = blob_ids[q_start : q_start + q_len].unsqueeze(1)
        kv_blob = blob_ids.unsqueeze(0)
        bidirectional = (q_blob == kv_blob) & (q_blob != 0)

        q_pos = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        causal = kv_pos <= q_pos

        mask = (causal | bidirectional).unsqueeze(0)
        masks.append(mask)
        max_q = max(max_q, q_len)
        max_kv = max(max_kv, kv_len)

    padded = []
    for mask in masks:
        _, q_len, kv_len = mask.shape
        padded.append(F.pad(mask, (0, max_kv - kv_len, 0, max_q - q_len), value=False))
    return torch.stack(padded, dim=0)


@gemma4_prepare_multimodal_mask.register_fake
def gemma4_prepare_multimodal_mask_fake(
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    input_pos: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del batch_info_host, cu_seqlen, mm_token_lengths, mm_item_cu_seqlen
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    num_seq = input_pos.shape[0]
    max_tokens = mm_token_positions.shape[0]
    return torch.empty(
        num_seq, 1, max_tokens, max_tokens, dtype=torch.bool, device=input_pos.device
    )
