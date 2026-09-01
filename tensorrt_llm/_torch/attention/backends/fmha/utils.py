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

import math
from collections.abc import MutableMapping
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.internal import thop

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _get_kv_page_offset(
    attn: "TrtllmAttention",
    metadata: "TrtllmAttentionMetadata",
    seq_offset: int,
    *,
    cache: Optional[MutableMapping[tuple[int, int], int]] = None,
) -> Optional[int]:
    """Return the V-page displacement relative to a K page ID."""
    manager = metadata.kv_cache_manager
    local_layer_idx = attn.local_layer_idx
    if local_layer_idx is None:
        local_layer_idx = int(attn.get_local_layer_idx(metadata))
    pool_mapping = metadata.host_kv_cache_pool_mapping
    pool_index = int(pool_mapping[local_layer_idx, 0])
    cache_key = (id(manager), pool_index)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    if isinstance(manager, KVCacheManagerV2):
        kv_offsets = manager.kv_offset
        kv_offset = int(kv_offsets[pool_index])
        if kv_offset > 0:
            if cache is not None:
                cache[cache_key] = kv_offset
            return kv_offset
        return None

    if isinstance(manager, KVCacheManager):
        host_block_offsets = manager.host_kv_cache_block_offsets
        if host_block_offsets is None or host_block_offsets.ndim != 4:
            return None
        if pool_index >= host_block_offsets.shape[0]:
            return None

        rows = host_block_offsets[pool_index]
        if 0 <= seq_offset < rows.shape[0]:
            row_deltas = rows[seq_offset, 1] - rows[seq_offset, 0]
            positive = row_deltas[row_deltas > 0]
            if positive.numel() > 0:
                kv_offset = int(positive[0])
                if cache is not None:
                    cache[cache_key] = kv_offset
                return kv_offset
        all_deltas = rows[:, 1] - rows[:, 0]
        positive = all_deltas[all_deltas > 0]
        if positive.numel() == 0:
            return None
        kv_offset = int(positive[0])
        if cache is not None:
            cache[cache_key] = kv_offset
        return kv_offset

    raise TypeError(f"Unsupported KV cache manager: {type(manager).__name__}.")


@lru_cache(maxsize=128)
def get_trtllm_gen_context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    """Return the fused context-preprocessing workspace size in bytes."""
    if max_num_tokens == 0:
        return 0
    layout = thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        fp8_context_fmha,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=None)
def get_multi_processor_count_for_device(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def get_bmm1_scale(attn: "TrtllmAttention") -> float:
    return 1.0 / (math.sqrt(attn.head_dim) * attn.q_scaling)


def get_attention_chunk_size(attn: "TrtllmAttention") -> int:
    return attn.attention_chunk_size if attn.attention_chunk_size is not None else 0
