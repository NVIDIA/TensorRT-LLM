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

from enum import Enum

DEEPSEEK_V4_SPARSE_RATIO = 4
DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO = 4


class DeepseekV4AttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6


def is_overlap_compressor(compress_ratio: int) -> bool:
    return compress_ratio == DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO


def is_sparse_layer(compress_ratio: int) -> bool:
    return compress_ratio == DEEPSEEK_V4_SPARSE_RATIO


def is_compress_layer(compress_ratio: int) -> bool:
    return compress_ratio > 1


def compress_ratio_has_attention(compress_ratio: int, attn_type: DeepseekV4AttentionType) -> bool:
    is_sparse = is_sparse_layer(compress_ratio)
    is_compress = is_compress_layer(compress_ratio)

    if attn_type == DeepseekV4AttentionType.SWA:
        return True
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return is_compress
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_STATE:
        return is_compress
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return is_compress
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return is_sparse
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE:
        return is_sparse
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return is_sparse
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


def get_attn_dim(
    head_dim: int, index_head_dim: int, compress_ratio: int, attn_type: DeepseekV4AttentionType
) -> int:
    state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
    if attn_type == DeepseekV4AttentionType.SWA:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_STATE:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE:
        return state_factor * index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return state_factor * index_head_dim
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


def get_token_bytes(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
) -> int:
    if not compress_ratio_has_attention(compress_ratio, attn_type):
        raise ValueError(
            f"Layer with compress ratio {compress_ratio} does not have attention type {attn_type}"
        )

    attn_dim = get_attn_dim(head_dim, index_head_dim, compress_ratio, attn_type)

    dtype_bytes = 1 if has_fp8_kv_cache else 2
    if attn_type in [
        DeepseekV4AttentionType.COMPRESSOR_STATE,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
    ]:
        dtype_bytes = 4

    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        if indexer_k_dtype == "fp8":
            return attn_dim + index_head_dim // 128 * 4
        if indexer_k_dtype == "fp4":
            return index_head_dim // 2 + index_head_dim // 32
        raise ValueError(
            f"Unsupported indexer_k_dtype {indexer_k_dtype!r}; expected 'fp8' or 'fp4'."
        )

    return attn_dim * dtype_bytes
