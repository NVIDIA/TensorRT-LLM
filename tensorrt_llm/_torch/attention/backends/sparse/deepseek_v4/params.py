# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 parameter and cache-role types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal

from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

from ..dsa.params import DSAMetadataParams, DSAParams

DEEPSEEK_V4_SPARSE_RATIO = 4
DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO = 4


class DeepseekV4AttentionType(Enum):
    """DeepSeek-V4 cache roles."""

    # Sliding-window roles must remain contiguous.
    SWA = 0
    COMPRESSOR_KV = 1
    COMPRESSOR_SCORE = 2
    INDEXER_COMPRESSOR_KV = 3
    INDEXER_COMPRESSOR_SCORE = 4

    # Non-sliding cache roles.
    COMPRESS = 5
    INDEXER_COMPRESS = 6

    @property
    def role(self) -> DataRole:
        return DataRole(f"deepseek_v4_{self.name.lower()}")


DEEPSEEK_V4_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.SWA,
    DeepseekV4AttentionType.COMPRESSOR_KV,
    DeepseekV4AttentionType.COMPRESSOR_SCORE,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
)
assert tuple(attn_type.value for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION) == tuple(
    range(len(DEEPSEEK_V4_SLIDING_ATTENTION))
)

DEEPSEEK_V4_NON_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.COMPRESS,
    DeepseekV4AttentionType.INDEXER_COMPRESS,
)


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
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_KV:
        return is_compress
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return is_compress
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return is_sparse
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV:
        return is_sparse
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return is_sparse
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


@dataclass(frozen=True)
class DeepSeekV4Params(DSAParams):
    """DeepSeek-V4 backend parameters."""

    algorithm: Literal["deepseek_v4"] = field(init=False, default="deepseek_v4")
    compress_ratios: List[int] = field(default_factory=list)
    window_size: int = 128


@dataclass(frozen=True)
class DeepSeekV4MetadataParams(DSAMetadataParams):
    """DeepSeek-V4 metadata parameters."""

    compress_ratios: List[int] = field(default_factory=list)
    window_size: int = 128
