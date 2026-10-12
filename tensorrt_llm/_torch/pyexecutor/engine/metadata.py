# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention metadata construction shared across model runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

from ..config_utils import is_mla

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

__all__ = ["build_attention_metadata"]


def _get_num_heads_per_kv(pretrained_config: object) -> int:
    num_attention_heads = getattr(pretrained_config, "num_attention_heads", None)
    num_key_value_heads = getattr(pretrained_config, "num_key_value_heads", None)
    if isinstance(num_key_value_heads, (list, tuple)):
        num_key_value_heads = min(
            (heads for heads in num_key_value_heads if heads and heads > 0),
            default=0,
        )
    if num_attention_heads and num_key_value_heads:
        return num_attention_heads // num_key_value_heads
    return 1


def build_attention_metadata(
    model_config: ModelConfig,
    *,
    max_batch_size: int,
    max_num_tokens: int,
    max_beam_width: int,
    attention_backend: type[AttentionBackend],
    attention_runtime_features: AttentionRuntimeFeatures,
    mapping: Mapping,
    cache_indirection: torch.Tensor | None,
    kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None = None,
    draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None = None,
    enable_context_mla_with_cached_kv: bool | None = None,
    num_heads_per_kv: int | None = None,
) -> AttentionMetadata:
    """Construct attention metadata and resolve model-derived inputs."""
    pretrained_config = model_config.pretrained_config
    if enable_context_mla_with_cached_kv is None:
        enable_context_mla_with_cached_kv = is_mla(pretrained_config) and (
            attention_runtime_features.cache_reuse or attention_runtime_features.chunked_prefill
        )
    if num_heads_per_kv is None:
        num_heads_per_kv = _get_num_heads_per_kv(pretrained_config)

    sparse_attention_config = model_config.sparse_attention_config
    sparse_metadata_params = (
        sparse_attention_config.to_sparse_metadata_params(pretrained_config=pretrained_config)
        if sparse_attention_config is not None
        else None
    )
    return attention_backend.Metadata(
        max_num_requests=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_num_sequences=max_batch_size * max_beam_width,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        mapping=mapping,
        runtime_features=attention_runtime_features,
        enable_flash_mla=model_config.enable_flash_mla,
        enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
        cache_indirection=cache_indirection,
        num_heads_per_kv=num_heads_per_kv,
        sparse_metadata_params=sparse_metadata_params,
    )
