# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata construction and request-time updates shared across model runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.mapping import Mapping

from ..config_utils import is_mla

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

__all__ = [
    "build_attention_metadata",
    "update_spec_metadata",
]


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


def update_spec_metadata(
    spec_metadata: SpecMetadata,
    scheduled_requests: ScheduledRequests,
    attn_metadata: AttentionMetadata,
    spec_tree_manager: SpecTreeManager | None,
    *,
    runtime_draft_len: int,
    runtime_tokens_per_gen_step: int,
    is_draft_model: bool,
    attention_backend: type[AttentionBackend],
    original_max_draft_len: int,
    original_max_total_draft_tokens: int,
    spec_dec_max_total_draft_tokens: int,
) -> None:
    """Update speculative and attention metadata for one scheduled batch."""
    spec_metadata.runtime_draft_len = runtime_draft_len
    spec_metadata.runtime_tokens_per_gen_step = runtime_tokens_per_gen_step

    is_spec_dec_mode = spec_metadata.spec_dec_mode.attention_need_spec_dec_mode(
        is_draft_model,
        attention_backend,
    )
    # Parallel-draft modes advertise their full generation width rather than a
    # conventional draft length, so attention needs the total-token capacity.
    if spec_metadata.spec_dec_mode.is_parallel_draft():
        max_draft_len = original_max_total_draft_tokens
        max_total_draft_tokens = original_max_total_draft_tokens
    else:
        max_draft_len = original_max_draft_len
        max_total_draft_tokens = spec_dec_max_total_draft_tokens

    if spec_tree_manager is not None and spec_tree_manager.use_dynamic_tree and not is_draft_model:
        spec_tree_manager.slot_storage.fill_all_slot_ids(
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
        )

    attn_metadata.update_spec_dec_param(
        batch_size=scheduled_requests.batch_size,
        is_spec_decoding_enabled=is_spec_dec_mode,
        is_spec_dec_tree=spec_metadata.is_spec_dec_tree,
        is_spec_dec_dynamic_tree=spec_metadata.is_spec_dec_dynamic_tree,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        spec_metadata=spec_metadata,
        spec_tree_manager=spec_tree_manager,
        num_contexts=scheduled_requests.num_context_requests,
    )
