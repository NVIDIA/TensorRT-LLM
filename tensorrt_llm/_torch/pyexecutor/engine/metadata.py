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
from tensorrt_llm._torch.attention.backends.sparse.hooks import has_sparse_mla_hooks
from tensorrt_llm._torch.model_config import ModelConfig, _get_kv_cache_layer_specs
from tensorrt_llm.mapping import Mapping

from ..config_utils import is_mla

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.metadata.core import TrtllmCoreMetadata
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

__all__ = ["build_attention_metadata", "build_feature_attention_metadata"]


def _get_max_num_heads_per_kv(model_config: ModelConfig) -> int:
    """Return the largest per-layer GQA ratio for shared mask allocation."""
    layer_specs = _get_kv_cache_layer_specs(model_config)
    if layer_specs:
        return max(spec.num_q_heads // spec.num_kv_heads for spec in layer_specs)

    pretrained_config = model_config.pretrained_config
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
    max_num_heads_per_kv: int | None = None,
    enable_ragged_verification: bool = False,
) -> AttentionMetadata:
    """Construct attention metadata and resolve model-derived inputs."""
    pretrained_config = model_config.pretrained_config
    sparse_attention_config = model_config.sparse_attention_config
    sparse_mla = has_sparse_mla_hooks(getattr(sparse_attention_config, "algorithm", None))
    if enable_context_mla_with_cached_kv is None:
        # Dense owner-only Helix prefill needs append indptrs even without cached tokens.
        enable_context_mla_with_cached_kv = is_mla(pretrained_config) and (
            attention_runtime_features.cache_reuse
            or attention_runtime_features.chunked_prefill
            or (not sparse_mla and mapping.has_cp_helix())
        )
    elif (
        enable_context_mla_with_cached_kv is False
        and is_mla(pretrained_config)
        and not sparse_mla
        and mapping.has_cp_helix()
    ):
        raise ValueError(
            "Helix MLA requires enable_context_mla_with_cached_kv=True "
            "to prepare owner-only KV append metadata"
        )
    if max_num_heads_per_kv is None:
        max_num_heads_per_kv = _get_max_num_heads_per_kv(model_config)

    sparse_metadata_params = (
        sparse_attention_config.to_sparse_metadata_params(pretrained_config=pretrained_config)
        if sparse_attention_config is not None
        else None
    )
    metadata_kwargs = {}
    if enable_ragged_verification:
        from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
            DSAtrtllmAttentionMetadata,
        )

        if not issubclass(attention_backend.Metadata, DSAtrtllmAttentionMetadata):
            raise ValueError("Ragged verification requires the DSA target metadata backend")
        metadata_kwargs["enable_ragged_verification"] = True
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
        max_num_heads_per_kv=max_num_heads_per_kv,
        sparse_metadata_params=sparse_metadata_params,
        **metadata_kwargs,
    )


def build_feature_attention_metadata(
    model_config: ModelConfig,
    *,
    max_batch_size: int,
    max_num_tokens: int,
    max_beam_width: int,
    attention_backend: type[AttentionBackend],
    attention_runtime_features: AttentionRuntimeFeatures,
    mapping: Mapping,
    kv_cache_manager: KVCacheManager | KVCacheManagerV2,
    draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None,
    spec_config: DecodingBaseConfig | None,
) -> TrtllmCoreMetadata:
    """Construct the feature-split attention metadata (TRTLLM_ATTENTION_METADATA=new).

    Returns:
        The core metadata; the other features are reached from it with ``get_metadata``.

    Raises:
        UnsupportedMetadataConfiguration: The configuration is outside what the
            feature-split metadata cover.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4TrtllmAttention
    from tensorrt_llm._torch.attention.metadata.base import (
        UnsupportedMetadataConfiguration,
        build_metadata,
    )
    from tensorrt_llm._torch.attention.metadata.batch_inputs import BatchInputs
    from tensorrt_llm._torch.attention.metadata.chunked_prefill import ChunkedPrefillMetadata
    from tensorrt_llm._torch.attention.metadata.core import TrtllmCoreMetadata
    from tensorrt_llm._torch.attention.metadata.deepseek_v4 import DeepseekV4Metadata
    from tensorrt_llm._torch.attention.metadata.draft import DraftMetadata
    from tensorrt_llm._torch.attention.metadata.mla_context import MlaContextMetadata
    from tensorrt_llm._torch.attention.metadata.spec_decoding import SpecDecodingMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    if not issubclass(attention_backend, DeepseekV4TrtllmAttention):
        raise UnsupportedMetadataConfiguration(
            f"the attention backend {attention_backend.__name__} (only DeepSeek-V4 on "
            "TrtllmAttention is covered)"
        )
    if max_beam_width != 1:
        raise UnsupportedMetadataConfiguration("beam search")
    if mapping.has_cp_helix():
        raise UnsupportedMetadataConfiguration("Helix context parallelism")
    if model_config.enable_flash_mla:
        raise UnsupportedMetadataConfiguration("FlashMLA")

    metadata_types = [
        TrtllmCoreMetadata,
        MlaContextMetadata,
        SpecDecodingMetadata,
        DeepseekV4Metadata,
    ]
    enable_context_mla_with_cached_kv = (
        attention_runtime_features.cache_reuse or attention_runtime_features.chunked_prefill
    )
    if attention_runtime_features.chunked_prefill:
        metadata_types.append(ChunkedPrefillMetadata)
    max_draft_tokens = 0
    if spec_config is not None:
        spec_dec_mode = spec_config.spec_dec_mode
        if not spec_dec_mode.is_mtp_eagle_one_model() or not spec_config.is_linear_tree:
            raise UnsupportedMetadataConfiguration(
                f"the speculative decoding mode {spec_dec_mode.name} (only linear MTP-Eagle "
                "one-model is covered)"
            )
        if getattr(spec_config, "enable_confidence_scheduling", False):
            raise UnsupportedMetadataConfiguration("ragged verification")
        max_draft_tokens = spec_config.max_draft_len
        metadata_types.append(DraftMetadata)

    sparse_attention_config = model_config.sparse_attention_config
    engine_inputs = BatchInputs(
        seq_lens=torch.zeros((0,), dtype=torch.int),
        num_contexts=0,
        request_ids=[],
        prompt_lens=[],
        kv_cache_params=KVCacheParams(use_cache=True),
        max_num_requests=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_num_sequences=max_batch_size * max_beam_width,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        mapping=mapping,
        runtime_features=attention_runtime_features,
        max_num_heads_per_kv=_get_max_num_heads_per_kv(model_config),
        sparse_metadata_params=sparse_attention_config.to_sparse_metadata_params(
            pretrained_config=model_config.pretrained_config
        ),
        enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
        max_draft_tokens=max_draft_tokens,
    )
    return build_metadata(metadata_types, engine_inputs)
