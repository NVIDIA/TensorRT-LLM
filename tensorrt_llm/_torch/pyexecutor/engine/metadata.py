# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention metadata construction shared across model runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.extensions import MetadataExtension
from tensorrt_llm._torch.attention.backends.features import FlashMlaMetadata, RecurrentConfig
from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingConfig
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.sparse.hooks import has_sparse_mla_hooks
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig, _get_kv_cache_layer_specs
from tensorrt_llm.mapping import Mapping

from ..config_utils import is_mla
from ..kv_cache.mamba_cache_manager import BaseMambaCacheManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

__all__ = ["build_attention_metadata", "metadata_extensions_of"]


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


def metadata_extensions_of(model: object) -> tuple[type[MetadataExtension], ...]:
    """The attention-metadata extensions ``model`` requires, or ().

    A model declares them as ``metadata_extensions`` (a class or instance
    attribute, or a property): a tuple of ``MetadataExtension`` types the
    engine attaches to the model's attention metadata with its own, a
    subclass of one of the engine's replacing it.
    """
    extensions = tuple(getattr(model, "metadata_extensions", ()))
    for extension_type in extensions:
        if not (isinstance(extension_type, type) and issubclass(extension_type, MetadataExtension)):
            raise TypeError(
                f"{type(model).__name__}.metadata_extensions must hold MetadataExtension "
                f"types, got {extension_type!r}"
            )
    return extensions


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
    recurrent_config: RecurrentConfig | None = None,
    spec_decoding: SpecDecodingConfig | None = None,
    one_model_draft: bool = False,
    max_seq_len: int | None = None,
    model_extension_types: tuple = (),
    model_extension_config: dict | None = None,
) -> AttentionMetadata:
    """Construct attention metadata and resolve model-derived inputs.

    ``max_seq_len`` bounds the sequences of no-KV-cache attention; it is given
    to the families whose core takes it. ``model_extension_types`` and
    ``model_extension_config`` are the extensions the model declares and their
    creation arguments (``metadata_extensions_of(model)`` /
    ``RuntimeHooks.metadata_config``), attached with the engine's own.
    """
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
    # Ragged verification is taken by the DSA extension, which the TRTLLM
    # family attaches for a DSA or DeepSeek-V4 sparse attention config.
    if enable_ragged_verification and not issubclass(
        attention_backend.Metadata, TrtllmAttentionMetadata
    ):
        raise ValueError("Ragged verification requires the DSA target metadata backend")
    # FlashMLA kernels run on the TRTLLM backend only.
    extension_types = (
        (FlashMlaMetadata,)
        if model_config.enable_flash_mla
        and issubclass(attention_backend.Metadata, TrtllmAttentionMetadata)
        else ()
    )
    # Each argument goes to the extension that takes it: speculative-decoding
    # state is TRTLLM's, and recurrent state exists only for hybrid models.
    family_settings = {}
    if issubclass(attention_backend.Metadata, TrtllmAttentionMetadata):
        family_settings["max_num_heads_per_kv"] = max_num_heads_per_kv
        if spec_decoding is not None:
            family_settings["spec_decoding"] = spec_decoding
    if isinstance(kv_cache_manager, BaseMambaCacheManager):
        family_settings["recurrent_config"] = recurrent_config
    core_type = attention_backend.Metadata.core_metadata_type
    if max_seq_len is not None and "max_seq_len" in core_type.declared_init_fields():
        family_settings["max_seq_len"] = max_seq_len
    if enable_ragged_verification:
        family_settings["enable_ragged_verification"] = True
    return attention_backend.Metadata(
        max_num_requests=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_num_sequences=max_batch_size * max_beam_width,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        one_model_draft=one_model_draft,
        mapping=mapping,
        runtime_features=attention_runtime_features,
        enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
        cache_indirection=cache_indirection,
        sparse_metadata_params=sparse_metadata_params,
        extension_types=(*extension_types, *model_extension_types),
        **family_settings,
        **(model_extension_config or {}),
    )
