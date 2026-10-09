# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder attention and speculative metadata."""

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import BaseMambaCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager, KVCacheManager
from tensorrt_llm.mapping import Mapping

from ...metadata import build_attention_metadata
from .config import DecoderRunnerConfig
from .inputs import DecoderInputBuffers
from .speculative import create_spec_metadata


def resolve_mamba_metadata_cls(model: torch.nn.Module) -> type[Mamba2Metadata]:
    """Resolve the model-specific Mamba metadata class with a default."""
    return getattr(model, "mamba_metadata_cls", None) or Mamba2Metadata


class DecoderMetadata:
    """Attention and speculative metadata reused across decoder iterations."""

    def __init__(
        self,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        model: torch.nn.Module,
        buffers: DecoderInputBuffers,
    ) -> None:
        self._config = config
        self.mapping = mapping
        self.model = model
        self._buffers = buffers
        # This field is initialized lazily on the first forward pass.
        # This is convenient because:
        # 1) The attention metadata depends on the KV cache manager.
        # 2) The KV cache manager depends on the model configuration.
        # 3) The model configuration is not loaded until the model engine
        # is initialized.
        #
        # NOTE: This can be simplified by decoupling the model config loading and
        # the model engine.
        self.attn_metadata = None
        self.spec_metadata = None

    def set_up_attn(
        self,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None = None,
    ):
        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        config = self.model.model_config.pretrained_config
        self.attn_metadata = build_attention_metadata(
            self.model.model_config,
            max_batch_size=self._config.max_batch_size,
            max_num_tokens=self._config.max_num_tokens,
            max_beam_width=self._config.max_beam_width,
            attention_backend=self._config.attention_backend,
            attention_runtime_features=self._config.attention_runtime_features,
            mapping=self.mapping,
            cache_indirection=self._buffers.cache_indirection_attention
            if self._config.attention_backend.Metadata is TrtllmAttentionMetadata
            else None,
            kv_cache_manager=kv_cache_manager,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            self.attn_metadata.mamba_chunk_size = getattr(
                config, "chunk_size", self.attn_metadata.mamba_chunk_size
            )
        self.attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(self.model)

        return self.attn_metadata

    def set_up_spec(self, spec_resource_manager: BaseResourceManager | None):
        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = create_spec_metadata(
            self._config, self.model.config, spec_resource_manager
        )
        return self.spec_metadata
