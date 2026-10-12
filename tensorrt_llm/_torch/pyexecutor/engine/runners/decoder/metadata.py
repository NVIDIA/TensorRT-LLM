# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder attention and speculative metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import BaseMambaCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.speculative import SpecMetadata, get_spec_metadata
from tensorrt_llm.mapping import Mapping

from ...metadata import build_attention_metadata
from .config import DecoderRunnerConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

    from .inputs import DecoderInputBuffers


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

    def setup_attention(
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
            cache_indirection=self._buffers.cache_indirection
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

    def setup_speculative(self, spec_resource_manager: BaseResourceManager | None):
        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = create_spec_metadata(
            self._config, self.model.config, spec_resource_manager
        )
        return self.spec_metadata


def get_spec_managers(
    resource_manager: ResourceManager,
) -> tuple[BaseResourceManager | None, SpecTreeManager | None]:
    """Return the speculative resource manager and its tree manager, if any."""
    spec_resource_manager = resource_manager.get_resource_manager(
        ResourceManagerType.SPEC_RESOURCE_MANAGER
    )
    return spec_resource_manager, getattr(spec_resource_manager, "spec_tree_manager", None)


def create_spec_metadata(
    config: DecoderRunnerConfig,
    pretrained_config: object,
    spec_resource_manager: BaseResourceManager | None,
) -> SpecMetadata | None:
    """Create speculative metadata sized by the runner capacities."""
    return get_spec_metadata(
        config.spec_config,
        pretrained_config,
        config.max_batch_size,
        max_num_tokens=config.max_num_tokens,
        spec_resource_manager=spec_resource_manager,
        max_seq_len=config.max_seq_len,
        num_seq_slots=config.num_seq_slots,
    )


def update_spec_metadata(
    spec_metadata: SpecMetadata,
    config: DecoderRunnerConfig,
    scheduled_requests: ScheduledRequests,
    attn_metadata: AttentionMetadata,
    spec_tree_manager: SpecTreeManager | None,
    *,
    runtime_draft_len: int,
) -> None:
    """Update speculative and attention metadata for one scheduled batch."""
    spec_metadata.runtime_draft_len = runtime_draft_len
    spec_metadata.runtime_tokens_per_gen_step = config.spec_config.get_runtime_tokens_per_gen_step(
        runtime_draft_len
    )

    is_spec_dec_mode = spec_metadata.spec_dec_mode.attention_need_spec_dec_mode(
        config.attention_backend
    )
    # Parallel-draft modes advertise their full generation width rather than a
    # conventional draft length, so attention needs the total-token capacity.
    if spec_metadata.spec_dec_mode.is_parallel_draft():
        max_draft_len = config.original_max_total_draft_tokens
        max_total_draft_tokens = config.original_max_total_draft_tokens
    else:
        max_draft_len = config.original_max_draft_len
        max_total_draft_tokens = config.spec_dec_max_total_draft_tokens

    if spec_tree_manager is not None:
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


def sync_group_all_greedy_sample(
    spec_metadata: SpecMetadata, mapping: Mapping, dist: Distributed | None
) -> None:
    """All-gather the per-rank greedy flags and store the group AND.

    Why the sampling-path choice must be group-uniform under
    ADP + LM-head TP is documented on the anchor,
    ``SpecMetadata.group_all_greedy_sample``. Local contract: called once
    per iteration, right after ``update_is_all_greedy_sample`` and BEFORE
    the CUDA graph key is built. The gate is pure config (identical on
    every rank), so ranks also agree on whether the exchange happens; the
    gather spans the whole TP group, a superset of any LM-head-TP
    subgroup. A dedicated host all-gather rather than a piggyback on the
    ``all_rank_num_tokens`` exchange, which runs in ``prepare_inputs`` --
    after the graph key, too late for the key to see the synced value.
    """
    # enable_lm_head_tp_in_adp implies enable_attention_dp (asserted in
    # Mapping.__init__), so ADP needs no separate check here.
    if not (mapping.enable_lm_head_tp_in_adp and spec_metadata.use_rejection_sampling):
        return
    local_flag = bool(spec_metadata.is_all_greedy_sample)
    all_flags = dist.tp_allgather_int64([local_flag])[:, 0]
    spec_metadata.group_all_greedy_sample = bool(all_flags.all())
    # Also overwrite the live flag directly: this iteration's scan already
    # ran (update_is_all_greedy_sample just returned) and the CUDA graph
    # key reads the flag next -- the stored override only takes effect on
    # the NEXT rescan (populate), which is after key selection.
    spec_metadata.is_all_greedy_sample = spec_metadata.group_all_greedy_sample


def set_spec_metadata_all_rank_num_tokens(
    spec_metadata: SpecMetadata,
    spec_all_rank_num_tokens: list[int],
    all_rank_num_seqs: list[int],
    all_rank_num_gens: list[int] | None = None,
) -> None:
    # Eagle3 / MTP-eagle one-model use subseq_all_rank_num_tokens for
    # draft loop iterations i>0 (per-sequence counts, since each
    # sequence contributes one token per iteration).
    spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
    spec_metadata.all_rank_num_seqs = all_rank_num_seqs
    # DSpark can draft only after the target processes the current bonus token,
    # because it consumes captured target-layer hidden states for that token.
    # Prefill computes hidden states for prompt tokens; the first generated token
    # is sampled from the last prompt logits and has not itself passed through the
    # target layers. Thus context requests seed the rolling window but do not run
    # the draft. On mixed steps, num_seqs therefore over-counts the draft MoE
    # workload; gen-only per-rank counts keep the FUSED_COMM (DeepGEMM MegaMoE)
    # chunk loop identical across EP ranks.
    if all_rank_num_gens is not None:
        spec_metadata.all_rank_num_gens = all_rank_num_gens
    if (
        spec_metadata.spec_dec_mode.is_mtp_eagle_one_model()
        or spec_metadata.spec_dec_mode.is_eagle3_one_model()
    ):
        spec_metadata.subseq_all_rank_num_tokens = all_rank_num_seqs
