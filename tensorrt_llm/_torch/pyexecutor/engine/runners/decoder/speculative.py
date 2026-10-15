# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding metadata for scheduled decoder execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.speculative import SpecMetadata, get_spec_metadata

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager, ResourceManager
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

    from .config import DecoderRunnerConfig


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
