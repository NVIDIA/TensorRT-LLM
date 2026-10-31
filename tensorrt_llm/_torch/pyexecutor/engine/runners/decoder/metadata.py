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
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    MsaRuntimePolicy,
    RuntimeHooks,
    runtime_hooks_of,
)
from tensorrt_llm._torch.speculative import SpecMetadata, get_spec_metadata
from tensorrt_llm._torch.speculative.interface import supports_sharded_target_sampling
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...metadata import build_attention_metadata
from .config import DecoderRunnerConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionMetadata,
        AttentionRuntimeFeatures,
    )
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

    # Step-loop behaviour the model opted into (RuntimeHooks); all off unless it declares otherwise.
    runtime_hooks: RuntimeHooks = DEFAULT_RUNTIME_HOOKS

    def __init__(
        self,
        *,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        model: torch.nn.Module,
        buffers: DecoderInputBuffers,
        attention_runtime_features: AttentionRuntimeFeatures,
    ) -> None:
        self._config = config
        self.mapping = mapping
        self.model = model
        self.runtime_hooks = runtime_hooks_of(model)
        self._buffers = buffers
        self._attention_runtime_features = attention_runtime_features
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
    ) -> AttentionMetadata:
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
            attention_runtime_features=self._attention_runtime_features,
            mapping=self.mapping,
            cache_indirection=self._buffers.cache_indirection
            if self._config.attention_backend.Metadata is TrtllmAttentionMetadata
            else None,
            kv_cache_manager=kv_cache_manager,
            draft_kv_cache_manager=draft_kv_cache_manager,
            enable_ragged_verification=self._ragged_verification_enabled,
        )
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            self.attn_metadata.mamba_chunk_size = getattr(
                config, "chunk_size", self.attn_metadata.mamba_chunk_size
            )
        self.attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(self.model)
        self._arm_msa_policy(kv_cache_manager)

        return self.attn_metadata

    def _arm_msa_policy(self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None) -> None:
        """Hand the model's MiniMax-M3 MSA preparation choices to the metadata and the KV cache."""
        msa = self.runtime_hooks.msa
        if msa == MsaRuntimePolicy():
            return
        # Imported here: only a model that armed an MSA policy needs the sparse backend.
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
            MiniMaxM3KVCacheManagerV2,
        )
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
            MiniMaxM3MsaSparseAttentionMetadata,
        )

        if not isinstance(
            self.attn_metadata, MiniMaxM3MsaSparseAttentionMetadata
        ) or not isinstance(kv_cache_manager, MiniMaxM3KVCacheManagerV2):
            raise ValueError(
                "RuntimeHooks.msa requires the MiniMax-M3 MSA sparse attention backend and KV cache, "
                f"got {type(self.attn_metadata).__name__} and {type(kv_cache_manager).__name__}"
            )
        self.attn_metadata.msa_policy = msa
        kv_cache_manager.configure_decode_shortcuts(
            skip_unchanged_block_offsets=msa.skip_unchanged_block_offsets,
            reuse_pool_views=msa.reuse_pool_views,
        )

    def setup_speculative(
        self, spec_resource_manager: BaseResourceManager | None
    ) -> SpecMetadata | None:
        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = create_spec_metadata(
            self._config, self.model.config, spec_resource_manager, self.mapping
        )
        self.spec_metadata.skip_greedy_rng_upload = self.runtime_hooks.skip_greedy_rng_upload
        return self.spec_metadata

    @property
    def _ragged_verification_enabled(self) -> bool:
        spec_config = self._config.spec_config
        return bool(
            spec_config is not None and getattr(spec_config, "enable_confidence_scheduling", False)
        )

    @torch.inference_mode()
    def _warmup_dspark_ragged_compressor_metadata(self) -> None:
        """Compile DSv4 ragged metadata for a non-CUDA-graph batch size.

        Whole-model CUDA-graph warmup exercises only configured graph batch
        sizes. The first mixed attention-DP iteration can contain an
        intermediate number of generation requests, which otherwise triggers
        a long ``torch.compile`` specialization immediately before a TP
        collective. Ranks without generation work reach the collective first
        and can trip the executor hang detector while their peers compile.

        One non-bucket probe forces the dynamic specialization during engine
        warmup, before the hang detector starts. It uses the metadata's real
        persistent buffers so the tensor shape and stride guards match serving.
        Those buffers are created under inference mode, so all probe and
        cleanup mutations must run under the same mode.
        """
        if not self._ragged_verification_enabled:
            return

        metadata = getattr(self, "attn_metadata", None)
        required_attributes = (
            "_compute_compressed_mask",
            "_compute_gen_compressed_position_ids",
            "_compress_ratios_sorted",
            "compressed_mask_cuda",
            "compressed_position_ids_cuda",
            "cu_new_comp_kv_cuda",
            "batch_size_cuda",
            "max_draft_tokens",
            "new_comp_kv_lens_cuda",
            "past_kv_lens_cuda",
        )
        missing_attributes = [
            name for name in required_attributes if metadata is None or not hasattr(metadata, name)
        ]
        if missing_attributes:
            raise RuntimeError(
                "DSpark ragged metadata warmup requires the DSv4 compressor "
                "contract; missing " + ", ".join(missing_attributes)
            )

        graph_batch_sizes = {int(batch_size) for batch_size in self._config.cuda_graph_batch_sizes}
        num_generations = next(
            (
                candidate
                for candidate in range(int(self._config.max_batch_size) - 1, 0, -1)
                if candidate not in graph_batch_sizes
            ),
            None,
        )
        if num_generations is None:
            return

        compress_ratios = [int(ratio) for ratio in metadata._compress_ratios_sorted]
        tokens_per_generation = 1 + int(metadata.max_draft_tokens)
        total_compressed_tokens = {}
        gen_output_offsets = {ratio: 0 for ratio in compress_ratios}
        prepared_ratios = []
        original_batch_size = metadata.batch_size_cuda.clone()
        try:
            for ratio in compress_ratios:
                compressed_tokens_per_generation = (tokens_per_generation + ratio - 1) // ratio
                total_tokens = num_generations * compressed_tokens_per_generation
                past_kv_lens = metadata.past_kv_lens_cuda[ratio]
                cu_new_comp = metadata.cu_new_comp_kv_cuda[ratio]
                new_comp = metadata.new_comp_kv_lens_cuda[ratio]
                compressed_positions = metadata.compressed_position_ids_cuda[ratio]
                compressed_mask = metadata.compressed_mask_cuda[ratio]
                if (
                    past_kv_lens.numel() < num_generations
                    or cu_new_comp.numel() < num_generations + 1
                    or new_comp.numel() < num_generations
                    or compressed_positions.numel() < total_tokens
                    or compressed_mask.numel() < total_tokens
                ):
                    raise RuntimeError(
                        "DSpark ragged compressor metadata buffers cannot "
                        f"hold the compile probe with {num_generations} "
                        "generation requests and compression ratio "
                        f"{ratio}"
                    )

                total_compressed_tokens[ratio] = total_tokens
                prepared_ratios.append(ratio)
                past_kv_lens[:num_generations].zero_()
                new_comp[:num_generations].fill_(compressed_tokens_per_generation)
                cu_new_comp[: num_generations + 1].copy_(
                    torch.arange(
                        num_generations + 1, dtype=cu_new_comp.dtype, device=cu_new_comp.device
                    )
                )
                cu_new_comp[: num_generations + 1].mul_(compressed_tokens_per_generation)

            logger.info(
                "DSpark ragged metadata warmup: compiling one non-graph "
                f"shape with {num_generations} generation requests"
            )
            metadata._compute_gen_compressed_position_ids(
                metadata.past_kv_lens_cuda,
                metadata.cu_new_comp_kv_cuda,
                metadata.compressed_position_ids_cuda,
                0,
                num_generations,
                tokens_per_generation,
                compress_ratios,
                gen_output_offsets,
            )
            metadata.batch_size_cuda.fill_(num_generations)
            metadata._compute_compressed_mask(
                metadata.new_comp_kv_lens_cuda,
                metadata.cu_new_comp_kv_cuda,
                metadata.compressed_mask_cuda,
                metadata.batch_size_cuda,
                compress_ratios,
            )
            torch.cuda.synchronize()
        finally:
            metadata.batch_size_cuda.copy_(original_batch_size)
            for ratio in prepared_ratios:
                total_tokens = total_compressed_tokens[ratio]
                metadata.past_kv_lens_cuda[ratio][:num_generations].zero_()
                metadata.new_comp_kv_lens_cuda[ratio][:num_generations].zero_()
                metadata.cu_new_comp_kv_cuda[ratio][: num_generations + 1].zero_()
                metadata.compressed_position_ids_cuda[ratio][:total_tokens].zero_()
                metadata.compressed_mask_cuda[ratio][:total_tokens].zero_()
            torch.cuda.synchronize()
        logger.info("DSpark ragged metadata warmup complete")


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
    mapping: Mapping | None = None,
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
        # The decode CUDA graphs then capture the temperature-only variant.
        temperature_only_graphs=config.cuda_graph_config is not None
        and config.spec_decode_fast_paths
        and supports_sharded_target_sampling(config.spec_config, mapping),
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
