# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 target-owned indexer extensions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    _INDEXER_MQA_LOGITS_ELEM_BUDGET,
    IndexerParams,
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _packed_logits_elem_budget,
    _prefill_q_split_threshold,
    _select_indexer_compress_ratio,
    _split_prefill_queries,
    compute_cu_seqlen_kv_bounds_with_cache,
    plan_mla_chunked_prefill_chunks,
    select_indexer_chunk_size,
    split_prefill_chunks,
    use_cute_dsl_paged_mqa_logits,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer as _BaseIndexer
from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather, cp_allgather
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ....ops.attention.convert_req_index_to_global import convert_req_index_to_global
from ....ops.attention.dsa_params import DSAParams
from ....ops.attention.indexer_k_cache_gather_op import indexer_k_cache_gather_op
from .top_k import TopK, TopKImplementation


def merge_helix_topk(
    logits: torch.Tensor,
    topk_rows: torch.Tensor,
    cp_rank: int,
    gather_cp: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Reduce per-rank Top-K rows to the global Top-K across Helix CP ranks.

    Under Helix each CP rank holds a slice of every sequence's KV cache, so its
    indexer only ranks local tokens. The global Top-K is a subset of the union
    of the local Top-Ks, so gathering every rank's candidate scores and taking
    the Top-K of that union reproduces the selection of an unsharded indexer.
    Local entries that lose to another rank's candidates become -1 in place and
    the surviving entries move to the front of each row.

    Args:
        logits: ``[num_tokens, kv_len]`` local indexer scores.
        topk_rows: ``[num_tokens, k]`` int32 local token indices, -1 for none.
        cp_rank: This rank's position in the CP group.
        gather_cp: Stacks a tensor over the CP group along a new leading dim.
    """
    num_tokens, k = topk_rows.shape
    if num_tokens == 0:
        return
    valid = topk_rows >= 0
    scores = torch.gather(logits.float(), 1, topk_rows.clamp(min=0).long())
    scores = scores.masked_fill(~valid, float("-inf"))
    # [cp, num_tokens, k] -> [num_tokens, cp * k], ordered by (rank, slot).
    all_scores = gather_cp(scores).transpose(0, 1).reshape(num_tokens, -1)
    top_scores, top_pos = torch.topk(all_scores, k, dim=1)
    owner = torch.div(top_pos, k, rounding_mode="floor")
    mine = (owner == cp_rank) & torch.isfinite(top_scores)
    hits = torch.zeros_like(topk_rows, dtype=torch.int32)
    hits.scatter_add_(1, torch.where(mine, top_pos - owner * k, 0), mine.to(torch.int32))
    keep = hits > 0
    # Stable compaction: kept entries first, in their original order.
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
    compacted = torch.gather(topk_rows.masked_fill(~keep, -1), 1, order)
    topk_rows.copy_(compacted)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached(layer_idx)
    page_index_scale, layer_offset = (
        attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(layer_idx)
    )

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    global_indices = convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        page_index_scale * attn_metadata._cached_tokens_per_block,
        layer_offset,
    )

    return global_indices, attn_metadata._cached_pool_view


class Indexer(_BaseIndexer):
    """Attention behavior owned by the NVIDIA K3.1 target."""

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DSAParams,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
        *,
        projection_free: bool = False,
        projection_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize cache and Top-K operators; the forward owns projection tensors."""
        nn.Module.__init__(
            self,
        )
        self.projection_free = True
        sm_version = get_sm_version() if torch.cuda.is_available() else 0
        self.hidden_size = mla_params.hidden_size if mla_params is not None else None
        self.q_lora_rank = mla_params.q_lora_rank if mla_params is not None else None
        if sparse_params.index_rope_dim is not None:
            self.rope_dim = sparse_params.index_rope_dim
        else:
            self.rope_dim = mla_params.qk_rope_head_dim if mla_params is not None else 0
        self.n_heads = sparse_params.index_n_heads  # 64
        self.head_dim = sparse_params.index_head_dim  # 128
        self.index_topk = sparse_params.index_topk  # 2048
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        # indexer_k_dtype controls both Q and K precision. DeepGEMM's
        # fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels only dispatch
        # to FP4xFP4 or FP8xFP8 (no mixed-precision variant). The DeepGEMM
        # kernel asserts SM100 + head_dim=128 at launch time under FP4.
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        # Fork/join events for the aux-stream prev_topk write-back.
        self.prev_topk_copy_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._prev_topk_copy_pending = False
        self.use_cute_dsl_topk = sparse_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        self.use_cute_dsl_paged_mqa_logits = use_cute_dsl_paged_mqa_logits(
            sparse_params.use_cute_dsl_paged_mqa_logits
        )
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = sparse_params.enable_heuristic_topk and sm_version >= 100
        # Two-level GVR dispatch: enable_heuristic_topk selects the GVR
        # family over the exact radix path; use_self_sampling_topk (default
        # True) selects the hint-free self-sampling engine over the
        # temporal-hint engines. Tuning is frozen from indexer_max_seq_len
        # at capture time, so the launch is CUDA-graph-replay safe. The
        # TopK module's hardware-format gate falls back to the exact
        # insertion/radix path with a one-time warning; contract violations
        # inside the engine raise.
        self._use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self._enable_heuristic_topk,
            use_self_sampling_topk=sparse_params.use_self_sampling_topk,
            index_topk=sparse_params.index_topk,
            compress_ratio=compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=sm_version,
        )
        if os.environ.get("TRTLLM_GVR_SELF_SAMPLING") is not None:
            logger.warning_once(
                "TRTLLM_GVR_SELF_SAMPLING is retired and ignored: the "
                "self-sampling GVR engine is selected by the "
                "use_self_sampling_topk sparse-attention config field "
                "(default True) when enable_heuristic_topk is set.",
                key="gvr_self_sampling_env_retired",
            )
        if (
            self._enable_heuristic_topk
            and sparse_params.use_self_sampling_topk
            and not self._use_self_sampling_topk
        ):
            logger.warning_once(
                "use_self_sampling_topk=True but the self-sampling GVR "
                "prerequisites are not met "
                f"(cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                f"sm={sm_version}, "
                f"index_topk={sparse_params.index_topk}, "
                f"compress_ratio={compress_ratio}); falling back to the "
                "temporal GVR path (exact radix when the DSL engine is "
                "unavailable).",
                key="gvr_self_sampling_prereq_fallback",
            )
        self.mtp_index_share = sparse_params.mtp_index_share

        if (
            self._enable_heuristic_topk
            and IS_CUTLASS_DSL_AVAILABLE
            # datacenter Blackwell only; consumer Blackwell (sm_120/121)
            # lacks the thread-block clusters both GVR engines use
            and sm_version in (100, 103)
        ):
            decode_top_k_implementation = TopKImplementation.CUTE_DSL_GVR
        else:
            if self._enable_heuristic_topk:
                logger.warning_once(
                    "enable_heuristic_topk=True but the DSL GVR engine is "
                    f"unavailable (cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                    f"sm={get_sm_version()}); using the exact radix decode "
                    "top-K instead.",
                    key="gvr_prereq_radix_fallback",
                )
            decode_top_k_implementation = (
                TopKImplementation.CUTE_DSL_RADIX
                if self.use_cute_dsl_topk
                else TopKImplementation.CUDA_RADIX
            )
        # Prefill uses the self-sampling engine on exactly the layers where the
        # decode dispatch picks it; the temporal-hint engine has no prefill form.
        prefill_top_k_implementation = (
            TopKImplementation.CUTE_DSL_GVR
            if (
                decode_top_k_implementation == TopKImplementation.CUTE_DSL_GVR
                and self._use_self_sampling_topk
            )
            else TopKImplementation.CUDA_RADIX
        )
        self.top_k = TopK(
            self.index_topk,
            prefill_implementation=prefill_top_k_implementation,
            decode_implementation=decode_top_k_implementation,
            compress_ratio=self.compress_ratio,
            gvr_self_sampling=self._use_self_sampling_topk,
            gvr_deterministic=sparse_params.gvr_deterministic_topk,
        )
        # Emission block-skip is a temporal-hint (V1) optimization: the FP4
        # indexer epilogue emits per-block max logits the GVR Top-K consumes to
        # skip whole blocks (see gvr_emission / gvr_routing; state lives on the
        # TopK module). Off on the self-sampling path (no cross-step state to
        # assist) and off non-FP4 / non-paged-MQA layers.
        self.use_gvr_emission = (
            sparse_params.use_gvr_emission
            and not self._use_self_sampling_topk
            and decode_top_k_implementation == TopKImplementation.CUTE_DSL_GVR
            and self.use_cute_dsl_paged_mqa_logits
            and self.use_fp4
        )

        # Fused wk + weights_proj weight for single FP32 cuBLAS GEMM
        # (populated in cache_derived_state; maps to TF32 tensor cores on Ampere+)
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

    @staticmethod
    def recompute_slot_mappings(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: Optional[IndexerParams] = None
    ):
        """Recompute slot_mapping_fp8/scale from current block offsets.

        This is the subset of prepare_for_update_k_cache() that maps each new
        compressed KV token to its flat cache position. It is safe to call in
        isolation after a caller has swapped the active KV cache manager and
        block-offset buffers, such as during draft KV-cache replay.
        """
        if indexer_params is None:
            indexer_params = Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        batch_size = indexer_params.batch_size
        tokens_per_block = indexer_params.tokens_per_block
        head_dim = indexer_params.head_dim
        new_kv_tokens = indexer_params.new_kv_tokens
        total_new_kv_tokens = new_kv_tokens.sum().item()
        data_bytes_per_token = indexer_params.data_bytes_per_token

        # Compute global positions for all kv tokens in the batch (fully vectorized)
        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device="cpu"), new_kv_tokens
        )
        # Vectorized token_offsets: arange(total) - cumulative start per request
        cu_new_kv = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
        cu_new_kv[1:] = new_kv_tokens.to(torch.int64).cumsum(0)
        token_offsets = torch.arange(
            total_new_kv_tokens, dtype=torch.int64, device="cpu"
        ) - cu_new_kv[:-1].repeat_interleave(new_kv_tokens)
        global_positions = indexer_params.cached_kv_tokens[req_indices] + token_offsets

        # Block indices/pos for all kv tokens in the batch
        block_indices_in_seq = global_positions // tokens_per_block

        max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            indexer_params.quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_new_kv_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_new_kv_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_new_kv_tokens], non_blocking=True
        )
        metadata.slot_mapping_scale[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_new_kv_tokens], non_blocking=True
        )
        if metadata.enable_helix and metadata.num_generations > 0:
            Indexer._helix_host_gen_slot_mappings(metadata, indexer_params)

    @staticmethod
    def _helix_host_gen_slot_mappings(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ) -> None:
        """Give every generation row its own Helix-aware indexer-K slot.

        The scatter pairs K row i with slot i, but under Helix a rank appends
        only the generation tokens it owns, so the counts from new_kv_tokens
        would leave rows paired with other tokens' (or stale) slots. Here the
        active rank writes its single token at the local tail and other ranks
        get -1, which the scatter skips. Verify groups get placeholders that
        on_update_kv_lens() replaces from the per-token Helix buffers.
        """
        num_ctx_tokens = metadata.num_ctx_tokens
        num_gen_tokens = metadata.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return
        if metadata._helix_spec_tokens_valid:
            fp8_flat_indices = torch.full((num_gen_tokens,), -1, dtype=torch.int64)
            scale_flat_indices = fp8_flat_indices
        else:
            assert num_gen_tokens == metadata.num_generations, (
                "Helix generation without verify groups appends one token per request"
            )
            gen = slice(metadata.num_contexts, metadata.num_seqs)
            inactive = metadata.helix_is_inactive_rank_cpu[: metadata.num_generations]
            positions = indexer_params.cached_kv_tokens[gen].to(torch.int64)
            positions = torch.where(inactive, torch.zeros_like(positions), positions)
            req_indices = torch.arange(
                metadata.num_contexts, metadata.num_seqs, dtype=torch.int64, device="cpu"
            )
            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                positions,
                metadata.host_indexer_k_cache_block_offsets,
                req_indices,
                indexer_params.head_dim,
                indexer_params.tokens_per_block,
                indexer_params.quant_block_size,
                data_bytes_per_token=indexer_params.data_bytes_per_token,
            )
            fp8_flat_indices = fp8_flat_indices.masked_fill(inactive, -1)
            scale_flat_indices = scale_flat_indices.masked_fill(inactive, -1)
        gen_rows = slice(num_ctx_tokens, num_ctx_tokens + num_gen_tokens)
        metadata.host_slot_mapping_fp8[gen_rows] = fp8_flat_indices
        metadata.host_slot_mapping_scale[gen_rows] = scale_flat_indices
        metadata.slot_mapping_fp8[gen_rows].copy_(
            metadata.host_slot_mapping_fp8[gen_rows], non_blocking=True
        )
        metadata.slot_mapping_scale[gen_rows].copy_(
            metadata.host_slot_mapping_scale[gen_rows], non_blocking=True
        )

    @staticmethod
    def recompute_context_kv_gather_mappings(
        metadata: DSAtrtllmAttentionMetadata,
        indexer_params: Optional[IndexerParams] = None,
    ) -> None:
        """Recompute context KV-cache gather mappings for the active cache manager."""
        if not metadata.enable_context_mla_with_cached_kv:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        if indexer_params is None:
            indexer_params = Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        num_contexts = indexer_params.num_contexts
        if num_contexts == 0:
            # Generation does not consume full-KV gather mappings. In
            # particular, avoid allocating temporary zero-length tensors
            # while capturing generation CUDA graphs.
            return

        if metadata.skip_indexer_for_ctx_reqs:
            # The dense short-context path does not gather from the indexer K
            # cache, so full-KV mappings would be allocated but never consumed.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        cached_kv_tokens = indexer_params.cached_kv_tokens[:num_contexts]
        if indexer_params.compress_ratio == 1 and cached_kv_tokens.count_nonzero().item() == 0:
            # Without compression or cached tokens, the regular mappings cover
            # every context KV token in the same order and remain equivalent
            # when on_update_kv_lens() refreshes them in place.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        total_kv_per_request = indexer_params.kv_lens[:num_contexts]
        total_kv_len = total_kv_per_request.sum().item()
        if total_kv_len == 0:
            # No chunk can gather indexer KV, so avoid zero-length allocation
            # and H2D work for short compressed contexts.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        host_slot_mapping_fp8_fullkv = torch.empty(
            total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
        )
        host_slot_mapping_scale_fullkv = torch.empty(
            total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
        )

        req_indices = torch.repeat_interleave(
            torch.arange(num_contexts, dtype=torch.int64, device="cpu"), total_kv_per_request
        )

        cu_kv = torch.zeros(num_contexts + 1, dtype=torch.int64, device="cpu")
        cu_kv[1:] = total_kv_per_request.to(torch.int64).cumsum(0)
        kv_positions = torch.arange(total_kv_len, dtype=torch.int64, device="cpu") - cu_kv[
            :-1
        ].repeat_interleave(total_kv_per_request)

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            kv_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            indexer_params.head_dim,
            indexer_params.tokens_per_block,
            indexer_params.quant_block_size,
            data_bytes_per_token=indexer_params.data_bytes_per_token,
        )

        host_slot_mapping_fp8_fullkv.copy_(fp8_flat_indices)
        host_slot_mapping_scale_fullkv.copy_(scale_flat_indices)
        metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(non_blocking=True)
        metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(non_blocking=True)

    @staticmethod
    def prepare_for_update_k_cache(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the update_k_cache stage.

        Compute slot_mapping for all requests (both context and generation)
        This maps each token to its flat cache position for vectorized KV cache updates
        """
        Indexer.recompute_slot_mappings(metadata, indexer_params)

    @staticmethod
    def prepare_for_chunked_prefill(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the chunked prefill.
        """
        num_contexts = indexer_params.num_contexts
        seq_lens = indexer_params.seq_lens
        metadata.indexer_prefill_q_split = None

        # When MLA chunked prefill is active, it already bounds the new tokens per
        # request, so requests are not split by length here. From
        # q_split_threshold context query tokens on, the batch's queries are
        # split evenly across TP ranks and each rank plans chunks for its share
        # only. Within that share, requests are grouped so that a chunk's packed
        # logits stay bounded: with cached KV, a request's KV can be far longer
        # than its new tokens.
        has_mla_chunked_prefill = (
            metadata.enable_context_mla_with_cached_kv and metadata.runtime_features.chunked_prefill
        )
        if has_mla_chunked_prefill:
            compress_ratio = _effective_compress_ratio_divisor(
                _select_indexer_compress_ratio(metadata.compress_ratios)
            )
            q_lens = seq_lens[:num_contexts].tolist()
            cached_indptr = metadata.host_ctx_cached_token_indptr[: num_contexts + 1].tolist()
            kv_lens = [
                (cached_indptr[i + 1] - cached_indptr[i] + q_lens[i]) // compress_ratio
                for i in range(num_contexts)
            ]
            q_begin, q_end, q_sizes = _split_prefill_queries(
                sum(q_lens), metadata.mapping, _prefill_q_split_threshold(metadata)
            )
            if q_sizes is not None:
                metadata.indexer_prefill_q_split = (q_begin, q_end, q_sizes)
            metadata.indexer_prefill_chunks = [
                Indexer.prepare_one_prefill_chunk(metadata, chunk_specs)
                for chunk_specs in plan_mla_chunked_prefill_chunks(
                    q_lens, kv_lens, _packed_logits_elem_budget(metadata), q_begin, q_end
                )
            ]
        else:
            # Use indexer's chunking to avoid quadratic indexer MQA logits
            # computation for long sequences.
            # This is only used when MLA chunked prefill is not enabled.
            # Adapt chunk size to the batch's largest K_compressed (see
            # select_indexer_chunk_size).
            max_k_compressed = (
                int(indexer_params.kv_lens[:num_contexts].max().item()) if num_contexts > 0 else 0
            )
            effective_chunk_size = select_indexer_chunk_size(
                metadata.indexer_max_chunk_size, max_k_compressed
            )
            chunk_groups = split_prefill_chunks(
                seq_lens[:num_contexts],
                effective_chunk_size,
                start_idx=0,
            )

            if len(chunk_groups) > 1 or metadata.enable_context_mla_with_cached_kv:
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                    for chunk_specs in chunk_groups
                ]
            else:
                metadata.indexer_prefill_chunks = None

        # Chunked prefill and KV-cache reuse require gather mappings that cover
        # all cached and newly appended indexer tokens.
        Indexer.recompute_context_kv_gather_mappings(metadata, indexer_params)

    @staticmethod
    def prepare(metadata: DSAtrtllmAttentionMetadata) -> None:
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """

        # Skip indexer preparation if the kv_cache_manager doesn't have index_head_dim.
        # This can happen when the metadata is being used with a draft KV cache manager
        # during MTP speculative decoding, which uses a regular KVCacheManager instead
        # of DSACacheManager.
        indexer_params = Indexer.build_indexer_params(metadata)
        if indexer_params is None:
            return

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        # Store compressed KV token count for context requests
        metadata.num_ctx_kv_tokens = indexer_params.new_kv_tokens[:num_contexts].sum().item()

        # Prepare for update_k_cache
        Indexer.prepare_for_update_k_cache(metadata, indexer_params)

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            host_num_past_tokens = indexer_params._num_past_tokens_tensor[:num_contexts]
            host_kv_lens = indexer_params.kv_lens[:num_contexts]
            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens,
                num_contexts,
                num_ctx_tokens,
                host_num_past_tokens,
                host_kv_lens,
                compress_ratio,
            )

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ks), non_blocking=True
            )
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ke), non_blocking=True
            )
            Indexer.prepare_for_chunked_prefill(metadata, indexer_params)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            Indexer.prepare_scheduler_metadata(metadata)

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for one phase or the full batch.

        When ``is_generation`` is ``None``, the inputs contain the full mixed
        batch. Otherwise they contain only the selected context or generation
        phase. ``q_scale`` is only consumed by the FP4 dispatch.

        """
        # DSACacheManager / DeepseekV4CacheManager force quant_block_size to
        # 128 (FP8 path) or 32 (MXFP4 path); both round-trip to the same
        # 4-byte scale word per token at index_head_dim=128, so the slot
        # mapping arithmetic is unchanged in either mode.
        assert metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size in (
            32,
            128,
        ), (
            "Unexpected quant_block_size "
            f"{metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        )
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        num_gen_tokens = num_tokens - num_ctx_tokens
        gvr_prior_indices = None
        if self.top_k.needs_gvr_prior:
            assert metadata.gvr_prior_indices is not None
            local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
            gvr_prior_indices = metadata.gvr_prior_indices[local_layer]
        if is_generation is None:
            has_prefill = num_contexts > 0
            has_decode = num_generations > 0
            token_offset = num_ctx_tokens
            cache_name = "indexer_topk_out_buffer"
        else:
            has_prefill = not is_generation and num_contexts > 0
            has_decode = is_generation and num_generations > 0
            token_offset = 0
            expected_tokens = num_gen_tokens if is_generation else num_ctx_tokens
            assert hidden_states.shape[0] == expected_tokens, (
                "Phase-specific DSA prediction received "
                f"{hidden_states.shape[0]} tokens, expected {expected_tokens}."
            )
            cache_name = (
                "indexer_topk_out_buffer_gen" if is_generation else "indexer_topk_out_buffer_ctx"
            )

        topk_indices_buffer = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (hidden_states.shape[0], self.index_topk),
            cache_name=cache_name,
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                q_split_threshold = _prefill_q_split_threshold(metadata)
                # Chunks planned for this rank's share of the queries only are
                # never split again; their top-k rows are gathered after the loop.
                batch_q_split = metadata.indexer_prefill_q_split
                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
                # FP4 packs two codes per byte so the gathered row holds half
                # as many bytes as in the FP8 path. The scale (4 bytes) is the
                # same in both modes because FP4 packs four UE8M0 exponents
                # into one int32 to match FP8's float32 scale width.
                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim

                for chunk in metadata.indexer_prefill_chunks:
                    # Skip chunks with no compressed KV tokens (e.g., warmup
                    # sequences shorter than compress_ratio produce zero KV).
                    if chunk.k_token_start >= chunk.k_token_end:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :].fill_(-1)
                        continue
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = indexer_k_cache_gather_op(
                        k_cache_4d,
                        metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        num_k_tokens,
                        gather_head_dim,
                    )

                    chunk_num_token = chunk.token_end - chunk.token_start
                    if batch_q_split is None:
                        chunk_q_start, chunk_q_end, q_sizes = _split_prefill_queries(
                            chunk_num_token, metadata.mapping, q_split_threshold
                        )
                    else:
                        chunk_q_start, chunk_q_end, q_sizes = 0, chunk_num_token, None

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    # Tile the query dimension so each fp8_mqa_logits call
                    # allocates at most [q_tile x num_k_tokens] instead of the
                    # full [local_q x num_k_tokens] (which can reach tens of GB
                    # on a long context and stall cuMemCreate under
                    # expandable_segments -> engine hang; see
                    # _INDEXER_MQA_LOGITS_ELEM_BUDGET). Results are identical:
                    # each query row's logits/top-k are independent and the KV
                    # (chunk_k_fp8) is unchanged across tiles, so the per-call
                    # allocation is the same size and the caching allocator
                    # reuses one block (peak ~= one tile, no extra sync).
                    local_q_len = chunk_q_end - chunk_q_start
                    budget = _INDEXER_MQA_LOGITS_ELEM_BUDGET
                    if self.use_fp4 and (
                        self.head_dim != 128 or not q_fp8.is_cuda or get_sm_version() < 100
                    ):
                        # Decoded MQA also materializes per-head dot products.
                        budget = max(1, budget // self.n_heads)
                    q_tile = max(1, min(local_q_len, budget // max(1, num_k_tokens)))
                    for tile_off in range(0, local_q_len, q_tile):
                        c0 = chunk_q_start + tile_off
                        c1 = min(c0 + q_tile, chunk_q_end)
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        tile_q_scale = q_scale[g0:g1, ...] if self.use_fp4 else None
                        self.forward_prepared(
                            q_fp8[g0:g1, ...],
                            chunk_k_fp8,
                            chunk_k_scale,
                            weights[g0:g1, ...],
                            chunk.cu_seqlen_ks[c0:c1],
                            chunk.cu_seqlen_ke[c0:c1],
                            topk_indices_buffer[g0:g1, :],
                            tile_q_scale,
                        )

                    if q_sizes is not None:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :] = allgather(
                            topk_indices_buffer[global_q_start:global_q_end, :],
                            metadata.mapping,
                            dim=0,
                            sizes=q_sizes,
                        )

                if batch_q_split is not None:
                    local_q_begin, local_q_end, q_sizes = batch_q_split
                    topk_indices_buffer[: sum(q_sizes), :] = allgather(
                        topk_indices_buffer[local_q_begin:local_q_end, :],
                        metadata.mapping,
                        dim=0,
                        sizes=q_sizes,
                    )
            elif metadata.num_ctx_kv_tokens == 0:
                # No compressed KV tokens — fill with -1 (no valid indices)
                topk_indices_buffer[:num_ctx_tokens, :].fill_(-1)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                num_ctx_kv_tokens = metadata.num_ctx_kv_tokens
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens, ...] if self.use_fp4 else None
                self.forward_prepared(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_kv_tokens, ...],
                    k_scale[:num_ctx_kv_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices_buffer[:num_ctx_tokens, :],
                    ctx_q_scale,
                )
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = metadata.topk_indices_buffer[
                :num_ctx_tokens, :
            ]

        # Dense skip outputs remain valid priors if a later step enters GVR.
        if has_prefill:
            # Device lengths: gathering the CUDA selections with the host
            # seq_lens would force a synchronous H2D copy every layer.
            self.top_k.update_gvr_prior_from_prefill(
                topk_indices_buffer[:num_ctx_tokens],
                metadata.seq_lens_cuda[:num_contexts],
                gvr_prior_indices,
                request_offset=num_generations,
            )
            if self.use_gvr_emission:
                # reused slots cold-start the emission closed loop; stale
                # lines only mis-place cuts - counts are re-measured
                # in-kernel, so exactness never rides on this reset
                self.top_k.reset_gvr_emission_rows(
                    slice(num_generations, num_generations + num_contexts)
                )

        reuse_topk = (
            self.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and metadata.shared_topk_indices is not None
        )

        if has_decode and not metadata.skip_indexer_for_gen_reqs and reuse_topk:
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.shared_topk_indices[:num_generations, :]
            )
        elif has_decode and not metadata.skip_indexer_for_gen_reqs:
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
            if not metadata.is_ragged_verify:
                max_decode_len = gen_seq_lens.max().item()
                min_decode_len = gen_seq_lens.min().item()
                assert max_decode_len == min_decode_len, (
                    "max_decode_len != min_decode_len, we need padding"
                )

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[token_offset : token_offset + num_gen_tokens, ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if metadata.is_ragged_verify:
                take_strided_path = False
            else:
                take_strided_path = not metadata.use_expanded_buffers_for_mtp or next_n == 1
            if take_strided_path:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts : num_contexts + num_generations
                ]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n. The DSL path
                # uses its own schedule buffer (built with num_next_n_atoms=1
                # via a (num_gen, 1) input shape) and overrides this below.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            if metadata.is_ragged_verify:
                assert num_gen_tokens == sum(metadata.ragged_verify_lens), (
                    f"ragged gen tokens {num_gen_tokens} != "
                    f"sum(verify_lens) {sum(metadata.ragged_verify_lens)}"
                )
            else:
                assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[token_offset : token_offset + num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits / fp8_fp4_paged_mqa_logits
            # with prepared decode metadata.
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            indexer_max_seq_len = metadata.get_indexer_max_seq_len()

            # Helix verify rows carry per-token bounds the DSL schedule cannot take.
            if (
                self.use_cute_dsl_paged_mqa_logits
                and not metadata.is_ragged_verify
                and metadata.helix_spec_row_kv_lens(num_gen_tokens) is None
            ):
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `Indexer.prepare()`
                # with a (num_gen, 1) input shape, which makes DeepGEMM's wrapper
                # compute `num_next_n_atoms = 1`. (DeepGEMM uses the same buffer
                # for its own next_n=1 kernel; DSL piggy-backs on it for all
                # real next_n values.) All next_n positions of a batch share
                # the same KV length on this path (kv_lens_cuda_2d broadcasts),
                # so passing the 1D contiguous kv_lens slice for context_lens
                # avoids materializing a 2D contiguous tensor per call.
                dsl_context_lens = metadata.gen_indexer_kv_lens_cuda_runtime
                assert dsl_context_lens is not None
                # Wave-aware atom-split: the picker in `_pick_dsl_expand` caches
                # (factor, atom) on metadata with invariant
                # `factor * atom == 1 + max_draft_tokens` (the target/verify-time
                # next_n). MTPEagle reuses the same metadata for its multi-step
                # draft loop; after i=0 it mutates seq_lens to 1, so i≥1
                # iterations run with next_n=1. The reshape
                # `(num_gen, next_n, ...) -> (num_gen*factor, atom, ...)` is only
                # valid when the caller actually supplies next_n == factor * atom
                # tokens; gate here so i≥1 draft calls fall back to the
                # kernel-native next_n=1 path.
                dsl_atom_split = (
                    metadata.dsl_expand_factor > 1
                    and next_n == metadata.dsl_expand_factor * metadata.dsl_atom
                )
                # In-kernel uneven split (next_n with no exact atom divisor):
                # native Q / context_lens / block_table, schedule over the
                # staggered per-atom rows. Same full-window gate as above.
                dsl_kernel_split = (
                    metadata.dsl_kernel_atoms > 1 and next_n == metadata.dsl_kernel_next_n
                )
                dsl_split_kwargs: dict = {}
                if dsl_kernel_split:
                    dsl_split_kwargs = dict(
                        num_next_n_atoms=metadata.dsl_kernel_atoms, atom=metadata.dsl_atom
                    )
                gvr_emit_kwargs: dict = {}
                # emitting for a step the Top-K cannot consume only churns
                # the closed-loop state, so gate on the consumable shape
                if (
                    self.use_gvr_emission
                    and gvr_prior_indices is not None
                    and next_n == 1
                    and not dsl_atom_split
                    and not dsl_kernel_split
                    and num_gen_tokens <= 256
                    # ext tiers are single-CTA/sort-path only; row reordering
                    # routes the Top-K through order_row, which excludes them
                    and metadata.kv_lens_row_reorder is None
                ):
                    gvr_emit_kwargs = self.top_k.prepare_gvr_emission(
                        num_generations,
                        indexer_max_seq_len,
                        torch.cuda.get_device_properties(q_decode.device).multi_processor_count,
                        gvr_prior_indices,
                    )
                if self.use_fp4:
                    # FP4 DSL signature splits DG's (q, sf_q) tuple into two
                    # separate args and requires q.dtype == uint8 (q_decode
                    # came in via the FP8 plumbing as int8; reinterpret with
                    # no copy). sf_q is the q_scale slice reshaped to
                    # (B, next_n, H) int32 -- mirrors the non-DSL FP4 branch.
                    decode_q_scale = q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                    dsl_q = q_decode.view(torch.uint8)
                    dsl_block_table = block_table
                    dsl_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        eff_next_n = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = dsl_q.reshape(exp_B, eff_next_n, self.n_heads, self.head_dim // 2)
                        decode_q_scale = decode_q_scale.reshape(exp_B, eff_next_n, self.n_heads)
                        dsl_context_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        dsl_block_table = metadata.block_table_expanded[:exp_B]
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    elif dsl_kernel_split:
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded

                    logits_decode = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q,
                        decode_q_scale,
                        k_cache,
                        weights_decode,
                        dsl_context_lens,
                        dsl_block_table,
                        dsl_schedule_meta,
                        indexer_max_seq_len,
                        **gvr_emit_kwargs,
                        **dsl_split_kwargs,
                    )
                else:
                    # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                    # Atom-split benefits small-batch / low-ntask configs by
                    # raising SM utilization at the cost of factorx KV HBM
                    # re-reads. Context lengths are indexer KV lengths, which
                    # may be compressed for DeepSeek-V4.
                    dsl_q = q_decode
                    fp8_ctx_lens = dsl_context_lens
                    fp8_block_table = block_table
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        atom = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = q_decode.reshape(exp_B, atom, self.n_heads, self.head_dim)
                        fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        fp8_block_table = metadata.block_table_expanded[:exp_B]
                        fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    elif dsl_kernel_split:
                        fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    logits_decode = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                        dsl_q,
                        k_cache,
                        weights_decode,
                        fp8_ctx_lens,
                        fp8_block_table,
                        fp8_schedule_meta,
                        indexer_max_seq_len,
                        **gvr_emit_kwargs,
                        **dsl_split_kwargs,
                    )
            else:
                decode_q_scale = (
                    q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    if self.use_fp4
                    else None
                )
                if self.use_fp4:
                    # q_decode shape is either (num_generations, next_n, n_heads,
                    # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                    # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                logits_decode = self._call_paged_mqa_logits(
                    q_decode,
                    k_cache,
                    weights_decode,
                    context_lens,
                    block_table,
                    scheduler_metadata_buffer,
                    indexer_max_seq_len,
                    decode_q_scale,
                )

            # Native/GVR use logical lengths; radix uses score-column lengths.
            gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                num_contexts : num_contexts + num_generations
            ]
            scan_lengths = metadata.gen_indexer_kv_lens_cuda_runtime
            assert scan_lengths is not None

            gvr_ext_kwargs = (
                {
                    "gvr_prior_indices": gvr_prior_indices[:num_generations],
                    "gvr_row_order": metadata.kv_lens_row_reorder,
                }
                if gvr_prior_indices is not None
                else None
            )
            assert metadata.radix_aux_indices is not None
            assert metadata.radix_aux_logits is not None
            # Helix verify tokens carry per-token rank-local bounds, like ragged rows.
            helix_row_kv_lens = metadata.helix_spec_row_kv_lens(num_gen_tokens)
            row_kv_lens = (
                helix_row_kv_lens
                if helix_row_kv_lens is not None
                else metadata.ragged_row_kv_lens(num_gen_tokens)
            )
            decode_rows = topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :]
            self.top_k(
                logits_decode,
                decode_rows,
                is_prefill=False,
                sequence_lengths=gen_kv_lens_cuda,
                scan_lengths=scan_lengths,
                row_kv_lens=row_kv_lens,
                next_n=1 if metadata.is_ragged_verify or helix_row_kv_lens is not None else next_n,
                max_seq_len=indexer_max_seq_len,
                gvr_ext_kwargs=gvr_ext_kwargs,
                radix_aux_indices=metadata.radix_aux_indices,
                radix_aux_logits=metadata.radix_aux_logits,
            )
            if metadata.enable_helix and metadata.mapping.cp_size > 1:
                if helix_row_kv_lens is not None:
                    # Tokens that see no KV on this rank were scored over a
                    # placeholder slot; they must not offer candidates.
                    decode_rows.masked_fill_((helix_row_kv_lens == 0).unsqueeze(1), -1)
                mapping = metadata.mapping
                merge_helix_topk(
                    logits_decode,
                    decode_rows,
                    mapping.cp_rank,
                    lambda t: cp_allgather(t.unsqueeze(0), mapping, dim=0),
                )
                metadata.set_helix_sparse_row_lens(decode_rows)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
            )

        # Keep the GVR prior current for computed, reused, and dense-skip TopK.
        if gvr_prior_indices is not None and has_decode:
            decode_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
            if metadata.is_ragged_verify:
                last_rows = (
                    torch.cumsum(metadata.gen_token_repeats_cuda[:num_generations], dim=0).to(
                        torch.long
                    )
                    - 1
                )
                last_mtp_topk = decode_topk[last_rows]
            else:
                next_n = num_gen_tokens // num_generations
                last_mtp_topk = decode_topk[next_n - 1 :: next_n]
            prev_topk_dst = gvr_prior_indices[:num_generations]
            if self.aux_stream is not None and torch.cuda.is_current_stream_capturing():
                # Overlap the GVR prior write-back with this layer's sparse attention.
                self.prev_topk_copy_events[0].record()
                with torch.cuda.stream(self.aux_stream):
                    self.prev_topk_copy_events[0].wait()
                    prev_topk_dst.copy_(last_mtp_topk)
                    self.prev_topk_copy_events[1].record()
                self._prev_topk_copy_pending = True
            else:
                prev_topk_dst.copy_(last_mtp_topk)

        if self.mtp_index_share and metadata.in_mtp_draft_loop and not reuse_topk:
            rows = None
            if has_decode:
                next_n = num_gen_tokens // num_generations
                gen_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
                rows = self._mtp_last_accepted_rows(
                    gen_topk, metadata, num_contexts, num_generations, next_n
                )
            if has_prefill:
                ctx_last = (
                    torch.cumsum(metadata.seq_lens_cuda[:num_contexts].to(torch.long), dim=0) - 1
                )
                ctx_rows = topk_indices_buffer[ctx_last]
                rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
            if rows is not None:
                shared_topk_indices = metadata.shared_topk_indices
                if shared_topk_indices is None:
                    metadata.shared_topk_indices = rows.contiguous()
                else:
                    row_start = num_contexts if is_generation else 0
                    shared_topk_indices[
                        row_start : row_start + rows.shape[0], : rows.shape[1]
                    ].copy_(rows)
        return topk_indices_buffer


if TYPE_CHECKING:
    from .metadata import DSAtrtllmAttentionMetadata
