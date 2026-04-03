# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.sparse.dsa.indexer import (
    IndexerPrefillChunkMetadata,
    _compute_slot_mappings,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.sparse.dsa.indexer import Indexer
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
    from tensorrt_llm._torch.speculative.utils import SpecDecodingTensor


class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # Topk for sparse MLA
    sparse_mla_topk: int
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False

    def __init__(self, *args, **kwargs):
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_stride_factor = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer

    def __post_init__(self):
        super().__post_init__()

        self.sparse_mla_topk = self.sparse_attention_config.index_topk
        self.enable_indexer_skip = self.sparse_attention_config.skip_indexer_for_short_seqs
        capture_graph = self.is_cuda_graph

        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # These buffers will be allocated dynamically in Indexer.prepare() based on actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Block table for topk_indices conversion (shared for context and generation)
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.sparse_mla_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        # Per-layer persistent buffers for heuristic TopK pre_idx.
        # Indexed by [local_layer_idx, generation_position, :].
        # The graph captures reads/writes on these stable-address buffers;
        # each replay's write becomes the next replay's read (feedback loop).
        self.enable_heuristic_topk = (
            self.sparse_attention_config.enable_heuristic_topk and get_sm_version() >= 100
        )
        if self.enable_heuristic_topk:
            num_local_layers = self.kv_cache_manager.num_local_layers
            self.heuristic_prev_topk = self.get_empty(
                self.cuda_graph_buffers,
                (num_local_layers, self.max_num_sequences, self.sparse_mla_topk),
                cache_name="heuristic_prev_topk",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            # Zero-initialize so the first decode step's pre_idx (kernel
            # adds +1 offset) points to index 1 — a valid but benign candidate.
            # Without this, uninitialized memory produces random hint indices.
            self.heuristic_prev_topk.zero_()
            # Scratch buffer for heuristic TopK kernel output values.
            # Pre-allocated with stable address for CUDA Graph compatibility
            # (replaces cudaMallocAsync/cudaFreeAsync inside the kernel launcher).
            # Shape: [max_gen_tokens, topK] where max_gen_tokens = max_batch * (1 + max_draft).
            max_gen_tokens = self.max_num_sequences * (1 + self.max_draft_tokens)
            self.heuristic_scratch_values = self.get_empty(
                self.cuda_graph_buffers,
                (max_gen_tokens, self.sparse_mla_topk),
                cache_name="heuristic_scratch_values",
                dtype=torch.float32,
                capture_graph=capture_graph,
            )

        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self.max_draft_tokens),),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self.max_draft_tokens),
                self.kv_cache_manager.max_blocks_per_seq,
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # The fp8_paged_mqa_logits kernel needs different layout of the metadata buffer for MTP=3.
        if self.max_draft_tokens == 3:
            self.scheduler_metadata_buffer_mtp3 = self.get_empty(
                self.cuda_graph_buffers,
                (self.num_sms // 2 + 1, 2),
                cache_name="scheduler_metadata_buffer_mtp3",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        model_is_wrapped: bool = False,
        spec_metadata: Optional["SpecMetadata"] = None,
        spec_tree_manager: Optional["SpecTreeManager"] = None,
        spec_decoding_tensor: Optional["SpecDecodingTensor"] = None,
    ):
        super().update_spec_dec_param(
            batch_size,
            is_spec_decoding_enabled,
            is_spec_dec_tree,
            is_spec_dec_dynamic_tree,
            max_draft_len,
            max_total_draft_tokens,
            model_is_wrapped,
            spec_metadata,
            spec_tree_manager,
            spec_decoding_tensor,
        )
        self.max_draft_tokens = max_draft_len
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            capture_graph = self.is_cuda_graph
            self.create_expanded_buffers(capture_graph=capture_graph)
            # Resize heuristic scratch buffer for new max_draft_tokens.
            if self.enable_heuristic_topk:
                max_gen_tokens = self.max_num_sequences * (1 + self.max_draft_tokens)
                self.heuristic_scratch_values = self.get_empty(
                    self.cuda_graph_buffers,
                    (max_gen_tokens, self.sparse_mla_topk),
                    cache_name="heuristic_scratch_values",
                    dtype=torch.float32,
                    capture_graph=capture_graph,
                )

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False

    def _ensure_pool_view_cached(self):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, stride factor, block table slices, request
        index slices) are constant across all layers sharing the same KV pool
        and batch dimensions within a forward pass. Caching them avoids
        redundant Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(self.kv_cache_manager):
            return

        pool = self.kv_cache_manager.get_unique_primary_pool()
        kv_cache_manager = self.kv_cache_manager
        num_blocks, num_layers, _, _ = pool.shape
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_stride_factor = num_layers * self._cached_tokens_per_block
        self._cached_block_table_ctx = self.block_table[: self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts : self.num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[: self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens : self.num_tokens] - self.num_contexts
        )
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._pool_cache_valid = True

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends, global_indices, side="right")
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.sparse_mla_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self, kv_lens, device=False):  # device=False means use CPU
        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[: self.num_contexts],
                        kv_lens[: self.num_contexts],
                        self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[ctx_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[: self.num_contexts],
                    kv_lens[: self.num_contexts],
                    self.num_ctx_tokens,
                )
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :], non_blocking=True
                )

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts : self.num_seqs],
                        kv_lens[self.num_contexts : self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[gen_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[self.num_contexts : self.num_seqs],
                    kv_lens[self.num_contexts : self.num_seqs],
                    self.num_tokens - self.num_ctx_tokens,
                )
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :], non_blocking=True
                )

    def _get_pool_block_indices(self) -> torch.Tensor:
        """Extract memory pool block indices from host_kv_cache_block_offsets.

        The C++ setOffsets() encodes offsets as:
            encoded = memPoolBlockIndex * numLayers * kvFactor
        For SELFKONLY (MLA/DSA), kvFactor=1, so:
            memPoolBlockIndex = encoded // num_local_layers

        Returns a (num_seqs, max_blocks_per_seq) int32 CPU tensor with valid
        pool indices clamped to [0, blocks_in_primary_pool - 1].
        """
        num_local_layers = self.kv_cache_manager.num_local_layers
        max_pool_idx = self.kv_cache_manager.blocks_in_primary_pool - 1
        # DSA uses SELFKONLY mode where only key cache is stored (kv_factor=1).
        # host_kv_cache_block_offsets shape: (num_pools, max_batch*beam, 2, max_blocks_per_seq)
        # Note: dim=2 is always 2 in the tensor layout (K and V slots), but for
        # SELFKONLY only the K slot (index 0) contains valid data.
        assert self.kv_cache_manager.kv_factor == 1, (
            f"DSA requires SELFKONLY mode (kv_factor=1), got kv_factor={self.kv_cache_manager.kv_factor}"
        )
        # Pool 0, first num_seqs entries, field 0 (key offsets)
        encoded = self.kv_cache_manager.host_kv_cache_block_offsets[0, : self.num_seqs, 0, :]
        pool_indices = encoded // num_local_layers
        # Clamp for safety: handles garbage padding from torch.empty in uninitialized slots
        pool_indices = pool_indices.clamp(min=0, max=max_pool_idx).to(torch.int32)
        return pool_indices

    def prepare(self):
        super().prepare()
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            # For Helix CP, inactive ranks only attend to previously cached
            # tokens (no new token appended), while active ranks add new tokens.
            # This mirrors the kv_lens logic in TrtllmAttentionMetadata.prepare().
            active_rank = ~self.helix_is_inactive_rank_cpu[: self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        # Prepare to support skip indexer
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = (
                kv_lens[: self.num_contexts].max().item()
                <= self.sparse_mla_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = (
                kv_lens[self.num_contexts : self.num_seqs].max().item()
                <= self.sparse_mla_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

        # Build indexer_k_cache_block_offsets using pool block indices derived
        # from host_kv_cache_block_offsets (populated by super().prepare()).
        # This correctly resolves block IDs to memory pool indices, which is
        # required when host cache offload is enabled (block IDs != pool indices
        # for onboarded secondary blocks).
        if self.kv_cache_manager is not None:
            pool_indices = self._get_pool_block_indices()
            self.host_indexer_k_cache_block_offsets[: self.num_seqs].copy_(pool_indices)
            self.indexer_k_cache_block_offsets[: self.num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[: self.num_seqs], non_blocking=True
            )
            # Safety clamp: prevent OOB from CUDA graph padding entries which
            # may contain stale negative or out-of-range values after block
            # eviction/onboarding with host cache offload.
            self.indexer_k_cache_block_offsets.clamp_(min=0)

        # Build req_idx_per_token for topk_indices conversion
        host_req_idx_per_token = torch.repeat_interleave(
            torch.arange(self.num_seqs, dtype=torch.int32), self.seq_lens, dim=0
        )
        self.req_idx_per_token[: self.num_tokens].copy_(host_req_idx_per_token, non_blocking=True)

        # Build block_table for topk_indices conversion (actual block allocation)
        if self.kv_cache_manager is not None:
            tokens_per_block = self.kv_cache_manager.tokens_per_block
            num_blocks_per_seq = (
                kv_lens[: self.num_seqs] + tokens_per_block - 1
            ) // tokens_per_block
            max_blocks_used = num_blocks_per_seq.max().item() if self.num_seqs > 0 else 1
            # pool_indices already has correct values; set padding to -1
            host_block_table = pool_indices[:, :max_blocks_used].clone()
            for i in range(self.num_seqs):
                if num_blocks_per_seq[i] < max_blocks_used:
                    host_block_table[i, num_blocks_per_seq[i] :] = -1
            # Copy to GPU
            self.block_table[: self.num_seqs, :max_blocks_used].copy_(
                host_block_table, non_blocking=True
            )

        # For mla_rope_append_paged_kv_assign_q
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[: self.num_contexts].sum().item()
            self.max_ctx_kv_len = kv_lens[: self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[: self.num_contexts].max().item()
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_cached_token_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[: self.num_contexts + 1], non_blocking=True
            )
            # context kv indptr
            torch.cumsum(
                kv_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_kv_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_kv_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[: self.num_contexts + 1], non_blocking=True
            )
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            self.max_gen_seq_len = self.seq_lens[self.num_contexts : self.num_seqs].max().item()
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1 : self.num_generations + 1],
            )
            self.gen_cached_token_indptr[: self.num_generations + 1].copy_(
                self.host_gen_cached_token_indptr[: self.num_generations + 1], non_blocking=True
            )
            # generation kv indptr
            torch.cumsum(
                kv_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_kv_indptr[1 : self.num_generations + 1],
            )
            self.gen_kv_indptr[: self.num_generations + 1].copy_(
                self.host_gen_kv_indptr[: self.num_generations + 1], non_blocking=True
            )
        else:
            self.max_gen_seq_len = 0

        # Because the fp8_paged_mqa_logits only supports seq_len == 1/2/4 (i.e., max_draft_tokens == 0/1/3) on sm100, and  # noqa: E501
        # seq_len == 1/2 (i.e., max_draft_tokens == 0/1) on sm90, for other cases, we need to flatten the q tensor and
        # expand the kv_lens and block_table for MTP support.
        # TODO:
        # - No distinction between sm90 and sm100 is needed once MTP3 is supported on sm90.
        # - Remove this once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
        self.use_expanded_buffers_for_mtp = (
            self.max_draft_tokens > 1 and get_sm_version() == 90
        ) or ((self.max_draft_tokens == 2 or self.max_draft_tokens > 3) and get_sm_version() >= 100)
        if self.use_expanded_buffers_for_mtp:
            # Expand kv_lens_cuda (only generation)
            num_tokens = self.num_generations * (1 + self.max_draft_tokens)
            gen_kv_lens = kv_lens[self.num_contexts : self.num_seqs]
            gen_kv_lens_expanded = torch.stack([gen_kv_lens] * (1 + self.max_draft_tokens), dim=0)
            gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(0, 1).contiguous().flatten()
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True
            )

            # Expand indexer_k_cache_block_offsets (only generation)
            # host_indexer_k_cache_block_offsets already contains correct pool
            # indices from _get_pool_block_indices() above.
            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts : self.num_seqs, :max_len
                ]
                expanded_blocks = gen_block_tensor.repeat_interleave(
                    1 + self.max_draft_tokens, dim=0
                )
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                    expanded_blocks, non_blocking=True
                )
                self.block_table_expanded[:num_tokens].copy_(
                    self.host_block_table_expanded[:num_tokens], non_blocking=True
                )
                self.block_table_expanded.clamp_(min=0)

        # Prepare metadata for indexer
        # Deferred import to avoid circular dependency (indexer uses
        # DSAtrtllmAttentionMetadata as a type hint).
        from .indexer import Indexer as _Indexer

        _Indexer.prepare(metadata=self)

    def on_update_kv_lens(self):
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside _preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        # _preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        if self.kv_cache_manager is not None and self.num_tokens > 0:
            seq_lens = self.seq_lens_cuda[: self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[: self.num_seqs] - seq_lens

            # Reuse request-per-token mapping prepared in metadata.prepare().
            # This avoids repeat_interleave in graph-capture mode.
            req_indices = self.req_idx_per_token[: self.num_tokens].to(dtype=torch.int64)
            seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = (
                torch.arange(self.num_tokens, device=seq_lens.device, dtype=torch.int64)
                - seq_starts[req_indices]
            )

            global_positions = start_positions[req_indices] + token_offsets
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                self.kv_cache_manager.index_head_dim,
                self.kv_cache_manager.tokens_per_block,
                self.kv_cache_manager.quant_block_size,
            )
            self.slot_mapping_fp8[: self.num_tokens] = fp8_indices
            self.slot_mapping_scale[: self.num_tokens] = scale_indices

        if self.num_generations > 0:
            torch.cumsum(
                self.kv_lens_cuda[self.num_contexts : self.num_seqs],  # num_contexts should be 0
                dim=0,
                dtype=torch.int64,
                out=self.gen_kv_indptr[1 : self.num_generations + 1],
            )
            torch.cumsum(
                (
                    self.kv_lens_cuda[self.num_contexts : self.num_seqs]
                    - self.seq_lens_cuda[self.num_contexts : self.num_seqs]
                ),
                dim=0,
                dtype=torch.int64,
                out=self.gen_cached_token_indptr[1 : self.num_generations + 1],
            )
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                self.kv_lens_cuda[self.num_contexts : self.num_seqs],
                self.kv_cache_manager.tokens_per_block,
                self.num_sms,
            )
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
                kv_lens_expanded = torch.stack([gen_kv_lens] * num_draft_tokens, dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = (
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                )
                # Expand schedule metadata buffer (only generation)
                kv_lens_expanded = self.kv_lens_expanded_cuda[:num_tokens]
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded, self.kv_cache_manager.tokens_per_block, self.num_sms
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
            elif self.max_draft_tokens == 3:
                scheduler_metadata_buffer_mtp3 = get_paged_mqa_logits_metadata(
                    self.kv_lens_cuda[self.num_contexts : self.num_seqs],
                    self.kv_cache_manager.tokens_per_block,
                    self.num_sms // 2,
                )
                self.scheduler_metadata_buffer_mtp3.copy_(
                    scheduler_metadata_buffer_mtp3, non_blocking=True
                )
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self):
        super().update_for_spec_dec()
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        self.on_update_kv_lens()


# Alias for backward compatibility (old all-caps "DSA" prefix).
# TODO: Remove once all downstream references have migrated to DSAtrtllmAttentionMetadata.
DSATrtllmAttentionMetadata = DSAtrtllmAttentionMetadata
