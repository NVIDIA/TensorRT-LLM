# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Sparse MLA state of DeepSeek-V4."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch

from tensorrt_llm._utils import get_sm_version, is_sm_100f, prefer_pinned

from .base import UnsupportedMetadataConfiguration, _Registry
from .dsa import DSAMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs

_SUPPORTED_COMPRESS_RATIOS = frozenset({1, 4, 128})


class DeepseekV4Metadata(DSAMetadata):
    """DeepSeek-V4's sparse MLA state, per compress ratio of the cache's layers.

    Every layer reads a sliding window (SWA) of the latest tokens. A layer of
    compress ratio 4 also reads the compressed rows its indexer selects, and
    a layer of ratio 128 every compressed row. This metadata holds the page
    tables and pool pointers of those caches, the per-token row indices and
    row counts the attention kernels read, and the compressor's lengths and
    positions.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        from ..backends.sparse.deepseek_v4.params import (
            DEEPSEEK_V4_SLIDING_ATTENTION,
            DeepSeekV4MetadataParams,
            is_compress_layer,
        )

        super().__init__(registry, config)
        params = config.sparse_metadata_params
        if not isinstance(params, DeepSeekV4MetadataParams):
            raise ValueError("DeepSeek-V4 sparse attention metadata params are not set")
        if not is_sm_100f(get_sm_version()):
            raise UnsupportedMetadataConfiguration(f"DeepSeek-V4 on SM{get_sm_version()}")
        manager = config.kv_cache_manager
        if getattr(manager, "mixed_precision_high_layers", None):
            raise UnsupportedMetadataConfiguration("mixed-precision KV cache")
        self.window_size = params.window_size
        assert self.window_size == 128, (
            "Dual-pool sparse MLA requires window_size == 128, the KV tile "
            f"size of the FMHA kernel (got {self.window_size})."
        )
        self.compress_ratio_set = set(self.compress_ratios)
        # Sorted for a deterministic iteration order across torch.compile traces.
        self._compress_ratios_sorted = sorted(self.compress_ratio_set)
        unsupported = self.compress_ratio_set - _SUPPORTED_COMPRESS_RATIOS
        assert not unsupported, (
            f"Unsupported compress ratios {unsupported}. Only "
            f"{set(_SUPPORTED_COMPRESS_RATIOS)} are supported."
        )
        ratios = self._compress_ratios_sorted
        max_sequences = config.max_num_sequences
        max_tokens = config.max_num_tokens
        max_blocks = manager.max_blocks_per_seq

        # Per-step host values, per compress ratio.
        self.num_total_compressed_tokens: Dict[int, int] = {}
        self.max_ctx_compressed_tokens: Dict[int, int] = {}
        self._ctx_output_sizes: Optional[Dict[int, int]] = None
        self.num_gen_tokens_per_seq = 0
        self.gen_new_tokens_per_seq = None
        self.nvfp4_compress_fp8_scratch: Optional[torch.Tensor] = None

        def per_ratio(name: str, shape: tuple, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
            return {ratio: self.get_buffer(shape, dtype, f"{name}_{ratio}") for ratio in ratios}

        # Cumulative query lengths of the requests.
        self.cu_seq_lens_cuda = self.get_buffer((max_sequences + 1,), torch.int, "cu_seq_lens_cuda")
        self.cu_seq_lens = torch.empty_like(
            self.cu_seq_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.cu_seq_lens[0] = 0
        # Compressed rows of each request: new this step, their prefix sums,
        # the total after this step and the total before it.
        self.new_comp_kv_lens_cuda = per_ratio("new_comp_kv_lens_cuda", (max_sequences,), torch.int)
        self.cu_new_comp_kv_cuda = per_ratio("cu_new_comp_kv_cuda", (max_sequences + 1,), torch.int)
        self.compressed_kv_lens_cuda = per_ratio(
            "compressed_kv_lens_cuda", (max_sequences,), torch.int
        )
        self.past_kv_lens_cuda = per_ratio("past_kv_lens_cuda", (max_sequences,), torch.int)
        # Position of each new compressed row, and whether its slot holds a
        # row (padded generation slots do not).
        self.compressed_position_ids_cuda = per_ratio(
            "compressed_position_ids_cuda", (max_tokens,), torch.int
        )
        self.compressed_mask_cuda = per_ratio("compressed_mask_cuda", (max_tokens,), torch.bool)
        # Device scalars of the shape-static compiled helpers: the batch
        # size, and (num_contexts, num_generations, tokens per generation).
        self.batch_size_cuda = self.get_buffer((1,), torch.int, "batch_size_cuda")
        self.gen_position_params_cuda = self.get_buffer((3,), torch.int, "gen_position_params_cuda")
        # All -1: the selection of a layer without compressed rows.
        self.empty_topk_indices_buffer = None
        if self.has_indexer:
            self.empty_topk_indices_buffer = self.get_buffer(
                (max_tokens, self.sparse_mla_topk), torch.int32, "empty_topk_indices_buffer"
            )
            self.empty_topk_indices_buffer.fill_(-1)
        # Window positions of each query token, -1 past its own position.
        self.swa_local_indices_cuda = self.get_buffer(
            (max_tokens, self.window_size), torch.int32, "swa_local_indices_cuda"
        )
        # Compressed rows a ratio-128 layer reads: all of them, with the row
        # count rounded up to a power of two for CUDA graphs. A ratio-4 layer
        # reads the indexer's top-k instead.
        num_rows_128 = math.ceil(manager.max_seq_len / 128)
        self.max_compressed_indices = {
            1: 0,
            4: self.sparse_mla_topk,
            128: 1 << (num_rows_128 - 1).bit_length() if num_rows_128 > 0 else 1,
        }
        self.compressed_local_indices_cuda = self.get_buffer(
            (max_tokens, self.max_compressed_indices[128] if 128 in self.compress_ratio_set else 1),
            torch.int32,
            "compressed_local_indices_cuda",
        )
        # Page tables of the sliding-window caches, [local layer, attention
        # type, sequence, block], and of the compressed rows, [sequence, block].
        self.sliding_block_tables = self.get_buffer(
            (
                manager.num_local_layers,
                len(DEEPSEEK_V4_SLIDING_ATTENTION),
                max_sequences,
                max_blocks,
            ),
            torch.int32,
            "sliding_block_tables",
        )
        self.compress_block_tables = {
            ratio: self.get_buffer(
                (max_sequences, max_blocks), torch.int32, f"compress_block_tables_{ratio}"
            )
            for ratio in ratios
            if is_compress_layer(ratio)
        }
        # Rows each query token attends: the window plus the compressed
        # rows, and the two parts on their own.
        self.sparse_mla_topk_lens = per_ratio("sparse_mla_topk_lens", (max_tokens,), torch.int32)
        self.sparse_mla_swa_lens = self.get_buffer(
            (max_tokens,), torch.int32, "sparse_mla_swa_lens"
        )
        self.sparse_mla_extra_lens = {
            ratio: self.get_buffer((max_tokens,), torch.int32, f"sparse_mla_extra_lens_{ratio}")
            for ratio in ratios
            if ratio > 1
        }
        # Tokens each request had in the cache before this step.
        self.cached_token_lens_cuda = self.get_buffer(
            (max_sequences,), torch.int32, "cached_token_lens_cuda"
        )
        self.cached_token_lens_cpu = torch.empty_like(
            self.cached_token_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )

        # Base addresses of the cache pools; constant once the cache exists.
        # The keys of the per-layer maps are global layer indices.
        from ..backends.sparse.deepseek_v4.params import DeepseekV4AttentionType

        self.sparse_mla_base_ptrs = {1: manager.swa_pool_ptr}
        self.sparse_mla_base_ptrs.update(manager.compress_pool_ptrs)
        self.swa_buffer_ptrs = {layer: manager.swa_pool_ptr for layer in manager.pp_layers}
        self.compressed_buffer_ptrs = {
            layer: manager.get_buffers(layer, DeepseekV4AttentionType.COMPRESS).data_ptr()
            for layer in manager.pp_layers
            if is_compress_layer(manager._compress_ratios[layer])
        }

    def prepare(self, inputs: "BatchInputs") -> None:
        if inputs.stage_host:
            self._stage_host(inputs)
        self._derive()

    def _stage_host(self, inputs: "BatchInputs") -> None:
        """Copy this batch's page tables and host lengths to the device."""
        from ..backends.sparse.dsa.indexer import Indexer

        core = self._core
        manager = core.kv_cache_manager
        request_ids = core.request_ids
        num_contexts = core.num_contexts
        num_seqs = core.num_seqs

        self.cached_token_lens_cpu[:num_seqs] = torch.tensor(
            inputs.kv_cache_params.num_cached_tokens_per_seq[:num_seqs], dtype=torch.int32
        )
        self.cached_token_lens_cuda[:num_seqs].copy_(
            self.cached_token_lens_cpu[:num_seqs], non_blocking=True
        )
        kv_lens = self.cached_token_lens_cpu[:num_seqs] + core.seq_lens

        self._prepare_req_idx_per_token()
        # The core already had the manager compute the sliding-window tables.
        manager.copy_batch_sliding_block_tables(
            self.sliding_block_tables, request_ids, num_contexts, num_seqs
        )
        for ratio, block_table in self.compress_block_tables.items():
            manager.copy_batch_compress_block_tables(
                block_table,
                request_ids,
                compress_ratio=ratio,
                beam_width=core.beam_width,
                num_contexts=num_contexts,
                num_seqs=num_seqs,
            )
        if self.has_indexer:
            manager.copy_batch_indexer_compress_block_tables(
                self.host_indexer_k_cache_block_offsets,
                request_ids,
                beam_width=core.beam_width,
                num_contexts=num_contexts,
                num_seqs=num_seqs,
            )
            self.indexer_k_cache_block_offsets[:num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[:num_seqs], non_blocking=True
            )
            # Columns past a sequence's blocks hold -1. Padded CUDA graph
            # slots still compute addresses from them, so map them to block 0.
            self.indexer_k_cache_block_offsets.clamp_(min=0)
            self._prepare_dsl_split(kv_lens)
            Indexer.prepare(metadata=core)

        # Context compressed-row counts as host ints, so that no device
        # scalar is read for the size of the context position ids.
        self._ctx_output_sizes = None
        if num_contexts > 0:
            self._ctx_output_sizes = {}
            cached = self.cached_token_lens_cpu[:num_seqs]
            for ratio in self._compress_ratios_sorted:
                new_rows = kv_lens // ratio - cached // ratio
                self._ctx_output_sizes[ratio] = int(new_rows[:num_contexts].sum())
                self.max_ctx_compressed_tokens[ratio] = int(new_rows[:num_contexts].max())

    def _derive(self) -> None:
        """Derive the device state from the core's current lengths."""
        # The compiled helpers live with the old metadata class, so both
        # metadata paths share one compiled graph each.
        from ..backends.sparse.deepseek_v4.metadata import (
            DeepseekV4TrtllmAttentionMetadata as helpers,
        )

        core = self._core
        self._derive_from_lengths()
        num_seqs = core.num_seqs
        num_tokens = core.num_tokens
        num_contexts = core.num_contexts
        num_generations = core.num_generations
        ratios = self._compress_ratios_sorted
        kv_lens = core.kv_lens_cuda[:num_seqs]
        seq_lens = core.seq_lens_cuda
        cached_tokens = kv_lens - seq_lens

        # Host values that follow from the step's shape.
        num_gen_tokens = num_tokens - core.num_ctx_tokens
        self.num_gen_tokens_per_seq = (
            num_gen_tokens // num_generations if num_generations > 0 else 0
        )
        for ratio in ratios:
            num_gen_rows = num_generations * ((self.num_gen_tokens_per_seq + ratio - 1) // ratio)
            if num_contexts > 0:
                num_gen_rows += self._ctx_output_sizes[ratio]
            else:
                self.max_ctx_compressed_tokens[ratio] = 0
            self.num_total_compressed_tokens[ratio] = num_gen_rows

        helpers._compute_per_ratio_kv_lens(
            kv_lens,
            cached_tokens,
            num_seqs,
            self.compressed_kv_lens_cuda,
            self.past_kv_lens_cuda,
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            ratios,
        )
        if num_contexts > 0:
            # Extend rows get their lengths corrected on the device, which
            # the host counts do not see.
            ctx_output_sizes = (
                self._ctx_output_sizes if core.num_chunked_ctx_requests == 0 else None
            )
            helpers._compute_ctx_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.cu_new_comp_kv_cuda,
                self.compressed_position_ids_cuda,
                num_contexts,
                ratios,
                ctx_output_sizes,
            )
        if self.num_gen_tokens_per_seq > 0 and num_generations > 0:
            gen_params = self.gen_position_params_cuda
            gen_params[0].fill_(num_contexts)
            gen_params[1].fill_(num_generations)
            gen_params[2].fill_(self.num_gen_tokens_per_seq)
            helpers._compute_gen_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.cu_new_comp_kv_cuda,
                self.compressed_position_ids_cuda,
                gen_params,
                ratios,
            )
        self.batch_size_cuda.fill_(num_seqs)
        helpers._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            self.batch_size_cuda,
            ratios,
        )

        token_positions = helpers._compute_token_positions(
            seq_lens,
            cached_tokens,
            num_seqs,
            num_tokens,
            self.cu_seq_lens_cuda,
            self.req_idx_per_token,
        )
        self.token_positions_cuda[:num_tokens] = token_positions
        helpers._prepare_deepseek_v4_indices_compiled(
            token_positions,
            self.window_size,
            self.compressed_local_indices_cuda.shape[1],
            self.sparse_mla_topk,
            self.swa_local_indices_cuda,
            self.compressed_local_indices_cuda,
            self.sparse_mla_topk_lens,
            ratios,
            self.sparse_mla_swa_lens,
            self.sparse_mla_extra_lens,
        )
