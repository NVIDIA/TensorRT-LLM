# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime metadata for DeepSeek-V4 sparse attention."""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import prefer_pinned

from ..dsa.metadata import DSAtrtllmAttentionMetadata
from .indexer import DeepseekV4Indexer
from .params import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    DeepSeekV4MetadataParams,
    is_compress_layer,
)


class DeepseekV4TrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    # The set of compress ratios for the layers
    compress_ratio_set: Set[int]
    # The set of (compress ratio, attention type) for the layers
    attention_type_set: Set[Tuple[int, DeepseekV4AttentionType]]
    # The number of total compressed tokens for each compress ratio
    num_total_compressed_tokens: Dict[int, int]
    # The max number of context compressed tokens for each compress ratio
    max_ctx_compressed_tokens: Dict[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.num_total_compressed_tokens = {}
        self.max_ctx_compressed_tokens = {}
        self._ctx_output_sizes: Optional[Dict[int, int]] = None
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DeepSeekV4MetadataParams):
            raise ValueError("DeepSeek-V4 sparse attention metadata params are not set")
        self.window_size = sparse_metadata_params.window_size
        window_size = self.window_size
        assert window_size == 128, (
            f"Dual-pool sparse MLA requires window_size == 128, which equals to the"
            f"TileSizeKV of the FMHA kernel. (got {window_size})."
        )
        capture_graph = self.is_cuda_graph
        self.compress_ratio_set = set(self.compress_ratios)
        # Cache a sorted list for deterministic iteration order across
        # torch.compile traces. Avoids repeated set→list conversion and
        # guarantees stable specialization keys.
        self._compress_ratios_sorted = sorted(self.compress_ratio_set)
        _supported_ratios = {1, 4, 128}
        _unsupported = self.compress_ratio_set - _supported_ratios
        assert not _unsupported, (
            f"Unsupported compress ratios {_unsupported}. Only {_supported_ratios} are supported."
        )

        attention_types = []
        for compress_ratio in self.compress_ratio_set:
            if compress_ratio == 1:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
            elif compress_ratio == 4:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESS))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_SCORE))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESS))
                attention_types.append(
                    (compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV)
                )
                attention_types.append(
                    (compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE)
                )
            else:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESS))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_SCORE))
        self.attention_type_set = set(attention_types)

        # Create buffers for the compressor
        # cu_seq_lens_cuda is the cumulative sequence lengths for the requests
        self.cu_seq_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int,
            cache_name="cu_seq_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cu_seq_lens = torch.empty_like(
            self.cu_seq_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.cu_seq_lens[0] = 0

        # new_comp_kv_lens_cuda is the number of new compressed tokens for the requests
        self.new_comp_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"new_comp_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # cu_new_comp_kv_cuda is the cumulative number of new compressed tokens for the requests
        self.cu_new_comp_kv_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences + 1,),
                dtype=torch.int,
                cache_name=f"cu_new_comp_kv_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_kv_lens_cuda is the number of compressed tokens for the requests
        self.compressed_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"compressed_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # past_kv_lens_cuda is the number of past compressed tokens for the requests
        self.past_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"past_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_position_ids_cuda is the compressed position ids for the requests
        self.compressed_position_ids_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                dtype=torch.int,
                cache_name=f"compressed_position_ids_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_mask_cuda: per-token bool mask for postprocess scatter.
        # Precomputed from new_comp_kv_lens to skip padded generation slots.
        self.compressed_mask_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                dtype=torch.bool,
                cache_name=f"compressed_mask_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.compressed_mask = {
            compress_ratio: torch.empty_like(
                self.compressed_mask_cuda[compress_ratio], device="cpu", pin_memory=prefer_pinned()
            )
            for compress_ratio in self.compress_ratio_set
        }

        # empty topk indices buffer with all -1s in the tensor
        self.empty_topk_indices_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.sparse_mla_topk),
            cache_name="empty_topk_indices_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.empty_topk_indices_buffer.fill_(-1)

        # SWA local indices
        self.swa_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.window_size),
            cache_name="swa_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Compute max_compressed_indices for CUDA graph compatibility.
        # For ratio=4, the indexer selects index_topk compressed tokens.
        # For ratio=128, use max_seq_len / 128 rounded up to next power of 2
        raw_128 = math.ceil(self.max_seq_len / 128)
        po2_128 = 1 << (raw_128 - 1).bit_length() if raw_128 > 0 else 1
        self.max_compressed_indices = {
            1: 0,  # No compressed indices
            4: self.sparse_mla_topk,  # index_topk from indexer
            128: po2_128,  # All compressed tokens, rounded to power of 2
        }

        # Compressed local indices for compress_ratio=128
        # Note: ratio=4 uses dynamic topk_indices from indexer, so we only pre-allocate for ratio=128
        self.compressed_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.max_compressed_indices[128]),
            cache_name="compressed_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Sliding-window caches use per-layer page indices and are indexed by
        # [local_layer_idx, attention_type.value, sequence, block]. COMPRESS
        # uses ratio-shared page indices, and INDEXER_COMPRESS keeps a separate
        # compatibility table for the generic DSA indexer path.
        block_table_shape = (
            self.kv_cache_manager.num_local_layers,
            len(DEEPSEEK_V4_SLIDING_ATTENTION),
            self.max_num_sequences,
            self.kv_cache_manager.max_blocks_per_seq,
        )
        self.sliding_block_tables = self.get_empty(
            self.cuda_graph_buffers,
            block_table_shape,
            cache_name="sliding_block_tables",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        compress_block_table_shape = (
            self.max_num_sequences,
            self.kv_cache_manager.max_blocks_per_seq,
        )
        self.compress_block_tables = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                compress_block_table_shape,
                cache_name=f"compress_block_tables_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
            if is_compress_layer(compress_ratio)
        }

        # sparse_mla_topk_lens: actual token count per token for each compress_ratio (SWA + compressed)
        # Shape: [max_num_tokens] per compress_ratio
        self.sparse_mla_topk_lens = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"sparse_mla_topk_lens_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # cached_token_lens_cuda: number of tokens already cached per request
        self.cached_token_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            cache_name="cached_token_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cached_token_lens_cpu = torch.empty_like(
            self.cached_token_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )

        # Cache buffer data pointers are constant after KV cache allocation,
        # so compute them once during initialization instead of every prepare().
        self._init_cache_buffer_data_pointers()

    def prepare_for_indexer_k_cache(self):
        """Prepare the shared indexer K-cache decode table for DSA kernels."""
        # INDEXER_COMPRESS uses shared page indices, so the generic DSA
        # indexer path only needs one 2D block table.
        self.kv_cache_manager.copy_batch_indexer_compress_block_tables(
            self.host_indexer_k_cache_block_offsets,
            self.request_ids,
            beam_width=self.beam_width,
            num_contexts=self.num_contexts,
            num_seqs=self.num_seqs,
        )
        self.indexer_k_cache_block_offsets[: self.num_seqs].copy_(
            self.host_indexer_k_cache_block_offsets[: self.num_seqs],
            non_blocking=True,
        )
        # Columns beyond each sequence's allocated indexer blocks contain BAD_PAGE_INDEX (-1).
        # CUDA-graph padded token slots may still compute scatter addresses from those columns
        # before being ignored, so map them to block 0, matching the base DSA metadata path.
        self.indexer_k_cache_block_offsets.clamp_(min=0)

    def prepare_for_block_tables(self):
        """Prepare block tables for sliding-window and compressed attention."""
        self.kv_cache_manager.copy_batch_sliding_block_tables(
            self.sliding_block_tables,
            self.request_ids,
            self.num_contexts,
            self.num_seqs,
        )
        for compress_ratio, compress_block_table in self.compress_block_tables.items():
            self.kv_cache_manager.copy_batch_compress_block_tables(
                compress_block_table,
                self.request_ids,
                compress_ratio=compress_ratio,
                beam_width=self.beam_width,
                num_contexts=self.num_contexts,
                num_seqs=self.num_seqs,
            )

    def prepare_for_deepseek_v4_indices(self, token_positions=None):
        """Prepare SWA/compressed local indices and sparse_mla_topk_lens."""
        window_size = self.window_size
        device = self.swa_local_indices_cuda.device

        if token_positions is None:
            # Initial prepare() path — build token_positions from CPU data.
            num_tokens = self.num_tokens
            num_requests = self.num_seqs

            # cu_seq_lens_cuda must already be populated before this call
            cu_seq_lens = self.cu_seq_lens_cuda[: num_requests + 1]
            cached_tokens = self.cached_token_lens_cuda[:num_requests]

            token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
            req_idx = torch.searchsorted(cu_seq_lens[1:].to(torch.int32), token_idx, right=True)
            offsets = token_idx - cu_seq_lens[req_idx].to(torch.int32)
            token_positions = cached_tokens[req_idx].to(torch.int32) + offsets

        self._prepare_deepseek_v4_indices_compiled(
            token_positions,
            window_size,
            self.max_compressed_indices[128],
            self.sparse_mla_topk,
            self.swa_local_indices_cuda,
            self.compressed_local_indices_cuda,
            self.sparse_mla_topk_lens,
            self._compress_ratios_sorted,
        )

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _prepare_deepseek_v4_indices_compiled(
        token_positions: torch.Tensor,
        window_size: int,
        max_compressed_indices_128: int,
        sparse_mla_topk: int,
        swa_local_indices_buf: torch.Tensor,
        compressed_local_indices_buf: torch.Tensor,
        sparse_mla_topk_lens_bufs: Dict[int, torch.Tensor],
        compress_ratios: list,
    ):
        """Build SWA indices, compressed indices, and topk_lens in one fused graph."""
        device = token_positions.device
        num_tokens = token_positions.shape[0]

        # ── SWA local indices ──
        positions = token_positions.unsqueeze(1)  # [num_tokens, 1]
        swa_offsets = torch.arange(window_size, dtype=torch.int32, device=device)
        swa_start = (positions - window_size + 1).clamp(min=0)
        swa_indices = swa_start + swa_offsets
        swa_indices = torch.where(swa_indices > positions, -1, swa_indices).to(torch.int32)
        swa_local_indices_buf[:num_tokens] = swa_indices

        # ── Compressed local indices (ratio=128) ──
        # Hardcoded 128: compressed_local_indices only applies to ratio=128
        # layers. This is a design constraint — see compress_ratio_set
        # validation in __post_init__ which restricts ratios to {1, 4, 128}.
        num_valid = (token_positions + 1) // 128
        comp_col = torch.arange(max_compressed_indices_128, dtype=torch.int32, device=device)
        valid_mask = comp_col.unsqueeze(0) < num_valid.unsqueeze(1)
        comp_indices = torch.where(
            valid_mask,
            comp_col.unsqueeze(0).expand(num_tokens, -1),
            torch.full(
                (num_tokens, max_compressed_indices_128), -1, dtype=torch.int32, device=device
            ),
        )
        compressed_local_indices_buf[:num_tokens] = comp_indices

        # ── sparse_mla_topk_lens per compress_ratio ──
        # Dual-pool layout:
        # - compress_ratio=1: min(kv_len, window_size) — actual valid SWA count
        # - compress_ratio=4: window_size + min(kv_len // 4, sparse_mla_topk)
        #   (fixed 128 SWA slots + compressed valid count)
        # - compress_ratio=128: window_size + kv_len // 128
        #   (fixed 128 SWA slots + compressed valid count)
        # For ratio>1, the SWA region always occupies exactly window_size (128)
        # slots. Invalid SWA positions are padded with -1 in the index buffer.
        kv_lens = token_positions + 1
        for compress_ratio in compress_ratios:
            if compress_ratio == 1:
                total_count = kv_lens.clamp(max=window_size)
            elif compress_ratio == 4:
                compressed_count = (kv_lens // compress_ratio).clamp(max=sparse_mla_topk)
                total_count = window_size + compressed_count
            elif compress_ratio == 128:
                compressed_count = kv_lens // compress_ratio
                total_count = window_size + compressed_count
            else:
                raise ValueError(f"Unsupported compress_ratio: {compress_ratio}")
            sparse_mla_topk_lens_bufs[compress_ratio][:num_tokens] = total_count.to(torch.int32)

    def _init_cache_buffer_data_pointers(self):
        # If MTP is enabled, enlarge the compress ratios by max_draft_tokens - 1
        extend_compress_ratios = self.compress_ratios + [self.compress_ratios[-1]] * (
            self.max_draft_tokens - 1
        )
        # SWA uses PER_LAYER indices; COMPRESS uses SHARED indices. The sparse
        # MLA conversion kernel receives a representative base pointer per pool
        # and a per-layer buffer pointer so it can account for any layer offset.
        self.sparse_mla_base_ptrs = {
            1: self.kv_cache_manager.swa_pool_ptr,
        }
        for ratio, compress_pool_ptr in self.kv_cache_manager.compress_pool_ptrs.items():
            self.sparse_mla_base_ptrs[ratio] = compress_pool_ptr

        self.swa_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.swa_pool_ptr
            for layer_idx in self.kv_cache_manager.pp_layers
        }
        self.compressed_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.get_buffers(
                layer_idx, DeepseekV4AttentionType.COMPRESS
            ).data_ptr()
            for layer_idx in self.kv_cache_manager.pp_layers
            if is_compress_layer(extend_compress_ratios[layer_idx])
        }

    def prepare(self):
        assert self.kv_cache_manager is not None
        assert self.request_ids is not None

        self.kv_cache_manager.compute_sliding_block_tables(
            self.request_ids,
            self.num_contexts,
        )

        TrtllmAttentionMetadata.prepare(self)

        num_requests = self.num_contexts + self.num_generations
        num_gen_tokens = self.num_tokens - self.num_ctx_tokens

        self.cached_token_lens_cpu[:num_requests] = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq[:num_requests],
            dtype=torch.int32,
        )
        cached_token_lens = self.cached_token_lens_cpu
        kv_lens = cached_token_lens[:num_requests] + self.seq_lens_kv[:num_requests]

        self.cached_token_lens_cuda[:num_requests].copy_(
            cached_token_lens[:num_requests], non_blocking=True
        )

        # Prepare cu_seq_lens early — needed by prepare_for_deepseek_v4_indices
        self.cu_seq_lens[1 : num_requests + 1] = self.seq_lens.cumsum(0)
        self.cu_seq_lens_cuda[: num_requests + 1].copy_(
            self.cu_seq_lens[: num_requests + 1], non_blocking=True
        )

        # For indices conversion
        self.prepare_for_indices_conversion()

        has_sparse_layers = DEEPSEEK_V4_SPARSE_RATIO in self.compress_ratio_set

        # For block offsets
        self.prepare_for_block_tables()

        # For indexer k cache (only needed when sparse layers exist)
        if has_sparse_layers:
            self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # For DeepSeek-V4 indices
        self.prepare_for_deepseek_v4_indices()

        # Prepare metadata for indexer (only needed when sparse layers exist)
        if has_sparse_layers:
            DeepseekV4Indexer.prepare(metadata=self)

        # --- Per-ratio metadata ---
        # 1) CPU-side: compute scalar metadata (num_total_compressed_tokens, etc.)
        # 2) CUDA-side: fill *_cuda buffers via prepare_compressed_kv_metadata()
        num_gen_tokens_per_seq = (
            num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
        )
        self.num_gen_tokens_per_seq = num_gen_tokens_per_seq
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        kv_lens_slice = kv_lens[:num_requests]
        cached_slice = cached_token_lens[:num_requests]

        # Host-side per-ratio ctx compressed-token counts (Python ints), so
        # _compute_ctx_compressed_position_ids never reads a device scalar
        # (implicit D2H + stream sync) for its arange size / slice bound.
        ctx_output_sizes: Optional[Dict[int, int]] = None
        if num_contexts > 0:
            # Prefill path: need per-request tensor ops for ctx scalar metadata.
            ctx_output_sizes = {}
            for compress_ratio in self.compress_ratio_set:
                new_comp_kv_lens = kv_lens_slice // compress_ratio - cached_slice // compress_ratio
                cu_new = new_comp_kv_lens.cumsum(0)
                num_ctx_compressed_tokens = cu_new[num_contexts - 1].item()
                ctx_output_sizes[compress_ratio] = num_ctx_compressed_tokens
                num_gen_compressed_tokens = num_generations * (
                    (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
                )
                self.num_total_compressed_tokens[compress_ratio] = (
                    num_ctx_compressed_tokens + num_gen_compressed_tokens
                )
                self.max_ctx_compressed_tokens[compress_ratio] = (
                    new_comp_kv_lens[:num_contexts].max().item()
                )
        else:
            # Decode-only: scalars depend only on num_generations and
            # num_gen_tokens_per_seq, no per-request tensor ops needed.
            for compress_ratio in self.compress_ratio_set:
                self.num_total_compressed_tokens[compress_ratio] = num_generations * (
                    (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
                )
                self.max_ctx_compressed_tokens[compress_ratio] = 0

        # Cached for on_update_kv_lens(); see the reuse gate there.
        self._ctx_output_sizes = ctx_output_sizes

        # 2) CUDA-side: fill *_cuda buffers on device.
        kv_lens_cuda = (
            self.cached_token_lens_cuda[:num_requests] + self._seq_lens_cuda[:num_requests]
        )
        cached_tokens_cuda = self.cached_token_lens_cuda[:num_requests]
        self.prepare_compressed_kv_metadata(kv_lens_cuda, cached_tokens_cuda, ctx_output_sizes)

        self._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            num_requests,
            self.num_total_compressed_tokens,
            self._compress_ratios_sorted,
        )

    def prepare_compressed_kv_metadata(
        self,
        kv_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
        ctx_output_sizes: Optional[Dict[int, int]] = None,
    ):
        """Compute per-ratio compressed KV lens and position IDs on device.

        Shared by prepare() and on_update_kv_lens() to avoid duplicated logic.

        Args:
            kv_lens: Total KV lengths per request (device tensor, [batch_size]).
            cached_tokens: Cached token counts per request (device tensor, [batch_size]).
            ctx_output_sizes: Optional per-ratio host-computed ctx
                compressed-token counts (Python ints); avoids implicit
                device-scalar reads (D2H + stream sync) in the ctx position-id
                computation. prepare() always passes it; on_update_kv_lens()
                reuses the cached copy unless the extend_ctx path may have
                mutated ctx-row kv_lens on device.
        """
        batch_size = kv_lens.shape[0]
        num_contexts = self.num_contexts
        num_generations = self.num_generations

        self._compute_per_ratio_kv_lens(
            kv_lens,
            cached_tokens,
            batch_size,
            self.compressed_kv_lens_cuda,
            self.past_kv_lens_cuda,
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self._compress_ratios_sorted,
        )

        if num_contexts > 0:
            self._compute_ctx_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.cu_new_comp_kv_cuda,
                self.compressed_position_ids_cuda,
                num_contexts,
                self._compress_ratios_sorted,
                ctx_output_sizes,
            )

        if self.num_gen_tokens_per_seq > 0 and num_generations > 0:
            # Extract output_offset as Python int per ratio to avoid
            # tensor-scalar slice inside compiled function.
            # For decode-only batches (num_contexts == 0), offset is 0.
            gen_output_offsets = {
                r: self.cu_new_comp_kv_cuda[r][num_contexts].item() if num_contexts > 0 else 0
                for r in self._compress_ratios_sorted
            }
            self._compute_gen_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.cu_new_comp_kv_cuda,
                self.compressed_position_ids_cuda,
                num_contexts,
                num_generations,
                self.num_gen_tokens_per_seq,
                self._compress_ratios_sorted,
                gen_output_offsets,
            )

    def on_update_kv_lens(self):
        """Recompute kv-lens-dependent DeepSeek-V4 metadata on device."""
        super().on_update_kv_lens()

        batch_size = self.num_seqs
        num_tokens = self.num_tokens
        kv_lens = self.kv_lens_cuda[:batch_size]
        seq_lens = self._seq_lens_cuda[:batch_size]
        cached_tokens = kv_lens - seq_lens

        num_gen_tokens = num_tokens - self.num_ctx_tokens
        self.num_gen_tokens_per_seq = (
            num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
        )

        # Reuse prepare()'s host-computed ctx sizes unless the extend_ctx path
        # (num_chunked_ctx_requests > 0) may have mutated ctx-row kv_lens on
        # device; every other path only changes gen rows.
        ctx_output_sizes = (
            self._ctx_output_sizes if getattr(self, "num_chunked_ctx_requests", 0) == 0 else None
        )
        self.prepare_compressed_kv_metadata(kv_lens, cached_tokens, ctx_output_sizes)

        self._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            batch_size,
            self.num_total_compressed_tokens,
            self._compress_ratios_sorted,
        )

        token_positions = self._compute_token_positions(
            seq_lens,
            cached_tokens,
            batch_size,
            num_tokens,
            self.cu_seq_lens_cuda,
            self.req_idx_per_token,
        )

        self.prepare_for_deepseek_v4_indices(token_positions)

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_per_ratio_kv_lens(
        kv_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
        batch_size: int,
        compressed_kv_lens_bufs: Dict[int, torch.Tensor],
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        new_comp_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compress_ratios: list,
    ):
        """Compute per-ratio compressed/past/new kv lens and cu_new_comp."""
        for compress_ratio in compress_ratios:
            compressed_kv = (kv_lens // compress_ratio).to(torch.int)
            compressed_kv_lens_bufs[compress_ratio][:batch_size] = compressed_kv

            past_kv = (cached_tokens // compress_ratio).to(torch.int)
            past_kv_lens_bufs[compress_ratio][:batch_size] = past_kv

            new_comp = compressed_kv - past_kv
            new_comp_kv_lens_bufs[compress_ratio][:batch_size] = new_comp

            cu_new_comp = cu_new_comp_kv_bufs[compress_ratio]
            cu_new_comp[: batch_size + 1] = torch.nn.functional.pad(
                torch.cumsum(new_comp, dim=0), (1, 0)
            )

    @staticmethod
    def _compute_token_positions(
        seq_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        cu_seq_lens_buf: torch.Tensor,
        req_idx_per_token_buf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cu_seq_lens, req_idx_per_token, and token_positions (eager)."""
        device = seq_lens.device

        # cu_seq_lens
        cu_seq_lens_buf[: batch_size + 1] = torch.nn.functional.pad(
            torch.cumsum(seq_lens.to(torch.int), dim=0), (1, 0)
        )

        # req_idx_per_token via searchsorted
        token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
        req_idx = torch.searchsorted(
            cu_seq_lens_buf[1 : batch_size + 1].to(torch.int32), token_idx, right=True
        )
        req_idx_per_token_buf[:num_tokens] = req_idx

        # token positions
        base_pos = cached_tokens[req_idx].to(torch.int32)
        offsets = token_idx - cu_seq_lens_buf[req_idx].to(torch.int32)
        return base_pos + offsets

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_gen_compressed_position_ids(
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compressed_position_ids_bufs: Dict[int, torch.Tensor],
        num_contexts: int,
        num_generations: int,
        num_gen_tokens_per_seq: int,
        compress_ratios: list,
        gen_output_offsets: Dict[int, int],
    ):
        """Generation position IDs in exact compact compressor output order.

        The decode kernel packs valid outputs according to ``cu_new_comp_kv``.
        The corresponding request and local offset are recovered for every
        compact output index. Reserved slots after the compact prefix are masked
        during postprocess, so their position IDs are irrelevant.

        gen_output_offsets: dict mapping compress_ratio -> Python int offset,
        pre-extracted by the caller to avoid tensor-scalar .item() inside
        compiled code.  0 for decode-only batches."""
        device = past_kv_lens_bufs[compress_ratios[0]].device
        batch_size = num_contexts + num_generations
        for compress_ratio in compress_ratios:
            new_gen_comp = (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
            gen_comp = num_generations * new_gen_comp
            output_offset = gen_output_offsets[compress_ratio]
            cu_new_comp = cu_new_comp_kv_bufs[compress_ratio]
            output_idx = torch.arange(gen_comp, dtype=torch.int32, device=device) + output_offset
            # Search the zero-offset prefix: Inductor miscompiles searchsorted on cu_new_comp[1:].
            req_idx = torch.searchsorted(cu_new_comp[: batch_size + 1], output_idx, right=True) - 1
            req_idx = req_idx.clamp(min=num_contexts, max=batch_size - 1)
            offset_in_req = output_idx - cu_new_comp[req_idx]
            past_kv_lens = past_kv_lens_bufs[compress_ratio]
            result = ((past_kv_lens[req_idx] + offset_in_req) * compress_ratio).to(torch.int)
            compressed_position_ids_bufs[compress_ratio][
                output_offset : output_offset + gen_comp
            ] = result

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_compressed_mask(
        new_comp_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compressed_mask_bufs: Dict[int, torch.Tensor],
        batch_size: int,
        num_total_compressed_tokens: Dict[int, int],
        compress_ratios: list,
    ):
        """Compute per-token compressed_mask on device (graph-safe, no .item()).

        For each token in [0, total_tokens), determine which sequence it
        belongs to via searchsorted on cu_new_comp_kv, then compare its
        within-sequence offset against the actual new_comp_kv_len.
        Context tokens (offset < actual) are always True.
        Generation tokens whose offset >= actual new_comp are padding → False.
        """
        device = new_comp_kv_lens_bufs[compress_ratios[0]].device
        for compress_ratio in compress_ratios:
            total_tokens = num_total_compressed_tokens[compress_ratio]
            new_comp = new_comp_kv_lens_bufs[compress_ratio][:batch_size]
            cu = cu_new_comp_kv_bufs[compress_ratio][: batch_size + 1]

            token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
            seq_idx = torch.searchsorted(cu[1:], token_idx, right=True)
            seq_idx = seq_idx.clamp_(max=batch_size - 1)
            offset_in_seq = token_idx - cu[seq_idx]
            compressed_mask_bufs[compress_ratio][:total_tokens] = offset_in_seq < new_comp[seq_idx]

    @staticmethod
    def _compute_ctx_compressed_position_ids(
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compressed_position_ids_bufs: Dict[int, torch.Tensor],
        num_contexts: int,
        compress_ratios: list,
        ctx_output_sizes: Optional[Dict[int, int]] = None,
    ):
        """Context-only compressed position IDs (eager, data-dependent shapes).

        ctx_output_sizes (host ints) keeps the arange size and slice bound off
        the device; the 0-dim-CUDA fallback costs two implicit D2H syncs per
        ratio.
        """
        device = past_kv_lens_bufs[compress_ratios[0]].device
        for compress_ratio in compress_ratios:
            past_kv = past_kv_lens_bufs[compress_ratio]
            cu_new_comp = cu_new_comp_kv_bufs[compress_ratio]

            total_ctx_comp = (
                ctx_output_sizes[compress_ratio]
                if ctx_output_sizes is not None
                else cu_new_comp[num_contexts]
            )
            ctx_idx = torch.arange(total_ctx_comp, dtype=torch.int32, device=device)
            ctx_cu = cu_new_comp[: num_contexts + 1].to(torch.int32)
            ctx_req = torch.searchsorted(ctx_cu[1:], ctx_idx, right=True)
            ctx_offset = ctx_idx - ctx_cu[ctx_req]
            compressed_position_ids_bufs[compress_ratio][:total_ctx_comp] = (
                (past_kv[:num_contexts][ctx_req] + ctx_offset) * compress_ratio
            ).to(torch.int)
