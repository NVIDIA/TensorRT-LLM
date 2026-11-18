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
"""Indexer and top-k state of DeepSeek Sparse Attention (DSA)."""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
from tensorrt_llm.logger import logger

from .base import UnsupportedMetadataConfiguration, _Registry
from .core import TrtllmCoreMetadata
from .mla_context import MlaContextMetadata
from .sparse import SparseMetadata

if TYPE_CHECKING:
    from ..backends.sparse.dsa.indexer import IndexerPrefillChunkMetadata
    from .batch_inputs import BatchInputs

# Indexer MQA logits are always fp32. The dtype is part of the CuTe DSL top-k
# compile key, so warmup must use the runtime dtype.
_INDEXER_LOGITS_DTYPE = torch.float32
_MAX_RADIX_BLOCKS_PER_ROW = 10


@functools.lru_cache(maxsize=1)
def _fused_dsa_metadata_enabled() -> bool:
    """Whether one fused launch fills the decode metadata (process-constant)."""
    return os.environ.get("TRTLLM_DISABLE_FUSED_DSA_METADATA", "0") == "0"


class DSAMetadata(SparseMetadata):
    """Base of DeepSeek-V4's sparse metadata: the indexer's buffers and state.

    It holds what the indexer and the top-k index conversion read: the indexer
    K-cache tables and slot mappings, the MQA-logits schedules, the per-token
    request map and the cached pool view. It has no ``prepare`` of its own:
    ``DeepseekV4Metadata`` prepares it as part of its step.
    """

    requires = (TrtllmCoreMetadata, MlaContextMetadata)

    # Ragged verification, mixed-precision KV, sparse KV offload and MTP index
    # sharing are not carried.
    is_ragged_verify = False
    ragged_verify_lens = None
    has_mixed_precision_high_rows = False
    sparse_offload_state = None
    sparse_offload_fetch_pending = None
    sparse_offload_write_pending = None
    indexer_skip_topk = False
    in_mtp_draft_loop = False
    mtp_num_accepted = None
    indexer_schedule_stale = False
    use_expanded_buffers_for_mtp = False

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        from ..backends.sparse.dsa.indexer import (
            _effective_compress_ratio_divisor,
            _select_indexer_compress_ratio,
        )
        from ..backends.sparse.dsa.params import DSAMetadataParams
        from ..backends.sparse.params import use_self_sampling_gvr

        super().__init__(registry, config)
        params = config.sparse_metadata_params
        if not isinstance(params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        if params.enable_kv_cache_offload:
            raise UnsupportedMetadataConfiguration("sparse KV cache offload")
        if params.mtp_index_share:
            raise UnsupportedMetadataConfiguration("MTP index sharing")
        manager = config.kv_cache_manager
        self._core = self.get_metadata(TrtllmCoreMetadata)
        self._mla_context = self.get_metadata(MlaContextMetadata)

        self.num_sms = get_num_sms()
        self.max_draft_tokens = config.max_draft_tokens
        self.indexer_max_chunk_size = params.indexer_max_chunk_size
        self.use_fp8_ds_mla = getattr(manager, "use_fp8_ds_mla", False)
        # Rows the top-k selects per query token.
        self.num_sparse_topk = params.max_sparse_topk
        self.sparse_mla_topk = self.num_sparse_topk
        self.indexer_head_dim = params.index_head_dim
        # Elements per four-byte scale word of the indexer K cache.
        self.indexer_quant_block_size = 128
        self.enable_indexer_skip = params.enable_indexer_skip
        self.use_cute_dsl_topk = params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        self.enable_gvr_topk = params.enable_heuristic_topk and get_sm_version() >= 100
        # Compress ratios of the layers this KV cache manager serves.
        self.compress_ratios: List[int] = getattr(params, "compress_ratios", [1])
        if hasattr(manager, "_compress_ratios"):
            self.compress_ratios = [manager._compress_ratios[layer] for layer in manager.pp_layers]
            # Only the ratio-4 layers of a compressed cache run the indexer.
            self.has_indexer = 4 in self.compress_ratios
        else:
            self.has_indexer = True
        # The indexer K cache stores one row per ``_indexer_compress_ratio``
        # tokens, so its slot mappings use the compressed block stride.
        self._indexer_compress_ratio = _select_indexer_compress_ratio(self.compress_ratios)
        self._tokens_per_block = manager.tokens_per_block
        if hasattr(manager, "compressed_block_sizes"):
            self._tokens_per_block //= _effective_compress_ratio_divisor(
                self._indexer_compress_ratio
            )
        self.use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self.enable_gvr_topk,
            use_self_sampling_topk=params.use_self_sampling_topk,
            index_topk=self.num_sparse_topk,
            compress_ratio=self._indexer_compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=get_sm_version(),
        )
        # The temporal GVR engine keeps the previous step's selection.
        self.needs_gvr_prior = (
            self.enable_gvr_topk
            and IS_CUTLASS_DSL_AVAILABLE
            and get_sm_version() in (100, 103)
            and not self.use_self_sampling_topk
        )

        # Per-step state.
        self.skip_indexer_for_ctx_reqs = False
        self.skip_indexer_for_gen_reqs = False
        self.indexer_prefill_chunks: Optional[List["IndexerPrefillChunkMetadata"]] = None
        # (local q_begin, local q_end, per-rank sizes) when the prefill chunks
        # cover only this TP rank's share of the context query tokens.
        self.indexer_prefill_q_split: Optional[Tuple[int, int, List[int]]] = None
        self.num_ctx_kv_tokens = 0
        self.num_ctx_mla_kv_tokens = 0
        self.max_gen_seq_len = 0
        self.gen_indexer_kv_lens_cuda_runtime: Optional[torch.Tensor] = None
        self.kv_lens_row_reorder: Optional[torch.Tensor] = None
        self.nvfp4_mla_context_fp8_scratch = None
        # Atom split of the CuTe DSL paged MQA-logits kernel, decided in
        # prepare() and read by the indexer forward.
        self.expand_for_dsl = False
        self.dsl_expand_factor = 1
        self.dsl_atom = 1
        self.dsl_kernel_atoms = 1
        self.dsl_kernel_next_n = 0
        self.dsl_atom_ctx_offsets: Optional[torch.Tensor] = None
        self.dsl_atom_ctx_offsets_key: Tuple[int, int] = (0, 0)
        # Full-KV gather mappings of the indexer; they alias the slot mappings
        # unless the context rows read cached indexer KV.
        self.slot_mapping_fp8_fullkv: Optional[torch.Tensor] = None
        self.slot_mapping_scale_fullkv: Optional[torch.Tensor] = None

        # Step-invariant values of the top-k index conversion, computed once
        # per step and reused by every layer.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_num_pool_tokens = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        # Layout of the full + shared indexer layer groups, and the current
        # forward's per-group remap output.
        self._group_remap_struct = None
        self._group_remap_struct_kv_id = 0
        self._group_remap_batched = {}
        self._fused_decode_metadata_logged = False

        self._create_buffers(manager)

    def _host_like(self, buffer: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(buffer, device="cpu", pin_memory=prefer_pinned())

    def _create_buffers(self, manager) -> None:
        params = self.sparse_metadata_params
        max_requests = self.config.max_num_requests
        max_sequences = self.config.max_num_sequences
        max_tokens = self.config.max_num_tokens
        max_blocks = manager.max_blocks_per_seq
        rows_per_sequence = 1 + self.max_draft_tokens

        # Exclusive prefix sums over the generation requests of the cached and
        # the total KV tokens.
        self.gen_cached_token_indptr = self.get_buffer(
            (max_requests + 1,), torch.int64, "gen_cached_token_indptr"
        )
        self.host_gen_cached_token_indptr = self._host_like(self.gen_cached_token_indptr)
        self.gen_kv_indptr = self.get_buffer((max_requests + 1,), torch.int64, "gen_kv_indptr")
        self.host_gen_kv_indptr = self._host_like(self.gen_kv_indptr)
        # Absolute position of each query token.
        self.token_positions_cuda = self.get_buffer(
            (max_tokens,), torch.int32, "token_positions_cuda"
        )
        # Request of each query token.
        self.req_idx_per_token = self.get_buffer((max_tokens,), torch.int32, "req_idx_per_token")
        self.host_req_idx_per_token = torch.empty_like(
            self.req_idx_per_token, device="cpu", pin_memory=prefer_pinned()
        )
        self.shared_topk_indices = None
        self.gvr_prior_indices = None
        self.nvfp4_mla_fp8_scratch = None
        self.nvfp4_mla_compact_indices = None
        if not self.has_indexer:
            return

        # Page table of the indexer K cache, [sequence, block].
        self.indexer_k_cache_block_offsets = self.get_buffer(
            (max_sequences, max_blocks), torch.int32, "indexer_k_cache_block_offsets"
        )
        self.host_indexer_k_cache_block_offsets = self._host_like(
            self.indexer_k_cache_block_offsets
        )
        # Local top-k reused by the layers that share an indexer.
        if params.has_shared_indexer_layers:
            self.shared_topk_indices = self.get_buffer(
                (max_tokens, self.num_sparse_topk), torch.int32, "shared_topk_indices"
            )
        # Flat byte positions of each new token's indexer K row and scale word.
        self.slot_mapping_fp8 = self.get_buffer((max_tokens,), torch.int64, "slot_mapping_fp8")
        self.host_slot_mapping_fp8 = self._host_like(self.slot_mapping_fp8)
        self.slot_mapping_scale = self.get_buffer((max_tokens,), torch.int64, "slot_mapping_scale")
        self.host_slot_mapping_scale = self._host_like(self.slot_mapping_scale)
        # Page table the top-k index conversion reads, [sequence, block].
        self.block_table = self.get_buffer((max_sequences, max_blocks), torch.int32, "block_table")
        # DeepGEMM MQA-logits schedules: one query token per request, the full
        # verify window, and the expanded rows of the DSL atom split.
        schedule_shape = (self.num_sms + 1, 2)
        self.scheduler_metadata_buffer = self.get_buffer(
            schedule_shape, torch.int32, "scheduler_metadata_buffer"
        )
        self.scheduler_metadata_buffer_full_next_n = self.get_buffer(
            schedule_shape, torch.int32, "scheduler_metadata_buffer_full_next_n"
        )
        self.scheduler_metadata_buffer_expanded = self.get_buffer(
            schedule_shape, torch.int32, "scheduler_metadata_buffer_expanded"
        )
        # Indexer KV length of each generation request, repeated per query token.
        self.kv_lens_cuda_2d = self.get_buffer(
            (max_sequences, rows_per_sequence), torch.int32, "kv_lens_cuda_2d"
        )
        self.kv_lens_expanded_cuda = self.get_buffer(
            (max_sequences * rows_per_sequence,), torch.int32, "kv_lens_expanded_cuda"
        )
        self.kv_lens_expanded_host = self._host_like(self.kv_lens_expanded_cuda)
        self.block_table_expanded = self.get_buffer(
            (max_sequences * rows_per_sequence, max_blocks), torch.int32, "block_table_expanded"
        )
        self.host_block_table_expanded = self._host_like(self.block_table_expanded)
        # Causal KV window of each context query token.
        self.cu_seqlen_ks = self.get_buffer((max_tokens,), torch.int32, "cu_seqlen_ks")
        self.cu_seqlen_ke = self.get_buffer((max_tokens,), torch.int32, "cu_seqlen_ke")
        if self.enable_indexer_skip:
            # Dense top-k of the requests short enough to skip the indexer.
            self.topk_indices_buffer = self.get_buffer(
                (max_tokens, self.num_sparse_topk), torch.int32, "topk_indices_buffer"
            )
            self.host_topk_indices_buffer = self._host_like(self.topk_indices_buffer)
        if self.needs_gvr_prior:
            self.gvr_prior_indices = self.get_buffer(
                (manager.num_local_layers, max_sequences, self.num_sparse_topk),
                torch.int32,
                "gvr_prior_indices",
            )
            self.gvr_prior_indices.zero_()
            self.kv_lens_row_reorder_buffer = self.get_buffer(
                (max_sequences,), torch.int32, "kv_lens_row_reorder_buffer"
            )
        radix_shape = (
            max_sequences * rows_per_sequence,
            _MAX_RADIX_BLOCKS_PER_ROW,
            self.num_sparse_topk,
        )
        self.radix_aux_indices = self.get_buffer(radix_shape, torch.int32, "radix_aux_indices")
        self.radix_aux_logits = self.get_buffer(radix_shape, torch.float32, "radix_aux_logits")
        # FP8 rows an NVFP4 latent cache is dequantized into for generation.
        if getattr(manager, "dtype", None) == tensorrt_llm.bindings.DataType.NVFP4:
            max_gen_tokens = max_sequences * rows_per_sequence
            self.nvfp4_mla_fp8_scratch = self.get_buffer(
                (max_gen_tokens, self.num_sparse_topk, manager.head_dim),
                torch.float8_e4m3fn,
                "nvfp4_mla_fp8_scratch",
            )
            self.nvfp4_mla_compact_indices = self.get_buffer(
                (max_gen_tokens, self.num_sparse_topk), torch.int32, "nvfp4_mla_compact_indices"
            )

    @property
    def use_cute_dsl_paged_mqa_logits(self) -> bool:
        """The indexer's decision: the config flag gated on SM100 and cutlass-dsl."""
        from ..backends.sparse.dsa.indexer import use_cute_dsl_paged_mqa_logits

        return use_cute_dsl_paged_mqa_logits(
            self.sparse_metadata_params.use_cute_dsl_paged_mqa_logits
        )

    @property
    def gen_token_stride(self) -> int:
        """Query tokens of each generation request."""
        return 1 + self.max_draft_tokens

    def gen_token_repeat_list(self) -> List[int]:
        return [self.gen_token_stride] * self._core.num_generations

    def ragged_row_kv_lens(self, num_tokens: int) -> None:
        return None

    def get_indexer_kv_lens(self, kv_lens: torch.Tensor) -> torch.Tensor:
        if self._indexer_compress_ratio <= 1:
            return kv_lens
        return kv_lens // self._indexer_compress_ratio

    def get_indexer_max_seq_len(self) -> int:
        max_seq_len = self._core.kv_cache_manager.max_seq_len
        if self._indexer_compress_ratio <= 1:
            return max_seq_len
        return max(1, max_seq_len // self._indexer_compress_ratio)

    def update_indexer_kv_lens_2d(self, gen_kv_lens: torch.Tensor) -> torch.Tensor:
        """Fill and return the active query-width view of ``kv_lens_cuda_2d``."""
        width = min(self.gen_token_stride, self.kv_lens_cuda_2d.shape[1])
        context_lens = self.kv_lens_cuda_2d[: self._core.num_generations, :width]
        context_lens.copy_(gen_kv_lens.unsqueeze(-1).expand(-1, width))
        return context_lens

    # Step preparation, called by the subclass in its own order.

    def _prepare_req_idx_per_token(self) -> None:
        core = self._core
        num_tokens = core.num_tokens
        self.host_req_idx_per_token[:num_tokens] = torch.repeat_interleave(
            torch.arange(core.num_seqs, dtype=torch.int32), core.seq_lens, dim=0
        )
        self.req_idx_per_token[:num_tokens].copy_(
            self.host_req_idx_per_token[:num_tokens], non_blocking=True
        )

    def _prepare_dsl_split(self, kv_lens: torch.Tensor) -> None:
        """Decide the DSL paged MQA-logits atom split and fill its row lengths."""
        from ..backends.sparse.dsa.indexer import _atom_ctx_offsets, _pick_dsl_split

        core = self._core
        manager = core.kv_cache_manager
        num_generations = core.num_generations
        self.expand_for_dsl = self.use_cute_dsl_paged_mqa_logits and self.max_draft_tokens >= 1
        if not (self.expand_for_dsl and num_generations > 0):
            self.dsl_expand_factor = 1
            self.dsl_atom = self.gen_token_stride
            self.dsl_kernel_atoms = 1
            return
        next_n = self.gen_token_stride
        kernel_atoms = (1, 2, 3) if manager.use_fp4 else (1, 2, 3, 4)
        gen_kv_lens = self.get_indexer_kv_lens(kv_lens[core.num_contexts : core.num_seqs])
        max_ctx = int(gen_kv_lens.max().item()) if gen_kv_lens.numel() else 0
        expand_factor, atom, kernel_split_atoms = _pick_dsl_split(
            next_n,
            batch_size=num_generations,
            max_ctx=max_ctx,
            num_sms=self.num_sms,
            kernel_atoms=kernel_atoms,
        )
        self.dsl_expand_factor = expand_factor
        self.dsl_atom = atom
        self.dsl_kernel_atoms = kernel_split_atoms
        self.dsl_kernel_next_n = next_n
        if kernel_split_atoms > 1:
            offsets = _atom_ctx_offsets(next_n, kernel_split_atoms)
            if self.dsl_atom_ctx_offsets_key != (next_n, kernel_split_atoms):
                # Built outside graph capture; the device derivation only subtracts.
                self.dsl_atom_ctx_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")
                self.dsl_atom_ctx_offsets_key = (next_n, kernel_split_atoms)
            num_rows = num_generations * kernel_split_atoms
            atom_lens = gen_kv_lens.unsqueeze(1) - torch.tensor(offsets, dtype=gen_kv_lens.dtype)
            self.kv_lens_expanded_host[:num_rows].copy_(atom_lens.reshape(-1))
            self.kv_lens_expanded_cuda[:num_rows].copy_(
                self.kv_lens_expanded_host[:num_rows], non_blocking=True
            )
        if expand_factor > 1:
            num_rows = num_generations * expand_factor
            self.kv_lens_expanded_host[:num_rows].copy_(
                gen_kv_lens.repeat_interleave(expand_factor)
            )
            self.kv_lens_expanded_cuda[:num_rows].copy_(
                self.kv_lens_expanded_host[:num_rows], non_blocking=True
            )
            self._refresh_expanded_block_table(expand_factor)

    def _refresh_expanded_block_table(self, repeat_factor: int) -> None:
        """Repeat each generation request's indexer page-table row per atom."""
        core = self._core
        num_rows = core.num_generations * repeat_factor
        gen_blocks = self.host_indexer_k_cache_block_offsets[core.num_contexts : core.num_seqs]
        self.host_block_table_expanded[:num_rows].copy_(
            gen_blocks.repeat_interleave(repeat_factor, dim=0)
        )
        self.block_table_expanded[:num_rows].copy_(
            self.host_block_table_expanded[:num_rows], non_blocking=True
        )
        self.block_table_expanded.clamp_(min=0)

    # Device derivation.

    def _derive_from_lengths(self) -> None:
        """Derive, on the device, what the indexer and the index conversion read
        from the core's current lengths."""
        from ..backends.sparse.dsa.indexer import _compute_slot_mappings

        core = self._core
        manager = core.kv_cache_manager
        num_tokens = core.num_tokens
        num_seqs = core.num_seqs
        num_contexts = core.num_contexts
        num_generations = core.num_generations
        self._invalidate_pool_view_cache()

        # One fused launch covers a generation-only step; it does not produce
        # the token positions, which the fp8_ds_mla layout reads.
        fused = (
            _fused_dsa_metadata_enabled()
            and num_tokens > 0
            and num_generations > 0
            and num_contexts == 0
            and not self.use_fp8_ds_mla
            and self.has_indexer
        )
        if fused:
            self._run_fused_decode_metadata()
        elif num_tokens > 0:
            seq_lens = core.seq_lens_cuda[:num_seqs]
            cu_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
            token_idx = torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int32)
            self.req_idx_per_token[:num_tokens] = torch.searchsorted(
                cu_seq_lens, token_idx, right=True
            ).to(self.req_idx_per_token.dtype)
            if self.has_indexer:
                # Cached lengths after the device-side correction.
                start_positions = core.kv_lens_cuda[:num_seqs] - seq_lens
                req_indices = self.req_idx_per_token[:num_tokens].to(torch.int64)
                seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(
                    torch.int64
                )
                token_offsets = (
                    torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int64)
                    - seq_starts[req_indices]
                )
                global_positions = start_positions[req_indices] + token_offsets
                if self.use_fp8_ds_mla:
                    self.token_positions_cuda[:num_tokens] = global_positions.to(torch.int32)
                index_head_dim = manager.index_head_dim
                use_fp4 = getattr(manager, "use_fp4", False)
                fp8_indices, scale_indices = _compute_slot_mappings(
                    global_positions,
                    self.indexer_k_cache_block_offsets,
                    req_indices,
                    index_head_dim,
                    self._tokens_per_block,
                    self.indexer_quant_block_size,
                    data_bytes_per_token=index_head_dim // 2 if use_fp4 else index_head_dim,
                )
                self.slot_mapping_fp8[:num_tokens] = fp8_indices
                self.slot_mapping_scale[:num_tokens] = scale_indices
            if num_generations > 0:
                gen_kv_lens = core.kv_lens_cuda[num_contexts:num_seqs]
                torch.cumsum(
                    gen_kv_lens,
                    dim=0,
                    dtype=torch.int64,
                    out=self.gen_kv_indptr[1 : num_generations + 1],
                )
                torch.cumsum(
                    gen_kv_lens - core.seq_lens_cuda[num_contexts:num_seqs],
                    dim=0,
                    dtype=torch.int64,
                    out=self.gen_cached_token_indptr[1 : num_generations + 1],
                )

        if num_generations > 0 and self.has_indexer:
            self._rebuild_indexer_schedule()
        if self.has_indexer:
            self._compute_kv_lens_row_reorder()
            self.prepare_dense_topk_indices(core.kv_lens_cuda, device=True)

    def _run_fused_decode_metadata(self) -> None:
        """Fill the request map, slot mappings and generation prefix sums in one launch."""
        from ..backends.sparse.dsa.kernels import fused_dsa_decode_metadata

        core = self._core
        manager = core.kv_cache_manager
        num_tokens = core.num_tokens
        num_seqs = core.num_seqs
        if not self._fused_decode_metadata_logged:
            logger.info(
                "Fused DSA decode-metadata kernel armed "
                f"(num_seqs={num_seqs}, num_tokens={num_tokens})."
            )
            self._fused_decode_metadata_logged = True
        index_head_dim = manager.index_head_dim
        use_fp4 = getattr(manager, "use_fp4", False)
        fused_dsa_decode_metadata(
            core.seq_lens_cuda[:num_seqs],
            core.kv_lens_cuda[:num_seqs],
            self.indexer_k_cache_block_offsets[:num_seqs],
            self.req_idx_per_token[:num_tokens],
            self.slot_mapping_fp8[:num_tokens],
            self.slot_mapping_scale[:num_tokens],
            self.gen_kv_indptr[: num_seqs + 1],
            self.gen_cached_token_indptr[: num_seqs + 1],
            num_tokens=num_tokens,
            max_query_len=1 + self.max_draft_tokens,
            tokens_per_block=self._tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=self.indexer_quant_block_size,
            data_bytes_per_token=index_head_dim // 2 if use_fp4 else index_head_dim,
        )

    def _rebuild_indexer_schedule(self) -> None:
        """Rebuild the MQA-logits schedules and indexer KV lengths of the generation rows."""
        from ..backends.sparse.dsa.indexer import _DG_SCHEDULE_BLOCK_KV

        core = self._core
        num_generations = core.num_generations
        gen_indexer_kv_lens = self.get_indexer_kv_lens(
            core.kv_lens_cuda[core.num_contexts : core.num_seqs]
        )
        self.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
        scheduler_context_lens = self.update_indexer_kv_lens_2d(gen_indexer_kv_lens)
        self.scheduler_metadata_buffer.copy_(
            get_paged_mqa_logits_metadata(
                gen_indexer_kv_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, self.num_sms
            ),
            non_blocking=True,
        )
        if self.max_draft_tokens > 0:
            self.scheduler_metadata_buffer_full_next_n.copy_(
                get_paged_mqa_logits_metadata(
                    scheduler_context_lens.contiguous(), _DG_SCHEDULE_BLOCK_KV, self.num_sms
                ),
                non_blocking=True,
            )
        if self.expand_for_dsl and self.dsl_kernel_atoms > 1:
            num_rows = num_generations * self.dsl_kernel_atoms
            atom_lens = gen_indexer_kv_lens.unsqueeze(1) - self.dsl_atom_ctx_offsets
            self.kv_lens_expanded_cuda[:num_rows].copy_(atom_lens.reshape(-1))
            self._schedule_expanded_rows(num_rows)
        if self.expand_for_dsl and self.dsl_expand_factor > 1:
            num_rows = num_generations * self.dsl_expand_factor
            self.kv_lens_expanded_cuda[:num_rows].copy_(
                gen_indexer_kv_lens.repeat_interleave(self.dsl_expand_factor)
            )
            self._schedule_expanded_rows(num_rows)

    def _schedule_expanded_rows(self, num_rows: int) -> None:
        from ..backends.sparse.dsa.indexer import _DG_SCHEDULE_BLOCK_KV

        self.scheduler_metadata_buffer_expanded.copy_(
            get_paged_mqa_logits_metadata(
                self.kv_lens_expanded_cuda[:num_rows].view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                self.num_sms,
            ),
            non_blocking=True,
        )

    def _compute_kv_lens_row_reorder(self) -> None:
        """Longest-job-first row order of the temporal GVR top-k."""
        core = self._core
        num_generations = core.num_generations
        if self.needs_gvr_prior and num_generations * self.gen_token_stride >= 2 * self.num_sms:
            gen_kv_lens = core.kv_lens_cuda[core.num_contexts : core.num_seqs]
            order = torch.argsort(gen_kv_lens, descending=True).to(torch.int32)
            self.kv_lens_row_reorder_buffer[:num_generations].copy_(order)
            self.kv_lens_row_reorder = self.kv_lens_row_reorder_buffer[:num_generations]
        else:
            self.kv_lens_row_reorder = None

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends, global_indices, side="right")
        position_ids = global_indices + per_seq_offsets[batch_indices]
        # Every position up to the query token's own, -1 past it.
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self, kv_lens: torch.Tensor, device: bool = False) -> None:
        """Dense top-k of the requests short enough to skip the indexer."""
        core = self._core
        seq_lens = core.seq_lens_cuda if device else core.seq_lens
        phases = (
            (
                self.skip_indexer_for_ctx_reqs,
                slice(0, core.num_contexts),
                slice(0, core.num_ctx_tokens),
            ),
            (
                self.skip_indexer_for_gen_reqs,
                slice(core.num_contexts, core.num_seqs),
                slice(core.num_ctx_tokens, core.num_tokens),
            ),
        )
        for skip, requests, tokens in phases:
            if not skip or requests.stop == requests.start:
                continue
            indices = self._get_dense_topk_indices(
                seq_lens[requests], kv_lens[requests], tokens.stop - tokens.start
            )
            if not device:
                self.host_topk_indices_buffer[tokens, :] = indices
                indices = self.host_topk_indices_buffer[tokens, :]
            self.topk_indices_buffer[tokens, :].copy_(indices, non_blocking=True)

    # Top-k index conversion.

    def _invalidate_pool_view_cache(self) -> None:
        self._pool_cache_valid = False
        self._group_remap_batched.clear()
        self.nvfp4_mla_context_fp8_scratch = None

    def _ensure_pool_view_cached(self) -> None:
        """Cache the pool view and the per-phase table slices for this step."""
        core = self._core
        manager = core.kv_cache_manager
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(manager):
            return
        pool = manager.get_unique_primary_pool()
        num_blocks, num_layers = pool.shape[:2]
        self._cached_tokens_per_block = manager.tokens_per_block
        self._cached_num_pool_tokens = num_blocks * num_layers * self._cached_tokens_per_block
        if manager.dtype == tensorrt_llm.bindings.DataType.NVFP4:
            # FP4 packs two values per byte; its gather op reads the raw pools.
            self._cached_pool_view = None
        else:
            self._cached_pool_view = pool.squeeze(2).view(-1, 1, manager.head_dim)
        self._cached_block_table_ctx = self.block_table[: core.num_contexts]
        self._cached_block_table_gen = self.block_table[core.num_contexts : core.num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[: core.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[core.num_ctx_tokens : core.num_tokens] - core.num_contexts
        )
        self._cached_kv_mgr_id = id(manager)
        self._pool_cache_valid = True

    def _ensure_group_remap_struct(self) -> dict:
        """Layout of the full + shared indexer layer groups (built once).

        A group is a full-indexer layer and the shared layers that follow it.
        It is active when it has at least two members with one page scale, so
        that one launch remaps the whole group.
        """
        manager = self._core.kv_cache_manager
        if self._group_remap_struct is not None and self._group_remap_struct_kv_id == id(manager):
            return self._group_remap_struct
        # The layer-offset tensors are built with a host-to-device copy,
        # which must not happen inside a CUDA graph capture.
        if torch.cuda.is_current_stream_capturing():
            return {}
        mask = getattr(manager, "indexer_k_cache_local_layer_mask", None)
        if mask is None:
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(manager)
            return self._group_remap_struct

        leader_of = [-1] * len(mask)
        slot_of = [0] * len(mask)
        members = {}
        leader = None
        for local_idx, is_full_indexer_layer in enumerate(mask):
            if is_full_indexer_layer or leader is None:
                leader = local_idx
                members[leader] = []
            members[leader].append(local_idx)
            leader_of[local_idx] = leader
            slot_of[local_idx] = len(members[leader]) - 1

        device = self.block_table.device
        group_active = {}
        group_scale = {}
        group_size = {}
        group_layer_ids = {}
        for leader, member_list in members.items():
            group_size[leader] = len(member_list)
            params = [manager.get_primary_pool_page_index_params(member) for member in member_list]
            scale = params[0][0]
            active = len(member_list) >= 2 and all(p[0] == scale for p in params)
            group_active[leader] = active
            group_scale[leader] = int(scale)
            if active:
                group_layer_ids[leader] = torch.tensor(
                    [int(p[1]) for p in params], dtype=torch.int32, device=device
                )
        self._group_remap_struct = {
            "leader_of": leader_of,
            "slot_of": slot_of,
            "group_active": group_active,
            "group_scale": group_scale,
            "group_size": group_size,
            "group_layer_ids": group_layer_ids,
        }
        self._group_remap_struct_kv_id = id(manager)
        return self._group_remap_struct

    # Warmup.

    def warmup_cute_dsl_radix_topk(self, next_n: int) -> None:
        """Pre-compile the CuTe DSL radix top-k variants engine warmup does not reach."""
        if not self.use_cute_dsl_topk or self.enable_gvr_topk:
            return
        if not self.sparse_mla_topk:
            return
        if self._indexer_compress_ratio > 1 and next_n > 1:
            return
        try:
            from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
                warmup_cute_dsl_radix_topk_decode,
            )
        except ImportError:
            return
        warmup_cute_dsl_radix_topk_decode(
            top_k=int(self.sparse_mla_topk),
            num_cols=int(self.get_indexer_max_seq_len()),
            next_n=next_n,
            dtype=_INDEXER_LOGITS_DTYPE,
            num_sms=self.num_sms,
        )

    def warmup_selfsampling_topk(
        self, next_n: int, batch_sizes: Optional[List[int]] = None
    ) -> None:
        """Pre-compile the self-sampling GVR engines for the common row counts."""
        if not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103):
            return
        if not self.enable_gvr_topk or not self.use_self_sampling_topk:
            return
        top_k = self.sparse_mla_topk
        if not top_k or int(top_k) not in (512, 1024, 2048):
            return
        compress_ratio = int(self._indexer_compress_ratio or 1)
        if compress_ratio not in (1, 4):
            return
        try:
            from tensorrt_llm._torch.kernels.blackwell.top_k import (
                gvr_topk_decode_self_sampling_host as self_sampling_host,
            )
        except ImportError:
            return
        next_n = int(next_n)
        # The small eager row counts plus every CUDA graph batch size.
        rows = set(range(next_n, 32 + 1, next_n)) or {next_n}
        for batch_size in batch_sizes or ():
            rows.add(int(batch_size) * next_n)
        max_seq_len = int(self.get_indexer_max_seq_len())
        options = (
            {"deterministic": True} if self.sparse_metadata_params.gvr_deterministic_topk else {}
        )
        try:
            self_sampling_host.warmup_prefill(int(top_k), max(max_seq_len, 32768), **options)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "self-sampling GVR prefill warmup ran out of memory; prefill "
                "engines will JIT-compile lazily on first touch instead."
            )
        if self.use_cute_dsl_paged_mqa_logits:
            # The DSL paged-MQA arena rounds rows up to 256 elements.
            row_stride = (max_seq_len + 255) // 256 * 256
        else:
            row_stride = max_seq_len
            if row_stride % 4:
                return
        try:
            self_sampling_host.warmup_varlen(
                int(top_k),
                max_seq_len * compress_ratio,
                compress_ratio=compress_ratio,
                next_n=next_n,
                num_rows_list=tuple(sorted(rows)),
                row_stride=row_stride,
                **options,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "self-sampling GVR warmup ran out of memory; varlen engines "
                "will JIT-compile lazily on first touch instead."
            )
