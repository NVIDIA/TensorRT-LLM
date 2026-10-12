# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime metadata for DeepSeek-V4 sparse attention."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import prefer_pinned

from ..dsa.metadata import DSAtrtllmAttentionMetadata
from .indexer import DeepseekV4Indexer
from .offload import SparseOffloadState
from .params import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    DeepSeekV4MetadataParams,
    is_compress_layer,
)
from .v41_layout import DeepseekV41Layout

if TYPE_CHECKING:
    from .cache_manager import DeepseekV4CacheManager


def _v41_layout_of(manager) -> Optional[DeepseekV41Layout]:
    """The DeepSeek-V4.1 layout a cache manager runs in, or None for DeepSeek-V4."""
    layout = getattr(manager, "v41_layout", None)
    return layout if isinstance(layout, DeepseekV41Layout) else None


def _num_sliding_table_columns(manager) -> int:
    return getattr(manager, "num_sliding_attention_columns", len(DEEPSEEK_V4_SLIDING_ATTENTION))


class DeepseekV4TrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    # The set of compress ratios for the layers
    compress_ratio_set: Set[int]
    # The set of (compress ratio, attention type) for the layers
    attention_type_set: Set[Tuple[int, DeepseekV4AttentionType]]
    # The number of total compressed tokens for each compress ratio
    num_total_compressed_tokens: Dict[int, int]
    # The max number of context compressed tokens for each compress ratio
    max_ctx_compressed_tokens: Dict[int, int]
    # DeepSeek-V4.1 layout of the cache manager, or None (DeepSeek-V4). A class
    # default, so an instance that skipped __post_init__ reads DeepSeek-V4.
    v41_layout: Optional["DeepseekV41Layout"] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _invalidate_mla_scheduler_buffers(self) -> None:
        """Invalidate request- and token-prefix buffers derived from sequence lengths."""
        super()._invalidate_mla_scheduler_buffers()
        self._fused_q_gen_cu_seqlens_valid = False

    @property
    def is_deepseek_v41(self) -> bool:
        return self.v41_layout is not None

    def __post_init__(self):
        super().__post_init__()
        self.num_total_compressed_tokens = {}
        self.max_ctx_compressed_tokens = {}
        self.nvfp4_compress_fp8_scratch: Optional[torch.Tensor] = None
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
        # DeepSeek-V4.1 layout mode follows the cache manager, which selects it
        # from the checkpoint config. The layer ratios then come from the
        # checkpoint (0 = window only, 1 / 2 = compressed); the sparse config's
        # normalized compress_ratios carry no layout.
        self.v41_layout = _v41_layout_of(self.kv_cache_manager)
        if self.v41_layout is not None:
            self._init_v41_ratio_sets(self.v41_layout)
        else:
            self._init_v4_ratio_sets()

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

        self._fused_q_gen_cu_seqlens_valid = False
        self.fused_q_gen_cu_seqlens = None
        self.gen_new_tokens_per_seq_cuda = None
        self.gen_new_tokens_per_seq: Optional[torch.Tensor] = None
        if self.enable_ragged_verification:
            # The fused Q normalization op consumes request-major token
            # prefixes, rather than the sparse-MLA token-major row view.
            self.fused_q_gen_cu_seqlens = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences + 1,),
                cache_name="fused_q_gen_cu_seqlens",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.fused_q_gen_cu_seqlens[:1].zero_()
            # Exact tokens appended by each generation request.
            self.gen_new_tokens_per_seq_cuda = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name="gen_new_tokens_per_seq_cuda",
                capture_graph=capture_graph,
            )

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

        if self.v41_layout is not None:
            # Every compressed V4.1 layer attends the window plus the index
            # top-k; the compressed local indices come from the indexer.
            self.max_compressed_indices = {
                ratio: (self.sparse_mla_topk if ratio > 0 else 0) for ratio in self._topk_len_ratios
            }
            self.compressed_local_indices_cuda = None
            self._init_v41_buffers(capture_graph)
        else:
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
            _num_sliding_table_columns(self.kv_cache_manager),
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
            for compress_ratio in self._compress_table_ratios
        }
        self._init_sparse_offload_state()

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
            for compress_ratio in self._topk_len_ratios
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

        # Draft-sized sparse buffers for one-model MTP separate draft KV cache.
        self._init_draft_sparse_buffers()

    def _init_sparse_offload_state(self) -> None:
        """Validate fixed offload configuration and allocate persistent metadata."""
        self.sparse_offload_state: SparseOffloadState | None = None
        if not self.sparse_metadata_params.enable_kv_cache_offload:
            return
        manager = self.kv_cache_manager
        # Per-batch beam/draft values start at defaults. Also check the manager's
        # configured limits before allocating any offload buffers.
        if (
            self.beam_width != 1
            or manager.max_beam_width != 1
            or self.max_draft_tokens
            or manager.max_draft_len
            or manager.max_total_draft_tokens
            or self.draft_kv_cache_manager is not None
        ):
            raise NotImplementedError(
                "Sparse offload requires single-beam, non-speculative execution"
            )
        # Compression policies are bound after manager construction, before
        # metadata initialization. Validate them once here, not on each batch.
        manager._validate_sparse_history_policy()
        # Descriptor setup also validates the fixed KV dtype and pool layout.
        layers = manager.get_sparse_offload_descriptors()
        if not layers:
            # This PP stage has no ratio-4 main KV to stage.
            return
        if self.sparse_mla_topk <= 0:
            raise ValueError("Sparse offload metadata requires positive top-k")
        batch = self.max_num_sequences
        max_blocks = self.kv_cache_manager.max_blocks_per_seq

        def allocate(name: str, shape: tuple[int, ...], value: int) -> torch.Tensor:
            tensor = self.get_empty(
                self.cuda_graph_buffers,
                shape,
                dtype=torch.int32,
                cache_name=f"dsv4_sparse_offload_{name}",
                capture_graph=self.is_cuda_graph,
            )
            tensor.fill_(value)
            return tensor

        history_blocks = allocate("history_blocks", (batch,), 0)
        self.sparse_offload_state = SparseOffloadState(
            layers=layers,
            base_page_tables={
                group: allocate(f"base_page_table_{group}", (batch, max_blocks), -1)
                for group in sorted({descriptor.group_id for descriptor in layers.values()})
            },
            history_blocks=history_blocks,
            history_blocks_host=torch.zeros(
                batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
            ),
            active_request_count=allocate("active_request_count", (1,), 0),
            selected_history_pages=allocate(
                "selected_history_pages", (batch, min(self.sparse_mla_topk, max_blocks)), -1
            ),
            fetched_page_table=allocate("fetched_page_table", (batch, max_blocks), -1),
            compress_read_table=allocate("compress_read_table", (batch, max_blocks), -1),
            read_table_valid=allocate("read_table_valid", (1,), 1),
        )

    def mla_prepare_fused_q_gen_cu_seqlens(self) -> Optional[torch.Tensor]:
        """Return stable request-major query prefixes for ragged fused Q-RoPE.

        The fused Q normalization op accepts token prefixes (not Q rows), so
        this deliberately uses request-major ``seq_lens_cuda`` rather than the
        one-row-per-token presentation used by sparse MLA. The cumsum writes
        into graph-owned storage once per iteration; device-window graph
        replay therefore updates values without changing the captured address.
        """
        if not self.is_ragged_verify or self.num_generations <= 0:
            return None

        num_contexts = self.num_contexts
        num_generations = self.num_generations
        cu_seqlens = getattr(self, "fused_q_gen_cu_seqlens", None)
        seq_lens_cuda = getattr(self, "seq_lens_cuda", None)
        if cu_seqlens is None or seq_lens_cuda is None or num_generations + 1 > cu_seqlens.shape[0]:
            return None

        if not self._fused_q_gen_cu_seqlens_valid:
            torch.cumsum(
                seq_lens_cuda[num_contexts : num_contexts + num_generations],
                0,
                dtype=torch.int32,
                out=cu_seqlens[1 : num_generations + 1],
            )
            self._fused_q_gen_cu_seqlens_valid = True
        return cu_seqlens[: num_generations + 1]

    def apply_device_ragged_layout(
        self,
        verify_lens: torch.Tensor,
        req_idx: torch.Tensor,
        kv_correction: torch.Tensor,
    ) -> None:
        """Install device-selected windows and invalidate derived Q prefixes."""
        super().apply_device_ragged_layout(verify_lens, req_idx, kv_correction)
        # This call precedes the first model layer. Its fixed-address cumsum is
        # therefore captured once after the layout update and replayed in order.
        self._invalidate_mla_scheduler_buffers()

    def _init_v4_ratio_sets(self) -> None:
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
        # sparse_mla_topk_lens keys: every layer ratio.
        self._topk_len_ratios = self.compress_ratio_set
        # Ratios with a COMPRESS page table.
        self._compress_table_ratios = [
            compress_ratio
            for compress_ratio in self._compress_ratios_sorted
            if is_compress_layer(compress_ratio)
        ]

    def _init_v41_ratio_sets(self, layout: "DeepseekV41Layout") -> None:
        """Ratio key spaces of the DeepSeek-V4.1 layout mode.

        Per-ratio compressed metadata (lengths, positions, masks, page tables)
        is keyed by the ratios that have a compressed stream, {1, 2};
        ``sparse_mla_topk_lens`` is keyed by every layer ratio, {0, 1, 2}.
        """
        self.compress_ratios = list(layout.compress_ratios)
        self.compress_ratio_set = set(layout.compressed_ratios)
        self._compress_ratios_sorted = sorted(self.compress_ratio_set)
        self._topk_len_ratios = list(layout.layer_ratios)
        self._compress_table_ratios = list(self._compress_ratios_sorted)
        attention_types = {(0, DeepseekV4AttentionType.SWA)}
        for layer in range(layout.num_layers):
            ratio = layout.compress_ratios[layer]
            for attn_type in layout.attention_types(layer):
                attention_types.add((ratio, attn_type))
        self.attention_type_set = attention_types

    def _init_v41_buffers(self, capture_graph: bool) -> None:
        """Buffers only the DeepSeek-V4.1 layout mode fills."""
        # Per-ratio INDEXER_COMPRESS (native MXFP4 index K) page tables, one
        # row per sequence of the batch, in pages of the owner's index-K pool
        # (tokens_per_block // ratio entries per page).
        self.indexer_compress_block_tables = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq),
                cache_name=f"indexer_compress_block_tables_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        self.host_indexer_compress_block_tables = {
            compress_ratio: torch.empty_like(
                self.indexer_compress_block_tables[compress_ratio],
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        # Per token: compressed entries visible at ratio r, (position + 1) // r.
        # Its generation rows, viewed [num_generations, tokens_per_generation],
        # are the paged indexer's context lengths.
        self.compressed_visible_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"compressed_visible_lens_cuda_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        max_positions = self.max_seq_len + max(self.max_draft_tokens, 0)
        # Static upper bound of compressed_visible_lens per ratio, e.g. the
        # paged indexer's max_context_len inside a captured step.
        self.max_compressed_kv_lens = {
            compress_ratio: max_positions // compress_ratio
            for compress_ratio in self._compress_ratios_sorted
        }

        # Address book: where this step's KV rows are written. Filled by
        # _prepare_v41_address_book() in prepare() and on_update_kv_lens();
        # -1 marks a row that must not be written. Rows at or past the step's
        # token / compressed-row count are not written.
        manager = self.kv_cache_manager
        # MXFP4 index-K entry: two E2M1 codes per byte, then one UE8M0 byte
        # per quant block (64 + 4 bytes at index_head_dim 128).
        self._v41_index_k_data_bytes = manager.index_head_dim // 2
        self._v41_index_k_scale_bytes = manager.index_head_dim // manager.quant_block_size
        # [num_local_layers, max_num_tokens]: row of token t's window KV in
        # get_buffers(L, SWA) viewed [pages * tokens_per_block, head_dim], at
        # local layer layer_offsets[L].
        self.v41_swa_slot_mapping = self.get_empty(
            self.cuda_graph_buffers,
            (manager.num_local_layers, self.max_num_tokens),
            cache_name="v41_swa_slot_mapping",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        # Per ratio r, [max_num_tokens], in the packed order of
        # compressed_position_ids_cuda[r]: row of the entry in
        # get_buffers(kv_source, COMPRESS) viewed [pages * block, head_dim],
        # block = tokens_per_block // r.
        self.v41_compress_slot_mapping = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"v41_compress_slot_mapping_{compress_ratio}",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        # Same rows: flat byte offsets of the entry's code bytes and scale word
        # in get_indexer_k_cache_buffers(index owner), uint8
        # [pages, block, 1, 68]; per page block * 64 code bytes, then block * 4
        # scale bytes.
        self.v41_index_k_slot_data = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"v41_index_k_slot_data_{compress_ratio}",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        self.v41_index_k_slot_scale = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"v41_index_k_slot_scale_{compress_ratio}",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            for compress_ratio in self._compress_ratios_sorted
        }
        # Context-only prefill index-K inputs; see _prepare_v41_ctx_index_k().
        self._reset_v41_ctx_index_k()
        self._init_v41_replay_buffers(capture_graph)

    def _init_v41_replay_buffers(self, capture_graph: bool) -> None:
        """Encoder SWA bounded replay fields, filled by _prepare_v41_replay().

        Without replay rows in the step they hold the plain values: no replay rows,
        every row a compute row, compute start = cached tokens, compute cu =
        cu_seq_lens, SWA floor 0, no Engram history rows.
        """
        num_seqs = self.max_num_sequences

        def buf(name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
            return self.get_empty(
                self.cuda_graph_buffers,
                shape,
                cache_name=name,
                dtype=dtype,
                capture_graph=capture_graph,
            )

        # Per request row: replay rows at the head of its rows this step.
        self.v41_num_replay_tokens_cuda = buf(
            "v41_num_replay_tokens_cuda", (num_seqs,), torch.int32
        )
        self.v41_num_replay_tokens_cpu = torch.zeros(
            (num_seqs,), dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.v41_num_replay_tokens_total = 0
        # Per request row: lowest position its window may attend (the replay start).
        self.v41_swa_floor_cuda = buf("v41_swa_floor_cuda", (num_seqs,), torch.int32)
        # Per context request row: the sparse attention's seq_lens, its kv length
        # minus its SWA floor (see _prepare_v41_swa_seq_lens()).
        self.v41_swa_seq_lens_cuda = buf("v41_swa_seq_lens_cuda", (num_seqs,), torch.int32)
        # Per token row: whether it is a replay row.
        self.v41_is_replay_token_cuda = buf(
            "v41_is_replay_token_cuda", (self.max_num_tokens,), torch.bool
        )
        # Rows that compute main KV / index K / compressor state, packed in row order.
        self.v41_compute_token_index_cuda = buf(
            "v41_compute_token_index_cuda", (self.max_num_tokens,), torch.int32
        )
        self.v41_num_compute_tokens = 0
        # Per request row: first computed position (cached + replay rows), and the
        # cumulative row counts of the packed compute rows.
        self.v41_compute_start_pos_cuda = buf(
            "v41_compute_start_pos_cuda", (num_seqs,), torch.int32
        )
        self.v41_compute_cu_seq_lens_cuda = buf(
            "v41_compute_cu_seq_lens_cuda", (num_seqs + 1,), torch.int32
        )
        # Engram history of the replay starts, as engram_hash input rows: ids,
        # context request row, per-context-row cu, and start position.
        history = max(1, self.v41_layout.engram_history_len)
        self.v41_replay_history_len = 0
        self.v41_replay_history_token_ids_cuda = buf(
            "v41_replay_history_token_ids_cuda", (num_seqs * history,), torch.int32
        )
        self.v41_replay_history_req_idx_cuda = buf(
            "v41_replay_history_req_idx_cuda", (num_seqs * history,), torch.int32
        )
        self.v41_replay_history_cu_seq_lens_cuda = buf(
            "v41_replay_history_cu_seq_lens_cuda", (num_seqs + 1,), torch.int32
        )
        self.v41_replay_history_start_pos_cuda = buf(
            "v41_replay_history_start_pos_cuda", (num_seqs,), torch.int32
        )

    def _prepare_v41_replay(self, num_requests: int) -> None:
        """Encoder SWA bounded replay rows of this step, from the cache manager's plans.

        A context request whose chunk starts at its replay start (a prefix-cache hit of
        ``reused`` tokens under replay; ``cached_token_lens`` is then ``reused - R``)
        has ``R = min(window, reused)`` replay rows at the head of its rows: cached
        tokens re-forwarded to rebuild the sliding windows. Filled here:

        - ``v41_num_replay_tokens_cuda/_cpu[b]`` = R (0 for other rows),
          ``v41_num_replay_tokens_total`` (host);
        - ``v41_is_replay_token_cuda[t]``;
        - ``v41_compute_token_index_cuda[:v41_num_compute_tokens]``: the other rows;
        - ``v41_compute_start_pos_cuda[b]`` = cached + R and
          ``v41_compute_cu_seq_lens_cuda``: the compute rows as a step starting at the
          reuse boundary;
        - ``v41_swa_floor_cuda[b]`` = the replay start (0 without replay): the
          window of a replay row starts no earlier (tech report 3.2.2);
        - ``v41_swa_seq_lens_cuda[b]`` = kv length - floor for each context
          request (``_prepare_v41_swa_seq_lens()``);
        - the Engram history rows of each replay start.

        The compressed stream (``past_kv_lens``, ``new_comp_kv_lens``, positions, the
        COMPRESS / index-K write slots) is then computed from the compute start, so no
        cached main-KV or index-K entry is rewritten.
        """
        manager = self.kv_cache_manager
        num_contexts = self.num_contexts
        num_tokens = self.num_tokens
        device = self.v41_num_replay_tokens_cuda.device
        replay = self.v41_num_replay_tokens_cpu
        replay[:num_requests] = 0
        cached = self.cached_token_lens_cpu[:num_requests]
        history_ids: List[int] = []
        history_req: List[int] = []
        history_lens = [0] * num_contexts
        history_start = [0] * num_contexts
        if num_contexts > 0 and getattr(manager, "v41_encoder_replay", False):
            plans = manager.get_v41_replay_plans(
                self.request_ids[:num_contexts], cached[:num_contexts].tolist()
            )
            for b, plan in enumerate(plans):
                if plan is None:
                    continue
                if int(self.seq_lens[b]) < plan.num_tokens:
                    raise RuntimeError(
                        f"request {self.request_ids[b]}: a chunk of {int(self.seq_lens[b])} "
                        f"rows cannot hold its {plan.num_tokens} replay rows"
                    )
                replay[b] = plan.num_tokens
                ids = plan.history_token_ids
                history_ids.extend(ids)
                history_req.extend([b] * len(ids))
                history_lens[b] = len(ids)
                history_start[b] = plan.start - len(ids)
        total = int(replay[:num_requests].sum())
        self.v41_num_replay_tokens_total = total
        self.v41_num_compute_tokens = num_tokens - total
        replay_cuda = self.v41_num_replay_tokens_cuda[:num_requests]
        replay_cuda.copy_(replay[:num_requests], non_blocking=True)
        cached_cuda = self.cached_token_lens_cuda[:num_requests]
        cu = self.cu_seq_lens_cuda[: num_requests + 1]
        if total == 0:
            self.v41_swa_floor_cuda[:num_requests].zero_()
            self.v41_compute_start_pos_cuda[:num_requests].copy_(cached_cuda)
            self.v41_compute_cu_seq_lens_cuda[: num_requests + 1].copy_(cu)
            self.v41_is_replay_token_cuda[:num_tokens].zero_()
            torch.arange(num_tokens, out=self.v41_compute_token_index_cuda[:num_tokens])
            self._prepare_v41_swa_seq_lens()
        else:
            cu = cu.to(torch.int32)
            self.v41_swa_floor_cuda[:num_requests] = torch.where(replay_cuda > 0, cached_cuda, 0)
            self._prepare_v41_swa_seq_lens()
            self.v41_compute_start_pos_cuda[:num_requests] = cached_cuda + replay_cuda
            compute_cu = torch.nn.functional.pad(
                torch.cumsum(self._seq_lens_cuda[:num_requests].to(torch.int32) - replay_cuda, 0),
                (1, 0),
            ).to(torch.int32)
            self.v41_compute_cu_seq_lens_cuda[: num_requests + 1] = compute_cu
            rows = torch.arange(num_tokens, dtype=torch.int32, device=device)
            req_idx = self.req_idx_per_token[:num_tokens].to(torch.int64)
            self.v41_is_replay_token_cuda[:num_tokens] = (rows - cu[req_idx]) < replay_cuda[req_idx]
            num_compute = num_tokens - total
            if num_compute > 0:
                crow = rows[:num_compute]
                creq = torch.searchsorted(compute_cu[1:], crow, right=True).to(torch.int64)
                self.v41_compute_token_index_cuda[:num_compute] = (
                    crow - compute_cu[creq] + cu[creq] + replay_cuda[creq]
                )
        num_history = len(history_ids)
        self.v41_replay_history_len = num_history
        if num_history > 0:
            history_cu = [0]
            for n in history_lens:
                history_cu.append(history_cu[-1] + n)
            self.v41_replay_history_token_ids_cuda[:num_history].copy_(
                torch.tensor(history_ids, dtype=torch.int32), non_blocking=True
            )
            self.v41_replay_history_req_idx_cuda[:num_history].copy_(
                torch.tensor(history_req, dtype=torch.int32), non_blocking=True
            )
            self.v41_replay_history_cu_seq_lens_cuda[: num_contexts + 1].copy_(
                torch.tensor(history_cu, dtype=torch.int32), non_blocking=True
            )
            self.v41_replay_history_start_pos_cuda[:num_contexts].copy_(
                torch.tensor(history_start, dtype=torch.int32), non_blocking=True
            )

    def _prepare_v41_swa_seq_lens(self) -> None:
        """``v41_swa_seq_lens_cuda[b]`` for each context request ``b``: its kv length
        minus its SWA floor, the ``seq_lens`` of the context sparse attention call.

        The kernel takes a query's window count from ``seq_lens``,
        ``min(window, seq_lens - q_len + j + 1)`` for query ``j`` of a request, and
        reads a -1 window column inside that count as an all-zero K/V row, not as
        a mask. A replay request's windows start at its floor, its replay start
        (the chunk's first position), so its kv length would count up to
        ``window - 1`` columns below the floor; ``kv_len - floor`` counts exactly
        its floored window, ``min(window, p - floor + 1)`` keys at position ``p``.
        With floor 0 (no replay) this is the kv length itself. Generation rows'
        windows are never floored and are not covered: they pass their kv
        lengths, which speculative decoding may move after prepare(). Refreshed
        by on_update_kv_lens() whenever the step has context requests.
        """
        n = self.num_contexts
        if n:
            torch.sub(
                self.kv_lens_cuda_runtime[:n],
                self.v41_swa_floor_cuda[:n],
                out=self.v41_swa_seq_lens_cuda[:n],
            )

    @property
    def engram_token_block_table(self) -> Optional[torch.Tensor]:
        """[max_num_sequences, max_blocks] page table of the Engram token history pool."""
        layout = self.v41_layout
        if layout is None or layout.engram_token_owner is None:
            return None
        local_layer = self.kv_cache_manager.layer_offsets.get(layout.engram_token_owner)
        if local_layer is None:
            return None
        return self.sliding_block_tables[local_layer, DeepseekV4AttentionType.ENGRAM_TOKEN.value]

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
            if (
                compress_ratio == DEEPSEEK_V4_SPARSE_RATIO
                and self.sparse_metadata_params.enable_kv_cache_offload
            ):
                if self.sparse_offload_state is None:
                    compress_block_table.fill_(-1)
                else:
                    self.kv_cache_manager.prepare_sparse_offload(
                        self.sparse_offload_state,
                        self.request_ids,
                        compress_block_table,
                        beam_width=self.beam_width,
                    )
                continue
            self.kv_cache_manager.copy_batch_compress_block_tables(
                compress_block_table,
                self.request_ids,
                compress_ratio=compress_ratio,
                beam_width=self.beam_width,
                num_contexts=self.num_contexts,
                num_seqs=self.num_seqs,
            )
        if self.v41_layout is not None:
            for compress_ratio, table in self.indexer_compress_block_tables.items():
                host_table = self.host_indexer_compress_block_tables[compress_ratio]
                self.kv_cache_manager.copy_batch_indexer_compress_block_tables(
                    host_table,
                    self.request_ids,
                    beam_width=self.beam_width,
                    num_contexts=self.num_contexts,
                    num_seqs=self.num_seqs,
                    compress_ratio=compress_ratio,
                )
                table[: self.num_seqs].copy_(host_table[: self.num_seqs], non_blocking=True)

    def prepare_for_deepseek_v4_indices(self, token_positions=None) -> torch.Tensor:
        """Prepare SWA/compressed local indices and sparse_mla_topk_lens.

        Returns the per-token absolute positions used, ``[num_tokens]`` int32.
        """
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

        if self.v41_layout is not None:
            num_tokens = token_positions.shape[0]
            swa_floor = self.v41_swa_floor_cuda[self.req_idx_per_token[:num_tokens].to(torch.int64)]
            self._prepare_deepseek_v41_indices_compiled(
                token_positions,
                swa_floor,
                window_size,
                self.sparse_mla_topk,
                self.swa_local_indices_cuda,
                self.sparse_mla_topk_lens,
                self.compressed_visible_lens_cuda,
                self._topk_len_ratios,
            )
            return token_positions

        if self.use_fp8_ds_mla:
            self.token_positions_cuda[: token_positions.shape[0]] = token_positions

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
        return token_positions

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

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _prepare_deepseek_v41_indices_compiled(
        token_positions: torch.Tensor,
        swa_floor: torch.Tensor,
        window_size: int,
        sparse_mla_topk: int,
        swa_local_indices_buf: torch.Tensor,
        sparse_mla_topk_lens_bufs: Dict[int, torch.Tensor],
        compressed_visible_lens_bufs: Dict[int, torch.Tensor],
        layer_ratios: list,
    ):
        """DeepSeek-V4.1 SWA local indices, visible compressed lengths and topk_lens.

        For a query at absolute position ``p`` (prefill, chunked prefill,
        decode and speculative-verify rows alike):
        - window: positions ``max(floor, p - window_size + 1) .. p`` in order,
          then -1, i.e. the reference's causal prefill window, and in decode the
          ring's slots as positions; ``floor`` is 0, or the replay start of an
          encoder-replay step's request (the window truncated to the replay
          segment, tech report 3.2.2);
        - ``compressed_visible_lens[r] = (p + 1) // r``, the reference's
          ``compress_lens``;
        - ``sparse_mla_topk_lens[r]``: ``window_size`` for window-only layers
          (the kernel takes the window count from ``seq_lens``), and
          ``window_size + min((p + 1) // r, index_topk)`` for compressed ones,
          the number of valid top-k entries the indexer packs at the front.
        """
        device = token_positions.device
        num_tokens = token_positions.shape[0]

        positions = token_positions.unsqueeze(1)
        swa_offsets = torch.arange(window_size, dtype=torch.int32, device=device)
        swa_start = torch.maximum(positions - window_size + 1, swa_floor.unsqueeze(1)).clamp(min=0)
        swa_indices = swa_start + swa_offsets
        swa_indices = torch.where(swa_indices > positions, -1, swa_indices).to(torch.int32)
        swa_local_indices_buf[:num_tokens] = swa_indices

        kv_lens = token_positions + 1
        for compress_ratio in layer_ratios:
            if compress_ratio == 0:
                total_count = torch.full_like(kv_lens, window_size)
            else:
                visible = kv_lens // compress_ratio
                compressed_visible_lens_bufs[compress_ratio][:num_tokens] = visible.to(torch.int32)
                total_count = window_size + visible.clamp(max=sparse_mla_topk)
            sparse_mla_topk_lens_bufs[compress_ratio][:num_tokens] = total_count.to(torch.int32)

    def _build_cache_buffer_data_pointers(
        self, manager: "DeepseekV4CacheManager", compress_ratios_by_layer: list[int]
    ) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
        """Build sparse cache pointers for a target or draft manager."""
        if _v41_layout_of(manager) is not None:
            return self._build_v41_cache_buffer_data_pointers(manager)
        sparse_mla_base_ptrs = {1: manager.swa_pool_ptr}
        for ratio, compress_pool_ptr in manager.compress_pool_ptrs.items():
            sparse_mla_base_ptrs[ratio] = compress_pool_ptr

        swa_buffer_ptrs = {layer_idx: manager.swa_pool_ptr for layer_idx in manager.pp_layers}
        compressed_buffer_ptrs = {
            layer_idx: manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS).data_ptr()
            for layer_idx in manager.pp_layers
            if is_compress_layer(compress_ratios_by_layer[layer_idx])
        }
        return sparse_mla_base_ptrs, swa_buffer_ptrs, compressed_buffer_ptrs

    @staticmethod
    def _build_v41_cache_buffer_data_pointers(
        manager: "DeepseekV4CacheManager",
    ) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
        """DeepSeek-V4.1 pool pointers keyed by layer ratio and by layer.

        ``sparse_mla_base_ptrs``: 0 -> the SWA pool, r -> the COMPRESS base of
        the first local KV source of ratio r. ``compressed_buffer_ptrs[L]`` is
        the COMPRESS base of ``kv_source(L)`` for every compressed layer, so a
        reader layer resolves to its owner's pool instead of a missing entry.
        """
        layout = manager.v41_layout
        sparse_mla_base_ptrs = {0: manager.swa_pool_ptr}
        sparse_mla_base_ptrs.update(manager.compress_pool_ptrs)
        swa_buffer_ptrs = {layer_idx: manager.swa_pool_ptr for layer_idx in manager.pp_layers}
        owner_ptrs: dict[int, int] = {}
        compressed_buffer_ptrs = {}
        for layer_idx in manager.pp_layers:
            owner = layout.kv_source(layer_idx)
            if owner is None:
                continue
            if owner not in owner_ptrs:
                owner_ptrs[owner] = manager.get_buffers(
                    owner, DeepseekV4AttentionType.COMPRESS
                ).data_ptr()
            compressed_buffer_ptrs[layer_idx] = owner_ptrs[owner]
        return sparse_mla_base_ptrs, swa_buffer_ptrs, compressed_buffer_ptrs

    def _init_cache_buffer_data_pointers(self):
        if self.v41_layout is not None:
            (
                self.sparse_mla_base_ptrs,
                self.swa_buffer_ptrs,
                self.compressed_buffer_ptrs,
            ) = self._build_v41_cache_buffer_data_pointers(self.kv_cache_manager)
            return
        # If MTP is enabled, enlarge the compress ratios by max_draft_tokens - 1.
        extend_compress_ratios = self.compress_ratios + [self.compress_ratios[-1]] * (
            self.max_draft_tokens - 1
        )
        (
            self.sparse_mla_base_ptrs,
            self.swa_buffer_ptrs,
            self.compressed_buffer_ptrs,
        ) = self._build_cache_buffer_data_pointers(self.kv_cache_manager, extend_compress_ratios)

    def _init_draft_sparse_buffers(self):
        """Initialize sparse buffers for a separate one-model MTP draft cache."""
        self.draft_sliding_block_tables = None
        self.draft_sparse_mla_base_ptrs = None
        self.draft_swa_buffer_ptrs = None

        draft_mgr = self.draft_kv_cache_manager
        from .cache_manager import DeepseekV4CacheManager

        if not isinstance(draft_mgr, DeepseekV4CacheManager):
            return
        if self.v41_layout is not None:
            raise NotImplementedError(
                "DeepSeek-V4.1 has no separate draft KV cache: its DSpark stages keep their "
                "windows in the drafter."
            )

        # Current DSv4 MTP layers are SWA-only. Fail fast if a future model
        # introduces a compressed or indexer MTP layer.
        draft_ratio = self.compress_ratios[-1]
        if draft_ratio != 1:
            raise NotImplementedError(
                "Separate DeepSeek-V4 draft KV cache supports only SWA-only "
                f"(compress_ratio 1) MTP draft layers; got ratio {draft_ratio}."
            )

        draft_block_table_shape = (
            draft_mgr.num_local_layers,
            _num_sliding_table_columns(draft_mgr),
            self.max_num_sequences,
            draft_mgr.max_blocks_per_seq,
        )
        self.draft_sliding_block_tables = self.get_empty(
            self.cuda_graph_buffers,
            draft_block_table_shape,
            cache_name="draft_sliding_block_tables",
            dtype=torch.int32,
            capture_graph=self.is_cuda_graph,
        )
        extend_compress_ratios = self.compress_ratios + [draft_ratio] * (self.max_draft_tokens - 1)
        (
            self.draft_sparse_mla_base_ptrs,
            self.draft_swa_buffer_ptrs,
            _,
        ) = self._build_cache_buffer_data_pointers(draft_mgr, extend_compress_ratios)

    _DRAFT_SPARSE_FIELDS = (
        "sliding_block_tables",
        "sparse_mla_base_ptrs",
        "swa_buffer_ptrs",
    )

    def prepare_for_draft_forward(self) -> dict | None:
        """Repoint sparse fields to the draft buffers for a draft forward."""
        if self.draft_sliding_block_tables is None:
            return None
        saved_state = {field: getattr(self, field) for field in self._DRAFT_SPARSE_FIELDS}
        for field in self._DRAFT_SPARSE_FIELDS:
            self.swap_for_draft(field, getattr(self, f"draft_{field}"))
        return saved_state

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        """No-op: restore_draft_swaps() restores the sparse fields, and DSA's hook must not run."""
        return None

    def validate_sparse_offload_batch(self) -> None:
        """Reject unsupported layouts on the host before any model writes.

        Only a fresh, unchunked context batch or one query per decode request
        is supported. This runs during prepare(), outside capture/replay;
        the per-layer path consumes the resulting phase and device metadata.
        Fixed configuration is validated during offload metadata initialization.
        """
        if not self.sparse_metadata_params.enable_kv_cache_offload:
            return
        state = self.sparse_offload_state
        if state is not None:
            state.prepared = False
            state.is_prefill = False
        if self.num_contexts and self.num_generations:
            raise NotImplementedError(
                "Sparse offload does not support mixed context/decode batches"
            )
        if self.seq_lens.device.type != "cpu" or self.seq_lens_kv.device.type != "cpu":
            raise ValueError("Sparse offload batch validation requires host sequence lengths")
        if not torch.equal(self.seq_lens, self.seq_lens_kv):
            raise NotImplementedError("Sparse offload requires matching query and new-KV lengths")
        if not self.num_contexts:
            if not torch.all(self.seq_lens == 1):
                raise NotImplementedError("Sparse offload decode requires one query per request")
            return
        cached = self.kv_cache_params.num_cached_tokens_per_seq
        if (
            self.prompt_lens is None
            or len(self.prompt_lens) < self.num_contexts
            or len(cached) < self.num_contexts
            or any(cached[: self.num_contexts])
            or self.seq_lens.tolist() != self.prompt_lens[: self.num_contexts]
            or any(
                self.kv_cache_manager.kv_cache_map[request_id].history_length != 0
                for request_id in self.request_ids
            )
        ):
            raise NotImplementedError(
                "Sparse offload prefill must be fresh, uncached, and unchunked"
            )
        if state is not None:
            state.is_prefill = True

    def prepare(self) -> None:
        self._validate_ragged_verification()
        assert self.kv_cache_manager is not None
        assert self.request_ids is not None
        self.validate_sparse_offload_batch()

        self.kv_cache_manager.compute_sliding_block_tables(
            self.request_ids,
            self.num_contexts,
        )

        # Prepare the draft manager's tables before the generic prepare copies
        # its block offsets, and populate the dedicated DSv4 draft sparse table.
        draft_mgr = self.draft_kv_cache_manager
        if draft_mgr is not None and hasattr(draft_mgr, "compute_sliding_block_tables"):
            draft_mgr.compute_sliding_block_tables(
                self.request_ids,
                self.num_contexts,
            )
            if self.draft_sliding_block_tables is not None:
                draft_mgr.copy_batch_sliding_block_tables(
                    self.draft_sliding_block_tables,
                    self.request_ids,
                    self.num_contexts,
                    self.num_seqs,
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

        kv_lens = self._prepare_device_window_kv_lens(kv_lens)

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

        if self.v41_layout is not None:
            self._prepare_v41_replay(num_requests)

        has_sparse_layers = DEEPSEEK_V4_SPARSE_RATIO in self.compress_ratio_set

        # For block offsets
        self.prepare_for_block_tables()

        # For indexer k cache (only needed when sparse layers exist)
        if has_sparse_layers:
            self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # For DeepSeek-V4 indices
        token_positions = self.prepare_for_deepseek_v4_indices()

        # Prepare metadata for indexer (only needed when sparse layers exist)
        if has_sparse_layers:
            DeepseekV4Indexer.prepare(metadata=self)

        # --- Per-ratio metadata ---
        # 1) CPU-side: compute scalar metadata (num_total_compressed_tokens, etc.)
        # 2) CUDA-side: fill *_cuda buffers via prepare_compressed_kv_metadata()
        num_gen_tokens_per_seq = self._sync_gen_tokens_per_seq(num_gen_tokens)
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        kv_lens_slice = kv_lens[:num_requests]
        cached_slice = cached_token_lens[:num_requests]
        if self.v41_layout is not None:
            # Replay rows compute no compressed entry: the stream starts after them.
            cached_slice = cached_slice + self.v41_num_replay_tokens_cpu[:num_requests]

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
        if self.v41_layout is not None:
            cached_tokens_cuda = self.v41_compute_start_pos_cuda[:num_requests]
        self.prepare_compressed_kv_metadata(kv_lens_cuda, cached_tokens_cuda, ctx_output_sizes)

        self._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            num_requests,
            self.num_total_compressed_tokens,
            self._compress_ratios_sorted,
        )

        if self.v41_layout is not None:
            self._prepare_v41_address_book(token_positions)
            self._prepare_v41_ctx_index_k(token_positions)

    def _sync_gen_tokens_per_seq(self, num_gen_tokens: int) -> int:
        """Synchronize the compressor's scalar and per-request token counts.

        The compressor uses ``num_gen_tokens_per_seq`` as a runtime launch
        bound. A ragged batch therefore uses the configured global maximum,
        while exact request lengths travel in a device vector. The vector is
        copied device-to-device from ``_seq_lens_cuda`` so CUDA-graph replay
        observes device-window rewrites instead of reloading the captured host
        shape split.
        """
        if self.is_ragged_verify and self.num_generations > 0:
            self.num_gen_tokens_per_seq = 1 + self.max_draft_tokens
            num_contexts = self.num_contexts
            num_generations = self.num_generations
            gen_new_tokens = self.gen_new_tokens_per_seq_cuda[:num_generations]
            gen_new_tokens.copy_(self._seq_lens_cuda[num_contexts : num_contexts + num_generations])
            self.gen_new_tokens_per_seq = gen_new_tokens
        else:
            self.num_gen_tokens_per_seq = (
                num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
            )
            self.gen_new_tokens_per_seq = None
        return self.num_gen_tokens_per_seq

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
        self._sync_gen_tokens_per_seq(num_gen_tokens)

        # Reuse prepare()'s host-computed ctx sizes unless the extend_ctx path
        # (num_chunked_ctx_requests > 0) may have mutated ctx-row kv_lens on
        # device; every other path only changes gen rows.
        ctx_output_sizes = (
            self._ctx_output_sizes if getattr(self, "num_chunked_ctx_requests", 0) == 0 else None
        )
        compressed_start = cached_tokens
        if self.v41_layout is not None:
            # Replay rows compute no compressed entry (see _prepare_v41_replay).
            compressed_start = cached_tokens + self.v41_num_replay_tokens_cuda[:batch_size]
        self.prepare_compressed_kv_metadata(kv_lens, compressed_start, ctx_output_sizes)

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

        if self.v41_layout is not None:
            self._prepare_v41_address_book(token_positions)
            self._prepare_v41_swa_seq_lens()

    @staticmethod
    def _v41_pages(
        block_table: torch.Tensor,
        req_idx: torch.Tensor,
        entry_idx: torch.Tensor,
        entries_per_page: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Page id and in-page slot of entry ``entry_idx`` of request row ``req_idx``.

        ``block_table`` is ``[..., rows, max_blocks]``; the page has its leading
        dims followed by ``entry_idx``'s. The page is -1 where the entry is
        negative, lies past the table, or sits on an unallocated page. The
        block index is clamped before the gather, so stale rows of a captured
        graph never read out of bounds. Graph-safe: no host synchronization.
        """
        max_blocks = block_table.shape[-1]
        block = torch.div(entry_idx, entries_per_page, rounding_mode="floor")
        in_table = (entry_idx >= 0) & (block < max_blocks)
        page = block_table[..., req_idx, block.clamp(0, max_blocks - 1)].to(torch.int64)
        page = torch.where(in_table & (page >= 0), page, -1)
        return page, torch.remainder(entry_idx, entries_per_page)

    def _v41_index_k_offsets(
        self,
        block_table: torch.Tensor,
        req_idx: torch.Tensor,
        entry_idx: torch.Tensor,
        entries_per_page: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flat byte offsets of an index-K entry's code bytes and scale word.

        A page of ``block`` entries holds ``block * data_bytes`` code bytes,
        then ``block * scale_bytes`` scale bytes (the DeepGEMM paged layout,
        as in the DSA indexer's ``_compute_slot_mappings``). -1 where the page
        is -1.
        """
        data_bytes = self._v41_index_k_data_bytes
        scale_bytes = self._v41_index_k_scale_bytes
        page, slot = self._v41_pages(block_table, req_idx, entry_idx, entries_per_page)
        page_base = page * (entries_per_page * (data_bytes + scale_bytes))
        valid = page >= 0
        data = torch.where(valid, page_base + slot * data_bytes, -1)
        scale = torch.where(
            valid, page_base + entries_per_page * data_bytes + slot * scale_bytes, -1
        )
        return data, scale

    def _prepare_v41_address_book(self, token_positions: torch.Tensor) -> None:
        """DeepSeek-V4.1 write addresses of this step's KV rows, on device.

        Runs at the end of prepare() and of on_update_kv_lens(), so it is
        recomputed inside CUDA-graph capture and replay: it only reads device
        tensors and host ints (the token count, ``num_seqs``,
        ``num_total_compressed_tokens``), and writes the persistent buffers in
        place.

        - ``v41_swa_slot_mapping[l, t]`` for local layer ``l`` and token ``t``
          at position ``p`` of request row ``b``:
          ``page * tokens_per_block + p % tokens_per_block`` with
          ``page = sliding_block_tables[l, SWA, b, p // tokens_per_block]``.
        - Per ratio r, row ``i`` of the packed compressed stream (the order of
          ``compressed_position_ids_cuda[r]``), request row ``b`` from
          ``cu_new_comp_kv_cuda[r]``, entry ``k = compressed_position_ids // r``,
          ``block = tokens_per_block // r``:
          ``v41_compress_slot_mapping[r][i] = page * block + k % block`` with
          ``page = compress_block_tables[r][b, k // block]``, and
          ``v41_index_k_slot_data[r][i]`` / ``v41_index_k_slot_scale[r][i]``
          from ``indexer_compress_block_tables[r]``.

        Rows whose ``compressed_mask_cuda[r]`` is False (reserved generation
        slots) or whose page is unallocated get -1. The masked rows' positions
        in ``compressed_position_ids_cuda[r]`` are set to 0: they are derived
        past the last request's entries and can reach ``max_seq_len``, and a
        RoPE table lookup reads every row.
        """
        manager = self.kv_cache_manager
        tokens_per_block = manager.tokens_per_block
        device = token_positions.device
        num_tokens = token_positions.shape[0]

        req_idx = self.req_idx_per_token[:num_tokens].to(torch.int64)
        positions = token_positions.to(torch.int64)
        swa_table = self.sliding_block_tables[:, DeepseekV4AttentionType.SWA.value]
        page, slot = self._v41_pages(swa_table, req_idx, positions, tokens_per_block)
        self.v41_swa_slot_mapping[:, :num_tokens] = torch.where(
            page >= 0, page * tokens_per_block + slot, -1
        )

        batch_size = self.num_seqs
        for compress_ratio in self._compress_ratios_sorted:
            total = self.num_total_compressed_tokens[compress_ratio]
            if total == 0:
                continue
            entries_per_page = tokens_per_block // compress_ratio
            cu_new_comp = self.cu_new_comp_kv_cuda[compress_ratio][: batch_size + 1]
            rows = torch.arange(total, dtype=cu_new_comp.dtype, device=device)
            row_req = torch.searchsorted(cu_new_comp, rows, right=True) - 1
            row_req = row_req.clamp(0, batch_size - 1).to(torch.int64)
            valid = self.compressed_mask_cuda[compress_ratio][:total]
            position_ids = self.compressed_position_ids_cuda[compress_ratio][:total]
            entry = torch.where(
                valid,
                torch.div(position_ids, compress_ratio, rounding_mode="floor").to(torch.int64),
                -1,
            )

            page, slot = self._v41_pages(
                self.compress_block_tables[compress_ratio], row_req, entry, entries_per_page
            )
            self.v41_compress_slot_mapping[compress_ratio][:total] = torch.where(
                page >= 0, page * entries_per_page + slot, -1
            )
            data, scale = self._v41_index_k_offsets(
                self.indexer_compress_block_tables[compress_ratio],
                row_req,
                entry,
                entries_per_page,
            )
            self.v41_index_k_slot_data[compress_ratio][:total] = data
            self.v41_index_k_slot_scale[compress_ratio][:total] = scale
            position_ids.masked_fill_(~valid, 0)

    def _reset_v41_ctx_index_k(self) -> None:
        ratios = self._compress_ratios_sorted
        self.v41_max_ctx_q_len = 0
        self.v41_ctx_index_k_len = {compress_ratio: 0 for compress_ratio in ratios}
        self.v41_ctx_index_k_gather_data = {compress_ratio: None for compress_ratio in ratios}
        self.v41_ctx_index_k_gather_scale = {compress_ratio: None for compress_ratio in ratios}
        self.v41_ctx_index_ks = {compress_ratio: None for compress_ratio in ratios}
        self.v41_ctx_index_ke = {compress_ratio: None for compress_ratio in ratios}

    def _prepare_v41_ctx_index_k(self, token_positions: torch.Tensor) -> None:
        """DeepSeek-V4.1 prefill indexer inputs of the context requests (eager).

        Per ratio r, with ``kv_len_b = cached_b + seq_len_b`` of context
        request ``b``:

        - ``v41_ctx_index_k_len[r]`` (host int): ``S_r = sum_b kv_len_b // r``.
        - ``v41_ctx_index_k_gather_data[r]`` / ``v41_ctx_index_k_gather_scale[r]``:
          int64 ``[S_r]`` byte offsets into the index-K pool of every
          compressed entry ``k = 0 .. kv_len_b // r - 1`` of every context
          request (cached entries included), request after request in batch
          order; -1 on an unallocated page.
        - ``v41_ctx_index_ks[r]`` / ``v41_ctx_index_ke[r]``: int32
          ``[num_ctx_tokens]``; context token ``t`` at position ``p`` of
          request ``b`` scores rows ``[ks, ke)`` of the gathered array: ``ks``
          = the first row of ``b``, ``ke = ks + (p + 1) // r``.
        - ``v41_max_ctx_q_len`` (host int): the longest context chunk.

        Sizes come from host lengths, so nothing reads the device. Every call
        replaces the previous step's values: without context requests the host
        ints are 0 and the tensors None. on_update_kv_lens() does not touch
        them.
        """
        self._reset_v41_ctx_index_k()
        num_contexts = self.num_contexts
        if num_contexts == 0:
            return
        device = token_positions.device
        num_ctx_tokens = self.num_ctx_tokens
        seq_lens = self.seq_lens[:num_contexts]
        kv_lens = self.cached_token_lens_cpu[:num_contexts] + seq_lens
        kv_lens_cuda = (
            self.cached_token_lens_cuda[:num_contexts] + self._seq_lens_cuda[:num_contexts]
        ).to(torch.int64)
        self.v41_max_ctx_q_len = int(seq_lens.max())

        ctx_req = self.req_idx_per_token[:num_ctx_tokens].to(torch.int64)
        visible_end = token_positions[:num_ctx_tokens].to(torch.int64) + 1
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        for compress_ratio in self._compress_ratios_sorted:
            total = int((kv_lens // compress_ratio).sum())
            cu_comp = torch.nn.functional.pad(
                torch.cumsum(kv_lens_cuda // compress_ratio, dim=0), (1, 0)
            )
            rows = torch.arange(total, dtype=torch.int64, device=device)
            row_req = torch.searchsorted(cu_comp, rows, right=True) - 1
            entry = rows - cu_comp[row_req]
            data, scale = self._v41_index_k_offsets(
                self.indexer_compress_block_tables[compress_ratio],
                row_req,
                entry,
                tokens_per_block // compress_ratio,
            )
            ks = cu_comp[ctx_req]
            self.v41_ctx_index_k_len[compress_ratio] = total
            self.v41_ctx_index_k_gather_data[compress_ratio] = data
            self.v41_ctx_index_k_gather_scale[compress_ratio] = scale
            self.v41_ctx_index_ks[compress_ratio] = ks.to(torch.int32)
            self.v41_ctx_index_ke[compress_ratio] = (
                ks + torch.div(visible_end, compress_ratio, rounding_mode="floor")
            ).to(torch.int32)

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
        """Compute cu_seq_lens and token_positions (eager)."""
        device = seq_lens.device

        # cu_seq_lens
        cu_seq_lens_buf[: batch_size + 1] = torch.nn.functional.pad(
            torch.cumsum(seq_lens.to(torch.int), dim=0), (1, 0)
        )

        # DSA metadata rebuilt this map for the current seq_lens.
        req_idx = req_idx_per_token_buf[:num_tokens].to(torch.int64)

        # token positions
        token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
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
