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

"""TRT-LLM attention backend for Auto-Deploy.

This module provides a TRT-LLM attention backend that wraps the optimized
`thop.attention` kernel for use in Auto-Deploy.

Architecture Overview:
---------------------
TRT-LLM's thop.attention expects:
- Combined KV cache or separate K/V with specific layout
- Many metadata tensors (sequence_length, context_lengths, request_types, etc.)
- Pool pointers for multi-pool KV cache management
- Per-layer wrapper state

AD provides:
- Separate K/V caches per layer: [num_pages, page_size, num_kv_heads, head_dim]
- Simpler metadata: batch_info, cu_seqlen, cu_num_pages, cache_loc, etc.

This implementation bridges the gap by:
1. Converting AD's metadata to TRT-LLM's format
2. Using AD's separate K/V caches with TRT-LLM's paged context FMHA
3. Managing per-layer state through global state dictionary
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType

from ..utils.cuda_graph import cuda_graph_state
from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    CacheConfig,
    Constant,
    MHACallable,
    PagedResourceHandler,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)


class TrtllmWorkspaceResourceHandler(ResourceHandler):
    """Resource handler for TRT-LLM workspace buffer.

    Allocates a global workspace buffer used by all attention layers.
    Returns the same buffer for all layers to avoid redundant allocations.

    The workspace is initially allocated at a modest size. The C++ attention
    kernel (thop.attention) automatically resizes the workspace via in-place
    ``resize_()`` if it determines more space is needed (see
    ``attentionOp.cpp`` – ``getWorkspaceSize`` / ``resize_`` logic). This
    avoids pre-allocating enormous buffers for large ``max_batch_size *
    max_seq_len`` configurations.
    """

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate workspace buffer - reuses existing if already allocated."""
        if _global_state.workspace is not None:
            return _global_state.workspace

        # Allocate a modest initial workspace; C++ thop.attention auto-resizes
        # via resize_() if more space is needed during warmup.
        buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=sequence_info.device)
        _global_state.init_workspace(buffer)
        return buffer


class TrtllmKVResourceHandler(PagedResourceHandler):
    """Resource handler for TRT-LLM unified KV cache.

    Extends PagedResourceHandler so the cache interface recognizes it as a paged resource
    and creates a proper KVCacheManager with correct parameters (num_layers, num_kv_heads, etc.).

    Uses kv_layout="HND" to request HND format from AD's cache manager.
    The cache is converted to HND format in interface.py via permute+contiguous.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_idx: int,
        trtllm_config: "TrtllmAttentionGlobalState",
        cache_config: CacheConfig,
    ) -> None:
        # Initialize parent class with token_shape = (2, num_kv_heads, head_dim)
        # The 2 is kv_factor for unified K+V cache
        # Use HND layout for thop.attention kernel compatibility
        super().__init__(2, num_kv_heads, head_dim, dtype=dtype, kv_layout="HND")

        # Store additional attributes for TRT-LLM attention
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_factor = 2  # Unified K+V cache
        self.layer_idx = layer_idx
        self._trtllm_config = trtllm_config
        self._cache_config = cache_config

    def __eq__(self, other: "TrtllmKVResourceHandler") -> bool:
        """Check compatibility for KVCacheManager resource grouping.

        Return True so KVCacheManager manages all layers' KV caches together.
        """
        if not isinstance(other, TrtllmKVResourceHandler):
            return False
        return (
            self.head_dim == other.head_dim
            and self.dtype == other.dtype
            and self.kv_factor == other.kv_factor
            and self.kv_layout == other.kv_layout
        )


@dataclass
class TrtllmLayerState:
    """Per-layer state for TRT-LLM attention wrapper."""

    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    tokens_per_block: int
    max_num_requests: int
    max_context_length: int
    num_layers: int = 0  # Total number of layers for block offset calculation

    # Pre-allocated tensors for metadata translation
    # Device tensors
    sequence_length: torch.Tensor = field(default=None)
    context_lengths: torch.Tensor = field(default=None)
    kv_cache_block_offsets: torch.Tensor = field(default=None)

    # Host tensors (pinned for async H2D)
    host_past_key_value_lengths: torch.Tensor = field(default=None)
    host_context_lengths: torch.Tensor = field(default=None)
    host_request_types: torch.Tensor = field(default=None)
    host_total_kv_lens: torch.Tensor = field(default=None)
    host_kv_cache_pool_pointers: torch.Tensor = field(default=None)
    host_kv_cache_pool_mapping: torch.Tensor = field(default=None)

    # Interleaved KV cache buffer for kernel (allocated lazily)
    interleaved_kv_cache: torch.Tensor = field(default=None)

    def __post_init__(self):
        """Initialize tensors - use shared tensors from global state where possible."""
        # Pool pointers and mapping are SHARED across layers.
        # For native HND mode with SELF cache type, all layers share the same pool
        # with different offsets. These will be linked in init_from_shared().
        pass

    def init_from_shared(self, global_state: "TrtllmAttentionGlobalState") -> None:
        """Initialize layer to use shared tensors from global state.

        For native HND mode with SELF cache type, ALL tensors are shared including
        pool pointers and mapping. The pool base pointer points to the entire pool,
        and pool_mapping contains per-layer offsets. Block indices from cache_loc
        are global indices relative to the pool base.
        """
        self.sequence_length = global_state._shared_sequence_length
        self.context_lengths = global_state._shared_context_lengths
        self.kv_cache_block_offsets = global_state._shared_kv_cache_block_offsets
        self.host_past_key_value_lengths = global_state._shared_host_past_key_value_lengths
        self.host_context_lengths = global_state._shared_host_context_lengths
        self.host_request_types = global_state._shared_host_request_types
        self.host_total_kv_lens = global_state._shared_host_total_kv_lens
        self.host_kv_cache_pool_pointers = global_state._shared_host_kv_cache_pool_pointers
        self.host_kv_cache_pool_mapping = global_state._shared_host_kv_cache_pool_mapping


class TrtllmAttentionGlobalState:
    """Global state manager for TRT-LLM attention layers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._layer_states: Dict[int, TrtllmLayerState] = {}
            cls._instance._workspace: Optional[torch.Tensor] = None
            cls._instance._max_blocks_per_seq: int = 0
            # Pre-allocated GPU buffers for vectorized block offset computation
            cls._instance._gpu_buffers_initialized: bool = False
            cls._instance._gpu_cu_pages: Optional[torch.Tensor] = None
            cls._instance._gpu_page_positions: Optional[torch.Tensor] = None
            cls._instance._gpu_seq_idx: Optional[torch.Tensor] = None
            cls._instance._gpu_page_idx: Optional[torch.Tensor] = None
            cls._instance._gpu_base_offset: Optional[torch.Tensor] = None
            # SHARED tensors across all layers (since KVCacheManager uses single pool)
            cls._instance._shared_tensors_initialized: bool = False
            cls._instance._shared_sequence_length: Optional[torch.Tensor] = None
            cls._instance._shared_context_lengths: Optional[torch.Tensor] = None
            cls._instance._shared_kv_cache_block_offsets: Optional[torch.Tensor] = None
            cls._instance._shared_host_past_key_value_lengths: Optional[torch.Tensor] = None
            cls._instance._shared_host_context_lengths: Optional[torch.Tensor] = None
            cls._instance._shared_host_request_types: Optional[torch.Tensor] = None
            cls._instance._shared_host_total_kv_lens: Optional[torch.Tensor] = None
            cls._instance._shared_host_kv_cache_pool_pointers: Optional[torch.Tensor] = None
            cls._instance._shared_host_kv_cache_pool_mapping: Optional[torch.Tensor] = None
            # Track if pool pointers have been initialized
            cls._instance._pool_pointers_initialized: bool = False
            # Correct num_layers from pool_mapping (may differ from _track_layer_config)
            cls._instance._pool_num_layers: int = 0
            # Track if host_prepare has been called (tensors are pre-filled)
            cls._instance._host_prepare_called: bool = False
            # Cache the current num_seq for tensor slicing
            cls._instance._current_num_seq: int = 0
            # Pre-allocated CPU buffers for host_prepare (avoid tensor allocation)
            cls._instance._cpu_buffers_initialized: bool = False
            cls._instance._cpu_input_seq_lens: Optional[torch.Tensor] = None
            cls._instance._cpu_seq_len_with_cache: Optional[torch.Tensor] = None
            cls._instance._cpu_past_kv_lens: Optional[torch.Tensor] = None
            cls._instance._cpu_cu_num_pages: Optional[torch.Tensor] = None
            cls._instance._cpu_pages_per_seq: Optional[torch.Tensor] = None
        return cls._instance

    def _init_shared_tensors(
        self, max_num_requests: int, max_context_length: int, tokens_per_block: int
    ) -> bool:
        """Initialize shared tensors used by all layers.

        Supports reallocation if a larger batch size or block count is needed after initial allocation.

        Returns:
            True if (re)allocation was performed, False if already sufficient.
        """
        device = "cuda"
        max_blocks_per_seq = (max_context_length + tokens_per_block - 1) // tokens_per_block

        # Check if reallocation is needed (current size < requested size)
        needs_realloc = (
            self._shared_tensors_initialized
            and self._shared_sequence_length is not None
            and (
                self._shared_sequence_length.shape[0] < max_num_requests
                or self._shared_kv_cache_block_offsets.shape[3] < max_blocks_per_seq
            )
        )

        if self._shared_tensors_initialized and not needs_realloc:
            return False

        # Shared device tensors
        self._shared_sequence_length = torch.zeros(
            max_num_requests, dtype=torch.int32, device=device
        )
        self._shared_context_lengths = torch.zeros(
            max_num_requests, dtype=torch.int32, device=device
        )
        self._shared_kv_cache_block_offsets = torch.zeros(
            1, max_num_requests, 2, max_blocks_per_seq, dtype=torch.int32, device=device
        )

        # Shared host tensors (pinned memory)
        self._shared_host_past_key_value_lengths = torch.zeros(
            max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._shared_host_context_lengths = torch.zeros(
            max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._shared_host_request_types = torch.zeros(
            max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._shared_host_total_kv_lens = torch.zeros(
            2, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._shared_host_kv_cache_pool_pointers = torch.zeros(
            1, 2, dtype=torch.int64, device="cpu", pin_memory=True
        )
        # Pool mapping: [num_layers, 2] - layer to pool mapping
        self._shared_host_kv_cache_pool_mapping = torch.zeros(
            64, 2, dtype=torch.int32, device="cpu", pin_memory=True
        )

        self._shared_tensors_initialized = True
        return True

    def _init_cpu_buffers(self, max_seqs: int) -> None:
        """Initialize pre-allocated CPU buffers to avoid tensor allocation in hot path."""
        needs_realloc = (
            self._cpu_buffers_initialized
            and self._cpu_input_seq_lens is not None
            and self._cpu_input_seq_lens.shape[0] < max_seqs
        )

        if self._cpu_buffers_initialized and not needs_realloc:
            return

        self._cpu_input_seq_lens = torch.zeros(
            max_seqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._cpu_seq_len_with_cache = torch.zeros(
            max_seqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._cpu_past_kv_lens = torch.zeros(
            max_seqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._cpu_cu_num_pages = torch.zeros(
            max_seqs + 1, dtype=torch.long, device="cpu", pin_memory=True
        )
        self._cpu_pages_per_seq = torch.zeros(
            max_seqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._cpu_buffers_initialized = True

    def _init_gpu_buffers(self, max_pages: int, max_seqs: int) -> None:
        """Initialize pre-allocated GPU buffers for vectorized operations."""
        needs_realloc = (
            self._gpu_buffers_initialized
            and self._gpu_cu_pages is not None
            and (
                self._gpu_cu_pages.shape[0] < max_seqs + 1
                or self._gpu_page_positions.shape[0] < max_pages
            )
        )

        if self._gpu_buffers_initialized and not needs_realloc:
            return

        self._gpu_cu_pages = torch.zeros(max_seqs + 1, dtype=torch.long, device="cuda")
        self._gpu_page_positions = torch.arange(max_pages, dtype=torch.long, device="cuda")
        self._gpu_seq_idx = torch.zeros(max_pages, dtype=torch.long, device="cuda")
        self._gpu_page_idx = torch.zeros(max_pages, dtype=torch.long, device="cuda")
        self._gpu_base_offset = torch.zeros(max_pages, dtype=torch.int32, device="cuda")
        self._gpu_buffers_initialized = True

    def _update_pool_pointers(
        self,
        ad_pool_pointers: Optional[torch.Tensor],
        ad_pool_mapping: Optional[torch.Tensor],
        num_layers: int,
    ) -> None:
        """Update pool pointers from AD's KVCacheManager.

        Called once to initialize pool pointers, or when the pool base address changes
        (after KVCacheManager resize). During steady-state serving, this is a no-op
        after initial setup.
        """
        # Check if AD pool pointers are valid
        use_ad_pool = (
            ad_pool_pointers is not None
            and ad_pool_mapping is not None
            and ad_pool_pointers.numel() > 0
            and ad_pool_pointers[0, 0].item() != 0
        )

        if not use_ad_pool:
            if not self._pool_pointers_initialized:
                ad_logger.warning("AD pool pointers not available. Pool initialization deferred.")
            return

        new_ptr = ad_pool_pointers[0, 0].item()
        current_ptr = self._shared_host_kv_cache_pool_pointers[0, 0].item()

        # Only update if pointer changed (e.g., after KVCacheManager resize)
        if current_ptr == new_ptr and self._pool_pointers_initialized:
            return

        self._shared_host_kv_cache_pool_pointers[0, 0] = new_ptr
        self._shared_host_kv_cache_pool_pointers[0, 1] = 0

        for layer_i in range(min(num_layers, ad_pool_mapping.shape[0])):
            self._shared_host_kv_cache_pool_mapping[layer_i, 0] = ad_pool_mapping[layer_i, 0].item()
            self._shared_host_kv_cache_pool_mapping[layer_i, 1] = ad_pool_mapping[layer_i, 1].item()

        # Use the actual pool_mapping size for num_layers (accurate for hybrid models)
        self._pool_num_layers = ad_pool_mapping.shape[0]
        self._pool_pointers_initialized = True

    def get_or_create_layer_state(
        self,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tokens_per_block: int,
        max_num_requests: int,
        max_context_length: int,
        num_layers: int = 0,
    ) -> TrtllmLayerState:
        """Get or create per-layer state."""
        did_realloc = self._init_shared_tensors(
            max_num_requests, max_context_length, tokens_per_block
        )

        if did_realloc:
            for state in self._layer_states.values():
                state.init_from_shared(self)

        if layer_idx not in self._layer_states:
            state = TrtllmLayerState(
                layer_idx=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_num_requests=max_num_requests,
                max_context_length=max_context_length,
                num_layers=num_layers,
            )
            state.init_from_shared(self)
            self._layer_states[layer_idx] = state
        return self._layer_states[layer_idx]

    def init_workspace(self, buffer: torch.Tensor) -> None:
        """Initialize the global workspace buffer."""
        self._workspace = buffer

    @property
    def workspace(self) -> Optional[torch.Tensor]:
        return self._workspace

    def set_max_blocks_per_seq(self, max_blocks: int) -> None:
        """Set max blocks per sequence (needed for block offset tensor sizing)."""
        self._max_blocks_per_seq = max(self._max_blocks_per_seq, max_blocks)

    @property
    def max_blocks_per_seq(self) -> int:
        return self._max_blocks_per_seq

    def create_host_prepare_function(self) -> Callable[..., None]:
        """Create host_prepare function for CUDA graph support.

        This function runs OUTSIDE the graph, before each forward (including replay).
        It updates tensors with current batch values using vectorized GPU operations.
        """
        layer_states = self._layer_states
        global_state = self

        def _host_prepare_trtllm_metadata(
            batch_info_host: torch.Tensor,
            cu_seqlen_host: torch.Tensor,
            cu_num_pages_host: torch.Tensor,
            cache_loc: torch.Tensor,
            seq_len_with_cache_host: torch.Tensor,
        ) -> None:
            """Fill device/host tensors before graph replay using vectorized ops.

            OPTIMIZED: Uses pre-allocated buffers with out= parameters to avoid
            tensor allocation overhead in the hot path.
            """
            if not layer_states:
                return

            # Fast integer extraction
            num_prefill = int(batch_info_host[0])
            num_prefill_tokens = int(batch_info_host[1])
            num_decode = int(batch_info_host[2])
            num_seq = num_prefill + num_decode

            # Set shared batch info in cuda_graph_state for other ops (e.g. Mamba)
            cuda_graph_state.set_batch_info(num_prefill, num_prefill_tokens, num_decode)

            if num_seq == 0:
                return

            first_state = next(iter(layer_states.values()))
            if first_state.sequence_length is None:
                return

            # Prefer pool_num_layers (most accurate for hybrid models)
            if global_state._pool_num_layers > 0:
                num_layers = global_state._pool_num_layers
            elif first_state.num_layers > 0:
                num_layers = first_state.num_layers
            else:
                num_layers = 32

            # Initialize buffers once (lazy initialization)
            if not global_state._cpu_buffers_initialized:
                global_state._init_cpu_buffers(first_state.max_num_requests)

            if not global_state._gpu_buffers_initialized:
                max_pages = first_state.max_num_requests * (
                    (first_state.max_context_length + first_state.tokens_per_block - 1)
                    // first_state.tokens_per_block
                )
                global_state._init_gpu_buffers(max_pages, first_state.max_num_requests)

            # Initialize pool pointers once (static for cache lifetime).
            # Pool pointers only change after KVCacheManager resize, which is handled
            # in _prepare_trtllm_metadata (initial setup path).
            if not global_state._pool_pointers_initialized:
                if _trtllm_config._sequence_info is not None:
                    ad_pool_pointers = _trtllm_config._sequence_info.kv_cache_pool_pointers
                    ad_pool_mapping = _trtllm_config._sequence_info.kv_cache_pool_mapping
                    global_state._update_pool_pointers(
                        ad_pool_pointers, ad_pool_mapping, num_layers
                    )

            # Use pre-allocated CPU buffers with out= to avoid tensor allocation
            input_seq_lens = global_state._cpu_input_seq_lens[:num_seq]
            torch.sub(
                cu_seqlen_host[1 : num_seq + 1],
                cu_seqlen_host[:num_seq],
                out=input_seq_lens,
            )

            seq_len_with_cache = global_state._cpu_seq_len_with_cache[:num_seq]
            seq_len_with_cache.copy_(seq_len_with_cache_host[:num_seq])

            past_kv_lens = global_state._cpu_past_kv_lens[:num_seq]
            torch.sub(seq_len_with_cache, input_seq_lens, out=past_kv_lens)

            # Compute totals
            context_total_kv = seq_len_with_cache[:num_prefill].sum() if num_prefill > 0 else 0
            gen_total_kv = seq_len_with_cache[num_prefill:num_seq].sum() if num_decode > 0 else 0

            # Block offsets info
            cu_num_pages = global_state._cpu_cu_num_pages[: num_seq + 1]
            cu_num_pages.copy_(cu_num_pages_host[: num_seq + 1])

            pages_per_seq = global_state._cpu_pages_per_seq[:num_seq]
            torch.sub(cu_num_pages[1 : num_seq + 1], cu_num_pages[:num_seq], out=pages_per_seq)

            max_blocks = int(pages_per_seq.max())
            global_state.set_max_blocks_per_seq(max_blocks)
            total_pages = int(cu_num_pages[num_seq])

            # H2D copies using pinned memory (non-blocking) - only fill active slots
            global_state._shared_sequence_length[:num_seq].copy_(
                seq_len_with_cache, non_blocking=True
            )
            global_state._shared_context_lengths[:num_seq].copy_(input_seq_lens, non_blocking=True)

            # Fill shared host tensors (CPU to CPU, fast) - only active slots
            global_state._shared_host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
            global_state._shared_host_context_lengths[:num_seq].copy_(input_seq_lens)
            if num_prefill > 0:
                global_state._shared_host_request_types[:num_prefill].fill_(0)
            if num_decode > 0:
                global_state._shared_host_request_types[num_prefill:num_seq].fill_(1)
            global_state._shared_host_total_kv_lens[0] = context_total_kv
            global_state._shared_host_total_kv_lens[1] = gen_total_kv

            # Compute block offsets on GPU (vectorized) - only zero active slice
            if total_pages > 0:
                global_state._shared_kv_cache_block_offsets[:, :num_seq, :, :].zero_()

                # Copy cu_pages to GPU (small tensor)
                global_state._gpu_cu_pages[: num_seq + 1].copy_(cu_num_pages)
                cu_num_pages_gpu = global_state._gpu_cu_pages[: num_seq + 1]

                page_positions = global_state._gpu_page_positions[:total_pages]

                # searchsorted on GPU
                torch.searchsorted(
                    cu_num_pages_gpu[1:],
                    page_positions,
                    right=True,
                    out=global_state._gpu_seq_idx[:total_pages],
                )
                seq_indices = global_state._gpu_seq_idx[:total_pages]

                # page_in_seq on GPU
                torch.sub(
                    page_positions,
                    cu_num_pages_gpu[seq_indices],
                    out=global_state._gpu_page_idx[:total_pages],
                )
                page_in_seq = global_state._gpu_page_idx[:total_pages]

                # Block offsets: multiplier = num_layers * kv_factor for interleaved K/V
                kv_factor = 2
                pool_layers = (
                    global_state._pool_num_layers
                    if global_state._pool_num_layers > 0
                    else num_layers
                )
                multiplier = pool_layers * kv_factor
                torch.mul(
                    cache_loc[:total_pages],
                    multiplier,
                    out=global_state._gpu_base_offset[:total_pages],
                )
                base_offsets = global_state._gpu_base_offset[:total_pages]

                # Fill block offsets using advanced indexing
                global_state._shared_kv_cache_block_offsets[0, seq_indices, 0, page_in_seq] = (
                    base_offsets  # K
                )
                global_state._shared_kv_cache_block_offsets[0, seq_indices, 1, page_in_seq] = (
                    base_offsets + 1  # V
                )

            # Mark that host_prepare has run
            global_state._host_prepare_called = True
            global_state._current_num_seq = num_seq

        return _host_prepare_trtllm_metadata

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        self._layer_states.clear()
        self._workspace = None
        self._max_blocks_per_seq = 0
        self._host_prepare_called = False
        self._current_num_seq = 0
        self._cpu_buffers_initialized = False
        self._gpu_buffers_initialized = False
        self._pool_pointers_initialized = False
        self._pool_num_layers = 0


# Global state singleton
_global_state = TrtllmAttentionGlobalState()


def _prepare_trtllm_metadata(
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    state: TrtllmLayerState,
    kv_cache: torch.Tensor,
    ad_pool_pointers: Optional[torch.Tensor] = None,
    ad_pool_mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Prepare TRT-LLM metadata from AD metadata.

    OPTIMIZED: All heavy computation is done in host_prepare_fn (outside graph).
    This function returns pre-computed tensors during normal operation.

    Flow:
    - First call (before host_prepare): Do full computation to initialize
    - CUDA graph capture: Set host tensors to MAX
    - Normal replay: Just return pre-computed tensors (FAST PATH)
    """
    # Check if in CUDA graph capture mode
    is_capturing = torch.cuda.is_current_stream_capturing()

    # Extract batch info early - needed for capacity check
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # Check if shared tensors need reallocation (batch size > allocated size)
    if state.sequence_length is not None and state.sequence_length.shape[0] < num_seq:
        did_realloc = _global_state._init_shared_tensors(
            num_seq, state.max_context_length, state.tokens_per_block
        )
        if did_realloc:
            state.init_from_shared(_global_state)

    # FAST PATH: If host_prepare has run and not capturing, just return pre-computed tensors
    # This is the key optimization - each layer does almost no work during replay
    if _global_state._host_prepare_called and not is_capturing:
        num_seq = _global_state._current_num_seq
        max_blocks_per_seq = state.kv_cache_block_offsets.shape[3]
        return (
            state.sequence_length[:num_seq],
            state.host_past_key_value_lengths[:num_seq],
            state.host_total_kv_lens,
            state.context_lengths[:num_seq],
            state.host_context_lengths[:num_seq],
            state.host_request_types[:num_seq],
            state.kv_cache_block_offsets[:, :num_seq, :, :max_blocks_per_seq],
            state.host_kv_cache_pool_pointers,
            state.host_kv_cache_pool_mapping,
        )

    # CUDA GRAPH CAPTURE: Set host tensors to MAX (kernel requirement)
    if is_capturing:
        max_seq = state.max_context_length
        state.host_past_key_value_lengths[:num_seq].fill_(max_seq)
        state.host_context_lengths[:num_seq].fill_(max_seq)
        state.host_request_types[:num_seq].fill_(1)
        state.host_total_kv_lens[0] = 0
        state.host_total_kv_lens[1] = max_seq * num_seq

        # Update pool pointers if not yet initialized
        if ad_pool_pointers is not None and ad_pool_mapping is not None:
            num_layers = state.num_layers if state.num_layers > 0 else 32
            _global_state._update_pool_pointers(ad_pool_pointers, ad_pool_mapping, num_layers)

        max_blocks_per_seq = state.kv_cache_block_offsets.shape[3]
        return (
            state.sequence_length[:num_seq],
            state.host_past_key_value_lengths[:num_seq],
            state.host_total_kv_lens,
            state.context_lengths[:num_seq],
            state.host_context_lengths[:num_seq],
            state.host_request_types[:num_seq],
            state.kv_cache_block_offsets[:, :num_seq, :, :max_blocks_per_seq],
            state.host_kv_cache_pool_pointers,
            state.host_kv_cache_pool_mapping,
        )

    # INITIAL SETUP: First call before host_prepare_fn is registered

    # Validate kv_cache shape
    if len(kv_cache.shape) != 5 or kv_cache.shape[1] != 2:
        raise RuntimeError(
            f"Expected kv_cache shape [pages, 2, heads, tokens, dim], got {kv_cache.shape}"
        )

    # Get num_layers - prefer from pool_mapping (accurate) over state.num_layers (may be stale)
    if ad_pool_mapping is not None and ad_pool_mapping.numel() > 0:
        num_layers = ad_pool_mapping.shape[0]
        # For hybrid models (e.g., Mamba+Attention), force-call set_model_config
        # with correct num_layers from pool_mapping.
        if _trtllm_config._num_layers == 0:
            kv_num_heads = kv_cache.shape[2]
            kv_head_dim = kv_cache.shape[4]
            kv_heads_list = TrtllmAttention._num_kv_heads_per_layer[:num_layers]
            if len(kv_heads_list) < num_layers:
                kv_heads_list = [kv_num_heads] * num_layers
            _trtllm_config.set_model_config(
                num_layers=num_layers,
                num_kv_heads_per_layer=kv_heads_list,
                head_dim=kv_head_dim,
                dtype=kv_cache.dtype,
            )
    elif state.num_layers > 0:
        num_layers = state.num_layers
    else:
        num_layers = 32

    # Compute input sequence lengths from cumulative sums
    input_seq_lens = (cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]).int()
    seq_len_with_cache = seq_len_with_cache_host[:num_seq].int()
    past_kv_lens = seq_len_with_cache - input_seq_lens.cpu()

    # Fill host tensors
    state.host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
    state.host_context_lengths[:num_seq].copy_(input_seq_lens.cpu())
    state.host_request_types[:num_prefill].fill_(0)
    state.host_request_types[num_prefill:num_seq].fill_(1)
    context_total_kv = seq_len_with_cache[:num_prefill].sum().item() if num_prefill > 0 else 0
    gen_total_kv = seq_len_with_cache[num_prefill:num_seq].sum().item() if num_decode > 0 else 0
    state.host_total_kv_lens[0] = context_total_kv
    state.host_total_kv_lens[1] = gen_total_kv

    # Fill device tensors
    state.sequence_length[:num_seq].copy_(seq_len_with_cache.cuda())
    state.context_lengths[:num_seq].copy_(input_seq_lens.cuda())

    # Block offsets: convert flat cache_loc to per-sequence block indices
    pages_per_seq = (cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]).int()
    max_blocks = pages_per_seq.max().item() if num_seq > 0 else 1
    _global_state.set_max_blocks_per_seq(max_blocks)

    # Ensure kv_cache_block_offsets is large enough for the required pages
    block_offsets_shape = state.kv_cache_block_offsets.shape
    if max_blocks > block_offsets_shape[3]:
        ad_logger.info(
            f"Reallocating block offsets: need {max_blocks} blocks but have {block_offsets_shape[3]}."
        )
        did_realloc = _global_state._init_shared_tensors(
            num_seq, max_blocks * state.tokens_per_block, state.tokens_per_block
        )
        if did_realloc:
            state.init_from_shared(_global_state)
            ad_logger.info(
                f"Block offsets reallocated: new shape {state.kv_cache_block_offsets.shape}"
            )

    # Set up KV cache pool pointers - use AD's pool pointers
    use_ad_pool = (
        ad_pool_pointers is not None
        and ad_pool_mapping is not None
        and ad_pool_pointers.numel() > 0
        and ad_pool_pointers[0, 0].item() != 0
    )

    if not use_ad_pool:
        raise RuntimeError(
            f"AD pool not available. ad_pool_pointers={ad_pool_pointers}, "
            f"ad_pool_mapping={ad_pool_mapping}"
        )

    # Use AD's pool pointers directly
    state.host_kv_cache_pool_pointers[0, 0] = ad_pool_pointers[0, 0].item()
    state.host_kv_cache_pool_pointers[0, 1] = 0

    # Use AD's pool mapping directly
    for layer_i in range(min(num_layers, ad_pool_mapping.shape[0])):
        state.host_kv_cache_pool_mapping[layer_i, 0] = ad_pool_mapping[layer_i, 0].item()
        state.host_kv_cache_pool_mapping[layer_i, 1] = ad_pool_mapping[layer_i, 1].item()

    _global_state._pool_pointers_initialized = True
    _global_state._pool_num_layers = ad_pool_mapping.shape[0]

    # Fill block offsets
    kv_factor = 2
    multiplier = num_layers * kv_factor
    state.kv_cache_block_offsets.zero_()
    offset = 0
    for i in range(num_seq):
        n_pages = pages_per_seq[i].item()
        if n_pages > 0:
            base_offsets = cache_loc[offset : offset + n_pages] * multiplier
            state.kv_cache_block_offsets[0, i, 0, :n_pages] = base_offsets  # K
            state.kv_cache_block_offsets[0, i, 1, :n_pages] = base_offsets + 1  # V
            offset += n_pages

    max_blocks_per_seq = state.kv_cache_block_offsets.shape[3]

    return (
        state.sequence_length[:num_seq],
        state.host_past_key_value_lengths[:num_seq],
        state.host_total_kv_lens,
        state.context_lengths[:num_seq],
        state.host_context_lengths[:num_seq],
        state.host_request_types[:num_seq],
        state.kv_cache_block_offsets[:, :num_seq, :, :max_blocks_per_seq],
        state.host_kv_cache_pool_pointers,
        state.host_kv_cache_pool_mapping,
    )


@torch.library.custom_op("auto_deploy::trtllm_attention_mha_with_cache", mutates_args=("kv_cache",))
def trtllm_mha_with_cache(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (AD format)
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # UNIFIED KV CACHE (kv_factor=2: K and V combined)
    kv_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    layer_idx: int,
    scale: Optional[float],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
) -> torch.Tensor:
    """TRT-LLM attention with unified KV cache for Auto-Deploy."""
    # Validate inputs
    if not q.is_cuda:
        raise RuntimeError(f"Q must be on CUDA, got {q.device}")
    if not kv_cache.is_cuda:
        raise RuntimeError(f"kv_cache must be on CUDA, got {kv_cache.device}")

    assert kv_cache.dim() == 5, f"kv_cache must be 5D, got {kv_cache.dim()}D"
    assert kv_cache.shape[1] == 2, (
        f"kv_cache must be in HND format [B, 2, H, T, D] with shape[1]=2, "
        f"got shape {kv_cache.shape}. Ensure TrtllmKVResourceHandler is used."
    )

    # Lazy initialization of model config (done once on first attention call)
    if _trtllm_config._num_layers == 0:
        TrtllmAttention._track_layer_config(num_kv_heads, head_dim, kv_cache.dtype)
        expected_num_layers = len(TrtllmAttention._num_kv_heads_per_layer)
        if layer_idx == expected_num_layers - 1 or expected_num_layers >= 32:
            _trtllm_config.set_model_config(
                num_layers=expected_num_layers,
                num_kv_heads_per_layer=TrtllmAttention._num_kv_heads_per_layer,
                head_dim=head_dim,
                dtype=kv_cache.dtype,
            )

    # Get batch dimensions
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    # Reshape inputs to TRT-LLM expected format: [num_tokens, hidden_dim]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim]
    q_flat = q.reshape(b * s, num_heads * head_dim)[:num_tokens]
    k_flat = k.reshape(b * s, num_kv_heads * head_dim)[:num_tokens]
    v_flat = v.reshape(b * s, num_kv_heads * head_dim)[:num_tokens]

    # TRT-LLM requires FUSED QKV: concatenate Q, K, V along hidden dimension
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # Prepare output tensor
    output = torch.empty(num_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device)

    # Get num_layers from config for block offset calculation
    if _global_state._pool_num_layers > 0:
        num_layers = _global_state._pool_num_layers
    elif _trtllm_config._num_layers > 0:
        num_layers = _trtllm_config._num_layers
    else:
        num_layers = 32
    state = _global_state.get_or_create_layer_state(
        layer_idx=layer_idx,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_num_requests=max_num_requests,
        max_context_length=max_context_length,
        num_layers=num_layers,
    )

    # Get AD's pool pointers if available
    ad_pool_pointers = None
    ad_pool_mapping = None
    if _trtllm_config._sequence_info is not None:
        ad_pool_pointers = _trtllm_config._sequence_info.kv_cache_pool_pointers
        ad_pool_mapping = _trtllm_config._sequence_info.kv_cache_pool_mapping

    # Prepare TRT-LLM metadata
    (
        sequence_length,
        host_past_key_value_lengths,
        host_total_kv_lens,
        context_lengths,
        host_context_lengths,
        host_request_types,
        kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
    ) = _prepare_trtllm_metadata(
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        state,
        kv_cache,
        ad_pool_pointers=ad_pool_pointers,
        ad_pool_mapping=ad_pool_mapping,
    )

    attention_window_size = max_context_length

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    sm_version = get_sm_version()
    if sm_version >= 89:  # Ada/Hopper
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    try:
        thop.attention(
            qkv_fused,  # q (actually fused QKV)
            None,  # k (None when using fused QKV)
            None,  # v (None when using fused QKV)
            output,  # output
            None,  # output_sf (NVFP4)
            workspace_buffer,  # workspace
            sequence_length,  # sequence_length
            host_past_key_value_lengths,  # host_past_key_value_lengths
            host_total_kv_lens,  # host_total_kv_lens
            context_lengths,  # context_lengths
            host_context_lengths,  # host_context_lengths
            host_request_types,  # host_request_types
            kv_cache_block_offsets,  # kv_cache_block_offsets
            host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            None,  # cache_indirection (beam search)
            _trtllm_config._kv_scale_orig_quant,  # kv_scale_orig_quant
            _trtllm_config._kv_scale_quant_orig,  # kv_scale_quant_orig
            None,  # out_scale
            None,  # rotary_inv_freq
            None,  # rotary_cos_sin
            None,  # latent_cache (MLA)
            None,  # q_pe (MLA)
            None,  # block_ids_per_seq
            None,  # attention_sinks
            True,  # is_fused_qkv
            True,  # update_kv_cache
            1,  # predicted_tokens_per_seq
            layer_idx,  # layer_idx
            num_heads,  # num_heads
            num_kv_heads,  # num_kv_heads
            head_dim,  # head_size
            kv_cache.shape[3],  # tokens_per_block
            max_num_requests,  # max_num_requests
            max_context_length,  # max_context_length
            attention_window_size,  # attention_window_size
            0,  # sink_token_length
            1,  # beam_width
            int(AttentionMaskType.causal),  # mask_type
            _trtllm_config._quant_mode,  # quant_mode
            1.0,  # q_scaling
            0,  # position_embedding_type
            0,  # rotary_embedding_dim
            10000.0,  # rotary_embedding_base
            0,  # rotary_embedding_scale_type
            rotary_embedding_scales,  # rotary_embedding_scales
            rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
            True,  # use_paged_context_fmha
            0,  # attention_input_type
            False,  # is_mla_enable
            max_num_requests,  # chunked_prefill_buffer_batch_size (constant for CUDA graph)
            None,  # q_lora_rank (MLA)
            None,  # kv_lora_rank (MLA)
            None,  # qk_nope_head_dim (MLA)
            None,  # qk_rope_head_dim (MLA)
            None,  # v_head_dim (MLA)
            None,  # mrope_rotary_cos_sin
            None,  # mrope_position_deltas
            mla_tensor_params,  # mla_tensor_params
            None,  # attention_chunk_size
            None,  # softmax_stats_tensor
            spec_decoding_bool_params,  # spec_decoding_bool_params
            spec_decoding_tensor_params,  # spec_decoding_tensor_params
            None,  # sparse_kv_indices
            None,  # sparse_kv_offsets
            None,  # sparse_attn_indices
            None,  # sparse_attn_offsets
            1,  # sparse_attn_indices_block_size
            0,  # sparse_mla_topk
            None,  # skip_softmax_threshold_scale_factor_prefill
            None,  # skip_softmax_threshold_scale_factor_decode
            None,  # skip_softmax_stat
            None,  # cu_q_seqlens
            None,  # cu_kv_seqlens
            None,  # fmha_scheduler_counter
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
        )
    except Exception as e:
        raise RuntimeError(
            f"TRT-LLM attention failed at layer {layer_idx}: {e}\n"
            f"  num_seq={num_seq}, num_tokens={num_tokens}\n"
            f"  q_flat.shape={q_flat.shape}, k_flat.shape={k_flat.shape}\n"
            f"  kv_cache.shape={kv_cache.shape}"
        ) from e

    # Reshape output back to AD format [b, s, num_heads * head_dim]
    # Pad back to original batch*seq size if needed
    if output.shape[0] < b * s:
        output_padded = torch.zeros(b * s, num_heads * head_dim, dtype=q.dtype, device=q.device)
        output_padded[: output.shape[0]] = output
        output = output_padded

    return output.view(*q_shape_og)


@trtllm_mha_with_cache.register_fake
def trtllm_mha_with_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    layer_idx: int,
    scale: Optional[float],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tokens_per_block: int,
    max_num_requests: int,
    max_context_length: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(q.contiguous())


class TrtllmAttentionConfig:
    """Configuration holder for TRT-LLM attention backend."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        """Reset configuration to defaults."""
        self.page_size: int = 64
        self.max_batch_size: int = 64
        self.max_seq_len: int = 65536
        self.max_num_tokens: int = 2048
        self.is_configured: bool = False

        # Model configuration
        self._num_layers: int = 0
        self._num_kv_heads_per_layer: List[int] = []
        self._head_dim: int = 0
        self._dtype: torch.dtype = torch.float16

        # FP8 KV cache support
        self._kv_scale_orig_quant: Optional[torch.Tensor] = None
        self._kv_scale_quant_orig: Optional[torch.Tensor] = None
        self._quant_mode: int = 0

        # Store SequenceInfo reference for AD pool pointer access
        self._sequence_info: Optional[SequenceInfo] = None

    def configure(self, si: SequenceInfo):
        """Configure from SequenceInfo."""
        self.page_size = si.tokens_per_block
        self.max_batch_size = si.max_batch_size
        self.max_seq_len = si.max_seq_len
        self.max_num_tokens = si.max_num_tokens
        self.is_configured = True
        self._sequence_info = si

    def set_model_config(
        self,
        num_layers: int,
        num_kv_heads_per_layer: List[int],
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Set model configuration."""
        self._num_layers = num_layers
        self._num_kv_heads_per_layer = num_kv_heads_per_layer
        self._head_dim = head_dim
        self._dtype = dtype

        # Initialize FP8 KV cache support if dtype is FP8
        if dtype == torch.float8_e4m3fn:
            self._quant_mode = 128
            self._kv_scale_orig_quant = torch.ones(1, dtype=torch.float32, device="cuda")
            self._kv_scale_quant_orig = torch.ones(1, dtype=torch.float32, device="cuda")
        else:
            self._quant_mode = 0
            self._kv_scale_orig_quant = None
            self._kv_scale_quant_orig = None


# Global config singleton
_trtllm_config = TrtllmAttentionConfig()


@AttentionRegistry.register("trtllm")
class TrtllmAttention(AttentionDescriptor):
    """TRT-LLM attention backend for Auto-Deploy."""

    # Class-level counter for layer indices
    _layer_counter: int = 0

    # Track num_kv_heads per layer for model config
    _num_kv_heads_per_layer: List[int] = []
    _head_dim: int = 0
    _dtype: torch.dtype = torch.float16

    @classmethod
    def _get_next_layer_idx(cls) -> int:
        """Get the next layer index and increment counter."""
        idx = cls._layer_counter
        cls._layer_counter += 1
        return idx

    @classmethod
    def _reset_layer_counter(cls) -> None:
        """Reset layer counter (for testing or new model builds)."""
        cls._layer_counter = 0
        cls._num_kv_heads_per_layer = []
        cls._head_dim = 0
        cls._dtype = torch.float16
        _global_state.reset()
        _trtllm_config.reset()

    @classmethod
    def _track_layer_config(cls, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        """Track layer configuration."""
        cls._num_kv_heads_per_layer.append(num_kv_heads)
        cls._head_dim = head_dim
        cls._dtype = dtype

    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Get the prepare_metadata op info."""
        return (None, 0, [])

    @classmethod
    def configure_from_sequence_info(cls, si: SequenceInfo) -> None:
        """Configure TRT-LLM backend from SequenceInfo."""
        _trtllm_config.configure(si)

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> ResourceHandlerDict:
        """Provide resource handlers for cache initialization."""
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype)
        layer_idx = cls._layer_counter

        return {
            "kv_cache": TrtllmKVResourceHandler(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                layer_idx=layer_idx,
                trtllm_config=_trtllm_config,
                cache_config=cache_config,
            ),
            "workspace_buffer": TrtllmWorkspaceResourceHandler(),
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Get function that performs host-side prep for attention."""
        return _global_state.create_host_prepare_function()

    @classmethod
    def get_host_prepare_metadata_args(cls) -> List[str]:
        """Get argument names for host_prepare function."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages_host",
            "cache_loc",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Provide constant arguments to be passed to the attention op."""
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        q_fake: FakeTensor = source_attn_node.args[0].meta["val"]
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]

        num_heads = q_fake.shape[2]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = k_fake.dtype

        cls._track_layer_config(num_kv_heads, head_dim, dtype)
        layer_idx = cls._get_next_layer_idx()

        tokens_per_block = _trtllm_config.page_size
        max_num_requests = _trtllm_config.max_batch_size
        max_context_length = _trtllm_config.max_seq_len

        return [
            layer_idx,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
            tokens_per_block,
            max_num_requests,
            max_context_length,
        ]


# =============================================================================
# Public API Functions
# =============================================================================


def reset_trtllm_attention_state() -> None:
    """Reset all TRT-LLM attention state."""
    TrtllmAttention._reset_layer_counter()
