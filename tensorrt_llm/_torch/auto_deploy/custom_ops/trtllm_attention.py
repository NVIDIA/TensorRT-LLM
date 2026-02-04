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

from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    CacheConfig,
    Constant,
    MHACallable,
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
    """

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate workspace buffer (64 MB) - reuses existing if already allocated."""
        # Check if workspace already exists (stored in _global_state)
        if _global_state.workspace is not None:
            return _global_state.workspace

        # Allocate new workspace
        buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=sequence_info.device)
        _global_state.init_workspace(buffer)
        return buffer


class TrtllmKVResourceHandler(ResourceHandler):
    """Resource handler for TRT-LLM unified KV cache.

    Uses ResourceHandler (not PagedResourceHandler) so the interface calls allocate()
    directly, allowing us to create the cache with the exact layout thop.attention expects.

    Uses kv_factor=2 (unified K+V) and kv_layout="HND" to match what thop.attention expects.
    The cache is allocated with shape [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim].
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
        # Store attributes for TRT-LLM attention
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.kv_factor = 2  # Unified K+V cache
        self.kv_layout = "HND"  # Matches thop.attention kernel's per-block layout
        self.layer_idx = layer_idx
        self._trtllm_config = trtllm_config
        self._cache_config = cache_config

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate cache via KVCacheManager or simple allocation."""
        # Configure global state first (first time only)
        if not self._trtllm_config.is_configured:
            self._trtllm_config.configure(sequence_info)

        # Set model config for FP8 KV cache support (first time only)
        if self._trtllm_config._num_layers == 0:
            cache_dtype = self.dtype
            self._trtllm_config.set_model_config(
                num_layers=len(TrtllmAttention._num_kv_heads_per_layer),
                num_kv_heads_per_layer=TrtllmAttention._num_kv_heads_per_layer,
                head_dim=TrtllmAttention._head_dim,
                dtype=cache_dtype,
            )

        # Allocate unified KV cache with correct layout for thop.attention
        # Shape: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim] (HND layout)
        cache = torch.empty(
            sequence_info.num_blocks,
            self.kv_factor,  # 2 for K and V
            self.num_kv_heads,
            sequence_info.tokens_per_block,
            self.head_dim,
            device=sequence_info.device,
            dtype=self.dtype,
        )
        return cache


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
        # Pool mapping needs to be pre-allocated before init_from_shared
        # Other tensors will come from shared state via init_from_shared()
        if self.host_kv_cache_pool_mapping is None:
            # Pool mapping: 2D [num_layers, 2] format expected by thop.attention
            max_layers = 256
            self.host_kv_cache_pool_mapping = torch.zeros(
                max_layers, 2, dtype=torch.int32, device="cpu", pin_memory=True
            )

    def init_from_shared(self, global_state: "TrtllmAttentionGlobalState") -> None:
        """Initialize layer to use shared tensors from global state."""
        # All layers share the same tensors (single KV cache pool)
        self.sequence_length = global_state._shared_sequence_length
        self.context_lengths = global_state._shared_context_lengths
        self.kv_cache_block_offsets = global_state._shared_kv_cache_block_offsets
        self.host_past_key_value_lengths = global_state._shared_host_past_key_value_lengths
        self.host_context_lengths = global_state._shared_host_context_lengths
        self.host_request_types = global_state._shared_host_request_types
        self.host_total_kv_lens = global_state._shared_host_total_kv_lens
        self.host_kv_cache_pool_pointers = global_state._shared_host_kv_cache_pool_pointers
        # Keep host_kv_cache_pool_mapping from __post_init__ - it's layer-specific


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
    ) -> None:
        """Initialize shared tensors used by all layers."""
        if self._shared_tensors_initialized:
            return

        device = "cuda"
        max_blocks_per_seq = (max_context_length + tokens_per_block - 1) // tokens_per_block

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
        # Using 64 as max layers (typical transformer max)
        self._shared_host_kv_cache_pool_mapping = torch.zeros(
            64, 2, dtype=torch.int32, device="cpu", pin_memory=True
        )

        self._shared_tensors_initialized = True

    def _init_cpu_buffers(self, max_seqs: int) -> None:
        """Initialize pre-allocated CPU buffers to avoid tensor allocation in hot path."""
        if self._cpu_buffers_initialized:
            return

        # Pre-allocate pinned CPU buffers for intermediate computations
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
        if self._gpu_buffers_initialized:
            return

        # Pre-allocate buffers with max sizes to avoid per-call allocations
        self._gpu_cu_pages = torch.zeros(max_seqs + 1, dtype=torch.long, device="cuda")
        self._gpu_page_positions = torch.arange(max_pages, dtype=torch.long, device="cuda")
        self._gpu_seq_idx = torch.zeros(max_pages, dtype=torch.long, device="cuda")
        self._gpu_page_idx = torch.zeros(max_pages, dtype=torch.long, device="cuda")
        self._gpu_base_offset = torch.zeros(max_pages, dtype=torch.int32, device="cuda")
        self._gpu_buffers_initialized = True

    def _init_pool_pointers(
        self, ad_pool_pointers: torch.Tensor, ad_pool_mapping: torch.Tensor, num_layers: int
    ) -> None:
        """Initialize pool pointers once from AD's KVCacheManager.

        This is called once during first host_prepare to set up the static pool info.
        Pool pointers don't change between requests - only block offsets do.
        """
        if self._pool_pointers_initialized:
            return

        if ad_pool_pointers is None or ad_pool_mapping is None:
            return

        if ad_pool_pointers.numel() == 0 or ad_pool_pointers[0, 0].item() == 0:
            return

        # Set pool pointers (these are static for the lifetime of the cache)
        self._shared_host_kv_cache_pool_pointers[0, 0] = ad_pool_pointers[0, 0].item()
        self._shared_host_kv_cache_pool_pointers[0, 1] = 0  # v_ptr=0 for interleaved

        # Set pool mapping for all layers
        for layer_i in range(min(num_layers, ad_pool_mapping.shape[0])):
            self._shared_host_kv_cache_pool_mapping[layer_i, 0] = ad_pool_mapping[layer_i, 0].item()
            self._shared_host_kv_cache_pool_mapping[layer_i, 1] = ad_pool_mapping[layer_i, 1].item()

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
        # Initialize shared tensors once (used by all layers)
        if not self._shared_tensors_initialized:
            self._init_shared_tensors(max_num_requests, max_context_length, tokens_per_block)

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
            # Link layer to shared tensors
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

            # Fast integer extraction (avoid .item() overhead by using int() directly)
            num_prefill = int(batch_info_host[0])
            num_decode = int(batch_info_host[2])
            num_seq = num_prefill + num_decode

            if num_seq == 0:
                return

            first_state = next(iter(layer_states.values()))
            if first_state.sequence_length is None:
                return

            num_layers = first_state.num_layers if first_state.num_layers > 0 else 32

            # Initialize buffers once (lazy initialization)
            if not global_state._cpu_buffers_initialized:
                global_state._init_cpu_buffers(first_state.max_num_requests)

            if not global_state._gpu_buffers_initialized:
                max_pages = first_state.max_num_requests * (
                    (first_state.max_context_length + first_state.tokens_per_block - 1)
                    // first_state.tokens_per_block
                )
                global_state._init_gpu_buffers(max_pages, first_state.max_num_requests)

            # Initialize pool pointers once (static for cache lifetime)
            if not global_state._pool_pointers_initialized:
                if _trtllm_config._sequence_info is not None:
                    ad_pool_pointers = _trtllm_config._sequence_info.kv_cache_pool_pointers
                    ad_pool_mapping = _trtllm_config._sequence_info.kv_cache_pool_mapping
                    global_state._init_pool_pointers(ad_pool_pointers, ad_pool_mapping, num_layers)

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

            # Compute totals (avoid .item() by keeping as tensor)
            context_total_kv = seq_len_with_cache[:num_prefill].sum() if num_prefill > 0 else 0
            gen_total_kv = seq_len_with_cache[num_prefill:num_seq].sum() if num_decode > 0 else 0

            # Block offsets info - use pre-allocated buffers
            cu_num_pages = global_state._cpu_cu_num_pages[: num_seq + 1]
            cu_num_pages.copy_(cu_num_pages_host[: num_seq + 1])

            pages_per_seq = global_state._cpu_pages_per_seq[:num_seq]
            torch.sub(cu_num_pages[1 : num_seq + 1], cu_num_pages[:num_seq], out=pages_per_seq)

            # Get max and total (unavoidable .item() for control flow)
            max_blocks = int(pages_per_seq.max())
            global_state.set_max_blocks_per_seq(max_blocks)
            total_pages = int(cu_num_pages[num_seq])

            # H2D copies using pinned memory (non-blocking)
            global_state._shared_sequence_length[:num_seq].copy_(
                seq_len_with_cache, non_blocking=True
            )
            global_state._shared_context_lengths[:num_seq].copy_(input_seq_lens, non_blocking=True)

            # Fill shared host tensors (CPU to CPU, fast)
            global_state._shared_host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
            global_state._shared_host_context_lengths[:num_seq].copy_(input_seq_lens)
            if num_prefill > 0:
                global_state._shared_host_request_types[:num_prefill].fill_(0)
            if num_decode > 0:
                global_state._shared_host_request_types[num_prefill:num_seq].fill_(1)
            global_state._shared_host_total_kv_lens[0] = context_total_kv
            global_state._shared_host_total_kv_lens[1] = gen_total_kv

            # Compute block offsets on GPU (vectorized)
            if total_pages > 0:
                # Copy cu_pages to GPU (small tensor)
                global_state._gpu_cu_pages[: num_seq + 1].copy_(cu_num_pages)
                cu_num_pages_gpu = global_state._gpu_cu_pages[: num_seq + 1]

                # Use pre-allocated page_positions slice
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

                # base_offsets on GPU
                kv_factor = 2
                multiplier = num_layers * kv_factor
                torch.mul(
                    cache_loc[:total_pages],
                    multiplier,
                    out=global_state._gpu_base_offset[:total_pages],
                )
                base_offsets = global_state._gpu_base_offset[:total_pages]

                # Fill block offsets using advanced indexing (only zero the slice we need)
                global_state._shared_kv_cache_block_offsets[:, :num_seq, :, :].zero_()
                global_state._shared_kv_cache_block_offsets[0, seq_indices, 0, page_in_seq] = (
                    base_offsets
                )
                global_state._shared_kv_cache_block_offsets[0, seq_indices, 1, page_in_seq] = (
                    base_offsets + 1
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

    Args:
        batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
        cu_seqlen_host: Cumulative sequence lengths [num_seq + 1]
        cu_num_pages: Cumulative page counts [num_seq + 1]
        cu_num_pages_host: Same as cu_num_pages but on host
        cache_loc: Flat page indices for all sequences
        last_page_len: Tokens in last page per sequence
        last_page_len_host: Same on host
        seq_len_with_cache_host: Total seq length including cached tokens
        state: Per-layer TRT-LLM state
        kv_cache: Unified KV cache tensor
        ad_pool_pointers: Optional AD pool pointers
        ad_pool_mapping: Optional AD pool mapping

    Returns:
        Tuple of tensors needed by thop.attention
    """
    # Check if in CUDA graph capture mode
    is_capturing = torch.cuda.is_current_stream_capturing()

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

    # Extract batch info
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # CUDA GRAPH CAPTURE: Set host tensors to MAX (kernel requirement)
    if is_capturing:
        max_seq = state.max_context_length
        state.host_past_key_value_lengths[:num_seq].fill_(max_seq)
        state.host_context_lengths[:num_seq].fill_(max_seq)
        state.host_request_types[:num_seq].fill_(1)
        state.host_total_kv_lens[0] = 0
        state.host_total_kv_lens[1] = max_seq * num_seq

        # Return tensors for capture
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
    # This handles model tracing/evaluation phase - do full computation once

    # Validate kv_cache shape
    if len(kv_cache.shape) != 5 or kv_cache.shape[1] != 2:
        raise RuntimeError(
            f"Expected kv_cache shape [pages, 2, heads, tokens, dim], got {kv_cache.shape}"
        )

    num_layers = state.num_layers if state.num_layers > 0 else 32

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

    # Set up KV cache pool pointers
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

    # Block offsets: convert flat cache_loc to per-sequence block indices
    pages_per_seq = (cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]).int()
    max_blocks = pages_per_seq.max().item() if num_seq > 0 else 1
    _global_state.set_max_blocks_per_seq(max_blocks)

    # Fill block offsets
    kv_factor = 2
    multiplier = num_layers * kv_factor
    state.kv_cache_block_offsets.zero_()
    offset = 0
    for i in range(num_seq):
        n_pages = pages_per_seq[i].item()
        if n_pages > 0:
            base_offsets = cache_loc[offset : offset + n_pages] * multiplier
            state.kv_cache_block_offsets[0, i, 0, :n_pages] = base_offsets
            state.kv_cache_block_offsets[0, i, 1, :n_pages] = base_offsets + 1
            offset += n_pages

    # Return tensors
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
    """TRT-LLM attention with unified KV cache for Auto-Deploy.

    This wraps thop.attention() with AD's metadata and cache interface.

    Args:
        kv_cache: Unified KV cache tensor with shape:
            - From KVCacheManager: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim] (HND layout)
            - The kv_factor dimension contains K at index 0 and V at index 1

    Note: This implementation assumes:
    - RoPE is applied OUTSIDE this kernel (AD's pattern)
    - No speculative decoding
    - No MLA
    - Causal attention mask
    """
    # Validate inputs
    if not q.is_cuda:
        raise RuntimeError(f"Q must be on CUDA, got {q.device}")
    if not kv_cache.is_cuda:
        raise RuntimeError(f"kv_cache must be on CUDA, got {kv_cache.device}")

    # Validate unified KV cache format
    # Expected shape: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim] (HND layout)
    # This shape is created by TrtllmKVResourceHandler.allocate() which permutes the base allocation
    assert kv_cache.dim() == 5, f"kv_cache must be 5D, got {kv_cache.dim()}D"
    assert kv_cache.shape[1] == 2, (
        f"kv_cache.shape[1] must be 2 (kv_factor), got {kv_cache.shape[1]}"
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
    # Shape: [num_tokens, (num_heads + 2 * num_kv_heads) * head_dim]
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # Prepare output tensor
    output = torch.empty(num_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device)

    # Get num_layers from config for block offset calculation
    num_layers = _trtllm_config._num_layers if _trtllm_config._num_layers > 0 else 32
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

    # Get AD's pool pointers if available (for proper integration)
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
        kv_cache,  # Unified KV cache
        ad_pool_pointers=ad_pool_pointers,
        ad_pool_mapping=ad_pool_mapping,
    )

    # Compute softmax scale
    # sm_scale = scale if scale is not None else (1.0 / (head_dim**0.5))

    # Attention window (full attention)
    attention_window_size = max_context_length

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    # Add extra params for newer TRT-LLM versions
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
            _trtllm_config._kv_scale_orig_quant,  # kv_scale_orig_quant (FP8 KV cache)
            _trtllm_config._kv_scale_quant_orig,  # kv_scale_quant_orig (FP8 KV cache)
            None,  # out_scale
            None,  # rotary_inv_freq
            None,  # rotary_cos_sin
            None,  # latent_cache (MLA)
            None,  # q_pe (MLA)
            None,  # block_ids_per_seq
            None,  # attention_sinks
            True,  # is_fused_qkv (Q contains [Q,K,V] concatenated)
            True,  # update_kv_cache
            1,  # predicted_tokens_per_seq
            layer_idx,  # layer_idx
            num_heads,  # num_heads
            num_kv_heads,  # num_kv_heads
            head_dim,  # head_size
            kv_cache.shape[3],  # tokens_per_block - use actual value from kv_cache shape!
            max_num_requests,  # max_num_requests
            max_context_length,  # max_context_length
            attention_window_size,  # attention_window_size
            0,  # sink_token_length
            1,  # beam_width
            int(AttentionMaskType.causal),  # mask_type
            _trtllm_config._quant_mode,  # quant_mode (128 for FP8 KV cache, 0 otherwise)
            1.0,  # q_scaling (scaling factor applied to Q, typically 1.0)
            0,  # position_embedding_type (none - RoPE applied outside)
            0,  # rotary_embedding_dim
            10000.0,  # rotary_embedding_base
            0,  # rotary_embedding_scale_type
            rotary_embedding_scales,  # rotary_embedding_scales
            rotary_embedding_max_position_info,  # rotary_embedding_max_position_info
            True,  # use_paged_context_fmha - True for paged KV cache
            0,  # attention_input_type - 0=mixed (context + generation)
            False,  # is_mla_enable
            max(1, num_prefill),  # chunked_prefill_buffer_batch_size
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
        output_padded[:num_tokens] = output
        output = output_padded

    return output.view(b, s, num_heads * head_dim)


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
    """Configuration holder for TRT-LLM attention backend.

    This class stores runtime configuration that's set during cache initialization
    and used when constructing the attention op constants.
    """

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

        # Store SequenceInfo reference for AD pool pointer access
        self._sequence_info = si

    def set_model_config(
        self,
        num_layers: int,
        num_kv_heads_per_layer: List[int],
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Set model configuration.

        Args:
            num_layers: Total number of attention layers
            num_kv_heads_per_layer: Number of KV heads for each layer
            head_dim: Head dimension
            dtype: Cache data type
        """
        self._num_layers = num_layers
        self._num_kv_heads_per_layer = num_kv_heads_per_layer
        self._head_dim = head_dim
        self._dtype = dtype

        # Initialize FP8 KV cache support if dtype is FP8
        if dtype == torch.float8_e4m3fn:
            # FP8_KV_CACHE quant mode = 128 (from tensorrt_llm.quantization.mode.QuantMode)
            self._quant_mode = 128
            # Default KV scales (1.0) - for FP8 models, scale is typically 1.0
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
    """TRT-LLM attention backend for Auto-Deploy.

    This backend uses the optimized thop.attention kernel from TRT-LLM,
    providing better performance than FlashInfer on certain workloads.

    Note: This backend assumes RoPE is applied outside the attention kernel,
    which matches AD's current pattern.

    Usage:
        Set `backend: trtllm` in your AD config under `insert_cached_attention`.
    """

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
        """Track layer configuration.

        This is called for each layer during graph analysis to collect
        the per-layer KV head counts needed for model configuration.
        """
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
        # TRT-LLM doesn't need extra metadata preparation like FlashInfer
        return (None, 0, [])

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> ResourceHandlerDict:
        """Provide resource handlers for cache initialization.

        Returns a single unified KV cache handler (kv_factor=2) that can be managed
        by AD's KVCacheManager for direct pool integration.
        """
        # Extract tensor shapes from source node
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        # Resolve dtype from config string to torch.dtype
        dtype = cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype)

        # Get current layer index
        layer_idx = cls._layer_counter  # Current layer being processed

        return {
            # Single unified KV cache with kv_factor=2 (K and V combined)
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
        """Get function that performs host-side prep for attention.

        Returns host_prepare function that runs OUTSIDE CUDA graphs to update tensors.
        """
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
        # Sanity check layout
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

        # Extract attention parameters
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        # Get scale
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        # Extract tensor shapes for constants
        q_fake: FakeTensor = source_attn_node.args[0].meta["val"]
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]

        num_heads = q_fake.shape[2]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = k_fake.dtype

        # Track layer configuration for model config
        cls._track_layer_config(num_kv_heads, head_dim, dtype)

        # Get layer index
        layer_idx = cls._get_next_layer_idx()

        # Use configured values if available, otherwise defaults
        tokens_per_block = _trtllm_config.page_size
        max_num_requests = _trtllm_config.max_batch_size
        max_context_length = _trtllm_config.max_seq_len

        # Return constants in order expected by trtllm_mha_with_cache
        return [
            layer_idx,  # layer_idx
            scale,  # scale
            num_heads,  # num_heads
            num_kv_heads,  # num_kv_heads
            head_dim,  # head_dim
            tokens_per_block,  # tokens_per_block (from AD's page_size)
            max_num_requests,  # max_num_requests (from AD's max_batch_size)
            max_context_length,  # max_context_length (from AD's max_seq_len)
        ]


# =============================================================================
# Public API Functions
# =============================================================================


def reset_trtllm_attention_state() -> None:
    """Reset all TRT-LLM attention state.

    Call this before building a new model to ensure clean state.
    This resets:
    - Layer counter
    - Global state (per-layer states, workspace)
    - Configuration (page size, batch size, etc.)
    """
    TrtllmAttention._reset_layer_counter()
