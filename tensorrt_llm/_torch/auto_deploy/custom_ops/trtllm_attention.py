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

Cache Backend Options:
---------------------
- SimpleCacheBackend (default): Per-layer cache allocation, Python metadata prep
- PTCacheBackend: Unified pool via PT's KVCacheManager, C++ fast path for metadata

Set `use_pt_cache_backend=True` in TrtllmAttentionConfig to use PTCacheBackend.
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
    KVPagedResourceHandler,
    MHACallable,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)

# Import cache backends

# PTCacheBackend is optional - only import if available
try:
    from .pt_cache_backend import PTCacheBackend, PTCacheConfig

    _HAS_PT_CACHE_BACKEND = True
except ImportError:
    _HAS_PT_CACHE_BACKEND = False
    PTCacheBackend = None
    PTCacheConfig = None


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
        ad_logger.debug(f"[TRT-LLM] Initialized workspace: {buffer.shape}, device={buffer.device}")
        return buffer


class TrtllmKVResourceHandler(KVPagedResourceHandler):
    """Resource handler for TRT-LLM unified KV cache.

    Extends KVPagedResourceHandler so the cache interface recognizes it as a paged resource
    and creates a proper KVCacheManager with correct parameters (num_layers, num_kv_heads, etc.).

    Uses kv_factor=2 (unified K+V) and kv_layout="HND" to match what thop.attention expects.
    When __eq__ returns True, KVCacheManager manages the cache and we use AD's pool directly.
    When use_pt_cache_backend=True, PTCacheBackend creates its own separate cache management.
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
        # Initialize parent class with kv_factor=2 (unified K+V) and HND layout
        # HND = [num_kv_heads, tokens_per_block, head_dim] per block - matches kernel expectation
        super().__init__(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_factor=2,  # Unified K+V cache
            kv_layout="HND",  # Matches thop.attention kernel's per-block layout
        )
        self.layer_idx = layer_idx
        self._trtllm_config = trtllm_config
        self._cache_config = cache_config

    def __eq__(self, other: "KVPagedResourceHandler") -> bool:
        """Check compatibility for KVCacheManager resource grouping.

        When use_pt_cache_backend=True, return False to use PTCacheBackend's separate management.
        When use_pt_cache_backend=False, return True so KVCacheManager manages all layers'
        KV caches with correct num_blocks calculation and we use AD's pool directly.
        """
        # When PTCacheBackend is enabled, don't let KVCacheManager manage this
        if self._trtllm_config.use_pt_cache_backend:
            return False

        # For direct AD pool integration, match compatible handlers
        if not isinstance(other, KVPagedResourceHandler):
            return False
        return (
            self.head_dim == other.head_dim
            and self.dtype == other.dtype
            and self.kv_factor == other.kv_factor
            and self.kv_layout == other.kv_layout
        )

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        """Allocate cache - either via PTCacheBackend, KVCacheManager, or simple allocation."""
        # Configure global state first (first time only)
        if not self._trtllm_config.is_configured:
            self._trtllm_config.configure(sequence_info)

        # Set model config for FP8 KV cache support (first time only, regardless of backend)
        # This must be done for BOTH PTCacheBackend and direct AD pool paths
        if self._trtllm_config._num_layers == 0:
            cache_dtype = self.dtype
            self._trtllm_config.set_model_config(
                num_layers=len(TrtllmAttention._num_kv_heads_per_layer),
                num_kv_heads_per_layer=TrtllmAttention._num_kv_heads_per_layer,
                head_dim=TrtllmAttention._head_dim,
                dtype=cache_dtype,
            )

        # Try PTCacheBackend if enabled
        if self._trtllm_config.use_pt_cache_backend and _HAS_PT_CACHE_BACKEND:
            # Get or create PT backend
            pt_backend = self._trtllm_config.get_or_create_pt_cache_backend(sequence_info)

            if pt_backend is not None:
                # Register host prepare function ONCE (only for layer 0)
                if self.layer_idx == 0:
                    host_fn = pt_backend.get_host_prepare_metadata_function()
                    if host_fn is not None:
                        host_args = pt_backend.get_host_prepare_metadata_args()
                        sequence_info.register_host_prepare_for_attention_forward(
                            host_fn, host_args
                        )

                # PTCacheBackend returns unified K+V cache for each layer
                return pt_backend.get_unified_cache(self.layer_idx)

        # Fallback: simple allocation (when __eq__=True, KVCacheManager handles this instead)
        # This is called when __eq__=False or as a safety fallback
        ad_logger.warning(
            f"[TRT-LLM] Fallback allocation for layer {self.layer_idx} - "
            "consider using PTCacheBackend or ensuring KVCacheManager manages the cache"
        )

        # Unified KV cache format: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        # This matches HND layout with kv_factor dimension
        cache = torch.empty(
            sequence_info.num_blocks,
            self.kv_factor,  # 2 for K and V
            self.num_kv_heads,
            sequence_info.tokens_per_block,
            self.head_dim,
            device=sequence_info.device,
            dtype=self.dtype,
        )
        ad_logger.debug(
            f"[TRT-LLM] Created unified KV cache: shape={cache.shape}, dtype={cache.dtype}, "
            f"device={cache.device}"
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
        """Allocate pre-sized tensors."""
        if self.sequence_length is None:
            device = "cuda"

            # Device tensors
            self.sequence_length = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device=device
            )
            self.context_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device=device
            )

            # Pre-allocate kv_cache_block_offsets with MAX size for CUDA graph stability
            max_blocks_per_seq = (
                self.max_context_length + self.tokens_per_block - 1
            ) // self.tokens_per_block
            self.kv_cache_block_offsets = torch.zeros(
                1,  # num_pools
                self.max_num_requests,
                2,  # K and V
                max_blocks_per_seq,
                dtype=torch.int32,
                device=device,
            )

            # Host tensors (pinned memory for async transfers)
            self.host_past_key_value_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_context_lengths = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_request_types = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.host_total_kv_lens = torch.zeros(
                2, dtype=torch.int64, device="cpu", pin_memory=True
            )
            # Pool pointers: [num_pools, 2] where each row is [k_cache_ptr, v_cache_ptr]
            # thop.attention expects 2D tensor: [num_pools, 2]
            self.host_kv_cache_pool_pointers = torch.zeros(
                1, 2, dtype=torch.int64, device="cpu", pin_memory=True
            )
            # Pool mapping: 2D [num_layers, 2] format expected by thop.attention
            # pool_mapping[layer, 0] = pool_idx (0 for single pool)
            # pool_mapping[layer, 1] = layer_offset (0 when using per-layer pointers)
            # Use max 256 layers to cover most models
            max_layers = 256
            self.host_kv_cache_pool_mapping = torch.zeros(
                max_layers, 2, dtype=torch.int32, device="cpu", pin_memory=True
            )


class TrtllmAttentionGlobalState:
    """Global state manager for TRT-LLM attention layers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._layer_states: Dict[int, TrtllmLayerState] = {}
            cls._instance._workspace: Optional[torch.Tensor] = None
            cls._instance._max_blocks_per_seq: int = 0
        return cls._instance

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
        if layer_idx not in self._layer_states:
            self._layer_states[layer_idx] = TrtllmLayerState(
                layer_idx=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_num_requests=max_num_requests,
                max_context_length=max_context_length,
                num_layers=num_layers,
            )
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
        It updates tensors with current batch values.
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
            """Fill device/host tensors before graph replay."""
            if not layer_states:
                return

            num_prefill = int(batch_info_host[0].item())
            num_decode = int(batch_info_host[2].item())
            num_seq = num_prefill + num_decode

            if num_seq == 0:
                return

            first_state = next(iter(layer_states.values()))
            if first_state.sequence_length is None:
                return

            # Compute metadata
            input_seq_lens = (cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]).int()
            seq_len_with_cache = seq_len_with_cache_host[:num_seq].int()
            past_kv_lens = seq_len_with_cache - input_seq_lens.cpu()

            context_total_kv = (
                seq_len_with_cache[:num_prefill].sum().item() if num_prefill > 0 else 0
            )
            gen_total_kv = (
                seq_len_with_cache[num_prefill:num_seq].sum().item() if num_decode > 0 else 0
            )

            num_layers = first_state.num_layers if first_state.num_layers > 0 else 32

            # Block offsets info
            pages_per_seq = (cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]).int()
            max_blocks = pages_per_seq.max().item() if num_seq > 0 else 1
            global_state.set_max_blocks_per_seq(max_blocks)

            kv_factor = 2
            multiplier = num_layers * kv_factor

            # Update ALL layer states
            for state in layer_states.values():
                if state.sequence_length is None:
                    continue

                # kv_cache_block_offsets is pre-allocated in TrtllmLayerState.__post_init__
                # Don't reallocate to maintain stable tensor address for CUDA graphs
                if state.kv_cache_block_offsets is None:
                    # This shouldn't happen after __post_init__, but just in case
                    ctx_len = state.max_context_length
                    tpb = state.tokens_per_block
                    max_blocks_per_seq = (ctx_len + tpb - 1) // tpb
                    state.kv_cache_block_offsets = torch.zeros(
                        1,
                        state.max_num_requests,
                        2,
                        max_blocks_per_seq,
                        dtype=torch.int32,
                        device="cuda",
                    )

                # Fill device tensors
                state.sequence_length[:num_seq].copy_(seq_len_with_cache.cuda(), non_blocking=True)
                state.context_lengths[:num_seq].copy_(input_seq_lens.cuda(), non_blocking=True)

                # Fill host tensors
                state.host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
                state.host_context_lengths[:num_seq].copy_(input_seq_lens.cpu())
                if num_prefill > 0:
                    state.host_request_types[:num_prefill].fill_(0)
                if num_decode > 0:
                    state.host_request_types[num_prefill:num_seq].fill_(1)
                state.host_total_kv_lens[0] = context_total_kv
                state.host_total_kv_lens[1] = gen_total_kv

                # Fill block offsets
                state.kv_cache_block_offsets.zero_()
                offset = 0
                for i in range(num_seq):
                    n_pages = pages_per_seq[i].item()
                    if n_pages > 0:
                        base_offsets = cache_loc[offset : offset + n_pages] * multiplier
                        state.kv_cache_block_offsets[0, i, 0, :n_pages] = base_offsets
                        state.kv_cache_block_offsets[0, i, 1, :n_pages] = base_offsets + 1
                        offset += n_pages

            # Synchronize to ensure all copies complete before graph replay
            torch.cuda.synchronize()

        return _host_prepare_trtllm_metadata

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        self._layer_states.clear()
        self._workspace = None
        self._max_blocks_per_seq = 0


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

    For CUDA graph support (like pt_cache_backend):
    - During capture: Set host tensors to MAX, skip device operations
    - Outside capture: Normal operation

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
        kv_cache: Unified KV cache tensor [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        ad_pool_pointers: Optional AD pool pointers from KVCacheManager (shape: [num_pools, 2])
        ad_pool_mapping: Optional AD pool mapping from KVCacheManager (shape: [num_layers, 2])

    Returns:
        Tuple of tensors needed by thop.attention
    """
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # Check if in CUDA graph capture mode
    is_capturing = torch.cuda.is_current_stream_capturing()

    # Compute input sequence lengths from cumulative sums
    input_seq_lens = (cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]).int()
    seq_len_with_cache = seq_len_with_cache_host[:num_seq].int()
    past_kv_lens = seq_len_with_cache - input_seq_lens.cpu()

    # CUDA GRAPH FIX: Set host tensors to MAX during capture (like pt_cache_backend)
    if is_capturing:
        max_seq = state.max_context_length
        state.host_past_key_value_lengths[:num_seq].fill_(max_seq)
        state.host_context_lengths[:num_seq].fill_(max_seq)
        state.host_request_types[:num_seq].fill_(1)
        state.host_total_kv_lens[0] = 0
        state.host_total_kv_lens[1] = max_seq * num_seq
    else:
        # Normal operation: fill host tensors
        state.host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
        state.host_context_lengths[:num_seq].copy_(input_seq_lens.cpu())
        state.host_request_types[:num_prefill].fill_(0)
        state.host_request_types[num_prefill:num_seq].fill_(1)
        context_total_kv = seq_len_with_cache[:num_prefill].sum().item() if num_prefill > 0 else 0
        gen_total_kv = seq_len_with_cache[num_prefill:num_seq].sum().item() if num_decode > 0 else 0
        state.host_total_kv_lens[0] = context_total_kv
        state.host_total_kv_lens[1] = gen_total_kv

    # Device operations - skip during capture (like pt_cache_backend's skip_device_ops)
    if not is_capturing:
        # Sync before copy to catch any previous async errors
        torch.cuda.synchronize()

        # Copy to pre-allocated tensors
        state.sequence_length[:num_seq].copy_(seq_len_with_cache.cuda())
        state.context_lengths[:num_seq].copy_(input_seq_lens.cuda())

    # Validate kv_cache shape (safe during capture - no device ops)
    if len(kv_cache.shape) != 5 or kv_cache.shape[1] != 2:
        raise RuntimeError(
            f"Expected kv_cache shape [pages, 2, heads, tokens, dim], got {kv_cache.shape}"
        )

    num_layers = state.num_layers if state.num_layers > 0 else 32

    # Pool pointer and block offset setup - skip during capture (contains .item() calls)
    if not is_capturing:
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

        # Log pool setup for debugging (only once)
        if state.layer_idx == 0 and not hasattr(state, "_pool_logged"):
            state._pool_logged = True
            ad_logger.debug(
                f"[TRT-LLM Attention] Using AD pool directly: "
                f"pool_ptr={state.host_kv_cache_pool_pointers[0, 0]}"
            )

        # Block offsets: convert flat cache_loc to per-sequence block indices
        pages_per_seq = (cu_num_pages_host[1 : num_seq + 1] - cu_num_pages_host[:num_seq]).int()
        max_blocks = pages_per_seq.max().item() if num_seq > 0 else 1
        _global_state.set_max_blocks_per_seq(max_blocks)

        # kv_cache_block_offsets is pre-allocated in __post_init__, don't reallocate

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
    # Use pre-allocated tensor size for block offsets (CUDA graph compatibility)
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
    # Expected shape: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim] (HND layout)
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

    # Check if PTCacheBackend is active - if so, use its metadata
    pt_backend = _trtllm_config.pt_cache_backend
    if pt_backend is not None:
        # PTCacheBackend's metadata is prepared via two mechanisms:
        # 1. During forward (warmup/capture/normal): host_prepare_fn is called here
        # 2. During inference: run_host_prepare_for_attention_forward() calls registered fn
        #    BEFORE graph replay (updates device tensors before replay)
        #
        # CUDA graph handling:
        # - HOST tensor VALUES are baked into the graph at capture time
        # - DEVICE tensor ADDRESSES are captured (data can be updated before replay)
        #
        # Therefore:
        # - During capture: call host_prepare_fn with skip_device_ops=True
        #   (sets host tensors correctly for capture, skips H2D copies that aren't allowed)
        # - Outside capture: call host_prepare_fn normally (sets all tensors)
        is_capturing = torch.cuda.is_current_stream_capturing()
        host_prepare_fn = pt_backend.get_host_prepare_metadata_function()
        if host_prepare_fn is not None:
            host_prepare_fn(
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages_host,
                cache_loc,
                seq_len_with_cache_host,
                skip_device_ops=is_capturing,  # Skip H2D during capture
            )

        # Get metadata from PTCacheBackend
        # NOTE: We slice tensors to num_seq because the TRT-LLM kernel uses
        # tensor SIZE to determine batch size. This is incompatible with CUDA
        # graphs since slicing creates different addresses each time.
        sequence_length = pt_backend.sequence_length[:num_seq]
        host_past_key_value_lengths = pt_backend.host_past_key_value_lengths[:num_seq]
        host_total_kv_lens = pt_backend.host_total_kv_lens
        context_lengths = pt_backend.context_lengths[:num_seq]
        host_context_lengths = pt_backend.host_context_lengths[:num_seq]
        host_request_types = pt_backend.host_request_types[:num_seq]

        # Get block offsets from PTCacheBackend - shape [1, num_seq, 2, max_blocks]
        # PTCacheBackend already sets K/V at different indices (*2 for K, *2+1 for V)
        # in _fill_block_offsets_from_cache_loc(), so use directly
        kv_cache_block_offsets = pt_backend.kv_cache_block_offsets[:, :num_seq, :, :]

        # Get pool pointers directly from C++ KVCacheManager: [[base_ptr, 0]]
        # The C++ pool stores data in [heads, tokens, dim] layout per block,
        # which matches what the kernel expects - no transpose needed!
        host_kv_cache_pool_pointers = pt_backend.get_pool_pointers()
        host_kv_cache_pool_mapping = pt_backend.get_pool_mapping()
    else:
        # Fall back to original metadata preparation
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

        # Prepare TRT-LLM metadata using fallback
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

    # Debug: log tensor info before attention call
    if layer_idx == 0:
        ad_logger.debug(
            f"[TRT-LLM Attention L{layer_idx}] qkv_fused={qkv_fused.shape}, dtype={qkv_fused.dtype}, "
            f"output={output.shape}, workspace={workspace_buffer.shape}"
        )
        ad_logger.debug(
            f"[TRT-LLM Attention L{layer_idx}] sequence_length={sequence_length.shape}, "
            f"context_lengths={context_lengths.shape}, kv_block_offsets={kv_cache_block_offsets.shape}"
        )
        ad_logger.debug(
            f"[TRT-LLM Attention L{layer_idx}] pool_pointers={host_kv_cache_pool_pointers.shape}, "
            f"pool_mapping={host_kv_cache_pool_mapping.shape}"
        )
        ad_logger.debug(
            f"[TRT-LLM Attention L{layer_idx}] num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, tokens_per_block={tokens_per_block}"
        )

    # DEBUG: Environment variable to skip thop.attention and use PyTorch SDPA
    import os

    use_debug_sdpa = os.environ.get("DEBUG_USE_SDPA", "0") == "1"

    if use_debug_sdpa:
        # Fall back to PyTorch SDPA for debugging (no cache update)
        ad_logger.warning(f"[DEBUG] Using PyTorch SDPA for layer {layer_idx} (DEBUG_USE_SDPA=1)")
        # Handle GQA: expand K/V to match Q's head count
        n_rep = num_heads // num_kv_heads  # Repeat factor for GQA
        q_sdpa = (
            q.reshape(num_tokens, num_heads, head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(torch.bfloat16)
        )
        k_for_sdpa = k.reshape(num_tokens, num_kv_heads, head_dim).transpose(
            0, 1
        )  # [kv_heads, tokens, dim]
        v_for_sdpa = v.reshape(num_tokens, num_kv_heads, head_dim).transpose(
            0, 1
        )  # [kv_heads, tokens, dim]
        # Repeat K/V for GQA
        k_sdpa = (
            k_for_sdpa.unsqueeze(1)
            .expand(-1, n_rep, -1, -1)
            .reshape(num_heads, num_tokens, head_dim)
            .unsqueeze(0)
            .to(torch.bfloat16)
        )
        v_sdpa = (
            v_for_sdpa.unsqueeze(1)
            .expand(-1, n_rep, -1, -1)
            .reshape(num_heads, num_tokens, head_dim)
            .unsqueeze(0)
            .to(torch.bfloat16)
        )
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True
        )
        output = (
            out_sdpa.squeeze(0)
            .transpose(0, 1)
            .reshape(num_tokens, num_heads * head_dim)
            .to(q.dtype)
        )
        # Pad output if needed
        if output.shape[0] < b * s:
            output_padded = torch.zeros(b * s, num_heads * head_dim, dtype=q.dtype, device=q.device)
            output_padded[:num_tokens] = output
            output = output_padded
        return output.view(b, s, num_heads * head_dim)

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
        ad_logger.error(f"TRT-LLM attention failed at layer {layer_idx}: {e}")
        ad_logger.error(f"  num_seq={num_seq}, num_tokens={num_tokens}")
        ad_logger.error(f"  q_flat.shape={q_flat.shape}, k_flat.shape={k_flat.shape}")
        ad_logger.error(f"  kv_cache.shape={kv_cache.shape}")
        # DEBUG: Fall back to PyTorch SDPA instead of crashing
        ad_logger.warning(f"[DEBUG] Falling back to PyTorch SDPA for layer {layer_idx}")
        # Simple SDPA without cache update (just for debugging)
        q_sdpa = q.reshape(num_tokens, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
        k_sdpa = k.reshape(num_tokens, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
        v_sdpa = v.reshape(num_tokens, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True
        )
        output = out_sdpa.squeeze(0).transpose(0, 1).reshape(num_tokens, num_heads * head_dim)
        # Skip the copy-back since we didn't use the kernel
        return (
            output.view(b, s, num_heads * head_dim)
            if output.shape[0] == b * s
            else torch.zeros(b, s, num_heads * head_dim, dtype=q.dtype, device=q.device)
        )

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

    Attributes:
        use_pt_cache_backend: If True, use PTCacheBackend with PT's KVCacheManager
            for efficient C++ metadata preparation. If False (default), use
            SimpleCacheBackend with Python-based metadata preparation.
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

        # Cache backend configuration
        self.use_pt_cache_backend: bool = False
        self._pt_cache_backend: Optional["PTCacheBackend"] = None
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
        # PR 11149 renamed page_size -> tokens_per_block
        self.page_size = si.tokens_per_block
        self.max_batch_size = si.max_batch_size
        self.max_seq_len = si.max_seq_len
        self.max_num_tokens = si.max_num_tokens
        self.is_configured = True

        # Store SequenceInfo reference for AD pool pointer access
        self._sequence_info = si

        ad_logger.info(
            f"[TRT-LLM Attention Config] page_size={self.page_size}, "
            f"max_batch_size={self.max_batch_size}, max_seq_len={self.max_seq_len}, "
            f"max_num_tokens={self.max_num_tokens}, use_pt_cache_backend={self.use_pt_cache_backend}"
        )

    def set_model_config(
        self,
        num_layers: int,
        num_kv_heads_per_layer: List[int],
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Set model configuration for PTCacheBackend.

        This should be called during model analysis before cache initialization.

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
            # These tensors must be on GPU for thop.attention
            self._kv_scale_orig_quant = torch.ones(1, dtype=torch.float32, device="cuda")
            self._kv_scale_quant_orig = torch.ones(1, dtype=torch.float32, device="cuda")
            ad_logger.info(
                f"[TRT-LLM] Enabled FP8 KV cache: quant_mode={self._quant_mode}, kv_scale=1.0"
            )
        else:
            self._quant_mode = 0
            self._kv_scale_orig_quant = None
            self._kv_scale_quant_orig = None

    def get_or_create_pt_cache_backend(self, si: SequenceInfo) -> Optional["PTCacheBackend"]:
        """Get or create the PTCacheBackend instance.

        This is called during cache initialization if use_pt_cache_backend is True.

        Args:
            si: SequenceInfo with cache configuration

        Returns:
            PTCacheBackend instance, or None if not configured to use it
        """
        if not self.use_pt_cache_backend:
            return None

        if not _HAS_PT_CACHE_BACKEND:
            ad_logger.warning(
                "[TRT-LLM] PTCacheBackend requested but not available. "
                "Falling back to SimpleCacheBackend."
            )
            return None

        if self._pt_cache_backend is not None:
            return self._pt_cache_backend

        # Validate we have model config
        if self._num_layers == 0 or not self._num_kv_heads_per_layer:
            ad_logger.error(
                "[TRT-LLM] Cannot create PTCacheBackend: model config not set. "
                "Call set_model_config() first."
            )
            return None

        # Calculate optimal num_blocks based on available GPU memory
        # si.num_blocks may be too small (from dummy KVCacheManager) - calculate our own
        #
        # Each block needs: tokens_per_block * num_kv_heads * head_dim * kv_factor * dtype_size
        # For all layers: multiply by num_layers
        dtype_size = (
            1 if self._dtype == torch.float8_e4m3fn else (2 if self._dtype == torch.float16 else 4)
        )
        kv_factor = 2  # K and V
        max_kv_heads = max(self._num_kv_heads_per_layer)
        bytes_per_block_per_layer = (
            si.tokens_per_block * max_kv_heads * self._head_dim * kv_factor * dtype_size
        )
        bytes_per_block_total = bytes_per_block_per_layer * self._num_layers

        # Get available GPU memory
        free_mem = torch.cuda.mem_get_info()[0]

        # Use 80% of free memory for KV cache (leave room for other allocations)
        mem_for_kv = int(free_mem * 0.80)
        optimal_blocks = max(64, mem_for_kv // bytes_per_block_total)

        # Cap at theoretical max to avoid wasting memory
        max_blocks_per_seq = (si.max_seq_len + si.tokens_per_block - 1) // si.tokens_per_block
        theoretical_max = max_blocks_per_seq * si.max_batch_size
        num_blocks = min(optimal_blocks, theoretical_max)

        ad_logger.info(
            f"[TRT-LLM PTCacheBackend] Calculated num_blocks={num_blocks} "
            f"(optimal={optimal_blocks}, theoretical_max={theoretical_max}, "
            f"free_mem={free_mem / 1e9:.2f}GB, bytes_per_block={bytes_per_block_total})"
        )

        # PR 11149 renamed: num_pages -> num_blocks, page_size -> tokens_per_block
        config = PTCacheConfig(
            num_layers=self._num_layers,
            num_kv_heads_per_layer=self._num_kv_heads_per_layer,
            head_dim=self._head_dim,
            tokens_per_block=si.tokens_per_block,
            max_num_sequences=si.max_batch_size,
            max_seq_len=si.max_seq_len,
            num_pages=num_blocks,  # Use our calculated optimal blocks
            dtype=self._dtype,
        )

        # Create and initialize backend
        self._pt_cache_backend = PTCacheBackend(config)
        self._pt_cache_backend.initialize(si, si.device)

        ad_logger.info(
            f"[TRT-LLM] Created PTCacheBackend: num_layers={self._num_layers}, "
            f"num_blocks={si.num_blocks}, tokens_per_block={si.tokens_per_block}"
        )

        return self._pt_cache_backend

    @property
    def pt_cache_backend(self) -> Optional["PTCacheBackend"]:
        """Get the PTCacheBackend instance if available."""
        return self._pt_cache_backend


# Global config singleton
_trtllm_config = TrtllmAttentionConfig()


@AttentionRegistry.register("trtllm")
class TrtllmAttention(AttentionDescriptor):
    """TRT-LLM attention backend for Auto-Deploy.

    This backend uses the optimized thop.attention kernel from TRT-LLM,
    providing better performance than FlashInfer on certain workloads.

    Note: This backend assumes RoPE is applied outside the attention kernel,
    which matches AD's current pattern.

    Cache Backend Options:
        - SimpleCacheBackend (default): Per-layer cache allocation
        - PTCacheBackend: Uses PT's KVCacheManager with C++ fast path

    To enable PTCacheBackend, set:
        TrtllmAttentionConfig().use_pt_cache_backend = True

    Usage:
        Set `backend: trtllm` in your AD config under `insert_cached_attention`.
    """

    # Class-level counter for layer indices
    _layer_counter: int = 0

    # Track num_kv_heads per layer for PTCacheBackend config
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
        """Track layer configuration for PTCacheBackend setup.

        This is called for each layer during graph analysis to collect
        the per-layer KV head counts needed by PTCacheBackend.
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
        # Check if we're using PTCacheBackend
        if _trtllm_config.use_pt_cache_backend and _trtllm_config.pt_cache_backend is not None:
            return _trtllm_config.pt_cache_backend.get_host_prepare_metadata_function()

        # Non-PTCacheBackend: Return global state's host_prepare function
        return _global_state.create_host_prepare_function()

    @classmethod
    def get_host_prepare_metadata_args(cls) -> List[str]:
        """Get argument names for host_prepare function."""
        if _trtllm_config.use_pt_cache_backend and _trtllm_config.pt_cache_backend is not None:
            return _trtllm_config.pt_cache_backend.get_host_prepare_metadata_args()

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
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                f"Unsupported attention arguments for {source_attn_node=}: "
                f"{attn_mask=}, {dropout_p=}, {is_causal=}"
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

        # Track layer configuration for PTCacheBackend
        cls._track_layer_config(num_kv_heads, head_dim, dtype)

        # Get layer index
        layer_idx = cls._get_next_layer_idx()

        # Use configured values if available, otherwise defaults
        tokens_per_block = _trtllm_config.page_size
        max_num_requests = _trtllm_config.max_batch_size
        max_context_length = _trtllm_config.max_seq_len

        ad_logger.debug(
            f"[TRT-LLM] Layer {layer_idx} constants: num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, scale={scale}, "
            f"tokens_per_block={tokens_per_block}, max_num_requests={max_num_requests}, "
            f"max_context_length={max_context_length}"
        )

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


def enable_pt_cache_backend(enable: bool = True) -> None:
    """Enable or disable PTCacheBackend for TRT-LLM attention.

    When enabled, the TRT-LLM attention backend uses PT's KVCacheManager
    for efficient metadata preparation via C++ code paths.

    Benefits of PTCacheBackend:
    - ~50% faster metadata preparation (C++ vs Python loops)
    - Pre-allocated tensors for CUDA graph compatibility
    - Direct access to unified pool pointers for thop.attention

    Limitations (current implementation):
    - Does not support block reuse (AD manages page assignments)
    - Does not support host offloading

    Usage:
        # Before building the model with AD
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            enable_pt_cache_backend
        )
        enable_pt_cache_backend(True)

        # Then proceed with AD model building
        # ...

    Args:
        enable: Whether to enable PTCacheBackend (default: True)
    """
    if enable and not _HAS_PT_CACHE_BACKEND:
        ad_logger.warning(
            "PTCacheBackend is not available (missing TensorRT-LLM bindings). "
            "Falling back to SimpleCacheBackend."
        )
        return

    _trtllm_config.use_pt_cache_backend = enable
    ad_logger.info(f"[TRT-LLM] PTCacheBackend {'enabled' if enable else 'disabled'}")


def get_pt_cache_backend() -> Optional["PTCacheBackend"]:
    """Get the current PTCacheBackend instance if available.

    Returns:
        The PTCacheBackend instance, or None if not initialized or disabled.
    """
    return _trtllm_config.pt_cache_backend


def is_pt_cache_backend_enabled() -> bool:
    """Check if PTCacheBackend is enabled.

    Returns:
        True if PTCacheBackend is enabled and available.
    """
    return _trtllm_config.use_pt_cache_backend and _HAS_PT_CACHE_BACKEND


def reset_trtllm_attention_state() -> None:
    """Reset all TRT-LLM attention state.

    Call this before building a new model to ensure clean state.
    This resets:
    - Layer counter
    - Global state (per-layer states, workspace)
    - Configuration (page size, batch size, etc.)
    - PTCacheBackend instance
    """
    TrtllmAttention._reset_layer_counter()
    ad_logger.info("[TRT-LLM] Attention state reset")
