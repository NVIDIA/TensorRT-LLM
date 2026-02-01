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

"""PT Cache Backend for Auto-Deploy TRT-LLM Attention.

This module provides a cache backend that integrates with TensorRT-LLM's
PyTorch executor (PT) KVCacheManager. This enables:

1. Efficient metadata preparation via C++ `copy_batch_block_offsets()`
2. Direct access to pool pointers for thop.attention
3. CUDA graph compatibility through pre-allocated tensors

Architecture:
-------------
The PTCacheBackend wraps the C++ KVCacheManagerCpp class which provides:
- Unified pool allocation across all layers
- Fast block offset population (no Python loops)
- Block reuse and host offloading capabilities (future)

Integration Flow:
-----------------
1. AD creates PTCacheBackend during model initialization
2. PTCacheBackend initializes KVCacheManager with AD's config
3. For each forward pass:
   a. AD calls host_prepare_metadata to update metadata tensors
   b. PTCacheBackend calls copy_batch_block_offsets (C++ fast path)
   c. Metadata tensors are passed to thop.attention
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.bindings.internal.batch_manager import CacheType

from ..utils.logger import ad_logger
from .cache_backend import CacheBackend

if TYPE_CHECKING:
    from .attention_interface import SequenceInfo

# Import KVCacheManager bindings
try:
    import tensorrt_llm.bindings.internal.batch_manager as bm

    KVCacheManagerCpp = bm.KVCacheManager
    _HAS_KV_CACHE_MANAGER = True
except ImportError:
    _HAS_KV_CACHE_MANAGER = False
    KVCacheManagerCpp = None


@dataclass
class PTCacheConfig:
    """Configuration for PT KVCacheManager integration.

    This contains all parameters needed to initialize the C++ KVCacheManager.
    """

    num_layers: int
    num_kv_heads_per_layer: List[int]  # Can vary per layer (e.g., GQA)
    head_dim: int
    tokens_per_block: int
    max_num_sequences: int
    max_seq_len: int
    num_pages: int = 0  # AD's calculated number of pages (0 = calculate from max)
    max_beam_width: int = 1
    dtype: torch.dtype = torch.float16
    enable_block_reuse: bool = False
    sink_token_length: int = 0

    def to_kv_cache_manager_kwargs(self) -> Dict:
        """Convert to kwargs for KVCacheManager constructor."""
        # Convert dtype to TRT-LLM DataType
        import tensorrt_llm.bindings as tllm_bindings

        dtype_map = {
            torch.float16: tllm_bindings.DataType.HALF,
            torch.bfloat16: tllm_bindings.DataType.BF16,
            torch.float32: tllm_bindings.DataType.FLOAT,
            torch.int8: tllm_bindings.DataType.INT8,
            torch.float8_e4m3fn: tllm_bindings.DataType.FP8,
        }
        trt_dtype = dtype_map.get(self.dtype, tllm_bindings.DataType.HALF)

        # Use AD's num_pages if provided, otherwise calculate from max
        if self.num_pages > 0:
            total_blocks = self.num_pages
        else:
            # Fallback: calculate theoretical maximum (may OOM!)
            max_blocks_per_seq = (
                self.max_seq_len + self.tokens_per_block - 1
            ) // self.tokens_per_block
            total_blocks = max_blocks_per_seq * self.max_num_sequences

        # Window size to layers mapping (single window for now)
        # Format: {window_size: (blocks_in_primary, blocks_in_secondary)}
        blocks_per_window = {self.max_seq_len: (total_blocks, 0)}

        return {
            "num_kv_heads_per_layer": self.num_kv_heads_per_layer,
            "size_per_head": self.head_dim,
            "tokens_per_block": self.tokens_per_block,
            "blocks_per_window": blocks_per_window,
            "max_num_sequences": self.max_num_sequences,
            "max_beam_width": self.max_beam_width,
            "max_attention_window_vec": [self.max_seq_len] * self.num_layers,
            "temp_attention_window_inputs": None,
            "dtype": trt_dtype,
            "sink_token_length": self.sink_token_length,
            "stream": True,  # Use CUDA stream
            "max_sequence_length": self.max_seq_len,
            "enable_block_reuse": self.enable_block_reuse,
            "onboard_blocks": True,
            "cache_type": CacheType.SELF,
        }


class PTCacheBackend(CacheBackend):
    """Cache backend using PT's KVCacheManager.

    This backend provides integration with TensorRT-LLM's C++ KVCacheManager,
    enabling efficient metadata preparation and direct pool access for
    thop.attention.

    Key Features:
    - Single unified pool across all layers (memory efficient)
    - C++ backed `copy_batch_block_offsets()` for fast metadata prep
    - Pre-allocated tensors for CUDA graph compatibility
    - Direct pool pointer access for thop.attention

    Limitations (current implementation):
    - Does not support block reuse (AD manages page assignments)
    - Does not support host offloading
    - Requires AD to provide cache_loc page assignments

    Usage:
        config = PTCacheConfig(num_layers=32, num_kv_heads_per_layer=[8]*32, ...)
        backend = PTCacheBackend(config)
        backend.initialize(sequence_info, device)

        # Get host prepare function for registration with SequenceInfo
        prep_fn = backend.get_host_prepare_metadata_function()
        prep_args = backend.get_host_prepare_metadata_args()
        sequence_info.register_host_prepare_for_attention_forward(prep_fn, prep_args)
    """

    def __init__(self, config: PTCacheConfig):
        """Initialize PTCacheBackend.

        Args:
            config: PTCacheConfig with KVCacheManager parameters.
        """
        if not _HAS_KV_CACHE_MANAGER:
            raise RuntimeError(
                "PTCacheBackend requires TensorRT-LLM bindings with KVCacheManager. "
                "Make sure tensorrt_llm is properly installed."
            )

        self._config = config
        self._kv_cache_manager: Optional["KVCacheManagerCpp"] = None
        self._device: Optional[torch.device] = None
        self._num_pages = 0
        self._initialized = False

        # Request ID mapping: AD sequence index -> PT request ID
        # PT's KVCacheManager uses request IDs for sequence tracking
        self._next_request_id = 0
        self._active_request_ids: Dict[int, int] = {}  # seq_idx -> request_id

        # Pre-allocated metadata tensors (for CUDA graph compatibility)
        self._kv_cache_block_offsets: Optional[torch.Tensor] = None
        self._sequence_length: Optional[torch.Tensor] = None
        self._context_lengths: Optional[torch.Tensor] = None
        self._host_past_key_value_lengths: Optional[torch.Tensor] = None
        self._host_context_lengths: Optional[torch.Tensor] = None
        self._host_request_types: Optional[torch.Tensor] = None
        self._host_total_kv_lens: Optional[torch.Tensor] = None

        # Cached pool pointers and mapping
        self._kv_cache_pool_pointers: Optional[torch.Tensor] = None
        self._kv_cache_pool_mapping: Optional[torch.Tensor] = None

        # Shared contiguous cache buffers for thop.attention
        # thop.attention expects contiguous K and V caches with specific block stride.
        # PTCacheBackend's interleaved pool has K/V interleaved per block, which
        # doesn't match the kernel's expected layout.
        #
        # MEMORY OPTIMIZATION: Use a single shared buffer pair for ALL layers.
        # Since only one layer is active at a time, we don't need per-layer buffers.
        # This avoids 2x memory multiplication (one buffer covers all layers sequentially).
        self._shared_contiguous_k_cache: Optional[torch.Tensor] = None
        self._shared_contiguous_v_cache: Optional[torch.Tensor] = None
        self._contiguous_pool_pointers: Optional[torch.Tensor] = None  # [1, 2]

    def initialize(
        self,
        sequence_info: "SequenceInfo",
        device: torch.device,
    ) -> None:
        """Initialize the KVCacheManager and allocate pools.

        This creates the C++ KVCacheManager, allocates GPU memory pools,
        and sets up pre-allocated metadata tensors.
        """
        if self._initialized:
            ad_logger.warning("PTCacheBackend already initialized, skipping")
            return

        self._device = device
        self._num_pages = sequence_info.num_pages

        # Verify config matches sequence_info
        if self._config.tokens_per_block != sequence_info.page_size:
            ad_logger.warning(
                f"PTCacheConfig tokens_per_block ({self._config.tokens_per_block}) "
                f"!= SequenceInfo page_size ({sequence_info.page_size}). "
                f"Using SequenceInfo value."
            )
            self._config.tokens_per_block = sequence_info.page_size

        if self._config.max_num_sequences != sequence_info.max_batch_size:
            ad_logger.warning(
                f"PTCacheConfig max_num_sequences ({self._config.max_num_sequences}) "
                f"!= SequenceInfo max_batch_size ({sequence_info.max_batch_size}). "
                f"Using SequenceInfo value."
            )
            self._config.max_num_sequences = sequence_info.max_batch_size

        # Create KVCacheManager
        kwargs = self._config.to_kv_cache_manager_kwargs()
        ad_logger.info(f"[PTCacheBackend] Creating KVCacheManager with: {kwargs}")

        try:
            self._kv_cache_manager = KVCacheManagerCpp(**kwargs)

            # Allocate pools
            self._kv_cache_manager.allocate_pools(False)  # useUvm=False

            # Get pool pointers and layer mapping from C++
            # NOTE: For unified interleaved pool, C++ returns [[base_ptr, 0]] where
            # v_ptr=0 indicates K and V are interleaved in the same pool.
            self._kv_cache_pool_pointers = self._kv_cache_manager.get_block_pool_pointers()
            self._kv_cache_pool_mapping = self._kv_cache_manager.get_layer_to_pool_mapping()

            ad_logger.info(
                f"[PTCacheBackend] Allocated pools: "
                f"num_pools={self._kv_cache_manager.num_pools}, "
                f"max_blocks_per_seq={self._kv_cache_manager.max_blocks_per_seq}, "
                f"tokens_per_block={self._kv_cache_manager.tokens_per_block}"
            )
            # Debug: Log pool pointer values
            ad_logger.info(
                f"[PTCacheBackend] Pool pointers shape={self._kv_cache_pool_pointers.shape}, "
                f"values={self._kv_cache_pool_pointers.tolist()}"
            )
            # Debug: Check pool data pointers for multiple layers
            for debug_layer in [0, 1, 2, 31]:
                pool_data_layer = self._kv_cache_manager.get_primary_pool_data(debug_layer)
                ad_logger.info(
                    f"[PTCacheBackend] Pool data layer {debug_layer}: shape={pool_data_layer.shape}, "
                    f"data_ptr={pool_data_layer.data_ptr()}"
                )

        except Exception as e:
            ad_logger.error(f"[PTCacheBackend] Failed to create KVCacheManager: {e}")
            raise

        # Allocate pre-allocated metadata tensors
        self._allocate_metadata_tensors(sequence_info)

        self._initialized = True

    def _allocate_metadata_tensors(self, sequence_info: "SequenceInfo") -> None:
        """Allocate pre-sized metadata tensors for CUDA graph compatibility."""
        max_batch = sequence_info.max_batch_size
        max_blocks = self._kv_cache_manager.max_blocks_per_seq

        # Device tensors
        device = self._device
        # Block offsets: [num_pools, batch, 2, max_blocks] - TRT-LLM expected layout
        # Dimension 0: pool index (we use 1 pool)
        # Dimension 1: sequence/batch index
        # Dimension 2: K cache (index 0) and V cache (index 1)
        # Dimension 3: block index within sequence
        self._kv_cache_block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks, dtype=torch.int32, device=device
        )
        self._sequence_length = torch.zeros(max_batch, dtype=torch.int32, device=device)
        self._context_lengths = torch.zeros(max_batch, dtype=torch.int32, device=device)

        # Host tensors (pinned for async H2D)
        self._host_past_key_value_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._host_context_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._host_request_types = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._host_total_kv_lens = torch.zeros(2, dtype=torch.int64, device="cpu", pin_memory=True)

        # Pool mapping for thop.attention: 1D tensor of pool indices per sequence
        # All sequences use pool 0 (single pool), so fill with zeros
        # NOTE: This is DIFFERENT from the C++ layer-to-pool mapping which is 2D [num_layers, 2]
        # thop.attention expects: [max_batch_size] with pool index per sequence
        self._host_kv_cache_pool_mapping_for_attention = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=True
        )

        ad_logger.debug(
            f"[PTCacheBackend] Allocated metadata tensors: "
            f"block_offsets={self._kv_cache_block_offsets.shape}, "
            f"sequence_length={self._sequence_length.shape}"
        )

        # Pre-allocate GPU work tensors for _fill_block_offsets_from_cache_loc
        # This avoids creating tensors and H2D transfers every iteration
        max_total_pages = sequence_info.num_pages
        self._gpu_page_positions = torch.arange(max_total_pages, dtype=torch.long, device=device)
        self._gpu_seq_idx = torch.empty(max_total_pages, dtype=torch.long, device=device)
        self._gpu_page_idx = torch.empty(max_total_pages, dtype=torch.long, device=device)
        self._gpu_cu_pages = torch.empty(max_batch + 1, dtype=torch.long, device=device)
        self._gpu_base_offset = torch.empty(max_total_pages, dtype=torch.int32, device=device)

        # Pre-allocate CPU work tensors for _prepare_trtllm_metadata
        self._cpu_input_seq_lens = torch.empty(max_batch, dtype=torch.int32, pin_memory=True)
        self._cpu_seq_len_with_cache = torch.empty(max_batch, dtype=torch.int32, pin_memory=True)
        self._cpu_past_kv_lens = torch.empty(max_batch, dtype=torch.int32, pin_memory=True)

    def get_cache(self, cache_name: str, layer_idx: int) -> torch.Tensor:
        """Get cache tensor view for a layer from the unified pool.

        Note: PT uses a unified pool, so we return a view into that pool
        for the specified layer. The cache layout is:
        [num_blocks, kv_factor, page_size * num_kv_heads * head_dim]

        We reshape to AD's expected format:
        [num_blocks, page_size, num_kv_heads, head_dim]
        """
        if not self._initialized:
            raise RuntimeError("PTCacheBackend not initialized")

        # Get the pool data for this layer and reshape to AD format
        pool_data = self._kv_cache_manager.get_primary_pool_data(layer_idx)

        # pool_data shape: [num_blocks, kv_factor, page_size * num_kv_heads * head_dim]
        # kv_factor = 2 (K and V)
        num_blocks = pool_data.shape[0]
        kv_factor = pool_data.shape[1]  # Should be 2

        num_kv_heads = self._config.num_kv_heads_per_layer[layer_idx]
        page_size = self._config.tokens_per_block
        head_dim = self._config.head_dim

        # Reshape to [num_blocks, kv_factor, page_size, num_kv_heads, head_dim]
        pool_reshaped = pool_data.view(num_blocks, kv_factor, page_size, num_kv_heads, head_dim)

        # Return K or V based on cache_name
        if cache_name == "k_cache":
            # K is at index 0, return [num_blocks, page_size, num_kv_heads, head_dim]
            return pool_reshaped[:, 0, :, :, :]
        elif cache_name == "v_cache":
            # V is at index 1
            return pool_reshaped[:, 1, :, :, :]
        else:
            raise ValueError(f"Unknown cache_name: {cache_name}")

    def get_contiguous_caches(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get shared contiguous K/V cache buffers.

        These buffers have a contiguous memory layout that matches what
        thop.attention expects (block stride = page_size * kv_dim).

        MEMORY OPTIMIZATION: A single shared buffer pair is used for ALL layers.
        Since only one layer is active at a time during forward pass, we don't
        need separate buffers per layer. This saves significant GPU memory.

        The buffers are allocated lazily on first use and reused across all
        layers, providing stable addresses for CUDA graph compatibility.

        Args:
            layer_idx: Layer index (used to get shape info, but buffer is shared)

        Returns:
            Tuple of (k_cache_contiguous, v_cache_contiguous) tensors
        """
        if not self._initialized:
            raise RuntimeError("PTCacheBackend not initialized")

        # Lazy allocation of shared contiguous buffers
        if self._shared_contiguous_k_cache is None:
            # Use layer 0 to get pool shape (all layers share same pool)
            pool_data = self._kv_cache_manager.get_primary_pool_data(0)
            num_blocks = pool_data.shape[0]
            # Use max num_kv_heads across all layers
            max_num_kv_heads = max(self._config.num_kv_heads_per_layer)
            page_size = self._config.tokens_per_block
            head_dim = self._config.head_dim
            dtype = pool_data.dtype

            # Allocate shared contiguous buffers with shape [num_blocks, max_kv_heads, page_size, head_dim]
            # This matches kernel's expected layout [heads, tokens, dim] per block
            self._shared_contiguous_k_cache = torch.empty(
                num_blocks, max_num_kv_heads, page_size, head_dim, dtype=dtype, device=self._device
            )
            self._shared_contiguous_v_cache = torch.empty(
                num_blocks, max_num_kv_heads, page_size, head_dim, dtype=dtype, device=self._device
            )
            ad_logger.info(
                f"[PTCacheBackend] Allocated SHARED contiguous buffers: "
                f"shape={self._shared_contiguous_k_cache.shape}, "
                f"size={self._shared_contiguous_k_cache.numel() * 2 * 2 / 1024 / 1024:.1f} MiB"
            )

        # Verify all layers have same num_kv_heads (required for shared buffer approach)
        # The kernel computes offsets based on num_kv_heads, so all layers must match
        num_kv_heads = self._config.num_kv_heads_per_layer[layer_idx]
        if num_kv_heads != max(self._config.num_kv_heads_per_layer):
            raise RuntimeError(
                f"Layer {layer_idx} has {num_kv_heads} kv_heads but max is "
                f"{max(self._config.num_kv_heads_per_layer)}. "
                f"PTCacheBackend's shared contiguous buffer requires uniform kv_heads."
            )

        return self._shared_contiguous_k_cache, self._shared_contiguous_v_cache

    def get_contiguous_pool_pointers(self, layer_idx: int) -> torch.Tensor:
        """Get pool pointers tensor pointing to contiguous K/V buffers.

        This returns a [1, 2] tensor with [k_ptr, v_ptr] that can be used
        with thop.attention. The addresses are stable across calls for
        CUDA graph compatibility.

        Args:
            layer_idx: Layer index

        Returns:
            Pool pointers tensor [1, 2] with [k_cache_ptr, v_cache_ptr]
        """
        k_cache, v_cache = self.get_contiguous_caches(layer_idx)

        # Allocate pool pointers tensor if needed (reused across calls)
        if self._contiguous_pool_pointers is None:
            self._contiguous_pool_pointers = torch.zeros(
                1, 2, dtype=torch.int64, device="cpu", pin_memory=True
            )

        # Update with current layer's buffer addresses
        self._contiguous_pool_pointers[0, 0] = k_cache.data_ptr()
        self._contiguous_pool_pointers[0, 1] = v_cache.data_ptr()

        return self._contiguous_pool_pointers

    def sync_to_contiguous(self, layer_idx: int) -> None:
        """Copy data from interleaved pool to contiguous buffers.

        Call this BEFORE thop.attention to ensure contiguous buffers
        have the latest cached K/V values.

        The kernel expects [heads, tokens, dim] layout per block, but the
        interleaved pool uses [tokens, heads, dim] layout, so we transpose.

        Args:
            layer_idx: Layer index
        """
        k_cont, v_cont = self.get_contiguous_caches(layer_idx)

        # Get interleaved pool views - shape [blocks, tokens, heads, dim]
        k_interleaved = self.get_cache("k_cache", layer_idx)
        v_interleaved = self.get_cache("v_cache", layer_idx)

        # Transpose from [blocks, tokens, heads, dim] -> [blocks, heads, tokens, dim]
        # then copy to contiguous buffer
        k_cont.copy_(k_interleaved.permute(0, 2, 1, 3))
        v_cont.copy_(v_interleaved.permute(0, 2, 1, 3))

    def sync_from_contiguous(self, layer_idx: int) -> None:
        """Copy data from contiguous buffers back to interleaved pool.

        Call this AFTER thop.attention to persist the kernel's cache
        writes to the main pool. This is needed because the kernel
        writes to the contiguous buffers, not the interleaved pool.

        The kernel uses [heads, tokens, dim] layout per block, but the
        interleaved pool uses [tokens, heads, dim] layout, so we transpose back.

        Args:
            layer_idx: Layer index
        """
        k_cont, v_cont = self.get_contiguous_caches(layer_idx)

        # Get interleaved pool views - shape [blocks, tokens, heads, dim]
        k_interleaved = self.get_cache("k_cache", layer_idx)
        v_interleaved = self.get_cache("v_cache", layer_idx)

        # Transpose from [blocks, heads, tokens, dim] -> [blocks, tokens, heads, dim]
        # and copy back to interleaved pool
        k_interleaved.copy_(k_cont.permute(0, 2, 1, 3))
        v_interleaved.copy_(v_cont.permute(0, 2, 1, 3))

    def get_interleaved_cache(self, layer_idx: int) -> torch.Tensor:
        """Get shared interleaved K/V cache buffer with alternating K/V blocks.

        The buffer has shape [total_kv_blocks, block_size] where total_kv_blocks = 2 * num_blocks.
        K blocks are at even indices (0, 2, 4, ...), V blocks at odd indices (1, 3, 5, ...).
        Each block has layout [heads, tokens, dim] as expected by the kernel.

        This is allocated lazily and shared across all layers.
        """
        if not self._initialized:
            raise RuntimeError("PTCacheBackend not initialized")

        # Lazy allocation
        if not hasattr(self, "_shared_interleaved_cache") or self._shared_interleaved_cache is None:
            pool_data = self._kv_cache_manager.get_primary_pool_data(0)
            num_blocks = pool_data.shape[0]
            max_num_kv_heads = max(self._config.num_kv_heads_per_layer)
            page_size = self._config.tokens_per_block
            head_dim = self._config.head_dim
            dtype = pool_data.dtype

            # Total blocks = 2 * num_blocks (alternating K/V)
            # Block size = heads * tokens * dim (kernel expected layout)
            block_size = max_num_kv_heads * page_size * head_dim
            self._shared_interleaved_cache = torch.empty(
                2 * num_blocks, block_size, dtype=dtype, device=self._device
            )
            ad_logger.info(
                f"[PTCacheBackend] Allocated SHARED interleaved buffer: "
                f"shape={self._shared_interleaved_cache.shape}, "
                f"size={self._shared_interleaved_cache.numel() * 2 / 1024 / 1024:.1f} MiB"
            )

        return self._shared_interleaved_cache

    def get_interleaved_pool_pointers(self, layer_idx: int) -> torch.Tensor:
        """Get pool pointers for interleaved buffer: [[base_ptr, 0]].

        v_ptr=0 signals interleaved mode to the kernel.
        """
        interleaved_cache = self.get_interleaved_cache(layer_idx)

        if (
            not hasattr(self, "_interleaved_pool_pointers")
            or self._interleaved_pool_pointers is None
        ):
            self._interleaved_pool_pointers = torch.zeros(
                1, 2, dtype=torch.int64, device="cpu", pin_memory=True
            )

        self._interleaved_pool_pointers[0, 0] = interleaved_cache.data_ptr()
        self._interleaved_pool_pointers[0, 1] = 0  # v_ptr=0 for interleaved mode

        return self._interleaved_pool_pointers

    def get_native_pool_pointers(self) -> torch.Tensor:
        """Get native pool pointers from C++ KVCacheManager.

        Returns the pool pointers in the format expected by the kernel:
        [[base_ptr, 0]] where 0 indicates K/V are interleaved in same pool.

        This allows the kernel to read/write directly to the C++ pool
        without any intermediate buffers or transposes.

        Returns:
            Pool pointers tensor on CPU with shape matching what C++ returned.
        """
        if not self._initialized:
            raise RuntimeError("PTCacheBackend not initialized")

        return self._kv_cache_pool_pointers

    def sync_to_interleaved(self, layer_idx: int) -> None:
        """Copy data from native pool to interleaved buffer before kernel.

        Copies K/V from native pool [blocks, tokens, heads, dim] to interleaved buffer
        with alternating K/V blocks and kernel layout [heads, tokens, dim] per block.
        """
        interleaved_cache = self.get_interleaved_cache(layer_idx)

        # Get native pool views - shape [blocks, tokens, heads, dim]
        k_native = self.get_cache("k_cache", layer_idx)
        v_native = self.get_cache("v_cache", layer_idx)

        num_blocks = k_native.shape[0]
        num_kv_heads = k_native.shape[2]
        page_size = k_native.shape[1]
        head_dim = k_native.shape[3]
        block_size = num_kv_heads * page_size * head_dim

        # Transpose and copy: [blocks, tokens, heads, dim] -> [blocks, heads, tokens, dim] -> [blocks, block_size]
        # K at even indices (0, 2, 4, ...), V at odd indices (1, 3, 5, ...)
        k_transposed = k_native.permute(0, 2, 1, 3).reshape(num_blocks, block_size)
        v_transposed = v_native.permute(0, 2, 1, 3).reshape(num_blocks, block_size)

        interleaved_cache[0::2, :block_size].copy_(k_transposed)
        interleaved_cache[1::2, :block_size].copy_(v_transposed)

    def sync_from_interleaved(self, layer_idx: int) -> None:
        """Copy data from interleaved buffer back to native pool after kernel.

        Copies K/V from interleaved buffer with kernel layout [heads, tokens, dim]
        back to native pool [blocks, tokens, heads, dim].
        """
        interleaved_cache = self.get_interleaved_cache(layer_idx)

        # Get native pool views - shape [blocks, tokens, heads, dim]
        k_native = self.get_cache("k_cache", layer_idx)
        v_native = self.get_cache("v_cache", layer_idx)

        num_blocks = k_native.shape[0]
        num_kv_heads = k_native.shape[2]
        page_size = k_native.shape[1]
        head_dim = k_native.shape[3]
        block_size = num_kv_heads * page_size * head_dim

        # K at even indices, V at odd indices
        k_from_kernel = interleaved_cache[0::2, :block_size].reshape(
            num_blocks, num_kv_heads, page_size, head_dim
        )
        v_from_kernel = interleaved_cache[1::2, :block_size].reshape(
            num_blocks, num_kv_heads, page_size, head_dim
        )

        # Transpose back: [blocks, heads, tokens, dim] -> [blocks, tokens, heads, dim]
        k_native.copy_(k_from_kernel.permute(0, 2, 1, 3))
        v_native.copy_(v_from_kernel.permute(0, 2, 1, 3))

    def resize(self, new_num_pages: int) -> bool:
        """Resize the KV cache pool by reallocating with new size.

        This creates a new KVCacheManager with the requested number of blocks
        and updates all internal references. The old pool is freed.

        Args:
            new_num_pages: New number of pages (blocks) to allocate

        Returns:
            True if resize was successful, False if skipped
        """
        if new_num_pages <= self._num_pages:
            ad_logger.info(f"[PTCacheBackend] Resize skipped: {new_num_pages} <= {self._num_pages}")
            return False

        if not self._initialized:
            ad_logger.warning("[PTCacheBackend] Cannot resize: not initialized")
            return False

        ad_logger.info(
            f"[PTCacheBackend] Resizing pool from {self._num_pages} to {new_num_pages} pages"
        )

        # Update config with new page count
        # This will be used in to_kv_cache_manager_kwargs() to compute blocks_per_window
        self._config.num_pages = new_num_pages

        # Free old pool (KVCacheManager destructor handles this)
        old_manager = self._kv_cache_manager

        try:
            # Create new KVCacheManager with updated config
            kwargs = self._config.to_kv_cache_manager_kwargs()
            ad_logger.info(f"[PTCacheBackend] Creating new KVCacheManager with: {kwargs}")

            self._kv_cache_manager = KVCacheManagerCpp(**kwargs)
            self._kv_cache_manager.allocate_pools(False)  # useUvm=False

            # Update pool pointers and mapping
            old_pool_ptr = (
                self._kv_cache_pool_pointers[0, 0].item()
                if self._kv_cache_pool_pointers is not None
                else 0
            )
            self._kv_cache_pool_pointers = self._kv_cache_manager.get_block_pool_pointers()
            self._kv_cache_pool_mapping = self._kv_cache_manager.get_layer_to_pool_mapping()
            new_pool_ptr = self._kv_cache_pool_pointers[0, 0].item()

            # Update internal page count
            self._num_pages = new_num_pages

            ad_logger.info(
                f"[PTCacheBackend] Resize complete: "
                f"num_pools={self._kv_cache_manager.num_pools}, "
                f"new_num_pages={new_num_pages}, "
                f"old_pool_ptr=0x{old_pool_ptr:x}, new_pool_ptr=0x{new_pool_ptr:x}"
            )

            # Synchronize all CUDA streams before freeing old memory
            torch.cuda.synchronize()

            # Keep old manager reference to prevent premature deallocation
            # This ensures any async operations using old memory complete first
            if not hasattr(self, "_old_managers"):
                self._old_managers = []
            self._old_managers.append(old_manager)

            # Try to free memory but don't delete old manager yet
            torch.cuda.empty_cache()

            # Clear shared contiguous buffers (will be reallocated with new size on next use)
            self._shared_contiguous_k_cache = None
            self._shared_contiguous_v_cache = None

            # Clear shared interleaved buffer (will be reallocated with new size on next use)
            if hasattr(self, "_shared_interleaved_cache"):
                self._shared_interleaved_cache = None
            if hasattr(self, "_interleaved_pool_pointers"):
                self._interleaved_pool_pointers = None

            # Resize GPU work tensors if needed
            if hasattr(self, "_gpu_page_positions") and new_num_pages > len(
                self._gpu_page_positions
            ):
                self._gpu_page_positions = torch.arange(
                    new_num_pages, dtype=torch.long, device=self._device
                )
                self._gpu_seq_idx = torch.empty(
                    new_num_pages, dtype=torch.long, device=self._device
                )
                self._gpu_page_idx = torch.empty(
                    new_num_pages, dtype=torch.long, device=self._device
                )
                self._gpu_base_offset = torch.empty(
                    new_num_pages, dtype=torch.int32, device=self._device
                )

            return True

        except Exception as e:
            ad_logger.error(f"[PTCacheBackend] Resize failed: {e}")
            # Restore old manager on failure
            self._kv_cache_manager = old_manager
            return False

    def resize_if_needed(self, num_pages: int) -> bool:
        """Resize if requested pages exceed current allocation."""
        if num_pages > self._num_pages:
            return self.resize(num_pages)
        return False

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def provides_pool_pointers(self) -> bool:
        return True

    def get_pool_pointers(self) -> Optional[torch.Tensor]:
        """Get KV cache pool pointers for thop.attention."""
        return self._kv_cache_pool_pointers

    def get_pool_mapping(self) -> Optional[torch.Tensor]:
        """Get layer-to-pool mapping tensor for thop.attention.

        For interleaved pool layout (v_ptr=0), the C++ kernel expects
        a 2D tensor [num_layers, 2] where each row is [pool_idx, layer_offset].

        Returns:
            2D tensor from C++ KVCacheManager's get_layer_to_pool_mapping().
        """
        return self._kv_cache_pool_mapping

    def get_host_prepare_metadata_function(self) -> Optional[Callable[..., None]]:
        """Get the host-side metadata preparation function.

        This function translates AD's metadata format to TRT-LLM's format.

        Returns:
            Callable that prepares TRT-LLM metadata from AD's SequenceInfo.
        """

        def _prepare_trtllm_metadata(
            batch_info_host: torch.Tensor,
            cu_seqlen_host: torch.Tensor,
            cu_num_pages_host: torch.Tensor,
            cache_loc: torch.Tensor,
            seq_len_with_cache_host: torch.Tensor,
            skip_device_ops: bool = False,
        ) -> None:
            """Prepare TRT-LLM metadata using PT's KVCacheManager.

            This is called before each forward pass to update metadata tensors.

            Args:
                batch_info_host: Batch info on CPU for control flow.
                cu_seqlen_host: Cumulative sequence lengths on CPU.
                cu_num_pages_host: Cumulative page counts on CPU.
                cache_loc: Cache locations (device tensor).
                seq_len_with_cache_host: Total sequence lengths on CPU.
                skip_device_ops: If True, only update host tensors (for CUDA graph capture).
            """
            # Fast integer extraction
            num_prefill = int(batch_info_host[0])
            num_decode = int(batch_info_host[2])
            num_seq = num_prefill + num_decode

            if num_seq == 0:
                return

            is_capturing = torch.cuda.is_current_stream_capturing()

            # Compute input sequence lengths using pre-allocated buffers
            # input_seq_lens = cu_seqlen[1:num_seq+1] - cu_seqlen[0:num_seq]
            input_seq_lens_host = self._cpu_input_seq_lens[:num_seq]
            torch.sub(
                cu_seqlen_host[1 : num_seq + 1],
                cu_seqlen_host[:num_seq],
                out=input_seq_lens_host,
            )

            # seq_len_with_cache_slice = seq_len_with_cache_host[:num_seq]
            seq_len_with_cache_slice = self._cpu_seq_len_with_cache[:num_seq]
            seq_len_with_cache_slice.copy_(seq_len_with_cache_host[:num_seq])

            # past_kv_lens = seq_len_with_cache - input_seq_lens
            past_kv_lens = self._cpu_past_kv_lens[:num_seq]
            torch.sub(seq_len_with_cache_slice, input_seq_lens_host, out=past_kv_lens)

            # CUDA GRAPH FIX: Set host tensors to max values during capture
            if is_capturing:
                max_seq = self._config.max_seq_len
                self._host_past_key_value_lengths[:num_seq].fill_(max_seq)
                self._host_context_lengths[:num_seq].fill_(max_seq)
                self._host_request_types[:num_seq].fill_(1)
                self._host_total_kv_lens[0] = 0
                self._host_total_kv_lens[1] = max_seq * num_seq
            else:
                # Normal operation: fill host tensors
                self._host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
                self._host_context_lengths[:num_seq].copy_(input_seq_lens_host)

                if num_prefill > 0:
                    self._host_request_types[:num_prefill].fill_(0)
                if num_decode > 0:
                    self._host_request_types[num_prefill:num_seq].fill_(1)

                if num_prefill > 0:
                    self._host_total_kv_lens[0] = seq_len_with_cache_slice[:num_prefill].sum()
                else:
                    self._host_total_kv_lens[0] = 0
                if num_decode > 0:
                    self._host_total_kv_lens[1] = seq_len_with_cache_slice[
                        num_prefill:num_seq
                    ].sum()
                else:
                    self._host_total_kv_lens[1] = 0

            # Device operations
            if not skip_device_ops:
                # H2D copy for device tensors (copy_ handles H2D directly, no intermediate)
                self._sequence_length[:num_seq].copy_(seq_len_with_cache_slice, non_blocking=True)
                self._context_lengths[:num_seq].copy_(input_seq_lens_host, non_blocking=True)

                # Fill block offsets (optimized GPU version)
                self._fill_block_offsets_from_cache_loc(cache_loc, cu_num_pages_host, num_seq)

        return _prepare_trtllm_metadata

    def _fill_block_offsets_from_cache_loc(
        self,
        cache_loc: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        num_seq: int,
    ) -> None:
        """Fill block offsets from AD's cache_loc (vectorized).

        This converts AD's flat cache_loc + cumulative pages to TRT-LLM's
        [2, batch_size, max_blocks_per_seq] format.
        """
        if num_seq == 0:
            return

        # Get the relevant slice of block_offsets
        block_offsets = self._kv_cache_block_offsets
        max_batch = block_offsets.shape[1]

        # Bounds check
        if num_seq > max_batch:
            num_seq = max_batch

        # cu_num_pages_host is cumulative, e.g., [0, 2, 5, 7] for 3 sequences
        cu_pages = cu_num_pages_host[: num_seq + 1].long()
        total_pages = cu_pages[num_seq].item()

        if total_pages == 0:
            block_offsets[:, :num_seq, :, :].zero_()
            return

        # Zero only the sequences we're updating
        block_offsets[:, :num_seq, :, :].zero_()

        # OPTIMIZED: Do all computation on GPU to avoid H2D transfers
        # 1. Copy cu_pages to GPU (small tensor, num_seq+1 elements)
        # NOTE: Must be blocking since we use it immediately in searchsorted
        self._gpu_cu_pages[: num_seq + 1].copy_(cu_pages)
        cu_pages_gpu = self._gpu_cu_pages[: num_seq + 1]

        # 2. page_positions is pre-allocated on GPU, just slice
        page_positions = self._gpu_page_positions[:total_pages]

        # 3. searchsorted on GPU into pre-allocated buffer
        torch.searchsorted(
            cu_pages_gpu[1:], page_positions, right=True, out=self._gpu_seq_idx[:total_pages]
        )
        seq_idx_dev = self._gpu_seq_idx[:total_pages]

        # 4. page_idx computed on GPU into pre-allocated buffer
        torch.sub(page_positions, cu_pages_gpu[seq_idx_dev], out=self._gpu_page_idx[:total_pages])
        page_idx_dev = self._gpu_page_idx[:total_pages]

        # cache_loc is already on GPU - compute base_offset into pre-allocated buffer
        # K and V use DIFFERENT block indices to index into the multi-layer pool
        num_layers = self._config.num_layers
        kv_factor = 2
        multiplier = num_layers * kv_factor
        # Use pre-allocated buffer: base_offset = cache_loc * multiplier
        base_offset = self._gpu_base_offset[:total_pages]
        torch.mul(cache_loc[:total_pages], multiplier, out=base_offset)

        # Pool 0, K cache (dim 2 = 0)
        block_offsets[0, seq_idx_dev, 0, page_idx_dev] = base_offset
        # Pool 0, V cache (dim 2 = 1)
        block_offsets[0, seq_idx_dev, 1, page_idx_dev] = base_offset + 1

    def get_host_prepare_metadata_args(self) -> List[str]:
        """Get argument names for the host prepare function."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages_host",
            "cache_loc",
            "seq_len_with_cache_host",
        ]

    # ==========================================================================
    # Properties for TRT-LLM attention metadata access
    # ==========================================================================

    @property
    def kv_cache_block_offsets(self) -> torch.Tensor:
        """Pre-allocated block offsets tensor."""
        return self._kv_cache_block_offsets

    @property
    def sequence_length(self) -> torch.Tensor:
        """Pre-allocated sequence length tensor (device)."""
        return self._sequence_length

    @property
    def context_lengths(self) -> torch.Tensor:
        """Pre-allocated context lengths tensor (device)."""
        return self._context_lengths

    @property
    def host_past_key_value_lengths(self) -> torch.Tensor:
        """Pre-allocated past KV lengths tensor (host, pinned)."""
        return self._host_past_key_value_lengths

    @property
    def host_context_lengths(self) -> torch.Tensor:
        """Pre-allocated context lengths tensor (host, pinned)."""
        return self._host_context_lengths

    @property
    def host_request_types(self) -> torch.Tensor:
        """Pre-allocated request types tensor (host, pinned)."""
        return self._host_request_types

    @property
    def host_total_kv_lens(self) -> torch.Tensor:
        """Pre-allocated total KV lens tensor (host, pinned)."""
        return self._host_total_kv_lens

    @property
    def tokens_per_block(self) -> int:
        """Number of tokens per cache block."""
        return self._config.tokens_per_block

    @property
    def max_blocks_per_seq(self) -> int:
        """Maximum blocks per sequence."""
        if self._kv_cache_manager is not None:
            return self._kv_cache_manager.max_blocks_per_seq
        return (
            self._config.max_seq_len + self._config.tokens_per_block - 1
        ) // self._config.tokens_per_block

    # ==========================================================================
    # Full PT Integration (Future Enhancement)
    # ==========================================================================

    def register_sequence(self, seq_idx: int, prompt_len: int, beam_width: int = 1) -> int:
        """Register a new sequence with the KVCacheManager.

        This is part of the full integration path where AD would use PT's
        block allocation. Currently not used as AD manages its own pages.

        Args:
            seq_idx: AD's sequence index
            prompt_len: Length of the prompt
            beam_width: Beam width for beam search

        Returns:
            PT request ID for this sequence
        """
        if not self._initialized:
            raise RuntimeError("PTCacheBackend not initialized")

        # Allocate a new request ID
        request_id = self._next_request_id
        self._next_request_id += 1

        # Register with KVCacheManager
        self._kv_cache_manager.add_sequence(request_id, prompt_len, beam_width)

        # Track mapping
        self._active_request_ids[seq_idx] = request_id

        return request_id

    def add_token(self, seq_idx: int) -> None:
        """Add a token to a sequence's KV cache allocation.

        Part of full integration path.
        """
        if seq_idx not in self._active_request_ids:
            raise ValueError(f"Sequence {seq_idx} not registered")

        request_id = self._active_request_ids[seq_idx]
        self._kv_cache_manager.add_token(request_id)

    def remove_sequence(self, seq_idx: int) -> None:
        """Remove a sequence from the KVCacheManager.

        Part of full integration path.
        """
        if seq_idx not in self._active_request_ids:
            return

        request_id = self._active_request_ids[seq_idx]
        self._kv_cache_manager.remove_sequence(request_id)
        del self._active_request_ids[seq_idx]

    def prepare_block_offsets_fast(self, seq_indices: List[int], beam_width: int = 1) -> None:
        """Use fast C++ path to populate block offsets.

        This is the optimal path when using full PT integration with
        registered sequences. Falls back to Python path otherwise.

        Args:
            seq_indices: List of AD sequence indices to prepare
            beam_width: Beam width
        """
        # Convert AD sequence indices to PT request IDs
        request_ids = []
        for idx in seq_indices:
            if idx in self._active_request_ids:
                request_ids.append(self._active_request_ids[idx])
            else:
                ad_logger.warning(f"Sequence {idx} not registered, skipping")

        if not request_ids:
            return

        # Use fast C++ path
        self._kv_cache_manager.copy_batch_block_offsets(
            self._kv_cache_block_offsets,
            request_ids,
            beam_width,
            offset=0,
        )
