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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

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

            # Get pool pointers and mapping
            self._kv_cache_pool_pointers = self._kv_cache_manager.get_block_pool_pointers()
            self._kv_cache_pool_mapping = self._kv_cache_manager.get_layer_to_pool_mapping()

            ad_logger.info(
                f"[PTCacheBackend] Allocated pools: "
                f"num_pools={self._kv_cache_manager.num_pools}, "
                f"max_blocks_per_seq={self._kv_cache_manager.max_blocks_per_seq}, "
                f"tokens_per_block={self._kv_cache_manager.tokens_per_block}"
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
        # Block offsets: [2, batch, max_blocks] - same layout as SimpleCacheBackend
        # Dimension 0: K cache (index 0) and V cache (index 1) - both get same block indices
        self._kv_cache_block_offsets = torch.zeros(
            2, max_batch, max_blocks, dtype=torch.int32, device=device
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

        ad_logger.debug(
            f"[PTCacheBackend] Allocated metadata tensors: "
            f"block_offsets={self._kv_cache_block_offsets.shape}, "
            f"sequence_length={self._sequence_length.shape}"
        )

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
        """Get layer-to-pool mapping tensor."""
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
                skip_device_ops: If True, only update host tensors (for CUDA graph capture).
                    During capture, host tensor VALUES are baked into the graph,
                    while device tensor ADDRESSES are captured (allowing data updates).
            """
            # Fast integer extraction without full .tolist()
            num_prefill = int(batch_info_host[0])
            num_decode = int(batch_info_host[2])
            num_seq = num_prefill + num_decode

            if num_seq == 0:
                return

            # Check if we're capturing a CUDA graph
            is_capturing = torch.cuda.is_current_stream_capturing()

            # Compute input sequence lengths from cumulative sums
            input_seq_lens = (cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]).int()

            # Total KV lengths (including cached)
            seq_len_with_cache = seq_len_with_cache_host[:num_seq].int()
            past_kv_lens = seq_len_with_cache - input_seq_lens

            # CUDA GRAPH FIX: During capture, host tensor VALUES are baked into the graph.
            # The kernel uses these values for grid sizing and memory calculations.
            # If we bake in small values (warmup), the kernel will crash when replayed
            # with larger sequences.
            #
            # Fix: During capture, set host tensors to MAXIMUM values so the kernel
            # grid is sized for the worst case. During replay, the device tensors
            # have actual values and the kernel processes only valid data.
            if is_capturing:
                # Use max sequence length for all sequences during capture
                # This ensures kernel grid is sized for maximum sequence length
                max_seq = self._config.max_seq_len
                self._host_past_key_value_lengths[:num_seq].fill_(max_seq)
                self._host_context_lengths[:num_seq].fill_(max_seq)
                # All decode during capture (ensures kernel handles generate path)
                self._host_request_types[:num_seq].fill_(1)
                # Max total KV lens
                self._host_total_kv_lens[0] = 0  # No prefill during capture
                self._host_total_kv_lens[1] = max_seq * num_seq
                ad_logger.debug(
                    f"[PTCacheBackend] CUDA graph capture: setting host tensors to max_seq={max_seq}"
                )
            else:
                # Normal operation: fill host tensors with actual values
                self._host_past_key_value_lengths[:num_seq].copy_(past_kv_lens)
                self._host_context_lengths[:num_seq].copy_(input_seq_lens)

                # Request types: 0 = context (prefill), 1 = generation (decode)
                if num_prefill > 0:
                    self._host_request_types[:num_prefill].fill_(0)
                if num_decode > 0:
                    self._host_request_types[num_prefill:num_seq].fill_(1)

                # Total KV lens
                if num_prefill > 0:
                    self._host_total_kv_lens[0] = seq_len_with_cache[:num_prefill].sum()
                else:
                    self._host_total_kv_lens[0] = 0
                if num_decode > 0:
                    self._host_total_kv_lens[1] = seq_len_with_cache[num_prefill:num_seq].sum()
                else:
                    self._host_total_kv_lens[1] = 0

            # Device operations - skip during CUDA graph capture
            # Device tensor ADDRESSES are captured, so we can update data before replay
            if not skip_device_ops:
                # Fill device tensors - H2D copy
                self._sequence_length[:num_seq].copy_(
                    seq_len_with_cache.to(self._device), non_blocking=True
                )
                self._context_lengths[:num_seq].copy_(
                    input_seq_lens.to(self._device), non_blocking=True
                )

                # Fill block offsets (device tensor, vectorized)
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
        [2, batch_size, max_blocks_per_seq] format (same as SimpleCacheBackend).
        Dimension 0 is K/V cache index (both get same block indices).

        Uses fully vectorized operations - no Python loops.
        """
        if num_seq == 0:
            return

        # Get the relevant slice of block_offsets
        # Shape: [2, batch, max_blocks] - same as SimpleCacheBackend
        block_offsets = self._kv_cache_block_offsets
        max_batch = block_offsets.shape[1]
        max_blocks = block_offsets.shape[2]
        device = block_offsets.device

        # Bounds check: num_seq <= max_batch
        if num_seq > max_batch:
            ad_logger.error(f"[PTCacheBackend] num_seq ({num_seq}) > max_batch ({max_batch})")
            num_seq = max_batch

        # cu_num_pages_host is cumulative, e.g., [0, 2, 5, 7] for 3 sequences
        cu_pages = cu_num_pages_host[: num_seq + 1].long()
        total_pages = cu_pages[num_seq].item()

        if total_pages == 0:
            block_offsets[:, :num_seq, :].zero_()  # Shape: [2, num_seq, max_blocks]
            return

        # DEBUG: Track statistics for diagnosing 2k OSL issue
        if not hasattr(self, "_fill_call_count"):
            self._fill_call_count = 0
            self._last_logged_pages = 0
        self._fill_call_count += 1

        # Log every 100 calls or when total_pages changes significantly
        pages_per_seq_avg = total_pages / num_seq if num_seq > 0 else 0
        should_log = self._fill_call_count % 100 == 0 or total_pages > self._last_logged_pages + 100
        if should_log:
            # Get max cache_loc value for bounds checking debug
            cache_loc_slice = cache_loc[:total_pages]
            max_cache_loc = cache_loc_slice.max().item() if total_pages > 0 else 0
            # Also log host tensor values for CUDA graph debugging
            max_seq_len = (
                self._host_past_key_value_lengths[:num_seq].max().item() if num_seq > 0 else 0
            )
            total_kv_lens = self._host_total_kv_lens.tolist()
            ad_logger.info(
                f"[PTCacheBackend DEBUG] call={self._fill_call_count}, num_seq={num_seq}, "
                f"total_pages={total_pages}, pages_per_seq_avg={pages_per_seq_avg:.1f}, "
                f"max_blocks={max_blocks}, num_pages_pool={self._num_pages}, "
                f"max_cache_loc={max_cache_loc}, max_past_kv_len={max_seq_len}, "
                f"total_kv_lens={total_kv_lens}"
            )
            self._last_logged_pages = total_pages

        # Zero only the sequences we're updating
        # Shape: [2, num_seq, max_blocks]
        block_offsets[:, :num_seq, :].zero_()

        # VECTORIZED: Create sequence indices for each page
        # Using searchsorted: for cu_pages=[0,2,5,7], page positions 0-6
        # gives seq_idx=[0,0,1,1,1,2,2]
        page_positions = torch.arange(total_pages, dtype=torch.long)
        seq_idx = torch.searchsorted(cu_pages[1:], page_positions, right=True)

        # VECTORIZED: Create within-sequence page indices
        # page_idx[i] = page_positions[i] - cu_pages[seq_idx[i]]
        page_idx = page_positions - cu_pages[seq_idx]

        # Bounds check: page_idx must be < max_blocks
        max_page_idx = page_idx.max().item() if total_pages > 0 else 0
        if max_page_idx >= max_blocks:
            ad_logger.error(
                f"[PTCacheBackend] page_idx ({max_page_idx}) >= max_blocks ({max_blocks}), "
                f"call={self._fill_call_count}, total_pages={total_pages}, num_seq={num_seq}"
            )
            # Clamp to valid range
            page_idx = page_idx.clamp(max=max_blocks - 1)

        # Bounds check: seq_idx must be < num_seq
        max_seq_idx = seq_idx.max().item() if total_pages > 0 else 0
        if max_seq_idx >= num_seq:
            ad_logger.error(
                f"[PTCacheBackend] seq_idx ({max_seq_idx}) >= num_seq ({num_seq}), "
                f"call={self._fill_call_count}"
            )
            # Clamp to valid range
            seq_idx = seq_idx.clamp(max=num_seq - 1)

        # Bounds check: cache_loc values must be < num_pages
        cache_loc_slice = cache_loc[:total_pages]
        max_cache_loc = cache_loc_slice.max().item() if total_pages > 0 else 0
        if max_cache_loc >= self._num_pages:
            ad_logger.error(
                f"[PTCacheBackend] cache_loc ({max_cache_loc}) >= num_pages ({self._num_pages}), "
                f"call={self._fill_call_count}, total_pages={total_pages}"
            )

        # Get cache_loc values and move indices to device
        cache_loc_vals = cache_loc_slice.int().to(device)
        seq_idx_dev = seq_idx.to(device)
        page_idx_dev = page_idx.to(device)

        # Use advanced indexing to scatter values (no loop)
        # Both K cache (index 0) and V cache (index 1) get same block indices
        # Layout: [2, batch, max_blocks] - same as SimpleCacheBackend
        block_offsets[0, seq_idx_dev, page_idx_dev] = cache_loc_vals  # K cache
        block_offsets[1, seq_idx_dev, page_idx_dev] = cache_loc_vals  # V cache

        # DEBUG: Optional sync to catch errors early (enable via env var)
        import os

        if os.environ.get("TRTLLM_AD_DEBUG_SYNC", "0") == "1":
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                ad_logger.error(
                    f"[PTCacheBackend] CUDA error after scatter at call={self._fill_call_count}, "
                    f"num_seq={num_seq}, total_pages={total_pages}, "
                    f"max_page_idx={max_page_idx}, max_seq_idx={max_seq_idx}, "
                    f"max_cache_loc={max_cache_loc}: {e}"
                )
                raise

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
