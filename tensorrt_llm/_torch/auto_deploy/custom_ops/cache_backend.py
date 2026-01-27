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

"""Cache Backend Abstraction for Auto-Deploy.

This module provides an abstraction layer for KV cache management that allows
different cache backends to be used with various attention implementations.

The abstraction enables:
1. SimpleCacheBackend: Default AD behavior with per-layer cache allocation
2. PTCacheBackend: Integration with PT's unified KVCacheManager (C++ backed)

The key benefit of PTCacheBackend is efficient metadata preparation via
C++ `copy_batch_block_offsets()` instead of Python loops.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .attention_interface import SequenceInfo


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    A cache backend is responsible for:
    1. Allocating and managing KV cache memory
    2. Providing cache tensors to attention operations
    3. Preparing metadata for attention kernels
    4. (Optionally) Providing host-side metadata preparation functions

    This abstraction allows AD to use either its simple per-layer cache allocation
    or PT's unified KVCacheManager with block management features.
    """

    @abstractmethod
    def initialize(
        self,
        sequence_info: "SequenceInfo",
        device: torch.device,
    ) -> None:
        """Initialize the cache backend with sequence configuration.

        This is called once during model initialization to set up cache storage.

        Args:
            sequence_info: SequenceInfo containing max_seq_len, max_batch_size,
                          page_size, num_pages, etc.
            device: Target device for cache tensors.
        """
        pass

    @abstractmethod
    def get_cache(self, cache_name: str, layer_idx: int) -> torch.Tensor:
        """Get the cache tensor for a specific layer.

        Args:
            cache_name: Name of the cache (e.g., "k_cache", "v_cache")
            layer_idx: Layer index for per-layer caches.

        Returns:
            Cache tensor with appropriate shape and dtype.
        """
        pass

    @abstractmethod
    def resize_if_needed(self, num_pages: int) -> bool:
        """Resize cache storage if needed.

        Args:
            num_pages: Required number of pages.

        Returns:
            True if resize occurred, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def num_pages(self) -> int:
        """Current number of allocated pages."""
        pass

    def get_host_prepare_metadata_function(
        self,
    ) -> Optional[Callable[..., None]]:
        """Get function that performs host-side metadata preparation.

        The returned function will be called before each forward pass to
        prepare attention metadata. This is NOT captured in CUDA graphs.

        Returns:
            Callable that takes SequenceInfo tensor arguments, or None if
            no host-side preparation is needed.
        """
        return None

    def get_host_prepare_metadata_args(self) -> List[str]:
        """Get the argument names required by the host prepare function.

        Returns:
            List of SequenceInfo argument names that should be passed to
            the host prepare function.
        """
        return []

    @property
    def provides_pool_pointers(self) -> bool:
        """Whether this backend provides KV cache pool pointers.

        Pool pointers are required for TRT-LLM's thop.attention when using
        paged KV cache. SimpleCacheBackend does not provide these.

        Returns:
            True if pool pointers are available via get_pool_pointers().
        """
        return False

    def get_pool_pointers(self) -> Optional[torch.Tensor]:
        """Get KV cache pool pointers (for TRT-LLM attention).

        Returns:
            Tensor of shape [num_pools, 2] with pool pointers, or None.
        """
        return None

    def get_pool_mapping(self) -> Optional[torch.Tensor]:
        """Get layer-to-pool mapping tensor.

        Returns:
            Tensor mapping layer indices to pool indices, or None.
        """
        return None


class SimpleCacheBackend(CacheBackend):
    """Simple cache backend that allocates per-layer caches.

    This is the default cache backend used by AD for FlashInfer and Triton
    attention backends. It allocates separate K and V cache tensors per layer.

    Cache layout: [num_pages, page_size, num_kv_heads, head_dim]
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        num_layers: Optional[int] = None,
    ):
        """Initialize SimpleCacheBackend.

        Args:
            num_kv_heads: Number of KV heads per layer.
            head_dim: Dimension of each attention head.
            dtype: Data type for cache tensors.
            num_layers: Optional number of layers (for pre-allocation info).
        """
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._num_layers = num_layers

        # Initialized during initialize()
        self._num_pages = 0
        self._page_size = 0
        self._device: Optional[torch.device] = None

        # Per-layer caches: {layer_idx: {"k_cache": tensor, "v_cache": tensor}}
        self._caches: Dict[int, Dict[str, torch.Tensor]] = {}

    def initialize(
        self,
        sequence_info: "SequenceInfo",
        device: torch.device,
    ) -> None:
        """Initialize cache storage based on sequence configuration."""
        self._num_pages = sequence_info.num_pages
        self._page_size = sequence_info.page_size
        self._device = device

    def _allocate_layer_cache(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Allocate K and V caches for a single layer."""
        cache_shape = (
            self._num_pages,
            self._page_size,
            self._num_kv_heads,
            self._head_dim,
        )
        k_cache = torch.empty(cache_shape, dtype=self._dtype, device=self._device)
        v_cache = torch.empty(cache_shape, dtype=self._dtype, device=self._device)
        return {"k_cache": k_cache, "v_cache": v_cache}

    def get_cache(self, cache_name: str, layer_idx: int) -> torch.Tensor:
        """Get cache tensor for a layer, allocating if needed."""
        if layer_idx not in self._caches:
            self._caches[layer_idx] = self._allocate_layer_cache(layer_idx)
        return self._caches[layer_idx][cache_name]

    def resize_if_needed(self, num_pages: int) -> bool:
        """Resize all layer caches if more pages are needed."""
        if num_pages <= self._num_pages:
            return False

        old_num_pages = self._num_pages
        self._num_pages = num_pages

        # Resize existing caches
        for layer_idx, layer_caches in self._caches.items():
            for cache_name, old_cache in layer_caches.items():
                new_cache = torch.empty(
                    (num_pages, self._page_size, self._num_kv_heads, self._head_dim),
                    dtype=self._dtype,
                    device=self._device,
                )
                # Copy existing data
                new_cache[:old_num_pages] = old_cache
                layer_caches[cache_name] = new_cache

        return True

    @property
    def num_pages(self) -> int:
        return self._num_pages
