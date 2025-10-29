from typing import Callable, Dict, List, Optional, Tuple, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from ..custom_ops.attention_interface import GetCacheCallable, SequenceInfo


@final
class CachedSequenceInterface:
    """An interface responsible for maintaining information about sequences and their caches."""

    def __init__(
        self, sequence_info: SequenceInfo, device: Optional[DeviceLikeType] = None
    ) -> None:
        # TODO (lucaslie): this is somewhat circular/confusing. Here `device` denotes the desired
        # device and not the actual device unlike, e.g., in SequenceInfo. We rely on the attribute
        # here to read the desired device across the inference optimizer pipeline. We should ideally
        # think about a better way to handle this,
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/8371
        self.device = device or "cuda"
        self.info = sequence_info
        self._cache_initializers: Dict[str, GetCacheCallable] = {}
        self._caches: Dict[str, torch.Tensor] = {}

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return all the graph arguments owned by this interface."""
        return (*self.info.args, *self._caches.values())

    @property
    def named_args(self) -> Dict[str, torch.Tensor]:
        """Return all the named arguments owned by this interface."""
        return {**self.info.named_args, **self._caches}

    @property
    def all_future_arg_names(self) -> List[str]:
        """Return all the argument names owned by this interface including uninitialized caches."""
        return list(self.info.named_args.keys()) + list(self._cache_initializers.keys())

    def to(self, *args, **kwargs) -> None:
        self.info.to(*args, **kwargs)
        if self._caches:
            for cache in self._caches.values():
                cache.to(*args, **kwargs)

    def add_cache(self, name: str, get_cache: GetCacheCallable) -> None:
        """Add a cache initializer to the cache interface."""
        self._cache_initializers[name] = get_cache

    def initialize_caches(self) -> int:
        """Initialize caches using the cache initializers."""
        assert not self._caches, "Caches already initialized."
        self.info.to(self.device)
        self._caches = {
            name: get_cache(self.info) for name, get_cache in self._cache_initializers.items()
        }
        return len(self._caches)

    def current_cache_size_bytes(self) -> int:
        """Calculate and return the total size of all caches in bytes."""
        total_size = 0
        for name, cache in self._caches.items():
            # this hack is needed since _caches also contains global buffers such as freqs_cis.
            if "cache" in name:
                total_size += cache.element_size() * cache.numel()
        return total_size

    def current_kv_cache_size_bytes(self) -> int:
        """Return size in bytes of KV caches only (k_cache_*, v_cache_*).

        Excludes SSM/conv/etc. which do not scale with num_pages.
        """
        total_size = 0
        for name, cache in self._caches.items():
            if name.startswith("k_cache_") or name.startswith("v_cache_"):
                total_size += cache.element_size() * cache.numel()
        return total_size

    def resize_cache(self, new_num_pages: int):
        """Resize the cache to the new number of pages."""
        # TODO: We should do some sanity check on the new number of pages.
        self.info.num_pages = new_num_pages
        for name, cache in self._caches.items():
            # We assume cache is a tensor of shape (max_batch_size, page_size, n_heads, head_dim)
            # TODO: cache resize should ideally be handled via a callback to the AttentionDescriptor
            # to avoid hard-coding any assumptions about the cache shape or its "pagedness"
            if "k_cache" in name or "v_cache" in name:
                current_shape = cache.shape
                new_shape = (new_num_pages, *current_shape[1:])
                cache.resize_(new_shape)


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
