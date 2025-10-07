from typing import Callable, Dict, List, Optional, Tuple, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from ..custom_ops.attention_interface import BufferHandler, CacheHandler, DynamicShape, SequenceInfo


@final
class CachedSequenceInterface:
    """An interface responsible for maintaining information about sequences and their caches."""

    def __init__(
        self, sequence_info: SequenceInfo, device: Optional[DeviceLikeType] = None
    ) -> None:
        self.device = device or "cuda"
        self.info = sequence_info
        self._buffer_handlers: Dict[str, BufferHandler] = {}
        self._buffers: Dict[str, torch.Tensor] = {}

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return all the graph arguments owned by this interface."""
        return (*self.info.args, *self._buffers.values())

    @property
    def named_args(self) -> Dict[str, torch.Tensor]:
        """Return all the named arguments owned by this interface."""
        return {**self.info.named_args, **self._buffers}

    @property
    def all_future_arg_names(self) -> List[str]:
        """Return all the argument names owned by this interface including uninitialized caches."""
        return list(self.info.named_args.keys()) + list(self._buffer_handlers.keys())

    @property
    def dynamic_shapes(self) -> Tuple[DynamicShape, ...]:
        """Return the dynamic shapes of all graph arguments owned by this interface (all static)."""
        return tuple(self.named_dynamic_shapes.values())

    @property
    def named_dynamic_shapes(self) -> Dict[str, DynamicShape]:
        """Return the dynamic shapes of all graph arguments owned by this interface (all static)."""
        named_dynamic_shapes = self.info.named_dynamic_shapes
        named_dynamic_shapes.update({k: {} for k in self._buffers})
        return named_dynamic_shapes

    def to(self, *args, **kwargs) -> None:
        self.info.to(*args, **kwargs)
        if self._buffers:
            for cache in self._buffers.values():
                cache.to(*args, **kwargs)

    def add_buffer_or_cache(self, name: str, buffer_or_cache_handler: BufferHandler) -> None:
        """Add a buffer or cache handler to the cache interface to be used later."""
        self._buffer_handlers[name] = buffer_or_cache_handler

    def initialize_buffers(self) -> int:
        """Initialize buffers+caches using the init methods of the buffer/cache handlers."""
        assert not self._buffers, "Caches already initialized."
        self.info.to(self.device)
        self._buffers = {
            name: c_handler.init(self.info) for name, c_handler in self._buffer_handlers.items()
        }
        return len(self._buffers)

    def named_paged_caches(self) -> Dict[str, torch.Tensor]:
        """Return the subset of caches that are paged."""
        return {
            k: self._buffers[k]
            for k, handler in self._buffer_handlers.items()
            if isinstance(handler, CacheHandler) and handler.is_paged
        }

    @property
    def is_paged(self) -> bool:
        """Return if all registered caches are paged."""
        return all(
            handler.is_paged
            for handler in self._buffer_handlers.values()
            if isinstance(handler, CacheHandler)
        )

    def current_cache_size_bytes(self) -> int:
        """Calculate and return the total size of all caches in bytes."""
        return sum(
            cache.element_size() * cache.numel() for cache in self.named_paged_caches().values()
        )

    def resize_cache(self, new_num_pages: int):
        """Resize the cache to the new number of pages."""
        # TODO: We should do some sanity check on the new number of pages.
        self.info.num_pages = new_num_pages
        for name, cache_paged in self.named_paged_caches().items():
            self._buffer_handlers[name].resize(cache_paged, new_num_pages)


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
