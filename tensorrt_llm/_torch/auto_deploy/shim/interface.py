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

        # Check if PTCacheBackend is available and handle resize
        pt_backend_resized = self._try_resize_pt_cache_backend(new_num_pages)
        if pt_backend_resized:
            # PTCacheBackend handled resize - regenerate cache views
            self._regenerate_cache_views()
            self.info.num_pages = new_num_pages
            return

        # Track if any resize was skipped (e.g., PTCacheBackend with non-resizable views)
        # resize_skipped = False
        actual_num_pages = new_num_pages

        for name, cache in self._caches.items():
            # We assume cache is a tensor of shape (max_batch_size, page_size, n_heads, head_dim)
            # TODO: cache resize should ideally be handled via a callback to the AttentionDescriptor
            # to avoid hard-coding any assumptions about the cache shape or its "pagedness"
            if "k_cache" in name or "v_cache" in name:
                current_shape = cache.shape
                new_shape = (new_num_pages, *current_shape[1:])

                # Check if tensor is resizable (not a view into another tensor's storage)
                # PTCacheBackend returns views into a unified pool which cannot be resized
                is_view = (
                    cache.storage_offset() != 0 or cache.data_ptr() != cache.storage().data_ptr()
                )
                if is_view:
                    # Skip resize for view tensors - the underlying pool manages its own size
                    # This happens when using PTCacheBackend with PT's unified KV pool
                    # Keep track of actual pages available
                    actual_num_pages = min(actual_num_pages, current_shape[0])
                    # resize_skipped = True
                    continue

                try:
                    cache.resize_(new_shape)
                except RuntimeError as e:
                    # Handle non-resizable storage (e.g., views from PTCacheBackend)
                    if "not resizable" in str(e):
                        # Skip resize for this cache - PT backend manages pool size
                        actual_num_pages = min(actual_num_pages, current_shape[0])
                        # resize_skipped = True
                        continue
                    raise

        # Update num_pages to actual value (may be less than requested if resize was skipped)
        self.info.num_pages = actual_num_pages

    def _try_resize_pt_cache_backend(self, new_num_pages: int) -> bool:
        """Try to resize using PTCacheBackend if available.

        Returns True if PTCacheBackend handled the resize, False otherwise.
        """
        try:
            from ..custom_ops.trtllm_attention import get_pt_cache_backend

            pt_backend = get_pt_cache_backend()
            if pt_backend is not None:
                return pt_backend.resize(new_num_pages)
        except ImportError:
            pass
        return False

    def _regenerate_cache_views(self):
        """Regenerate cache tensors after pool reallocation.

        This is called after PTCacheBackend reallocates its pool.
        It re-invokes cache initializers to get new views into the pool.
        """
        from ..utils.logger import ad_logger

        regenerated = 0
        # Only regenerate k_cache and v_cache (KV caches that are views)
        for name in list(self._caches.keys()):
            if "k_cache" in name or "v_cache" in name:
                if name in self._cache_initializers:
                    old_ptr = self._caches[name].data_ptr()
                    # Re-invoke initializer to get new view
                    self._caches[name] = self._cache_initializers[name](self.info)
                    new_ptr = self._caches[name].data_ptr()
                    regenerated += 1
                    if regenerated <= 2:  # Only log first 2
                        ad_logger.info(
                            f"[CachedSequenceInterface] Regenerated {name}: "
                            f"old_ptr=0x{old_ptr:x}, new_ptr=0x{new_ptr:x}, "
                            f"shape={self._caches[name].shape}"
                        )

        ad_logger.info(f"[CachedSequenceInterface] Regenerated {regenerated} cache views")


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]
