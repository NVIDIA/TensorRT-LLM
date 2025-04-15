from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, final

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from ...pyexecutor.config import PyTorchConfig
from ..custom_ops.attention_interface import GetCacheCallable, SequenceInfo
from ..utils.logger import ad_logger


@final
class CachedSequenceInterface:
    """An interface responsible for maintaining information about sequences and their caches."""

    def __init__(
        self, sequence_info: SequenceInfo, device: Optional[DeviceLikeType] = None
    ) -> None:
        self.device = device or "cuda"
        self.info = sequence_info
        self._cache_initializers: Dict[str, GetCacheCallable] = {}
        self._caches: Dict[str, torch.Tensor] = {}

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        """Return all the graph arguments owned by this interface."""
        return (*self.info.args, *self._caches.values())

    @property
    def dynamic_shapes(self) -> Tuple[Dict[int, Any], ...]:
        """Return the dynamic shapes of all graph arguments owned by this interface (all static)."""
        return self.info.dynamic_shapes + ({},) * len(self._caches)

    def add_cache(self, name: str, get_cache: GetCacheCallable) -> None:
        """Add a cache initializer to the cache interface."""
        self._cache_initializers[name] = get_cache

    def initialize_caches(self) -> None:
        """Initialize caches using the cache initializers."""
        assert not self._caches, "Caches already initialized."
        ad_logger.info("Setting up caches + moving info args to device")
        self.info.to(self.device)
        self._caches = {
            name: get_cache(self.info) for name, get_cache in self._cache_initializers.items()
        }

    def current_cache_size_bytes(self) -> int:
        """Calculate and return the total size of all caches in bytes."""
        total_size = 0
        for name, cache in self._caches.items():
            # this hack is needed since _caches also contains global buffers such as freqs_cis.
            if "cache" in name:
                total_size += cache.element_size() * cache.numel()
        return total_size

    def resize_cache(self, new_num_pages: int):
        """Resize the cache to the new number of pages."""
        # TODO: We should do some sanity check on the new number of pages.
        self.info.num_pages = new_num_pages
        for name, cache in self._caches.items():
            # We assume cache is a tensor of shape (max_batch_size, page_size, n_heads, head_dim)
            if "cache" in name:
                current_shape = cache.shape
                new_shape = (new_num_pages, *current_shape[1:])
                cache.resize_(new_shape)


GetInferenceModel = Callable[[CachedSequenceInterface], nn.Module]


@dataclass
class AutoDeployConfig(PyTorchConfig):
    ### MODEL EXTRA KWARGS ###
    # Extra kwargs for the model config class to customize the model config. Those arguments will
    # take precedence over the default values or config values in the model config file in the HF
    # directory. Arguments are resolved in the following order:
    # 1. Default values in the model config class
    # 2. Values in the model config file in the HF directory
    # 3. Values in the model_kwargs
    # Note that that if the kwarg does not exist in the model config class, it will be ignored.
    # An example model config class can be found [here](https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/models/llama/configuration_llama.py#L26).
    model_kwargs: Dict = field(
        default_factory=lambda: {
            "use_cache": False,  # to avoid using built-in cache
        }
    )

    # attention backend to choose from
    attn_backend: str = "TritonWithFlattenedInputs"
    mla_backend: str = "MultiHeadLatentAttention"

    # check if we should skip loading weights
    skip_loading_weights: bool = False

    def __post_init__(self):
        super().__post_init__()

        # we don't want to loose the default values for model_kwargs unless explicitly set by the
        # user. They are not preserved by the standard initialization process since they whole dict
        # gets replaced by the user provided one. We don't want that though.
        f_default = self.__dataclass_fields__["model_kwargs"].default_factory()
        setattr(self, "model_kwargs", {**f_default, **getattr(self, "model_kwargs")})
