from typing import TYPE_CHECKING

# ``.utils`` imports the attention backend and KV-cache manager, which would
# create an import cycle through ``interface`` if pulled in eagerly here. So
# expose the accessors lazily (PEP 562); importing this package, or a submodule
# like ``sparse.skip_softmax``, then stays light.
if TYPE_CHECKING:
    from .utils import (get_flashinfer_sparse_attn_attention_backend,
                        get_sparse_attn_kv_cache_manager,
                        get_trtllm_sparse_attn_attention_backend,
                        get_vanilla_sparse_attn_attention_backend)

__all__ = [
    "get_sparse_attn_kv_cache_manager",
    "get_vanilla_sparse_attn_attention_backend",
    "get_trtllm_sparse_attn_attention_backend",
    "get_flashinfer_sparse_attn_attention_backend",
]


def __getattr__(name: str):
    if name in __all__:
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
