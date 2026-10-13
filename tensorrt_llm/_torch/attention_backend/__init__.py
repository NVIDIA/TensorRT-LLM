from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, AttentionForwardArgs, AttentionMetadata
from .sparse import get_sparse_attn_kv_cache_manager
from .trtllm import AttentionInputType, TrtllmAttention, TrtllmAttentionMetadata
from .vanilla import VanillaAttention, VanillaAttentionMetadata

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "AttentionForwardArgs",
    "AttentionInputType",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "VanillaAttention",
    "VanillaAttentionMetadata",
    "get_sparse_attn_kv_cache_manager",
]

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer import FlashInferAttention, FlashInferAttentionMetadata
    from .star_flashinfer import StarAttention, StarAttentionMetadata
else:

    class FlashInferAttentionMetadata(AttentionMetadata):
        """Stub class for when FlashInfer is not available."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FlashInferAttentionMetadata requires FlashInfer backend, but FlashInfer is not available. "
                "Please install flashinfer or use a GPU with compute capability >= 7.5 (Turing or newer)."
            )

    class FlashInferAttention(AttentionBackend):
        """Stub class for when FlashInfer is not available."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FlashInferAttention requires FlashInfer backend, but FlashInfer is not available. "
                "Please install flashinfer or use a GPU with compute capability >= 7.5 (Turing or newer)."
            )

    class StarAttentionMetadata(AttentionMetadata):
        """Stub class for when FlashInfer is not available."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "StarAttentionMetadata requires FlashInfer backend, but FlashInfer is not available. "
                "Please install flashinfer or use a GPU with compute capability >= 7.5 (Turing or newer)."
            )

    class StarAttention(AttentionBackend):
        """Stub class for when FlashInfer is not available."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "StarAttention requires FlashInfer backend, but FlashInfer is not available. "
                "Please install flashinfer or use a GPU with compute capability >= 7.5 (Turing or newer)."
            )


__all__ += [
    "FlashInferAttention", "FlashInferAttentionMetadata", "StarAttention",
    "StarAttentionMetadata"
]
