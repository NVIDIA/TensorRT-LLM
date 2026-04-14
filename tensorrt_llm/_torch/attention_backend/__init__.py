from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, AttentionMetadata
from .sparse import get_sparse_attn_kv_cache_manager
from .trtllm import AttentionInputType, TrtllmAttention, TrtllmAttentionMetadata
from .vanilla import VanillaAttention, VanillaAttentionMetadata

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
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
    __all__ += [
        "FlashInferAttention", "FlashInferAttentionMetadata", "StarAttention",
        "StarAttentionMetadata"
    ]
else:
    # Provide stub classes so that model files (e.g. Gemma3, Cohere2)
    # can be imported even when FlashInfer is not installed.  These are
    # intentionally *not* subclasses of TrtllmAttention/Metadata so that
    # isinstance() guards correctly return False, and instantiation gives
    # a clear error instead of a late AttributeError.

    class FlashInferAttention:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FlashInferAttention requires flashinfer to be installed. "
                "Please install flashinfer or use a different attention backend."
            )

    class FlashInferAttentionMetadata:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FlashInferAttentionMetadata requires flashinfer to be "
                "installed. Please install flashinfer or use a different "
                "attention backend.")

    class StarAttention:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "StarAttention requires flashinfer to be installed. "
                "Please install flashinfer or use a different attention backend."
            )

    class StarAttentionMetadata:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "StarAttentionMetadata requires flashinfer to be installed. "
                "Please install flashinfer or use a different attention backend."
            )
