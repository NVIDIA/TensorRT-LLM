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
    # Provide fallback names so that model files (e.g. Gemma3, Cohere2)
    # can be imported even when FlashInfer is not installed.  These names
    # are intentionally *not* added to __all__ because they are not real
    # implementations and should not be relied upon at runtime.
    FlashInferAttention = TrtllmAttention
    FlashInferAttentionMetadata = TrtllmAttentionMetadata
    StarAttention = TrtllmAttention
    StarAttentionMetadata = TrtllmAttentionMetadata
