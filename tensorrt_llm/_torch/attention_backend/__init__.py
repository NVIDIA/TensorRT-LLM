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
