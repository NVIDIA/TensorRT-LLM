# Re-export names that were module-level in the old flat dsa.py, needed for
# mock.patch targets in tests (e.g., 'sparse.dsa.RotaryEmbedding').
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

from .backend import DSATrtllmAttention
from .cache_manager import DSACacheManager
from .indexer import (
    Indexer,
    IndexerPrefillChunkMetadata,
    _compute_slot_mappings,
    compute_cu_seqlen_kv_bounds_with_cache,
    rotate_activation,
    split_prefill_chunks,
    transform_local_topk_and_prepare_pool_view,
)
from .metadata import DSATrtllmAttentionMetadata, DSAtrtllmAttentionMetadata

__all__ = [
    "DSATrtllmAttention",
    "DSATrtllmAttentionMetadata",
    "DSAtrtllmAttentionMetadata",
    "DSACacheManager",
    "Indexer",
    "IndexerPrefillChunkMetadata",
    "RotaryEmbedding",
    "compute_cu_seqlen_kv_bounds_with_cache",
    "split_prefill_chunks",
    "transform_local_topk_and_prepare_pool_view",
    "_compute_slot_mappings",
    "rotate_activation",
]
