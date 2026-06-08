# Concrete executor subclasses are not re-exported at package level — they
# are accessed via the factory dispatchers ``create_sparse_attention_manager``
# and ``create_behavior_coordinator``, which select and instantiate the right
# subclass from the Pydantic discriminator in ``SparseAttentionConfig``. Users
# configure via Config objects in LLM(...); production code goes through the
# dispatcher, not direct class imports.
from .coordinator import KVCacheBehaviorCoordinator
from .kv_cache_compression_executor import (BaseKVCacheCompressionExecutor,
                                            SparseAttentionExecutor)
from .utils import (create_behavior_coordinator,
                    create_sparse_attention_manager,
                    get_flashinfer_sparse_attn_attention_backend,
                    get_sparse_attn_kv_cache_manager,
                    get_trtllm_sparse_attn_attention_backend,
                    get_vanilla_sparse_attn_attention_backend)

__all__ = [
    "BaseKVCacheCompressionExecutor",
    "SparseAttentionExecutor",
    "KVCacheBehaviorCoordinator",
    "create_sparse_attention_manager",
    "create_behavior_coordinator",
    "get_sparse_attn_kv_cache_manager",
    "get_vanilla_sparse_attn_attention_backend",
    "get_trtllm_sparse_attn_attention_backend",
    "get_flashinfer_sparse_attn_attention_backend",
]
