from typing import TYPE_CHECKING, Optional

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.logger import logger

from .dsa import DSACacheManager, DSATrtllmAttention
from .kv_cache_compression_manager import BaseKVCacheCompressionManager
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

# Methods that own their own cache manager instead of the compression framework;
# they return None from the factory below by design.
_LEGACY_ALGORITHMS = frozenset({"rocket", "dsa", "skip_softmax"})


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    """Return the KV-cache manager *class* for a sparse method that owns its
    own sparse-aware physical storage (``rocket`` / ``dsa`` / ``skip_softmax``),
    or ``None`` for a compression-framework method (it runs its algorithm in a
    compression manager and uses the standard ``KVCacheManagerV2``).
    """
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    return None


def create_kv_cache_compression_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Return the KV-cache compression manager for the configured algorithm,
    or ``None`` if the algorithm does not use the compression framework (e.g.
    legacy rocket / dsa)."""
    # A framework method returns from its own branch above; reaching here with a
    # non-legacy algorithm means it was never registered.
    if sparse_attn_config.algorithm not in _LEGACY_ALGORITHMS:
        logger.warning(
            f"No compression manager is registered for sparse-attention "
            f"algorithm '{sparse_attn_config.algorithm}'; add a branch in "
            f"create_kv_cache_compression_manager. Running without compression."
        )
    return None


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    return None


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    return None


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    return None
