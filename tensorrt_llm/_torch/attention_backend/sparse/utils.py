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

# Hardcoded (not derived from the config union) so adding a new algorithm
# without registering it in the dispatch below trips the warning.
_KNOWN_ALGORITHMS = frozenset({"rocket", "dsa", "skip_softmax"})


def _warn_if_unregistered(sparse_attn_config: "SparseAttentionConfig",
                          fallback: str) -> None:
    """Warn when a configured algorithm isn't registered in this dispatch;
    ``fallback`` is what is used instead."""
    if sparse_attn_config.algorithm not in _KNOWN_ALGORITHMS:
        logger.warning(
            f"Sparse-attention algorithm '{sparse_attn_config.algorithm}' is not "
            f"registered here; {fallback}.")


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
    _warn_if_unregistered(sparse_attn_config,
                          "using the standard KV-cache manager")
    return None


def create_kv_cache_compression_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Return the KV-cache compression manager for the configured algorithm,
    or ``None`` if the algorithm does not use the compression framework (e.g.
    legacy rocket / dsa)."""
    _warn_if_unregistered(sparse_attn_config,
                          "running without a compression manager")
    return None


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    _warn_if_unregistered(sparse_attn_config,
                          "using the base vanilla attention backend")
    return None


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    _warn_if_unregistered(sparse_attn_config,
                          "using the base TRTLLM attention backend")
    return None


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    _warn_if_unregistered(sparse_attn_config,
                          "using the base FlashInfer attention backend")
    return None
