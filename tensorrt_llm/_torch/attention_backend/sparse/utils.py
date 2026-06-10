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

# Every sparse-attention algorithm the framework knows about (mirrors the
# SparseAttentionConfig union). One not in this set that reaches a dispatch
# fall-through below was never registered.
_KNOWN_ALGORITHMS = frozenset({"rocket", "dsa", "skip_softmax"})


def _warn_if_unregistered(sparse_attn_config: "SparseAttentionConfig") -> None:
    """Warn if a configured algorithm is unregistered (added to the config but
    not wired into the dispatch here). A known algorithm that returns None from
    one dispatch because it is handled elsewhere does not warn."""
    if sparse_attn_config.algorithm not in _KNOWN_ALGORITHMS:
        logger.warning(
            f"Sparse-attention algorithm '{sparse_attn_config.algorithm}' is not "
            f"registered in attention_backend/sparse/utils.py; it will run "
            f"without its sparse path. Add it to the relevant dispatch.")


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
    _warn_if_unregistered(sparse_attn_config)
    return None


def create_kv_cache_compression_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Return the KV-cache compression manager for the configured algorithm,
    or ``None`` if the algorithm does not use the compression framework (e.g.
    legacy rocket / dsa)."""
    _warn_if_unregistered(sparse_attn_config)
    return None


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    _warn_if_unregistered(sparse_attn_config)
    return None


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    _warn_if_unregistered(sparse_attn_config)
    return None


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    _warn_if_unregistered(sparse_attn_config)
    return None
