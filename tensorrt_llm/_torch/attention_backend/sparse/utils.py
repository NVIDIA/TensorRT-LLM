from typing import TYPE_CHECKING, Optional

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .kv_cache_compression_manager import BaseKVCacheCompressionManager
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


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


def create_sparse_attention_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Build the compression manager for the configured algorithm, or ``None``
    if the algorithm does not run on the compression framework (it is one of the
    legacy methods that owns its own cache manager, e.g. ``rocket`` / ``dsa``).

    This PR ships the framework only; no concrete algorithm is registered yet.
    A concrete method adds its dispatch branch here in its own PR, returning its
    :class:`BaseKVCacheCompressionManager` subclass. Compression managers need
    :class:`KVCacheManagerV2` (the older ``KVCacheManager`` lacks the page-table
    / block-read API they use).
    """
    return None


def create_compression_manager(
    sparse_attn_config: "Optional[SparseAttentionConfig]",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Build the single KV-cache compression manager to register with
    PyExecutor from the user-facing config, or ``None`` if no compression-
    framework method is configured (no config, or only legacy methods such as
    ``rocket`` / ``dsa`` / ``skip_softmax``).

    Returns one :class:`BaseKVCacheCompressionManager`. It inherits
    :class:`BaseResourceManager`, so PyExecutor's main loop auto-drives its
    lifecycle hooks once it is added to the resource-manager registry.
    Multi-method stacking (composing several axes) is future work and
    intentionally unsupported here.
    """
    if sparse_attn_config is None:
        return None
    return create_sparse_attention_manager(sparse_attn_config, kv_cache_manager)


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
