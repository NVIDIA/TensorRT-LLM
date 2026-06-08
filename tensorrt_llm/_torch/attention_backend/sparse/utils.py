from typing import TYPE_CHECKING, Optional

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .kv_cache_compression_manager import (BaseKVCacheCompressionManager,
                                           SparseAttentionManager)
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    """Legacy memory-layer dispatch: returns a KV-cache manager *class* whose
    instance owns sparse-aware physical storage (e.g. legacy ``rocket`` / DSA /
    plain).

    Behavior-layer methods (where ``config.is_behavior_layer_method == True``)
    do NOT own a sparse-aware cache manager — they use the standard V2 manager
    and run their algorithm in a :class:`SparseAttentionManager` subclass.
    This function returns ``None`` for such configs (caller falls through to
    the standard V2 manager); top-level callers in ``_util.py`` short-circuit
    earlier via ``is_behavior_layer_method``, so this defensive None-return is
    mainly for tests and direct callers.
    """
    if sparse_attn_config.is_behavior_layer_method:
        return None
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def create_sparse_attention_manager(
    sparse_attn_config: "SparseAttentionConfig",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[SparseAttentionManager]:
    """Behavior-layer factory: returns a constructed
    :class:`SparseAttentionManager` subclass for the configured algorithm, or
    ``None`` when no behavior-layer (compression-manager) method applies — in
    which case the caller falls through to the legacy memory-layer path.

    This framework PR ships the abstraction only; no concrete behavior-layer
    algorithm is registered yet. A concrete sparse-attention method adds its
    dispatch branch here in its own PR, returning its
    :class:`SparseAttentionManager` subclass instance. Behavior-layer managers
    require :class:`KVCacheManagerV2` as the underlying KV-cache manager (the
    legacy ``KVCacheManager`` lacks the page-table / block-read API the
    behavior layer depends on).
    """
    return None


def create_compression_manager(
    sparse_attn_config: "Optional[SparseAttentionConfig]",
    kv_cache_manager: "KVCacheManagerV2",
) -> Optional[BaseKVCacheCompressionManager]:
    """Build the single KV-cache compression manager to register with
    PyExecutor from the user-facing config, or ``None`` if no behavior-layer
    method is configured (no config, or only legacy memory-layer configs).

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
    # Only legacy memory-layer methods reach here. Behavior-layer methods use
    # whatever backend the user selected: get_attention_backend nulls their
    # config (see attention_backend/utils.py) so they get the base class and
    # never reach this per-method dispatch.
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    # Only legacy memory-layer methods reach here (behavior-layer methods are
    # short-circuited to the user-selected backend in get_attention_backend).
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    # No legacy memory-layer sparse algorithm supports the flashinfer backend.
    # (Behavior-layer methods never reach here: get_attention_backend nulls
    # their config and returns FlashInferAttention directly.)
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
