from typing import TYPE_CHECKING, List, Optional

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .kv_cache_compression_executor import (BaseKVCacheCompressionExecutor,
                                            SparseAttentionExecutor)
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)
from .rocketkv import (RocketKV, RocketKVCacheManagerV2,
                       RocketKVTrtllmAttention, RocketKVVanillaAttention)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

    from .coordinator import KVCacheBehaviorCoordinator


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    """Legacy memory-layer dispatch: returns a KV-cache manager *class* whose
    instance owns sparse-aware physical storage (RocketKV / DSA / plain).

    Behavior-layer methods (where ``config.is_behavior_layer_method == True``)
    do NOT own a sparse-aware cache manager — they use the standard V2 manager
    and run their algorithm in a :class:`SparseAttentionExecutor` subclass.
    This function returns ``None`` for such configs (caller falls through to
    the standard V2 manager); top-level callers in ``_util.py`` short-circuit
    earlier via ``is_behavior_layer_method``, so this defensive None-return is
    mainly for tests and direct callers.
    """
    if sparse_attn_config.is_behavior_layer_method:
        return None
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "rocketkv":
        return RocketKVCacheManagerV2
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
) -> Optional[SparseAttentionExecutor]:
    """Behavior-layer factory: dispatches on ``config.algorithm`` (the same
    Pydantic ``Literal`` discriminator as the rest of the sparse stack) and
    returns a fully constructed :class:`SparseAttentionExecutor` subclass
    instance, or ``None`` when the configured algorithm is *not* a behavior-
    layer method (legacy RocketKV / DSA / skip_softmax stay on the legacy
    cache-manager path).

    Callers should also check ``sparse_attn_config.is_behavior_layer_method``
    to decide whether to invoke this factory at all; this function additionally
    guards against misuse by returning ``None`` for non-behavior-layer configs.

    Behavior-layer methods require :class:`KVCacheManagerV2` as the underlying
    KV-cache manager (see :class:`SparseAttentionExecutor` docstring). The
    legacy ``KVCacheManager`` lacks the page-table / block-read API the
    behavior layer depends on; passing it raises ``TypeError`` here so the
    misconfiguration
    is caught at LLM init rather than at the first eviction.
    """
    if sparse_attn_config.algorithm != "rocketkv":
        # Only methods with a SparseAttentionExecutor use the hook framework.
        # Legacy memory-layer methods (rocket / dsa / skip_softmax) have no
        # executor -> return None (caller skips coordinator creation).
        return None

    # Local import to avoid circular dependency at module load time.
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    if not isinstance(kv_cache_manager, KVCacheManagerV2):
        raise TypeError(
            f"Sparse attention algorithm '{sparse_attn_config.algorithm}' "
            f"requires KVCacheManagerV2 but received "
            f"{type(kv_cache_manager).__name__}. Enable V2 by setting "
            f"`KvCacheConfig.use_kv_cache_manager_v2=True` in your LLM config.")

    if sparse_attn_config.algorithm == "rocketkv":
        # The ``rocketkv`` algorithm (RocketKV executor in sparse/rocketkv.py).
        # Coexists with the legacy ``rocket`` algorithm, which routes through
        # get_sparse_attn_kv_cache_manager → RocketKVCacheManager.
        return RocketKV(
            kv_cache_manager=kv_cache_manager,
            page_size=sparse_attn_config.page_size,
            prompt_budget=sparse_attn_config.prompt_budget,
            kt_cache_dtype=sparse_attn_config.kt_cache_dtype,
            kt_tokens_per_block=sparse_attn_config.kt_tokens_per_block,
            window_size=sparse_attn_config.window_size,
            kernel_size=sparse_attn_config.kernel_size,
            topk=sparse_attn_config.topk,
            topr=sparse_attn_config.topr,
        )
    raise ValueError(f"Unsupported behavior-layer sparse attention algorithm: "
                     f"{sparse_attn_config.algorithm}")


def create_behavior_coordinator(
    sparse_attn_config: "Optional[SparseAttentionConfig]",
    kv_cache_manager: "KVCacheManagerV2",
) -> "Optional[KVCacheBehaviorCoordinator]":
    """Multi-manager factory: build a :class:`KVCacheBehaviorCoordinator`
    from the user-facing config(s).

    Currently only accepts a single ``sparse_attention_config``; when a
    multi-manager ``LlmArgs`` field (``behavior_managers: List[...]``)
    lands, this factory will accept a list of axis-discriminated configs
    instead. The wrapping path constructs at most one manager
    (sparse-attention executor) and returns a coordinator owning that single
    manager. Returns ``None`` if no manager is configured (no behavior-layer
    config or only legacy memory-layer configs).

    Note: PyExecutor does not yet call this factory — the
    :func:`create_sparse_attention_manager` path is still active. This
    factory exists so the PyExecutor wire can slot it in without further
    factory changes.
    """
    if sparse_attn_config is None:
        return None

    managers: List[BaseKVCacheCompressionExecutor] = []
    sparse_mgr = create_sparse_attention_manager(sparse_attn_config,
                                                 kv_cache_manager)
    if sparse_mgr is not None:
        managers.append(sparse_mgr)

    if not managers:
        return None

    # Local import to avoid circular dependency at module load time.
    from .coordinator import KVCacheBehaviorCoordinator
    return KVCacheBehaviorCoordinator(managers)


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    # ``rocketkv`` is a behavior-layer method but ALSO ships its own
    # attention shim (in sparse/rocketkv.py) — the executor produces a
    # sparse mask via on_*_attention hooks but the attention shim is what
    # actually consumes the mask in the kernel forward. So check rocketkv
    # BEFORE the generic behavior-layer short-circuit.
    if sparse_attn_config.algorithm == "rocketkv":
        return RocketKVVanillaAttention
    # Other behavior-layer methods use the base attention class without a
    # method-specific shim; their work happens out-of-band in a
    # SparseAttentionExecutor subclass invoked by PyExecutor. Top-level
    # callers in attention_backend.utils.get_attention_backend short-circuit
    # before reaching here; short-circuit defensively for tests / direct
    # callers too.
    if sparse_attn_config.is_behavior_layer_method:
        from ..vanilla import VanillaAttention
        return VanillaAttention
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    # ``rocketkv`` is a behavior-layer method but ALSO ships its own
    # attention shim (in sparse/rocketkv.py); check BEFORE the generic
    # behavior-layer short-circuit.
    if sparse_attn_config.algorithm == "rocketkv":
        return RocketKVTrtllmAttention
    # Other behavior-layer methods use the base attention class without a
    # method-specific shim. Top-level callers short-circuit before reaching
    # here; this defensive guard catches tests / direct callers.
    if sparse_attn_config.is_behavior_layer_method:
        return TrtllmAttention
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
    # Behavior-layer sparse methods use the base attention class. Defensive
    # short-circuit (mirrors get_trtllm / get_vanilla variants).
    if sparse_attn_config.is_behavior_layer_method:
        from ..flashinfer import FlashInferAttention
        return FlashInferAttention
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
