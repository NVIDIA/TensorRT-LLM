"""L2 behavior layer for KV cache compression methods.

This module defines:

- :class:`BaseKVCacheCompressionExecutor`: framework-level abstract base for
  every L2 behavior executor (sparse attention, KV storage
  transform-coding). Subclasses set an ``axis`` ClassVar and
  override any subset of the 6 lifecycle hooks (default no-op).

- :class:`SparseAttentionExecutor`: subclass for sparse attention methods
  (e.g. RocketKV); the only executor shipped today.

Architecture (3-layer stack):

- L0 attention kernel — attention math, optional kernel-fused quant decode.
- L1 ``KVCacheManagerV2`` — physical page management, dtype storage, 3-tier.
- L2 behavior (this module) — algorithm orchestration, lifecycle hooks.

Behavior executors hold the underlying ``KVCacheManagerV2`` as a tool and
never inherit from it. The behavior/memory split mirrors how speculative
decoding wires Eagle3 / MTPHiddenStatesManager into PyExecutor.

V2 specialization patterns:

- **Pattern 1 (default)** — Subclass uses default plain ``KVCacheManagerV2``;
  any per-method state (scoring buffer / compressed pool / TTL pool) is owned
  by the subclass instance directly. ``kv_cache_manager_class`` ClassVar stays
  ``None``. Used by methods whose per-request state is self-contained.

- **Pattern 2 (declarative BufferConfig)** — Subclass uses plain
  ``KVCacheManagerV2`` but PyExecutor factory adds extra ``BufferConfig`` per
  layer (new ``DataRole``) at V2 instantiation, for page-aligned auxiliary
  pools locked to KEY/VALUE lifecycle. ``kv_cache_manager_class`` ClassVar
  stays ``None``. Applies to: RocketKV (KT_CACHE landmark pool). Same mechanism as
  NVFP4 ``KEY_BLOCK_SCALE`` (already in production).

- **Pattern 3 (V2 subclass)** — Escape hatch when V2 behavioral
  specialization is needed (custom allocator, custom tier policy, etc.).
  Subclass declares ``kv_cache_manager_class = MyV2Subclass``; PyExecutor
  factory consults this ClassVar to instantiate the right V2 type.
  No shipped method uses this.

Hooks (6 total):

  - ``on_request_init`` / ``on_request_finish``
        Request lifecycle (entry / exit).
  - ``on_context_attention`` / ``on_context_end``
        Prefill phase: per attention layer, and once at phase boundary.
  - ``on_generation_attention`` / ``on_generation_step_end``
        Decode phase: per attention layer, and once at end of step.

A :class:`KVCacheBehaviorCoordinator` (see ``coordinator.py``) owns a list of
``BaseKVCacheCompressionExecutor`` instances and dispatches each hook to them
in deterministic order (see ``coordinator.HOOK_ORDER``).
"""

from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# ``(indices, offsets)`` tuple consumed by the attention kernel as an input-
# side sparse mask; ``None`` falls back to dense attention.
SparseAttentionIndices = Tuple[torch.Tensor, torch.Tensor]


class BaseKVCacheCompressionExecutor:
    """Framework-level base class for all L2 KV-cache compression executors.

    Subclasses must set ``axis`` ClassVar to one of:

    - ``"sparse"`` — sparse attention / per-token eviction
      (:class:`SparseAttentionExecutor`, shipped now).
    - ``"storage"`` — KV storage / transform-coding (no executor yet).

    All 6 hooks default to no-op; subclasses override what they need. A
    :class:`KVCacheBehaviorCoordinator` dispatches each hook to all
    registered executors in deterministic order.

    The behavior layer never inherits from any cache / resource manager
    because this layer decides *how* the physical KV is used, not *what*
    physical KV exists. Subclasses hold ``KVCacheManagerV2`` as a tool.

    V2 selection (see module docstring for full patterns):

    - Default (Patterns 1 + 2): ``kv_cache_manager_class`` stays ``None``;
      PyExecutor factory passes plain ``KVCacheManagerV2`` instance.
    - Pattern 3 escape hatch: subclass sets
      ``kv_cache_manager_class = MyV2Subclass``; factory instantiates that
      type instead. No current method uses this.
    """

    # ------------------------------------------------------------------ #
    # Class-level metadata                                                #
    # ------------------------------------------------------------------ #

    # Axis identifier — subclass MUST override.
    axis: ClassVar[str] = ""

    # Whether this executor class is compatible with prefix / block KV
    # cache reuse (radix-tree / APC). Default conservative ``False``;
    # subclasses opt in explicitly. The LLM init factory may enforce a mutex:
    # combining an executor with ``supports_kv_cache_reuse=False`` and
    # ``KvCacheConfig.enable_block_reuse=True`` raises a config error at init.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Pattern 3 (V2 subclass) escape hatch: subclass may declare a specific
    # ``KVCacheManagerV2`` subclass type via this ClassVar. PyExecutor factory
    # uses this to instantiate the right V2 type. ``None`` (default) means use
    # plain ``KVCacheManagerV2`` — adequate for Patterns 1 and 2 (i.e., for
    # essentially all known methods, including RocketKV since
    # the KT_CACHE auxiliary pool is added via Pattern 2 declarative
    # ``BufferConfig`` rather than V2 subclassing).
    kv_cache_manager_class: ClassVar[Optional[Type["KVCacheManagerV2"]]] = None

    def __init__(self, kv_cache_manager: "KVCacheManagerV2"):
        if not self.axis:
            raise NotImplementedError(
                f"{type(self).__name__} must set the 'axis' ClassVar "
                f"to one of: 'sparse', 'storage'."
            )
        # Pattern 3 type assertion: if subclass declared a V2 subclass
        # requirement, verify the injected instance matches.
        if self.kv_cache_manager_class is not None:
            assert isinstance(kv_cache_manager, self.kv_cache_manager_class), (
                f"{type(self).__name__} declared "
                f"kv_cache_manager_class={self.kv_cache_manager_class.__name__} "
                f"but received {type(kv_cache_manager).__name__}. "
                f"PyExecutor factory should consult the ClassVar to pick "
                f"the right V2 instance."
            )
        self.kv_cache_manager = kv_cache_manager

    # ------------------------------------------------------------------ #
    # Request lifecycle hooks                                            #
    # ------------------------------------------------------------------ #

    def on_request_init(self, request: "LlmRequest") -> None:
        """Per-request init hook.

        Override to allocate per-request accumulators (e.g. per-request
        scoring buffers).
        """
        pass

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Per-request finish / abort hook.

        Override to release per-request state allocated in
        ``on_request_init``. Underlying KV blocks are still freed by the
        ``KVCacheManagerV2``; subclasses must not free them here.
        """
        pass

    # ------------------------------------------------------------------ #
    # Context (prefill) phase hooks                                      #
    # ------------------------------------------------------------------ #

    def on_context_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Per-layer hook after every context-phase attention forward.

        Fires once per chunk per layer under chunked prefill; once per
        layer otherwise. ``attn_scores`` is populated only when the
        attention kernel instantiation exposes scores (compile-time
        template flag); ``None`` when scores are not materialized.

        Sparse-mask executors (e.g. RocketKV) return an
        ``(indices, offsets)`` tuple as an input-side sparse mask;
        physical-evict and storage executors return ``None``.
        The coordinator enforces single-source: at most one executor may
        return non-None per attention call.
        """
        return None

    def on_context_end(
        self,
        request: "LlmRequest",
        metadata: "AttentionMetadata",
    ) -> None:
        """Per-request hook fired once after the whole prompt has been
        consumed across all chunks and all layers (phase boundary).

        Override for one-shot prefill-end physical eviction (e.g. RocketKV
        Stage I-b).
        """
        pass

    # ------------------------------------------------------------------ #
    # Generation (decode) phase hooks                                    #
    # ------------------------------------------------------------------ #

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Per-layer hook after every generation-phase attention forward.

        Same single-source invariant as ``on_context_attention``: at most
        one executor may return non-None metadata. Used by sparse-mask
        executors (e.g. RocketKV Stage II HSA) to return query-aware masks;
        physical-evict and non-sparse executors return ``None``.
        """
        return None

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Cross-layer, cross-batch hook fired once per generation step
        after every layer's forward completes.

        Override for periodic or budget-triggered eviction, or runtime
        cleanup (e.g. RocketKV rewind). Storage executors
        may invalidate active compressed copies here if the cache shape
        changed (e.g., after a sparse evict).
        """
        pass

    # ------------------------------------------------------------------ #
    # Capability introspection                                            #
    # ------------------------------------------------------------------ #

    def implements(self, hook_name: str) -> bool:
        """Return ``True`` if this subclass actually overrides
        ``hook_name`` (treating the default no-op inherited from
        :class:`BaseKVCacheCompressionExecutor` as not implementing).

        Used by the coordinator to optionally skip-iterate executors that
        don't implement a particular hook (perf micro-optimization, off by
        default).
        """
        own_method = getattr(type(self), hook_name, None)
        if own_method is None:
            return False
        base_method = getattr(BaseKVCacheCompressionExecutor, hook_name, None)
        if base_method is None:
            return False
        # In Python 3, accessing a method via class returns the function directly
        # (no .__func__ needed). MRO lookup means an inherited (non-overridden) hook
        # returns the SAME function object as the base, so identity check suffices.
        return own_method is not base_method


class SparseAttentionExecutor(BaseKVCacheCompressionExecutor):
    """Convenience subclass for sparse-attention methods.

    Subclasses: RocketKV (``rocketkv.py``). The legacy ``rocket`` / ``dsa`` /
    ``skip_softmax`` algorithms follow the older plugin pattern (separate
    cache-manager + attention shim classes, in ``rocket.py`` / ``dsa.py``)
    and do NOT inherit from this base.
    """

    axis: ClassVar[str] = "sparse"

    # ------------------------------------------------------------------ #
    # Sparse-specific capability declarations (subclass overrides)        #
    # ------------------------------------------------------------------ #

    # ``True`` if this method physically deletes (compacts) tokens from the
    # KV cache. ``False`` if it only returns a sparse mask via
    # :class:`SparseAttentionIndices`, leaving cache contents unchanged
    # (e.g. RocketKV Stage II HSA).
    physically_evicts_kv: ClassVar[bool] = False
