"""L2 behavior layer for KV cache compression methods.

This module defines:

- :class:`BaseKVCacheCompressionManager`: framework-level abstract base for
  every L2 compression manager (sparse attention, KV storage transform-coding).
  Subclasses override any subset of the 8 lifecycle hooks (default no-op).

- :class:`SparseAttentionManager`: subclass for sparse-attention methods
  (e.g. RocketKV).

- :class:`KVCacheStorageManager`: subclass for KV-storage / transform-coding
  methods (e.g. KVTC) -- stub base; no shipped method yet.

Architecture (3-layer stack):

- L0 attention kernel — attention math, optional kernel-fused quant decode.
- L1 ``KVCacheManagerV2`` — physical page management, dtype storage, 3-tier.
- L2 behavior (this module) — algorithm orchestration, lifecycle hooks.

Compression managers hold the underlying ``KVCacheManagerV2`` as a tool and
never inherit from it. The behavior/memory split mirrors how speculative
decoding wires Eagle3 / MTPHiddenStatesManager into PyExecutor.

Resource-manager integration (Path A):

- :class:`BaseKVCacheCompressionManager` inherits :class:`BaseResourceManager`,
  so PyExecutor's main loop already invokes ``prepare_resources`` /
  ``update_resources`` / ``free_resources`` on every registered resource
  manager each iteration — no PyExecutor code changes. Those three callbacks
  translate into the semantic lifecycle hooks (request init / context end /
  generation-step end / request finish), gated on the same PyExecutor-provided
  signals the peer resource managers use (``is_first_context_chunk`` /
  ``context_requests_last_chunk``). The two per-layer *attention* hooks fire
  directly from ``TrtllmAttention.forward`` via ``metadata.compression_manager``.
- A single compression manager is registered today (no multi-method stacking).
  Composing several axes (sparse + storage + cross-request) is future work; it
  is intentionally NOT supported here to keep the resource-manager wiring flat.

V2 specialization patterns:

- **Pattern 1 (default)** — Subclass uses default plain ``KVCacheManagerV2``;
  any per-method state (scoring buffer / compressed pool / TTL pool) is owned
  by the subclass instance directly. ``kv_cache_manager_class`` ClassVar stays
  ``None``. Used by methods whose per-request state is self-contained.

- **Pattern 2 (declarative BufferConfig)** — Subclass uses plain
  ``KVCacheManagerV2`` but PyExecutor factory adds extra ``BufferConfig`` per
  layer (new ``DataRole``) at V2 instantiation, for page-aligned auxiliary
  pools locked to KEY/VALUE lifecycle. ``kv_cache_manager_class`` ClassVar
  stays ``None``. Same mechanism as NVFP4 ``KEY_BLOCK_SCALE``.

- **Pattern 3 (V2 subclass)** — Escape hatch when V2 behavioral
  specialization is needed (custom allocator, custom tier policy, etc.).
  Subclass declares ``kv_cache_manager_class = MyV2Subclass``; PyExecutor
  factory consults this ClassVar to instantiate the right V2 type.

Hooks (8 total, temporal order):

  - ``on_request_init``
        Per-request entry.
  - ``on_context_attention`` / ``on_context_attention_end``
        Prefill phase: per attention layer (mask-supplying / post-kernel).
  - ``on_context_end``
        Prefill phase boundary (whole prompt consumed).
  - ``on_generation_attention`` / ``on_generation_attention_end``
        Decode phase: per attention layer (mask-supplying / post-kernel).
  - ``on_generation_step_end``
        Decode phase: once per generation step.
  - ``on_request_finish``
        Per-request exit / abort.
"""

from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# ``(indices, offsets)`` tuple consumed by the attention kernel as an input-
# side sparse mask; ``None`` falls back to dense attention.
SparseAttentionIndices = Tuple[torch.Tensor, torch.Tensor]


class BaseKVCacheCompressionManager(BaseResourceManager):
    """Framework-level base class for all L2 KV-cache compression managers.

    Inherits :class:`BaseResourceManager` so PyExecutor's main loop
    auto-invokes ``prepare_resources`` / ``update_resources`` /
    ``free_resources`` each iteration without any PyExecutor code changes; the
    base implementations below translate those callbacks into the semantic
    lifecycle hooks. The two per-layer attention hooks
    (``on_context_attention`` / ``on_generation_attention``) fire directly from
    ``TrtllmAttention.forward`` via ``metadata.compression_manager``.

    Concrete managers subclass :class:`SparseAttentionManager` (sparse
    attention / per-token eviction) or :class:`KVCacheStorageManager` (KV
    storage / transform-coding).

    All 8 hooks default to no-op; subclasses override what they need. The
    manager never inherits from any cache manager because this layer decides
    *how* the physical KV is used, not *what* physical KV exists. Subclasses
    hold ``KVCacheManagerV2`` as a tool.

    V2 selection (see module docstring for full patterns):

    - Default (Patterns 1 + 2): ``kv_cache_manager_class`` stays ``None``;
      PyExecutor factory passes a plain ``KVCacheManagerV2`` instance.
    - Pattern 3 escape hatch: subclass sets
      ``kv_cache_manager_class = MyV2Subclass``; factory instantiates that type.
    """

    # ------------------------------------------------------------------ #
    # Class-level metadata                                                #
    # ------------------------------------------------------------------ #

    # Whether this manager class is compatible with prefix / block KV
    # cache reuse (radix-tree / APC). Default conservative ``False``;
    # subclasses opt in explicitly. The LLM init factory may enforce a mutex:
    # combining a manager with ``supports_kv_cache_reuse=False`` and
    # ``KvCacheConfig.enable_block_reuse=True`` raises a config error at init.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # Pattern 3 (V2 subclass) escape hatch: subclass may declare a specific
    # ``KVCacheManagerV2`` subclass type via this ClassVar. PyExecutor factory
    # uses this to instantiate the right V2 type. ``None`` (default) means use
    # plain ``KVCacheManagerV2`` — adequate for Patterns 1 and 2.
    kv_cache_manager_class: ClassVar[Optional[Type["KVCacheManagerV2"]]] = None

    def __init__(self, kv_cache_manager: "KVCacheManagerV2"):
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

    # ================================================================== #
    # Semantic lifecycle hooks (8 total, temporal order).                #
    # Subclasses override what they need; all default to no-op.          #
    # ================================================================== #

    def on_request_init(self, request: "LlmRequest") -> None:
        """Per-request init hook.

        Override to allocate per-request accumulators (e.g. per-request
        scoring buffers).
        """
        pass

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

        Sparse-mask managers (e.g. RocketKV) return an ``(indices, offsets)``
        tuple as an input-side sparse mask; physical-evict and storage
        managers return ``None``. Called directly from
        ``TrtllmAttention.forward`` via ``metadata.compression_manager``.
        """
        return None

    def on_context_attention_end(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_output: torch.Tensor,
        metadata: "AttentionMetadata",
    ) -> None:
        """Post-attention hook — per-layer, fired AFTER the context-phase attention output
        is computed (post-kernel), unlike :meth:`on_context_attention` which
        fires before/at the kernel to supply an input-side sparse mask.

        Side-effect only (returns ``None``): the sparse mask, if any, was
        already applied. Use this when the algorithm conceptually runs *after*
        attention — e.g. stash per-layer ``q``/``k``/``attn_output`` so a
        unified eviction can be computed in :meth:`on_context_end`,
        rather than per-layer during attention.
        """
        pass

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

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Per-layer hook after every generation-phase attention forward.

        Used by sparse-mask managers (e.g. RocketKV) to return query-aware
        masks; physical-evict and non-sparse managers return ``None``. Called
        directly from ``TrtllmAttention.forward`` via
        ``metadata.compression_manager``.
        """
        return None

    def on_generation_attention_end(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_output: torch.Tensor,
        metadata: "AttentionMetadata",
    ) -> None:
        """Post-attention hook — per-layer, fired AFTER the generation-phase attention
        output is computed (post-kernel). Side-effect only (returns ``None``);
        the decode-phase analogue of :meth:`on_context_attention_end`.
        """
        pass

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Cross-layer, cross-batch hook fired once per generation step
        after every layer's forward completes.

        Override for periodic or budget-triggered eviction, or runtime
        cleanup (e.g. RocketKV rewind). Storage managers may invalidate active
        compressed copies here if the cache shape changed (e.g., after a
        sparse evict).
        """
        pass

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Per-request finish / abort hook.

        Override to release per-request state allocated in
        ``on_request_init``. Underlying KV blocks are still freed by the
        ``KVCacheManagerV2``; subclasses must not free them here.
        """
        pass

    # ================================================================== #
    # BaseResourceManager interface — PyExecutor auto-invokes these each  #
    # iteration; they translate into the semantic lifecycle hooks above.  #
    # ================================================================== #

    def get_max_resource_count(self) -> int:
        """The compression manager does not own physical resources (the V2
        cache manager does). Returns 0 so PyExecutor's scheduler does not gate
        on us."""
        return 0

    def get_needed_resource_to_completion(self, request: "LlmRequest") -> int:
        """The compression manager does not own physical resources (the V2
        cache manager does). Returns 0 so PyExecutor's scheduler does not block
        on us."""
        return 0

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Fire :meth:`on_request_init` once per request, on its first prefill
        chunk -- the same ``is_first_context_chunk`` gate ``KVCacheManager``
        uses, so no manager-side dedup bookkeeping is needed.
        """
        for req in scheduled_batch.context_requests:
            if req.is_first_context_chunk:
                self.on_request_init(req)

    def update_resources(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ) -> None:
        """Fire :meth:`on_context_end` once per request, on the iteration its
        final prefill chunk runs, then :meth:`on_generation_step_end` once.

        Uses the scheduler's ``context_requests_last_chunk`` split (computed at
        schedule time from ``is_last_context_chunk``) rather than tracking
        request-state transitions: it is iteration-exact and immune to a
        short-output request going straight to ``GENERATION_TO_COMPLETE``
        (which, under the overlap scheduler, never passes through
        ``GENERATION_IN_PROGRESS``). Signature matches the other resource
        managers so PyExecutor passes ``attn_metadata`` /
        ``kv_cache_dtype_byte_size`` through transparently.
        """
        for req in scheduled_batch.context_requests_last_chunk:
            self.on_context_end(req, attn_metadata)
        self.on_generation_step_end(scheduled_batch, attn_metadata)

    def free_resources(self, request: "LlmRequest") -> None:
        """Fire :meth:`on_request_finish`."""
        self.on_request_finish(request)

    # ------------------------------------------------------------------ #
    # Capability introspection                                            #
    # ------------------------------------------------------------------ #

    def implements(self, hook_name: str) -> bool:
        """Return ``True`` if this subclass actually overrides ``hook_name``
        (treating the default no-op inherited from
        :class:`BaseKVCacheCompressionManager` as not implementing).

        Useful to skip per-layer attention-hook calls for managers that don't
        supply masks (perf micro-optimization).
        """
        own_method = getattr(type(self), hook_name, None)
        if own_method is None:
            return False
        base_method = getattr(BaseKVCacheCompressionManager, hook_name, None)
        if base_method is None:
            return False
        # In Python 3, accessing a method via class returns the function directly.
        # MRO lookup means an inherited (non-overridden) hook returns the SAME
        # function object as the base, so identity check suffices.
        return own_method is not base_method


class SparseAttentionManager(BaseKVCacheCompressionManager):
    """Convenience subclass for sparse-attention methods.

    Concrete sparse-attention managers subclass this and override the
    attention hooks (``on_*_attention``) to return an input-side sparse
    mask. The legacy ``rocket`` / ``dsa`` / ``skip_softmax`` algorithms
    follow the older plugin pattern (separate cache-manager + attention
    shim classes, in ``rocket.py`` / ``dsa.py``) and do NOT inherit from
    this base.
    """

    # ------------------------------------------------------------------ #
    # Sparse-specific capability declarations (subclass overrides)        #
    # ------------------------------------------------------------------ #

    # ``True`` if this method physically deletes (compacts) tokens from the
    # KV cache. ``False`` if it only returns a sparse mask via
    # :class:`SparseAttentionIndices`, leaving cache contents unchanged
    # (e.g. RocketKV).
    physically_evicts_kv: ClassVar[bool] = False


class KVCacheStorageManager(BaseKVCacheCompressionManager):
    """Convenience subclass for KV-storage / transform-coding methods
    (e.g. KVTC).

    Stub base — no shipped storage method yet. Concrete storage methods
    override the post-attention / phase-boundary hooks to compress the
    materialized KV in place (the attention hooks stay no-op since storage
    managers do not supply sparse masks).
    """
