# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Base classes for KV-cache compression managers.

A compression manager runs a KV-cache reduction algorithm (for example sparse
attention or KV-storage compression) next to the normal cache manager, without
changing PyExecutor or the cache manager itself. It plugs into the runtime
through lifecycle hooks that fire at request init, end of prefill, each
generation step, before/after each attention layer, and request finish; every
hook defaults to no-op, so a subclass overrides only the ones it needs.

The manager holds a KVCacheManagerV2 as a tool and never inherits from it: the
cache manager owns the physical keys and values, and the compression manager
only decides how they are used.

It subclasses BaseResourceManager, so PyExecutor's loop already calls
prepare_resources / update_resources / free_resources each iteration; those
calls are forwarded to the hooks below.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


class BaseKVCacheCompressionManager(BaseResourceManager):
    """Framework-level base class for all KV-cache compression managers.

    Inherits :class:`BaseResourceManager` so PyExecutor's main loop
    auto-invokes ``prepare_resources`` / ``update_resources`` /
    ``free_resources`` each iteration without any PyExecutor code changes; the
    base implementations below translate those callbacks into the semantic
    lifecycle hooks. The two per-layer attention hooks
    (``on_context_attention`` / ``on_generation_attention``) fire directly from
    ``TrtllmAttention.forward`` via ``metadata.compression_manager``.

    Concrete compression methods subclass this directly.

    All 8 hooks default to no-op; subclasses override what they need. The
    manager never inherits from any cache manager because this layer decides
    *how* the physical KV is used, not *what* physical KV exists. Subclasses
    hold ``KVCacheManagerV2`` as a tool.
    """

    def __init__(self, kv_cache_manager: "KVCacheManagerV2"):
        self.kv_cache_manager = kv_cache_manager
        # Compression evicts/rewrites stored keys and values, so a shared prefix
        # block is no longer safe to reuse (same constraint as RocketKVCacheManager).
        if kv_cache_manager.enable_block_reuse:
            raise ValueError(
                f"{type(self).__name__} changes stored keys and values and cannot "
                f"run with KV-cache block reuse. Set "
                f"KvCacheConfig.enable_block_reuse to False."
            )

    # ================================================================== #
    # Semantic lifecycle hooks (8 total, temporal order).                #
    # Subclasses override what they need; all default to no-op.          #
    # ================================================================== #

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
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
        **kwargs,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer hook fired during each context-phase attention, from
        ``TrtllmAttention.forward`` via ``metadata.compression_manager`` (once
        per layer, or per chunk per layer under chunked prefill).

        Return the sparse-attention mask ``(indices, offsets)`` to apply for
        this layer, or ``None`` for dense attention. ``attn_scores`` is set only
        when the attention kernel exposes scores, else ``None``.
        """
        pass

    def on_context_attention_end(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_output: torch.Tensor,
        metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Per-layer hook fired after the context-phase attention output is
        computed. Read the output here -- e.g. stash ``q``/``k``/``attn_output``
        for an eviction computed later in :meth:`on_context_step_end`.
        """
        pass

    def on_context_step_end(
        self,
        request: "LlmRequest",
        metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Fired once per request, when its prefill finishes (its final
        chunk). Override for a one-shot prefill-end eviction.
        """
        pass

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
        **kwargs,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer hook fired during each generation-phase attention, from
        ``TrtllmAttention.forward`` via ``metadata.compression_manager``.

        Return the sparse-attention mask ``(indices, offsets)`` to apply for
        this layer, or ``None`` for dense attention.
        """
        pass

    def on_generation_attention_end(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_output: torch.Tensor,
        metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Per-layer hook fired after the generation-phase attention output is
        computed; the generation-phase analogue of
        :meth:`on_context_attention_end`.
        """
        pass

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Fired once per generation step, after every layer's forward
        completes. Override for periodic or budget-triggered eviction.
        """
        pass

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
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
        """Fire :meth:`on_context_step_end` once per request, on the iteration its
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
            self.on_context_step_end(req, attn_metadata)
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
