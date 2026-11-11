# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft model's attention metadata of one-model speculative decoding."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

from ....memory_buffer_utils import Buffers
from ..extensions import MetadataExtension, init_field_owner

if TYPE_CHECKING:
    from ..interface import AttentionMetadata

# Creation settings every draft metadata gets, each given only when the
# extension that takes it is attached: a draft builds no decoder replay
# metadata and no mixed-precision high tables of DeepSeek-V4's MTP layers, and
# its mixed-precision staging copy has buffer names of its own.
_DRAFT_SETTINGS: Dict[str, Any] = {
    "builds_decoder_replay": False,
    "builds_mixed_precision_high_tables": False,
    "mixed_precision_staging_name": "draft_mixed",
}


class DraftMetadata(MetadataExtension):
    """The draft forwards' own metadata, ``md.draft``, and the draft loop.

    One-model speculative decoding runs the draft forwards on a metadata of
    their own: a copy of the target's (all of its extensions but this one and
    cross attention) over the draft KV cache manager, or the target's when the
    cache is shared. ``md.draft`` is an ordinary metadata: it holds no
    reference to the target. Everything that needs both sides is here.

    - Which state the draft takes from the target is fixed at creation. Over
      the target's KV cache, when every extension adopts the target's state,
      the draft adopts it at the start of each draft loop (inside the CUDA
      graph that records the loop) and is never prepared. Any other draft,
      such as one over a separate draft KV cache (whose core then stages the
      whole block table), is prepared in the target's ``prepare`` from the same
      batch, and takes the target's lengths and the state of the extensions
      that adopt it always (speculative decoding) at the loop start.
    - A draft loop runs inside ``with md.draft_loop() as draft``. It changes
      only the draft's lengths, through its core's length primitives, and
      calls ``md.refresh_draft()`` before each draft forward, which
      re-derives the draft's length-dependent state when they changed.
      ``refresh_draft`` raises outside a draft loop.

    A CUDA graph copy of the target builds a graph copy of the draft the same
    way, over a buffer pool every graph copy of this draft shares.
    """

    fields = frozenset(
        {
            "_draft_adopts",
            "draft",
            "draft_or_none",
            "one_model_draft",
            "draft_kv_cache_manager",
            "in_draft_loop",
            "draft_adopts_target_state",
        }
    )
    methods = frozenset({"draft_loop", "refresh_draft"})
    init_fields = {
        # Whether the draft forwards run in this engine and get a metadata of
        # their own (one-model speculative decoding).
        "one_model_draft": False,
        # The draft model's KV cache manager when the draft keeps a separate
        # KV cache; None shares the target's.
        "draft_kv_cache_manager": None,
    }
    # Its draft is a copy of the complete target metadata.
    constructed_last = True
    # The draft's CUDA graph copies share one pool, apart from the target's.
    shared = frozenset({"_draft_graph_pool"})
    # A draft's own copy of this extension is inert: it has no draft of its
    # own, so it never stops its draft from adopting the target's state.
    adopts_target_state = True
    draft_private = frozenset(
        {
            "one_model_draft",
            "draft_kv_cache_manager",
            "_draft",
            "_draft_adopts",
            "_in_draft_loop",
            "_draft_graph_pool",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self._draft: Optional["AttentionMetadata"] = None
        self._draft_adopts = False
        self._in_draft_loop = False
        # A copy of this metadata takes the source's pool, so every CUDA graph
        # copy's draft allocates from the same one.
        self._draft_graph_pool = self.shared_or("_draft_graph_pool", Buffers)
        if self.one_model_draft:
            self._draft = self._build_draft()
            self._draft_adopts = (
                not self._separate_draft_cache() and self._draft.adopts_target_state()
            )

    @property
    def draft(self) -> "AttentionMetadata":
        """The draft forwards' metadata."""
        if self._draft is None:
            raise RuntimeError("This metadata has no one-model draft metadata")
        return self._draft

    @property
    def draft_or_none(self) -> Optional["AttentionMetadata"]:
        """The draft forwards' metadata, or None without one-model drafting."""
        return self._draft

    @property
    def draft_adopts_target_state(self) -> bool:
        """Whether the draft takes the target's prepared state at each draft-loop
        start (shared KV cache, every extension adopting) instead of being
        prepared with the target."""
        return self._draft_adopts

    @property
    def in_draft_loop(self) -> bool:
        """Whether a ``draft_loop()`` is running."""
        return self._in_draft_loop

    def _separate_draft_cache(self) -> bool:
        manager = self.draft_kv_cache_manager
        return manager is not None and manager is not self.metadata.kv_cache_manager

    def _build_draft(self) -> "AttentionMetadata":
        """The draft metadata: a copy of the complete target metadata.

        The draft KV cache manager (or the target's when the cache is shared),
        the target's extensions but this one and cross attention, and buffers
        of its own: from the target's pool when eager, from the shared draft
        pool in a CUDA graph.
        """
        from .cross import CrossAttentionMetadata

        md = self.metadata
        pool = self._draft_graph_pool if md.is_cuda_graph else md.cuda_graph_buffers
        separate = self._separate_draft_cache()
        settings = {
            name: value
            for name, value in _DRAFT_SETTINGS.items()
            if (owner := init_field_owner(name)) is not None and md.find_ext(owner) is not None
        }
        return md.constructed_copy(
            share=False,
            without=(DraftMetadata, CrossAttentionMetadata),
            kv_cache_manager=self.draft_kv_cache_manager if separate else md.kv_cache_manager,
            cuda_graph_buffers=pool,
            # Draft steps advance the device KV lengths past the host
            # snapshot, so a draft over its own cache stages the whole table.
            full_block_table_width=separate or md.full_block_table_width,
            # Its lengths come from its first prepare: a CUDA graph target is
            # built before its graph-sized lengths are bound.
            seq_lens=None,
            seq_lens_kv=None,
            **settings,
        )

    def prepare(self) -> None:
        """Prepare a draft that derives its own state from the target's batch.

        A draft that adopts the target's state does so at the start of its
        draft loop instead. After a steady decode step of the target the draft
        takes its own steady step.
        """
        draft = self._draft
        if draft is None or self._draft_adopts:
            return
        from ..interface import BatchInputs

        md = self.metadata
        inputs = BatchInputs(
            seq_lens=md.seq_lens.clone() if md.seq_lens is not None else None,
            num_contexts=md.num_contexts,
            request_ids=md.request_ids,
            prompt_lens=md.prompt_lens,
            kv_cache_params=md.kv_cache_params,
            beam_width=md.beam_width,
            num_chunked_ctx_requests=md.num_chunked_ctx_requests,
            kv_lens_offsets=md.kv_lens_offsets,
            position_id_offsets=md.position_id_offsets,
        )
        kv_lens = md.steady_decode_kv_lens
        if kv_lens is not None:
            draft.prepare_steady_decode_step(inputs, kv_lens)
        else:
            draft.prepare(inputs)

    @contextmanager
    def draft_loop(self) -> Iterator["AttentionMetadata"]:
        """Run one draft loop on ``md.draft``, started from the target's state.

        An adopting draft takes the target's prepared state (``adopt_target_state``
        of each extension, core first); any other draft takes the target's lengths
        and counts (``copy_lengths_from``, which also brings the engine's
        attention-DP token counts) and the state of the extensions that adopt it
        always. Both run inside a captured graph, so every replay starts from the
        target. Yields the draft; the loop scope ends with the block.
        """
        draft = self.draft
        target = self.metadata
        core = draft.core()
        if self._draft_adopts:
            for extension in draft._attached_extensions():
                if isinstance(extension, DraftMetadata):
                    continue
                source = target.core() if extension.is_core else target.find_ext(type(extension))
                extension.adopt_target_state(source)
        else:
            core.copy_lengths_from(target.core())
            for extension in draft._attached_extensions():
                if type(extension).adopts_target_state_always:
                    extension.adopt_target_state(target.find_ext(type(extension)))
        core.clear_lengths_changed()
        self._in_draft_loop = True
        try:
            yield draft
        finally:
            self._in_draft_loop = False

    def refresh_draft(self) -> None:
        """Re-derive ``md.draft``'s length-dependent state after the draft loop
        changed its lengths in place; nothing when they did not change.

        The core re-derives its eagerly derived state
        (``resync_length_derived_state``) and every other extension refreshes
        from the draft's current lengths (``refresh_from_draft_lengths``), in
        dependency order. Raises outside ``draft_loop()``.
        """
        if not self._in_draft_loop:
            raise RuntimeError("refresh_draft() runs inside md.draft_loop() only")
        draft = self._draft
        core = draft.core()
        if not core.lengths_changed:
            return
        core.resync_length_derived_state()
        for extension in draft._attached_extensions():
            if extension is not core:
                extension.refresh_from_draft_lengths()
        core.clear_lengths_changed()
