# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba metadata of the Gated-DeltaNet layers.

The GDN models (Qwen3-Next, Qwen3.5, Qwen4-Exp) and their cache managers name
``GDNMambaMetadata`` as ``mamba_metadata_cls``.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.modules.mamba.replay_metadata import (
    REPLAY_WORK_ITEM_WIDTH,
    ReplaySlotIndices,
    build_replay_work_items,
    snapshot_gdn_mtp_replay_work_items,
)
from tensorrt_llm._utils import prefer_pinned

from .flashinfer_chunk import invalidate_int32_cu_seqlens_cache
from .inforward_snapshot import InforwardSnapshotEntry

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import GdnIterationContext


class InforwardSnapshotSplit(NamedTuple):
    """Index tensors of one step's in-forward Mamba snapshot split.

    See ``GDNMambaMetadata._prepare_inforward_split``. ``k`` requests are split;
    ``num_tail_tokens`` tokens lie in their tails.
    """

    k: int
    num_tail_tokens: int
    cu: torch.Tensor
    gather: torch.Tensor
    live: torch.Tensor
    main_rows: torch.Tensor
    split_rows: torch.Tensor
    stage_rows: torch.Tensor
    tail_tok: torch.Tensor
    cu_tail: torch.Tensor
    tail_slots: torch.Tensor
    conv_win: torch.Tensor


# Fields of InforwardSnapshotSplit, in upload order.
_INFORWARD_SPLIT_FIELDS = (
    "cu",
    "gather",
    "live",
    "main_rows",
    "split_rows",
    "stage_rows",
    "tail_tok",
    "cu_tail",
    "tail_slots",
    "conv_win",
)


class GDNMambaMetadata(Mamba2Metadata):
    """Mamba metadata plus what the GDN mixer reads besides the chunked-scan layout.

    That is the per-iteration ``GdnIterationContext``, the host gate of the
    prefill state reset, the decode requests' slots in an aligned buffer, this
    step's in-forward snapshot split, and each request's replay slot with the
    replay work items of the partitioned cached replay.
    """

    # Warmup-only knob, set via ``inforward_split_for_warmup``: ``prepare()``
    # plans an in-forward snapshot split for the context requests, so warmup
    # captures the CUDA graphs of the split GDN prefill core.
    _warmup_inforward_split: bool = False

    @classmethod
    @contextlib.contextmanager
    def inforward_split_for_warmup(cls):
        # Set on this class whichever subclass the warmup resolved: prepare()
        # reads it here.
        prev = GDNMambaMetadata._warmup_inforward_split
        GDNMambaMetadata._warmup_inforward_split = True
        try:
            yield
        finally:
            GDNMambaMetadata._warmup_inforward_split = prev

    def __init__(
        self,
        max_batch_size: int,
        chunk_size: int,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        # Host gate for the GDN state reset (see gdn_mixer._reset_prefill_states):
        # True when a context request of this iteration starts without a
        # recurrent state; state_reset_done flips once the first GDN layer has
        # cleared those slots for every layer.
        self.prefill_needs_state_reset = True
        self.state_reset_done = False
        # The decode requests' slots of a mixed iteration in their own buffer:
        # the FlashInfer decode kernel wants an aligned base pointer, which the
        # tail slice ``state_indices[num_contexts:]`` only has when the number
        # of context requests is a multiple of 4 (see _mixed_decode_recurrent).
        self.state_indices_decode = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        # Per-iteration GdnIterationContext (Qwen3NextGatedDeltaNet.forward_core):
        # the first GDN layer of an iteration builds it once and the other
        # layers reuse it. Cleared by prepare() and prepare_steady_gen_step().
        self.gdn_iteration_context: GdnIterationContext | None = None

        # In-forward Mamba snapshot split of this step (None: no split), and
        # the buffer its index tensors live in at fixed field offsets, so they
        # keep their addresses across steps (sized on the first split from the
        # cache manager's staging capacity).
        self.inforward_split: InforwardSnapshotSplit | None = None
        self._inforward_split_buf: torch.Tensor | None = None
        self._inforward_split_offsets: list[int] = []

        # Each request's replay slot (see ReplaySlotIndices) and the replay
        # work items, which order the generation rows write-first for the
        # partitioned cached replay.
        self.replay_state_indices = self.state_indices
        self._replay_slots = ReplaySlotIndices(max_batch_size)
        self.replay_work_items = torch.zeros(
            max_batch_size, REPLAY_WORK_ITEM_WIDTH, dtype=torch.int32, device="cuda"
        )
        self.replay_n_writes = torch.zeros(1, dtype=torch.int32, device="cuda")
        # Decode rows of the step the work items were prepared for; the
        # all-layer checkpoint commit checks it.
        self.replay_num_decodes = 0

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        # The cu_seqlens buffers are rewritten in place: drop the GDN launch
        # path's per-iteration int32 casts of them first.
        invalidate_int32_cu_seqlens_cache()
        self.state_reset_done = False
        self.gdn_iteration_context = None
        super().prepare(attn_metadata)
        batch_size = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        kv_cache_manager = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids

        # Only context requests without a state need their slots zeroed;
        # continuation chunks (the common case under prefix reuse) skip the
        # reset launches in every GDN layer. Warmup forces the flag on so the
        # reset kernel is compiled before the first real iteration.
        self.prefill_needs_state_reset = num_contexts > 0 and bool(
            Mamba2Metadata._warmup_force_initial_states
            or not self.has_initial_states_cpu[:num_contexts].all()
        )
        num_decodes = batch_size - num_contexts
        if num_contexts > 0 and num_decodes > 0:
            self.state_indices_decode[:num_decodes].copy_(
                self.state_indices[num_contexts:batch_size], non_blocking=True
            )

        self.inforward_split = None
        plan = getattr(kv_cache_manager, "inforward_snapshot_plan", None)
        if (
            GDNMambaMetadata._warmup_inforward_split
            and num_contexts > 0
            and request_ids is not None
        ):
            plan = self._warmup_inforward_split_plan(attn_metadata, kv_cache_manager, num_contexts)
        if isinstance(plan, dict) and plan and num_contexts > 0 and request_ids is not None:
            self._prepare_inforward_split(
                attn_metadata, kv_cache_manager, plan, num_contexts, batch_size
            )

        if (
            kv_cache_manager is not None
            and hasattr(kv_cache_manager, "get_state_indices")
            and request_ids is not None
        ):
            self.replay_state_indices = self._replay_slots.resolve(
                kv_cache_manager, request_ids[:batch_size], self.state_indices
            )
        else:
            self.replay_state_indices = self.state_indices
        self._prepare_replay_work_items(kv_cache_manager, batch_size, num_contexts)

    def prepare_steady_gen_step(self, attn_metadata: AttentionMetadata) -> None:
        # The replay slots stay in place with the state slots; the work items follow the
        # accepted tokens of the previous step.
        self.prefill_needs_state_reset = False
        self.state_reset_done = False
        self.gdn_iteration_context = None
        super().prepare_steady_gen_step(attn_metadata)
        batch_size = attn_metadata.seq_lens.shape[0]
        self._prepare_replay_work_items(attn_metadata.kv_cache_manager, batch_size, 0)

    def _prepare_replay_work_items(
        self, kv_cache_manager, batch_size: int, num_contexts: int
    ) -> None:
        self.replay_num_decodes = 0
        get_replay_metadata = getattr(kv_cache_manager, "get_replay_state_update_metadata", None)
        replay_metadata = get_replay_metadata() if get_replay_metadata is not None else None
        if replay_metadata is None:
            return
        num_decodes = batch_size - num_contexts
        self.replay_num_decodes = num_decodes
        if num_decodes == 0:
            return
        # FlashInfer ring replay folds in-kernel and never reads work items.
        if getattr(kv_cache_manager, "use_gdn_flashinfer_ring_replay", False):
            return
        if getattr(kv_cache_manager, "use_gdn_mtp_replay", False):
            snapshot_gdn_mtp_replay_work_items(
                self.state_indices[num_contexts:batch_size],
                self.replay_state_indices[num_contexts:batch_size],
                replay_metadata.prev_num_accepted_tokens,
                self.replay_work_items,
            )
            return
        use_gdn_all_layer_commit = getattr(
            kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False
        )
        # The fused small-batch GDN kernel commits its checkpoint in-layer
        # and indexes cache metadata directly. Work items are only consumed
        # by the partitioned replay + all-layer commit path.
        if use_gdn_all_layer_commit and not kv_cache_manager.gdn_replay_partitioned(num_decodes):
            return
        build_replay_work_items(
            replay_metadata,
            self.state_indices[num_contexts:batch_size],
            self.replay_state_indices[num_contexts:batch_size],
            self.replay_work_items,
            self.replay_n_writes,
        )

    def _prepare_inforward_split(
        self,
        attn_metadata: AttentionMetadata,
        kv_cache_manager,
        plan: dict[int, InforwardSnapshotEntry],
        num_contexts: int,
        batch_size: int,
    ) -> None:
        """Index tensors of the in-forward Mamba snapshot split.

        A planned request's chunk [start, start + len) becomes two chunked
        prefill sequences, [start, start + before) and the tail [start + before,
        start + len); both gather the request's live state. The GDN layer
        writes the first one's final state (the state at P) to the live slot,
        recomputes the tail from it with the outputs written back at
        ``tail_tok``, and stages the tail's final state (the state at the
        prompt end) in the request's row. ``conv_win`` holds the token
        positions of the convolution window that ends at P.
        """
        request_ids = attn_metadata.request_ids[:num_contexts]
        entries = [plan.get(rid) for rid in request_ids]
        k = sum(entry is not None for entry in entries)
        if k == 0:
            return
        if batch_size != num_contexts:
            raise RuntimeError(
                "In-forward Mamba snapshot planned for a batch with generation requests"
            )
        width = kv_cache_manager.conv_state_shape[-1]
        rows_cap = kv_cache_manager.inforward_snapshot_rows
        tail_cap = rows_cap * kv_cache_manager.inforward_snapshot_max_tail
        lens = attn_metadata.seq_lens[:num_contexts].tolist()
        vals = {name: [] for name in _INFORWARD_SPLIT_FIELDS}
        vals["cu"].append(0)
        vals["cu_tail"].append(0)
        start = 0
        for rid, length, entry in zip(request_ids, lens, entries):
            slot = kv_cache_manager.state_index(rid)
            vals["live"].append(slot)
            vals["main_rows"].append(len(vals["gather"]))
            vals["gather"].append(slot)
            if entry is None:
                vals["cu"].append(start + length)
            else:
                before, after = entry.num_before, entry.num_after
                if before + after != length or before < width:
                    raise RuntimeError(
                        f"In-forward Mamba snapshot: request {rid} chunk of "
                        f"{length} tokens does not match the plan ({before} + "
                        f"{after}, window {width})"
                    )
                vals["split_rows"].append(len(vals["gather"]) - 1)
                vals["stage_rows"].append(entry.row)
                vals["cu"].append(start + before)
                # The tail as a sequence of its own: its outputs are recomputed
                # from the state at P, its final state is dropped.
                vals["gather"].append(slot)
                vals["cu"].append(start + length)
                vals["tail_tok"].extend(range(start + before, start + length))
                vals["cu_tail"].append(vals["cu_tail"][-1] + after)
                vals["tail_slots"].append(slot)
                vals["conv_win"].extend(range(start + before - width, start + before))
            start += length
        num_tail_tokens = len(vals["tail_tok"])
        if k > rows_cap or num_tail_tokens > tail_cap:
            raise RuntimeError(
                f"In-forward Mamba snapshot: {k} requests / {num_tail_tokens} "
                "tail tokens exceed the staging capacity"
            )
        if self._inforward_split_buf is None:
            batch_cap = self.max_batch_size
            capacity = (
                batch_cap + rows_cap + 1,
                batch_cap + rows_cap,
                batch_cap,
                batch_cap,
                rows_cap,
                rows_cap,
                tail_cap,
                rows_cap + 1,
                rows_cap,
                width * rows_cap,
            )
            self._inforward_split_offsets = [0, *itertools.accumulate(capacity)]
            self._inforward_split_buf = torch.zeros(
                self._inforward_split_offsets[-1], dtype=torch.long, device="cuda"
            )
        offsets = self._inforward_split_offsets
        flat = [0] * offsets[-1]
        for name, start in zip(_INFORWARD_SPLIT_FIELDS, offsets):
            flat[start : start + len(vals[name])] = vals[name]
        # One upload through a fresh pinned tensor: no later host write can
        # race the copy.
        self._inforward_split_buf.copy_(
            torch.tensor(flat, dtype=torch.long, pin_memory=prefer_pinned()), non_blocking=True
        )
        views = {
            name: self._inforward_split_buf[start : start + len(vals[name])]
            for name, start in zip(_INFORWARD_SPLIT_FIELDS, offsets)
        }
        self.inforward_split = InforwardSnapshotSplit(k=k, num_tail_tokens=num_tail_tokens, **views)

    @staticmethod
    def _warmup_inforward_split_plan(
        attn_metadata: AttentionMetadata, kv_cache_manager, num_contexts: int
    ) -> dict[int, InforwardSnapshotEntry]:
        """Split each context request with room for the longest tail, up to the
        staging rows (``inforward_split_for_warmup``)."""
        tail = kv_cache_manager.inforward_snapshot_max_tail
        plan = {}
        for rid, length in zip(
            attn_metadata.request_ids[:num_contexts],
            attn_metadata.seq_lens[:num_contexts].tolist(),
        ):
            if (
                len(plan) < kv_cache_manager.inforward_snapshot_rows
                and length - tail >= kv_cache_manager.inforward_snapshot_min_main
            ):
                plan[rid] = InforwardSnapshotEntry(len(plan), length - tail, length - tail, tail)
        return plan


def gdn_metadata_for_batch(attn_metadata: AttentionMetadata) -> GDNMambaMetadata | None:
    """The attention metadata's GDN Mamba metadata, rebuilt when its capacity is not the batch's.

    A rebuilt instance keeps the GDN metadata class in use and replaces
    ``attn_metadata.mamba_metadata``, so the GDN layers and the cache manager's
    acceptance update read the same instance; it is prepared for this batch.
    Attention metadata without a Mamba cache manager holds ``False``: it is
    rebuilt too, with no state to prepare. None (not yet prepared) is returned
    as is.
    """
    metadata = attn_metadata.mamba_metadata
    if metadata is None or (
        getattr(metadata, "max_batch_size", None) == attn_metadata.max_num_requests
    ):
        return metadata
    metadata_cls = type(metadata) if isinstance(metadata, GDNMambaMetadata) else GDNMambaMetadata
    rebuilt = metadata_cls(attn_metadata.max_num_requests, chunk_size=128)
    if metadata is not False:
        rebuilt.prepare(attn_metadata)
    attn_metadata.mamba_metadata = rebuilt
    return rebuilt


__all__ = [
    "GDNMambaMetadata",
    "InforwardSnapshotSplit",
    "gdn_metadata_for_batch",
]
