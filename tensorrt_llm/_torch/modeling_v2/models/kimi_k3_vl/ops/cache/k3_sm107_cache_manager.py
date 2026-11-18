# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache manager of the Kimi K3 SM107 targets: the engine's K3 manager plus KDA mid-chunk snapshots.

The engine resolves ``KimiK3HybridCacheManager`` for a Kimi K3 model; each SM107 target declares one of
the subclasses here as ``RuntimeHooks.kv_cache_manager_cls``, and the engine builds it instead:

* :class:`KimiK3Sm107MidchunkCacheManager` (tp1, the context servers): KDA mid-chunk snapshots at
  ``KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT``, the prefill kernel's chunk;
* :class:`KimiK3Sm107CacheManager` (dep16, helix8): mid-chunk snapshots off, which leaves every step
  as the engine's ``KimiK3HybridCacheManager`` runs it.

A target may subclass either one to turn on the recurrent-state cache policies, class constants of
:class:`KimiK3Sm107CacheManager`: the lowest eviction priority for short prompts' recurrent-state
pages (``LOW_PRIORITY_SNAPSHOT_PROMPT_LEN``) and the one-kernel upload of the recurrent-state indices
and the dummy-request mask (``STATE_INDEX_UPLOAD_KERNEL``). They change how recurrent-state pages are
kept and uploaded, never what a forward computes.

Both keep the KDA replay buffers of speculative decoding in the recurrent-state slot, which the
SM107 KDA kernels address them by, instead of the engine's replay region.

The KV cache managers are runtime interfaces, not Modeling V1 model code (``modeling_v2/README.md``,
"Modeling V1 code"), so the family subclasses the engine's class rather than reimplementing it. The
switch is a class constant of the declared class; nothing here reads the environment.

Mid-chunk snapshots: a context request whose last snapshot point lies inside its final chunk is
prefilled in one step, and a step may pass one other snapshot point (a branch point, a point
requested by a request waiting for this prefix, an additional offset), ending at the following point
at the latest. The KDA layers write each snapshot into a pending V2 SSM page that the context update
commits by move. The model's KDA metadata fetches the step's snapshots with
:meth:`KimiK3Sm107CacheManager.get_kda_midchunk_snapshots`; a snapshot it does not fetch is not
committed. The scheduler reads the manager's per-request marks ``py_midchunk_snapshot_alignment`` and
``py_midchunk_pass_point`` to size the context chunks.
"""

from __future__ import annotations

import sys
import time
from typing import ClassVar, NamedTuple

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    PAGE_PRIORITY_DEFAULT,
    PAGE_PRIORITY_MIN,
    PagePriorityCallback,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

from ...common.kda_metadata import KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT
from ...kernels.k3r_state_upload.state_upload_kernel import copy_small_pinned_pair

__all__ = ["KimiK3Sm107CacheManager", "KimiK3Sm107MidchunkCacheManager"]


class _KDAMidchunkSnapshot(NamedTuple):
    """A recurrent-state snapshot taken inside a request's context step."""

    # Prompt position of the snapshot; tokens [0, point) are committed with it.
    point: int
    # Tokens of the step before the snapshot point.
    offset: int
    # Pending SSM page that receives the snapshot. None on a rank without
    # local KDA layers, or when no page was available (snapshot skipped).
    slot: int | None


def _format_ratio(ratio) -> str:
    return "[" + ", ".join(f"{share:.4f}" for share in ratio) + "]"


def _low_priority_recurrent_state(_ordinal: int, life_cycle: object) -> int:
    """Lowest eviction priority for recurrent-state pages, the default for attention pages."""
    if isinstance(life_cycle, AttnLifeCycle):
        return PAGE_PRIORITY_DEFAULT
    return PAGE_PRIORITY_MIN


class _StateSlotKDAReplayState(KDAReplayState):
    """``KDAReplayState`` whose replay buffers live in the recurrent-state slot.

    The SM107 KDA kernels (the fused verify, the replay conv-window commit) address the replay
    caches and the accepted-draft counts by the recurrent-state slot, so the manager declares the
    replay buffers with the recurrent state rather than as a replay region of their own.
    """

    @property
    @override
    def replay_shares_state_slot(self) -> bool:
        return True


class KimiK3Sm107CacheManager(KimiK3HybridCacheManager):
    """``KimiK3HybridCacheManager`` with KDA mid-chunk snapshots (see the module).

    Mid-chunk snapshots are on when the class's ``KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT`` is positive, block
    reuse is on and the manager is not a draft model's; this class keeps them off.

    The KDA replay buffers of speculative decoding share the recurrent-state slot
    (:class:`_StateSlotKDAReplayState`), and the KV cache size estimate counts them per state slot.

    With block reuse on, the recurrent snapshot and reuse counters are logged at INFO at most once a
    minute, and the pool ratio a unified draft pool resolves to is logged once.
    """

    # Step alignment, in tokens, at which the model's KDA prefill stores a snapshot inside a context
    # step; 0: no mid-chunk snapshots.
    KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT: ClassVar[int] = 0
    # Step alignment of mid-chunk snapshot points; None when they are off.
    _kda_midchunk_snapshot_alignment: int | None = None
    # Context requests whose prompt is shorter than this many tokens commit their recurrent-state
    # pages (live state and snapshots) at the lowest eviction priority; attention pages keep theirs.
    # A snapshot is reusable only at its exact token boundary and takes a whole state slot whatever
    # the prompt length, while a short prompt is cheap to recompute. 0: off. Needs block reuse; a
    # draft model's manager keeps it off.
    LOW_PRIORITY_SNAPSHOT_PROMPT_LEN: ClassVar[int] = 0
    _low_priority_snapshot_prompt_len = 0
    # When the batch's recurrent-state indices or dummy-request mask change, write both device
    # buffers with one SM kernel from pinned snapshots instead of two copy-engine copies, whose
    # engine hand-offs delay the execution stream in front of the next step's inputs. Unchanged
    # buffers are not written either way.
    STATE_INDEX_UPLOAD_KERNEL: ClassVar[bool] = False
    # The recurrent snapshot and reuse counters are logged at INFO at most this often.
    _BRANCH_SNAPSHOT_COUNTERS_LOG_INTERVAL_S: ClassVar[float] = 60.0
    _branch_snapshot_counters_logged_at = float("-inf")

    @override
    def __init__(self, *args, **kwargs) -> None:
        self._kda_midchunk_snapshots: dict[int, _KDAMidchunkSnapshot] = {}
        # Requests whose snapshot plan the model fetched this step.
        self._kda_midchunk_handed_out: set[int] = set()
        self._kda_midchunk_snapshots_taken_total = 0
        self._kda_midchunk_snapshots_skipped_total = 0
        # Pinned snapshots read by in-flight state-index upload kernels, with the event after
        # which each is free (_upload_state_indices_and_dummy_mask).
        self._state_upload_snapshots: list = []
        super().__init__(*args, **kwargs)
        self._kda_midchunk_snapshot_alignment = self._get_kda_midchunk_snapshot_alignment()
        self._low_priority_snapshot_prompt_len = self._get_low_priority_snapshot_prompt_len()

    @override
    def _initialize_spec_state(self):
        state = super()._initialize_spec_state()
        if not isinstance(state, KDAReplayState) or state.replay_shares_state_slot:
            return state
        self._kda_replay = _StateSlotKDAReplayState(
            state.num_speculative_tokens,
            stochastic_rounding=self._mamba_ssm_stochastic_rounding,
            draft_records=state.draft_records,
        )
        self._kda_replay.validate(self._state_layout)
        return self._kda_replay

    @classmethod
    @override
    def _extra_state_bytes_per_rank(cls, model_config, mapping, **kwargs) -> int:
        # Every recurrent-state slot holds the replay buffers besides the seeds.
        return KDAReplayState.estimate_seed_bytes_per_slot(
            model_config, mapping, **kwargs
        ) + KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)

    @classmethod
    @override
    def _estimate_fixed_region_bytes(cls, model_config, mapping, **kwargs) -> int:
        # No replay region: the replay buffers are counted per recurrent-state slot.
        return 0

    def _get_kda_midchunk_snapshot_alignment(self) -> int | None:
        """Step alignment of mid-chunk snapshot points, or None when they are off.

        Depends only on the declared class and configuration shared by all
        ranks, so that every rank merges the same context chunks.
        """
        alignment = self.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT
        if alignment <= 0 or not self.enable_block_reuse or self.is_draft:
            return None
        logger.info("KDA mid-chunk recurrent-state snapshots are enabled")
        return alignment

    def _get_low_priority_snapshot_prompt_len(self) -> int:
        """``LOW_PRIORITY_SNAPSHOT_PROMPT_LEN`` where it applies, else 0."""
        length = self.LOW_PRIORITY_SNAPSHOT_PROMPT_LEN
        if length <= 0 or not self.enable_block_reuse or self.is_draft:
            return 0
        logger.info(
            f"Recurrent-state pages of context requests shorter than {length} tokens use the "
            "lowest eviction priority"
        )
        return length

    @override
    def _context_priority_callback(self, req) -> PagePriorityCallback | None:
        """Lowest eviction priority for the recurrent state of short prompts.

        See ``LOW_PRIORITY_SNAPSHOT_PROMPT_LEN``.
        """
        if (
            self.local_num_mamba_layers > 0
            and not req.is_dummy
            and req.prompt_len < self._low_priority_snapshot_prompt_len
        ):
            return _low_priority_recurrent_state
        return super()._context_priority_callback(req)

    @override
    def _upload_state_indices_and_dummy_mask(self, is_dummy: list[bool]) -> bool:
        """Both uploads of ``_setup_state_indices`` as one SM kernel.

        See ``STATE_INDEX_UPLOAD_KERNEL``. Like the two ``DeviceCopyBuffer.update`` calls it
        replaces, nothing is written while both buffers already hold the host values. Otherwise one
        kernel writes both at the same stream position. The kernel reads fresh pinned snapshots of
        the host buffers (an admission and the scheduled batch may refill those in one iteration
        before the first upload has run); the pinned allocator does not see a kernel's reads, so
        the snapshots are held until an event recorded after the kernel. Returns False (nothing
        uploaded) to fall back to the copies.
        """
        if not self.STATE_INDEX_UPLOAD_KERNEL or self._dummy_request_mask_buffer is None:
            return super()._upload_state_indices_and_dummy_mask(is_dummy)
        self._fill_dummy_request_mask_host(is_dummy)
        indices, mask = self._state_indices_buffer, self._dummy_request_mask_buffer
        if indices.matches(self._host_state_indices) and mask.matches(
            self._dummy_request_mask_host
        ):
            return True
        snapshots = tuple(
            torch.empty_like(host, pin_memory=prefer_pinned()).copy_(host)
            for host in (self._host_state_indices, self._dummy_request_mask_host)
        )
        if not copy_small_pinned_pair(indices.tensor, snapshots[0], mask.tensor, snapshots[1]):
            return False
        indices.record(self._host_state_indices)
        mask.record(self._dummy_request_mask_host)
        done = torch.cuda.Event()
        done.record()
        self._state_upload_snapshots = [
            (held, event) for held, event in self._state_upload_snapshots if not event.query()
        ]
        self._state_upload_snapshots.append((snapshots, done))
        return True

    @override
    def _with_draft_pool_ratio(self, config):
        resolved = super()._with_draft_pool_ratio(config)
        if config.initial_pool_ratio is not None and self._appended_draft_layer_specs:
            logger.info(
                f"Unified draft pool ratio: layer-group ratios "
                f"{_format_ratio(resolved.initial_pool_ratio)} "
                f"(configured {_format_ratio(config.initial_pool_ratio)})"
            )
        return resolved

    def _maybe_log_branch_snapshot_counters(self) -> None:
        """Log the recurrent snapshot and reuse counters at INFO, at most once a minute.

        Called on every context admission of a block-reuse manager, so the line shows whether
        branch, waiter and mid-chunk snapshots are being taken.
        """
        now = time.monotonic()
        if (
            now - self._branch_snapshot_counters_logged_at
            < self._BRANCH_SNAPSHOT_COUNTERS_LOG_INTERVAL_S
        ):
            return
        self._branch_snapshot_counters_logged_at = now
        logger.info(
            f"[MambaHybridCacheManagerV2] recurrent snapshot counters "
            f"{self._format_branch_snapshot_counters()}"
        )

    @override
    def _record_branch_snapshot_point(self, req, kv_cache, num_lookup_tokens) -> None:
        if not req.is_dummy_request:
            self._maybe_log_branch_snapshot_counters()
        super()._record_branch_snapshot_point(req, kv_cache, num_lookup_tokens)

    @override
    def _format_branch_snapshot_counters(self) -> str:
        counters = super()._format_branch_snapshot_counters()
        if self._kda_midchunk_snapshot_alignment is None:
            return counters
        return (
            f"{counters} "
            f"midchunk_snapshots_taken={self._kda_midchunk_snapshots_taken_total} "
            f"midchunk_snapshots_skipped={self._kda_midchunk_snapshots_skipped_total}"
        )

    @override
    def prepare_expect_snapshot_points(self, requests) -> None:
        super().prepare_expect_snapshot_points(requests)
        # Read by the scheduler (_get_forced_context_chunk_size): the last
        # snapshot point no longer splits the context chunk, and a step may
        # pass the next other point.
        alignment = self._kda_midchunk_snapshot_alignment
        for request in requests:
            request.py_midchunk_snapshot_alignment = (
                alignment if not request.is_dummy_request else None
            )
            request.py_midchunk_pass_point = self._midchunk_pass_point(
                request, request.context_current_position
            )

    def _midchunk_pass_point(self, request, current_position: int) -> int | None:
        """The next snapshot point a context step may pass, if any.

        Every point ahead of the cursor but the last one qualifies; the last
        point merges by the scheduler's ``_is_midchunk_snapshot_point`` instead.
        """
        if self._kda_midchunk_snapshot_alignment is None or request.is_dummy_request:
            return None
        points = sorted(
            point for point in request.expect_snapshot_points if point > current_position
        )
        return points[0] if len(points) > 1 else None

    @override
    def _apply_branch_snapshot_point(self, request, context_current_position=None) -> None:
        super()._apply_branch_snapshot_point(request, context_current_position)
        current_position = (
            request.context_current_position
            if context_current_position is None
            else context_current_position
        )
        request.py_midchunk_pass_point = self._midchunk_pass_point(request, current_position)

    @override
    def _merge_waiter_snapshot_points(self, request) -> None:
        # A point requested by a waiting request (request_prefix_snapshot) can
        # arrive after the scheduler pass placed this request's points.
        super()._merge_waiter_snapshot_points(request)
        request.py_midchunk_pass_point = self._midchunk_pass_point(
            request, request.context_current_position
        )

    def _plan_kda_midchunk_snapshot(self, request) -> tuple[int, int] | None:
        """``(point, offset)`` of a snapshot inside this context step, if any."""
        alignment = getattr(request, "py_midchunk_snapshot_alignment", None)
        if not isinstance(alignment, int) or self._kda_midchunk_snapshot_alignment is None:
            return None
        start = request.context_current_position
        end = start + request.context_chunk_size
        # The scheduler lets at most one point into a step, but a point
        # requested after the step was scheduled can add another; the
        # deepest one is taken then.
        points = [
            point
            for point in request.expect_snapshot_points
            if start < point < end and (point - start) % alignment == 0
        ]
        if not points:
            return None
        return points[-1], points[-1] - start

    def _prepare_kda_midchunk_snapshots(self, context_requests) -> None:
        self._kda_midchunk_handed_out.clear()
        for request in context_requests:
            request_id = request.py_request_id
            stale = self._kda_midchunk_snapshots.pop(request_id, None)
            kv_cache = self.kv_cache_map.get(request_id)
            if stale is not None and kv_cache is not None:
                kv_cache.release_ssm_snapshot_slot()
            plan = self._plan_kda_midchunk_snapshot(request)
            if plan is None or kv_cache is None:
                continue
            point, offset = plan
            slot = None
            if self.local_num_mamba_layers > 0:
                slot = kv_cache.acquire_ssm_snapshot_slot()
                if slot is None:
                    self._kda_midchunk_snapshots_skipped_total += 1
                    logger.debug(
                        "KDA mid-chunk snapshot skipped for request "
                        f"{request_id} at {point}: no SSM page available"
                    )
            self._kda_midchunk_snapshots[request_id] = _KDAMidchunkSnapshot(point, offset, slot)

    def get_kda_midchunk_snapshots(self, request_ids: list[int]) -> list[tuple[int, int, int]]:
        """``(batch index, slot, offset)`` of this step's mid-chunk snapshots.

        ``request_ids`` are the step's context requests in batch order. The KDA
        layers store the recurrent state and convolution window after the
        request's first ``offset`` tokens of this step (a multiple of the
        declared alignment, strictly inside the step) into SSM pool slot
        ``slot`` (the index space of the state indices). Called by the KDA
        metadata of a model that declares mid-chunk snapshots; only snapshots
        handed out here are committed.
        """
        if not self._kda_midchunk_snapshots:
            return []
        snapshots = []
        for index, request_id in enumerate(request_ids):
            snapshot = self._kda_midchunk_snapshots.get(request_id)
            if snapshot is not None and snapshot.slot is not None:
                snapshots.append((index, snapshot.slot, snapshot.offset))
                self._kda_midchunk_handed_out.add(request_id)
        return snapshots

    def _disable_kda_midchunk_snapshots(self, request_id: int) -> None:
        """Stop planning snapshots that the model does not take.

        The KDA layers of the running model never fetched a planned snapshot,
        so its pending page holds no state. Every rank runs the same model and
        reaches this on the same step.
        """
        logger.warning(
            "KDA mid-chunk snapshots are turned off: the KDA layers did not take "
            f"the mid-chunk snapshot of request {request_id} (the model declares "
            f"{type(self).__name__}, whose KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT turns them on, but "
            "its KDA metadata does not call get_kda_midchunk_snapshots)."
        )
        self._kda_midchunk_snapshot_alignment = None

    def _commit_kda_midchunk_snapshot(self, request, snapshot: _KDAMidchunkSnapshot) -> None:
        """Commit the prompt up to the snapshot point with the pending SSM page.

        Runs before the context update of the step, in the order of a step that
        ended at the point: history to the point, then the commit.
        """
        needs_page = self.local_num_mamba_layers > 0
        if needs_page and snapshot.slot is None:
            # Already counted as skipped when no page was available.
            return
        request_id = request.py_request_id
        kv_cache = self.kv_cache_map.get(request_id)
        if kv_cache is None or not kv_cache.is_active:
            # Suspension released the pending page.
            self._kda_midchunk_snapshots_skipped_total += int(needs_page)
            return
        has_page = not needs_page or kv_cache.pending_ssm_snapshot_slot == snapshot.slot
        # Without a fetch by the KDA metadata the pending page holds no state.
        filled = not needs_page or request_id in self._kda_midchunk_handed_out
        self._kda_midchunk_handed_out.discard(request_id)
        # A step that did not run leaves the context position before the point.
        step_ran = request.context_current_position >= snapshot.point
        if (
            not has_page
            or not filled
            or not step_ran
            or kv_cache.num_committed_tokens >= snapshot.point
        ):
            kv_cache.release_ssm_snapshot_slot()
            self._kda_midchunk_snapshots_skipped_total += int(needs_page)
            if has_page and step_ran and not filled:
                self._disable_kda_midchunk_snapshots(request_id)
            return
        if not kv_cache.resize(None, snapshot.point):
            raise ValueError(
                "Failed to resize history length of V2 Mamba cache for "
                f"request {request_id} to {snapshot.point} tokens "
                "at a mid-chunk snapshot"
            )
        tokens = self._augment_tokens_for_block_reuse(
            self._reuse_token_source(request),
            request,
            start=kv_cache.num_committed_tokens,
            end=snapshot.point,
        )
        kv_cache.commit(tokens, ssm_from_pending=needs_page)
        if needs_page:
            self._kda_midchunk_snapshots_taken_total += 1
            self._count_snapshot_commit(request, snapshot.point)

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        if self._kda_midchunk_snapshots:
            for request in scheduled_batch.context_requests:
                snapshot = self._kda_midchunk_snapshots.pop(request.py_request_id, None)
                if snapshot is not None:
                    self._commit_kda_midchunk_snapshot(request, snapshot)
        super().update_context_resources(scheduled_batch)

    @override
    def prepare_resources(self, scheduled_batch) -> None:
        super().prepare_resources(scheduled_batch)
        if self._kda_midchunk_snapshot_alignment is not None:
            self._prepare_kda_midchunk_snapshots(scheduled_batch.context_requests)

    @override
    def free_resources(self, request, pin_on_release: bool = False) -> None:
        self._kda_midchunk_handed_out.discard(request.py_request_id)
        if self._kda_midchunk_snapshots.pop(request.py_request_id, None) is not None:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if kv_cache is not None and kv_cache.is_active:
                kv_cache.release_ssm_snapshot_slot()
        super().free_resources(request, pin_on_release)

    @override
    def shutdown(self) -> None:
        self._kda_midchunk_snapshots.clear()
        self._kda_midchunk_handed_out.clear()
        super().shutdown()


class KimiK3Sm107MidchunkCacheManager(KimiK3Sm107CacheManager):
    """:class:`KimiK3Sm107CacheManager` with mid-chunk snapshots at the KDA prefill kernel's chunk."""

    KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT: ClassVar[int] = KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT
