# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache manager of the Kimi K3 SM107 targets: the engine's K3 manager plus KDA mid-chunk snapshots.

The engine resolves ``KimiK3HybridCacheManager`` for a Kimi K3 model; each SM107 target declares one of
the subclasses here as ``RuntimeHooks.kv_cache_manager_cls``, and the engine builds it instead:

* :class:`KimiK3Sm107MidchunkCacheManager` (tp1, the context servers): KDA mid-chunk snapshots at
  ``KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT``, the prefill kernel's chunk;
* :class:`KimiK3Sm107CacheManager` (dep16, helix8): mid-chunk snapshots off, which leaves every step
  as the engine's ``KimiK3HybridCacheManager`` runs it.

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
from typing import ClassVar, NamedTuple

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm.logger import logger

from ...common.kda_metadata import KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT

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


class KimiK3Sm107CacheManager(KimiK3HybridCacheManager):
    """``KimiK3HybridCacheManager`` with KDA mid-chunk snapshots (see the module).

    Mid-chunk snapshots are on when the class's ``KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT`` is positive, block
    reuse is on and the manager is not a draft model's; this class keeps them off.
    """

    # Step alignment, in tokens, at which the model's KDA prefill stores a snapshot inside a context
    # step; 0: no mid-chunk snapshots.
    KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT: ClassVar[int] = 0
    # Step alignment of mid-chunk snapshot points; None when they are off.
    _kda_midchunk_snapshot_alignment: int | None = None

    @override
    def __init__(self, *args, **kwargs) -> None:
        self._kda_midchunk_snapshots: dict[int, _KDAMidchunkSnapshot] = {}
        # Requests whose snapshot plan the model fetched this step.
        self._kda_midchunk_handed_out: set[int] = set()
        self._kda_midchunk_snapshots_taken_total = 0
        self._kda_midchunk_snapshots_skipped_total = 0
        super().__init__(*args, **kwargs)
        self._kda_midchunk_snapshot_alignment = self._get_kda_midchunk_snapshot_alignment()

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
