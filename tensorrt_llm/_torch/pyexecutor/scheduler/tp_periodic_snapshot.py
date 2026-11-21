# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agree on a periodic-snapshot expansion after the baseline batch is scheduled."""

from typing import TYPE_CHECKING

from ..kv_cache.mamba_cache_manager.multi_snapshot import plan_periodic_snapshot_chunk

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
        MambaHybridCacheManagerV2,
    )

    from .scheduler import ScheduledRequests


def coordinate_periodic_snapshot(
    batch: "ScheduledRequests",
    manager: "MambaHybridCacheManagerV2",
    dist: "Distributed",
    *,
    extra_budget: int | None,
    max_context_length: int | None,
    chunk_unit_size: int,
    enabled: bool = True,
) -> None:
    """Expand one context only after every TP rank agrees and reserves resources.

    All ranks in the TP group must call this once per scheduling pass. A failed
    reservation returns the scheduled batch only after every rank cleans up.
    """
    batch.periodic_snapshot_tp = False
    contexts = batch.context_requests
    baseline = (
        tuple(
            (
                req.py_request_id,
                req.context_current_position,
                req.context_chunk_size,
                req.prompt_len,
                req.context_remaining_length,
                req.is_last_context_chunk,
                req.py_mamba_inforward_snapshot_point,
                tuple(
                    sorted(
                        point
                        for point in req.expect_snapshot_points
                        if req.context_current_position
                        < point
                        <= req.context_current_position + req.context_chunk_size
                    )
                ),
            )
            for req in contexts
        ),
        tuple(
            (
                req.py_request_id,
                req.py_beam_width,
                getattr(req, "num_draft_tokens", 0),
                getattr(req, "py_batch_idx", None) is not None,
            )
            for req in batch.generation_requests
        ),
        tuple(req.py_request_id for req in batch.encoder_requests),
        tuple(req.py_request_id for req in batch.paused_requests),
        tuple(req.py_request_id for req in batch.recompute_paused_requests),
        tuple(sorted(batch.paused_in_place_ids)),
    )
    proposal = None
    if enabled and len(contexts) == 1 and not batch.encoder_requests:
        req = contexts[0]
        start = req.context_current_position
        baseline_end = start + req.context_chunk_size
        points = tuple(sorted(set(req.expect_snapshot_points)))
        cache = manager.kv_cache_map.get(req.py_request_id)
        if (
            cache is not None
            and cache.is_active
            and not req.is_dummy_request
            and not req.py_multimodal_data
            and req.py_mamba_inforward_snapshot_point is None
            and manager.context_replay_tokens(req) is None
            and not any(start < point < baseline_end for point in points)
            and all(getattr(item, "num_draft_tokens", 0) == 0 for item in batch.generation_requests)
        ):
            identity = (
                points,
                manager.kv_cache_config.mamba_state_config.periodic_snapshot_interval,
                manager.periodic_snapshot_alignment,
                max_context_length,
                chunk_unit_size,
            )
            proposal = (identity, manager.periodic_snapshot_capacity, extra_budget)

    exchanged = dist.tp_allgather((baseline, proposal))
    if any(item[0] != exchanged[0][0] for item in exchanged):
        raise RuntimeError("TP baseline batches differ before periodic snapshot agreement")
    proposals = [item[1] for item in exchanged]
    if any(item is None for item in proposals):
        return
    if any(item[0] != proposals[0][0] for item in proposals):
        return

    req = contexts[0]
    start = req.context_current_position
    interval = manager.kv_cache_config.mamba_state_config.periodic_snapshot_interval
    capacity = min(item[1] for item in proposals)
    if capacity < 2:
        return
    available = min(
        req.context_remaining_length if item[2] is None else item[2] for item in proposals
    )
    desired = min(req.context_remaining_length, req.context_chunk_size + available)
    if max_context_length is not None:
        desired = min(desired, max_context_length)
    plan = plan_periodic_snapshot_chunk(
        start,
        start + desired,
        req.expect_snapshot_points,
        interval,
        manager.periodic_snapshot_alignment,
        capacity - 1,
    )
    if plan.end < req.prompt_len and plan.end not in req.expect_snapshot_points:
        desired = ((plan.end - start) // chunk_unit_size) * chunk_unit_size
        plan = plan_periodic_snapshot_chunk(
            start,
            start + desired,
            req.expect_snapshot_points,
            interval,
            manager.periodic_snapshot_alignment,
            capacity - 1,
        )
    if plan.end <= start + req.context_chunk_size or not plan.points:
        return

    cache = manager.kv_cache_map[req.py_request_id]
    baseline_capacity = cache.capacity
    target_capacity = max(baseline_capacity, plan.end + manager._kv_alloc_extra_tokens)
    reserved = False
    grown = False
    error = None
    try:
        reserved = manager.reserve_periodic_snapshots(
            req, plan.end, plan.points, without_eviction=True
        )
        if reserved:
            grown = cache.try_resize_capacity_without_eviction(target_capacity)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    votes = dist.tp_allgather_int64([int(reserved and grown), int(error is not None)])
    errors = dist.tp_allgather(error) if any(vote[1] for vote in votes) else ()
    if not all(vote[0] for vote in votes):
        error_detail = next((message for message in errors if message is not None), None)
        cleanup_errors = []
        cleanup_exception = None
        try:
            if grown and not cache.try_resize_capacity_without_eviction(baseline_capacity):
                raise RuntimeError("Failed to restore the scheduled context capacity")
        except Exception as exc:
            cleanup_exception = exc
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        try:
            if reserved:
                manager.release_periodic_snapshots(req)
        except Exception as exc:
            if cleanup_exception is None:
                cleanup_exception = exc
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        # All ranks observed the failed resource vote and must finish cleanup
        # before any rank can return the baseline batch to the executor.
        cleanup_votes = dist.tp_allgather_int64([int(bool(cleanup_errors))])
        if any(vote[0] for vote in cleanup_votes):
            gathered = dist.tp_allgather("; ".join(cleanup_errors) if cleanup_errors else None)
            detail = next(message for message in gathered if message is not None)
            if error_detail is not None:
                detail = f"{detail}; original reservation error: {error_detail}"
            raise RuntimeError(
                f"Periodic snapshot cleanup failed on a TP rank: {detail}"
            ) from cleanup_exception
        if error_detail is not None:
            raise RuntimeError(f"Periodic snapshot reservation failed on a TP rank: {error_detail}")
        return

    commit_error = None
    commit_exception = None
    try:
        if not cache.resize(target_capacity):
            raise RuntimeError("Failed to accept the reserved context capacity")
        manager._fill_fresh_kv_pages(req.py_request_id)
        if target_capacity > baseline_capacity and req.py_ctx_pre_resize_cap is None:
            req.py_ctx_pre_resize_cap = baseline_capacity
        req.context_chunk_size = plan.end - start
        batch.reset_context_requests(contexts)
    except Exception as exc:
        commit_exception = exc
        commit_error = f"{type(exc).__name__}: {exc}"

    # The preceding vote guarantees every rank reaches this agreement. Once
    # local commits begin, a partial commit cannot be safely resumed as the
    # baseline batch; every rank must fail before any rank runs the forward.
    commit_votes = dist.tp_allgather_int64([int(commit_error is not None)])
    if any(vote[0] for vote in commit_votes):
        errors = dist.tp_allgather(commit_error)
        error_detail = next(message for message in errors if message is not None)
        try:
            manager.release_periodic_snapshots(req)
        except Exception as exc:
            raise RuntimeError(
                f"Periodic snapshot commit failed on a TP rank: {error_detail}; "
                f"local snapshot release failed: {type(exc).__name__}: {exc}"
            ) from exc
        raise RuntimeError(
            f"Periodic snapshot commit failed on a TP rank: {error_detail}"
        ) from commit_exception
    batch.periodic_snapshot_tp = True


def finish_periodic_snapshots_tp(
    batch: "ScheduledRequests", manager: "MambaHybridCacheManagerV2", dist: "Distributed"
) -> None:
    """Finish an agreed TP expansion and fail every rank if a snapshot commit fails."""
    local_exception = None
    error = None
    try:
        manager.finish_periodic_snapshots(batch)
    except Exception as exc:
        local_exception = exc
        error = f"{type(exc).__name__}: {exc}"
    votes = dist.tp_allgather_int64([int(error is not None)])
    if any(vote[0] for vote in votes):
        errors = dist.tp_allgather(error)
        detail = next(message for message in errors if message is not None)
        raise RuntimeError(
            f"Periodic snapshot forward commit failed on a TP rank: {detail}"
        ) from local_exception
