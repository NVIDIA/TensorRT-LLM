# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current ready host snapshot ownership; decode device mode is untouched."""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .dspark_mixed_host_capacity import LocalCurve, prepare_local_curve


@dataclass(frozen=True)
class AuthenticatedCurve:
    curve: LocalCurve
    request_slots: Tuple[Tuple[int, int], ...]
    producing_iteration: int
    snapshot_staged_iteration: int
    staging_sequence: int
    ranking_mode: str = "mixed_current_ready_host"
    request_incarnations: Tuple[int, ...] = ()


def prepare_current_mixed_curve(
    *,
    planner,
    request_ids: Tuple[int, ...],
    rows: Tuple[int, ...],
    iteration: int,
    staging_sequence: int,
    context_rows: int,
    fixed_generation_rows: int,
    body_capacities: Tuple[int, ...],
    original_compute_survival: Callable,
    requests: Tuple[object, ...] = (),
    incarnations: Tuple[int, ...] = (),
) -> AuthenticatedCurve | None:
    """Return an owned current-host curve, or native fallback without waiting.

    This is an explicit mixed host-window contract, not fresh device ranking.
    It uses the existing ready/event and request-addressed gather API even when
    decode planner.device_windows is true. It never toggles that global flag.
    Source/runtime admission and all-rank voting belong to the caller.
    """
    if (
        type(iteration) is not int
        or iteration < 0
        or type(staging_sequence) is not int
        or staging_sequence < 0
        or len(request_ids) != len(rows)
    ):
        return None
    k, floor = int(planner.max_verify_len), int(planner.cfg.min_verify_len)
    if not request_ids:
        # A context/dummy-only rank has no invented confidence ownership/yield.
        curve = prepare_local_curve(
            survival=torch.empty((0, k), dtype=torch.float32),
            request_ids=(),
            context_rows=context_rows,
            fixed_generation_rows=fixed_generation_rows,
            physical_k=k,
            min_verify_len=floor,
            body_capacities=body_capacities,
        )
        return AuthenticatedCurve(curve, (), iteration, -1, staging_sequence)
    meta = getattr(planner, "_mixed_current_snapshot_meta", None)
    snapshot = planner._ready_snapshot()  # Existing query(), never synchronize().
    if (
        snapshot is None
        or meta is None
        or meta.get("buffer") is not snapshot
        or meta.get("event") is not planner._copy_event
        or meta.get("staging_sequence") != staging_sequence
        or type(meta.get("staged_iteration")) is not int
        or not 0 <= meta["staged_iteration"] < iteration
        or snapshot.device.type != "cpu"
    ):
        return None
    owners = meta.get("owners")
    if (
        not isinstance(owners, dict)
        or len(set(request_ids)) != len(request_ids)
        or len(set(rows)) != len(rows)
        or any(
            type(request_id) is not int
            or request_id < 0
            or type(row) is not int
            or row < 0
            or row >= snapshot.shape[0]
            or owners.get(request_id) != row
            for request_id, row in zip(request_ids, rows)
        )
    ):
        return None
    # Ownership consistency above remains fail-closed. Freshness is a separate
    # conservative partition: no writer proof means full K, never fake scores.
    stamp = meta.get("confidence_stamp")
    attempts = meta.get("producer_attempts")
    expected_writer = staging_sequence - 1  # Copy precedes forward q.
    valid = []
    if (
        expected_writer > 0
        and len(requests) == len(rows)
        and len(incarnations) == len(rows)
        and isinstance(attempts, dict)
        and stamp is getattr(planner, "_host_confidence_stamp", None)
        and isinstance(stamp, torch.Tensor)
        and stamp.device.type == "cpu"
        and stamp.dtype == torch.int32
        and stamp.ndim == 1
        and stamp.shape[0] == snapshot.shape[0]
    ):
        for index, (request_id, row, request, incarnation) in enumerate(
            zip(request_ids, rows, requests, incarnations)
        ):
            attempt = attempts.get(request_id)
            if (
                type(incarnation) is int
                and incarnation > 0
                and type(attempt) is tuple
                and len(attempt) == 3
                and attempt[0] is request
                and type(attempt[1]) is int
                and type(attempt[2]) is int
                and attempt[1] == row
                and attempt[2] == incarnation
                and int(stamp[row]) == expected_writer
            ):
                valid.append(index)
    fixed_generation_rows += (len(rows) - len(valid)) * (k + 1)
    request_ids = tuple(request_ids[index] for index in valid)
    rows = tuple(rows[index] for index in valid)
    incarnations = tuple(incarnations[index] for index in valid)
    if not rows:
        curve = prepare_local_curve(
            survival=torch.empty((0, k), dtype=torch.float32),
            request_ids=(),
            context_rows=context_rows,
            fixed_generation_rows=fixed_generation_rows,
            physical_k=k,
            min_verify_len=floor,
            body_capacities=body_capacities,
        )
        return AuthenticatedCurve(curve, (), iteration, meta["staged_iteration"], staging_sequence)
    selected = planner._gather_rows(num_gen_requests=len(rows), rows=rows, snapshot=snapshot)
    if selected is None:
        return None
    survival = original_compute_survival(planner.apply_calibration(selected))
    try:
        curve = prepare_local_curve(
            survival=survival,
            request_ids=tuple(request_ids),
            context_rows=context_rows,
            fixed_generation_rows=fixed_generation_rows,
            physical_k=k,
            min_verify_len=floor,
            body_capacities=body_capacities,
        )
    except ValueError:
        return None
    return AuthenticatedCurve(
        curve,
        tuple(zip(request_ids, rows)),
        iteration,
        meta["staged_iteration"],
        staging_sequence,
        request_incarnations=incarnations,
    )
