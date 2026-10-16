# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure host mixed geometry with context/full-width capacity reservations.

The caller must authenticate the current ready host survival matrix and retain
request ownership. This module neither reads device tensors nor prices routes.
Actual allocation is delegated to the existing DSpark top-k implementation.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Capacity:
    proposed_body_rows: int
    eligible_rows: int
    optional_budget: int
    total_real_rows: int
    expected_yield: float


@dataclass(frozen=True)
class LocalCurve:
    request_ids: Tuple[int, ...]
    survival: torch.Tensor
    context_rows: int
    fixed_generation_rows: int
    physical_k: int
    min_verify_len: int
    native_expected_yield: float
    candidates: Tuple[Capacity, ...]

    def candidate(self, body_rows: int) -> Capacity:
        for item in self.candidates:
            if item.proposed_body_rows == body_rows:
                return item
        raise ValueError("Unlisted or locally infeasible body capacity")


def _nonnegative_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def prepare_local_curve(
    *,
    survival: torch.Tensor,
    request_ids: Tuple[int, ...],
    context_rows: int,
    fixed_generation_rows: int,
    physical_k: int,
    min_verify_len: int,
    body_capacities: Tuple[int, ...],
) -> LocalCurve:
    """Reserve context/full-width rows before filling eligible draft prefixes.

    Infeasible capacities are omitted. Context work contributes to paid rows,
    not fictitious generation progress. Final-context progress is preserved by
    the caller as a separate, candidate-independent component.
    """
    for name, value in (
        ("context_rows", context_rows),
        ("fixed_generation_rows", fixed_generation_rows),
        ("physical_k", physical_k),
        ("min_verify_len", min_verify_len),
    ):
        _nonnegative_integer(value, name)
    if not 1 <= min_verify_len <= physical_k:
        raise ValueError("Invalid draft prefix bounds")
    if (
        not isinstance(survival, torch.Tensor)
        or survival.device.type != "cpu"
        or survival.ndim != 2
        or tuple(survival.shape) != (len(request_ids), physical_k)
    ):
        raise ValueError("Requires a ready CPU [eligible requests, physical K] matrix")
    if any(type(item) is not int or item < 0 for item in request_ids) or len(
        set(request_ids)
    ) != len(request_ids):
        raise ValueError("Request identities must be unique nonnegative integers")
    if (
        not body_capacities
        or any(type(item) is not int or item <= 0 for item in body_capacities)
        or tuple(sorted(set(body_capacities))) != body_capacities
    ):
        raise ValueError("Body capacities must be positive, strictly increasing")
    # Own the admitted host matrix: later copy-buffer reuse cannot change the
    # capacity curve between the group vote and its allocation.
    matrix = survival.detach().to(dtype=torch.float32).clone()
    values = matrix.numpy().astype(np.float64, copy=False)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or (np.diff(values, axis=1) > 0).any()
    ):
        raise ValueError("Survival must be finite probabilities and prefix-monotone")
    num_real = len(request_ids)
    floor_rows = num_real * (min_verify_len + 1)
    native_rows = num_real * (physical_k + 1)
    base_yield = float(num_real) + float(values[:, :min_verify_len].sum(dtype=np.float64))
    ordered = np.sort(values[:, min_verify_len:physical_k].reshape(-1))[::-1]
    prefix = np.concatenate(([0.0], np.cumsum(ordered, dtype=np.float64)))
    native_yield = base_yield + float(prefix[-1])
    reserve = context_rows + fixed_generation_rows
    candidates = []
    for body_rows in body_capacities:
        available = body_rows - reserve
        if available < floor_rows:
            continue
        eligible_rows = min(available, native_rows)
        optional_budget = eligible_rows - floor_rows
        candidates.append(
            Capacity(
                body_rows,
                eligible_rows,
                optional_budget,
                reserve + eligible_rows,
                base_yield + float(prefix[optional_budget]),
            )
        )
    return LocalCurve(
        tuple(request_ids),
        matrix,
        context_rows,
        fixed_generation_rows,
        physical_k,
        min_verify_len,
        native_yield,
        tuple(candidates),
    )


def allocate_candidate(
    curve: LocalCurve,
    body_rows: int,
    original_allocate: Callable[[torch.Tensor, int, int, int], torch.Tensor],
):
    """Use original DSpark top-k with survival_eps=0 and validate exact rows."""
    candidate = curve.candidate(body_rows)
    lengths = original_allocate(
        curve.survival, candidate.optional_budget, curve.physical_k, curve.min_verify_len
    )
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.device.type != "cpu"
        or lengths.ndim != 1
        or len(lengths) != len(curve.request_ids)
    ):
        raise ValueError("Allocator returned incompatible request lengths")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Allocator lengths must be integer")
    windows = tuple(int(value) for value in lengths.tolist())
    if (
        any(not curve.min_verify_len <= value <= curve.physical_k for value in windows)
        or sum(value + 1 for value in windows) != candidate.eligible_rows
    ):
        raise ValueError("Allocator did not spend the admitted eligible-row budget")
    return tuple(zip(curve.request_ids, windows))


def realized_paid_bucket(
    curves: Tuple[LocalCurve, ...], proposed_body_rows: int, captured_body_rows: Tuple[int, ...]
) -> int:
    """Canonicalize actual all-rank maximum; do not manufacture padding work."""
    if not curves or proposed_body_rows not in captured_body_rows:
        raise ValueError("Requires a captured proposal and nonempty rank group")
    if (
        any(type(item) is not int or item <= 0 for item in captured_body_rows)
        or tuple(sorted(set(captured_body_rows))) != captured_body_rows
    ):
        raise ValueError("Captured body grid must be positive and strictly increasing")
    needed = max(curve.candidate(proposed_body_rows).total_real_rows for curve in curves)
    for bucket in captured_body_rows:
        if bucket >= needed:
            return bucket
    raise ValueError("Actual rows exceed the captured body grid")
