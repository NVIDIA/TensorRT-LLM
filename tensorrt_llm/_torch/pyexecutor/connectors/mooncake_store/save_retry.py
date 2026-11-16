# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded synchronous retries while the caller owns immutable save buffers."""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

MAX_PUT_ATTEMPTS = 3
RETRY_START_WINDOW_SECONDS = 0.1


@dataclass(frozen=True)
class SaveRetryResult:
    failed_indices: Tuple[int, ...]
    put_attempts: int
    attempted_keys: int
    retried_keys: int
    recovered_keys: int
    recovered_by_probe: int
    exception_attempts: int
    invalid_results: int
    failure_codes: Tuple[int, ...]


def _statuses(result, count: int) -> Tuple[List[Optional[int]], int]:
    """Keep only explicitly returned, correctly positioned integer statuses."""
    if not isinstance(result, (list, tuple)) or len(result) > count:
        return [None] * count, count
    values: List[Optional[int]] = [None] * count
    invalid = count - len(result)
    for index, value in enumerate(result):
        if isinstance(value, int) and not isinstance(value, bool):
            values[index] = value
        else:
            invalid += 1
    return values, invalid


def put_with_retries(
    store,
    keys: Sequence[str],
    addresses: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> SaveRetryResult:
    """Retry unresolved keys in-place, never retain pointers beyond this call.

    The caller must keep GPU pages and any staging slots immutable and owned
    until this function returns. Store calls remain synchronous. The time limit
    bounds when a retry may START; it cannot interrupt a running store RPC.
    """
    if len(keys) != len(addresses) or len(keys) != len(sizes):
        raise ValueError("Save key/buffer lengths differ")
    unresolved = list(range(len(keys)))
    deadline = clock() + RETRY_START_WINDOW_SECONDS
    attempts = attempted_keys = retried_keys = exceptions = invalid = 0
    first_failures = recovered_by_probe = 0
    codes = set()
    while unresolved and attempts < MAX_PUT_ATTEMPTS:
        if attempts:
            if clock() >= deadline:
                break
            try:
                probe, bad = _statuses(
                    store.batch_is_exist([keys[index] for index in unresolved]), len(unresolved)
                )
                invalid += bad
            except Exception:  # External Store boundary; unknown keys still need PUT.
                exceptions += 1
                probe = [None] * len(unresolved)
            remaining = [index for index, value in zip(unresolved, probe) if value != 1]
            recovered_by_probe += len(unresolved) - len(remaining)
            unresolved = remaining
            if not unresolved or clock() >= deadline:
                break
            retried_keys += len(unresolved)
        attempted_keys += len(unresolved)
        attempts += 1
        try:
            result, bad = _statuses(
                store.batch_put_from_multi_buffers(
                    [keys[index] for index in unresolved],
                    [addresses[index] for index in unresolved],
                    [sizes[index] for index in unresolved],
                ),
                len(unresolved),
            )
            invalid += bad
        except Exception:  # The caller's ownership and synchronous-call contract still apply.
            exceptions += 1
            result = [None] * len(unresolved)
        codes.update(value for value in result if value is not None and value < 0)
        unresolved = [
            index for index, value in zip(unresolved, result) if value is None or value < 0
        ]
        if attempts == 1:
            first_failures = len(unresolved)
    return SaveRetryResult(
        tuple(unresolved),
        attempts,
        attempted_keys,
        retried_keys,
        first_failures - len(unresolved),
        recovered_by_probe,
        exceptions,
        invalid,
        tuple(sorted(codes)),
    )
