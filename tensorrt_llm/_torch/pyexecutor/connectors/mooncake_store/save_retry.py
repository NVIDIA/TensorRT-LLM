# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded synchronous put retries while the caller owns immutable save buffers.

A put the pool refuses (for example while the master is evicting to make room)
only costs a future cache miss, but under load such refusals come in bursts,
and every refused page is a prefix hole for later requests. A short retry of
just the refused keys recovers most of them. Before each retry the keys are
probed, because a put that raised may still have landed.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

#: Set to 1 to put each batch once, as without retries.
DISABLE_ENV = "TLLM_MOONCAKE_STORE_DISABLE_SAVE_RETRY"

MAX_PUT_ATTEMPTS = 3
#: Retries may start only this long after the first put. It bounds when a retry
#: starts, not how long a running store call takes.
RETRY_START_WINDOW_SECONDS = 0.1


def save_retry_enabled() -> bool:
    """Whether refused puts are retried; on unless `DISABLE_ENV` is 1."""
    return os.environ.get(DISABLE_ENV, "0").strip() != "1"


@dataclass(frozen=True)
class SaveRetryResult:
    """What one `put_with_retries` call did, for the caller's counters."""

    #: Indices into the caller's keys that are still not stored.
    failed_indices: Tuple[int, ...]
    put_attempts: int
    attempted_keys: int
    retried_keys: int
    #: Keys a retry stored after the first put had refused them.
    recovered_keys: int
    #: Keys a probe found stored after a put had reported them as failed.
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
    config_for: Optional[Callable[[Sequence[int]], Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> SaveRetryResult:
    """Put `keys`, retrying the refused ones a bounded number of times.

    The caller must keep the source pages and any staging slots immutable and
    owned until this returns; no address is kept past the call. Store calls
    stay synchronous.

    Args:
        store: The Mooncake store handle.
        keys: Keys to put.
        addresses: Source buffer addresses, one sequence per key.
        sizes: Source buffer sizes, one sequence per key.
        config_for: Maps the indices of the keys one put carries to the
            `ReplicateConfig` for that put, or None for the three-argument
            form. Called again for the subset each retry carries.
        clock: Monotonic clock, replaceable in tests.
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
            except Exception:  # noqa: BLE001 - the store boundary; an unknown key still needs a put
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
        put_args = (
            [keys[index] for index in unresolved],
            [addresses[index] for index in unresolved],
            [sizes[index] for index in unresolved],
        )
        config = config_for(unresolved) if config_for is not None else None
        if config is not None:
            put_args += (config,)
        try:
            result, bad = _statuses(store.batch_put_from_multi_buffers(*put_args), len(unresolved))
            invalid += bad
        except Exception:  # noqa: BLE001 - the store boundary; the keys stay unresolved
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
