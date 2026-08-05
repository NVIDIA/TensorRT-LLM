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
"""Bounded exact replay state for the native disaggregated protocol."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Generic, TypeVar
from uuid import UUID

_K = TypeVar("_K")
_V = TypeVar("_V")
_MISSING = object()


class ReplayCapacityError(RuntimeError):
    """Raised when exact replay safety cannot admit another identity."""


class ReplayConflictError(RuntimeError):
    """Raised when one exact identity is associated with conflicting state."""


def _validate_epoch(epoch: UUID) -> UUID:
    if not isinstance(epoch, UUID) or epoch.int == 0:
        raise ValueError("replay epoch must be a non-nil UUID")
    return epoch


def _validate_capacity(capacity: int) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("replay capacity must be a positive integer")
    return capacity


class BoundedExactReplaySet(Generic[_K]):
    """A set that never silently evicts an exact replay decision.

    Exhaustion latches a fail-stop fault. Existing identities remain queryable
    and idempotent additions remain accepted, but a new identity cannot be
    admitted until the owning endpoint rotates to a different incarnation.
    """

    def __init__(self, capacity: int, epoch: UUID) -> None:
        self._capacity = _validate_capacity(capacity)
        self._epoch = _validate_epoch(epoch)
        self._entries: set[_K] = set()
        self._fault: ReplayCapacityError | None = None
        self._lock = threading.RLock()

    @property
    def fault(self) -> ReplayCapacityError | None:
        with self._lock:
            return self._fault

    @property
    def epoch(self) -> UUID:
        with self._lock:
            return self._epoch

    def add(self, key: _K) -> bool:
        with self._lock:
            if key in self._entries:
                return False
            if self._fault is not None:
                raise self._fault
            if len(self._entries) >= self._capacity:
                self._fault = ReplayCapacityError(
                    f"exact replay set exhausted its {self._capacity}-record capacity"
                )
                raise self._fault
            self._entries.add(key)
            return True

    def rotate(self, epoch: UUID) -> bool:
        """Clear state only after the owning endpoint incarnation changes."""
        epoch = _validate_epoch(epoch)
        with self._lock:
            if epoch == self._epoch:
                return False
            self._entries.clear()
            self._fault = None
            self._epoch = epoch
            return True

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def __iter__(self) -> Iterator[_K]:
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __eq__(self, other: object) -> bool:
        with self._lock:
            if isinstance(other, BoundedExactReplaySet):
                return self._entries == set(other)
            if isinstance(other, set):
                return self._entries == other
            return NotImplemented


class BoundedExactReplayMap(MutableMapping[_K, _V], Generic[_K, _V]):
    """A bounded exact mapping with conflict detection and epoch rotation."""

    def __init__(self, capacity: int, epoch: UUID) -> None:
        self._capacity = _validate_capacity(capacity)
        self._epoch = _validate_epoch(epoch)
        self._entries: dict[_K, _V] = {}
        self._fault: ReplayCapacityError | None = None
        self._lock = threading.RLock()

    @property
    def fault(self) -> ReplayCapacityError | None:
        with self._lock:
            return self._fault

    @property
    def epoch(self) -> UUID:
        with self._lock:
            return self._epoch

    def put_exact(self, key: _K, value: _V) -> bool:
        """Insert one immutable association, rejecting conflicting replays."""
        with self._lock:
            existing = self._entries.get(key, _MISSING)
            if existing is not _MISSING:
                if existing != value:
                    raise ReplayConflictError(
                        "exact replay identity is associated with conflicting state"
                    )
                return False
            if self._fault is not None:
                raise self._fault
            if len(self._entries) >= self._capacity:
                self._fault = ReplayCapacityError(
                    f"exact replay map exhausted its {self._capacity}-record capacity"
                )
                raise self._fault
            self._entries[key] = value
            return True

    def get(self, key: _K, default: _V | None = None) -> _V | None:
        with self._lock:
            return self._entries.get(key, default)

    def __getitem__(self, key: _K) -> _V:
        with self._lock:
            return self._entries[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        self.put_exact(key, value)

    def __delitem__(self, key: _K) -> None:
        with self._lock:
            del self._entries[key]

    def pop_exact(self, key: _K, value: _V) -> bool:
        """Remove only the exact association observed by the caller."""
        with self._lock:
            existing = self._entries.get(key, _MISSING)
            if existing is _MISSING:
                return False
            if existing is not value and existing != value:
                raise ReplayConflictError("exact replay identity changed before retirement")
            self._entries.pop(key)
            return True

    def items_snapshot(self) -> tuple[tuple[_K, _V], ...]:
        with self._lock:
            return tuple(self._entries.items())

    def rotate(self, epoch: UUID) -> bool:
        """Clear state only after the owning endpoint incarnation changes."""
        epoch = _validate_epoch(epoch)
        with self._lock:
            if epoch == self._epoch:
                return False
            self._entries.clear()
            self._fault = None
            self._epoch = epoch
            return True

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def __iter__(self) -> Iterator[_K]:
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __eq__(self, other: object) -> bool:
        with self._lock:
            if isinstance(other, BoundedExactReplayMap):
                return self._entries == dict(other.items_snapshot())
            if isinstance(other, Mapping):
                return self._entries == dict(other)
            return NotImplemented
