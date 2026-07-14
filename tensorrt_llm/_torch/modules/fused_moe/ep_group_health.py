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

"""Committed EP data-plane membership for WideEP fault tolerance.

This module provides :class:`EPGroupHealth`, a process-local, thread-safe data
structure that records which Expert Parallel (EP) ranks the recovery coordinator
has committed as included vs. excluded from the data plane. It is consumed by:

  * AlltoAll communication backends (rank masking on dispatch / combine)
  * The host-side AlltoAll watchdog (read-only expected-peer snapshot)
  * The MoE load balancer (emergency-mask reconfiguration)
  * The model engine and PyExecutor (degraded health reporting)

Detected or suspected physical liveness is separate evidence. Higher-layer
coordination reconciles that evidence and commits membership; detectors and
telemetry consumers must not mutate this object to drive recovery.
"""

import threading
from typing import NamedTuple

# Default number of uint64 words for EPGroupHealth.get_mask_words().
# Matches the active_rank_mask ABI of the NVLink AlltoAll kernels
# (uint64_t[2]). Two words cover 128 ranks, sufficient for NVL72 with
# headroom.
EP_MASK_NUM_WORDS: int = 2


class EPGroupHealthSnapshot(NamedTuple):
    """Atomic snapshot of :class:`EPGroupHealth` state.

    Returned by :meth:`EPGroupHealth.snapshot`; all four fields reflect a
    single point in time (a single internal lock acquisition). Use this when
    the consumer needs a coherent view across multiple fields rather than
    calling individual accessors back-to-back (which can observe torn state
    if a mutator runs between calls).
    """

    # Active-rank bitmask (bit i set iff rank i is active).
    mask: int
    # Number of ranks currently marked active.
    active_count: int
    # Immutable snapshot of failed ranks.
    failed_ranks: frozenset[int]
    # Monotonic counter; bumps only on effective state changes.
    generation: int


class EPGroupHealth:
    """Thread-safe health tracker for the ranks of an EP group.

    Internally backed by an arbitrary-precision Python ``int`` bitmask
    (bit ``i`` set iff rank ``i`` is currently active). The kernel-side
    representation as a fixed-width array of ``uint64`` words is exposed via
    :meth:`get_mask_words`; this matches the ``active_rank_mask`` parameter
    expected by the NVLink AlltoAll kernels.

    All read and write operations take an internal lock. Read operations that
    return collection types return defensive snapshots so the caller cannot
    mutate internal state.

    .. note::
       Observations across multiple individual accessors are **not** atomic
       (a mutator can run between calls). Consumers that need a coherent
       multi-field view should call :meth:`snapshot`, or compare a cached
       :attr:`generation` to detect concurrent changes.

    Args:
        moe_world_size: Total number of ranks in the MoE world. Must be ``> 0``.

    Raises:
        ValueError: If ``moe_world_size <= 0``.
    """

    def __init__(self, moe_world_size: int) -> None:
        """Initialize all MoE ranks as active."""
        if moe_world_size <= 0:
            raise ValueError(f"moe_world_size must be > 0, got {moe_world_size}")
        # _moe_world_size and _all_active_mask are set once here and never mutated;
        # they are therefore safe to read without holding _lock (e.g. in
        # _validate_rank, moe_world_size property).
        self._moe_world_size: int = moe_world_size
        self._all_active_mask: int = (1 << moe_world_size) - 1
        self._active_mask: int = self._all_active_mask
        self._active_count: int = moe_world_size
        self._failed_ranks: set[int] = set()
        self._generation: int = 0
        self._lock = threading.Lock()

    @property
    def moe_world_size(self) -> int:
        """Total number of ranks in the MoE world (immutable)."""
        return self._moe_world_size

    @property
    def generation(self) -> int:
        """Monotonic counter incremented on every effective state change.

        Idempotent calls (e.g. marking an already-failed rank as failed) do not
        bump the counter. Consumers that need to react to mask changes can
        cache the last-seen generation and compare on each iteration boundary
        instead of diffing the full mask.
        """
        # The lock is taken to ensure visibility of writes by other threads
        # without relying on GIL-based ordering (relevant for free-threaded
        # CPython, PEP 703 / 3.13+). The read itself is atomic either way.
        with self._lock:
            return self._generation

    def _validate_rank(self, rank: int) -> None:
        """Raise if ``rank`` is outside the MoE world."""
        if not 0 <= rank < self._moe_world_size:
            raise ValueError(f"rank must be in [0, {self._moe_world_size}), got {rank}")

    def mark_failed(self, rank: int) -> bool:
        """Mark ``rank`` as failed. Idempotent.

        Args:
            rank: Index of the rank to mark, in ``[0, moe_world_size)``.

        Returns:
            ``True`` if this call changed state (rank was previously active),
            ``False`` if the rank was already marked failed.

        Raises:
            ValueError: If ``rank`` is outside ``[0, moe_world_size)``.
        """
        self._validate_rank(rank)
        bit = 1 << rank
        with self._lock:
            if not self._active_mask & bit:
                return False
            self._active_mask &= ~bit
            self._active_count -= 1
            self._failed_ranks.add(rank)
            self._generation += 1
            return True

    def mark_active(self, rank: int) -> bool:
        """Mark ``rank`` as active. Idempotent.

        Used when a replacement rank rejoins the group after a failure.
        Higher layers may impose a "monotonic failure" policy (do not
        reactivate a failed rank until the process group has been
        reconstructed); this primitive does not enforce that policy on its own.

        Args:
            rank: Index of the rank to mark, in ``[0, moe_world_size)``.

        Returns:
            ``True`` if this call changed state (rank was previously failed),
            ``False`` if the rank was already marked active.

        Raises:
            ValueError: If ``rank`` is outside ``[0, moe_world_size)``.
        """
        self._validate_rank(rank)
        bit = 1 << rank
        with self._lock:
            if self._active_mask & bit:
                return False
            self._active_mask |= bit
            self._active_count += 1
            self._failed_ranks.discard(rank)
            self._generation += 1
            return True

    def is_active(self, rank: int) -> bool:
        """Return ``True`` iff ``rank`` is currently marked active.

        Raises:
            ValueError: If ``rank`` is outside ``[0, moe_world_size)``.
        """
        self._validate_rank(rank)
        with self._lock:
            return bool(self._active_mask & (1 << rank))

    def get_mask(self) -> int:
        """Return the active-rank bitmask as a Python ``int``.

        Bit ``i`` is set iff rank ``i`` is currently active.
        """
        with self._lock:
            return self._active_mask

    def get_mask_words(self, num_words: int = EP_MASK_NUM_WORDS) -> tuple[int, ...]:
        """Return the active-rank bitmask split into little-endian uint64 words.

        Suitable for passing to CUDA kernels that accept ``uint64_t[num_words]``.
        Word ``0`` covers ranks ``0..63``, word ``1`` covers ranks ``64..127``,
        etc. The default of two words covers the NVL72 case (72 ranks) with
        headroom for future expansion.

        Args:
            num_words: Number of 64-bit words to produce. Must be large enough
                to hold ``moe_world_size`` bits (``num_words * 64 >= moe_world_size``).

        Returns:
            A tuple of ``num_words`` non-negative ``int`` values, each in
            ``[0, 2**64)``.

        Raises:
            ValueError: If ``num_words`` is non-positive or too small to hold
                ``moe_world_size`` bits.
        """
        if num_words <= 0:
            raise ValueError(f"num_words must be > 0, got {num_words}")
        if num_words * 64 < self._moe_world_size:
            raise ValueError(
                f"num_words={num_words} cannot represent moe_world_size={self._moe_world_size}"
            )
        word_mask = (1 << 64) - 1
        with self._lock:
            mask = self._active_mask
        return tuple((mask >> (i * 64)) & word_mask for i in range(num_words))

    def get_active_count(self) -> int:
        """Return the number of ranks currently marked active."""
        with self._lock:
            return self._active_count

    def get_failed_ranks(self) -> frozenset[int]:
        """Return an immutable snapshot of the set of failed ranks."""
        with self._lock:
            return frozenset(self._failed_ranks)

    def all_active(self) -> bool:
        """Return ``True`` iff every rank in the group is currently active."""
        with self._lock:
            return self._active_mask == self._all_active_mask

    def snapshot(self) -> EPGroupHealthSnapshot:
        """Return an atomic snapshot of mask, active_count, failed_ranks, generation.

        All four fields reflect a single point in time (one lock acquisition).
        Prefer this over calling individual accessors back-to-back when the
        consumer needs a coherent multi-field view.
        """
        with self._lock:
            return EPGroupHealthSnapshot(
                mask=self._active_mask,
                active_count=self._active_count,
                failed_ranks=frozenset(self._failed_ranks),
                generation=self._generation,
            )

    def __len__(self) -> int:
        """Return the MoE world size (``moe_world_size``), not the active count.

        Provided for compatibility with code that treats this object as a
        sized container of ranks (alive or dead). Use :meth:`get_active_count`
        when you specifically want the surviving-rank count.
        """
        return self._moe_world_size

    def __repr__(self) -> str:
        """Return a concise debug representation of the current health state."""
        with self._lock:
            return (
                f"EPGroupHealth(moe_world_size={self._moe_world_size}, "
                f"active_count={self._active_count}, "
                f"failed_ranks={sorted(self._failed_ranks)}, "
                f"generation={self._generation})"
            )
