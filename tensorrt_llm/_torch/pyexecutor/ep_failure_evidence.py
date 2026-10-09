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

"""Runtime-independent failure evidence, distinct from committed EP membership.

Evidence reconciliation is not survivor consensus or a quiescence proof. The
membership authority must fence failed producers, qualify the data plane and
authorize the next execution generation separately. This module has no MPI,
Ray, CUDA or backend imports.
"""

from __future__ import annotations

import operator
import threading
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

FailureDetectedCallback = Callable[[int, int, float], None]
"""Handoff of failed logical rank, reporting logical rank and local monotonic time.

The timestamp is local observation time, not a cross-host clock or ordering
token. The owning adapter must associate this evidence with its execution and
producer identities before a membership authority consumes it.
"""


def _require_integer(value: object, name: str) -> int:
    """Normalize one integer-like value while rejecting booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from error


@dataclass(frozen=True)
class FailureEvidenceSnapshot:
    """Atomic evidence for one immutable transport lifetime.

    Attributes:
        failed_ranks: Logical EP ranks with observed failure evidence, not
            authority-committed membership removals.
        evidence_epoch: Local evidence version. This is not an execution
            generation, producer incarnation, or authorization token.
    """

    failed_ranks: frozenset[int]
    evidence_epoch: int


class FailureEvidenceState:
    """Thread-safe, process-local rank-failure evidence.

    This state is deliberately distinct from `EPGroupHealth`. Only its owning
    evidence transport records failures; consumers may inspect but must not
    mutate it. Evidence is monotonic within one immutable transport lifetime.

    Args:
        ep_size: Number of logical EP ranks represented by this state.

    Raises:
        TypeError: If `ep_size` is not an integer.
        ValueError: If `ep_size` is not positive.
    """

    def __init__(self, ep_size: int) -> None:
        ep_size = _require_integer(ep_size, "ep_size")
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        self._ep_size = ep_size
        self._failed_ranks: set[int] = set()
        self._evidence_epoch = 0
        self._lock = threading.Lock()

    @property
    def ep_size(self) -> int:
        """Number of logical ranks in this immutable transport lifetime."""
        return self._ep_size

    def _record_failure(self, rank: int) -> bool:
        """Record evidence on behalf of the owning transport.

        Args:
            rank: Logical EP rank in `[0, ep_size)`.

        Returns:
            Whether the failed-rank set changed.
        """
        rank = self._validate_rank(rank)
        with self._lock:
            if rank in self._failed_ranks:
                return False
            self._failed_ranks.add(rank)
            self._evidence_epoch += 1
            return True

    def has_failure(self, rank: int) -> bool:
        """Inspect failure evidence without changing membership.

        Args:
            rank: Logical EP rank in `[0, ep_size)`.

        Returns:
            Whether this rank has recorded failure evidence.
        """
        rank = self._validate_rank(rank)
        with self._lock:
            return rank in self._failed_ranks

    def has_failures(self) -> bool:
        """Return whether this transport lifetime contains failure evidence."""
        with self._lock:
            return bool(self._failed_ranks)

    def snapshot(self) -> FailureEvidenceSnapshot:
        """Return one coherent failed-rank set and local-version snapshot."""
        with self._lock:
            return FailureEvidenceSnapshot(
                failed_ranks=frozenset(self._failed_ranks),
                evidence_epoch=self._evidence_epoch,
            )

    def _validate_rank(self, rank: int) -> int:
        rank = _require_integer(rank, "rank")
        if not 0 <= rank < self._ep_size:
            raise ValueError(f"rank must be in [0, {self._ep_size}), got {rank}")
        return rank


class FailureEvidenceTransport(Protocol):
    """Detection-plane interface consumed by a runtime adapter.

    Implementations own independent evidence progress and bounded shutdown.
    This interface deliberately exposes no membership commit, fencing,
    communicator reconstruction or request-resume operation. A reconciled
    report is evidence of delivery only, never permission to resume execution.
    """

    @property
    @abstractmethod
    def last_error(self) -> BaseException | None:
        """First terminal evidence-transport error, or `None`."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start independent progress once; fail if startup cannot complete."""
        ...

    @abstractmethod
    def stop(self, timeout: float | None = None) -> None:
        """Stop progress within the implementation's bounded shutdown policy.

        Args:
            timeout: Positive finite bound in seconds, or the transport default.
        """
        ...

    @abstractmethod
    def report_detected_failure(self, failed_rank: int) -> bool:
        """Record and enqueue evidence without waiting for remote progress.

        Args:
            failed_rank: Logical EP rank in the transport's immutable rank space.

        Returns:
            Whether this report changed the local evidence state.
        """
        ...

    @abstractmethod
    def failure_detection_is_reconciled(self, failed_rank: int) -> bool:
        """Inspect report propagation without performing transport operations.

        Args:
            failed_rank: Logical EP rank whose report is being inspected.

        Returns:
            Whether the required recipients propagated that report. A terminal
            transport error must return `False`.
        """
        ...

    @abstractmethod
    def failure_evidence_is_reconciled(self) -> bool:
        """Return whether all local evidence propagated, failing closed on error."""
        ...
