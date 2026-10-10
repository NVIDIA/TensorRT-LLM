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
"""The public types and protocols of the lender. Numpy and the standard library only, so any
side can import them without a live KV cache manager."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ...llm_request import LlmRequest

# A row's name: a fixed-length opaque key.
_NAME_BYTES = 54


def _freeze(array: np.ndarray) -> np.ndarray:
    """``array`` marked read-only (a view when it is not already)."""
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def _positive(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


@dataclass(frozen=True)
class StagingOptions:
    """Staging in whole fetches: ``max_fetches`` of ``fetch_tokens`` tokens, capped by ``max_bytes``
    but never below one fetch, which always fits. A capacity budget, not concurrency: a lease takes
    one contiguous run per pool group, and holes released leases leave can make it wait."""

    fetch_tokens: int
    max_fetches: int = 1
    max_bytes: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "fetch_tokens", _positive("fetch_tokens", self.fetch_tokens))
        object.__setattr__(self, "max_fetches", _positive("max_fetches", self.max_fetches))
        if self.max_bytes is not None:
            object.__setattr__(self, "max_bytes", _positive("max_bytes", self.max_bytes))


@dataclass(frozen=True)
class Part:
    """One device pool group's staging host region: ``nbytes`` bytes at ``address``, holding
    ``slots`` slots of ``slot_bytes`` bytes, one row each. Instances laid out alike give it the
    same ``name``."""

    name: str
    address: int
    nbytes: int
    slot_bytes: int
    slots: int


@dataclass(frozen=True, eq=False)
class GroupRun:
    """One layer group's read-only rows: row ``i`` is block ``ordinals[i]``. Staging rows also
    carry an opaque 54-byte name (a store key: ``names[i].tobytes()``), their slot's address and
    their part's index in ``StagingLender.parts``; in-place rows carry none of the three."""

    layer_group: int
    ordinals: np.ndarray
    names: Optional[np.ndarray] = None
    addresses: Optional[np.ndarray] = None
    part: Optional[int] = None

    def __post_init__(self):
        group = self.layer_group
        if isinstance(group, bool) or not isinstance(group, numbers.Integral):
            raise TypeError(f"layer_group must be an integer, got {type(group).__name__}")
        if self.layer_group < 0:
            raise ValueError(f"negative layer group {self.layer_group}")
        object.__setattr__(self, "layer_group", int(self.layer_group))
        ordinals = _freeze(np.ascontiguousarray(self.ordinals, dtype=np.int64))
        if ordinals.ndim != 1:
            raise ValueError(f"ordinals must be one-dimensional, got shape {ordinals.shape}")
        object.__setattr__(self, "ordinals", ordinals)
        placed = (self.names is not None, self.addresses is not None, self.part is not None)
        if any(placed) and not all(placed):
            raise ValueError("names, addresses and part are all given or all None")
        if not all(placed):
            return
        names = np.asarray(self.names)
        if names.dtype != np.uint8 or names.shape != (len(ordinals), _NAME_BYTES):
            raise ValueError(
                f"names must be uint8 ({len(ordinals)}, {_NAME_BYTES}), "
                f"got {names.dtype} {names.shape}"
            )
        addresses = np.asarray(self.addresses)
        if not np.issubdtype(addresses.dtype, np.integer) or addresses.shape != ordinals.shape:
            raise ValueError(
                f"addresses must be integers of shape {ordinals.shape}, "
                f"got {addresses.dtype} {addresses.shape}"
            )
        if isinstance(self.part, bool) or not isinstance(self.part, numbers.Integral):
            raise TypeError(f"part must be an integer, got {type(self.part).__name__}")
        if self.part < 0:
            raise ValueError(f"negative part {self.part}")
        object.__setattr__(self, "names", _freeze(np.ascontiguousarray(names)))
        object.__setattr__(
            self, "addresses", _freeze(np.ascontiguousarray(addresses, dtype=np.int64))
        )
        object.__setattr__(self, "part", int(self.part))

    def __len__(self) -> int:
        return int(self.ordinals.shape[0])

    def select(self, mask: np.ndarray) -> GroupRun:
        """The rows where the boolean ``mask`` (one entry per row) is true, in order."""
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.ordinals.shape:
            raise ValueError(
                f"mask must be bool of shape {self.ordinals.shape}, got {mask.dtype} {mask.shape}"
            )
        return GroupRun(
            self.layer_group,
            self.ordinals[mask],
            None if self.names is None else self.names[mask],
            None if self.addresses is None else self.addresses[mask],
            self.part,
        )


@dataclass(frozen=True, eq=False)
class RegionView:
    """What a ready lease lends: at most one run per layer group. Backends may read its arrays on
    their own threads until the lease is released."""

    runs: Tuple[GroupRun, ...]

    def __post_init__(self):
        runs = tuple(self.runs)
        seen = set()
        for run in runs:
            if not isinstance(run, GroupRun):
                raise TypeError(f"runs hold GroupRun, got {type(run).__name__}")
            if run.layer_group in seen:
                raise ValueError(f"layer group {run.layer_group} appears twice")
            seen.add(run.layer_group)
        object.__setattr__(self, "runs", runs)

    @property
    def num_rows(self) -> int:
        """Rows over all runs."""
        return sum(len(run) for run in self.runs)

    def row_masks(self, value: bool = False) -> Tuple[np.ndarray, ...]:
        """One writable boolean array per run, filled with ``value``: the shape ``mark_arrived``
        takes."""
        return tuple(np.full(len(run), bool(value), dtype=bool) for run in self.runs)


class Readiness(NamedTuple):
    """Resume at any ``p`` with ``restart_floor <= p <= usable_until``; ranks take the largest floor
    and the smallest end, and an empty interval means drop the cache and compute from 0. Below the
    floor a sliding window has released earlier blocks, so resuming there cannot continue."""

    usable_until: int
    restart_floor: int


@runtime_checkable
class Lease(Protocol):
    """One lent range: poll until the view or ``failure``, mark a write once, release. Its methods
    run only on the manager's thread; a backend's own threads read the view, access the memory it
    points to, call no lease or lender method and tell the holder through their own channel."""

    def poll(self) -> Optional[RegionView]:
        """Does pending work. The view once ready (the same object each time); ``None`` while
        pending and for good once failed. ``RuntimeError`` after release."""
        ...

    @property
    def failure(self) -> Optional[str]:
        """Why the lease will never be ready, once that is known; for logs."""
        ...

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """Write leases, once, after ready, before or after release: one boolean mask per run, true
        for rows that arrived whole. Required for staging writes, whose marked rows alone reach the
        request's pages, in one batch: split long fetches. Optional in place; it checks shapes."""
        ...

    def release(self) -> None:
        """The backend has stopped touching the lent memory. Legal in every state; later calls do
        nothing."""
        ...


@runtime_checkable
class PartsHold(Protocol):
    """A hold still open at the manager's shutdown keeps the staging memory until the process exits.
    Take one before registering ``StagingLender.parts``; release it on the manager's thread once
    deregistration is confirmed. The lender holds it, so dropping it unreleased keeps the memory."""

    def release(self) -> None:
        """The backend can no longer reach the parts. Runs only on the manager's thread, like every
        lender and lease method; legal in every state, later calls do nothing."""
        ...


# TODO: a naming entry that allocates nothing, for a flow that looks up remote hits before it
# reserves pages for them; a pure addition beside lend_read and lend_write.
@runtime_checkable
class StagingLender(Protocol):
    """Relays whole blocks through host staging slots, first come, first served. Lender and lease
    methods run only on the manager's thread; a backend's own threads read a view, access the memory
    it points to, call no lease or lender method, and tell the holder through their own channel."""

    @property
    def parts(self) -> Tuple[Part, ...]:
        """The staging host regions, one per device pool group, fixed for the lender's life and
        registrable once. Freed at the manager's shutdown unless an unreleased lease, an unreleased
        hold or a slot lost to a failed copy keeps them until the process exits."""
        ...

    def hold_parts(self) -> PartsHold:
        """A new hold on the staging memory, taken before registering ``parts``. After the
        manager's shutdown the hold is inert: the memory was freed or kept then."""
        ...

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """A copy of the committed blocks ``[start, end)``; ``ValueError`` for bad bounds, an end
        past the committed tokens or a misfit range. Its outcome is this rank's own, failed at the
        call or later on this rank's state; the caller combines every rank's outcome."""
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Empty slots for blocks ``[start, end)``, the cache grown to ``end``. ``ValueError``: bad
        bounds, a start in the committed blocks, an end below a window's history, a misfit range,
        an unsettled fetch. Fails as ``lend_read``, also on no free pages or SWA scratch reuse."""
        ...

    def readiness(self, request: LlmRequest) -> Optional[Readiness]:
        """``None`` while a fetch into the request is unsettled, else where it may resume. Copies
        queue on the manager's stream, serial with the forward on the GPU; with page-locked staging
        no lender call waits for them on the CPU. ``ValueError`` if the request has no KV cache."""
        ...


@runtime_checkable
class InPlaceLender(Protocol):
    """Lends a request's own device pages, addressed by the caller's own page-table code. Lender and
    lease methods run only on the manager's thread; a backend's own threads access the lent memory,
    call no lease or lender method, and tell the holder through their own channel."""

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """The paged blocks ``[start, end)`` touches, ready at once; ``ValueError`` for a bad range.
        Its outcome is this rank's own; the caller combines every rank's outcome, and ensures work
        the manager's stream queued that still writes those pages completed."""
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """As ``lend_read`` for writing; a block without a page fails it too, and all work the
        manager's stream queued for those pages has completed. While either is lent the caller keeps
        the request unscheduled, unsuspended, unshrunk and its window still, and owns validity."""
        ...
