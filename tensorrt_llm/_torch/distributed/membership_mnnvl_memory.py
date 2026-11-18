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
"""MNNVL memory that can be mapped again when the active ranks change.

Each rank backs its own row of an allocation with physical pages, and every peer maps that row.
A failed rank's process may still write into the rows it maps. ``remap`` maps the allocation again
among the active ranks at the same virtual addresses and can give rows new physical pages. Late
writes then land on the old pages, which no active rank maps, and captured CUDA graphs keep their
addresses. At a membership change every active rank calls ``rows_needing_fresh_pages`` and then
``remap`` with its result. Both calls are collective over a ``StoreGroup`` of the allocation's
members, whose gathers return None for an inactive member.
"""

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch

from tensorrt_llm.logger import logger

from .mnnvl_memory import (
    MembershipComm,
    MnnvlMappingError,
    MnnvlMemory,
    MnnvlPeerHandleError,
    _MnnvlAllocationState,
)

if TYPE_CHECKING:
    from .membership import StoreGroup


def comm_members(comm: Any) -> tuple[int, ...]:
    """Return the global ranks of ``comm``'s members.

    A ``MembershipComm`` lists them. Any other communicator is taken as the ranks from 0 in order.
    """
    if isinstance(comm, MembershipComm):
        return tuple(comm.membership)
    return tuple(range(comm.Get_size()))


class _MappingFailure(NamedTuple):
    """A mapping failure that a rank reports to its peers in the outcome exchange of a remap."""

    error: str
    # Global ranks whose handles this rank could not import.
    failed_peers: list[int]


def _mapping_outcome(failure: Exception, rank: int) -> _MappingFailure | tuple[()]:
    """Return what global rank ``rank`` reports for ``failure``, or ``()`` if it is another rank's."""
    if isinstance(failure, MnnvlMappingError) and rank not in failure.ranks:
        return ()
    peers = failure.ranks if isinstance(failure, MnnvlPeerHandleError) else ()
    return _MappingFailure(f"{type(failure).__name__}: {failure}", list(peers))


def _raise_reported_failures(
    what: str, outcomes: dict[int, Any], local_failure: Optional[Exception]
) -> None:
    """Raise the same error on every active rank for the failures the active ranks reported.

    ``outcomes`` maps each global rank to its outcome: ``()`` for success, a
    :class:`_MappingFailure`, or None for an inactive rank.
    """
    failures = [(rank, reported) for rank, reported in outcomes.items() if reported]
    if not failures:
        if local_failure is not None:
            raise local_failure
        return
    message = f"{what} failed on active ranks:\n" + "\n".join(
        f"rank {rank}: {text}" for rank, (text, _) in failures
    )
    failed_peers = [peer for _, (_, peers) in failures for peer in peers]
    if failed_peers:
        raise MnnvlPeerHandleError(failed_peers, message) from local_failure
    raise MnnvlMappingError([rank for rank, _ in failures], message) from local_failure


class MembershipMnnvlMemory(MnnvlMemory):
    """MNNVL memory that can be mapped again among the active ranks at the same addresses."""

    # Map the rows of inactive ranks on one local scratch allocation instead of leaving them
    # unmapped, for users whose copies span every row.
    absent_rows_on_scratch: bool = False
    # Communicator indices of the rows mapped on the local scratch allocation.
    _scratch_row_indices: frozenset[int] = frozenset()

    @property
    def local_device_bytes(self) -> int:
        """Return the bytes of device memory this rank's row of the allocation takes."""
        return type(self).allocated_map[self.ptr].aligned_size

    def rows_needing_fresh_pages(self, comm: "StoreGroup") -> frozenset[int]:
        """Return every row if a former member may still write into the allocation, else no row.

        The answer is all or nothing. The pages stay only if every rank that maps them is active
        and maps them with the same ranks. This call is collective over ``comm``.
        """
        record = self._record_of(comm)
        mapped_with = frozenset()
        if record.state is _MnnvlAllocationState.MAPPED:
            mapped_with = frozenset(
                rank
                for index, (rank, handle) in enumerate(
                    zip(record.comm_membership, record.mem_handles)
                )
                if handle is not None and index not in self._scratch_row_indices
            )
        views = dict(zip(record.comm_membership, comm.allgather(mapped_with)))
        # An active rank that maps no row, or maps the row of a rank that is inactive or does not
        # map it back, means that a former member may map the pages.
        stale = any(
            not view or any(views[peer] is None or rank not in views[peer] for peer in view)
            for rank, view in views.items()
            if view is not None
        )
        return frozenset(record.comm_membership) if stale else frozenset()

    def remap(self, comm: "StoreGroup", fresh_rows: frozenset[int]) -> None:
        """Map the allocation again among the active members of ``comm`` at the same addresses.

        The rows of the global ranks in ``fresh_rows`` get new physical pages, so late writes of a
        former member land in pages that no active rank maps. The other rows keep their pages,
        which is safe only when :meth:`rows_needing_fresh_pages` returned no row. This call is
        collective over ``comm``. Every active rank either maps the allocation or raises the same
        error type.

        Raises:
            MnnvlPeerHandleError: Some active rank could not import a peer's handle.
            MnnvlMappingError: Mapping failed on some active ranks for another reason.
            RuntimeError: This rank's row is unmapped but not in ``fresh_rows``, or a kept row's
                rank is not active.
        """
        start_s = time.perf_counter()
        cls = type(self)
        record = self._record_of(comm)
        own_row_fresh = record.comm_membership[record.comm_rank] in fresh_rows
        if record.state is _MnnvlAllocationState.MAPPED:
            drop = [
                rank in fresh_rows or index in self._scratch_row_indices
                for index, rank in enumerate(record.comm_membership)
            ]
            if any(drop):
                torch.cuda.synchronize()
                try:
                    cls._unmap_and_release_handles(
                        replace(
                            record,
                            mem_handles=[
                                h if d else None for h, d in zip(record.mem_handles, drop)
                            ],
                        )
                    )
                except RuntimeError:
                    record.state = _MnnvlAllocationState.BROKEN
                    raise
                record.mem_handles = [None if d else h for h, d in zip(record.mem_handles, drop)]
        elif record.state is not _MnnvlAllocationState.UNMAPPED:
            raise RuntimeError(f"Cannot remap MNNVL allocation in {record.state.value} state")
        elif not own_row_fresh:
            raise RuntimeError("This rank's row of an unmapped MNNVL allocation needs fresh pages")
        kept = list(record.mem_handles)
        handles, outcome, local_failure = None, (), None

        def release_after_failure() -> None:
            """Unmap and release every row, and mark the allocation broken if that fails too."""
            if handles is not None:
                record.mem_handles = handles
            try:
                cls._unmap_and_release_handles(record)
            except RuntimeError as cleanup_error:
                logger.error(
                    f"Failed to release the pages of a failed MNNVL remap: {cleanup_error}"
                )
                record.state = _MnnvlAllocationState.BROKEN
                return
            record.state = _MnnvlAllocationState.UNMAPPED

        # The handle exchange inside _create_and_map_handles is followed by an outcome exchange, so
        # that every active rank raises the same error if any of them failed to map.
        try:
            handles = cls._create_and_map_handles(
                comm,
                record.aligned_size,
                record.start_address,
                record.rank_stride,
                record.address_offset,
                absent_rows_on_scratch=cls.absent_rows_on_scratch,
                mapped_handles=kept,
                members=record.comm_membership,
            )
            if own_row_fresh:
                self._reset_own_row()
        except Exception as failure:
            # A failed handle exchange leaves the peers unreachable, and an error other than a
            # RuntimeError is a bug. Neither goes to the outcome exchange.
            exchange_failed = isinstance(comm, MembershipComm) and comm.exchange_failed
            if exchange_failed or not isinstance(failure, RuntimeError):
                release_after_failure()
                raise
            local_failure = failure
            outcome = _mapping_outcome(failure, record.comm_membership[record.comm_rank])
        try:
            outcomes = dict(zip(record.comm_membership, comm.allgather(outcome)))
            _raise_reported_failures("MNNVL remap", outcomes, local_failure)
        except BaseException:
            release_after_failure()
            raise
        record.mem_handles = handles
        record.state = _MnnvlAllocationState.MAPPED
        self._scratch_row_indices = frozenset(
            index
            for index, (rank, handle) in enumerate(zip(record.comm_membership, handles))
            if handle is not None and outcomes[rank] is None
        )
        active = sum(reported is not None for reported in outcomes.values())
        # Every active row except the kept rows and this rank's own new row.
        imported = active - sum(handle is not None for handle in kept) - int(own_row_fresh)
        fresh = sum(outcomes[rank] is not None for rank in fresh_rows)
        logger.debug(
            f"MNNVL remap of {cls.__name__} at {self.ptr:#x} among {active} active ranks: "
            f"{fresh} rows on fresh pages, imported {imported} rows in "
            f"{(time.perf_counter() - start_s) * 1e3:.1f} ms"
        )

    def _record_of(self, comm: "StoreGroup") -> Any:
        """Return this allocation's record after checking that ``comm`` has its members.

        Raises:
            RuntimeError: ``comm`` has other members, orders them differently or is another
                member's.
        """
        record = type(self).allocated_map[self.ptr]
        members = comm_members(comm)
        if (members, comm.Get_rank()) != (record.comm_membership, record.comm_rank):
            raise RuntimeError(
                f"Rank {record.comm_rank} of the MNNVL allocation of ranks {record.comm_membership} "
                f"cannot remap it as rank {comm.Get_rank()} of ranks {members}"
            )
        return record

    def _reset_own_row(self) -> None:
        """Initialize this rank's row after :meth:`remap` gave it new pages.

        The default leaves the new pages as they are.
        """
