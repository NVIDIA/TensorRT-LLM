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

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._core._kv_cache_manager import KVCacheManager


@dataclass(slots=True, frozen=True)
class AllocationIdentity:
    """Identity of one concrete allocation owned by one allocator."""

    allocator_domain_id: str
    request_id: int | None
    allocation_generation: int

    def __post_init__(self) -> None:
        if not self.allocator_domain_id:
            raise ValueError("allocator_domain_id must not be empty")
        if self.allocation_generation <= 0:
            raise ValueError("allocation_generation must be positive")


@dataclass(slots=True, frozen=True)
class AllocationRange:
    """One immutable contiguous segment of exact transferable GPU pages.

    An attention allocation may contain several ranges for one beam and layer
    group when SWA has intentionally released stale pages. Attention ranges
    contain one pinned GPU page index per represented logical block. SSM ranges
    contain one pinned state-page index and no attention page indices.
    Scratch-backed, unexpectedly unallocated, and otherwise unpinned pages are
    never represented by a range.
    """

    layer_group_id: int
    beam_index: int
    block_begin: int
    block_end: int
    page_indices: tuple[int, ...]
    ssm_page_index: int | None = None

    def __post_init__(self) -> None:
        if self.layer_group_id < 0:
            raise ValueError("layer_group_id must be non-negative")
        if self.beam_index < 0:
            raise ValueError("beam_index must be non-negative")
        if self.block_begin < 0 or self.block_end < self.block_begin:
            raise ValueError("invalid block range")
        if self.ssm_page_index is not None:
            if self.ssm_page_index < 0:
                raise ValueError("ssm_page_index must identify a valid GPU page")
            if self.page_indices:
                raise ValueError("SSM ranges must not contain attention page indices")
        else:
            if len(self.page_indices) != self.block_end - self.block_begin:
                raise ValueError("page_indices must cover the complete block range")
            if any(page_index < 0 for page_index in self.page_indices):
                raise ValueError("page_indices must identify valid GPU pages")


@dataclass(slots=True, frozen=True)
class AllocationLeaseSnapshot:
    """Immutable descriptors borrowed by one allocation lease."""

    lease_id: int
    identity: AllocationIdentity
    ranges: tuple[AllocationRange, ...]

    def __post_init__(self) -> None:
        if self.lease_id <= 0:
            raise ValueError("lease_id must be positive")


class AllocationReuseProof(str, enum.Enum):
    """Allocator-local proof supplied when settling an allocation lease.

    This enum intentionally mirrors the backend-neutral physical disposition
    values without importing the PyTorch disaggregation layer into the runtime
    allocator. The lifecycle integration must map by value explicitly.
    """

    NOT_EXPOSED = "NOT_EXPOSED"
    ACTIVE = "ACTIVE"
    QUIESCING = "QUIESCING"
    QUIESCED_SUCCESS = "QUIESCED_SUCCESS"
    QUIESCED_FAILURE = "QUIESCED_FAILURE"
    IN_DOUBT = "IN_DOUBT"

    @property
    def is_reusable(self) -> bool:
        """Whether this proof authorizes allocator reuse."""
        return self in (
            AllocationReuseProof.NOT_EXPOSED,
            AllocationReuseProof.QUIESCED_SUCCESS,
            AllocationReuseProof.QUIESCED_FAILURE,
        )


class LeaseSettlement(str, enum.Enum):
    """Result of an idempotent allocation-lease settlement."""

    RELEASED = "RELEASED"
    ALREADY_RELEASED = "ALREADY_RELEASED"
    IN_DOUBT = "IN_DOUBT"
    NOT_QUIESCED = "NOT_QUIESCED"
    STALE_GENERATION = "STALE_GENERATION"
    NOT_FOUND = "NOT_FOUND"


@dataclass(slots=True, frozen=True)
class AllocationLeaseHandle:
    """Explicit lease handle whose garbage collection never releases memory."""

    snapshot: AllocationLeaseSnapshot
    _manager: "KVCacheManager"

    def settle(self, proof: AllocationReuseProof) -> LeaseSettlement:
        """Settle this lease using allocator-recognized quiescence evidence."""
        return self._manager.settle_allocation_lease(
            self.snapshot.lease_id,
            self.snapshot.identity,
            proof,
        )


__all__ = [
    "AllocationIdentity",
    "AllocationLeaseHandle",
    "AllocationLeaseSnapshot",
    "AllocationRange",
    "AllocationReuseProof",
    "LeaseSettlement",
]
