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

"""Backend-neutral ownership for allocator-generation KV cache leases."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from tensorrt_llm._torch.disaggregation.lifecycle import PhysicalDisposition


class AllocationLeaseValidationError(RuntimeError):
    """The copied transfer descriptors do not belong to the leased allocation."""


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", maxsplit=1)[-1]


class AllocationLease:
    """One exact V1 or V2 allocator lease and its immutable snapshot.

    Destruction deliberately has no release semantics. A caller may release
    this lease only with reusable physical evidence after its transfer session
    has closed, or with ``NOT_EXPOSED`` before any descriptor is published.
    """

    _RELEASED_SETTLEMENTS = frozenset({"RELEASED", "ALREADY_RELEASED"})

    def __init__(self, handle: object) -> None:
        snapshot = getattr(handle, "snapshot", None)
        if snapshot is None:
            raise TypeError("allocation lease handle does not expose an immutable snapshot")
        has_v1_blocks = hasattr(snapshot, "blocks")
        has_v2_ranges = hasattr(snapshot, "ranges")
        if has_v1_blocks == has_v2_ranges:
            raise TypeError(
                "allocation lease snapshot must be exactly one of V1 blocks or V2 ranges"
            )
        self._handle = handle
        self._is_v2 = has_v2_ranges
        self._lock = threading.Lock()
        self._settled = False
        self._last_settlement: object | None = None

    @classmethod
    def acquire(cls, manager: object, request_id: int) -> "AllocationLease":
        """Acquire before any mutable block-table descriptor is copied."""
        acquire = getattr(manager, "snapshot_and_lease", None)
        if not callable(acquire):
            raise TypeError("KV cache manager does not support allocation leases")
        return cls(acquire(request_id))

    @property
    def snapshot(self) -> object:
        return self._handle.snapshot

    @property
    def lease_id(self) -> int:
        return int(self.snapshot.lease_id)

    @property
    def identity(self) -> object:
        return self.snapshot.identity

    @property
    def settled(self) -> bool:
        with self._lock:
            return self._settled

    @property
    def last_settlement(self) -> object | None:
        with self._lock:
            return self._last_settlement

    @staticmethod
    def _pack_v1_beams(blocks: Sequence[object]) -> tuple[int, ...]:
        by_beam: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for block in blocks:
            by_beam[int(block.beam_index)].append(
                (int(block.block_index), int(block.primary_pool_index))
            )
        if not by_beam:
            return ()
        beam_indices = sorted(by_beam)
        if beam_indices != list(range(beam_indices[-1] + 1)):
            raise AllocationLeaseValidationError("V1 snapshot has non-contiguous beam indices")
        beams = [
            tuple(pool_index for _, pool_index in sorted(by_beam[beam_index]))
            for beam_index in beam_indices
        ]
        packed = list(beams[0])
        beam_zero_last = beams[0][-1] if beams[0] else None
        for beam in beams[1:]:
            if beam and beam[-1] != beam_zero_last:
                packed.append(beam[-1])
        return tuple(packed)

    def _validate_v1(
        self,
        copied_block_ids: Sequence[np.ndarray],
        group_window_sizes: Sequence[int | None],
        mamba_group_indices: frozenset[int],
    ) -> None:
        blocks_by_window: dict[int, list[object]] = defaultdict(list)
        for block in self.snapshot.blocks:
            blocks_by_window[int(block.window_size)].append(block)

        for group_index, copied in enumerate(copied_block_ids):
            if group_index in mamba_group_indices:
                if copied.size:
                    raise AllocationLeaseValidationError(
                        f"Mamba layer group {group_index} unexpectedly copied attention pages"
                    )
                continue
            window_size = group_window_sizes[group_index]
            if window_size is None:
                raise AllocationLeaseValidationError(
                    f"V1 attention layer group {group_index} has no allocator window identity"
                )
            expected = self._pack_v1_beams(blocks_by_window.get(window_size, ()))
            actual = tuple(int(block_id) for block_id in copied)
            if actual != expected:
                raise AllocationLeaseValidationError(
                    f"V1 layer group {group_index} copied {actual}, but lease "
                    f"{self.lease_id} owns {expected}"
                )

    def _validate_v2(
        self,
        copied_block_ids: Sequence[np.ndarray],
        mamba_group_indices: frozenset[int],
        mamba_state_index: int | None,
    ) -> None:
        ranges_by_group: dict[int, list[object]] = defaultdict(list)
        for allocation_range in self.snapshot.ranges:
            ranges_by_group[int(allocation_range.layer_group_id)].append(allocation_range)

        expected_mamba_indices: set[int] = set()
        for group_index, copied in enumerate(copied_block_ids):
            ranges = ranges_by_group.get(group_index, ())
            if group_index in mamba_group_indices:
                if copied.size:
                    raise AllocationLeaseValidationError(
                        f"Mamba layer group {group_index} unexpectedly copied attention pages"
                    )
                expected_mamba_indices.update(
                    int(allocation_range.ssm_page_index)
                    for allocation_range in ranges
                    if allocation_range.ssm_page_index is not None
                )
                continue

            beam_zero_ranges = sorted(
                (
                    allocation_range
                    for allocation_range in ranges
                    if int(allocation_range.beam_index) == 0
                    and allocation_range.ssm_page_index is None
                ),
                key=lambda allocation_range: int(allocation_range.block_begin),
            )
            expected = tuple(
                int(page_index)
                for allocation_range in beam_zero_ranges
                for page_index in allocation_range.page_indices
            )
            actual = tuple(int(block_id) for block_id in copied)
            if actual != expected:
                raise AllocationLeaseValidationError(
                    f"V2 layer group {group_index} copied {actual}, but lease "
                    f"{self.lease_id} owns {expected}"
                )

        if expected_mamba_indices:
            if len(expected_mamba_indices) != 1:
                raise AllocationLeaseValidationError(
                    "V2 snapshot contains different Mamba state pages across layer groups or beams"
                )
            expected_mamba_state_index = next(iter(expected_mamba_indices))
            if mamba_state_index != expected_mamba_state_index:
                raise AllocationLeaseValidationError(
                    f"copied Mamba state index {mamba_state_index} does not match "
                    f"lease {self.lease_id} index {expected_mamba_state_index}"
                )

    def validate_copied_descriptors(
        self,
        copied_block_ids: Sequence[np.ndarray],
        group_window_sizes: Sequence[int | None],
        mamba_group_indices: frozenset[int],
        mamba_state_index: int | None,
    ) -> None:
        """Verify mutable manager reads against the immutable leased snapshot."""
        if len(copied_block_ids) != len(group_window_sizes):
            raise ValueError("layer-group descriptor metadata has inconsistent lengths")
        if self._is_v2:
            self._validate_v2(
                copied_block_ids,
                mamba_group_indices,
                mamba_state_index,
            )
        else:
            self._validate_v1(
                copied_block_ids,
                group_window_sizes,
                mamba_group_indices,
            )

    def _backend_proof(self, disposition: PhysicalDisposition) -> object:
        if self._is_v2:
            from tensorrt_llm.runtime.kv_cache_manager_v2 import AllocationReuseProof

            return AllocationReuseProof(disposition.value)

        import tensorrt_llm.bindings

        return getattr(
            tensorrt_llm.bindings.internal.batch_manager.PhysicalDisposition,
            disposition.name,
        )

    def settle(self, disposition: PhysicalDisposition) -> bool:
        """Settle once, returning whether the allocator accepted release.

        Non-reusable evidence is intentionally not submitted as release
        authority. The exact handle remains live for quarantine and diagnosis.
        """
        if not disposition.is_reusable:
            return False
        with self._lock:
            if self._settled:
                return True
            settlement = self._handle.settle(self._backend_proof(disposition))
            self._last_settlement = settlement
            if _enum_name(settlement) in self._RELEASED_SETTLEMENTS:
                self._settled = True
                return True
            return False


__all__ = [
    "AllocationLease",
    "AllocationLeaseValidationError",
]
