# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data classes for DWDP Weight Buffer specifications and layout computation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Literal, Sequence, Tuple

import torch

# ``(start, end)``: one DWDP rank stores the routed experts ``[start, end)`` in rows
# ``0 .. end - start - 1`` of its expert weights.  ``ExpertPartition.ranges`` holds one per rank;
# adjacent ranges overlap only in a redundant partition.
PeerRange = Tuple[int, int]
# Indexed by peer DWDP rank; ``len(PeerRanges) == dwdp_size``.
PeerRanges = List[PeerRange]

# How an ``ExpertPartition`` was made; see its docstring.
PartitionKind = Literal["balanced", "uniform", "redundant", "capped", "custom"]


def lookup_owner(expert_id: int, peer_ranges: PeerRanges) -> int:
    """Return the lowest peer rank whose range contains ``expert_id``.

    Without overlap that is the one rank storing it.  For redundancy
    (overlapping ranges) the first-match policy picks the lowest-rank
    owner, matching the IPC implementation's behavior so reads of the
    same expert id are deterministic across peers.

    Raises:
        ValueError: If no peer's range contains ``expert_id``.  The ranges
            of an ``ExpertPartition`` cover every expert, so this never
            happens for them.
    """
    for peer_rank, (start, end) in enumerate(peer_ranges):
        if start <= expert_id < end:
            return peer_rank
    raise ValueError(f"expert_id={expert_id} not owned by any peer in peer_ranges={peer_ranges}")


_BALANCED_HINT = (
    "Leave num_experts_per_worker and num_prefetch_experts both at 0 for the balanced partition, "
    "which stores every expert once on any dwdp_size."
)


@dataclass(frozen=True)
class ExpertPartition:
    """Which routed experts each DWDP rank stores: one contiguous range per rank, every MoE layer.

    ``ranges[r] == (start, end)``: rank ``r`` stores experts ``[start, end)`` in rows
    ``0 .. end - start - 1`` of its expert weights, allocates exactly those rows and exports them
    in its handles.  Every rank computes the same partition (``from_config``) from
    ``DwdpConfig``, ``dwdp_size`` and the model's expert count; it is the only source of any
    rank's range for ``DwdpManager``, ``MoE._init_dwdp_expert_layout``, ``setup_dwdp`` and the
    transport.  The ranges ascend, the first starts at 0, the last ends at ``num_experts`` and
    none starts after the previous one ends, so every expert has an owner: ``lookup_owner``
    picks the lowest rank storing it.

    Kinds:

    * ``balanced`` (both config fields 0): with ``divmod(num_experts, dwdp_size) == (base,
      rem)``, ranks below ``rem`` store ``base + 1`` experts and the others ``base``, as
      ``_compute_ep_partition`` splits EP ranks.  No expert is stored twice.
    * ``uniform``, ``redundant`` and ``capped`` (``strided``, both fields set): rank ``r`` stores
      ``num_experts_per_worker`` experts from ``r * num_prefetch_experts``, capped at
      ``num_experts``.
    * ``custom`` (``from_counts``): contiguous ranges of given counts, for tests and tools.

    Attributes:
        num_experts: Routed experts of every MoE layer.
        ranges: ``(start, end)`` per DWDP rank.
        kind: How the ranges were made.
    """

    num_experts: int
    ranges: Tuple[PeerRange, ...]
    kind: PartitionKind

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"[DWDP] num_experts must be positive, got {self.num_experts}")
        if not self.ranges:
            raise ValueError("[DWDP] an expert partition needs at least one rank")
        prev_start, prev_end = 0, 0
        for rank, (start, end) in enumerate(self.ranges):
            if not 0 <= start < end <= self.num_experts:
                raise ValueError(
                    f"[DWDP] rank {rank} stores experts [{start}, {end}), which is empty or not "
                    f"within [0, {self.num_experts})"
                )
            if rank == 0 and start != 0:
                raise ValueError(
                    f"[DWDP] rank 0 stores experts [{start}, {end}); experts [0, {start}) have "
                    f"no owner"
                )
            if rank > 0 and (not prev_start < start <= prev_end or end <= prev_end):
                raise ValueError(
                    f"[DWDP] rank {rank} stores experts [{start}, {end}) after rank {rank - 1}'s "
                    f"[{prev_start}, {prev_end}); the ranges must ascend without a gap"
                )
            prev_start, prev_end = start, end
        if prev_end != self.num_experts:
            raise ValueError(
                f"[DWDP] the last rank's range ends at {prev_end}; experts "
                f"[{prev_end}, {self.num_experts}) have no owner"
            )

    @classmethod
    def balanced(cls, *, num_experts: int, dwdp_size: int) -> ExpertPartition:
        """Floor or ceil of ``num_experts / dwdp_size`` per rank, the larger ones on the low ranks.

        Raises:
            ValueError: ``num_experts < dwdp_size``: a rank would store no expert.
        """
        if dwdp_size <= 0 or num_experts < dwdp_size:
            raise ValueError(
                f"[DWDP] cannot split {num_experts} routed experts over dwdp_size={dwdp_size} "
                f"ranks: every rank must store at least one expert"
            )
        base, rem = divmod(num_experts, dwdp_size)
        ranges = tuple(
            (rank * base + min(rank, rem), (rank + 1) * base + min(rank + 1, rem))
            for rank in range(dwdp_size)
        )
        return cls(num_experts=num_experts, ranges=ranges, kind="balanced")

    @classmethod
    def strided(
        cls, *, num_experts: int, dwdp_size: int, size: int, stride: int
    ) -> ExpertPartition:
        """``size`` experts per rank, ``stride`` apart: the two strided ``DwdpConfig`` fields.

        ``size`` is ``num_experts_per_worker`` and ``stride`` is ``num_prefetch_experts``.  Rank
        ``r`` stores ``[r * stride, min(r * stride + size, num_experts))``.  Accepted:

        * ``uniform``: ``stride == size`` and ``dwdp_size * size == num_experts``.
        * ``redundant``: ``stride < size`` and ``(dwdp_size - 1) * stride + size ==
          num_experts``; adjacent ranks both store ``size - stride`` experts, which the lower
          rank serves.
        * ``capped``: ``stride == size`` and ``(dwdp_size - 1) * size < num_experts < dwdp_size *
          size``; the last rank stores the ``num_experts - (dwdp_size - 1) * size`` experts left
          and allocates only those rows.

        Raises:
            ValueError: Any other combination: a value that is not positive, a stride larger
                than the size (experts nobody stores), too few experts in total, overlapping
                ranges past ``num_experts``, or a last rank left without experts.
        """
        if size <= 0 or stride <= 0:
            raise ValueError(
                f"[DWDP] num_experts_per_worker ({size}) and num_prefetch_experts ({stride}) "
                f"must both be positive in a strided expert partition. {_BALANCED_HINT}"
            )
        if stride > size:
            raise ValueError(
                f"[DWDP] num_prefetch_experts ({stride}) must be <= num_experts_per_worker "
                f"({size}): a stride larger than the range size leaves experts nobody stores. "
                f"{_BALANCED_HINT}"
            )
        coverage = (dwdp_size - 1) * stride + size
        if coverage == num_experts:
            kind = "uniform" if stride == size else "redundant"
        elif stride == size and (dwdp_size - 1) * size < num_experts < coverage:
            kind = "capped"
        else:
            if coverage < num_experts:
                problem = f"so experts [{coverage}, {num_experts}) have no owner"
            elif stride < size:
                problem = f"past the {num_experts} routed experts, with overlapping ranges"
            else:
                problem = f"and the last rank would store none of the {num_experts} routed experts"
            raise ValueError(
                f"[DWDP] num_experts_per_worker={size}, num_prefetch_experts={stride} on "
                f"dwdp_size={dwdp_size}: the last rank's range ends at (dwdp_size - 1) * "
                f"num_prefetch_experts + num_experts_per_worker = {coverage}, {problem}. A strided "
                f"partition must end at {num_experts}, or use equal values and cut the last "
                f"range short at {num_experts}. {_BALANCED_HINT}"
            )
        ranges = tuple(
            (rank * stride, min(rank * stride + size, num_experts)) for rank in range(dwdp_size)
        )
        return cls(num_experts=num_experts, ranges=ranges, kind=kind)

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> ExpertPartition:
        """Contiguous ranges of ``counts[r]`` experts each, from expert 0 (tests and tools)."""
        ranges = []
        start = 0
        for count in counts:
            ranges.append((start, start + count))
            start += count
        return cls(num_experts=start, ranges=tuple(ranges), kind="custom")

    @classmethod
    def from_config(
        cls,
        *,
        num_experts: int,
        dwdp_size: int,
        num_experts_per_worker: int,
        num_prefetch_experts: int,
    ) -> ExpertPartition:
        """The partition ``DwdpConfig`` asks for, for a model with ``num_experts`` routed experts.

        Both fields 0: ``balanced``; otherwise ``strided``, which needs both positive.

        Raises:
            ValueError: The config gives no valid partition of ``num_experts`` experts over
                ``dwdp_size`` ranks.
        """
        if num_experts_per_worker == 0 and num_prefetch_experts == 0:
            return cls.balanced(num_experts=num_experts, dwdp_size=dwdp_size)
        return cls.strided(
            num_experts=num_experts,
            dwdp_size=dwdp_size,
            size=num_experts_per_worker,
            stride=num_prefetch_experts,
        )

    @property
    def dwdp_size(self) -> int:
        """Number of DWDP ranks."""
        return len(self.ranges)

    @property
    def peer_ranges(self) -> PeerRanges:
        """``ranges`` as the ``PeerRanges`` list ``lookup_owner`` and the copy plans take."""
        return list(self.ranges)

    def range_of(self, rank: int) -> PeerRange:
        """``(start, end)`` of the experts ``rank`` stores."""
        return self.ranges[rank]

    def count_of(self, rank: int) -> int:
        """Number of experts ``rank`` stores: the rows of its expert weights."""
        start, end = self.ranges[rank]
        return end - start

    def describe(self) -> str:
        """The kind and the per-rank counts, e.g. ``balanced, 8 x 75 + 4 x 74 of 896``.

        Overlapping ranges add the extra copies: ``redundant, 12 x 82 of 896, 88 duplicate
        copies``.
        """
        counts = [end - start for start, end in self.ranges]
        groups = " + ".join(
            f"{counts.count(count)} x {count}" for count in sorted(set(counts), reverse=True)
        )
        duplicates = sum(counts) - self.num_experts
        extra = f", {duplicates} duplicate copies" if duplicates else ""
        return f"{self.kind}, {groups} of {self.num_experts}{extra}"


def handle_geometry(expert_bytes: int, start: int, end: int, granularity: int) -> Tuple[int, int]:
    """``(phys_size, data_offset)`` of the handle that holds experts ``[start, end)`` of one weight.

    The handle spans the ``granularity`` pages the experts' bytes touch, and they start
    ``data_offset`` bytes into it.  A VMM handle maps only whole (offset 0 and its full size,
    whatever its type), and an imported handle carries no size: the exporter allocates exactly
    ``phys_size`` and every importer maps exactly ``phys_size``, computed from the exporter's
    range.  ``PageAlignedLayout.compute`` gives the same ``mnnvl_size`` and ``data_offset``.
    """
    start_bytes = start * expert_bytes
    page_start = _align_down(start_bytes, granularity)
    page_end = _align_up(end * expert_bytes, granularity)
    return page_end - page_start, start_bytes - page_start


@dataclass(frozen=True)
class WeightSpec:
    """Shape and dtype of one expert weight parameter.

    Attributes:
        num_experts: Total experts for this layer (e.g., 256).
        chunk_shape: Per-rank chunk shape (e.g., (64, 4096, 448)).
        full_shape: Full expert shape (e.g., (256, 4096, 448)).
        dtype: Data type of the weight tensor.
    """

    num_experts: int
    chunk_shape: Tuple[int, ...]
    full_shape: Tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if len(self.chunk_shape) == 0:
            raise ValueError("chunk_shape cannot be empty")
        if len(self.full_shape) == 0:
            raise ValueError("full_shape cannot be empty")
        if self.full_shape[0] != self.num_experts:
            raise ValueError(
                f"full_shape[0]={self.full_shape[0]} must equal num_experts={self.num_experts}"
            )

    @cached_property
    def expert_bytes(self) -> int:
        """Bytes per single expert (computed from full_shape and dtype)."""
        n = 1
        for d in self.full_shape[1:]:
            n *= d
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return n * element_size

    @cached_property
    def chunk_bytes(self) -> int:
        """Total bytes for the chunk (all local experts)."""
        n = 1
        for d in self.chunk_shape:
            n *= d
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return n * element_size

    @property
    def local_experts(self) -> int:
        """Number of local experts in the chunk."""
        return self.chunk_shape[0]


# Type alias for per-layer weight specs
LayerWeightSpecs = Dict[int, Dict[str, WeightSpec]]


@dataclass
class MnnvlHandleSet:
    """MNNVL handles produced by Transport, consumed by WeightBuffer.

    One handle per (layer_idx, weight_name). Each handle is a
    CUmemGenericAllocationHandle pointing to local physical memory
    containing this rank's expert chunk.

    Attributes:
        handles: Mapping from (layer_idx, weight_name) to handle integer.
        sizes: Mapping from (layer_idx, weight_name) to physical size in bytes.
    """

    handles: Dict[Tuple[int, str], int]
    sizes: Dict[Tuple[int, str], int]

    def __post_init__(self) -> None:
        if set(self.handles.keys()) != set(self.sizes.keys()):
            raise ValueError("handles and sizes must have the same keys")

    def get_handle(self, layer_idx: int, name: str) -> int:
        """Get handle for a specific (layer, weight) pair."""
        key = (layer_idx, name)
        if key not in self.handles:
            raise KeyError(f"No handle for {key}")
        return self.handles[key]

    def get_size(self, layer_idx: int, name: str) -> int:
        """Get physical size for a specific (layer, weight) pair."""
        key = (layer_idx, name)
        if key not in self.sizes:
            raise KeyError(f"No size for {key}")
        return self.sizes[key]

    @property
    def layer_indices(self) -> list[int]:
        """Get sorted list of unique layer indices."""
        return sorted(set(layer_idx for layer_idx, _ in self.handles.keys()))

    def weight_names(self, layer_idx: int) -> list[str]:
        """Get weight names for a specific layer."""
        return [name for (lidx, name) in self.handles.keys() if lidx == layer_idx]


@dataclass(frozen=True)
class EdgeInfo:
    """Page-alignment edge information for setup-time peer data fill.

    When the local expert data doesn't align to page boundaries, there are
    edge regions within the MNNVL pages that must be filled with peer data.

    Attributes:
        data_offset: Bytes from handle VA start to local data start.
        leading_edge: Bytes [page_start, local_start) - peer data region.
        trailing_edge: Bytes [local_end, page_end) - peer data region.
        page_start: Byte offset of first MNNVL page.
        page_end: Byte offset past last MNNVL page.
        expert_bytes: Bytes per single expert for this weight.
    """

    data_offset: int
    leading_edge: int
    trailing_edge: int
    page_start: int
    page_end: int
    expert_bytes: int

    def __post_init__(self) -> None:
        if self.data_offset < 0:
            raise ValueError(f"data_offset must be non-negative, got {self.data_offset}")
        if self.leading_edge < 0:
            raise ValueError(f"leading_edge must be non-negative, got {self.leading_edge}")
        if self.trailing_edge < 0:
            raise ValueError(f"trailing_edge must be non-negative, got {self.trailing_edge}")
        if self.page_end < self.page_start:
            raise ValueError(f"page_end={self.page_end} must be >= page_start={self.page_start}")


@dataclass(frozen=True)
class PageAlignedLayout:
    """Page-aligned layout for a specific (layer, weight) pair.

    This computes the VA layout for the composite buffer including:
    - Pre-remote region: experts before local_start
    - MNNVL region: local expert pages (may include edge bytes)
    - Post-remote region: experts after local_end

    Two granularities control alignment:

    - ``granularity`` (typically 2MB): the CUDA VMM page size used for MNNVL
      handle mapping.  ``page_start`` and ``page_end`` are aligned to this.
    - ``pool_granularity`` (typically 16MB = 8 * granularity): the pool page
      handle size.  ``pre_size`` and ``post_size`` are rounded UP to this so
      that each pool-backed region is an exact multiple of pool page handles,
      avoiding cuMemMap overlap between pool pages and the MNNVL handle.

    When ``pool_granularity > granularity``, ``pre_size`` is ``page_start``
    rounded up: the first ``pre_padding`` bytes of the reservation (before
    expert 0) are unused pool-backed space, and the tensor view must start at
    ``va_base + pre_padding`` so that expert indices line up with the correct
    physical memory.  Likewise the last ``post_padding`` bytes of the
    reservation are unused.  The MNNVL region abuts both pool-backed regions.

    Attributes:
        expert_bytes: Bytes per single expert.
        num_experts: Total number of experts for this layer.
        local_start: First local expert index (inclusive).
        local_end: Last local expert index (exclusive).
        granularity: CUDA VMM page granularity in bytes.
        pool_granularity: Pool page handle size in bytes (>= granularity,
            must be a multiple of granularity).
        page_start: align_down(local_start_bytes, granularity).
        page_end: align_up(local_end_bytes, granularity).
        pre_size: VA bytes for pool-backed pre-region (multiple of
            pool_granularity).
        mnnvl_size: VA bytes for MNNVL region (multiple of granularity).
        post_size: VA bytes for pool-backed post-region (multiple of
            pool_granularity).
        pre_padding: Unused bytes at the start of the pre-region (before
            expert 0) due to pool_granularity rounding (pre_size - page_start).
        post_padding: Extra bytes at end of post-region due to pool_granularity
            rounding.
        data_offset: local_start_bytes - page_start (same as leading_edge).
        leading_edge: Bytes before local data within first MNNVL page.
        trailing_edge: Bytes after local data within last MNNVL page.
        total_size: Total VA size for the composite buffer.
        handle_phys_size: Physical size of the MNNVL handle.
    """

    expert_bytes: int
    num_experts: int
    local_start: int
    local_end: int
    granularity: int
    pool_granularity: int
    page_start: int
    page_end: int
    pre_size: int
    mnnvl_size: int
    post_size: int
    pre_padding: int
    post_padding: int
    data_offset: int
    leading_edge: int
    trailing_edge: int
    total_size: int
    handle_phys_size: int

    @classmethod
    def compute(
        cls,
        expert_bytes: int,
        num_experts: int,
        local_start: int,
        local_end: int,
        granularity: int,
        handle_phys_size: int,
        pool_granularity: int | None = None,
    ) -> PageAlignedLayout:
        """Compute page-aligned layout for given parameters.

        Args:
            expert_bytes: Bytes per single expert.
            num_experts: Total number of experts.
            local_start: First local expert index (inclusive).
            local_end: Last local expert index (exclusive).
            granularity: CUDA VMM page granularity in bytes.
            handle_phys_size: Physical size of the MNNVL handle.
            pool_granularity: Pool page handle size in bytes.  Must be a
                positive multiple of ``granularity``.  Defaults to
                ``granularity`` (i.e. each pool page = one VMM page).

        Returns:
            PageAlignedLayout with all computed fields.

        Raises:
            ValueError: If parameters are invalid.
        """
        if local_start < 0 or local_end > num_experts or local_start >= local_end:
            raise ValueError(
                f"Invalid expert range: local_start={local_start}, "
                f"local_end={local_end}, num_experts={num_experts}"
            )
        if granularity <= 0 or (granularity & (granularity - 1)) != 0:
            raise ValueError(f"granularity must be a positive power of 2, got {granularity}")

        if pool_granularity is None:
            pool_granularity = granularity
        if pool_granularity <= 0 or pool_granularity % granularity != 0:
            raise ValueError(
                f"pool_granularity must be a positive multiple of granularity "
                f"({granularity}), got {pool_granularity}"
            )
        if (pool_granularity & (pool_granularity - 1)) != 0:
            raise ValueError(f"pool_granularity must be a power of 2, got {pool_granularity}")

        # Compute byte offsets
        local_start_bytes = local_start * expert_bytes
        local_end_bytes = local_end * expert_bytes
        total_expert_bytes = num_experts * expert_bytes

        # Page-align the local region to VMM granularity (for MNNVL handle mapping)
        page_start = _align_down(local_start_bytes, granularity)
        page_end = _align_up(local_end_bytes, granularity)

        # Edge bytes (within the MNNVL region)
        data_offset = local_start_bytes - page_start
        leading_edge = data_offset
        trailing_edge = page_end - local_end_bytes

        # MNNVL region size is the page-aligned span (unchanged by pool_granularity)
        mnnvl_size = page_end - page_start

        # Validate: mnnvl_size must not exceed the handle's physical size.
        # cuMemMap maps a handle only whole (offset 0, size == phys_size; any
        # handle type, else CUDA_ERROR_NOT_SUPPORTED), and the transport
        # allocates it with handle_geometry(), whose size is mnnvl_size.
        if mnnvl_size > handle_phys_size:
            raise ValueError(
                f"mnnvl_size ({mnnvl_size}) exceeds handle_phys_size "
                f"({handle_phys_size}). The Transport must allocate handles "
                f"with phys_size >= page_end - page_start. "
                f"page_start={page_start}, page_end={page_end}, "
                f"local_start={local_start}, local_end={local_end}, "
                f"expert_bytes={expert_bytes}, granularity={granularity}"
            )

        # Pre region: everything before the MNNVL region, rounded UP to
        # pool_granularity so each pool page maps cleanly without overlapping
        # the MNNVL handle.
        pre_size = _align_up(page_start, pool_granularity)
        pre_padding = pre_size - page_start

        # Post region: everything after the MNNVL region, rounded UP to
        # pool_granularity.
        post_size_raw = _align_up(total_expert_bytes, granularity) - page_end
        post_size = _align_up(post_size_raw, pool_granularity)
        post_padding = post_size - post_size_raw

        # Total VA size
        total_size = pre_size + mnnvl_size + post_size

        return cls(
            expert_bytes=expert_bytes,
            num_experts=num_experts,
            local_start=local_start,
            local_end=local_end,
            granularity=granularity,
            pool_granularity=pool_granularity,
            page_start=page_start,
            page_end=page_end,
            pre_size=pre_size,
            mnnvl_size=mnnvl_size,
            post_size=post_size,
            pre_padding=pre_padding,
            post_padding=post_padding,
            data_offset=data_offset,
            leading_edge=leading_edge,
            trailing_edge=trailing_edge,
            total_size=total_size,
            handle_phys_size=handle_phys_size,
        )

    def get_edge_info(self) -> EdgeInfo:
        """Get EdgeInfo for this layout."""
        return EdgeInfo(
            data_offset=self.data_offset,
            leading_edge=self.leading_edge,
            trailing_edge=self.trailing_edge,
            page_start=self.page_start,
            page_end=self.page_end,
            expert_bytes=self.expert_bytes,
        )

    @property
    def pre_pages(self) -> int:
        """Number of pool pages in the pre-remote region."""
        return self.pre_size // self.pool_granularity

    @property
    def mnnvl_pages(self) -> int:
        """Number of VMM pages in the MNNVL region."""
        return self.mnnvl_size // self.granularity

    @property
    def post_pages(self) -> int:
        """Number of pool pages in the post-remote region."""
        return self.post_size // self.pool_granularity

    @property
    def remote_pages(self) -> int:
        """Total number of pool pages needed for remote regions."""
        return self.pre_pages + self.post_pages


def _align_up(value: int, alignment: int) -> int:
    """Align value up to the nearest multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    """Align value down to the nearest multiple of alignment."""
    return (value // alignment) * alignment
