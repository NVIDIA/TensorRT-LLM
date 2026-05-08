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
from typing import Dict, List, Tuple

import torch


# (local_start, local_end_capped) for one peer DWDP rank.  ``local_start``
# is ``peer_rank * num_prefetch_experts`` (the rank-to-rank stride).
# ``local_end_capped`` is the *valid* upper bound — i.e.,
# ``min(local_start + num_experts_per_worker, num_experts_total)``; the
# tail rank's storage may extend past ``num_experts_total`` but the cap
# truncates the valid range.  Adjacent ranks' valid ranges may overlap
# when ``num_prefetch_experts < num_experts_per_worker`` (redundancy mode).
PeerRange = Tuple[int, int]
# Indexed by peer DWDP rank; ``len(PeerRanges) == dwdp_size``.
PeerRanges = List[PeerRange]


def compute_peer_ranges(
    *,
    dwdp_size: int,
    num_experts_per_worker: int,
    num_prefetch_experts: int,
    num_experts_total: int,
) -> PeerRanges:
    """Compute every peer rank's valid expert range.

    Each peer's storage holds ``num_experts_per_worker`` slots starting
    at ``peer_rank * num_prefetch_experts``.  The valid range is the
    storage range capped at ``num_experts_total`` (the model's total
    expert count), which truncates any padding on the tail rank.

    All ranks deterministically compute the same ``PeerRanges`` from the
    shared ``DwdpConfig``, so no allgather is required.
    """
    ranges: PeerRanges = []
    for peer_rank in range(dwdp_size):
        start = peer_rank * num_prefetch_experts
        end_capped = min(start + num_experts_per_worker, num_experts_total)
        ranges.append((start, end_capped))
    return ranges


def lookup_owner(expert_id: int, peer_ranges: PeerRanges) -> int:
    """Return the lowest peer rank whose valid range contains ``expert_id``.

    For uniform partition this is equivalent to
    ``expert_id // (num_experts_total // dwdp_size)``.  For redundancy
    (overlapping ranges) the first-match policy picks the lowest-rank
    owner, matching the IPC implementation's behavior so reads of the
    same expert id are deterministic across peers.

    Raises:
        ValueError: If no peer's valid range contains ``expert_id``.
            This is impossible when the partition passes the coverage
            check ``(dwdp_size - 1) * stride + size >= num_experts``.
    """
    for peer_rank, (start, end) in enumerate(peer_ranges):
        if start <= expert_id < end:
            return peer_rank
    raise ValueError(
        f"expert_id={expert_id} not owned by any peer in "
        f"peer_ranges={peer_ranges}"
    )


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

    When ``pool_granularity > granularity``, a padding region appears between
    the pool-backed pre-region and the MNNVL region (``pre_padding`` bytes) and
    between the MNNVL region and the pool-backed post-region (``post_padding``
    bytes).  The tensor view must start at ``va_base + pre_padding`` so that
    expert indices line up with the correct physical memory.

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
        pre_padding: Extra bytes at end of pre-region due to pool_granularity
            rounding (pre_size - page_start).
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
            raise ValueError(
                f"pool_granularity must be a power of 2, got {pool_granularity}"
            )

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
        # On GB200, cuMemMap with fabric handles requires size == phys_size
        # (partial mapping returns CUDA_ERROR_NOT_SUPPORTED).
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
