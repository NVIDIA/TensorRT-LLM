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

"""Page pool for DWDP remote double buffer.

The page pool manages per-page fabric handles for the remote regions of the
composite buffer. Two pools (slot 0, slot 1) support double buffering.

Each pool is a list of page-sized fabric handles that can be mapped at different
VA positions for different layers. Pages are reused across layers in the same slot:
    - Layer 3 (buf_idx=0): pool[0] pages 0..K for pre/post regions
    - Layer 5 (buf_idx=0): SAME pool[0] pages (double buffer reuse)
    - Layer 4 (buf_idx=1): pool[1] pages 0..M

This naturally handles heterogeneous layers: each layer maps exactly the pages
it needs, even if remote sizes differ across layers.

Verified on GB200:
    - Per-page fabric handles composited into one VA: PASS
    - Same page handles reused across layers: PASS
    - Non-fabric handles cannot be mapped: FAIL (not supported on GB200)
"""

from __future__ import annotations

from typing import List, Tuple

from tensorrt_llm.logger import logger

from .vmm import (
    align_up,
    create_local_handle,
    get_allocation_granularity,
    map_handle,
    release_handle,
    unmap_va,
)


class PagePool:
    """Pool of per-page fabric handles for remote double buffer.

    Two pools (slot 0, slot 1) for double buffering. Each pool is a list
    of page-sized fabric handles. Different layers map these pages at
    different VA positions within their composite VA.

    Pages are reused across layers in the same slot:
        - Layer 3 (buf_idx=0): pool[0] pages 0..K for pre/post regions
        - Layer 5 (buf_idx=0): SAME pool[0] pages (double buffer reuse)
        - Layer 4 (buf_idx=1): pool[1] pages 0..M

    Pool page size vs VMM granularity:
        Each pool handle is ``page_size`` bytes (default 8 * granularity =
        16MB on GB200).  Larger pool pages reduce the number of cuMemMap
        calls by 8x, avoiding the CUDA driver internal tracking limit
        (~130K mappings per process) that causes OOM when 4 processes share
        a GB200 node with 2MB pages.

    Attributes:
        device_id: CUDA device ordinal.
        granularity: VMM page granularity in bytes (typically 2MB).
        page_size: Pool page handle size in bytes (typically 16MB).
        slot_sizes: Size of each slot in bytes.
        slot_pages: Number of pool pages in each slot.
    """

    __slots__ = (
        "_device_id",
        "_granularity",
        "_page_size",
        "_slot_sizes",
        "_slot_pages",
        "_page_handles",
        "_released",
    )

    # Default pool page multiplier: each pool handle = 8 * granularity.
    # On GB200 with 2MB granularity this gives 16MB pool pages, reducing
    # cuMemMap calls by 8x compared to 1 * granularity.
    DEFAULT_PAGE_SIZE_MULTIPLIER = 8

    def __init__(
        self,
        slot_sizes: List[int],
        device_id: int,
        granularity: int | None = None,
        page_size: int | None = None,
    ):
        """Create a page pool with the specified slot sizes.

        Args:
            slot_sizes: [max_remote_bytes_slot0, max_remote_bytes_slot1]
                Computed as max over all layers assigned to each slot.
            device_id: CUDA device ordinal.
            granularity: VMM page granularity in bytes. If None, queries the
                device.
            page_size: Pool page handle size in bytes. Must be a positive
                multiple of ``granularity``. If None, defaults to
                ``DEFAULT_PAGE_SIZE_MULTIPLIER * granularity`` (16MB on GB200).
        """
        self._device_id = device_id

        if granularity is None:
            self._granularity = get_allocation_granularity(device_id)
        else:
            self._granularity = granularity

        if page_size is None:
            self._page_size = self.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        else:
            if page_size <= 0 or page_size % self._granularity != 0:
                raise ValueError(
                    f"page_size must be a positive multiple of granularity "
                    f"({self._granularity}), got {page_size}"
                )
            # page_size must also be a power of 2 because vmm.align_up
            # requires it (used in map_pages for safety alignment).
            if (page_size & (page_size - 1)) != 0:
                raise ValueError(f"page_size must be a power of 2, got {page_size}")
            self._page_size = page_size

        self._slot_sizes = list(slot_sizes)
        # Number of pool pages per slot, using page_size (not granularity).
        self._slot_pages = [
            align_up(size, self._page_size) // self._page_size for size in slot_sizes
        ]

        # Allocate per-page LOCAL handles for each slot.
        # Uses HANDLE_TYPE_NONE (not fabric) — these pages are only accessed
        # locally (P2P writes + local reads).  Non-fabric handles do NOT
        # consume NVLink fabric routing table entries, which are limited to
        # ~928 per GPU on GB200 and must be reserved for MNNVL expert data.
        #
        # Each handle is page_size bytes (e.g. 16MB), NOT granularity bytes.
        # This is the key change that reduces cuMemMap calls by 8x.
        self._page_handles: List[List[int]] = []
        self._released = False

        for slot_idx, num_pages in enumerate(self._slot_pages):
            handles = []
            for page_idx in range(num_pages):
                try:
                    handle = create_local_handle(self._page_size, device_id)
                    handles.append(handle)
                except Exception as e:
                    # Cleanup on failure
                    for h in handles:
                        release_handle(h)
                    for prev_handles in self._page_handles:
                        for h in prev_handles:
                            release_handle(h)
                    raise RuntimeError(
                        f"Failed to allocate page {page_idx} in slot {slot_idx}: {e}"
                    )
            self._page_handles.append(handles)
            logger.debug(
                f"[PagePool] Allocated {num_pages} pages ({self._page_size} B each) "
                f"for slot {slot_idx} ({self._slot_sizes[slot_idx]} bytes)"
            )

    @classmethod
    def create(
        cls,
        slot_sizes: List[int],
        device_id: int,
        page_size: int | None = None,
    ) -> PagePool:
        """Factory method to create a page pool.

        Args:
            slot_sizes: [max_remote_bytes_slot0, max_remote_bytes_slot1]
                Computed as max over all layers assigned to each slot.
            device_id: CUDA device ordinal.
            page_size: Pool page handle size in bytes. If None, uses default
                (8 * granularity = 16MB on GB200).

        Returns:
            Initialized PagePool instance.
        """
        return cls(slot_sizes, device_id, page_size=page_size)

    @property
    def device_id(self) -> int:
        """Get CUDA device ordinal."""
        return self._device_id

    @property
    def granularity(self) -> int:
        """Get VMM page granularity in bytes (typically 2MB)."""
        return self._granularity

    @property
    def page_size(self) -> int:
        """Get pool page handle size in bytes (typically 16MB)."""
        return self._page_size

    @property
    def num_slots(self) -> int:
        """Get number of double buffer slots."""
        return len(self._page_handles)

    def num_pages(self, slot: int) -> int:
        """Get number of pages in a slot.

        Args:
            slot: Double buffer slot (0 or 1).

        Returns:
            Number of pages in the slot.
        """
        return self._slot_pages[slot]

    def slot_size(self, slot: int) -> int:
        """Get the size of a slot in bytes.

        Args:
            slot: Double buffer slot (0 or 1).

        Returns:
            Size of the slot in bytes.
        """
        return self._slot_sizes[slot]

    def get_page_handle(self, slot: int, page_idx: int) -> int:
        """Get handle for a specific page.

        Args:
            slot: Double buffer slot (0 or 1).
            page_idx: Page index within the slot.

        Returns:
            Page handle as integer.

        Raises:
            IndexError: If slot or page_idx is out of range.
            RuntimeError: If pool has been released.
        """
        if self._released:
            raise RuntimeError("PagePool has been released")
        if slot < 0 or slot >= len(self._page_handles):
            raise IndexError(f"Invalid slot {slot}, must be in [0, {len(self._page_handles)})")
        if page_idx < 0 or page_idx >= len(self._page_handles[slot]):
            raise IndexError(
                f"Invalid page_idx {page_idx} for slot {slot}, "
                f"must be in [0, {len(self._page_handles[slot])})"
            )
        return self._page_handles[slot][page_idx]

    def map_pages(
        self,
        slot: int,
        va_start: int,
        size: int,
        page_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        """Map pages from pool[slot] starting at page_offset into VA.

        Maps ceil(size / page_size) pool page handles sequentially:
            cuMemMap(va_start + i*page_size, page_size, 0,
                     pages[slot][page_offset + i], 0)

        Each cuMemMap call maps exactly ``page_size`` bytes (the full pool
        handle). Using 16MB pool pages instead of 2MB VMM pages reduces the
        number of cuMemMap calls by 8x.

        Args:
            slot: Double buffer slot (0 or 1).
            va_start: Page-aligned VA position to start mapping.
            size: Total bytes to map (must be a multiple of page_size;
                callers should use pool_granularity-aligned pre/post sizes).
            page_offset: First page index within the pool slot.

        Returns:
            List of (va, size) tuples for each mapping made.

        Raises:
            ValueError: If mapping would exceed available pages.
            RuntimeError: If pool has been released.
        """
        if self._released:
            raise RuntimeError("PagePool has been released")

        aligned_size = align_up(size, self._page_size)
        num_pages_needed = aligned_size // self._page_size

        if page_offset + num_pages_needed > len(self._page_handles[slot]):
            raise ValueError(
                f"Mapping {num_pages_needed} pages from offset {page_offset} "
                f"exceeds slot {slot} capacity of {len(self._page_handles[slot])} pages"
            )

        mappings = []
        for i in range(num_pages_needed):
            va = va_start + i * self._page_size
            handle = self._page_handles[slot][page_offset + i]
            map_handle(va, self._page_size, handle, offset=0)
            mappings.append((va, self._page_size))

        # NOTE: set_access is intentionally NOT called here.
        # The caller (WeightBuffer._setup_layer) issues a single
        # cuMemSetAccess for the entire composite VA (pre + mnnvl + post)
        # after all sub-regions are mapped.  This reduces the total number
        # of cuMemSetAccess calls from 3 per weight per layer to 1, which
        # is necessary to stay within the CUDA VMM NVLink fabric routing
        # table limit on GB200 MNNVL nodes.

        return mappings

    def unmap_pages(self, mappings: List[Tuple[int, int]]) -> None:
        """Unmap previously mapped pages.

        Args:
            mappings: List of (va, size) tuples from map_pages().
        """
        for va, size in mappings:
            try:
                unmap_va(va, size)
            except Exception as e:
                logger.warning(f"[PagePool] Failed to unmap VA {va:#x}: {e}")

    def release(self) -> None:
        """Release all page handles. Idempotent. Safe from __del__."""
        if self._released:
            return

        self._released = True

        for slot_idx, handles in enumerate(self._page_handles):
            for page_idx, handle in enumerate(handles):
                try:
                    release_handle(handle)
                except Exception as e:
                    logger.warning(
                        f"[PagePool] Failed to release page {page_idx} in slot {slot_idx}: {e}"
                    )

        self._page_handles = [[], []]
        logger.debug("[PagePool] Released all page handles")

    def __del__(self):
        """Clean up on destruction (best-effort; errors logged to debug)."""
        try:
            self.release()
        except Exception as e:
            logger.debug(f"[PagePool] __del__ release failed (ignored): {e!r}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def compute_slot_sizes(
    layouts: dict[int, dict[str, "PageAlignedLayout"]],  # noqa: F821
    buffer_slot_assignments: dict[int, int],
) -> List[int]:
    """Compute the required slot sizes for a set of layouts.

    For each slot, computes the maximum remote buffer size across all
    layers assigned to that slot.

    Args:
        layouts: Mapping from layer_idx to {weight_name: PageAlignedLayout}.
        buffer_slot_assignments: Mapping from layer_idx to slot (0 or 1).

    Returns:
        [slot0_size, slot1_size] in bytes.
    """

    slot_sizes = [0, 0]

    for layer_idx, weight_layouts in layouts.items():
        slot = buffer_slot_assignments.get(layer_idx, layer_idx % 2)

        # Sum remote sizes across all weight names within a layer.
        # Each weight name needs its own pool pages (they hold different data
        # and are prefetched independently by the WeightManager).
        layer_remote_total = 0
        for layout in weight_layouts.values():
            layer_remote_total += layout.pre_size + layout.post_size

        slot_sizes[slot] = max(slot_sizes[slot], layer_remote_total)

    return slot_sizes
