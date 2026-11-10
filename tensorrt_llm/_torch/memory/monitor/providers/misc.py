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
"""Providers that read a component's own accounting.

Each one declares its ``Basis``, because being able to read a number says
nothing about what the number means. ``DeepEP`` reports what it asked for;
``PagePool`` reports a configured capacity that survives the pages being
released. Summing those with physical backing produces a plausible, wrong
total.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm._torch.modules.dwdp.page_pool import unreleased_page_bytes
from tensorrt_llm.bindings import MemoryCounters

from ..schema import Basis, Category, Entry, Placement, Source

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.dwdp.page_pool import PagePool
    from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager

__all__ = [
    "DeepEpProvider",
    "DwdpLocalWeightsProvider",
    "DwdpPagePoolProvider",
    "MemoryCountersProvider",
]


class DeepEpProvider:
    """DeepEP NVL/RDMA buffers.

    Basis is :attr:`Basis.REQUESTED`: ``num_nvl_bytes``/``num_rdma_bytes`` are
    the sizes DeepEP asked for, not the NVSHMEM heap actually reserved, so a
    small positive residual against NVML is expected rather than a bug.

    Only the module-level default ``buffer_pool`` is visible; callers may
    construct their own pool, and nothing in the tree currently does.
    """

    name = "deep_ep"
    category = Category.COMM_DEEPEP

    def available(self) -> bool:
        # Imported on use: it pulls in the MoE communication stack, which a
        # process without DeepEP never loads.
        from tensorrt_llm._torch.moe.fused_moe.deep_ep_utils import deep_ep_installed

        return bool(deep_ep_installed)

    def collect(self) -> list[Entry]:
        if not self.available():
            return []
        from tensorrt_llm._torch.moe.fused_moe.deep_ep_utils import buffer_pool

        total = 0
        for pool in (buffer_pool.buffers, buffer_pool.low_latency_buffers):
            for ref in list(pool.values()):
                wrapper = ref()  # values are weakrefs
                if wrapper is None:
                    continue
                buffer = getattr(wrapper, "buffer", None)
                if buffer is None:
                    continue
                total += int(buffer.num_nvl_bytes)
                total += int(buffer.num_rdma_bytes)

        return [
            Entry(
                category=Category.COMM_DEEPEP,
                size_bytes=total,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.REQUESTED,
                owner="deep_ep.BufferPool",
                note="requested bytes; NVSHMEM heap may be larger",
            )
        ]


def _dwdp_manager() -> DwdpManager | None:
    """The process's DWDP manager, or None without DWDP."""
    # Imported on use: the executor package imports this one.
    from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager

    return get_global_dwdp_manager()


class DwdpPagePoolProvider:
    """DWDP page pool.

    Reads ``held_pages``, the handles the pool owns now, not ``num_pages``,
    which keeps its value after ``release()`` has handed every page back.

    Page handles whose release raised may still be mapped after their pool
    is gone. Their bytes are listed separately for diagnosis and not summed:
    whether the GPU still backs them is unknown, and if it does they are
    already inside the process's NVML figure.
    """

    name = "dwdp_page_pool"
    category = Category.DWDP_PAGES

    @staticmethod
    def _pool() -> PagePool | None:
        manager = _dwdp_manager()
        weight_manager = manager.weight_manager if manager is not None else None
        if weight_manager is None:
            return None
        return weight_manager.weight_buffer.page_pool

    @staticmethod
    def _unreleased() -> int:
        return unreleased_page_bytes()

    def available(self) -> bool:
        return self._pool() is not None or self._unreleased() > 0

    def collect(self) -> list[Entry]:
        entries = []
        pool = self._pool()
        if pool is not None:
            held = sum(pool.held_pages(slot) for slot in range(pool.num_slots))
            entries.append(
                Entry(
                    category=Category.DWDP_PAGES,
                    size_bytes=held * pool.page_size,
                    source=Source.COMPONENT,
                    placement=Placement.NON_TORCH,
                    basis=Basis.PHYSICAL_BACKING,
                    owner="dwdp.PagePool",
                )
            )
        unreleased = self._unreleased()
        if unreleased:
            entries.append(
                Entry(
                    category=Category.DWDP_PAGES,
                    size_bytes=unreleased,
                    source=Source.COMPONENT,
                    placement=Placement.NON_TORCH,
                    basis=Basis.REQUESTED,
                    owner="dwdp.PagePool, failed to release",
                    note="page handles whose release raised; may still be resident",
                    counts_toward_total=False,
                )
            )
        return entries


class DwdpLocalWeightsProvider:
    """This rank's own DWDP weight shards.

    DWDP moves each rank's expert chunks and dense shards out of PyTorch into
    fabric handles it creates itself, which peers then import. Only the
    local handles are counted: an imported handle is a peer's memory.
    """

    name = "dwdp_local_weights"
    category = Category.WEIGHTS

    @staticmethod
    def _sources() -> list[tuple[str, int]]:
        parts = []
        manager = _dwdp_manager()
        weight_manager = manager.weight_manager if manager is not None else None
        if weight_manager is not None:
            parts.append(("DWDP expert shards", weight_manager.local_weight_bytes))
        # Imported on use, like the DWDP manager above.
        from tensorrt_llm._torch.modules.dwdp.dense.manager import get_global_dense_dwdp_manager

        dense_manager = get_global_dense_dwdp_manager()
        dense = dense_manager.dense if dense_manager is not None else None
        if dense is not None:
            parts.append(("DWDP dense shards", dense.local_store_bytes))
        return parts

    def available(self) -> bool:
        return any(size for _, size in self._sources())

    def collect(self) -> list[Entry]:
        return [
            Entry(
                category=Category.WEIGHTS,
                size_bytes=size,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.PHYSICAL_BACKING,
                owner=owner,
                note="this rank's shards, in local fabric handles",
            )
            for owner, size in self._sources()
            if size
        ]


class MemoryCountersProvider:
    """C++ ``MemoryCounters``.

    Reported for diagnosis only -- ``counts_toward_total=False`` -- because it
    overlaps ``torch_reserved`` rather than adding to it: an allocation inside
    a ``virtual_memory.scope`` raises both by its size while the device
    consumes it once.

    The two are the physical and the logical view of the same bytes. After
    ``release_with_tag()`` the device gets the memory back and
    ``MemoryCounters`` follows, while ``torch_reserved`` keeps accounting for
    the VMM-backed pool whose pages are gone, so during sleep the torch view
    overstates occupancy by the released amount.

    Two further caveats before this could ever become additive: it is a single
    process-wide counter with no device dimension, and
    ``CudaAllocatorAsync`` decrements it when ``cudaFreeAsync`` is *submitted*,
    while the CUDA pool may still hold the memory. It also excludes the KV
    cache v2 pool entirely.
    """

    name = "cpp_memory_counters"
    category = Category.CPP_BUFFERS

    def available(self) -> bool:
        return True

    def collect(self) -> list[Entry]:
        return [
            Entry(
                category=Category.CPP_BUFFERS,
                size_bytes=int(MemoryCounters.instance().gpu),
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.PHYSICAL_BACKING,
                owner="MemoryCounters",
                note=(
                    "diagnostic only: overlaps torch_reserved, no device "
                    "dimension, excludes the KV cache v2 pool"
                ),
                counts_toward_total=False,
            )
        ]
