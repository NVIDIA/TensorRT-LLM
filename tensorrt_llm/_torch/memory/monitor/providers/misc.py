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

from ..schema import Basis, Category, Entry, Placement, Source

__all__ = ["DeepEpProvider", "DwdpPagePoolProvider", "MemoryCountersProvider"]


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
        try:
            from tensorrt_llm._torch.moe.fused_moe.deep_ep_utils import deep_ep_installed
        except ImportError:
            return False
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
                total += int(getattr(buffer, "num_nvl_bytes", 0))
                total += int(getattr(buffer, "num_rdma_bytes", 0))

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


class DwdpPagePoolProvider:
    """DWDP page pool.

    Reads ``held_pages``, the handles the pool owns now, not
    ``num_pages``/``configured_pages``, which keep their value after
    ``release()`` has handed every page back. Measured on a live pool: after
    releasing every page, ``sum(num_pages(s)) * page_size`` still reported
    128 MiB while the device had taken all 128 MiB back.

    A handle whose release raised is no longer held but may still be mapped,
    so a non-zero ``release_failed`` turns the figure into a lower bound and
    the note says so.
    """

    name = "dwdp_page_pool"
    category = Category.DWDP_PAGES

    def _pool(self):
        try:
            from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager
        except ImportError:
            return None
        manager = get_global_dwdp_manager()
        if manager is None:
            return None
        # DwdpManager keeps the weight manager private; there is no public
        # accessor to go through.
        weight_manager = getattr(manager, "_weight_manager", None)
        if weight_manager is None:
            return None
        buffer = getattr(weight_manager, "weight_buffer", None)
        if buffer is None:
            return None
        return getattr(buffer, "_page_pool", None)

    def available(self) -> bool:
        return self._pool() is not None

    def collect(self) -> list[Entry]:
        pool = self._pool()
        if pool is None:
            return []
        held = sum(pool.held_pages(slot) for slot in range(pool.num_slots))
        failed = pool.release_failed

        note = None
        if failed:
            # release() drops a handle it could not free, so held_pages stops
            # counting memory that is plausibly still mapped. Reporting the
            # smaller number without saying so would read as a clean release.
            note = (
                f"at least this much: {failed} handle(s) failed to release "
                "and are no longer tracked, so the true figure may be higher"
            )

        return [
            Entry(
                category=Category.DWDP_PAGES,
                size_bytes=held * pool.page_size,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.PHYSICAL_BACKING,
                owner="dwdp.PagePool",
                note=note,
            )
        ]


class MemoryCountersProvider:
    """C++ ``MemoryCounters``.

    Reported for diagnosis only -- ``counts_toward_total=False`` -- because it
    overlaps ``torch_reserved`` rather than adding to it. Measured: allocating
    512 MiB inside a ``virtual_memory.scope`` raised ``MemoryCounters.gpu``
    *and* ``torch_reserved`` by 512 MiB each while the device consumed 512 MiB
    once. Summing them would have doubled the figure.

    Its real value is as a physical-truth correction. After
    ``release_with_tag()`` the device gave back 512 MiB and ``MemoryCounters``
    followed, while ``torch_reserved`` did not move at all: torch keeps
    accounting for a VMM-backed pool whose pages are gone. So the two are the
    physical and the logical view of the same bytes, and during sleep the
    torch view overstates occupancy by the full released amount.

    Two further caveats before this could ever become additive: it is a single
    process-wide counter with no device dimension, and
    ``CudaAllocatorAsync`` decrements it when ``cudaFreeAsync`` is *submitted*,
    while the CUDA pool may still hold the memory. It also excludes the KV
    cache v2 pool entirely.
    """

    name = "cpp_memory_counters"
    category = Category.CPP_BUFFERS

    def available(self) -> bool:
        try:
            from tensorrt_llm.bindings import MemoryCounters  # noqa: F401
        except ImportError:
            return False
        return True

    def collect(self) -> list[Entry]:
        if not self.available():
            return []
        from tensorrt_llm.bindings import MemoryCounters

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
