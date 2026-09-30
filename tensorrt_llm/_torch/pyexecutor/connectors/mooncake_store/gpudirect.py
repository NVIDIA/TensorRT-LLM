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
"""What the driver says about a KV pool range, for diagnosing registration failures.

Registration failures surface as one opaque status code from `register_buffer`,
several layers below anything this connector can see. Two causes account for
most of them.

`ibv_reg_mr` with `EFAULT`. Mooncake's `RdmaContext::exportDmabuf` chooses
between `ibv_reg_mr` and `ibv_reg_dmabuf_mr`, preferring the former whenever
`WITH_NVIDIA_PEERMEM` is set, which it is by default. That path needs the
`nvidia_peermem` module, so on a host without it every range fails.

`ibv_reg_dmabuf_mr` with `EINVAL`. A registration succeeds only while it stays
inside one GPU pool mapping, as described on
`KvCacheLayout.gpu_pool_mapping_bytes`. On GB300 with a 32 MiB granularity,
32 MiB from a mapping boundary registers and 64 MiB does not, while a 16 MiB
window registers at every offset, so the boundary is what matters rather than
the length or the offset alone.

`CU_POINTER_ATTRIBUTE_RANGE_SIZE` does not expose that bound. It reports the
whole reservation, and `cuMemGetHandleForAddressRange` exports a dma-buf over
all of it, so both look healthy on a pool that cannot be registered in one
call. They are reported here as context; the bound itself is the cache
manager's granularity, which the caller supplies.

`CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` does matter, and not only as context:
the mappings tile the reservation from its base, so that address is where the
boundaries are counted from. `reservation_start` reads it for that purpose.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "RangeFacts",
    "REGISTRATION_DEBUG_ENV",
    "describe_range",
    "format_diagnosis",
    "peermem_loaded",
    "reservation_start",
]

#: Dump the per-range facts below at registration time, not only after a
#: failure. One line per range, so off by default.
REGISTRATION_DEBUG_ENV = "TRTLLM_MOONCAKE_STORE_DEBUG_REGISTRATION"
#: The variable Mooncake reads to choose between `ibv_reg_mr` and
#: `ibv_reg_dmabuf_mr`. Reported but never set: which path works is a property
#: of the host.
PEERMEM_ENV = "WITH_NVIDIA_PEERMEM"

_MIB = 1 << 20


def _driver():
    """The CUDA driver bindings, or None where they are unavailable."""
    try:
        from cuda.bindings import driver
    except ImportError:
        try:
            from cuda import cuda as driver
        except ImportError:
            return None
    return driver


def _pointer_attribute(driver, address: int, name: str):
    """One `cuPointerGetAttribute`, or None where the driver will not answer."""
    try:
        enum = getattr(driver.CUpointer_attribute, name)
    except AttributeError:
        return None
    status, value = driver.cuPointerGetAttribute(enum, driver.CUdeviceptr(address))
    return None if int(status) != 0 else value


def reservation_start(address: int) -> Optional[int]:
    """Base of the virtual reservation holding `address`, or None if unreadable.

    This is where the mappings of a V2 GPU pool begin, and therefore where a
    caller cutting ranges on mapping boundaries has to count from. Nothing puts
    it at a multiple of the mapping size: `VirtMem` reserves with
    `cuMemAddressReserve(size, 0, 0, 0)`, and the driver satisfies a default
    alignment with its minimum granularity, which is smaller than a mapping
    whenever `pool_size_granularity` is more than 2 MiB.
    """
    driver = _driver()
    if driver is None:
        return None
    try:
        value = _pointer_attribute(driver, address, "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR")
    except Exception:  # noqa: BLE001
        # An address the driver does not recognize, or no context on this
        # thread. Either way the caller has a documented fallback, and a
        # question about a pointer must not be what fails startup.
        return None
    return None if value is None else int(value)


def peermem_loaded() -> Optional[bool]:
    """Whether `nvidia_peermem` is loaded, or None if that cannot be read.

    The module list is the host's, so this is meaningful from inside a
    container.
    """
    try:
        with open("/proc/modules") as handle:
            return any(line.startswith("nvidia_peermem") for line in handle)
    except OSError:
        return None


@dataclass(frozen=True)
class RangeFacts:
    """Driver facts about one range the connector is about to register."""

    address: int
    length: int
    #: 1 for host, 2 for device. Mooncake only considers dma-buf for device.
    memory_type: Optional[int] = None
    #: Start and size of the whole reservation holding `address`, not of the
    #: single mapping within it. See the note in the module docstring.
    range_start: Optional[int] = None
    range_size: Optional[int] = None
    #: Whether the driver considers this memory GPUDirect-RDMA capable. The V2
    #: allocator asks for it but falls back to allocations without it, so this
    #: can be false without anything having logged a warning.
    gdr_capable: Optional[bool] = None
    #: `CUresult` from exporting a dma-buf over the mapping, 0 on success.
    dmabuf_status: Optional[int] = None
    #: Whatever went wrong collecting the above.
    error: Optional[str] = None

    def fits_one_mapping(self, mapping_bytes: Optional[int]) -> Optional[bool]:
        """Whether this range stays inside one `mapping_bytes` sized mapping.

        Mappings tile the reservation from its base, so a range crosses a
        boundary unless it sits within a single tile. None when the granularity
        is unknown, since nothing the driver reports substitutes for it.
        """
        if not mapping_bytes or self.range_start is None:
            return None
        offset = self.address - self.range_start
        return offset % mapping_bytes + self.length <= mapping_bytes

    def describe(self) -> str:
        """One line for a log."""
        if self.error is not None:
            return (
                f"[{self.address:#x}, {self.address + self.length:#x}) "
                f"{self.length / _MIB:.1f} MiB: {self.error}"
            )
        reservation = (
            f"reservation [{self.range_start:#x}, +{self.range_size / _MIB:.1f} MiB)"
            if self.range_size is not None
            else "reservation unknown"
        )
        return (
            f"[{self.address:#x}, {self.address + self.length:#x}) "
            f"{self.length / _MIB:.1f} MiB: memory_type={self.memory_type} "
            f"gdr_capable={self.gdr_capable} {reservation} "
            f"dmabuf_export={self.dmabuf_status}"
        )


def describe_range(address: int, length: int) -> RangeFacts:
    """Collect what Mooncake's `exportDmabuf` would see for `[address, +length)`.

    Every failure is folded into the returned value so this can be called from
    an error path without masking the original fault. The exported fd is closed
    immediately.
    """
    driver = _driver()
    if driver is None:
        return RangeFacts(address, length, error="cuda.bindings.driver unavailable")

    def attribute(name: str):
        return _pointer_attribute(driver, address, name)

    try:
        memory_type = attribute("CU_POINTER_ATTRIBUTE_MEMORY_TYPE")
        range_start = attribute("CU_POINTER_ATTRIBUTE_RANGE_START_ADDR")
        range_size = attribute("CU_POINTER_ATTRIBUTE_RANGE_SIZE")
        gdr = attribute("CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE")

        dmabuf_status: Optional[int] = None
        if range_start is not None and range_size is not None and int(range_size) > 0:
            status, fd = driver.cuMemGetHandleForAddressRange(
                driver.CUdeviceptr(int(range_start)),
                int(range_size),
                driver.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                0,
            )
            dmabuf_status = int(status)
            if dmabuf_status == 0:
                os.close(int(fd))

        return RangeFacts(
            address=address,
            length=length,
            memory_type=None if memory_type is None else int(memory_type),
            range_start=None if range_start is None else int(range_start),
            range_size=None if range_size is None else int(range_size),
            gdr_capable=None if gdr is None else bool(int(gdr)),
            dmabuf_status=dmabuf_status,
        )
    except Exception as exc:  # noqa: BLE001
        return RangeFacts(address, length, error=f"{type(exc).__name__}: {exc}")


def format_diagnosis(
    ranges: Sequence[Tuple[int, int]], rank: int, mapping_bytes: Optional[int] = None
) -> str:
    """A report explaining why registering `ranges` did or will fail.

    Args:
        ranges: `(start, end)` byte ranges as handed to `register_buffer`.
        rank: Reporting rank, for logs that interleave.
        mapping_bytes: The cache manager's GPU `pool_size_granularity`, if
            known. Without it the report can describe the ranges but not say
            which of them are unregisterable.

    Returns:
        A multi-line report: the host's GPUDirect situation, the driver's view
        of each range, and the conclusion those facts support.
    """
    peermem = peermem_loaded()
    peermem_requested = os.getenv(PEERMEM_ENV)
    dmabuf_selected = peermem_requested in ("0", "false", "False", "no", "off")
    lines: List[str] = [
        f"mooncake-store rank {rank} GPU registration diagnosis:",
        f"  nvidia_peermem loaded: {peermem}",
        f"  {PEERMEM_ENV}={peermem_requested!r} "
        f"(anything but 0 selects Mooncake's ibv_reg_mr path, which needs the module)",
        f"  GPU pool mapping granularity: "
        f"{'unknown' if not mapping_bytes else f'{mapping_bytes / _MIB:.0f} MiB'}",
    ]

    facts = [describe_range(start, end - start) for start, end in ranges]
    lines.append(f"  {len(facts)} range(s) to register:")
    lines.extend(f"    {fact.describe()}" for fact in facts)

    crossing = [f for f in facts if f.fits_one_mapping(mapping_bytes) is False]
    not_gdr = [fact for fact in facts if fact.gdr_capable is False]

    lines.append("  conclusion:")
    if peermem is False and not dmabuf_selected:
        lines.append(
            f"    Mooncake will call ibv_reg_mr because {PEERMEM_ENV} is not 0, but "
            "nvidia_peermem is not loaded, so every range fails with EFAULT "
            '("Bad address"). Either load the module or set '
            f"{PEERMEM_ENV}=0 to select the dma-buf path."
        )
    if crossing:
        lines.append(
            f"    {len(crossing)} range(s) cross a {mapping_bytes / _MIB:.0f} MiB pool "
            "mapping boundary. ibv_reg_dmabuf_mr rejects those with EINVAL "
            '("Invalid argument") however the call is framed, so the dma-buf '
            "path needs each registration confined to one mapping."
        )
    elif dmabuf_selected and not mapping_bytes:
        lines.append(
            "    On the dma-buf path a registration must stay inside one of the "
            "cache manager's GPU pool mappings. That granularity is not "
            "reported here, so if these ranges are larger than it, EINVAL is "
            "expected and is not visible in the facts above."
        )
    if not_gdr:
        lines.append(
            f"    {len(not_gdr)} range(s) are not GPUDirect-RDMA capable. The V2 "
            "allocator asks for that property and silently drops it when the "
            "platform refuses, so no RDMA registration can work on them."
        )
    if not crossing and not not_gdr and peermem is not False:
        lines.append("    Nothing here explains a registration failure; see the Mooncake log.")
    return "\n".join(lines)
