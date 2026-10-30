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
"""What the driver says about a KV pool range, for registering and diagnosing it.

Mooncake registers device memory one of two ways, chosen by `WITH_NVIDIA_PEERMEM`:

- `ibv_reg_mr`, the default, needs the `nvidia_peermem` module and VMM memory
  created with `gpuDirectRDMACapable=1`. Either missing fails with `EFAULT`.
- `ibv_reg_dmabuf_mr`, with `WITH_NVIDIA_PEERMEM=0`, needs neither.

Both refuse a range that crosses two `cuMemCreate` allocations, under peermem
with the same `EFAULT`. The report below tells these causes apart.

Mooncake reads the variable once per process, when the first client starts, so
the choice is made before that and only checked afterwards.
"""

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = ["PEERMEM_ENV", "allocation_range", "format_diagnosis", "select_registration_path"]

#: Selects Mooncake's registration path for the whole process.
PEERMEM_ENV = "WITH_NVIDIA_PEERMEM"

_NOT_STARTED = object()
#: `PEERMEM_ENV` as this process's first Mooncake client saw it, or
#: `_NOT_STARTED`. Guarded by `_first_client_lock`.
_peermem_at_first_client = _NOT_STARTED
_first_client_lock = threading.Lock()

#: Kernel modules that let `ibv_reg_mr` pin GPU memory. `nv_peer_mem` is the
#: out-of-tree predecessor of `nvidia_peermem`.
_PEERMEM_MODULES = ("nvidia_peermem", "nv_peer_mem")

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


def allocation_range(address: int) -> Optional[Tuple[int, int]]:
    """`(base, size)` of the `cuMemCreate` allocation mapped at `address`, or None.

    This is the bound a GPUDirect registration must stay within. The
    `CU_POINTER_ATTRIBUTE_RANGE_*` attributes report the whole reservation
    instead, which is too wide.
    """
    driver = _driver()
    if driver is None:
        return None
    try:
        status, base, size = driver.cuMemGetAddressRange(driver.CUdeviceptr(address))
    except Exception:  # noqa: BLE001
        # An unknown address or no context on this thread. The caller falls
        # back, so a pointer query must not fail startup.
        return None
    if int(status) != 0 or int(size) <= 0:
        return None
    return int(base), int(size)


def peermem_loaded() -> Optional[bool]:
    """Whether a peer-memory module is loaded, or None if that cannot be read.

    The module list is the host's, so this is meaningful from inside a container.
    """
    try:
        with open("/proc/modules") as handle:
            return any(line.split(" ", 1)[0] in _PEERMEM_MODULES for line in handle)
    except OSError:
        return None


def _selects_dmabuf(value: Optional[str]) -> bool:
    return value in ("0", "false", "False", "no", "off")


def peermem_value_in_effect() -> Optional[str]:
    """The `WITH_NVIDIA_PEERMEM` value Mooncake uses in this process.

    The value its first client read, or, before any client has started, the
    value the next one will read.
    """
    with _first_client_lock:
        if _peermem_at_first_client is not _NOT_STARTED:
            return _peermem_at_first_client
    return os.getenv(PEERMEM_ENV)


def dmabuf_selected() -> bool:
    """Whether Mooncake registers device memory through dma-buf in this process."""
    return _selects_dmabuf(peermem_value_in_effect())


def dmabuf_supported(device: int) -> Optional[bool]:
    """Whether `device` reports `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`, or None if unknown."""
    driver = _driver()
    if driver is None:
        return None
    try:
        status, handle = driver.cuDeviceGet(device)
        if int(status) != 0:
            return None
        status, value = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, handle
        )
    except Exception:  # noqa: BLE001
        # An uninitialized driver or bindings without the attribute.
        return None
    return None if int(status) != 0 else bool(int(value))


def _visible_device_count() -> int:
    driver = _driver()
    if driver is None:
        return 0
    try:
        # Idempotent, and needed when nothing in the process has used CUDA yet.
        driver.cuInit(0)
        status, count = driver.cuDeviceGetCount()
    except Exception:  # noqa: BLE001
        # A driver that cannot initialize; nothing can be probed.
        return 0
    return int(count) if int(status) == 0 else 0


def dmabuf_supported_everywhere() -> Optional[bool]:
    """Whether every visible device supports dma-buf, or None if unknown.

    With dma-buf selected, Mooncake requires it of every GPU it maps to a NIC
    and fails context creation otherwise, so one device is not enough to ask.
    """
    count = _visible_device_count()
    if count == 0:
        return None
    answers = [dmabuf_supported(device) for device in range(count)]
    if any(answer is False for answer in answers):
        return False
    return True if all(answers) else None


def dmabuf_needed() -> bool:
    """Whether `ibv_reg_mr` is certain to fail and dma-buf is available."""
    return peermem_loaded() is False and dmabuf_supported_everywhere() is True


def select_registration_path(protocol: str, *, registers_device_memory: bool) -> bool:
    """Choose Mooncake's registration path; call before each client starts.

    Before the process's first client, sets `WITH_NVIDIA_PEERMEM=0` when the
    protocol is RDMA, the variable is unset and `dmabuf_needed()` holds, then
    records the value that client reads. Later calls never write it, since
    Mooncake would ignore the change.

    Args:
        protocol: The transport the client is about to set up.
        registers_device_memory: Whether the client will register KV pools.

    Returns:
        Whether this call selected dma-buf.

    Raises:
        RuntimeError: A later client registering device memory over RDMA
            needs a path other than the recorded one.
    """
    global _peermem_at_first_client
    rdma = protocol == "rdma"
    with _first_client_lock:
        if _peermem_at_first_client is _NOT_STARTED:
            selected = rdma and os.getenv(PEERMEM_ENV) is None and dmabuf_needed()
            if selected:
                os.environ[PEERMEM_ENV] = "0"
            _peermem_at_first_client = os.getenv(PEERMEM_ENV)
            return selected
        in_effect = _peermem_at_first_client

    if not (rdma and registers_device_memory):
        return False
    current = os.getenv(PEERMEM_ENV)
    if _selects_dmabuf(current) != _selects_dmabuf(in_effect):
        raise RuntimeError(
            f"{PEERMEM_ENV} is {current!r}, but this process's first Mooncake client "
            f"started with {in_effect!r}, and Mooncake reads it only then. Set "
            f"{PEERMEM_ENV} before the first Mooncake client in the process starts "
            "and leave it unchanged, or set stage_through_host: true."
        )
    if not _selects_dmabuf(in_effect) and dmabuf_needed():
        raise RuntimeError(
            f"This process's first Mooncake client started with {PEERMEM_ENV}="
            f"{in_effect!r}, so Mooncake registers GPU memory with ibv_reg_mr, "
            "which fails without a peer-memory module, and none is loaded. Mooncake "
            f"reads {PEERMEM_ENV} only once, so export {PEERMEM_ENV}=0 before the "
            "first Mooncake client in the process starts, or set "
            "stage_through_host: true."
        )
    return False


@dataclass(frozen=True)
class RangeFacts:
    """Driver facts about one range the connector is about to register."""

    address: int
    length: int
    #: 1 for host, 2 for device. Mooncake only considers dma-buf for device.
    memory_type: Optional[int] = None
    #: The allocation holding `address`; see `allocation_range`.
    allocation_start: Optional[int] = None
    allocation_size: Optional[int] = None
    #: Whether the memory was created GPUDirect-RDMA capable.
    gdr_capable: Optional[bool] = None
    #: `CUresult` from exporting a dma-buf over the allocation, 0 on success.
    dmabuf_status: Optional[int] = None
    #: Whatever went wrong collecting the above.
    error: Optional[str] = None

    @property
    def fits_one_allocation(self) -> Optional[bool]:
        """Whether this range ends within the allocation it starts in."""
        if self.allocation_start is None or self.allocation_size is None:
            return None
        return self.address + self.length <= self.allocation_start + self.allocation_size

    def describe(self) -> str:
        """One line for a log."""
        if self.error is not None:
            return (
                f"[{self.address:#x}, {self.address + self.length:#x}) "
                f"{self.length / _MIB:.1f} MiB: {self.error}"
            )
        allocation = (
            f"allocation [{self.allocation_start:#x}, +{self.allocation_size / _MIB:.1f} MiB)"
            if self.allocation_size is not None
            else "allocation unknown"
        )
        return (
            f"[{self.address:#x}, {self.address + self.length:#x}) "
            f"{self.length / _MIB:.1f} MiB: memory_type={self.memory_type} "
            f"gdr_capable={self.gdr_capable} {allocation} "
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

    try:
        memory_type = _pointer_attribute(driver, address, "CU_POINTER_ATTRIBUTE_MEMORY_TYPE")
        gdr = _pointer_attribute(driver, address, "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE")
        allocation = allocation_range(address)

        dmabuf_status: Optional[int] = None
        if allocation is not None:
            status, fd = driver.cuMemGetHandleForAddressRange(
                driver.CUdeviceptr(allocation[0]),
                allocation[1],
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
            allocation_start=None if allocation is None else allocation[0],
            allocation_size=None if allocation is None else allocation[1],
            gdr_capable=None if gdr is None else bool(int(gdr)),
            dmabuf_status=dmabuf_status,
        )
    except Exception as exc:  # noqa: BLE001
        return RangeFacts(address, length, error=f"{type(exc).__name__}: {exc}")


def format_diagnosis(ranges: Sequence[Tuple[int, int]], rank: int) -> str:
    """A report explaining why registering `ranges` did or will fail.

    Args:
        ranges: `(start, end)` byte ranges as handed to `register_buffer`.
        rank: Reporting rank, for logs that interleave.

    Returns:
        A multi-line report: the host's GPUDirect situation, the driver's view
        of each range, and the conclusion those facts support.
    """
    peermem = peermem_loaded()
    in_effect = peermem_value_in_effect()
    on_dmabuf = _selects_dmabuf(in_effect)
    lines: List[str] = [
        f"mooncake-store rank {rank} GPU registration diagnosis:",
        f"  nvidia_peermem loaded: {peermem}",
        f"  {PEERMEM_ENV}={in_effect!r} as Mooncake read it "
        f"({'dma-buf' if on_dmabuf else 'ibv_reg_mr, which needs nvidia_peermem'})",
    ]
    current = os.getenv(PEERMEM_ENV)
    if current != in_effect:
        lines.append(
            f"  {PEERMEM_ENV} is {current!r} in the environment now, which Mooncake "
            "ignores: it reads the variable once, when the first client in the "
            "process starts."
        )

    facts = [describe_range(start, end - start) for start, end in ranges]
    lines.append(f"  {len(facts)} range(s) to register:")
    lines.extend(f"    {fact.describe()}" for fact in facts)

    crossing = [fact for fact in facts if fact.fits_one_allocation is False]
    not_gdr = [fact for fact in facts if fact.gdr_capable is False]

    lines.append("  conclusion:")
    if peermem is False and not on_dmabuf:
        lines.append(
            f"    Mooncake will call ibv_reg_mr because {PEERMEM_ENV} is not 0, but "
            "nvidia_peermem is not loaded, so every range fails with EFAULT "
            '("Bad address"). Either load the module or set '
            f"{PEERMEM_ENV}=0 to select the dma-buf path."
        )
    if crossing:
        lines.append(
            f"    {len(crossing)} range(s) run past the cuMemCreate allocation "
            "they start in. Both paths refuse those: ibv_reg_mr with EFAULT, the "
            "same error as a missing nvidia_peermem, and ibv_reg_dmabuf_mr with "
            "EINVAL."
        )
    if not_gdr and not on_dmabuf:
        lines.append(
            f"    {len(not_gdr)} range(s) were allocated without "
            "gpuDirectRDMACapable, which the V2 allocator drops where the "
            "platform refuses it. nvidia_peermem cannot register such memory "
            f"even one allocation at a time; {PEERMEM_ENV}=0 selects dma-buf, "
            "which can."
        )
    if not crossing and not (not_gdr and not on_dmabuf) and (peermem is not False or on_dmabuf):
        lines.append("    Nothing here explains a registration failure; see the Mooncake log.")
    return "\n".join(lines)
