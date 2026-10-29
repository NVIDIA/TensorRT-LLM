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
"""Die-local weight placement for GPUs with two memory locality domains (Rubin VR200).

Each half of a weight is homed on one die, and the CTAs running on that die's SMs
process it first. This module provides the pieces the die-local routed-expert path
(``TRTLLM_K3_DIE_LOCAL_EXPERTS``) needs:

* :func:`stack_halves` copies an ``[L, N, ...]`` weight into one virtual range
  holding the stacked N-halves ``[2L, N/2, ...]``; half ``h`` is physically
  homed on die ``h`` (``cuMemCreate`` with a locality-domain location).
* :func:`smid_die_table` maps ``%smid`` to a die with a cold-load latency probe
  (the probe finds each SM's locality domain).
* :class:`StreamScratch` holds the kernels' tile-queue counters, one zeroed set
  per CUDA stream.

Kernels consume both: the CTAs on die ``h`` work on half ``h``.
"""

from __future__ import annotations

import ctypes
import functools
import statistics
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...logger import logger

# A probe line per SM is timed this many times per die; the per-die median decides.
_PROBE_LOADS = 16
# Smallest latency separation (cycles) accepted as a clean split.
_PROBE_MIN_GAP_CYCLES = 150
# One CTA per SM: a large dynamic shared-memory request keeps a second CTA from co-residing.
_PROBE_SMEM_BYTES = 150 * 1024

_PROBE_SRC = r"""
extern "C" __global__ void die_probe(const unsigned long long* __restrict__ addrs, int nloads,
                                     int nslots, long long* __restrict__ cycles,
                                     int* __restrict__ smids)
{
    if (threadIdx.x != 0) {
        return;
    }
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smids[blockIdx.x] = (int) sm;
    if ((int) sm >= nslots) {
        return;
    }
    for (int i = 0; i < nloads; ++i) {
        unsigned long long const a = addrs[(unsigned long long) sm * nloads + i];
        long long dt;
        unsigned v;
        // The address depends on the first clock read; the second waits for the data.
        asm volatile(
            "{\n\t.reg .u64 t0, t1, z, aa;\n\t.reg .pred p;\n\t"
            "mov.u64 t0, %%clock64;\n\t"
            "shr.u64 z, t0, 63;\n\t"
            "add.u64 aa, %2, z;\n\t"
            "ld.global.cg.u32 %1, [aa];\n\t"
            "setp.ne.u32 p, %1, 0x7fffffff;\n\t"
            "mov.u64 t1, 0;\n\t"
            "@p mov.u64 t1, %%clock64;\n\t"
            "sub.u64 %0, t1, t0;\n\t}"
            : "=l"(dt), "=r"(v)
            : "l"(a)
            : "memory");
        cycles[(unsigned long long) sm * nloads + i] = dt;
    }
}
"""


def _driver():
    import cuda.bindings.driver as cu

    return cu


def _check(result, what: str):
    cu = _driver()
    status = result[0] if isinstance(result, tuple) else result
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {status}")
    if isinstance(result, tuple):
        return result[1] if len(result) == 2 else result[1:]
    return None


def _warn_on_error(result, what: str) -> None:
    """Log a failed driver call on a release path, which must not raise."""
    cu = _driver()
    status = result[0] if isinstance(result, tuple) else result
    if status != cu.CUresult.CUDA_SUCCESS:
        logger.warning(f"die-local: {what} failed: {status}")


def locality_domain_count(device: int) -> int:
    """Number of memory locality domains (dies) of ``device``; 0 if the driver API lacks it.

    Not cached: a query before CUDA is initialized fails and must not stick.
    """
    try:
        cu = _driver()
        attr = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT
        _ = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    except (ImportError, AttributeError):
        return 0
    status, count = cu.cuDeviceGetAttribute(attr, device)
    return int(count) if status == cu.CUresult.CUDA_SUCCESS else 0


def _granularity(device: int) -> int:
    cu = _driver()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.localized.deviceId = device
    prop.location.localized.localityDomainId = 0
    return int(
        _check(
            cu.cuMemGetAllocationGranularity(
                prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            ),
            "cuMemGetAllocationGranularity",
        )
    )


class _LocalizedRange:
    """One virtual range whose granularity-sized chunks are homed on given dies.

    Exposes ``__cuda_array_interface__`` (uint8) so ``torch.as_tensor`` can wrap it;
    the tensor keeps this owner alive and the mapping is released with it.
    """

    def __init__(self, homes: List[int], granularity: int, device: int):
        cu = _driver()
        self.nbytes = len(homes) * granularity
        self.device = device
        handles: list = []
        mapped: list = []  # (address, size) of every mapped run, unmapped one by one
        self.base = int(
            _check(cu.cuMemAddressReserve(self.nbytes, granularity, 0, 0), "cuMemAddressReserve")
        )
        # Registered before any mapping so that a failure part way through still
        # releases what was created when the half-built object is collected.
        self._finalizer = weakref.finalize(
            self, _LocalizedRange._release, self.base, self.nbytes, handles, mapped, device
        )
        # Not at interpreter exit: process teardown returns the memory anyway, and the
        # device sync in _release could hang there behind a peer stuck in a collective.
        self._finalizer.atexit = False
        start = 0
        for i in range(1, len(homes) + 1):
            if i < len(homes) and homes[i] == homes[start]:
                continue
            size = (i - start) * granularity
            prop = cu.CUmemAllocationProp()
            prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
            prop.location.localized.deviceId = device
            prop.location.localized.localityDomainId = homes[start]
            handle = _check(cu.cuMemCreate(size, prop, 0), "cuMemCreate(locality domain)")
            handles.append(handle)
            address = self.base + start * granularity
            _check(cu.cuMemMap(address, size, 0, handle, 0), "cuMemMap")
            mapped.append((address, size))
            start = i
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cu.cuMemSetAccess(self.base, self.nbytes, [access], 1), "cuMemSetAccess")

    @staticmethod
    def _release(base: int, nbytes: int, handles: list, mapped: list, device: int) -> None:
        cu = _driver()
        if mapped:
            # Work queued on the device may still read the range; unmapping under it faults.
            try:
                torch.cuda.synchronize(device)
            except RuntimeError as exc:  # e.g. a sticky CUDA error: still release the range
                logger.warning(f"die-local: no device sync before unmapping ({exc})")
        for address, size in mapped:
            _warn_on_error(cu.cuMemUnmap(address, size), "cuMemUnmap")
        for handle in handles:
            _warn_on_error(cu.cuMemRelease(handle), "cuMemRelease")
        _warn_on_error(cu.cuMemAddressFree(base, nbytes), "cuMemAddressFree")

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
        }


def _chunk_homes(nbytes: int, split_at: int, granularity: int) -> List[int]:
    """Die (0/1) of each ``granularity`` chunk of an ``nbytes`` range split at ``split_at``.

    A chunk goes to die 0 when most of its used bytes lie before ``split_at``; a
    chunk that straddles the split therefore follows its majority.
    """
    chunks = -(-nbytes // granularity)
    return [
        0 if (i * granularity + min((i + 1) * granularity, nbytes)) / 2 < split_at else 1
        for i in range(chunks)
    ]


def _localized_bytes(nbytes: int, split_at: int, device: int) -> torch.Tensor:
    """uint8 tensor of ``nbytes``: bytes before ``split_at`` on die 0, the rest on die 1.

    Placement is per allocation-granularity chunk (2 MiB, see :func:`_chunk_homes`).
    """
    granularity = _granularity(device)
    homes = _chunk_homes(nbytes, split_at, granularity)
    with torch.cuda.device(device):
        owner = _LocalizedRange(homes, granularity, device)
        return torch.as_tensor(owner, device=torch.device("cuda", device))[:nbytes]


def stack_halves(weight: torch.Tensor, device: Optional[int] = None) -> torch.Tensor:
    """Copy ``weight`` [L, N, ...] into die-homed stacked N-halves [2L, N/2, ...] on ``device``.

    ``out[h * L + e] == weight[e, h * N/2:(h + 1) * N/2]``; rows ``[0, L)`` live on die 0,
    rows ``[L, 2L)`` on die 1. The copy is exact (bytes are moved, not converted).
    ``weight`` may be on the host; ``device`` defaults to its CUDA device, else the
    current one.
    """
    if weight.dim() < 2 or weight.size(1) % 2:
        raise ValueError(f"stack_halves needs [L, N, ...] with even N, got {tuple(weight.shape)}")
    if device is None:
        device = weight.device.index if weight.is_cuda else torch.cuda.current_device()
    weight = weight.contiguous()
    num, n_half = weight.size(0), weight.size(1) // 2
    shape: Tuple[int, ...] = (2 * num, n_half, *weight.shape[2:])
    nbytes = weight.numel() * weight.element_size()
    raw = _localized_bytes(nbytes, nbytes // 2, device)
    out = raw.view(weight.dtype).view(shape)
    out[:num].copy_(weight[:, :n_half])
    out[num:].copy_(weight[:, n_half:])
    return out


def _compile_probe(device: int):
    from cuda.bindings import nvrtc

    def nv_check(result, what):
        status = result[0] if isinstance(result, tuple) else result
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{what} failed: {status}")
        return result[1] if isinstance(result, tuple) and len(result) > 1 else None

    program = nv_check(
        nvrtc.nvrtcCreateProgram(_PROBE_SRC.encode(), b"die_probe.cu", 0, [], []),
        "nvrtcCreateProgram",
    )
    try:
        # PTX for a base virtual arch: the driver JIT-compiles it for the device.
        options = [b"--gpu-architecture=compute_90"]
        result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = nv_check(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"die probe compile failed: {log.decode(errors='replace')}")
        size = nv_check(nvrtc.nvrtcGetPTXSize(program), "nvrtcGetPTXSize")
        ptx = b" " * size
        nv_check(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    cu = _driver()
    image = ctypes.create_string_buffer(ptx)
    module = _check(cu.cuModuleLoadData(ctypes.addressof(image)), "cuModuleLoadData")
    kernel = _check(cu.cuModuleGetFunction(module, b"die_probe"), "cuModuleGetFunction")
    _check(
        cu.cuFuncSetAttribute(
            kernel,
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            _PROBE_SMEM_BYTES,
        ),
        "cuFuncSetAttribute",
    )
    return module, kernel


def _probe_cycles(kernel, device: int, addrs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cu = _driver()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    slots, nloads = addrs.shape
    cycles = torch.zeros_like(addrs)
    smids = torch.full((sms,), -1, dtype=torch.int32, device=addrs.device)
    holders = [
        ctypes.c_void_p(addrs.data_ptr()),
        ctypes.c_int(nloads),
        ctypes.c_int(slots),
        ctypes.c_void_p(cycles.data_ptr()),
        ctypes.c_void_p(smids.data_ptr()),
    ]
    params = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders]
    )
    stream = torch.cuda.current_stream(device)
    # Evict the probe lines from L2 so every timed load goes to HBM.
    props = torch.cuda.get_device_properties(device)
    flush = torch.ones(
        max(2 * getattr(props, "L2_cache_size", 0), 256 << 20) // 4,
        dtype=torch.int32,
        device=addrs.device,
    )
    flush.sum()
    del flush
    _check(
        cu.cuLaunchKernel(
            kernel,
            sms,
            1,
            1,
            32,
            1,
            1,
            _PROBE_SMEM_BYTES,
            cu.CUstream(stream.cuda_stream),
            ctypes.addressof(params),
            0,
        ),
        "cuLaunchKernel(die_probe)",
    )
    stream.synchronize()
    return cycles.cpu(), smids.cpu()


@functools.cache
def _smid_die_table_cached(device: int) -> Optional[Tuple[int, ...]]:
    try:
        return _probe_die_table(device)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        # NVRTC/driver failures, a missing cuda-python, or an unusable probe result:
        # die-local stays off for this device (cached, so the probe is not retried per layer).
        logger.warning(f"die-local: SM die probe failed ({exc!r}); no die table")
        return None


def _probe_die_table(device: int) -> Optional[Tuple[int, ...]]:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    slots = 2 * sms
    # Two small buffers, one per die; every SM times its own cold lines, 4 KiB apart.
    span = slots * _PROBE_LOADS * 4096
    buffers = [_localized_bytes(span, span if die == 0 else 0, device) for die in (0, 1)]
    for buffer in buffers:
        buffer.fill_(1)
    torch.cuda.synchronize(device)
    lines = torch.arange(slots * _PROBE_LOADS, dtype=torch.int64).view(slots, _PROBE_LOADS) * 4096
    module, kernel = _compile_probe(device)
    try:
        medians = []
        smids = None
        for buffer in buffers:
            cycles, smids = _probe_cycles(
                kernel, device, (lines + buffer.data_ptr()).to(buffer.device)
            )
            medians.append(cycles.float().median(dim=1).values)
    finally:
        _driver().cuModuleUnload(module)
    seen = sorted({int(s) for s in smids.tolist()})
    if len(seen) != sms or seen[0] < 0 or seen[-1] >= slots:
        logger.warning(f"die-local: probe saw {len(seen)} distinct SMs of {sms}; no die table")
        return None
    table = [-1] * (seen[-1] + 1)
    gaps = []
    for s in seen:
        near0, near1 = float(medians[0][s]), float(medians[1][s])
        table[s] = 0 if near0 < near1 else 1
        gaps.append(abs(near0 - near1))
    per_die = (table.count(0), table.count(1))
    if min(per_die) == 0 or statistics.median(gaps) < _PROBE_MIN_GAP_CYCLES:
        logger.warning(
            f"die-local: no clean die split (SMs per die {per_die}, median gap "
            f"{statistics.median(gaps):.0f} cycles); no die table"
        )
        return None
    logger.info(
        f"die-local: SMs per die {per_die}, median near/far gap "
        f"{statistics.median(gaps):.0f} cycles"
    )
    return tuple(table)


def smid_die_table(device: int) -> Optional[torch.Tensor]:
    """int32 tensor on ``device`` mapping ``%smid`` to its die (0/1), or None when the
    device does not have exactly two locality domains or the probe finds no clean split.

    Measured once per device and process (one probe kernel, two 2 MiB-granular buffers).
    """
    device = int(device)
    if locality_domain_count(device) != 2:
        return None
    table = _smid_die_table_cached(device)
    if table is None:
        return None
    return torch.tensor(table, dtype=torch.int32, device=torch.device("cuda", device))


class StreamScratch:
    """Zeroed int32 scratch tensors for the die-local tile queues, one set per CUDA stream.

    The die-local GEMMs zero their queue counters before they exit, so launches ordered on
    one stream can reuse a set without a memset. Launches on two streams can overlap (the
    chunked MoE path runs alternate chunks of a layer on the main and an aux stream), so each
    stream gets its own set. Every set is zeroed here, outside any CUDA-graph capture, so a
    stream first seen during capture takes a set that is already zero. Beyond ``max_streams``
    streams, each call gets freshly zeroed tensors (correct, one memset more per call).

    CUDA graphs keep the set of the stream they were captured on, so two graphs captured on
    one stream must not be replayed concurrently.
    """

    def __init__(self, sizes: Sequence[int], device: torch.device, max_streams: int = 8):
        if max_streams < 1 or not sizes or min(sizes) < 1:
            raise ValueError(f"bad scratch layout: sizes={tuple(sizes)}, max_streams={max_streams}")
        self._sizes = tuple(int(n) for n in sizes)
        self._device = torch.device(device)
        self._sets = [self._zeros() for _ in range(max_streams)]
        self._slot_of_stream: Dict[int, int] = {}

    def _zeros(self) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.zeros(n, dtype=torch.int32, device=self._device) for n in self._sizes)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """The scratch set of the current stream."""
        stream = torch.cuda.current_stream(self._device).cuda_stream
        slot = self._slot_of_stream.get(stream)
        if slot is None:
            if len(self._slot_of_stream) == len(self._sets):
                return self._zeros()
            slot = len(self._slot_of_stream)
            self._slot_of_stream[stream] = slot
        return self._sets[slot]

    def all_sets(self) -> List[Tuple[torch.Tensor, ...]]:
        """Every pre-zeroed set (for checks that the kernels left them zeroed)."""
        return list(self._sets)
