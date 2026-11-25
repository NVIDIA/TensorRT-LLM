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
"""Die locality on two-die GPUs (Rubin R200, SM107): die-alternating placement, SM -> die table.

An R200 is two dies (the CUDA driver's "locality domains"), each with its own L2 and HBM stacks.
Default allocations interleave the dies every 4 KiB, so a kernel streaming a weight sends about
half of its bytes over the die-to-die link, which bounds it at ~13 TB/s; reads of memory homed on
the reading SM's own die reach ~18 TB/s.

- :func:`die_alternating_copy` copies a tensor into one virtual address range whose 2 MiB chunks
  alternate between the dies: the range starts on a 4 MiB boundary and chunk ``i`` is physical
  memory of die ``i % 2`` (``cuMemCreate`` with ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``).
  So the die of any byte of the copy is bit 21 of its address, and the copy keeps the original's
  byte layout (strides, TMA descriptors and results are unchanged).
- :func:`sm_die_table` is the die of every SM (``%smid``), measured once per device: every SM
  times cold loads into memory of either die and the faster die is its own.

A kernel that knows both can give each SM the tiles whose bytes sit on its die.

The method (localized 2 MiB chunks mapped into one VA range, the cold-load latency probe and its
timed-load PTX sequence) follows vLLM's Kimi-K3 Rubin ``dies.py`` (Apache-2.0, Copyright
contributors to the vLLM project); the code here is written for TensorRT-LLM.
"""

import ctypes
import functools
import statistics
import weakref
from typing import Optional

import cuda.bindings.driver as cuda
import torch

from tensorrt_llm.logger import logger

__all__ = [
    "DIE_CHUNK_BYTES",
    "die_alternating_allocations",
    "die_alternating_copy",
    "is_die_alternating",
    "locality_domain_count",
    "pack_sm_die_words",
    "sm_die_table",
]

DIE_CHUNK_BYTES = 2 << 20
# Chunk i of a die-alternating range is on die i % 2 and the range starts on a multiple of two
# chunks, so bit 21 of an address is its die.
_RANGE_ALIGNMENT = 2 * DIE_CHUNK_BYTES
# Cold loads: ~700 cycles to the near die, ~1100 to the far one; every SM must show this gap.
_MIN_GAP_CYCLES = 150
_PROBE_LOADS = 16
# Smallest share of the SMs each die must get for the table to be trusted.
_MIN_DIE_SHARE = 0.4

# Base address -> (size, physical allocations) of every live die-alternating range.
_ALTERNATING_RANGES: dict[int, tuple[int, int]] = {}


def _checked(result, operation: str):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def locality_domain_count(device_index: int) -> int:
    """Locality domains (dies) of a CUDA device; 1 when the driver reports none."""
    torch.cuda.init()
    device = _checked(cuda.cuDeviceGet(device_index), "cuDeviceGet")
    attribute = getattr(cuda.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None)
    if attribute is None:
        return 1
    status, count = cuda.cuDeviceGetAttribute(attribute, device)
    return count if status == cuda.CUresult.CUDA_SUCCESS and count > 1 else 1


def _die_prop(device_index: int, die: int) -> "cuda.CUmemAllocationProp":
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.localized.deviceId = device_index
    prop.location.localized.localityDomainId = die
    return prop


class _DieAlternatingMemory:
    """One VA range of ``nbytes`` (rounded up to 2 MiB chunks) whose chunk ``i`` is on die ``i % 2``.

    Each die's chunks come from one physical allocation mapped piecewise into the range; where
    the driver does not map at a non-zero offset into an allocation, every chunk gets its own.
    Exposes ``__cuda_array_interface__`` (uint8): ``torch.as_tensor`` keeps this object, and with
    it the mapping, alive for the tensor's lifetime.
    """

    def __init__(self, nbytes: int, device_index: int):
        chunks = -(-nbytes // DIE_CHUNK_BYTES)
        self.nbytes = chunks * DIE_CHUNK_BYTES
        self.device_index = device_index
        self._handles: list = []
        self._mapped = [False]
        self.base = int(
            _checked(
                cuda.cuMemAddressReserve(self.nbytes, _RANGE_ALIGNMENT, 0, 0),
                "cuMemAddressReserve",
            )
        )
        self._finalizer = weakref.finalize(
            self,
            _DieAlternatingMemory._release,
            self.base,
            self.nbytes,
            self._handles,
            self._mapped,
        )
        if not self._map_from_one_allocation_per_die(chunks):
            self._map_one_allocation_per_chunk(chunks)
        self._mapped[0] = True
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _checked(cuda.cuMemSetAccess(self.base, self.nbytes, [access], 1), "cuMemSetAccess")
        _ALTERNATING_RANGES[self.base] = (self.nbytes, len(self._handles))

    def _map_from_one_allocation_per_die(self, chunks: int) -> bool:
        """Map chunk ``i`` from offset ``(i // 2) * 2 MiB`` of die ``i % 2``'s allocation; False
        (nothing left mapped or allocated) if the driver rejects mapping at an offset."""
        handles = []
        for die in (0, 1):
            die_chunks = (chunks + 1 - die) // 2
            if die_chunks == 0:
                handles.append(None)
                continue
            prop = _die_prop(self.device_index, die)
            handles.append(
                _checked(cuda.cuMemCreate(die_chunks * DIE_CHUNK_BYTES, prop, 0), "cuMemCreate")
            )
        mapped = 0
        for chunk in range(chunks):
            status, *_ = cuda.cuMemMap(
                self.base + chunk * DIE_CHUNK_BYTES,
                DIE_CHUNK_BYTES,
                (chunk // 2) * DIE_CHUNK_BYTES,
                handles[chunk % 2],
                0,
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                if mapped:
                    cuda.cuMemUnmap(self.base, mapped * DIE_CHUNK_BYTES)
                for handle in handles:
                    if handle is not None:
                        cuda.cuMemRelease(handle)
                return False
            mapped += 1
        self._handles.extend(handle for handle in handles if handle is not None)
        return True

    def _map_one_allocation_per_chunk(self, chunks: int) -> None:
        props = [_die_prop(self.device_index, die) for die in (0, 1)]
        for chunk in range(chunks):
            handle = _checked(cuda.cuMemCreate(DIE_CHUNK_BYTES, props[chunk % 2], 0), "cuMemCreate")
            self._handles.append(handle)
            _checked(
                cuda.cuMemMap(self.base + chunk * DIE_CHUNK_BYTES, DIE_CHUNK_BYTES, 0, handle, 0),
                "cuMemMap",
            )

    @staticmethod
    def _release(base: int, nbytes: int, handles: list, mapped: list) -> None:
        _ALTERNATING_RANGES.pop(base, None)
        if mapped[0]:
            cuda.cuMemUnmap(base, nbytes)
        for handle in handles:
            cuda.cuMemRelease(handle)
        cuda.cuMemAddressFree(base, nbytes)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
        }


@functools.cache
def _libcuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _chunk_die(address: int) -> Optional[int]:
    """Locality domain the driver reports for ``address`` (None if it reports none)."""
    attribute = getattr(
        cuda.CUpointer_attribute, "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL", None
    )
    if attribute is None:
        return None
    value = ctypes.c_int(-1)
    status = _libcuda().cuPointerGetAttribute(
        ctypes.byref(value), ctypes.c_int(int(attribute)), ctypes.c_uint64(address)
    )
    return value.value if status == 0 and value.value >= 0 else None


def die_alternating_copy(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """A contiguous copy of ``tensor`` (same shape, dtype, bytes) in die-alternating memory.

    The copy starts on a 4 MiB boundary and its 2 MiB chunk ``i`` is homed on die ``i % 2``, so
    the die of a byte at address ``a`` is ``(a >> 21) & 1``. The copy is issued on the current
    stream. Returns None on devices with fewer than two dies or when the driver reports a chunk
    on another die than requested.

    Args:
        tensor: A CUDA tensor of any shape and dtype.
    """
    if not tensor.is_cuda:
        raise ValueError("die_alternating_copy needs a CUDA tensor")
    device_index = tensor.device.index if tensor.device.index is not None else 0
    if locality_domain_count(device_index) < 2:
        return None
    nbytes = tensor.numel() * tensor.element_size()
    with torch.cuda.device(device_index):
        memory = _DieAlternatingMemory(max(nbytes, 1), device_index)
        chunks = memory.nbytes // DIE_CHUNK_BYTES
        for chunk in sorted({0, 1, chunks // 2, chunks - 1} & set(range(chunks))):
            die = _chunk_die(memory.base + chunk * DIE_CHUNK_BYTES)
            if die is not None and die != chunk % 2:
                logger.warning(
                    f"die_alternating_copy: chunk {chunk} reports die {die}, "
                    f"requested {chunk % 2}; keeping the default placement"
                )
                return None
        raw = torch.as_tensor(memory, device=tensor.device)
    out = raw[:nbytes].view(tensor.dtype).view(tensor.shape)
    out.copy_(tensor)
    return out


def is_die_alternating(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` starts a live :func:`die_alternating_copy` range."""
    return tensor.is_cuda and tensor.data_ptr() in _ALTERNATING_RANGES


def die_alternating_allocations(tensor: torch.Tensor) -> int:
    """Physical allocations behind a :func:`die_alternating_copy` (2, or one per 2 MiB chunk where
    the driver does not map at an offset into an allocation); 0 for other tensors."""
    if not tensor.is_cuda:
        return 0
    return _ALTERNATING_RANGES.get(tensor.data_ptr(), (0, 0))[1]


def pack_sm_die_words(table: tuple[int, ...]) -> tuple[int, ...]:
    """Bit ``s % 32`` of word ``s // 32`` is the die of smid ``s`` (unsigned 32-bit words)."""
    words = [0] * (-(-len(table) // 32))
    for smid, die in enumerate(table):
        if die == 1:
            words[smid // 32] |= 1 << (smid % 32)
    return tuple(words)


class _SmDieProbe:
    """One CTA per SM (shared memory forces it): thread 0 records ``%smid`` at ``smids[cta]``
    and times one cold load at each non-zero ``addresses[smid, i]`` into ``cycles[smid, i]``."""

    def __init__(self, smem_bytes: int):
        self.smem_bytes = smem_bytes

    def compile(self, smids, addresses, cycles, stream):
        import cutlass.cute as cute

        return cute.compile(self._launch, smids, addresses, cycles, stream)

    @functools.cached_property
    def _launch(self):
        import cutlass
        import cutlass.cute as cute
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import dsl_user_op

        @dsl_user_op
        def timed_load(address, *, loc=None, ip=None):
            # The address depends on the first clock read and the second clock read waits for
            # the loaded value, so the interval covers exactly one load (sequence from vLLM's
            # dies.py ``timed_load``).
            result = llvm.inline_asm(
                cutlass.Int64.mlir_type,
                [cutlass.Int64(address).ir_value(loc=loc, ip=ip)],
                "{ .reg .u64 t0, t1, z, a; .reg .u32 v; .reg .pred p;"
                " mov.u64 t0, %clock64; shr.u64 z, t0, 63; add.u64 a, $1, z;"
                " ld.global.cg.u32 v, [a]; setp.ne.u32 p, v, 0x7fffffff;"
                " mov.u64 t1, 0; @p mov.u64 t1, %clock64; sub.u64 $0, t1, t0; }",
                "=l,l,~{memory}",
                has_side_effects=True,
                loc=loc,
                ip=ip,
            )
            return cutlass.Int64(result)

        smem_bytes = self.smem_bytes

        @cute.kernel
        def kernel(smids: cute.Tensor, addresses: cute.Tensor, cycles: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            if tidx == 0:
                sm = cute.arch.smid()
                smids[bidx] = sm
                if sm < addresses.shape[0]:
                    for i in cutlass.range(addresses.shape[1]):
                        address = addresses[sm, i]
                        if address != 0:
                            cycles[sm, i] = timed_load(address)

        @cute.jit
        def launch(smids: cute.Tensor, addresses: cute.Tensor, cycles: cute.Tensor, stream):
            kernel(smids, addresses, cycles).launch(
                grid=(smids.shape[0], 1, 1),
                block=(32, 1, 1),
                smem=smem_bytes,
                stream=stream,
            )

        return launch


def _probe_cycles(device_index: int) -> tuple[torch.Tensor, list[int]]:
    """Cycles ``[slots, 2 * (1 + loads)]`` of every SM's loads (per die: one TLB warm-up load,
    then ``loads`` cold ones) and the smids of the CTAs that ran."""
    from cutlass.cute.runtime import from_dlpack

    props = torch.cuda.get_device_properties(device_index)
    sms = props.multi_processor_count
    slots = 2 * sms
    device = torch.device("cuda", device_index)
    memory = _DieAlternatingMemory(2 * DIE_CHUNK_BYTES, device_index)
    loads = _PROBE_LOADS
    addresses = torch.zeros((slots, 2, 1 + loads), dtype=torch.int64)
    # Lines 256 B apart, distinct for every (smid, load) up to 8190 of them; the last 256 B of
    # each chunk take the TLB warm-up loads.
    lines = torch.arange(slots * loads, dtype=torch.int64).view(slots, loads)
    offsets = 256 * (lines % (DIE_CHUNK_BYTES // 256 - 2) + 1)
    for die in (0, 1):
        chunk = memory.base + die * DIE_CHUNK_BYTES
        addresses[:, die, 0] = chunk + DIE_CHUNK_BYTES - 128
        addresses[:, die, 1:] = chunk + offsets
    addresses = addresses.view(slots, -1).to(device)
    cycles = torch.zeros(addresses.shape, dtype=torch.int64, device=device)
    smids = torch.full((sms,), -1, dtype=torch.int32, device=device)
    # Every probed line cold, no dirty lines left to write back during the probe: write 4x the
    # L2, then read it back.
    l2_bytes = getattr(props, "L2_cache_size", 128 << 20) or (128 << 20)
    flush = torch.ones(max(4 * l2_bytes, 256 << 20) // 4, dtype=torch.int32, device=device)
    flush.sum()
    del flush
    smem_bytes = max(props.shared_memory_per_multiprocessor // 2 + 4096, 48 << 10)
    smem_bytes = min(smem_bytes, props.shared_memory_per_block_optin)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    args = (from_dlpack(smids), from_dlpack(addresses), from_dlpack(cycles))
    compiled = _SmDieProbe(smem_bytes).compile(*args, stream)
    compiled(*args, stream)
    torch.cuda.synchronize(device)
    del memory
    return cycles.view(slots, 2, 1 + loads)[:, :, 1:].cpu(), sorted(int(s) for s in smids.cpu())


# Device index -> its measured SM -> die table (None: no trusted table).
_SM_DIE_TABLES: dict[int, Optional[tuple[int, ...]]] = {}


def sm_die_table(device_index: int) -> Optional[tuple[int, ...]]:
    """Die (0 or 1) of every smid of a two-die device, measured once per process and device.

    Every SM times cold loads to memory of die 0 and of die 1 (one CTA per SM); the die with the
    lower median latency is the SM's own. Returns None on devices with fewer than two dies, or
    when the measurement is not clear-cut: an SM without a near / far gap of at least 150 cycles,
    or a die with fewer than 40% of the SMs. The measurement allocates, runs a kernel and
    synchronizes: during CUDA graph capture a table not measured before is None.
    """
    if device_index not in _SM_DIE_TABLES:
        if torch.cuda.is_current_stream_capturing():
            return None
        _SM_DIE_TABLES[device_index] = _measure_sm_die_table(device_index)
    return _SM_DIE_TABLES[device_index]


def _measure_sm_die_table(device_index: int) -> Optional[tuple[int, ...]]:
    if locality_domain_count(device_index) < 2:
        return None
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    with torch.cuda.device(device_index):
        cycles, seen = _probe_cycles(device_index)
    if seen[0] < 0 or len(set(seen)) != sms or seen[-1] >= cycles.shape[0]:
        logger.warning(f"sm_die_table: probe CTAs not one per SM ({seen[:4]}..{seen[-4:]})")
        return None
    table = [0] * (seen[-1] + 1)
    unclear = []
    for smid in seen:
        near, far = (statistics.median(cycles[smid, die].tolist()) for die in (0, 1))
        if abs(near - far) < _MIN_GAP_CYCLES:
            unclear.append(smid)
        table[smid] = 0 if near < far else 1
    counts = (table.count(0), table.count(1))
    if unclear or min(counts) < _MIN_DIE_SHARE * sms:
        logger.warning(
            f"sm_die_table: no clear SM -> die map (unclear smids {unclear[:8]}, "
            f"SMs per die {counts})"
        )
        return None
    logger.info(f"sm_die_table: device {device_index}: SMs per die {counts}")
    return tuple(table)
