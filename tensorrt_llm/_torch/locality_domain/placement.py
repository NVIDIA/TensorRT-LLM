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
"""Which locality domain holds a tensor's bytes, and which domain each SM belongs to.

* :func:`split_domain_copy` copies a tensor into one virtual address range whose leading bytes
  are memory of locality domain 0 and the rest memory of domain 1, so a kernel that knows where
  the split is can read every byte from its own domain.
* :func:`sm_locality_domains` is the domain of every SM (``%smid``), observed by running a probe
  kernel on each domain's localized stream.
"""

import functools
import weakref

import torch
from cuda.bindings import driver as cuda

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._torch.locality_domain_utils import (
    get_locality_domain_stream,
    initialize_locality_domain_resources,
)

__all__ = ["split_domain_copy", "sm_locality_domains"]

# CU_MEM_HANDLE_TYPE_NONE and the automatic allocation usage of non-RDMA memory.
_HANDLE_TYPES = 0
_USAGE = 0


def _check(result, operation: str):
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {err}")
    return values[0] if len(values) == 1 else values


class _SplitDomainMemory:
    """One VA range of ``nbytes`` whose first ``domain0_bytes`` are one physical allocation on
    locality domain 0 and the rest one allocation on domain 1 (exposed as a uint8 CUDA array)."""

    def __init__(self, nbytes: int, domain0_bytes: int, granularity: int, device_index: int):
        self.nbytes = nbytes
        self.base = int(_check(cuda.cuMemAddressReserve(nbytes, granularity, 0, 0), "reserve"))
        handles: list[int] = []
        mapped: list[tuple[int, int]] = []
        self._finalizer = weakref.finalize(
            self, _SplitDomainMemory._release, self.base, nbytes, handles, mapped
        )
        localization = _tbr.LocalizationHandle()
        for domain, offset, size in (
            (0, 0, domain0_bytes),
            (1, domain0_bytes, nbytes - domain0_bytes),
        ):
            if size == 0:
                continue
            err, handle = localization.try_create_localized_allocation_handle(
                size, domain, _HANDLE_TYPES, False, _USAGE
            )
            if err == int(cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY):
                # Bytes freed by earlier re-homed tensors sit in PyTorch's cache, not the driver's.
                torch.cuda.empty_cache()
                err, handle = localization.try_create_localized_allocation_handle(
                    size, domain, _HANDLE_TYPES, False, _USAGE
                )
            if err != 0:
                raise RuntimeError(f"localized allocation on domain {domain} failed: {err}")
            handles.append(handle)
            _check(cuda.cuMemMap(self.base + offset, size, 0, handle, 0), "cuMemMap")
            mapped.append((self.base + offset, size))
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cuda.cuMemSetAccess(self.base, nbytes, [access], 1), "cuMemSetAccess")

    @staticmethod
    def _release(base: int, nbytes: int, handles: list, mapped: list) -> None:
        for address, size in mapped:
            cuda.cuMemUnmap(address, size)
        for handle in handles:
            cuda.cuMemRelease(handle)
        cuda.cuMemAddressFree(base, nbytes)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {"shape": (self.nbytes,), "typestr": "|u1", "data": (self.base, False), "version": 3}


def split_domain_copy(tensor: torch.Tensor, split_bytes: int) -> torch.Tensor | None:
    """A contiguous copy of ``tensor`` (same shape, dtype and bytes) whose bytes before the
    allocation-granularity boundary nearest ``split_bytes`` are memory of locality domain 0 and
    the rest memory of domain 1.

    The copy is issued on the current stream and owns its memory. Returns None when the device
    does not support localized allocations.
    """
    device_index = tensor.device.index
    localization = _tbr.LocalizationHandle()
    if not localization.supports_memory_localization():
        return None
    granularity = 0
    for domain in (0, 1):
        err, domain_granularity = localization.try_get_localized_allocation_granularity(
            domain, _HANDLE_TYPES, False, _USAGE
        )
        if err != 0 or domain_granularity == 0:
            return None
        granularity = max(granularity, domain_granularity)
    nbytes = tensor.numel() * tensor.element_size()
    padded = -(-nbytes // granularity) * granularity
    domain0_bytes = min((split_bytes + granularity // 2) // granularity * granularity, padded)
    with torch.cuda.device(device_index):
        memory = _SplitDomainMemory(padded, domain0_bytes, granularity, device_index)
        raw = torch.as_tensor(memory, device=tensor.device)
    out = raw[:nbytes].view(tensor.dtype).view(tensor.shape)
    out.copy_(tensor)
    return out


@functools.cache
def _sm_probe():
    """A kernel whose CTA (one per SM: the shared memory admits no second one) marks its SM and
    stays resident for a while, so a grid of one CTA per SM visits every SM it may run on."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    def _asm(res, text, constraints, operands, *, loc=None, ip=None):
        return llvm.inline_asm(
            res,
            operands,
            text,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def smid(*, loc=None, ip=None):
        return Int32(_asm(T.i32(), "mov.u32 $0, %smid;", "=r", [], loc=loc, ip=ip))

    @dsl_user_op
    def globaltimer(*, loc=None, ip=None):
        return Int64(_asm(T.i64(), "mov.u64 $0, %globaltimer;", "=l", [], loc=loc, ip=ip))

    @cute.kernel
    def kernel(seen: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            seen[smid()] = Int32(1)
            start = globaltimer()
            while globaltimer() - start < Int64(50000):
                pass

    @cute.jit
    def launch(seen: cute.Tensor, num_ctas: cutlass.Constexpr, smem: cutlass.Constexpr, stream):
        kernel(seen).launch(grid=[num_ctas, 1, 1], block=[32, 1, 1], smem=smem, stream=stream)

    return launch


@functools.cache
def sm_locality_domains(device_index: int) -> tuple[int, ...]:
    """Locality domain of every SM of the device (SMs outside both localized partitions: 0).

    Runs one probe grid on each domain's localized stream, which only schedules onto that
    domain's SMs; call it outside CUDA graph capture.
    """
    import cutlass.cute as cute
    from cuda.bindings import driver as cu
    from cutlass.cute.runtime import from_dlpack

    props = torch.cuda.get_device_properties(device_index)
    num_sms = props.multi_processor_count
    smem = props.shared_memory_per_block_optin // 2 + 1024
    launch = _sm_probe()
    domains = [0] * num_sms
    with torch.cuda.device(device_index):
        initialize_locality_domain_resources()
        for domain in (0, 1):
            stream = get_locality_domain_stream(domain)
            seen = torch.zeros(num_sms, dtype=torch.int32, device=f"cuda:{device_index}")
            stream.wait_stream(torch.cuda.current_stream())
            cute.compile(launch, from_dlpack(seen), num_sms, smem, cu.CUstream(stream.cuda_stream))(
                from_dlpack(seen), cu.CUstream(stream.cuda_stream)
            )
            stream.synchronize()
            for sm in seen.nonzero().flatten().tolist():
                domains[sm] = domain
    return tuple(domains)
