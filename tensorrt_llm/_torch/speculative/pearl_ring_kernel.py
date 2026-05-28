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
"""Single-thread CUDA kernel that copies a PEARL packet from the kernel
parameter buffer into a peer's IPC ring slot.

Stage 2 of the CUDA-IPC progression uses this kernel **instead of**
``cudaMemcpyAsync(host->device, 100B)`` on the send path. The win
(measurable in stage 3 once the launch is graph-captured) is that the
100-byte payload travels through the kernel parameter buffer rather
than via a host pinned staging buffer + async memcpy. Per packet that
saves one pinned-buffer write, one ``cudaMemcpyAsync`` API call, and
one stream sync; in graph-replay mode it also eliminates the per-cycle
host-device sync entirely.

The kernel is compiled once at module load via NVRTC -> CUBIN
(``sm_100`` on Blackwell B200). PTX is intentionally skipped because
the locally-installed driver rejected our ``compute_90`` PTX with
``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``; building directly for the
target arch sidesteps the JIT mismatch.

The kernel issues ``__threadfence_system()`` after the store so the
ring slot is visible to the consumer in the peer process before the
caller bumps the meta head counter.
"""

from __future__ import annotations

import ctypes
import threading

# Source kept inline so we don't need a compile-time toolchain or a
# separate .cu file in the wheel. NVRTC produces a tiny CUBIN (~5 KB).
_KERNEL_SRC = b"""
extern "C" {

struct PearlPacket {
    unsigned int imm;          // [0:4]
    unsigned char payload[96]; // [4:100]
};

// Write a 100-byte PEARL frame into the ring slot at ``offset_bytes``,
// then issue a system-scope fence so the peer process's consumer can
// observe the data before its CPU reads our bumped head counter.
__global__ void pearl_write_ring_slot(
    unsigned char* ring,
    int offset_bytes,
    PearlPacket pkt
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        PearlPacket* slot = (PearlPacket*)(ring + offset_bytes);
        *slot = pkt;
        __threadfence_system();
    }
}

}  // extern "C"
"""


class _PearlPacket(ctypes.Structure):
    """Mirror of the ``PearlPacket`` struct in the kernel source.

    Used to pack the 100-byte frame as a kernel parameter (CUDA copies
    pass-by-value struct args into the kernel param buffer; the GPU
    reads them from constant memory, not from a separate device alloc).
    """

    _fields_ = [
        ("imm", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * 96),
    ]


_module_handle = None
_function_handle = None
_lock = threading.Lock()


def _ensure_compiled():
    """Compile + module-load the kernel on first use, cached process-wide."""
    global _module_handle, _function_handle
    if _function_handle is not None:
        return _function_handle
    with _lock:
        if _function_handle is not None:
            return _function_handle
        from cuda.bindings import driver, nvrtc
        from cuda.bindings import runtime as cudart

        # Force the CUDA primary context to exist before any driver-level
        # call. Without this, ``cuModuleLoadData`` fires CUDA_ERROR_INVALID_CONTEXT
        # because the runtime hasn't lazily attached one to this thread.
        cudart.cudaFree(0)

        status, prog = nvrtc.nvrtcCreateProgram(_KERNEL_SRC, b"pearl_ring_kernel.cu", 0, [], [])
        if int(status) != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {int(status)}")
        # Build directly for the device architecture so we get a CUBIN
        # the local driver can load without going through PTX JIT (see
        # module docstring).
        # Detect device CC at compile time so we don't hard-code sm_100.
        status, props = cudart.cudaGetDeviceProperties(0)
        if int(status) != 0:
            raise RuntimeError(f"cudaGetDeviceProperties failed: {int(status)}")
        arch = f"--gpu-architecture=sm_{int(props.major)}{int(props.minor)}".encode()
        opts = [arch]
        (status,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(status) != 0:
            (_, log_size) = nvrtc.nvrtcGetProgramLogSize(prog)
            log_buf = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(
                f"nvrtcCompileProgram failed: {int(status)}; "
                f"log:\n{log_buf.decode(errors='replace')}"
            )
        status, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = bytearray(cubin_size)
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        nvrtc.nvrtcDestroyProgram(prog)

        status, mod = driver.cuModuleLoadData(bytes(cubin))
        if int(status) != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {int(status)}")
        status, fn = driver.cuModuleGetFunction(mod, b"pearl_write_ring_slot")
        if int(status) != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {int(status)}")
        _module_handle = mod
        _function_handle = fn
        return _function_handle


class RingWriteLauncher:
    """Pre-allocated launcher state for repeatedly invoking the ring-write
    kernel from the hot path.

    Reusing the ctypes args (``_PearlPacket``, ``c_uint64`` ring, ``c_int32``
    offset, ``(c_void_p * 3)`` arg vector) across calls saves ~30% wall time
    per send vs allocating fresh on every launch -- measured on the PEARL
    cycle (~265 sends per perf run). The launcher is per-endpoint instance
    so the ring pointer and stream are baked in once.
    """

    __slots__ = (
        "_fn",
        "_driver",
        "_stream",
        "_ring_arg",
        "_off_arg",
        "_pkt",
        "_args",
        "_payload_addr",
    )

    def __init__(self, stream_handle: int):
        from cuda.bindings import driver

        self._fn = _ensure_compiled()
        self._driver = driver
        self._stream = stream_handle
        # Reusable ctypes storage. ``ctypes.addressof`` is cheap; keeping
        # references to the underlying objects keeps the addresses stable.
        self._ring_arg = ctypes.c_uint64(0)
        self._off_arg = ctypes.c_int32(0)
        self._pkt = _PearlPacket()
        self._args = (ctypes.c_void_p * 3)(
            ctypes.addressof(self._ring_arg),
            ctypes.addressof(self._off_arg),
            ctypes.addressof(self._pkt),
        )
        # Cache the payload's destination address so memmove can target it
        # without repeated attribute access.
        self._payload_addr = ctypes.addressof(self._pkt) + _PearlPacket.payload.offset

    def write(self, ring_ptr: int, slot_offset: int, imm: int, payload: bytes) -> None:
        if len(payload) != 96:
            raise ValueError(f"payload must be 96 bytes; got {len(payload)}")
        self._ring_arg.value = int(ring_ptr)
        self._off_arg.value = int(slot_offset)
        self._pkt.imm = int(imm) & 0xFFFFFFFF
        ctypes.memmove(self._payload_addr, payload, 96)
        (status,) = self._driver.cuLaunchKernel(
            self._fn,
            1,
            1,
            1,  # grid
            1,
            1,
            1,  # block
            0,  # shared mem
            self._stream,
            self._args,
            0,
        )
        if int(status) != 0:
            raise RuntimeError(f"cuLaunchKernel(pearl_write_ring_slot) failed: {int(status)}")


def write_ring_slot(
    stream_handle: int,
    ring_ptr: int,
    slot_offset: int,
    imm: int,
    payload: bytes,
) -> None:
    """One-shot wrapper for callers that don't want to manage launcher state.
    Per-call ctypes allocation -- prefer ``RingWriteLauncher`` in hot paths."""
    if len(payload) != 96:
        raise ValueError(f"payload must be 96 bytes; got {len(payload)}")
    from cuda.bindings import driver

    fn = _ensure_compiled()
    pkt = _PearlPacket()
    pkt.imm = int(imm) & 0xFFFFFFFF
    ctypes.memmove(pkt.payload, payload, 96)
    ring_arg = ctypes.c_uint64(int(ring_ptr))
    off_arg = ctypes.c_int32(int(slot_offset))
    args = (ctypes.c_void_p * 3)(
        ctypes.addressof(ring_arg),
        ctypes.addressof(off_arg),
        ctypes.addressof(pkt),
    )
    (status,) = driver.cuLaunchKernel(
        fn,
        1,
        1,
        1,  # grid
        1,
        1,
        1,  # block
        0,  # shared mem
        stream_handle,
        args,
        0,
    )
    if int(status) != 0:
        raise RuntimeError(f"cuLaunchKernel(pearl_write_ring_slot) failed: {int(status)}")


__all__ = ["write_ring_slot", "RingWriteLauncher"]


def _selftest():  # pragma: no cover -- manual smoke test
    from cuda.bindings import runtime as cudart

    cudart.cudaFree(0)
    (st, ring) = cudart.cudaMalloc(256 * 100)
    (st, stream) = cudart.cudaStreamCreate()
    payload = b"abcdefgh" * 12  # 96 bytes
    write_ring_slot(stream, ring, slot_offset=200, imm=0xDEADBEEF, payload=payload)
    cudart.cudaStreamSynchronize(stream)
    host = (ctypes.c_uint8 * 100)()
    cudart.cudaMemcpy(host, ring + 200, 100, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    blob = bytes(host)
    assert blob[:4] == (0xDEADBEEF).to_bytes(4, "little"), blob[:4].hex()
    assert blob[4:] == payload, blob[4:][:32]
    print("OK")


if __name__ == "__main__":
    _selftest()
