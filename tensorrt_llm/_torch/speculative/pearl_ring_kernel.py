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
#
# Two kernels:
#   * ``pearl_write_ring_slot`` (stage 2): writes a fully CPU-built frame.
#   * ``pearl_compose_and_send`` (stage 3): receives scalar protocol fields
#     and a GPU pointer to the verify output; reads the last accepted
#     token directly from GPU and assembles the protocol frame on-the-fly
#     before writing it to the peer ring slot. This eliminates the
#     CPU-side ``.item()`` on the verify path -- the last-token read
#     happens entirely on the GPU, in the same kernel that emits the
#     ring write.
_KERNEL_SRC = b"""
extern "C" {

struct PearlPacket {
    unsigned int imm;          // [0:4]
    unsigned char payload[96]; // [4:100]
};

// Stage 2: write a 100-byte PEARL frame into the ring slot at
// ``offset_bytes``, then issue a system-scope fence so the peer
// process's consumer can observe the data before its CPU reads our
// bumped head counter.
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

// Stage 3: assemble the protocol frame directly from scalar fields +
// a GPU read of the last accepted token. The wire layout mirrors
// ``DraftApiProtocol._WIRE_STRUCT`` (LE: II BBB 5s 20I): 4-byte imm,
// then 96-byte payload with round_seq / position / version /
// message_type / num_tokens / 5 reserved bytes / 20 tokens.
__global__ void pearl_compose_and_send(
    unsigned char* ring,
    int offset_bytes,
    unsigned int imm,
    unsigned int round_seq,
    unsigned int position,
    unsigned char version,
    unsigned char message_type,
    unsigned char num_tokens,
    const int* accepted_tokens_row,    // length >= max_draft_len
    const int* num_accepted_ptr,       // single int on GPU
    int max_draft_len
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Read the last accepted token entirely on the GPU. The verify
        // sampler wrote ``num_accepted`` and ``accepted_tokens`` upstream;
        // the launching CUDA stream guarantees those stores are visible
        // here.
        int n = num_accepted_ptr ? num_accepted_ptr[0] : 1;
        int idx = n - 1;
        if (idx < 0) idx = 0;
        if (idx >= max_draft_len) idx = max_draft_len - 1;
        int last_token = accepted_tokens_row[idx];

        unsigned char* dst = ring + offset_bytes;

        // Frame [0:4] = imm_data
        *(unsigned int*)(dst + 0) = imm;
        // Payload starts at offset 4 (DraftApiProtocol layout).
        *(unsigned int*)(dst + 4) = round_seq;     // [4:8]
        *(unsigned int*)(dst + 8) = position;      // [8:12]
        dst[12] = version;                         // [12]
        dst[13] = message_type;                    // [13]
        dst[14] = num_tokens;                      // [14]
        // Reserved bytes [15:20].
        dst[15] = 0; dst[16] = 0; dst[17] = 0; dst[18] = 0; dst[19] = 0;
        // tokens[0] -- the verified last token.
        *(int*)(dst + 20) = last_token;            // [20:24]
        // tokens[1..19] zero out (4 * 19 = 76 bytes).
        for (int i = 1; i < 20; ++i) {
            *(int*)(dst + 20 + i * 4) = 0;
        }

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
_compose_function_handle = None
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
            raise RuntimeError(f"cuModuleGetFunction(pearl_write_ring_slot) failed: {int(status)}")
        status, fn_compose = driver.cuModuleGetFunction(mod, b"pearl_compose_and_send")
        if int(status) != 0:
            raise RuntimeError(f"cuModuleGetFunction(pearl_compose_and_send) failed: {int(status)}")
        _module_handle = mod
        _function_handle = fn
        global _compose_function_handle
        _compose_function_handle = fn_compose
        return _function_handle


def _ensure_compose_compiled():
    """Return the compose kernel handle, compiling the module on first use."""
    _ensure_compiled()
    return _compose_function_handle


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


class RingComposeLauncher:
    """Stage 3 launcher: assembles the protocol frame inside the kernel,
    reading the verified ``last_token`` from a GPU tensor pointer.

    The CPU only has to hand over scalar protocol fields (round_seq,
    position, version, message_type, num_tokens) plus device pointers
    to ``accepted_tokens_row`` and ``num_accepted_ptr``. The kernel
    does the indexed read on the GPU and emits the ring write, removing
    the per-send ``.item()`` that the stage-2 path still needs.

    Like ``RingWriteLauncher``, ctypes scratch is pre-allocated and
    reused across launches to keep Python overhead off the hot path.
    """

    __slots__ = (
        "_fn",
        "_driver",
        "_stream",
        "_ring_arg",
        "_off_arg",
        "_imm",
        "_round_seq",
        "_position",
        "_ver",
        "_msg_type",
        "_num_tokens",
        "_acc_ptr",
        "_num_acc_ptr",
        "_max_draft_len",
        "_args",
    )

    def __init__(self, stream_handle: int):
        from cuda.bindings import driver

        self._fn = _ensure_compose_compiled()
        self._driver = driver
        self._stream = stream_handle
        self._ring_arg = ctypes.c_uint64(0)
        self._off_arg = ctypes.c_int32(0)
        self._imm = ctypes.c_uint32(0)
        self._round_seq = ctypes.c_uint32(0)
        self._position = ctypes.c_uint32(0)
        self._ver = ctypes.c_uint8(0)
        self._msg_type = ctypes.c_uint8(0)
        self._num_tokens = ctypes.c_uint8(0)
        self._acc_ptr = ctypes.c_uint64(0)
        self._num_acc_ptr = ctypes.c_uint64(0)
        self._max_draft_len = ctypes.c_int32(0)
        self._args = (ctypes.c_void_p * 11)(
            ctypes.addressof(self._ring_arg),
            ctypes.addressof(self._off_arg),
            ctypes.addressof(self._imm),
            ctypes.addressof(self._round_seq),
            ctypes.addressof(self._position),
            ctypes.addressof(self._ver),
            ctypes.addressof(self._msg_type),
            ctypes.addressof(self._num_tokens),
            ctypes.addressof(self._acc_ptr),
            ctypes.addressof(self._num_acc_ptr),
            ctypes.addressof(self._max_draft_len),
        )

    def compose_and_send(
        self,
        ring_ptr: int,
        slot_offset: int,
        imm: int,
        round_seq: int,
        position: int,
        version: int,
        message_type: int,
        num_tokens: int,
        accepted_tokens_row_ptr: int,
        num_accepted_ptr: int,
        max_draft_len: int,
    ) -> None:
        self._ring_arg.value = int(ring_ptr)
        self._off_arg.value = int(slot_offset)
        self._imm.value = int(imm) & 0xFFFFFFFF
        self._round_seq.value = int(round_seq) & 0xFFFFFFFF
        self._position.value = int(position) & 0xFFFFFFFF
        self._ver.value = int(version) & 0xFF
        self._msg_type.value = int(message_type) & 0xFF
        self._num_tokens.value = int(num_tokens) & 0xFF
        self._acc_ptr.value = int(accepted_tokens_row_ptr)
        self._num_acc_ptr.value = int(num_accepted_ptr)
        self._max_draft_len.value = int(max_draft_len)
        (status,) = self._driver.cuLaunchKernel(
            self._fn,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            self._stream,
            self._args,
            0,
        )
        if int(status) != 0:
            raise RuntimeError(f"cuLaunchKernel(pearl_compose_and_send) failed: {int(status)}")


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


__all__ = ["write_ring_slot", "RingWriteLauncher", "RingComposeLauncher"]


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
