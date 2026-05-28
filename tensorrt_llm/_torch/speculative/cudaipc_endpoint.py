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
"""CUDA IPC transport implementing the ``Endpoint`` protocol (stage 1).

Same-machine peer of ``_ShmBackend``, but with the **data slots** living
in GPU device memory shared across processes via
``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``. The ring head/tail
metadata still lives in POSIX shared memory so polling stays a cheap
CPU memory read.

Stage 1 keeps the host-orchestrated send/recv path: each frame is
``cudaMemcpyAsync``-ed between a pinned host staging buffer and the GPU
ring slot. This is **not** a perf win over the pure-CPU shm backend on
its own — there's an extra host↔device hop per frame. The point of
this stage is to put the data plane in GPU memory so later stages can:

  Stage 2: replace the cudaMemcpy on the send side with a single-thread
           CUDA kernel that writes the prebuilt packet into the ring
           directly (no host->device API call per packet).
  Stage 3: tail-fuse the ring write into the model's verify/sampler
           kernel chain so the whole verify+send sequence is captured
           into the same CUDA graph as the forward pass.
  Stage 4: mirror stages 2-3 on the draft side so the full round trip
           is GPU-resident.

Layout (per direction):

  CPU shm region ``/dev/shm/<name>_meta_{t2d,d2t}`` (small, header only):
    [0:8]     head u64 (producer writes monotonically)
    [8:16]    tail u64 (consumer writes monotonically)
    [16:17]   stopped u8
    [17:24]   reserved
    [24:28]   ready magic u32 (set by server last)
    [28:32]   reserved
    [32:96]   IPC mem handle (64 bytes; server writes, client reads)
    [96:128]  reserved

  GPU device memory (256 slots × 100 bytes = 25 600 bytes per direction).
  Server allocates both rings on its own GPU; client maps both via IPC.

Memory ownership: the server (draft side) creates both rings + both
CPU meta regions and is responsible for cleanup. The client (target
side) only opens / closes the IPC handles and detaches from
``resource_tracker``.
"""

from __future__ import annotations

import ctypes
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

from .draft_api_protocol import DraftApiProtocol
from .shm_endpoint import _detach_from_resource_tracker
from .spec_decode_channel import EndpointPacket, EndpointStatus

# ---------------------------------------------------------------------------
# CUDA runtime API binding -- imported lazily so this module stays importable
# on hosts without a CUDA toolkit (mirrors the rest of the speculative
# module's import-time discipline).
# ---------------------------------------------------------------------------
_cudart = None


def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    from cuda.bindings import runtime as cudart  # type: ignore

    _cudart = cudart
    return _cudart


def _check(status, what: str) -> None:
    """Raise on non-zero CUDA runtime status, with the API context for
    debugging. We use ``RuntimeError`` rather than a CUDA-specific class
    so callers don't need to import cuda.bindings to ``except``."""
    if int(status) != 0:
        raise RuntimeError(f"cudaipc-endpoint: {what} failed, cudart status={int(status)}")


_FRAME_HEADER_BYTES = 4  # uint32 imm_data, little-endian
_FRAME_BYTES = _FRAME_HEADER_BYTES + DraftApiProtocol.kMessageBytes  # = 100

_RING_SLOTS = 256
_RING_DATA_BYTES = _RING_SLOTS * _FRAME_BYTES

_META_BYTES = 128
_META_HEAD_OFFSET = 0
_META_TAIL_OFFSET = 8
_META_STOPPED_OFFSET = 16
_META_READY_OFFSET = 24
_META_IPC_HANDLE_OFFSET = 32
_META_IPC_HANDLE_BYTES = 64
_READY_MAGIC = 0x50454152  # 'PEAR'


@dataclass
class CudaIpcEndpointConfig:
    """Connection config for ``_CudaIpcBackend``.

    ``name`` is the prefix of the two POSIX shared-memory metadata regions
    (``<name>_meta_t2d`` and ``<name>_meta_d2t``); the IPC handle for each
    direction's GPU ring lives inside its meta region.
    """

    is_server: bool = False
    name: str = "pearl_ipc_default"
    handshake_timeout_s: float = 120.0
    recv_queue_depth: int = 256
    payload_bytes: int = DraftApiProtocol.kMessageBytes


class _RingMeta:
    """Writable view onto a single CPU shm meta region."""

    __slots__ = ("_mv",)

    def __init__(self, mv):
        self._mv = mv

    def head(self) -> int:
        return struct.unpack_from("<Q", self._mv, _META_HEAD_OFFSET)[0]

    def tail(self) -> int:
        return struct.unpack_from("<Q", self._mv, _META_TAIL_OFFSET)[0]

    def stopped(self) -> int:
        return self._mv[_META_STOPPED_OFFSET]

    def ready(self) -> int:
        return struct.unpack_from("<I", self._mv, _META_READY_OFFSET)[0]

    def ipc_handle_bytes(self) -> bytes:
        return bytes(
            self._mv[_META_IPC_HANDLE_OFFSET : _META_IPC_HANDLE_OFFSET + _META_IPC_HANDLE_BYTES]
        )

    def set_head(self, v: int) -> None:
        struct.pack_into("<Q", self._mv, _META_HEAD_OFFSET, v)

    def set_tail(self, v: int) -> None:
        struct.pack_into("<Q", self._mv, _META_TAIL_OFFSET, v)

    def set_stopped(self) -> None:
        self._mv[_META_STOPPED_OFFSET] = 1

    def set_ready(self) -> None:
        struct.pack_into("<I", self._mv, _META_READY_OFFSET, _READY_MAGIC)

    def set_ipc_handle_bytes(self, b: bytes) -> None:
        if len(b) != _META_IPC_HANDLE_BYTES:
            raise ValueError(f"ipc handle must be {_META_IPC_HANDLE_BYTES} bytes, got {len(b)}")
        self._mv[_META_IPC_HANDLE_OFFSET : _META_IPC_HANDLE_OFFSET + _META_IPC_HANDLE_BYTES] = b

    def reset_header(self) -> None:
        self._mv[:_META_BYTES] = b"\x00" * _META_BYTES


class _CudaIpcBackend:
    """Endpoint Protocol implementation over CPU-meta + GPU-data shm.

    See module docstring for the data-flow / ownership contract.
    """

    def __init__(self, config: CudaIpcEndpointConfig):
        self._cfg = config

        # Side-relative names so the producer/consumer roles fall out
        # naturally regardless of which peer this instance lives on.
        if config.is_server:
            self._out_suffix = "_meta_d2t"  # this peer (draft) produces
            self._in_suffix = "_meta_t2d"  # this peer (draft) consumes
        else:
            self._out_suffix = "_meta_t2d"  # this peer (target) produces
            self._in_suffix = "_meta_d2t"  # this peer (target) consumes

        # CPU metadata
        self._shm_meta_out: Optional[shared_memory.SharedMemory] = None
        self._shm_meta_in: Optional[shared_memory.SharedMemory] = None
        self._meta_out: Optional[_RingMeta] = None
        self._meta_in: Optional[_RingMeta] = None

        # GPU data slot pointers (int, GPU virtual address):
        #   server side: _dev_out / _dev_in are local cudaMalloc'd pointers;
        #   client side: they are remote pointers obtained from
        #                cudaIpcOpenMemHandle.
        self._dev_out: int = 0
        self._dev_in: int = 0
        # Whether the GPU pointer is locally owned (server) or IPC-imported
        # (client). Drives the right free path on cleanup.
        self._dev_out_is_local: bool = False
        self._dev_in_is_local: bool = False

        # Pinned host staging buffers + CUDA stream for send/recv copies.
        self._pinned_send: int = 0
        self._pinned_recv: int = 0
        self._stream: int = 0

        self._recv_credits = 0
        self._started = False

    # ------------------------------------------------------------------
    # Endpoint Protocol
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            return EndpointStatus.kOk
        try:
            cudart = _load_cudart()

            # Stream + pinned host buffers used in send/recv. One stream
            # is enough since stage 1 issues sync after every copy.
            status, self._stream = cudart.cudaStreamCreate()
            _check(status, "cudaStreamCreate")
            status, self._pinned_send = cudart.cudaMallocHost(_FRAME_BYTES)
            _check(status, "cudaMallocHost(send)")
            status, self._pinned_recv = cudart.cudaMallocHost(_FRAME_BYTES)
            _check(status, "cudaMallocHost(recv)")
            ctypes.memset(self._pinned_send, 0, _FRAME_BYTES)
            ctypes.memset(self._pinned_recv, 0, _FRAME_BYTES)

            if self._cfg.is_server:
                # Allocate the two device rings on this peer's GPU.
                status, dev_out = cudart.cudaMalloc(_RING_DATA_BYTES)
                _check(status, "cudaMalloc(out ring)")
                self._dev_out = dev_out
                self._dev_out_is_local = True
                status, dev_in = cudart.cudaMalloc(_RING_DATA_BYTES)
                _check(status, "cudaMalloc(in ring)")
                self._dev_in = dev_in
                self._dev_in_is_local = True
                # Zero-init device memory so a stale slot can't be mistaken
                # for a real frame (head/tail logic in the channel layer
                # already prevents this, but defence-in-depth is cheap).
                (status,) = cudart.cudaMemset(self._dev_out, 0, _RING_DATA_BYTES)
                _check(status, "cudaMemset(out)")
                (status,) = cudart.cudaMemset(self._dev_in, 0, _RING_DATA_BYTES)
                _check(status, "cudaMemset(in)")

                # Get IPC handles to share with the client.
                status, h_out = cudart.cudaIpcGetMemHandle(self._dev_out)
                _check(status, "cudaIpcGetMemHandle(out)")
                status, h_in = cudart.cudaIpcGetMemHandle(self._dev_in)
                _check(status, "cudaIpcGetMemHandle(in)")
                ipc_out_bytes = bytes(h_out.reserved)
                ipc_in_bytes = bytes(h_in.reserved)

                # Create CPU meta regions and publish the IPC handles.
                self._shm_meta_out = self._create_or_recreate_shm(self._cfg.name + self._out_suffix)
                self._shm_meta_in = self._create_or_recreate_shm(self._cfg.name + self._in_suffix)
                _detach_from_resource_tracker(self._shm_meta_out)
                _detach_from_resource_tracker(self._shm_meta_in)
                self._meta_out = _RingMeta(self._shm_meta_out.buf)
                self._meta_in = _RingMeta(self._shm_meta_in.buf)
                self._meta_out.reset_header()
                self._meta_in.reset_header()
                # The OUT ring's IPC handle is the LOCAL d2t allocation
                # (server writes to it locally; client reads it via IPC).
                # The IN ring's IPC handle is the LOCAL t2d allocation
                # (server reads it locally; client writes to it via IPC).
                self._meta_out.set_ipc_handle_bytes(ipc_out_bytes)
                self._meta_in.set_ipc_handle_bytes(ipc_in_bytes)
                # Mark ready last so the client only observes a fully-
                # initialized region.
                self._meta_out.set_ready()
                self._meta_in.set_ready()
            else:
                deadline = time.monotonic() + max(0.0, self._cfg.handshake_timeout_s)
                while True:
                    try:
                        self._shm_meta_out = shared_memory.SharedMemory(
                            name=self._cfg.name + self._out_suffix
                        )
                        self._shm_meta_in = shared_memory.SharedMemory(
                            name=self._cfg.name + self._in_suffix
                        )
                        _detach_from_resource_tracker(self._shm_meta_out)
                        _detach_from_resource_tracker(self._shm_meta_in)
                        self._meta_out = _RingMeta(self._shm_meta_out.buf)
                        self._meta_in = _RingMeta(self._shm_meta_in.buf)
                        if (
                            self._meta_out.ready() == _READY_MAGIC
                            and self._meta_in.ready() == _READY_MAGIC
                        ):
                            break
                        # One ring published but not the other yet; drop
                        # the handles and retry.
                        self._detach_meta_only()
                    except FileNotFoundError:
                        self._detach_meta_only()
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "cudaipc endpoint: timed out waiting for server-side meta regions "
                            f"(name={self._cfg.name!r}, "
                            f"timeout_s={self._cfg.handshake_timeout_s})"
                        )
                    time.sleep(0.05)

                # Import the server's IPC handles.
                ipc_out = cudart.cudaIpcMemHandle_t()
                ipc_out.reserved = list(self._meta_out.ipc_handle_bytes())
                ipc_in = cudart.cudaIpcMemHandle_t()
                ipc_in.reserved = list(self._meta_in.ipc_handle_bytes())
                status, dev_out = cudart.cudaIpcOpenMemHandle(
                    ipc_out, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                _check(status, "cudaIpcOpenMemHandle(out)")
                status, dev_in = cudart.cudaIpcOpenMemHandle(
                    ipc_in, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                _check(status, "cudaIpcOpenMemHandle(in)")
                # Client's OUT direction is t2d, but the meta_out slot stores
                # the IPC handle for the server's d2t allocation -- because
                # both peers' "out" meta is the same physical ring as the
                # other peer's "in" meta. We swap pointers so:
                #   dev_out (client) ↔ server's t2d allocation
                #   dev_in  (client) ↔ server's d2t allocation
                #
                # Wait -- re-read the bookkeeping above. The server stored
                # h_out (== d2t alloc handle) into meta_out (== meta_d2t)
                # and h_in (== t2d alloc handle) into meta_in (== meta_t2d).
                # For the client, meta_out is meta_t2d (because client's
                # "out" suffix is "_meta_t2d"), which carries the t2d
                # alloc handle -- exactly the ring the client should write
                # to. Symmetrically meta_in carries the d2t alloc handle.
                # So no swap; dev_out / dev_in already correspond to the
                # right ring on the client side.
                self._dev_out = dev_out
                self._dev_in = dev_in
                self._dev_out_is_local = False
                self._dev_in_is_local = False

            self._started = True
            return EndpointStatus.kOk
        except Exception:
            print(
                "[cudaipc-endpoint] start failed:\n" + traceback.format_exc(),
                flush=True,
            )
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        # Best-effort: tell the peer we're going away.
        try:
            if self._meta_out is not None:
                self._meta_out.set_stopped()
            if self._meta_in is not None:
                self._meta_in.set_stopped()
        except Exception:
            pass
        self._cleanup()
        self._started = False
        return EndpointStatus.kOk

    def prime_recv(self, count):
        if not self._started:
            return EndpointStatus.kNotStarted
        if count <= 0:
            return EndpointStatus.kInvalidPayload
        if self._recv_credits + count > self._cfg.recv_queue_depth:
            return EndpointStatus.kQueueFull
        self._recv_credits += count
        return EndpointStatus.kOk

    def poll_once(self):
        if not self._started or self._meta_in is None:
            return EndpointStatus.kNotStarted
        return (
            EndpointStatus.kOk
            if self._meta_in.head() != self._meta_in.tail()
            else EndpointStatus.kEmpty
        )

    def send(self, packet):
        if not self._started or self._meta_out is None:
            return EndpointStatus.kNotStarted
        if self._recv_credits <= 0:
            return EndpointStatus.kNoRecvCredits
        payload = bytes(packet.payload)
        if len(payload) != self._cfg.payload_bytes:
            return EndpointStatus.kInvalidPayload
        frame = struct.pack("<I", int(packet.imm_data) & 0xFFFFFFFF) + payload
        if len(frame) != _FRAME_BYTES:
            return EndpointStatus.kInvalidPayload

        head = self._meta_out.head()
        tail = self._meta_out.tail()
        if head - tail >= _RING_SLOTS:
            return EndpointStatus.kQueueFull

        cudart = _load_cudart()
        # Stage the frame in the pinned host buffer, then push it to the
        # ring slot on whichever GPU owns that ring.
        ctypes.memmove(self._pinned_send, frame, _FRAME_BYTES)
        slot_offset = (head % _RING_SLOTS) * _FRAME_BYTES
        (status,) = cudart.cudaMemcpyAsync(
            self._dev_out + slot_offset,
            self._pinned_send,
            _FRAME_BYTES,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self._stream,
        )
        if int(status) != 0:
            return EndpointStatus.kError
        (status,) = cudart.cudaStreamSynchronize(self._stream)
        if int(status) != 0:
            return EndpointStatus.kError
        # Stream sync above acts as the release fence: every byte of the
        # frame is committed to GPU memory before the consumer observes
        # the head bump below.
        self._meta_out.set_head(head + 1)
        self._recv_credits -= 1
        return EndpointStatus.kOk

    def recv(self):
        if not self._started or self._meta_in is None:
            return EndpointStatus.kNotStarted, None
        head = self._meta_in.head()
        tail = self._meta_in.tail()
        if head == tail:
            return EndpointStatus.kEmpty, None

        cudart = _load_cudart()
        slot_offset = (tail % _RING_SLOTS) * _FRAME_BYTES
        (status,) = cudart.cudaMemcpyAsync(
            self._pinned_recv,
            self._dev_in + slot_offset,
            _FRAME_BYTES,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self._stream,
        )
        if int(status) != 0:
            return EndpointStatus.kError, None
        (status,) = cudart.cudaStreamSynchronize(self._stream)
        if int(status) != 0:
            return EndpointStatus.kError, None

        frame = bytes((ctypes.c_uint8 * _FRAME_BYTES).from_address(self._pinned_recv))
        imm = struct.unpack_from("<I", frame, 0)[0]
        self._meta_in.set_tail(tail + 1)
        return EndpointStatus.kOk, EndpointPacket(
            imm_data=int(imm), payload=frame[_FRAME_HEADER_BYTES:]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _create_or_recreate_shm(name: str) -> shared_memory.SharedMemory:
        """Fresh CPU shm region, scrubbing any stale segment from a
        previous run. Same gotcha as the shm backend: a crashed draft
        can leave a /dev/shm file we must unlink before re-creating."""
        try:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            import os

            try:
                os.unlink(f"/dev/shm/{name}")
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return shared_memory.SharedMemory(name=name, create=True, size=_META_BYTES)

    def _detach_meta_only(self) -> None:
        """Drop the meta views/handles without unlinking. Used during the
        client retry loop where the server's regions may still be in flux."""
        self._meta_out = None
        self._meta_in = None
        for shm in (self._shm_meta_out, self._shm_meta_in):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
        self._shm_meta_out = None
        self._shm_meta_in = None

    def _cleanup(self) -> None:
        # Drop ring/meta views first so SharedMemory.close() doesn't trip
        # on outstanding memoryviews.
        self._meta_out = None
        self._meta_in = None

        cudart = _load_cudart() if _cudart is not None else None

        # GPU pointers
        if cudart is not None:
            if self._dev_out:
                try:
                    if self._dev_out_is_local:
                        cudart.cudaFree(self._dev_out)
                    else:
                        cudart.cudaIpcCloseMemHandle(self._dev_out)
                except Exception:
                    pass
            if self._dev_in:
                try:
                    if self._dev_in_is_local:
                        cudart.cudaFree(self._dev_in)
                    else:
                        cudart.cudaIpcCloseMemHandle(self._dev_in)
                except Exception:
                    pass
            if self._pinned_send:
                try:
                    cudart.cudaFreeHost(self._pinned_send)
                except Exception:
                    pass
            if self._pinned_recv:
                try:
                    cudart.cudaFreeHost(self._pinned_recv)
                except Exception:
                    pass
            if self._stream:
                try:
                    cudart.cudaStreamDestroy(self._stream)
                except Exception:
                    pass
        self._dev_out = 0
        self._dev_in = 0
        self._pinned_send = 0
        self._pinned_recv = 0
        self._stream = 0

        # CPU shm
        for shm in (self._shm_meta_out, self._shm_meta_in):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            if self._cfg.is_server:
                try:
                    shm.unlink()
                except (FileNotFoundError, OSError):
                    pass
        self._shm_meta_out = None
        self._shm_meta_in = None


# Quiet linters: the Optional in field annotations is used via Optional[T].
_ = (Optional[int], Optional[shared_memory.SharedMemory])
__all__ = ["CudaIpcEndpointConfig", "_CudaIpcBackend"]


if sys.platform == "win32":  # pragma: no cover -- POSIX only
    raise RuntimeError(
        "cudaipc_endpoint is POSIX-only: it relies on multiprocessing.shared_memory "
        "and cudaIpc* APIs which are unavailable on Windows."
    )
