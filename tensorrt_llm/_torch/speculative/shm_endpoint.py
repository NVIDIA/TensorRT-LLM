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
"""POSIX shared-memory transport implementing the ``Endpoint`` protocol.

Drop-in replacement for ``_IbverbsRdmaBackend`` / ``_TcpRdmaBackend`` for the
same-machine case where draft and target run as separate processes on the
same host but do NOT have RDMA hardware available. The wire frame is
identical to the TCP backend (4-byte ``imm_data`` LE + 96-byte
``DraftApiProtocol`` payload = 100 bytes), so the upper protocol/channel
layers are unchanged.

Mechanism: two single-producer / single-consumer (SPSC) ring buffers in
POSIX shared memory.

    target ──── out_ring (t2d) ────► draft
    target ◄─── in_ring  (d2t) ──── draft

Each ring is laid out as::

    [0:8]    head (u64, written only by the producer)
    [8:16]   tail (u64, written only by the consumer)
    [16:17]  stopped flag (u8, peer sets to 1 on shutdown)
    [24:28]  ready magic (u32, written by the server once init is complete)
    [28:64]  padding (cache-line align)
    [64:..]  N slots × 100 bytes (one ``EndpointPacket`` frame each)

Aligned 8-byte stores/loads on x86_64 are atomic under TSO, so the SPSC
counters need no locks. Receive is busy-polled (matches the channel layer's
``while pending: pump_once_...`` loop); the polling overhead is ~1-2 µs per
empty poll, well below the existing ibverbs busy-poll cost.
"""

from __future__ import annotations

import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

from .draft_api_protocol import DraftApiProtocol
from .spec_decode_channel import EndpointPacket, EndpointStatus


def _detach_from_resource_tracker(shm: shared_memory.SharedMemory) -> None:
    """Stop ``multiprocessing.resource_tracker`` from auto-``unlink``-ing this
    shm segment when the holding process exits.

    On Python 3.8-3.12, ``SharedMemory.__init__`` calls
    ``resource_tracker.register`` unconditionally — including on attach-side
    processes that never created the segment (see
    ``Lib/multiprocessing/shared_memory.py``). As a result, a client process
    can silently unlink a server-owned segment on its own exit, breaking any
    follow-up reconnect by the next client. We unregister explicitly on
    BOTH peers; the server then calls ``shm.unlink()`` from ``stop()`` to
    actually destroy the segment when it's done with it.
    """
    if sys.platform == "win32":
        return
    try:
        from multiprocessing import resource_tracker
    except Exception:
        return
    name = getattr(shm, "_name", None)
    if not isinstance(name, str):
        return
    # The registered name in 3.12 is the POSIX form with leading slash, but
    # the historical record on older interpreters varies. Try both to stay
    # forward-compatible.
    candidates = [name]
    if name.startswith("/"):
        candidates.append(name.lstrip("/"))
    else:
        candidates.append("/" + name)
    for candidate in candidates:
        try:
            resource_tracker.unregister(candidate, "shared_memory")
        except Exception:
            continue


_FRAME_HEADER_BYTES = 4  # uint32 imm_data, little-endian
_FRAME_BYTES = _FRAME_HEADER_BYTES + DraftApiProtocol.kMessageBytes  # = 100

# Ring layout constants. Keep ``_RING_SLOTS`` a power of two so a future
# optimization could replace ``% _RING_SLOTS`` with a bit-mask; the python
# fast path here uses ``%`` directly.
_RING_SLOTS = 256
_HEADER_SIZE = 64
_HEAD_OFFSET = 0
_TAIL_OFFSET = 8
_STOPPED_OFFSET = 16
_READY_OFFSET = 24
_READY_MAGIC = 0x50454152  # 'PEAR'
_RING_BYTES = _HEADER_SIZE + _RING_SLOTS * _FRAME_BYTES


@dataclass
class ShmEndpointConfig:
    """Connection config for ``_ShmBackend``.

    ``shm_name`` is the prefix of the two POSIX shared-memory regions; the
    backend appends ``_t2d`` / ``_d2t`` suffixes so a single name uniquely
    identifies a draft/target pair on the host.
    """

    is_server: bool = False
    shm_name: str = "pearl_shm_default"
    handshake_timeout_s: float = 120.0
    recv_queue_depth: int = 256
    payload_bytes: int = DraftApiProtocol.kMessageBytes


class _ShmRing:
    """SPSC ring backed by a writable memoryview into shared memory.

    Producer writes ``frame`` to ``slots[head % N]`` then increments
    ``head``; consumer observes ``head != tail`` then reads
    ``slots[tail % N]`` and increments ``tail``. On x86_64 the aligned
    8-byte stores/loads are atomic under TSO and the producer's write of
    the frame is ordered before the increment of ``head``, so no explicit
    barrier is required. (A future port to ARMv8 would need
    ``release``/``acquire`` semantics — easiest via a C extension.)
    """

    __slots__ = ("_mv",)

    def __init__(self, mv):
        self._mv = mv

    def head(self) -> int:
        return struct.unpack_from("<Q", self._mv, _HEAD_OFFSET)[0]

    def tail(self) -> int:
        return struct.unpack_from("<Q", self._mv, _TAIL_OFFSET)[0]

    def stopped(self) -> int:
        return self._mv[_STOPPED_OFFSET]

    def ready(self) -> int:
        return struct.unpack_from("<I", self._mv, _READY_OFFSET)[0]

    def set_head(self, v: int) -> None:
        struct.pack_into("<Q", self._mv, _HEAD_OFFSET, v)

    def set_tail(self, v: int) -> None:
        struct.pack_into("<Q", self._mv, _TAIL_OFFSET, v)

    def set_stopped(self) -> None:
        self._mv[_STOPPED_OFFSET] = 1

    def set_ready(self) -> None:
        struct.pack_into("<I", self._mv, _READY_OFFSET, _READY_MAGIC)

    def reset_header(self) -> None:
        # Zero the whole header in one shot. The slot region is left
        # uninitialized — slots are overwritten before being read.
        self._mv[:_HEADER_SIZE] = b"\x00" * _HEADER_SIZE

    def push(self, frame: bytes) -> bool:
        head = self.head()
        if head - self.tail() >= _RING_SLOTS:
            return False  # full
        slot_idx = head % _RING_SLOTS
        offset = _HEADER_SIZE + slot_idx * _FRAME_BYTES
        self._mv[offset : offset + _FRAME_BYTES] = frame
        # x86 TSO: the slot store above is committed to memory before this
        # 8-byte head store becomes visible to the consumer.
        self.set_head(head + 1)
        return True

    def pop(self) -> Optional[bytes]:
        tail = self.tail()
        if tail == self.head():
            return None  # empty
        slot_idx = tail % _RING_SLOTS
        offset = _HEADER_SIZE + slot_idx * _FRAME_BYTES
        frame = bytes(self._mv[offset : offset + _FRAME_BYTES])
        self.set_tail(tail + 1)
        return frame


class _ShmBackend:
    """Endpoint Protocol implementation over two SPSC shared-memory rings.

    Server (draft, ``is_server=True``) creates both rings via
    ``SharedMemory(create=True)`` and writes the ready magic last; the
    server is also responsible for ``unlink``-ing the segments at stop
    time. Client (target, ``is_server=False``) attaches by name and busy-
    waits until both rings report ready.
    """

    def __init__(self, config: ShmEndpointConfig):
        self._cfg = config
        # SHM is symmetric; pick the producer/consumer side by role.
        if config.is_server:
            self._out_suffix = "_d2t"  # this peer (draft) writes
            self._in_suffix = "_t2d"  # this peer (draft) reads
        else:
            self._out_suffix = "_t2d"  # this peer (target) writes
            self._in_suffix = "_d2t"  # this peer (target) reads
        self._shm_out: Optional[shared_memory.SharedMemory] = None
        self._shm_in: Optional[shared_memory.SharedMemory] = None
        self._out: Optional[_ShmRing] = None
        self._in: Optional[_ShmRing] = None
        self._recv_credits = 0
        self._started = False

    # ------------------------------------------------------------------
    # Endpoint Protocol
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            return EndpointStatus.kOk
        try:
            if self._cfg.is_server:
                self._shm_out = self._create_or_recreate(self._cfg.shm_name + self._out_suffix)
                self._shm_in = self._create_or_recreate(self._cfg.shm_name + self._in_suffix)
                # Take ownership of unlink — stop() will call shm.unlink()
                # explicitly. Otherwise resource_tracker may race with our
                # cleanup at process exit.
                _detach_from_resource_tracker(self._shm_out)
                _detach_from_resource_tracker(self._shm_in)
                self._out = _ShmRing(self._shm_out.buf)
                self._in = _ShmRing(self._shm_in.buf)
                self._out.reset_header()
                self._in.reset_header()
                # Order matters: only mark ready after the headers are
                # zeroed, so the client never observes a partially-init
                # region.
                self._out.set_ready()
                self._in.set_ready()
            else:
                deadline = time.monotonic() + max(0.0, self._cfg.handshake_timeout_s)
                while True:
                    try:
                        self._shm_out = shared_memory.SharedMemory(
                            name=self._cfg.shm_name + self._out_suffix
                        )
                        self._shm_in = shared_memory.SharedMemory(
                            name=self._cfg.shm_name + self._in_suffix
                        )
                        # CRITICAL: Python 3.8-3.12 registers attach-side
                        # SharedMemory with resource_tracker unconditionally
                        # (Lib/multiprocessing/shared_memory.py:120). Without
                        # this detach, the client's exit would unlink the
                        # server-owned segment and break the next reconnect.
                        _detach_from_resource_tracker(self._shm_out)
                        _detach_from_resource_tracker(self._shm_in)
                        self._out = _ShmRing(self._shm_out.buf)
                        self._in = _ShmRing(self._shm_in.buf)
                        if self._out.ready() == _READY_MAGIC and self._in.ready() == _READY_MAGIC:
                            break
                        # Server published one ring but not the other yet;
                        # detach and retry.
                        self._detach_handles_only()
                    except FileNotFoundError:
                        # Server hasn't created yet; retry below.
                        self._detach_handles_only()
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "shm endpoint: timed out waiting for server-side shm regions "
                            f"(name={self._cfg.shm_name!r}, timeout_s={self._cfg.handshake_timeout_s})"
                        )
                    time.sleep(0.05)

            self._started = True
            return EndpointStatus.kOk
        except Exception:
            print(
                "[shm-endpoint] start failed:\n" + traceback.format_exc(),
                flush=True,
            )
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        # Best-effort: tell the peer we're going away so it can stop
        # polling at its leisure.
        try:
            if self._out is not None:
                self._out.set_stopped()
            if self._in is not None:
                self._in.set_stopped()
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
        if not self._started or self._in is None:
            return EndpointStatus.kNotStarted
        return EndpointStatus.kOk if self._in.head() != self._in.tail() else EndpointStatus.kEmpty

    def send(self, packet):
        if not self._started or self._out is None:
            return EndpointStatus.kNotStarted
        if self._recv_credits <= 0:
            return EndpointStatus.kNoRecvCredits
        payload = bytes(packet.payload)
        if len(payload) != self._cfg.payload_bytes:
            return EndpointStatus.kInvalidPayload
        frame = struct.pack("<I", int(packet.imm_data) & 0xFFFFFFFF) + payload
        if len(frame) != _FRAME_BYTES:
            return EndpointStatus.kInvalidPayload
        if not self._out.push(frame):
            return EndpointStatus.kQueueFull
        self._recv_credits -= 1
        return EndpointStatus.kOk

    def recv(self):
        if not self._started or self._in is None:
            return EndpointStatus.kNotStarted, None
        frame = self._in.pop()
        if frame is None:
            return EndpointStatus.kEmpty, None
        imm = struct.unpack_from("<I", frame, 0)[0]
        return EndpointStatus.kOk, EndpointPacket(
            imm_data=int(imm), payload=frame[_FRAME_HEADER_BYTES:]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _create_or_recreate(name: str) -> shared_memory.SharedMemory:
        """Create a fresh shm region, scrubbing any stale segment from a
        previous run. Without this, a crashed draft leaves /dev/shm
        entries that would collide on the next start."""
        try:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            # Best-effort: even if we can't read the stale segment, try to
            # remove it via the underlying /dev/shm path so the create
            # below has a clean slate.
            stale_path = os.path.join("/dev/shm", name)
            try:
                os.unlink(stale_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return shared_memory.SharedMemory(name=name, create=True, size=_RING_BYTES)

    def _detach_handles_only(self) -> None:
        """Drop ring views + shm handles without unlinking. Used during
        client retry where the server's shm may not yet have stable
        contents."""
        self._out = None
        self._in = None
        for shm in (self._shm_out, self._shm_in):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
        self._shm_out = None
        self._shm_in = None

    def _cleanup(self) -> None:
        # Drop the memoryview-holding ring objects first so SharedMemory.close()
        # doesn't trip on outstanding views.
        self._out = None
        self._in = None
        for shm in (self._shm_out, self._shm_in):
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
        self._shm_out = None
        self._shm_in = None
