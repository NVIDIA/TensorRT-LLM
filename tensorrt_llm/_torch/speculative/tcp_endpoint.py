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
"""Pure-TCP transport implementing the ``Endpoint`` protocol.

Drop-in replacement for ``_IbverbsRdmaBackend`` that ships the 96-byte
``DraftApiProtocol`` frame plus a 4-byte ``imm_data`` over a plain TCP
socket. Useful as:

* a baseline for Draft API correctness work that doesn't need RDMA
  hardware,
* the first leg of the 4-phase draft-offload transport progression
  (TCP → ibverbs WRITE_WITH_IMM → DOCA CPU_PROXY → DOCA GPU SM doorbell).

Wire frame on the socket is exactly 100 bytes per message:

    bytes [0:4]    little-endian uint32 ``imm_data``
    bytes [4:100]  96-byte ``DraftApiProtocol`` payload

The receive side runs a daemon thread draining the socket into a thread-
safe inbox so the spec-decode polling loop can issue ``poll_once`` /
``recv`` without blocking the engine thread on socket I/O.

Flow control: TCP has no native receive-buffer credits like ibverbs, so
``prime_recv(n)`` just bumps a local counter that ``send()`` decrements.
Bounded by ``recv_queue_depth`` from the config. This is identical
semantics to the ``InMemoryEndpointBackend`` and good enough for
spec-decode where the application drains promptly.
"""

from __future__ import annotations

import socket
import struct
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .draft_api_protocol import DraftApiProtocol
from .spec_decode_channel import EndpointPacket, EndpointStatus

_FRAME_HEADER_BYTES = 4  # uint32 imm_data, little-endian
_FRAME_BYTES = _FRAME_HEADER_BYTES + DraftApiProtocol.kMessageBytes  # = 100


@dataclass
class TcpEndpointConfig:
    """Connection config for ``_TcpRdmaBackend``.

    Mirrors the relevant fields of ``IbverbsEndpointConfig`` so upper
    layers can switch transports by changing a single config field.
    """

    # Direction: client connects to (remote_host, remote_port); server
    # binds to (bind_host, bind_port). Exactly one of those pairs is
    # active depending on ``is_server``.
    is_server: bool = False
    remote_host: str = "127.0.0.1"
    remote_port: int = 47000
    bind_host: str = "0.0.0.0"
    bind_port: int = 47000
    handshake_timeout_s: float = 120.0

    # Shared with the ibverbs config so configs are swappable.
    recv_queue_depth: int = 256
    payload_bytes: int = DraftApiProtocol.kMessageBytes


class _TcpRdmaBackend:
    """Endpoint Protocol implementation over a single full-duplex TCP socket."""

    def __init__(self, config: TcpEndpointConfig):
        self._cfg = config
        self._sock: Optional[socket.socket] = None
        self._listen_sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._inbox = deque()
        self._inbox_lock = threading.Lock()
        self._inbox_cv = threading.Condition(self._inbox_lock)

        # See module-level note: TCP-side flow control is purely local.
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
                self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._listen_sock.bind((self._cfg.bind_host, int(self._cfg.bind_port)))
                self._listen_sock.listen(1)
                self._listen_sock.settimeout(self._cfg.handshake_timeout_s)
                conn, _peer = self._listen_sock.accept()
                self._sock = conn
            else:
                # Retry connect: the draft may still be opening the data
                # port (it does that after loading the model, which can
                # take 60s+). Retry until handshake_timeout_s, then fail.
                deadline = (
                    None
                    if self._cfg.handshake_timeout_s is None
                    else (__import__("time").monotonic() + self._cfg.handshake_timeout_s)
                )
                last_err = None
                import time as _time

                while True:
                    try:
                        self._sock = socket.create_connection(
                            (self._cfg.remote_host, int(self._cfg.remote_port)),
                            timeout=10.0,
                        )
                        break
                    except (ConnectionRefusedError, socket.timeout, OSError) as exc:
                        last_err = exc
                        if deadline is not None and _time.monotonic() >= deadline:
                            raise
                        _time.sleep(0.2)
                if self._sock is None:
                    raise last_err or RuntimeError("TcpRdmaBackend connect failed")
            self._sock.settimeout(None)  # blocking after handshake
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self._stop_evt.clear()
            self._recv_thread = threading.Thread(
                target=self._recv_loop, name="TcpRdmaBackend-recv", daemon=True
            )
            self._recv_thread.start()
            self._started = True
            return EndpointStatus.kOk
        except Exception:
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        self._stop_evt.set()
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
        if not self._started:
            return EndpointStatus.kNotStarted
        with self._inbox_lock:
            return EndpointStatus.kOk if self._inbox else EndpointStatus.kEmpty

    def send(self, packet):
        if not self._started:
            return EndpointStatus.kNotStarted
        if self._recv_credits <= 0:
            return EndpointStatus.kNoRecvCredits
        payload = bytes(packet.payload)
        if len(payload) != self._cfg.payload_bytes:
            return EndpointStatus.kInvalidPayload
        try:
            frame = struct.pack("<I", int(packet.imm_data) & 0xFFFFFFFF) + payload
            self._sock.sendall(frame)
            self._recv_credits -= 1
            return EndpointStatus.kOk
        except Exception:
            return EndpointStatus.kError

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        with self._inbox_lock:
            if not self._inbox:
                return EndpointStatus.kEmpty, None
            return EndpointStatus.kOk, self._inbox.popleft()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recv_loop(self):
        try:
            while not self._stop_evt.is_set():
                frame = self._recv_exact(_FRAME_BYTES)
                if frame is None:
                    return
                imm = struct.unpack("<I", frame[:_FRAME_HEADER_BYTES])[0]
                packet = EndpointPacket(imm_data=int(imm), payload=frame[_FRAME_HEADER_BYTES:])
                with self._inbox_cv:
                    self._inbox.append(packet)
                    self._inbox_cv.notify_all()
        except Exception:
            return
        finally:
            self._started = False

    def _recv_exact(self, n):
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            if self._stop_evt.is_set():
                return None
            try:
                chunk = self._sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _cleanup(self):
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        if self._recv_thread is not None and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        self._recv_thread = None
        with self._inbox_lock:
            self._inbox.clear()
