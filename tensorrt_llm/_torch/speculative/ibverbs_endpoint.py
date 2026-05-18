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
"""libibverbs endpoint backends for the speculative Draft API channel.

This module is the interface peer of izzy's ``doca_endpoint.py`` — the
``EndpointBackend`` Protocol matches verbatim so ``SpecDecodeChannel``
does not care which backend it sits on top of.

Two backends ship here:

- ``InMemoryEndpointBackend`` — used by every layer's unit tests and by
  test harnesses that don't have RDMA hardware.
- ``_IbverbsRdmaBackend`` — production backend on top of libibverbs +
  ``libibv_wrapper.so``.  Uses ``IBV_WR_RDMA_WRITE_WITH_IMM`` so the
  receiver picks up ``imm_data`` from the recv CQE and uses the slot
  field to demultiplex into the right per-request buffer.

Hardware testing is exercised end to end by
``scripts/smoke_write_with_imm.py``.  This file refines that prototype
into the multi-buffer, slot-indexed shape needed for many concurrent
requests.
"""

import ctypes
import json
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .spec_decode_channel import EndpointPacket, EndpointStatus

# ---------------------------------------------------------------------------
# Env helpers + debug trace
# ---------------------------------------------------------------------------


def _env_enabled(name):
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _debug_trace(message, *args):
    trace_path = os.environ.get("TLLM_RDMA_DEBUG_TRACE_PATH", "/tmp/tllm_rdma_worker_trace.log")
    if not trace_path:
        return
    try:
        rendered = message % args if args else message
    except Exception:
        rendered = "%s args=%r" % (message, args)
    try:
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(
                "%s pid=%d ibverbs_endpoint %s\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(), rendered)
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public config + Protocol
# ---------------------------------------------------------------------------


@dataclass
class IbverbsEndpointConfig:
    """Connection and queue configuration for the ibverbs endpoint adapter.

    Mirrors izzy's ``DocaEndpointConfig`` field-for-field so upper layers
    can accept either backend transparently.
    """

    gpu_id: int = 0
    nic_name: str = ""
    remote_host: str = "localhost"
    remote_port: int = 47000
    remote_peer_name: str = "draft_lpu"
    recv_queue_depth: int = 32
    payload_bytes: int = 96
    use_torch_ops: bool = False
    torch_ops_lib_path: Optional[str] = None
    # ibverbs-specific:
    max_num_requests: int = 256
    sgid_index: int = 0
    pkey_index: int = 0
    ib_port: int = 1
    is_server: bool = False  # True ⇒ listen on local_port; False ⇒ connect out
    local_port: int = 0
    # Default 120s — long enough for the peer process to finish heavy
    # startup (e.g. TRT-LLM target loading an 8B model takes ~40s).
    # The client side typically connects within a few seconds; the server
    # side may wait longer.
    handshake_timeout_s: float = 120.0
    wrapper_path: Optional[str] = None  # override path to libibv_wrapper.so


class EndpointBackend(Protocol):
    def start(self): ...
    def stop(self): ...
    def prime_recv(self, count): ...
    def poll_once(self): ...
    def send(self, packet): ...
    def recv(self): ...


# ---------------------------------------------------------------------------
# In-memory backend (no hardware required)
# ---------------------------------------------------------------------------


class InMemoryEndpointBackend:
    """Queue-backed endpoint backend used for non-hardware tests.

    Verbatim port of izzy's ``InMemoryEndpointBackend``.
    """

    def __init__(self, max_queue_depth=32):
        self._max_queue_depth = max_queue_depth
        self._started = False
        self._recv_credits = 0
        self._packets = deque()

    def start(self):
        self._started = True
        return EndpointStatus.kOk

    def stop(self):
        self._started = False
        self._recv_credits = 0
        self._packets.clear()
        return EndpointStatus.kOk

    def prime_recv(self, count):
        if not self._started:
            return EndpointStatus.kNotStarted
        if count <= 0:
            return EndpointStatus.kInvalidPayload
        if self._recv_credits + count > self._max_queue_depth:
            return EndpointStatus.kQueueFull
        self._recv_credits += count
        return EndpointStatus.kOk

    def poll_once(self):
        if not self._started:
            return EndpointStatus.kNotStarted
        if not self._packets:
            return EndpointStatus.kEmpty
        return EndpointStatus.kOk

    def send(self, packet):
        if not self._started:
            return EndpointStatus.kNotStarted
        if self._recv_credits == 0:
            return EndpointStatus.kNoRecvCredits
        if len(self._packets) >= self._max_queue_depth:
            return EndpointStatus.kQueueFull
        self._recv_credits -= 1
        self._packets.append(
            EndpointPacket(imm_data=int(packet.imm_data), payload=bytes(packet.payload))
        )
        return EndpointStatus.kOk

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        if not self._packets:
            return EndpointStatus.kEmpty, None
        return EndpointStatus.kOk, self._packets.popleft()


# ---------------------------------------------------------------------------
# libibverbs constants & low-level helpers
# ---------------------------------------------------------------------------

# ibv_access_flags
_IBV_ACCESS_LOCAL_WRITE = 1
_IBV_ACCESS_REMOTE_WRITE = 2
_IBV_ACCESS_REMOTE_READ = 4
_IBV_ACCESS_REMOTE_ATOMIC = 8

# ibv_qp_type
_IBV_QPT_RC = 2

# qp states
_IBV_QPS_RESET = 0
_IBV_QPS_INIT = 1
_IBV_QPS_RTR = 2
_IBV_QPS_RTS = 3

# wr opcodes
_IBV_WR_RDMA_WRITE = 0
_IBV_WR_RDMA_WRITE_WITH_IMM = 1

# wr send_flags
_IBV_SEND_SIGNALED = 2

# wc statuses / opcodes
_IBV_WC_SUCCESS = 0
_IBV_WC_RECV_RDMA_WITH_IMM = 0x81

# qp_attr_mask bits
_IBV_QP_STATE = 1 << 0
_IBV_QP_ACCESS_FLAGS = 1 << 3
_IBV_QP_PKEY_INDEX = 1 << 4
_IBV_QP_PORT = 1 << 5
_IBV_QP_AV = 1 << 7
_IBV_QP_PATH_MTU = 1 << 8
_IBV_QP_TIMEOUT = 1 << 9
_IBV_QP_RETRY_CNT = 1 << 10
_IBV_QP_RNR_RETRY = 1 << 11
_IBV_QP_RQ_PSN = 1 << 12
_IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13
_IBV_QP_MIN_RNR_TIMER = 1 << 15
_IBV_QP_SQ_PSN = 1 << 16
_IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17
_IBV_QP_DEST_QPN = 1 << 20


class _ibv_wc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
        ("src_qp", ctypes.c_uint32),
        ("wc_flags", ctypes.c_uint),
        ("pkey_index", ctypes.c_uint16),
        ("slid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("dlid_path_bits", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 2),
    ]


def _buf(size):
    return (ctypes.c_uint8 * size)()


def _voidp(obj):
    return ctypes.cast(ctypes.addressof(obj), ctypes.c_void_p)


def _set8(buf, off, val):
    struct.pack_into("B", buf, off, val & 0xFF)


def _set16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def _set32(buf, off, val):
    struct.pack_into("<I", buf, off, val & 0xFFFFFFFF)


def _set64(buf, off, val):
    struct.pack_into("<Q", buf, off, val & 0xFFFFFFFFFFFFFFFF)


def _read_mr_fields(mr_p):
    """Return (addr, lkey, rkey) for an ``ibv_mr*`` pointer."""
    raw = ctypes.cast(mr_p, ctypes.POINTER(ctypes.c_uint8))
    addr = struct.unpack_from("<Q", bytes(raw[16:24]))[0]
    lkey = struct.unpack_from("<I", bytes(raw[36:40]))[0]
    rkey = struct.unpack_from("<I", bytes(raw[40:44]))[0]
    return addr, lkey, rkey


def _read_qp_num(qp_p):
    raw = ctypes.cast(qp_p, ctypes.POINTER(ctypes.c_uint8))
    return struct.unpack_from("<I", bytes(raw[52:56]))[0]


# ---------------------------------------------------------------------------
# libibverbs loader (lazy: only on hardware backend construction)
# ---------------------------------------------------------------------------


_LIBVERBS = None
_LIBWRAP = None
_LIB_LOAD_LOCK = threading.Lock()


def _packaged_wrapper_path():
    path = Path(__file__).resolve().parent / "ibverbs_ext" / "libibv_wrapper.so"
    return str(path) if path.exists() else None


def _load_libs(wrapper_path=None):
    """Load and (idempotently) configure libibverbs + the wrapper."""
    global _LIBVERBS, _LIBWRAP
    if _LIBVERBS is not None and _LIBWRAP is not None:
        return _LIBVERBS, _LIBWRAP
    with _LIB_LOAD_LOCK:
        if _LIBVERBS is None:
            _LIBVERBS = ctypes.CDLL("libibverbs.so.1", use_errno=True)
            _v = _LIBVERBS
            _v.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
            _v.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
            _v.ibv_free_device_list.restype = None
            _v.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            _v.ibv_get_device_name.restype = ctypes.c_char_p
            _v.ibv_get_device_name.argtypes = [ctypes.c_void_p]
            _v.ibv_open_device.restype = ctypes.c_void_p
            _v.ibv_open_device.argtypes = [ctypes.c_void_p]
            _v.ibv_close_device.restype = ctypes.c_int
            _v.ibv_close_device.argtypes = [ctypes.c_void_p]
            _v.ibv_query_port.restype = ctypes.c_int
            _v.ibv_query_port.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_void_p]
            _v.ibv_query_gid.restype = ctypes.c_int
            _v.ibv_query_gid.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint8,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            _v.ibv_alloc_pd.restype = ctypes.c_void_p
            _v.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
            _v.ibv_dealloc_pd.restype = ctypes.c_int
            _v.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
            _v.ibv_reg_mr.restype = ctypes.c_void_p
            _v.ibv_reg_mr.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _v.ibv_dereg_mr.restype = ctypes.c_int
            _v.ibv_dereg_mr.argtypes = [ctypes.c_void_p]
            _v.ibv_create_cq.restype = ctypes.c_void_p
            _v.ibv_create_cq.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _v.ibv_destroy_cq.restype = ctypes.c_int
            _v.ibv_destroy_cq.argtypes = [ctypes.c_void_p]
            _v.ibv_create_qp.restype = ctypes.c_void_p
            _v.ibv_create_qp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _v.ibv_destroy_qp.restype = ctypes.c_int
            _v.ibv_destroy_qp.argtypes = [ctypes.c_void_p]
            _v.ibv_modify_qp.restype = ctypes.c_int
            _v.ibv_modify_qp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        if _LIBWRAP is None:
            path = (
                wrapper_path or os.environ.get("TLLM_RDMA_WRAPPER_PATH") or _packaged_wrapper_path()
            )
            if not path:
                # Default to project repo layout.
                path = "/home/scratch.zhaoyangw_gpu/rdma/libibv_wrapper.so"
            _LIBWRAP = ctypes.CDLL(path, use_errno=True)
            # IMPORTANT: set argtypes so 64-bit pointers aren't truncated
            # when crossing the Python/C boundary on x86_64.
            _LIBWRAP.wrap_ibv_post_send.restype = ctypes.c_int
            _LIBWRAP.wrap_ibv_post_send.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _LIBWRAP.wrap_ibv_post_recv.restype = ctypes.c_int
            _LIBWRAP.wrap_ibv_post_recv.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _LIBWRAP.wrap_ibv_poll_cq.restype = ctypes.c_int
            _LIBWRAP.wrap_ibv_poll_cq.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return _LIBVERBS, _LIBWRAP


# ---------------------------------------------------------------------------
# TCP handshake utilities
# ---------------------------------------------------------------------------


def _send_json_line(sock, obj):
    payload = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(payload)


def _recv_json_line(sock, timeout_s):
    sock.settimeout(timeout_s)
    chunks = []
    while True:
        ch = sock.recv(4096)
        if not ch:
            raise RuntimeError("peer closed during handshake")
        chunks.append(ch)
        if b"\n" in ch:
            break
    line = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


# ---------------------------------------------------------------------------
# _IbverbsRdmaBackend
# ---------------------------------------------------------------------------


@dataclass
class _LocalQpInfo:
    qpn: int
    psn: int
    lid: int
    gid: bytes
    active_mtu: int
    recv_base: int
    rkey: int
    max_num_requests: int
    payload_bytes: int


class _IbverbsRdmaBackend:
    """Production backend over libibverbs ``WRITE_WITH_IMM``.

    Lifecycle
    ---------
    - ``start()`` opens the NIC, builds an RC QP, allocates per-slot recv
      buffers + a single send buffer, and finishes a TCP handshake that
      exchanges QP metadata with the peer.
    - ``prime_recv(n)`` posts ``n`` recv WRs (one per slot, slot index
      taken from a free list) so the receiver has credits.
    - ``send(packet)`` uses ``packet.imm_data`` to derive the destination
      slot (bits [20:32)) and writes the 96-byte payload to the peer's
      ``recv_base + slot * payload_bytes``.
    - ``poll_once()`` and ``recv()`` drain the recv CQ; ``imm_data`` is
      copied straight from the CQE.
    """

    def __init__(self, config: IbverbsEndpointConfig):
        self._cfg = config
        self._libverbs = None
        self._libwrap = None

        self._ctx = None
        self._pd = None
        self._cq_send = None
        self._cq_recv = None
        self._qp = None

        self._send_buf = None  # ctypes array (96 bytes)
        self._send_mr = None
        self._send_addr = 0
        self._send_lkey = 0

        self._recv_buf = None  # ctypes array (max_num_requests * 96)
        self._recv_mr = None
        self._recv_base = 0
        self._recv_lkey = 0
        self._recv_rkey = 0

        # Holders for posted WR memory (must outlive post calls).
        # Each entry: (sge_buf, wr_buf).  WRITE_WITH_IMM consumes recv WRs
        # in FIFO order regardless of which "slot" they were posted with —
        # the receive WR is purely a credit.  The actual destination slot
        # comes from the imm_data on the CQE.  So we don't try to map
        # wr_id → slot; we just keep memory alive for posted WRs.
        self._posted_recv_wrs = {}  # wr_id -> (sge_buf, wr_buf)

        # Dummy 1-byte recv landing buffer.  WRITE_WITH_IMM doesn't write
        # into the recv WR's SGE, but the SGE must still be valid lkey.
        self._recv_dummy_buf = None
        self._recv_dummy_addr = 0
        self._recv_dummy_lkey = 0

        # Peer info filled by handshake.
        self._peer_qpn = 0
        self._peer_psn = 0
        self._peer_lid = 0
        self._peer_gid = b"\x00" * 16
        self._peer_recv_base = 0
        self._peer_rkey = 0
        self._peer_max_num_requests = 0

        self._next_recv_wr_id = 1

        self._started = False
        self._stopping = False
        self._recv_credits = 0

        # Buffer of packets received on the wire, awaiting recv() consumption.
        self._inbox = deque()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            return EndpointStatus.kOk
        try:
            self._libverbs, self._libwrap = _load_libs(self._cfg.wrapper_path)
            self._open_device()
            self._allocate_buffers()
            self._create_qp()
            self._qp_to_init()
            local = self._collect_local_info()
            self._tcp_handshake(local)
            self._qp_to_rtr()
            self._qp_to_rts()
            self._started = True
            _debug_trace(
                "started nic=%s qpn=%d peer_qpn=%d max_req=%d",
                self._cfg.nic_name,
                local.qpn,
                self._peer_qpn,
                self._cfg.max_num_requests,
            )
            return EndpointStatus.kOk
        except Exception as exc:
            _debug_trace("start failed: %r", exc)
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        self._stopping = True
        self._cleanup()
        self._started = False
        return EndpointStatus.kOk

    def prime_recv(self, count):
        if not self._started:
            return EndpointStatus.kNotStarted
        if count <= 0:
            return EndpointStatus.kInvalidPayload
        max_outstanding = self._cfg.recv_queue_depth
        if self._recv_credits + count > max_outstanding:
            return EndpointStatus.kQueueFull
        for _ in range(count):
            try:
                self._post_recv_credit()
            except Exception as exc:
                _debug_trace("post_recv failed: %r", exc)
                return EndpointStatus.kError
            self._recv_credits += 1
        return EndpointStatus.kOk

    def poll_once(self):
        if not self._started:
            return EndpointStatus.kNotStarted
        self._drain_recv_cq()
        if not self._inbox:
            return EndpointStatus.kEmpty
        return EndpointStatus.kOk

    def send(self, packet):
        if not self._started:
            return EndpointStatus.kNotStarted
        try:
            self._do_send(packet)
            return EndpointStatus.kOk
        except Exception as exc:
            _debug_trace("send failed: %r", exc)
            return EndpointStatus.kError

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        # Drain CQ first in case caller didn't call poll_once.
        self._drain_recv_cq()
        if not self._inbox:
            return EndpointStatus.kEmpty, None
        return EndpointStatus.kOk, self._inbox.popleft()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _open_device(self):
        cnt = ctypes.c_int(0)
        devs = self._libverbs.ibv_get_device_list(ctypes.byref(cnt))
        if not devs:
            raise RuntimeError("ibv_get_device_list returned NULL")
        try:
            target = None
            first_name = None
            for i in range(cnt.value):
                dev = devs[i]
                if not dev:
                    continue
                nm = self._libverbs.ibv_get_device_name(dev)
                if not nm:
                    continue
                name = nm.decode()
                if first_name is None:
                    first_name = name
                if self._cfg.nic_name and name == self._cfg.nic_name:
                    target = dev
                    break
                if not self._cfg.nic_name and target is None:
                    target = dev
            if target is None:
                raise RuntimeError("device %r not found" % self._cfg.nic_name)
            if not self._cfg.nic_name and first_name is not None:
                self._cfg.nic_name = first_name
            self._ctx = self._libverbs.ibv_open_device(target)
            if not self._ctx:
                raise RuntimeError("ibv_open_device(%r) failed" % self._cfg.nic_name)
        finally:
            self._libverbs.ibv_free_device_list(devs)

        # Port info
        port_attr = _buf(56)
        if self._libverbs.ibv_query_port(self._ctx, self._cfg.ib_port, _voidp(port_attr)) != 0:
            raise RuntimeError("ibv_query_port failed")
        self._local_lid = struct.unpack_from("<H", port_attr, 34)[0]
        self._local_active_mtu = struct.unpack_from("<i", port_attr, 8)[0]
        gid_buf = _buf(16)
        if (
            self._libverbs.ibv_query_gid(
                self._ctx, self._cfg.ib_port, self._cfg.sgid_index, _voidp(gid_buf)
            )
            != 0
        ):
            raise RuntimeError("ibv_query_gid failed")
        self._local_gid = bytes(gid_buf)

        self._pd = self._libverbs.ibv_alloc_pd(self._ctx)
        if not self._pd:
            raise RuntimeError("ibv_alloc_pd failed")

    def _allocate_buffers(self):
        payload = self._cfg.payload_bytes
        n = self._cfg.max_num_requests
        access = _IBV_ACCESS_LOCAL_WRITE | _IBV_ACCESS_REMOTE_WRITE | _IBV_ACCESS_REMOTE_READ

        # Send buffer (single 96-byte staging buffer).
        self._send_buf = _buf(payload)
        self._send_mr = self._libverbs.ibv_reg_mr(self._pd, _voidp(self._send_buf), payload, access)
        if not self._send_mr:
            raise RuntimeError("ibv_reg_mr(send) failed errno=%d" % ctypes.get_errno())
        self._send_addr, self._send_lkey, _ = _read_mr_fields(self._send_mr)

        # Recv pool — one big MR, indexed by slot.  Slots are written by
        # the peer's WRITE_WITH_IMM directly into recv_buf[slot*payload].
        recv_total = payload * n
        self._recv_buf = _buf(recv_total)
        self._recv_mr = self._libverbs.ibv_reg_mr(
            self._pd, _voidp(self._recv_buf), recv_total, access
        )
        if not self._recv_mr:
            raise RuntimeError("ibv_reg_mr(recv) failed errno=%d" % ctypes.get_errno())
        self._recv_base, self._recv_lkey, self._recv_rkey = _read_mr_fields(self._recv_mr)

        # 1-byte dummy buffer that the recv WR's SGE points to.
        # WRITE_WITH_IMM does not write into this — the buffer is a
        # placeholder so the recv WR has a valid lkey.
        self._recv_dummy_buf = _buf(8)
        self._recv_dummy_addr = ctypes.addressof(self._recv_dummy_buf)
        # Reuse the recv MR's lkey since dummy_buf overlaps PD scope; we
        # need a different MR if the address isn't covered.  Easier: use
        # the send buffer's lkey (its MR covers different memory) — but
        # the cleanest fix is to register the dummy as part of the recv
        # MR.  For simplicity, register a tiny 8-byte dummy MR.
        self._recv_dummy_mr = self._libverbs.ibv_reg_mr(
            self._pd, _voidp(self._recv_dummy_buf), 8, access
        )
        if not self._recv_dummy_mr:
            raise RuntimeError("ibv_reg_mr(dummy) failed errno=%d" % ctypes.get_errno())
        _, self._recv_dummy_lkey, _ = _read_mr_fields(self._recv_dummy_mr)

    def _create_qp(self):
        cqe = max(self._cfg.recv_queue_depth, 16)
        self._cq_send = self._libverbs.ibv_create_cq(self._ctx, cqe, None, None, 0)
        self._cq_recv = self._libverbs.ibv_create_cq(self._ctx, cqe, None, None, 0)
        if not self._cq_send or not self._cq_recv:
            raise RuntimeError("ibv_create_cq failed")

        init = _buf(96)
        _set64(init, 8, self._cq_send)
        _set64(init, 16, self._cq_recv)
        _set32(init, 32, max(16, cqe))  # max_send_wr
        _set32(init, 36, max(16, cqe))  # max_recv_wr
        _set32(init, 40, 1)  # max_send_sge
        _set32(init, 44, 1)  # max_recv_sge
        _set32(init, 52, _IBV_QPT_RC)
        _set32(init, 56, 0)  # sq_sig_all
        self._qp = self._libverbs.ibv_create_qp(self._pd, _voidp(init))
        if not self._qp:
            raise RuntimeError("ibv_create_qp failed errno=%d" % ctypes.get_errno())

    def _qp_to_init(self):
        attr = _buf(144)
        _set32(attr, 0, _IBV_QPS_INIT)
        _set32(
            attr,
            32,
            _IBV_ACCESS_LOCAL_WRITE
            | _IBV_ACCESS_REMOTE_WRITE
            | _IBV_ACCESS_REMOTE_READ
            | _IBV_ACCESS_REMOTE_ATOMIC,
        )
        _set16(attr, 120, self._cfg.pkey_index)
        _set8(attr, 129, self._cfg.ib_port)
        mask = _IBV_QP_STATE | _IBV_QP_PKEY_INDEX | _IBV_QP_PORT | _IBV_QP_ACCESS_FLAGS
        if self._libverbs.ibv_modify_qp(self._qp, _voidp(attr), mask) != 0:
            raise RuntimeError("modify_qp INIT failed errno=%d" % ctypes.get_errno())

    def _qp_to_rtr(self):
        path_mtu = min(self._local_active_mtu, self._peer_active_mtu)
        attr = _buf(144)
        _set32(attr, 0, _IBV_QPS_RTR)
        _set32(attr, 8, path_mtu)
        _set32(attr, 20, self._peer_psn)
        _set32(attr, 28, self._peer_qpn)
        _set16(attr, 80, self._peer_lid)
        _set8(attr, 82, 0)  # sl
        _set8(attr, 83, 0)  # src_path_bits
        _set8(attr, 86, self._cfg.ib_port)
        # Always populate GRH; cross-NIC requires it and same-NIC tolerates it.
        for i, b in enumerate(self._peer_gid[:16]):
            _set8(attr, 56 + i, b)
        _set8(attr, 76, self._cfg.sgid_index)
        _set8(attr, 77, 64)  # hop_limit
        _set8(attr, 85, 1)  # ah_attr.is_global = 1
        _set8(attr, 127, 1)  # max_dest_rd_atomic
        _set8(attr, 128, 12)  # min_rnr_timer
        mask = (
            _IBV_QP_STATE
            | _IBV_QP_AV
            | _IBV_QP_PATH_MTU
            | _IBV_QP_DEST_QPN
            | _IBV_QP_RQ_PSN
            | _IBV_QP_MAX_DEST_RD_ATOMIC
            | _IBV_QP_MIN_RNR_TIMER
        )
        if self._libverbs.ibv_modify_qp(self._qp, _voidp(attr), mask) != 0:
            raise RuntimeError("modify_qp RTR failed errno=%d" % ctypes.get_errno())

    def _qp_to_rts(self):
        attr = _buf(144)
        _set32(attr, 0, _IBV_QPS_RTS)
        _set32(attr, 24, self._local_psn)
        _set8(attr, 126, 1)  # max_rd_atomic
        _set8(attr, 130, 14)  # timeout
        _set8(attr, 131, 7)  # retry_cnt
        _set8(attr, 132, 7)  # rnr_retry
        mask = (
            _IBV_QP_STATE
            | _IBV_QP_TIMEOUT
            | _IBV_QP_RETRY_CNT
            | _IBV_QP_RNR_RETRY
            | _IBV_QP_SQ_PSN
            | _IBV_QP_MAX_QP_RD_ATOMIC
        )
        if self._libverbs.ibv_modify_qp(self._qp, _voidp(attr), mask) != 0:
            raise RuntimeError("modify_qp RTS failed errno=%d" % ctypes.get_errno())

    def _collect_local_info(self):
        self._local_qpn = _read_qp_num(self._qp)
        # Pick a stable PSN; a deterministic value is fine for RC, but real
        # deployments often randomize.  Keep it simple here.
        self._local_psn = 0
        return _LocalQpInfo(
            qpn=self._local_qpn,
            psn=self._local_psn,
            lid=self._local_lid,
            gid=self._local_gid,
            active_mtu=self._local_active_mtu,
            recv_base=self._recv_base,
            rkey=self._recv_rkey,
            max_num_requests=self._cfg.max_num_requests,
            payload_bytes=self._cfg.payload_bytes,
        )

    def _tcp_handshake(self, local):
        my_payload = {
            "qpn": local.qpn,
            "psn": local.psn,
            "lid": local.lid,
            "gid": local.gid.hex(),
            "active_mtu": local.active_mtu,
            "recv_base": local.recv_base,
            "rkey": local.rkey,
            "max_num_requests": local.max_num_requests,
            "payload_bytes": local.payload_bytes,
            "peer_name": self._cfg.remote_peer_name,
        }

        if self._cfg.is_server:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.bind(("0.0.0.0", self._cfg.local_port))
                srv.listen(1)
                srv.settimeout(self._cfg.handshake_timeout_s)
                client, _ = srv.accept()
            finally:
                srv.close()
            try:
                # Server reads first then writes (avoids deadlock if peer
                # writes before reading).
                peer = _recv_json_line(client, self._cfg.handshake_timeout_s)
                _send_json_line(client, my_payload)
            finally:
                client.close()
        else:
            deadline = time.monotonic() + float(self._cfg.handshake_timeout_s)
            last_err = None
            client = None
            while time.monotonic() < deadline:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(10.0)
                try:
                    client.connect((self._cfg.remote_host, self._cfg.remote_port))
                    break
                except (ConnectionRefusedError, socket.timeout, OSError) as exc:
                    last_err = exc
                    try:
                        client.close()
                    except OSError:
                        pass
                    client = None
                    time.sleep(0.2)
            if client is None:
                raise last_err or RuntimeError("ibverbs TCP handshake connect failed")
            client.settimeout(self._cfg.handshake_timeout_s)
            try:
                _send_json_line(client, my_payload)
                peer = _recv_json_line(client, self._cfg.handshake_timeout_s)
            finally:
                client.close()

        # Trust but sanity-check.
        if int(peer.get("payload_bytes", 0)) != self._cfg.payload_bytes:
            raise RuntimeError(
                "peer payload_bytes mismatch (peer=%r local=%d)"
                % (peer.get("payload_bytes"), self._cfg.payload_bytes)
            )
        self._peer_qpn = int(peer["qpn"])
        self._peer_psn = int(peer["psn"])
        self._peer_lid = int(peer["lid"])
        self._peer_gid = bytes.fromhex(peer["gid"])
        self._peer_active_mtu = int(peer["active_mtu"])
        self._peer_recv_base = int(peer["recv_base"])
        self._peer_rkey = int(peer["rkey"])
        self._peer_max_num_requests = int(peer["max_num_requests"])

    def _post_recv_credit(self):
        """Post one recv WR as a credit.

        WRITE_WITH_IMM consumes recv WRs in FIFO regardless of the SGE
        in the WR — the data lands at the address the peer chose with
        ``wr.wr.rdma.remote_addr``.  So this WR is purely a credit; the
        SGE points to a tiny dummy buffer with a valid lkey.
        """
        sge = _buf(16)
        _set64(sge, 0, self._recv_dummy_addr)
        _set32(sge, 8, 1)
        _set32(sge, 12, self._recv_dummy_lkey)

        wr = _buf(32)
        wr_id = self._next_recv_wr_id
        self._next_recv_wr_id = (self._next_recv_wr_id + 1) & 0xFFFFFFFFFFFFFFFF
        _set64(wr, 0, wr_id)
        _set64(wr, 8, 0)  # next = NULL
        _set64(wr, 16, ctypes.addressof(sge))
        _set32(wr, 24, 1)
        bad = ctypes.c_void_p()
        rc = self._libwrap.wrap_ibv_post_recv(self._qp, _voidp(wr), ctypes.byref(bad))
        if rc != 0:
            raise RuntimeError("ibv_post_recv rc=%d errno=%d" % (rc, ctypes.get_errno()))
        self._posted_recv_wrs[wr_id] = (sge, wr)

    def _do_send(self, packet):
        payload = self._cfg.payload_bytes
        data = bytes(packet.payload)
        if len(data) != payload:
            raise RuntimeError("payload size mismatch: got %d want %d" % (len(data), payload))
        # Slot is encoded in imm_data bits [20:32) — see SpecDecodeChannel.
        slot = (int(packet.imm_data) >> 20) & 0x0FFF
        if slot >= self._peer_max_num_requests:
            raise RuntimeError(
                "slot %d exceeds peer max_num_requests=%d" % (slot, self._peer_max_num_requests)
            )

        # Stage payload into send buffer.
        ctypes.memmove(self._send_buf, data, payload)

        sge = _buf(16)
        _set64(sge, 0, self._send_addr)
        _set32(sge, 8, payload)
        _set32(sge, 12, self._send_lkey)

        wr = _buf(128)
        _set64(wr, 0, 0)
        _set64(wr, 8, 0)
        _set64(wr, 16, ctypes.addressof(sge))
        _set32(wr, 24, 1)
        _set32(wr, 28, _IBV_WR_RDMA_WRITE_WITH_IMM)
        _set32(wr, 32, _IBV_SEND_SIGNALED)
        _set32(wr, 36, int(packet.imm_data) & 0xFFFFFFFF)
        remote_addr = self._peer_recv_base + slot * payload
        _set64(wr, 40, remote_addr)
        _set32(wr, 48, self._peer_rkey)

        bad = ctypes.c_void_p()
        rc = self._libwrap.wrap_ibv_post_send(self._qp, _voidp(wr), ctypes.byref(bad))
        if rc != 0:
            raise RuntimeError("ibv_post_send rc=%d errno=%d" % (rc, ctypes.get_errno()))

        # Wait for send completion (small timeout — RC + WRITE is fast).
        deadline = time.monotonic() + 5.0
        while True:
            wc = _ibv_wc()
            n = self._libwrap.wrap_ibv_poll_cq(self._cq_send, 1, ctypes.byref(wc))
            if n < 0:
                raise RuntimeError("send-cq poll failed")
            if n > 0:
                if wc.status != _IBV_WC_SUCCESS:
                    raise RuntimeError("send wc status=%d" % wc.status)
                break
            if time.monotonic() > deadline:
                raise TimeoutError("send CQ poll timeout")
            time.sleep(0)

    def _drain_recv_cq(self):
        if self._cq_recv is None:
            return
        wc = _ibv_wc()
        while True:
            n = self._libwrap.wrap_ibv_poll_cq(self._cq_recv, 1, ctypes.byref(wc))
            if n < 0:
                raise RuntimeError("recv-cq poll failed")
            if n == 0:
                return
            wr_id = int(wc.wr_id)
            # Release the WR's bookkeeping memory regardless of status.
            self._posted_recv_wrs.pop(wr_id, None)
            self._recv_credits = max(0, self._recv_credits - 1)
            if wc.status != _IBV_WC_SUCCESS:
                _debug_trace("recv WC bad status=%d wr_id=%d", wc.status, wr_id)
                continue
            # Decode the destination slot from imm_data — that's where
            # the WRITE_WITH_IMM landed our payload.
            imm = int(wc.imm_data) & 0xFFFFFFFF
            slot = (imm >> 20) & 0x0FFF
            if slot >= self._cfg.max_num_requests:
                _debug_trace("recv slot=%d out of range", slot)
                continue
            payload_off = slot * self._cfg.payload_bytes
            payload = bytes(self._recv_buf[payload_off : payload_off + self._cfg.payload_bytes])
            self._inbox.append(EndpointPacket(imm_data=imm, payload=payload))

    def _cleanup(self):
        try:
            if self._qp:
                self._libverbs.ibv_destroy_qp(self._qp)
                self._qp = None
            if self._cq_send:
                self._libverbs.ibv_destroy_cq(self._cq_send)
                self._cq_send = None
            if self._cq_recv:
                self._libverbs.ibv_destroy_cq(self._cq_recv)
                self._cq_recv = None
            if self._send_mr:
                self._libverbs.ibv_dereg_mr(self._send_mr)
                self._send_mr = None
            if self._recv_mr:
                self._libverbs.ibv_dereg_mr(self._recv_mr)
                self._recv_mr = None
            if getattr(self, "_recv_dummy_mr", None):
                self._libverbs.ibv_dereg_mr(self._recv_dummy_mr)
                self._recv_dummy_mr = None
            if self._pd:
                self._libverbs.ibv_dealloc_pd(self._pd)
                self._pd = None
            if self._ctx:
                self._libverbs.ibv_close_device(self._ctx)
                self._ctx = None
        except Exception as exc:
            _debug_trace("cleanup error: %r", exc)
        finally:
            self._send_buf = None
            self._recv_buf = None
            self._recv_dummy_buf = None
            self._posted_recv_wrs.clear()
            self._inbox.clear()
            self._recv_credits = 0
