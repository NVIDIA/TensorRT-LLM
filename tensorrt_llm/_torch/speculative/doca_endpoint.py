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
"""Endpoint protocol implementation over DOCA RDMA WRITE_WITH_IMM.

This follows Izzy's original data path: the GPU submits an RDMA
WRITE_WITH_IMM WQE, the 96-byte DraftApiProtocol payload is written to
the peer's receive buffer, and the 32-bit routing word is delivered as
hardware immediate data in the receive CQE. The receiver pre-posts empty
receive WRs and polls the GPU receive CQ for completions.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .draft_api_protocol import DraftApiProtocol
from .spec_decode_channel import EndpointPacket, EndpointStatus

# izzy's allocator packs small allocations into a shared CUDA region,
# which can alias send and recv buffers. Ask for a whole page each so
# the addresses are distinct.
_BUF_BYTES = 4096
_POST_RECV_BATCH = 32


def _env_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _debug_trace(message, *args) -> None:
    if not _env_enabled("TLLM_RDMA_DEBUG_TRACE"):
        return
    trace_path = os.environ.get("TLLM_RDMA_DEBUG_TRACE_PATH", "/tmp/tllm_rdma_trace.log")
    if not trace_path:
        return
    try:
        rendered = message % args if args else message
    except Exception:
        rendered = "%s args=%r" % (message, args)
    try:
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(
                "%s pid=%d doca_endpoint %s\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(), rendered)
            )
    except Exception:
        pass


@dataclass
class DocaEndpointConfig:
    """Connection config for ``_DocaEndpointBackend``.

    Mirrors ``IbverbsEndpointConfig`` so upper layers can swap transports
    by flipping a single string.
    """

    gpu_id: int = 0
    nic_name: str = "mlx5_0"
    is_server: bool = False
    remote_host: str = "127.0.0.1"
    remote_port: int = 47000
    bind_port: int = 0  # 0 ⇒ same as remote_port (single-side test)
    remote_peer_name: str = "draft_doca"
    recv_queue_depth: int = 256
    payload_bytes: int = DraftApiProtocol.kMessageBytes
    handshake_timeout_s: float = 180.0
    # Optional override for the packaged DOCA RDMA extension directory
    # (contains ``doca_rdma.cpython-*.so`` and ``doca_rdma_ops.so``).
    extension_dir: Optional[str] = None


def _packaged_extension_dir() -> Optional[str]:
    path = Path(__file__).resolve().parent / "doca_rdma_ext"
    if list(path.glob("doca_rdma*.so")) and (path / "doca_rdma_ops.so").exists():
        return str(path)
    return None


def _resolve_extension_dir(extension_dir: Optional[str]) -> str:
    return (
        extension_dir
        or os.environ.get("DOCA_RDMA_EXTENSION_DIR")
        or _packaged_extension_dir()
        or os.environ.get("DOCA_RDMA_BUILD")
        or "/opt/doca_rdma_izzy/build"
    )


def _load_doca_extension(extension_dir: Optional[str]):
    """Import the DOCA RDMA pybind module and load its torch ops library."""
    path = _resolve_extension_dir(extension_dir)
    _debug_trace("load doca extension_dir=%s resolved_path=%s", extension_dir, path)
    if path and path not in sys.path:
        sys.path.insert(0, path)
    import torch

    ops_path = os.path.join(path, "doca_rdma_ops.so")
    if os.path.exists(ops_path):
        try:
            _debug_trace("load torch ops path=%s", ops_path)
            torch.ops.load_library(ops_path)
        except RuntimeError:
            # Already loaded — fine.
            _debug_trace("torch ops already loaded path=%s", ops_path)
            pass
    else:
        _debug_trace("torch ops missing path=%s", ops_path)
    import doca_rdma  # noqa: E402

    _debug_trace("imported doca_rdma module=%s", getattr(doca_rdma, "__file__", None))
    return doca_rdma


class _DocaEndpointBackend:
    """``Endpoint`` Protocol over DOCA RDMA WRITE_WITH_IMM."""

    def __init__(self, config: DocaEndpointConfig):
        self._cfg = config
        self._doca = None
        self._ep = None
        self._send_buf = None
        self._recv_buf = None
        self._send_payload = None  # torch tensor view of send buffer
        self._recv_payload = None
        self._send_seq = 0
        self._progress_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._recv_wq_idx = None
        self._recv_cq_idx = None
        self._recv_imm_out = None

        self._inbox = deque()
        self._inbox_lock = threading.Lock()

        self._recv_credits = 0
        self._started = False

        # Dedicated sender thread + stream. ``send()`` enqueues the
        # outgoing 96-byte payload here; the worker thread issues
        # ``write_with_imm`` on its own CUDA stream so successive sends
        # don't interleave with TRT-LLM model.forward kernels.
        self._send_queue = deque()
        self._send_queue_lock = threading.Lock()
        self._send_queue_cv = threading.Condition(self._send_queue_lock)
        self._send_done_cv = threading.Condition()
        self._send_completed_seq = 0
        self._send_submitted_seq = 0
        self._send_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Endpoint Protocol
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            _debug_trace("start skipped already started")
            return EndpointStatus.kOk
        try:
            mode = "server" if self._cfg.is_server else "client"
            port = int(
                (self._cfg.bind_port or self._cfg.remote_port)
                if self._cfg.is_server
                else self._cfg.remote_port
            )
            _debug_trace(
                "start mode=%s gpu_id=%s nic=%s remote_host=%s port=%s bind_port=%s "
                "peer=%s queue_depth=%s payload_bytes=%s timeout_s=%s extension_dir=%s",
                mode,
                self._cfg.gpu_id,
                self._cfg.nic_name,
                self._cfg.remote_host,
                port,
                self._cfg.bind_port,
                self._cfg.remote_peer_name,
                self._cfg.recv_queue_depth,
                self._cfg.payload_bytes,
                self._cfg.handshake_timeout_s,
                _resolve_extension_dir(self._cfg.extension_dir),
            )
            self._doca = _load_doca_extension(self._cfg.extension_dir)
            import torch

            _debug_trace("cuda set_device gpu_id=%s", self._cfg.gpu_id)
            torch.cuda.set_device(int(self._cfg.gpu_id))
            self._ep = self._doca.DocaRdmaEndpoint()
            _debug_trace(
                "endpoint init begin gpu_id=%s nic=%s",
                self._cfg.gpu_id,
                self._cfg.nic_name,
            )
            self._ep.init(int(self._cfg.gpu_id), str(self._cfg.nic_name))
            _debug_trace("endpoint init ok")
            _debug_trace("alloc send buffer bytes=%s", _BUF_BYTES)
            self._send_buf = self._ep.alloc_doca_gpu_buffer(_BUF_BYTES)
            _debug_trace(
                "alloc send buffer ok addr=%s lkey=%s rkey=%s",
                getattr(self._send_buf, "addr", None),
                getattr(self._send_buf, "lkey", None),
                getattr(self._send_buf, "rkey", None),
            )
            _debug_trace("alloc recv buffer bytes=%s", _BUF_BYTES)
            self._recv_buf = self._ep.alloc_doca_gpu_buffer(_BUF_BYTES)
            _debug_trace(
                "alloc recv buffer ok addr=%s lkey=%s rkey=%s",
                getattr(self._recv_buf, "addr", None),
                getattr(self._recv_buf, "lkey", None),
                getattr(self._recv_buf, "rkey", None),
            )

            _debug_trace("wrap send payload")
            self._send_payload = torch.ops.doca_rdma.wrap_gpu_memory(
                self._send_buf.addr, _BUF_BYTES, int(self._cfg.gpu_id)
            ).view(torch.uint8)
            _debug_trace("wrap recv payload")
            self._recv_payload = torch.ops.doca_rdma.wrap_gpu_memory(
                self._recv_buf.addr, _BUF_BYTES, int(self._cfg.gpu_id)
            ).view(torch.uint8)

            _debug_trace("zero buffers begin")
            self._send_payload.zero_()
            self._recv_payload.zero_()
            torch.cuda.synchronize()
            _debug_trace("zero buffers ok")

            peer = str(self._cfg.remote_peer_name)
            if self._cfg.is_server:
                port = int(self._cfg.bind_port or self._cfg.remote_port)
                _debug_trace("accept peer begin port=%s peer=%s", port, peer)
                ok = self._ep.accept_peer(port, peer)
                _debug_trace("accept peer result ok=%s", ok)
            else:
                # The draft side may still be opening its DOCA accept
                # port after loading the model. Retry connect until
                # handshake_timeout_s elapses.
                deadline = time.monotonic() + float(self._cfg.handshake_timeout_s)
                ok = False
                attempt = 0
                while time.monotonic() < deadline:
                    attempt += 1
                    _debug_trace(
                        "connect peer attempt=%s host=%s port=%s peer=%s",
                        attempt,
                        self._cfg.remote_host,
                        self._cfg.remote_port,
                        peer,
                    )
                    ok = self._ep.connect_to_peer(
                        str(self._cfg.remote_host), int(self._cfg.remote_port), peer
                    )
                    if ok:
                        _debug_trace("connect peer ok attempt=%s", attempt)
                        break
                    time.sleep(1.0)
                if not ok:
                    _debug_trace("connect peer failed attempts=%s", attempt)
            if not ok:
                _debug_trace("peer handshake failed")
                return EndpointStatus.kError
            _debug_trace(
                "exchange buffer info begin peer=%s send_addr=%s recv_addr=%s",
                peer,
                getattr(self._send_buf, "addr", None),
                getattr(self._recv_buf, "addr", None),
            )
            ok = self._ep.exchange_buffer_info(peer, self._send_buf, self._recv_buf)
            if not ok:
                _debug_trace("exchange buffer info failed")
                return EndpointStatus.kError
            _debug_trace("exchange buffer info ok")

            qp_ptr = self._ep.get_qp_dev_ptr(peer)
            self._recv_wq_idx = torch.zeros(
                1, dtype=torch.int64, device=f"cuda:{int(self._cfg.gpu_id)}"
            )
            self._recv_cq_idx = torch.zeros(
                1, dtype=torch.int64, device=f"cuda:{int(self._cfg.gpu_id)}"
            )
            self._recv_imm_out = torch.zeros(
                1, dtype=torch.int32, device=f"cuda:{int(self._cfg.gpu_id)}"
            )
            post_count = min(max(1, int(self._cfg.recv_queue_depth)), _POST_RECV_BATCH)
            _debug_trace("post recv WRs begin peer=%s count=%s", peer, post_count)
            torch.ops.doca_rdma.post_recv_wrs(qp_ptr, self._recv_wq_idx, post_count)
            torch.cuda.synchronize()
            _debug_trace("post recv WRs ok")

            self._progress_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="DocaEp-poll", daemon=True
            )
            self._poll_thread.start()
            self._started = True
            self._send_thread = threading.Thread(
                target=self._send_worker_loop, name="DocaEp-send", daemon=True
            )
            self._send_thread.start()
            _debug_trace("start ok threads started")
            return EndpointStatus.kOk
        except Exception as exc:
            _debug_trace("start exception: %r", exc)
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        self._started = False
        self._progress_stop.set()
        with self._send_queue_cv:
            self._send_queue_cv.notify_all()
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

        self._send_seq = (self._send_seq + 1) & 0x7FFFFFFF
        imm32 = int(packet.imm_data) & 0xFFFFFFFF
        my_seq = self._send_seq

        # Hand the WQE to the dedicated sender thread. It runs the
        # write_with_imm kernel on its own CUDA stream so the SQ submit
        # never interleaves with kernels that TRT-LLM's model.forward
        # is queueing on the caller's stream.
        with self._send_queue_cv:
            self._send_queue.append((my_seq, imm32, payload))
            self._send_queue_cv.notify()
        with self._send_done_cv:
            deadline = time.monotonic() + 30.0
            while self._send_completed_seq < my_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return EndpointStatus.kError
                self._send_done_cv.wait(timeout=remaining)
        self._recv_credits -= 1
        return EndpointStatus.kOk

    def _send_worker_loop(self):
        """Pop one outgoing packet at a time, run write_with_imm on a
        dedicated CUDA stream, wait for the kernel, then notify the
        caller. Single-threaded — guarantees strict round-by-round
        ordering of WQEs into the SQ.

        The payload is copied into the DOCA-allocated GPU send buffer,
        then sent with hardware immediate data. The receiver gets the
        routing word from the CQE, matching Izzy's original mock client.
        """
        import torch

        torch.cuda.set_device(int(self._cfg.gpu_id))
        sender_stream = torch.cuda.Stream(device=int(self._cfg.gpu_id))
        peer = self._cfg.remote_peer_name
        with torch.inference_mode():
            while True:
                with self._send_queue_cv:
                    while self._started and not self._send_queue:
                        self._send_queue_cv.wait()
                    if not self._started:
                        return
                    my_seq, imm32, payload = self._send_queue.popleft()

                with torch.cuda.stream(sender_stream):
                    src = torch.tensor(
                        list(payload),
                        dtype=torch.uint8,
                        device=f"cuda:{int(self._cfg.gpu_id)}",
                    )
                    self._send_payload[: DraftApiProtocol.kMessageBytes].copy_(src)
                    remote = self._ep.get_remote_recv_buf(peer)
                    local_send = self._ep.get_local_send_buf(peer)
                    qp_ptr = self._ep.get_qp_dev_ptr(peer)
                    wqe_out = torch.zeros(
                        1, dtype=torch.int64, device=f"cuda:{int(self._cfg.gpu_id)}"
                    )
                    torch.ops.doca_rdma.write_with_imm(
                        qp_ptr,
                        local_send.addr,
                        DraftApiProtocol.kMessageBytes,
                        local_send.lkey,
                        remote.addr,
                        remote.rkey,
                        imm32,
                        wqe_out,
                    )
                    torch.ops.doca_rdma.wait_send(qp_ptr, self._send_payload, wqe_out)
                    sender_stream.synchronize()
                # Force the wqe_idx readback so the kernel is observed
                # to have completed before we mark the seq done.
                _ = int(wqe_out.cpu().item())
                self._send_submitted_seq = my_seq
                with self._send_done_cv:
                    self._send_completed_seq = my_seq
                    self._send_done_cv.notify_all()

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        with self._inbox_lock:
            if not self._inbox:
                return EndpointStatus.kEmpty, None
            return EndpointStatus.kOk, self._inbox.popleft()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _poll_loop(self):
        import torch

        torch.cuda.set_device(int(self._cfg.gpu_id))
        poll_stream = torch.cuda.Stream(device=int(self._cfg.gpu_id))
        peer = self._cfg.remote_peer_name
        qp_ptr = self._ep.get_qp_dev_ptr(peer)
        with torch.inference_mode():
            while not self._progress_stop.is_set():
                try:
                    with torch.cuda.stream(poll_stream):
                        completed = torch.ops.doca_rdma.poll_recv(
                            qp_ptr, self._recv_cq_idx, self._recv_imm_out
                        )
                        poll_stream.synchronize()
                    if int(completed.cpu().item()) == 0:
                        time.sleep(0.0005)
                        continue
                    imm = int(self._recv_imm_out.cpu().item()) & 0xFFFFFFFF
                    payload = bytes(
                        self._recv_payload[: DraftApiProtocol.kMessageBytes].cpu().tolist()
                    )
                    with self._inbox_lock:
                        self._inbox.append(EndpointPacket(imm_data=imm, payload=payload))
                    with torch.cuda.stream(poll_stream):
                        torch.ops.doca_rdma.post_recv_wrs(qp_ptr, self._recv_wq_idx, 1)
                        poll_stream.synchronize()
                except Exception:
                    return

    def _cleanup(self):
        self._progress_stop.set()
        # Threads are daemon — they'll exit when the program exits.
        # Best effort: nudge the endpoint shutdown.
        try:
            if self._ep is not None:
                self._ep.cleanup()
        except Exception:
            pass
        self._ep = None
