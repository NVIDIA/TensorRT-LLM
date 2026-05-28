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
"""libibverbs Draft Token Offload Layer.

Business-level peer of izzy's ``DocaDraftOffloadLayer`` — same surface,
same forward shape, same round-seq semantics, but the transport sits on
``_IbverbsRdmaBackend`` instead of DOCA.  Because the upper three layers
(business / route / protocol) are identical, swapping endpoints leaves
this code Largely unchanged from the izzy reference.

Phase 4 supports ``batch_size == 1`` only (matches izzy's default
``_serialize_multi_request=True``); multi-request lands in Phase 9.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from torch import nn

    _NN_MODULE_BASE = nn.Module
except Exception:  # pragma: no cover — kept importable without torch
    torch = None  # type: ignore
    _NN_MODULE_BASE = object  # type: ignore

from tensorrt_llm._utils import prefer_pinned

from .pearl_trace import enabled as _pearl_trace_enabled
from .pearl_trace import log as _pearl_log
from .pearl_trace import to_int_list as _pearl_to_int_list

HEADER_NUM_FIELDS = 8
HEADER_BYTES = HEADER_NUM_FIELDS * 4  # kept for parity with izzy / future use
LLAMA3_TERMINAL_TOKENS = frozenset({128001, 128008, 128009})


@dataclass
class IbverbsDraftOffloadConfig:
    """Configuration for libibverbs RDMA draft token offloading.

    Mirrors ``DraftOffloadConfig`` from the izzy reference implementation.
    """

    nic_name: str = ""
    server_host: str = "localhost"
    server_port: int = 47000
    remote_peer_name: str = "draft_lpu"
    max_num_tokens: int = 8192
    max_num_requests: int = 256
    max_draft_len: int = 5
    mock: bool = False
    use_torch_ops: bool = False  # ibverbs has no GPU-initiated path; kept for API parity
    torch_ops_lib_path: Optional[str] = None
    protocol_max_position: int = (1 << 32) - 1
    protocol_enforce_reserved_zeros: bool = True
    # Phase 7 TCP prompt-init control channel.  0 = disabled.
    tcp_prompt_port: int = 0
    # ibverbs-specific knobs:
    is_server: bool = False
    local_port: int = 0
    sgid_index: int = 0
    pkey_index: int = 0
    ib_port: int = 1
    # Transport selection. Default is ``ibverbs``; ``tcp`` selects the
    # pure-TCP endpoint (used for the spec-decode-over-RDMA progression's
    # phase 1 baseline). When ``tcp`` is used, ``nic_name`` / ``sgid_*``
    # / ``ib_port`` are ignored, and the endpoint connects to
    # ``server_host:server_port`` (or binds, if ``is_server`` is set).
    transport: str = "ibverbs"
    # When ``transport == "tcp"``, the target sends a ``TcpModelInit`` to
    # the draft server before the first RDMA round so the draft can lazy-
    # load this exact model. ``None`` skips the lazy init (draft is
    # expected to already have the model loaded out-of-band).
    draft_model_path: Optional[str] = None
    draft_model_dtype: str = "bfloat16"
    draft_kv_cache_free_fraction: float = 0.4
    # When ``transport == "shm"``, both peers attach the same pair of
    # POSIX shared-memory regions (``<shm_name>_t2d`` / ``<shm_name>_d2t``).
    # Default chosen to be unique enough for a typical single-pair test.
    shm_name: str = "pearl_shm_default"
    # When ``transport == "cudaipc"``, the prefix of the CPU meta region
    # names; the GPU data rings are exchanged via cudaIpc handles embedded
    # in those meta regions. Same uniqueness requirement as ``shm_name``.
    cudaipc_name: str = "pearl_ipc_default"


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_enabled(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
                "%s pid=%d ibverbs_draft_offload %s\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(), rendered)
            )
    except Exception:
        pass


def _tensor_row_preview(tensor, row, *, count=None, limit=8):
    if tensor is None:
        return []
    try:
        row_tensor = tensor[row].detach().reshape(-1).cpu()
    except Exception:
        return []
    total = int(row_tensor.numel())
    visible = total if count is None else max(0, min(int(count), total))
    return row_tensor[: min(visible, int(limit))].tolist()


class IbverbsDraftOffloadLayer(_NN_MODULE_BASE):
    """Draft offload transport over SpecDecodeChannel + ibverbs backend."""

    def __init__(self, config: IbverbsDraftOffloadConfig, endpoint=None):
        super().__init__()
        _debug_trace(
            "__init__ entering nic=%s remote=%s:%s peer=%s max_requests=%s max_draft_len=%s",
            config.nic_name,
            config.server_host,
            config.server_port,
            config.remote_peer_name,
            config.max_num_requests,
            config.max_draft_len,
        )
        from .draft_api_protocol import DraftApiProtocol
        from .spec_decode_channel import SpecDecodeChannel

        self.config = config
        self._protocol = DraftApiProtocol
        self._channel_cls = SpecDecodeChannel

        if endpoint is not None:
            self._endpoint = endpoint
        elif config.transport == "tcp":
            from .tcp_endpoint import TcpEndpointConfig, _TcpRdmaBackend

            tcp_cfg = TcpEndpointConfig(
                is_server=config.is_server,
                remote_host=config.server_host,
                remote_port=config.server_port,
                bind_host="0.0.0.0",
                bind_port=config.local_port or config.server_port,
                recv_queue_depth=config.max_num_requests,
                payload_bytes=DraftApiProtocol.kMessageBytes,
            )
            self._endpoint = _TcpRdmaBackend(tcp_cfg)
        elif config.transport == "shm":
            from .shm_endpoint import ShmEndpointConfig, _ShmBackend

            shm_cfg = ShmEndpointConfig(
                is_server=config.is_server,
                shm_name=getattr(config, "shm_name", "pearl_shm_default"),
                recv_queue_depth=config.max_num_requests,
                payload_bytes=DraftApiProtocol.kMessageBytes,
            )
            self._endpoint = _ShmBackend(shm_cfg)
        elif config.transport == "cudaipc":
            from .cudaipc_endpoint import CudaIpcEndpointConfig, _CudaIpcBackend

            ipc_cfg = CudaIpcEndpointConfig(
                is_server=config.is_server,
                name=getattr(config, "cudaipc_name", "pearl_ipc_default"),
                recv_queue_depth=config.max_num_requests,
                payload_bytes=DraftApiProtocol.kMessageBytes,
            )
            self._endpoint = _CudaIpcBackend(ipc_cfg)
        elif config.transport == "doca":
            from .doca_endpoint import DocaEndpointConfig, _DocaEndpointBackend

            # ``CUDA_VISIBLE_DEVICES`` already masks the visible GPU, so
            # torch sees the worker's chosen device as index 0. izzy's
            # init() takes the masked index.
            doca_cfg = DocaEndpointConfig(
                gpu_id=0,
                nic_name=config.nic_name,
                is_server=config.is_server,
                remote_host=config.server_host,
                remote_port=config.server_port,
                bind_port=config.local_port or config.server_port,
                remote_peer_name=config.remote_peer_name,
                recv_queue_depth=config.max_num_requests,
                payload_bytes=DraftApiProtocol.kMessageBytes,
            )
            self._endpoint = _DocaEndpointBackend(doca_cfg)
        else:
            from .ibverbs_endpoint import IbverbsEndpointConfig, _IbverbsRdmaBackend

            endpoint_cfg = IbverbsEndpointConfig(
                nic_name=config.nic_name,
                remote_host=config.server_host,
                remote_port=config.server_port,
                remote_peer_name=config.remote_peer_name,
                recv_queue_depth=config.max_num_requests,
                payload_bytes=DraftApiProtocol.kMessageBytes,
                max_num_requests=config.max_num_requests,
                sgid_index=config.sgid_index,
                pkey_index=config.pkey_index,
                ib_port=config.ib_port,
                is_server=config.is_server,
                local_port=config.local_port,
                use_torch_ops=False,
                torch_ops_lib_path=None,
            )
            self._endpoint = _IbverbsRdmaBackend(endpoint_cfg)
        _debug_trace(
            "__init__ endpoint ready type=%s transport=%s",
            type(self._endpoint).__name__,
            config.transport,
        )

        validation_options = DraftApiProtocol.ValidationOptions(
            max_position=config.protocol_max_position,
            enforce_reserved_zeros=config.protocol_enforce_reserved_zeros,
        )
        self._channel = SpecDecodeChannel(self._endpoint, validation_options=validation_options)

        self._started = False
        self._bound_requests = set()
        self._retired_requests = set()
        self._next_route_probe = {}
        self._last_sent_round_seq_by_request = {}
        self._last_recv_round_seq_by_request = {}
        self._pending_round_seq_by_request = {}
        self._round_seq_tensor_by_request = {}
        # ibverbs has no GPU-initiated graph path; keep field for parity.
        self._batch1_graph_fast_path_enabled = False
        self._serialize_multi_request = _env_enabled(
            "TLLM_RDMA_DRAFT_OFFLOAD_SERIALIZE_MULTI_REQUEST", True
        )
        self._cached_routes = {}  # request_id -> Route
        self.last_response_token_counts = None
        # Phase 7 TCP prefill
        self._tcp_prompt_host = config.server_host
        self._tcp_prompt_port = getattr(config, "tcp_prompt_port", 0)
        self._tcp_prompt_timeout_s = 30.0
        self._prompt_pushed_requests = set()

        # Sync-batching: every cycle's send path needs accepted_tokens /
        # num_accepted_tokens / position_ids values on the host (to build the
        # protocol message). Naively each is read via .item()/.cpu().tolist(),
        # producing ~5-8 host-device syncs per cycle. Prefetch all three into
        # pre-allocated pinned host buffers with a SINGLE stream sync; the
        # CPU-side reads that follow become memory loads, not GPU round-trips.
        self._pearl_trace_enabled_send = _pearl_trace_enabled("target")
        self._pinned_accepted_buf = None  # lazy-init on first forward()
        self._pinned_num_accepted_buf = None
        self._pinned_positions_buf = None

    # ------------------------------------------------------------------
    # Lifecycle / route management
    # ------------------------------------------------------------------

    def _ensure_started(self):
        if self._started:
            return
        # Order matters when transport is tcp/ibverbs/doca: tell the draft
        # server which model to load FIRST (slow — minutes for an 8B
        # model), then open the data-plane socket. Otherwise data-plane
        # connect() races a not-yet-ready draft.
        if (
            self.config.transport in ("tcp", "ibverbs", "doca", "shm", "cudaipc")
            and self.config.draft_model_path
        ):
            self._push_tcp_model_init()
        status = self._channel.start()
        if status != self._channel_cls.Status.kOk:
            raise RuntimeError(
                "Failed to start draft offload channel: " + self._channel_cls.to_string(status)
            )
        status = self._channel.prime_recv(self.config.max_num_requests)
        if status not in (
            self._channel_cls.Status.kOk,
            self._channel_cls.Status.kEndpointQueueFull,
        ):
            raise RuntimeError(
                "Failed to prime draft offload receives: " + self._channel_cls.to_string(status)
            )
        self._started = True

    @staticmethod
    def _validate_request_ids(request_ids, batch_size):
        if request_ids is None:
            return list(range(batch_size))
        out = [int(r) for r in request_ids]
        if len(out) != batch_size:
            raise ValueError(
                "request_ids length (%d) must match batch_size (%d)" % (len(out), batch_size)
            )
        return out

    def _unbind_request(self, request_id):
        self._channel.unbind_request_route(request_id)
        self._bound_requests.discard(request_id)
        self._next_route_probe.pop(request_id, None)
        self._last_sent_round_seq_by_request.pop(request_id, None)
        self._last_recv_round_seq_by_request.pop(request_id, None)
        self._pending_round_seq_by_request.pop(request_id, None)
        self._round_seq_tensor_by_request.pop(request_id, None)
        self._cached_routes.pop(request_id, None)
        self._prompt_pushed_requests.discard(request_id)

    def _bind_route_for_request(self, request_id):
        route_status, route = self._channel.route_for_request(request_id)
        if route_status == self._channel_cls.Status.kOk:
            self._bound_requests.add(request_id)
            self._cached_routes[request_id] = route
            return
        if route_status != self._channel_cls.Status.kRouteNotFound:
            raise RuntimeError(
                "Route lookup failed for request %d: %s"
                % (request_id, self._channel_cls.to_string(route_status))
            )

        slot_max = self._channel_cls.kImmSlotMax
        base_slot = self._next_route_probe.get(request_id, request_id & slot_max)
        for offset in range(slot_max + 1):
            slot = (base_slot + offset) & slot_max
            route = self._channel_cls.Route(stream_id=0, slot=slot)
            bind_status = self._channel.bind_request_route(request_id, route)
            if bind_status == self._channel_cls.Status.kOk:
                self._bound_requests.add(request_id)
                self._cached_routes[request_id] = route
                self._next_route_probe[request_id] = (slot + 1) & slot_max
                return
            if bind_status == self._channel_cls.Status.kRouteInUse:
                continue
            raise RuntimeError(
                "Failed binding route for request %d: %s"
                % (request_id, self._channel_cls.to_string(bind_status))
            )
        raise RuntimeError("No route slots available for request %d" % request_id)

    def _ensure_routes(self, request_ids):
        for request_id in request_ids:
            self._bind_route_for_request(request_id)

    def _next_round_seq(self, request_id):
        round_seq = self._last_sent_round_seq_by_request.get(request_id, 0) + 1
        self._last_sent_round_seq_by_request[request_id] = round_seq
        return round_seq

    def bind_requests(self, request_ids):
        if not request_ids:
            return
        unique = list(dict.fromkeys(int(r) for r in request_ids))
        unbound = [
            r for r in unique if r not in self._bound_requests and r not in self._retired_requests
        ]
        if not unbound:
            return
        self._ensure_started()
        self._ensure_routes(unbound)

    def unbind_requests(self, request_ids):
        for r in dict.fromkeys(int(r) for r in request_ids):
            self._unbind_request(r)

    def retire_requests(self, request_ids):
        for r in dict.fromkeys(int(r) for r in request_ids):
            self._retired_requests.add(r)
            self._unbind_request(r)

    def reactivate_requests(self, request_ids):
        for r in dict.fromkeys(int(r) for r in request_ids):
            self._retired_requests.discard(r)

    # ------------------------------------------------------------------
    # TCP control plane
    # ------------------------------------------------------------------

    def _push_tcp_model_init(self):
        """Send ``TcpModelInit`` so the draft server lazy-loads its model.

        Only used by the TCP transport. Runs once during ``_ensure_started``
        before ``prime_recv``, so by the time the first 96-byte frame goes
        out the draft side already has the model warm.
        """
        if not self.config.draft_model_path:
            return
        import json as _json
        import socket as _socket

        msg = {
            "msg_type": "model_init",
            "model_path": str(self.config.draft_model_path),
            "dtype": str(self.config.draft_model_dtype),
            "max_draft_len": int(self.config.max_draft_len),
            "kv_cache_free_fraction": float(self.config.draft_kv_cache_free_fraction),
            "extra_kwargs_json": _json.dumps(
                {
                    "transport": str(self.config.transport),
                    "data_port": int(self.config.server_port),
                    "nic_name": str(self.config.nic_name),
                    "max_num_requests": int(self.config.max_num_requests),
                    "shm_name": str(getattr(self.config, "shm_name", "")),
                    "cudaipc_name": str(getattr(self.config, "cudaipc_name", "")),
                }
            ),
        }
        _pearl_log("target", "control_model_init_send", message=msg)
        port = int(self._tcp_prompt_port if self._tcp_prompt_port > 0 else self.config.server_port)
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        # Loading a model on the draft side can take 60s+ (8B model + KV
        # cache warmup), so the ack timeout is intentionally generous.
        # First-time JIT kernel compilation can push it past 5 min.
        sock.settimeout(900.0)
        try:
            sock.connect((self._tcp_prompt_host, port))
            sock.sendall((_json.dumps(msg) + "\n").encode("utf-8"))
            chunks = []
            while True:
                ch = sock.recv(4096)
                if not ch:
                    raise RuntimeError("draft server closed during model-init ack")
                chunks.append(ch)
                if b"\n" in ch:
                    break
            line = b"".join(chunks).split(b"\n", 1)[0]
            ack = _json.loads(line.decode("utf-8"))
        finally:
            sock.close()
        if ack.get("status") != "ok":
            raise RuntimeError(
                "TcpModelInit failed: " + str(ack.get("error", "<no error message>"))
            )
        data_port = int(ack.get("data_port", 0) or 0)
        if data_port > 0 and data_port != int(self.config.server_port):
            self.config.server_port = data_port
            cfg = getattr(self._endpoint, "_cfg", None)
            if cfg is not None:
                if hasattr(cfg, "remote_port"):
                    cfg.remote_port = data_port
                if hasattr(cfg, "bind_port") and not getattr(cfg, "is_server", False):
                    cfg.bind_port = data_port
        _debug_trace(
            "TcpModelInit ack model=%s vocab=%s eos=%s data_port=%s",
            self.config.draft_model_path,
            ack.get("vocab_size"),
            ack.get("eos_token_id"),
            data_port,
        )
        _pearl_log("target", "control_model_init_ack", ack=ack)

    def push_prompt(self, request_id, prompt_tokens):
        """Send a ``TcpPromptInit`` over TCP to the draft server.

        Must be called before the first ``forward`` for the new request,
        otherwise the draft server has no KV-cache state to step from.
        Idempotent — repeated calls for the same request_id are no-ops
        until the request is unbound / retired.

        The TCP push must use the **slot** (the wire-level identifier
        carried in imm_data) as the session key, not the logical
        ``request_id``.  The draft server has no view of the target's
        request_id namespace — it only ever sees the slot on incoming
        RDMA packets — so the TCP push and the later RDMA traffic must
        agree on the slot.  We bind the request first to obtain the
        slot, then push under that slot.
        """
        if int(request_id) in self._prompt_pushed_requests:
            return
        if self._tcp_prompt_port <= 0:
            raise RuntimeError("tcp_prompt_port not configured; cannot push prompt")

        # Ensure the channel is started and the request has a route bound.
        self.bind_requests([request_id])
        route_status, route = self._channel.route_for_request(int(request_id))
        if route_status != self._channel_cls.Status.kOk or route is None:
            raise RuntimeError(
                "Cannot push prompt for request %d: no route bound (%s)"
                % (request_id, self._channel_cls.to_string(route_status))
            )
        wire_handle = int(route.slot)

        import json as _json
        import socket as _socket

        msg = {
            "msg_type": "prompt_init",
            "request_id": wire_handle,
            "prompt_tokens": [int(t) for t in prompt_tokens],
            "max_draft_len": int(self.config.max_draft_len),
        }
        _pearl_log(
            "target",
            "prompt_init_send",
            logical_request_id=int(request_id),
            wire_request_id=wire_handle,
            prompt_tokens=[int(t) for t in prompt_tokens],
            prompt_token_count=len(prompt_tokens),
            max_draft_len=int(self.config.max_draft_len),
        )
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(self._tcp_prompt_timeout_s)
        try:
            sock.connect((self._tcp_prompt_host, int(self._tcp_prompt_port)))
            sock.sendall((_json.dumps(msg) + "\n").encode("utf-8"))
            chunks = []
            while True:
                ch = sock.recv(4096)
                if not ch:
                    raise RuntimeError("draft server closed during prompt-init ack")
                chunks.append(ch)
                if b"\n" in ch:
                    break
            line = b"".join(chunks).split(b"\n", 1)[0]
            ack = _json.loads(line.decode("utf-8"))
        finally:
            sock.close()
        if ack.get("status") != "ok":
            raise RuntimeError(
                "prompt_init failed for request %d: %s" % (request_id, ack.get("error", "unknown"))
            )
        self._prompt_pushed_requests.add(int(request_id))
        _debug_trace(
            "push_prompt request_id=%s tokens=%d last_pos=%s",
            request_id,
            len(prompt_tokens),
            ack.get("last_token_position", -1),
        )
        _pearl_log(
            "target",
            "prompt_init_ack",
            logical_request_id=int(request_id),
            wire_request_id=wire_handle,
            ack=ack,
        )

    # ------------------------------------------------------------------
    # Per-request send / receive
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_accepted_token(accepted_tokens, num_accepted_tokens, row):
        width = accepted_tokens.shape[1]
        accepted = int(num_accepted_tokens[row].item())
        token_idx = max(0, min(accepted - 1, width - 1))
        return int(accepted_tokens[row, token_idx].item())

    @staticmethod
    def _resolve_request_start_positions(position_ids, batch_size):
        flat_positions = position_ids.flatten()
        if int(flat_positions.numel()) != int(batch_size):
            raise ValueError(
                "position_ids must provide exactly one start position per request "
                "(got %d, expected %d)" % (int(flat_positions.numel()), int(batch_size))
            )
        return [int(v) for v in flat_positions.detach().cpu().tolist()]

    def _build_target_to_draft_message(
        self, request_id, accepted_tokens, num_accepted_tokens, row, request_start_position
    ):
        # One packet carries exactly the target-verified/correction token and
        # that token's absolute position.  The draft side compares it against
        # its local speculative token at the same position.  A match commits
        # through that position; a mismatch rolls back to the previous token
        # and appends this target correction.
        tokens = [0] * self._protocol.kMaxTokens
        tokens[0] = self._extract_last_accepted_token(accepted_tokens, num_accepted_tokens, row)
        return self._protocol.Message(
            version=self._protocol.kVersionV1,
            message_type=self._protocol.MessageType.kTargetToDraft,
            round_seq_num=self._next_round_seq(request_id),
            position=int(request_start_position),
            num_tokens=1,
            tokens=tokens,
        )

    def _send_requests(
        self,
        request_ids,
        accepted_tokens,
        num_accepted_tokens,
        request_start_positions=None,
        gpu_accepted_tokens=None,
        gpu_num_accepted_tokens=None,
    ):
        if request_start_positions is None:
            request_start_positions = [0] * len(request_ids)
        if len(request_start_positions) != len(request_ids):
            raise ValueError("request_start_positions length must match request_ids")

        # Stage 3 (cudaipc + GPU compose): if the endpoint advertises the
        # ``set_next_send_gpu_tokens`` hook AND the caller handed us the
        # underlying GPU tensors, the kernel can read ``last_token``
        # directly from device memory and skip the CPU-side ``.item()`` in
        # ``_build_target_to_draft_message``.
        endpoint = getattr(self._channel, "_endpoint", None)
        use_gpu_compose = (
            gpu_accepted_tokens is not None
            and gpu_num_accepted_tokens is not None
            and endpoint is not None
            and hasattr(endpoint, "set_next_send_gpu_tokens")
        )
        max_draft_width = int(gpu_accepted_tokens.shape[1]) if use_gpu_compose else 0

        for row, request_id in enumerate(request_ids):
            if use_gpu_compose:
                round_seq = self._next_round_seq(request_id)
                # Build a placeholder message; the kernel overwrites
                # tokens[0] with the GPU-resident last accepted token. The
                # CPU-side message still flows through the channel layer
                # so the imm_data packing / route lookup paths stay shared.
                message = self._protocol.Message(
                    version=self._protocol.kVersionV1,
                    message_type=self._protocol.MessageType.kTargetToDraft,
                    round_seq_num=round_seq,
                    position=int(request_start_positions[row]),
                    num_tokens=1,
                    tokens=[0] * self._protocol.kMaxTokens,
                )
                # Hand the GPU pointers to the endpoint. Row slicing is a
                # contiguous view in the common case (max_draft_len
                # innermost dim), so ``.data_ptr()`` is the row start.
                endpoint.set_next_send_gpu_tokens(
                    int(gpu_accepted_tokens[row].data_ptr()),
                    int(gpu_num_accepted_tokens[row : row + 1].data_ptr()),
                    max_draft_width,
                    round_seq,
                    int(request_start_positions[row]),
                    int(self._protocol.kVersionV1),
                    int(self._protocol.MessageType.kTargetToDraft),
                    1,
                )
            else:
                message = self._build_target_to_draft_message(
                    request_id,
                    accepted_tokens,
                    num_accepted_tokens,
                    row,
                    request_start_positions[row],
                )
            status = self._channel.send_for_request(
                request_id,
                msg_type=int(self._protocol.MessageType.kTargetToDraft),
                message=message,
            )
            if status != self._channel_cls.Status.kOk:
                raise RuntimeError(
                    "Failed sending draft request %d: %s"
                    % (request_id, self._channel_cls.to_string(status))
                )
            self._pending_round_seq_by_request[request_id] = int(message.round_seq_num)
            # Trace fields are eagerly evaluated by Python, so even with the
            # log path disabled the build cost is paid. Gate the whole call.
            if self._pearl_trace_enabled_send:
                _pearl_log(
                    "target",
                    "send_target_to_draft",
                    request_id=int(request_id),
                    round_seq=int(message.round_seq_num),
                    position=int(message.position),
                    num_tokens=int(message.num_tokens),
                    last_token=int(message.tokens[0]),
                    accepted_token_count=int(num_accepted_tokens[row].item()),
                    accepted_tokens=_pearl_to_int_list(
                        accepted_tokens[row],
                        limit=int(
                            max(0, min(accepted_tokens.shape[1], num_accepted_tokens[row].item()))
                        ),
                    ),
                    packet={
                        "message_type": int(self._protocol.MessageType.kTargetToDraft),
                        "round_seq_num": int(message.round_seq_num),
                        "position": int(message.position),
                        "num_tokens": int(message.num_tokens),
                        "tokens": [int(t) for t in message.tokens],
                    },
                )
            _debug_trace(
                "_send_requests req_id=%s round_seq=%s start=%s last_token=%s",
                request_id,
                int(message.round_seq_num),
                int(request_start_positions[row]),
                int(message.tokens[0]),
            )

    def _validate_response_round_seq(self, request_id, received):
        expected = self._pending_round_seq_by_request.get(request_id)
        recv_round = int(received.message.round_seq_num)
        if expected is None:
            raise RuntimeError(
                "Received response for request %d without a pending round sequence" % request_id
            )
        if recv_round != int(expected):
            raise RuntimeError(
                "Round sequence mismatch for request %d: received=%d, expected=%d"
                % (request_id, recv_round, int(expected))
            )
        prev = self._last_recv_round_seq_by_request.get(request_id, 0)
        if recv_round <= prev:
            raise RuntimeError(
                "Non-monotonic round sequence for request %d: received=%d, previous=%d"
                % (request_id, recv_round, prev)
            )
        self._last_recv_round_seq_by_request[request_id] = recv_round
        self._pending_round_seq_by_request.pop(request_id, None)

    def _is_pending_route(self, pending_request_ids, route):
        for request_id in pending_request_ids:
            if self._cached_routes.get(request_id) == route:
                return True
        return False

    def _receive_responses(self, request_ids, device):
        next_draft_tokens = torch.zeros(
            (len(request_ids), self.config.max_draft_len),
            dtype=torch.int32,
            device=device,
        )
        next_draft_token_counts = torch.zeros(
            (len(request_ids),),
            dtype=torch.int32,
            device=device,
        )

        row_by_request = {r: i for i, r in enumerate(request_ids)}
        pending = set(request_ids)
        max_empty_polls = _env_int("TLLM_RDMA_DRAFT_OFFLOAD_MAX_EMPTY_POLLS", 0)
        timeout_s = _env_float("TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S", 2.0)
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        empty_polls = 0

        while pending:
            status, request_id, received = self._channel.pump_once_for_bound_request()

            if status == self._channel_cls.Status.kEndpointEmpty:
                empty_polls += 1
                if max_empty_polls > 0 and empty_polls > max_empty_polls:
                    raise RuntimeError(
                        "Timed out waiting for draft offload responses after %d empty polls"
                        % empty_polls
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Timed out waiting for draft offload responses "
                        "(pending=%s, timeout_s=%s)" % (sorted(pending), timeout_s)
                    )
                continue

            if status == self._channel_cls.Status.kRouteNotFound and received is not None:
                stale_route = self._channel_cls.Route(
                    stream_id=int(received.imm_data.stream_id),
                    slot=int(received.imm_data.slot),
                )
                if not self._is_pending_route(pending, stale_route):
                    _debug_trace(
                        "dropping unbound draft response route=%s pending=%s",
                        stale_route,
                        sorted(pending),
                    )
                    continue

            if status != self._channel_cls.Status.kOk:
                raise RuntimeError(
                    "Failed receiving draft response: " + self._channel_cls.to_string(status)
                )
            if received is None or request_id is None:
                raise RuntimeError("Draft response missing routed request metadata")

            if int(received.imm_data.msg_type) != int(self._protocol.MessageType.kDraftToTarget):
                raise RuntimeError(
                    "Unexpected response msg_type=%s, expected %d"
                    % (received.imm_data.msg_type, int(self._protocol.MessageType.kDraftToTarget))
                )

            row = row_by_request.get(request_id)
            if row is None:
                continue

            self._validate_response_round_seq(request_id, received)
            token_count = max(0, min(int(received.message.num_tokens), self.config.max_draft_len))
            next_draft_token_counts[row] = token_count
            if token_count > 0:
                next_draft_tokens[row, :token_count] = torch.tensor(
                    received.message.tokens[:token_count],
                    dtype=torch.int32,
                    device=device,
                )
            if self._pearl_trace_enabled_send:
                _pearl_log(
                    "target",
                    "recv_draft_to_target",
                    request_id=int(request_id),
                    route={
                        "stream_id": int(received.imm_data.stream_id),
                        "slot": int(received.imm_data.slot),
                        "msg_type": int(received.imm_data.msg_type),
                    },
                    round_seq=int(received.message.round_seq_num),
                    position=int(received.message.position),
                    num_tokens=int(received.message.num_tokens),
                    draft_tokens=[int(t) for t in received.message.tokens[:token_count]],
                    packet={
                        "message_type": int(self._protocol.MessageType.kDraftToTarget),
                        "round_seq_num": int(received.message.round_seq_num),
                        "position": int(received.message.position),
                        "num_tokens": int(received.message.num_tokens),
                        "tokens": [int(t) for t in received.message.tokens],
                    },
                )
            pending.discard(request_id)

        self.last_response_token_counts = next_draft_token_counts
        return next_draft_tokens

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _prefetch_send_data_to_pinned(
        self, accepted_tokens, num_accepted_tokens, position_ids, batch_size
    ):
        """Copy GPU verify-state tensors to pinned host buffers in one shot.

        Returns CPU-resident views backed by pinned memory; downstream
        ``.item()`` / ``.tolist()`` calls on these are pure memory loads and
        no longer trigger host-device syncs. A single
        ``current_stream().synchronize()`` here replaces the per-value syncs
        previously scattered across ``_extract_last_accepted_token`` (x2),
        ``_resolve_request_start_positions`` (x1), and the eager kwarg
        evaluation inside ``_pearl_log`` (x3+).
        """
        max_req = int(self.config.max_num_requests)
        width = int(accepted_tokens.shape[1])

        if (
            self._pinned_accepted_buf is None
            or self._pinned_accepted_buf.shape[0] < max_req
            or self._pinned_accepted_buf.shape[1] < width
        ):
            self._pinned_accepted_buf = torch.empty(
                (max_req, width), dtype=torch.int32, pin_memory=prefer_pinned()
            )
        if (
            self._pinned_num_accepted_buf is None
            or self._pinned_num_accepted_buf.shape[0] < max_req
        ):
            self._pinned_num_accepted_buf = torch.empty(
                (max_req,), dtype=torch.int32, pin_memory=prefer_pinned()
            )
        if self._pinned_positions_buf is None or self._pinned_positions_buf.shape[0] < max_req:
            self._pinned_positions_buf = torch.empty(
                (max_req,), dtype=torch.int64, pin_memory=prefer_pinned()
            )

        pinned_acc = self._pinned_accepted_buf[:batch_size, :width]
        pinned_acc.copy_(accepted_tokens, non_blocking=True)
        pinned_num = self._pinned_num_accepted_buf[:batch_size]
        pinned_num.copy_(num_accepted_tokens.flatten()[:batch_size], non_blocking=True)
        pinned_pos = self._pinned_positions_buf[:batch_size]
        pinned_pos.copy_(position_ids.flatten()[:batch_size], non_blocking=True)
        # One host-device sync covers all three async copies above. Without
        # this, the first .item() read below would force the sync anyway and
        # subsequent reads would each issue their own.
        torch.cuda.current_stream().synchronize()
        return pinned_acc, pinned_num, pinned_pos

    def _run_batch1_graph_fast_path(
        self, request_id, position_ids, accepted_tokens, num_accepted_tokens
    ):
        # Phase 4 disables the GPU-initiated fast path.  ibverbs has no
        # equivalent of DOCA's ``graph_round_trip_batch1`` — Phase 10
        # would re-enable this with the DOCA backend swap.
        return None

    def forward(
        self,
        input_ids,
        position_ids,
        accepted_tokens,
        num_accepted_tokens,
        batch_size,
        num_contexts,
        request_ids=None,
    ):
        request_ids = self._validate_request_ids(request_ids, batch_size)

        # Step 1: filter retired requests.
        active_rows = [i for i, r in enumerate(request_ids) if r not in self._retired_requests]
        if len(active_rows) != len(request_ids):
            if not active_rows:
                self.last_response_token_counts = torch.zeros(
                    (batch_size,), dtype=torch.int32, device=accepted_tokens.device
                )
                return torch.zeros(
                    (batch_size, self.config.max_draft_len),
                    dtype=torch.int32,
                    device=accepted_tokens.device,
                )
            # Recurse on the active subset and scatter back.
            active_index = torch.tensor(
                active_rows, dtype=torch.int64, device=accepted_tokens.device
            )
            full_tokens = torch.zeros(
                (batch_size, self.config.max_draft_len),
                dtype=torch.int32,
                device=accepted_tokens.device,
            )
            full_counts = torch.zeros(
                (batch_size,), dtype=torch.int32, device=accepted_tokens.device
            )
            sub_request_ids = [request_ids[i] for i in active_rows]
            sub_positions = position_ids.flatten().index_select(0, active_index)
            sub_accepted = accepted_tokens.index_select(0, active_index)
            sub_num_acc = num_accepted_tokens.flatten().index_select(0, active_index)
            sub_num_ctx = sum(1 for i in active_rows if i < num_contexts)
            sub_result = self.forward(
                input_ids=input_ids,
                position_ids=sub_positions,
                accepted_tokens=sub_accepted,
                num_accepted_tokens=sub_num_acc,
                batch_size=len(active_rows),
                num_contexts=sub_num_ctx,
                request_ids=sub_request_ids,
            )
            full_tokens.index_copy_(0, active_index, sub_result)
            full_counts.index_copy_(0, active_index, self.last_response_token_counts)
            self.last_response_token_counts = full_counts
            return full_tokens

        # Step 2: serialize multi-request when configured.
        if self._serialize_multi_request and len(request_ids) > 1:
            return self._forward_serialized_requests(
                input_ids=input_ids,
                position_ids=position_ids,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_contexts=num_contexts,
                request_ids=request_ids,
            )

        self.bind_requests(request_ids)

        # Pull verify-state values to pinned host once. Downstream code
        # (EOS check, position resolution, message build, optional trace)
        # operates on these CPU tensors so .item()/.tolist() reads no longer
        # round-trip to the GPU.
        pinned_accepted, pinned_num_accepted, pinned_positions = self._prefetch_send_data_to_pinned(
            accepted_tokens, num_accepted_tokens, position_ids, batch_size
        )

        # Step 3: short-circuit Llama-3 terminal tokens (don't bother LPU).
        if len(request_ids) == 1 and int(batch_size) == 1:
            last_token = self._extract_last_accepted_token(
                pinned_accepted, pinned_num_accepted, row=0
            )
            if last_token in LLAMA3_TERMINAL_TOKENS:
                self.unbind_requests([request_ids[0]])
                self.last_response_token_counts = torch.zeros(
                    (1,), dtype=torch.int32, device=accepted_tokens.device
                )
                return torch.zeros(
                    (1, self.config.max_draft_len),
                    dtype=torch.int32,
                    device=accepted_tokens.device,
                )

        # Step 4: graph fast path (disabled in Phase 4).
        if len(request_ids) == 1 and int(batch_size) == 1:
            res = self._run_batch1_graph_fast_path(
                request_ids[0],
                position_ids,
                accepted_tokens,
                num_accepted_tokens,
            )
            if res is not None:
                return res

        # Step 5: ordinary send / recv round-trip.
        request_start_positions = [int(pinned_positions[i]) for i in range(batch_size)]
        self._send_requests(
            request_ids,
            pinned_accepted,
            pinned_num_accepted,
            request_start_positions=request_start_positions,
            # Stage 3 path (cudaipc): if the endpoint accepts a GPU-tensor
            # handoff, pass the underlying device tensors so the kernel
            # can read ``last_token`` directly from GPU memory. Other
            # transports ignore these and use the pinned-CPU views above.
            gpu_accepted_tokens=accepted_tokens,
            gpu_num_accepted_tokens=num_accepted_tokens,
        )
        return self._receive_responses(request_ids, accepted_tokens.device)

    def _forward_serialized_requests(
        self,
        input_ids,
        position_ids,
        accepted_tokens,
        num_accepted_tokens,
        num_contexts,
        request_ids,
    ):
        flat_positions = position_ids.flatten()
        flat_num_accepted = num_accepted_tokens.flatten()
        batch_size = len(request_ids)
        next_draft_tokens = torch.zeros(
            (batch_size, self.config.max_draft_len),
            dtype=torch.int32,
            device=accepted_tokens.device,
        )
        next_draft_token_counts = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=accepted_tokens.device,
        )
        # Generation rows first, then context rows (matches izzy ordering).
        row_order = list(range(int(num_contexts), batch_size)) + list(range(0, int(num_contexts)))
        for row in row_order:
            request_id = request_ids[row]
            row_result = self.forward(
                input_ids=input_ids,
                position_ids=flat_positions[row : row + 1],
                accepted_tokens=accepted_tokens[row : row + 1],
                num_accepted_tokens=flat_num_accepted[row : row + 1],
                batch_size=1,
                num_contexts=1 if row < int(num_contexts) else 0,
                request_ids=[request_id],
            )
            next_draft_tokens[row : row + 1].copy_(row_result)
            if self.last_response_token_counts is not None:
                row_counts = self.last_response_token_counts.reshape(-1)
                if int(row_counts.numel()) > 0:
                    next_draft_token_counts[row : row + 1].copy_(
                        row_counts[:1].to(
                            device=next_draft_token_counts.device,
                            dtype=next_draft_token_counts.dtype,
                        )
                    )
        self.last_response_token_counts = next_draft_token_counts
        return next_draft_tokens
