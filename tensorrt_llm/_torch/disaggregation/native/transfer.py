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

from __future__ import annotations

import os
import queue
import secrets
import struct
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import msgpack
import numpy as np
import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm.bindings
from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import (
    BaseTransferAgent,
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    ReceiverBase,
    RxSessionBase,
    SenderBase,
    SessionArgsBase,
    SessionStatus,
    TxSessionBase,
    WaitResult,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer, perf_log_manager
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import CUASSERT, nvtx_range
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

if TYPE_CHECKING:
    from .bounce import Config

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType

# Number of worker threads for KV transfer queues (default: 1)
KV_TRANSFER_NUM_THREADS = int(os.environ.get("TRTLLM_KV_TRANSFER_NUM_THREADS", "1"))

_ASYNC_CONSENSUS_ENVS = (
    "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS",
    "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS",
)
_NATIVE_PROTOCOL_VERSION = 2
_NATIVE_CAPABILITY_PREFIX = b"TRTLLM_NATIVE_TRANSFER_CAPABILITIES\0"
_CONTROL_RETRY_INTERVAL_S = 0.01
_CONTROL_QUEUE_LIMIT = 65_536
_LIVE_PROTOCOL_STATE_LIMIT = 65_536


@dataclass(frozen=True)
class _NativeProtocolCapabilities:
    version: int = 1
    drain_ack: bool = False


_LOCAL_PROTOCOL_CAPABILITIES = _NativeProtocolCapabilities(
    version=_NATIVE_PROTOCOL_VERSION,
    drain_ack=True,
)
_LEGACY_PROTOCOL_CAPABILITIES = _NativeProtocolCapabilities()


def _encode_protocol_capabilities() -> bytes:
    return _NATIVE_CAPABILITY_PREFIX + msgpack.packb(
        {
            "version": _LOCAL_PROTOCOL_CAPABILITIES.version,
            "drain_ack": _LOCAL_PROTOCOL_CAPABILITIES.drain_ack,
        }
    )


def _decode_protocol_capabilities(
    frame: Optional[bytes],
) -> _NativeProtocolCapabilities:
    if frame is None:
        return _LEGACY_PROTOCOL_CAPABILITIES
    if not frame.startswith(_NATIVE_CAPABILITY_PREFIX):
        raise RuntimeError("invalid native-transfer capability frame")
    values = msgpack.unpackb(frame[len(_NATIVE_CAPABILITY_PREFIX) :], raw=False)
    return _NativeProtocolCapabilities(
        version=int(values.get("version", 1)),
        drain_ack=bool(values.get("drain_ack", False)),
    )


def _requires_drain_ack_protocol() -> bool:
    return any(os.environ.get(name, "0") == "1" for name in _ASYNC_CONSENSUS_ENVS)


def _supports_native_protocol_v2(capabilities: _NativeProtocolCapabilities) -> bool:
    """Return whether a peer implements the exact protocol used by this process.

    ``drain_ack`` alone is not sufficient: a future peer may retain the bit
    while changing the request-incarnation wire contract.  Treat unknown
    versions as incompatible instead of silently mixing lifetime semantics.
    """
    return capabilities.version == _NATIVE_PROTOCOL_VERSION and capabilities.drain_ack


@dataclass
class _ControlSend:
    endpoint: str
    message: list[bytes]
    retry: bool
    on_sent: Optional[Callable[[], None]] = None
    repeat_until: Optional[Callable[[], bool]] = None
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None
    callback_ran: bool = False


class _ControlPlane:
    """Own every control DEALER on one thread and serialize its message stream.

    ZMQ sockets are thread-affine. Callers only enqueue immutable frame lists;
    the owner thread creates, uses, and closes every socket. Durable messages
    retry local send failures until accepted by ZeroMQ, whose reconnect queue
    then owns delivery while this progress thread remains alive.
    """

    def __init__(self, name: str):
        self._name = name
        self._queue: queue.Queue[Optional[_ControlSend]] = queue.Queue(maxsize=_CONTROL_QUEUE_LIMIT)
        self._accepting = True
        self._state_lock = threading.Lock()
        self._owner_ident: Optional[int] = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"trtllm-{name}-control",
            daemon=True,
        )
        self._thread.start()

    @property
    def owner_ident(self) -> Optional[int]:
        return self._owner_ident

    def send(
        self,
        endpoint: Optional[str],
        message: list[bytes],
        *,
        retry: bool = False,
        wait: bool = True,
        on_sent: Optional[Callable[[], None]] = None,
        repeat_until: Optional[Callable[[], bool]] = None,
    ) -> _ControlSend:
        if endpoint is None:
            raise ValueError("control peer endpoint is None; peer may not have registered yet")
        request = _ControlSend(
            endpoint,
            list(message),
            retry,
            on_sent=on_sent,
            repeat_until=repeat_until,
        )
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError(f"{self._name} control plane is shutting down")
            try:
                self._queue.put_nowait(request)
            except queue.Full as error:
                raise RuntimeError(
                    f"{self._name} control queue reached its safety limit "
                    f"({_CONTROL_QUEUE_LIMIT}); rejecting new control work"
                ) from error
        if wait:
            request.done.wait()
            if request.error is not None:
                raise RuntimeError(
                    f"{self._name} control send to {endpoint} failed"
                ) from request.error
        return request

    def flush(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        with self._state_lock:
            if not self._accepting:
                should_join = True
            else:
                self._accepting = False
                self._queue.put(None)
                should_join = True
        if should_join and threading.current_thread() is not self._thread:
            self._thread.join()

    def _run(self) -> None:
        self._owner_ident = threading.get_ident()
        dealers: dict[str, ZMQMessenger] = {}
        pending: list[tuple[float, _ControlSend]] = []
        try:
            while True:
                now = time.monotonic()
                ready_index = next(
                    (index for index, (deadline, _) in enumerate(pending) if deadline <= now),
                    None,
                )
                if ready_index is not None:
                    _, request = pending.pop(ready_index)
                else:
                    timeout = None
                    if pending:
                        timeout = max(0.0, min(deadline for deadline, _ in pending) - now)
                    try:
                        request = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        continue
                if request is None:
                    self._queue.task_done()
                    break
                try:
                    if request.repeat_until is not None and request.repeat_until():
                        if not request.done.is_set():
                            request.done.set()
                        self._queue.task_done()
                        continue
                    dealer = dealers.get(request.endpoint)
                    if dealer is None:
                        dealer = ZMQMessenger(mode="DEALER", endpoint=request.endpoint)
                        dealers[request.endpoint] = dealer
                    dealer.send(request.message)
                    if request.on_sent is not None and not request.callback_ran:
                        request.on_sent()
                        request.callback_ran = True
                    if not request.done.is_set():
                        request.done.set()
                    if request.repeat_until is not None and not request.repeat_until():
                        pending.append((time.monotonic() + _CONTROL_RETRY_INTERVAL_S, request))
                    else:
                        self._queue.task_done()
                except Exception as error:
                    dealer = dealers.pop(request.endpoint, None)
                    if dealer is not None:
                        try:
                            dealer.stop()
                        except Exception:
                            pass
                    if request.retry:
                        logger.warning(
                            "%s control send to %s failed; retrying without "
                            "blocking unrelated control work: %s",
                            self._name,
                            request.endpoint,
                            error,
                        )
                        pending.append((time.monotonic() + _CONTROL_RETRY_INTERVAL_S, request))
                    else:
                        request.error = error
                        request.done.set()
                        self._queue.task_done()
        finally:
            for dealer in dealers.values():
                try:
                    dealer.stop()
                except Exception as error:
                    logger.warning("%s control DEALER shutdown failed: %s", self._name, error)


@dataclass
class RecvReqInfo:
    sender_req_id: int
    instance_name: str
    instance_rank: int
    block_ids_per_layer_groups: list[
        np.ndarray
    ]  # Block IDs per layer group, each np.ndarray(dtype=np.int64)
    unique_rid: int
    # Block-aligned token offset where the receiver's block list starts.
    # None means "end-of-range suffix" — sender derives it from len(blocks).
    dst_start_token: Optional[int] = None
    aux_slot: Optional[int] = None
    mamba_state_index: Optional[int] = None
    slice_id: Optional[int] = None
    bounce_dst_base: Optional[int] = None
    request_epoch: Optional[int] = None

    def to_bytes(self) -> bytes:
        values = {
            "sender_req_id": self.sender_req_id,
            "instance_name": self.instance_name,
            "instance_rank": self.instance_rank,
            "block_ids_per_layer_groups": [
                arr.tobytes() for arr in self.block_ids_per_layer_groups
            ],
            "unique_rid": self.unique_rid,
            "dst_start_token": self.dst_start_token,
            "aux_slot": self.aux_slot,
            "mamba_state_index": self.mamba_state_index,
            "slice_id": self.slice_id,
            "bounce_dst_base": self.bounce_dst_base,
        }
        # Keep legacy peers byte-compatible. An epoch is emitted only after
        # capability negotiation proves that every writer understands v2.
        if self.request_epoch is not None:
            values["request_epoch"] = self.request_epoch
        return msgpack.packb(values)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RecvReqInfo":
        d = msgpack.unpackb(data, raw=False)
        d["block_ids_per_layer_groups"] = [
            np.frombuffer(b, dtype=np.int64).copy() for b in d["block_ids_per_layer_groups"]
        ]
        return cls(**d)


@dataclass
class ReadMeta:
    unique_rid: int
    slice_id: int
    target_ranks: Optional[List[int]] = None


class WriteMetaType(Enum):
    KV = "KV"
    AUX = "AUX"


@dataclass
class WriteMeta:
    task: Union[SendTaskBase, "KVRecvTask"]
    expected_transfers: int
    peer_name: str
    peer_rank: int
    peer_endpoint: str
    unique_rid: int
    src_ptrs: np.ndarray  # dtype=np.int64
    dst_ptrs: np.ndarray  # dtype=np.int64
    sizes: np.ndarray  # dtype=np.int64
    dst_device_id: Optional[int] = None
    slice_id: Optional[int] = None
    is_last_slice: bool = False
    meta_type: WriteMetaType = WriteMetaType.KV
    bounce_dst_base: Optional[int] = None
    request_epoch: Optional[int] = None
    # Strongly retain the session until this queued response obligation has
    # sent a terminal result to the peer and retired its lifetime credit.
    operation_owner: Optional["TxSession"] = None
    terminal_sent: bool = False
    terminal_queued: bool = False
    operation_retired: bool = False
    lifetime_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class MessageType:
    TERMINATION = b"TERMINATION"
    KV_AGENT_RESULT = b"KV_AGENT_RESULT"
    REQUEST_DATA = b"REQUEST_DATA"
    REQUEST_INSTANCE_INFO = b"REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = b"REGISTER_RANK_INFO"
    AUX_AGENT_RESULT = b"AUX_AGENT_RESULT"
    CANCEL_SESSION = b"CANCEL_SESSION"
    CANCEL_SESSION_ACK = b"CANCEL_SESSION_ACK"


class TaskStatus(Enum):
    INIT = "INIT"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ERROR = "ERROR"


class AgentResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


# KV_AGENT_RESULT prefix in one struct frame (was 5 ascii frames serialized/parsed under the
# GIL per slice per writer): instance_rank, unique_rid, slice_id, is_last, status. The optional
# bounce tail follows at message[2:].
_KV_RESULT_PREFIX = struct.Struct("<qqq?B")
_KV_RESULT_PREFIX_V2 = struct.Struct("<qqqq?B")
_AGENT_RESULT_CODE = {AgentResult.SUCCESS: 0, AgentResult.FAILED: 1}
_AGENT_RESULT_BY_CODE = {0: AgentResult.SUCCESS, 1: AgentResult.FAILED}


def _make_kv_result_msg(
    instance_rank,
    unique_rid,
    slice_id,
    is_last_slice,
    agent_result,
    tail=None,
    *,
    request_epoch: Optional[int] = None,
    sender_endpoint: Optional[str] = None,
):
    """Build a KV_AGENT_RESULT message. ALL result sends (success AND failed/cancelled) must go
    through this single binary frame so the receiver's _KV_RESULT_PREFIX.unpack never hits a stale
    ascii payload (which would fail to decode and leave the RX task stuck forever)."""
    if request_epoch is None:
        prefix = _KV_RESULT_PREFIX.pack(
            int(instance_rank),
            int(unique_rid),
            int(slice_id),
            bool(is_last_slice),
            _AGENT_RESULT_CODE[agent_result],
        )
        msg = [MessageType.KV_AGENT_RESULT, prefix]
    else:
        if sender_endpoint is None:
            raise ValueError("v2 KV result requires the sender endpoint")
        prefix = _KV_RESULT_PREFIX_V2.pack(
            int(instance_rank),
            int(unique_rid),
            int(request_epoch),
            int(slice_id),
            bool(is_last_slice),
            _AGENT_RESULT_CODE[agent_result],
        )
        msg = [MessageType.KV_AGENT_RESULT, prefix, sender_endpoint.encode("utf-8")]
    if tail:
        msg += tail
    return msg


class _AmbiguousTransferError(RuntimeError):
    """The transfer agent may still own an operation whose terminal state is unknown."""


@dataclass(frozen=True)
class _ExpectedReceiveOperation:
    sender_endpoint: str
    request_epoch: Optional[int]


class SendTaskBase:
    def __init__(self, params: DisaggregatedParams):
        self.status = TaskStatus.INIT
        self._event = threading.Event()
        self._exception: Optional[Exception] = None
        self.lock = threading.Lock()
        self._params = params
        self._unique_rid: Optional[int] = params.disagg_request_id
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

    def fail(self, exc: Exception) -> None:
        self._exception = exc
        self.status = TaskStatus.ERROR
        self._event.set()

    def complete(self) -> None:
        self.status = TaskStatus.TRANSFERRED
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until terminal state. Returns True if done, False on timeout."""
        return self._event.wait(timeout=timeout)

    @property
    def is_done(self) -> bool:
        return self._event.is_set()

    def print_perf_info(self, peer_rank: int, instance_name: str, instance_rank: int):
        if self._perf_timer is None:
            return
        perf_log_manager.log_task_perf(
            type(self).__name__,
            self._unique_rid,
            peer_rank,
            instance_name,
            instance_rank,
            self._perf_timer,
        )


class AuxSendTask(SendTaskBase):
    def __init__(self, params: DisaggregatedParams, slot: Optional[int]):
        super().__init__(params)
        self._slot = slot
        self._transfer_count = 0


class KVSendTask(SendTaskBase):
    def __init__(
        self,
        kv_slice: KVSlice,
        params: DisaggregatedParams,
        slice_id: int,
        prompt_len: Optional[int] = None,
        beam_width: int = 1,
    ):
        super().__init__(params)
        self.slice_id = slice_id
        self.transferred_count = 0
        self._slice = kv_slice
        self._prompt_len = prompt_len
        self._beam_width = beam_width


class Sender(SenderBase):
    # Time-to-live for orphaned RecvReqInfo entries (seconds).
    # In gen-first ADP broadcast, non-assigned DP ranks accumulate
    # RecvReqInfo that never gets consumed.  Entries older than this
    # are evicted during periodic sweeps.
    _STALE_REQ_INFO_TTL_S = 120.0
    _TOMBSTONE_LIMIT = 65_536

    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        agent: BaseTransferAgent,
        bounce=None,
    ):
        self._registrar = peer_registrar
        self._device_id = peer_registrar.self_rank_info.device_id
        self._agent = agent
        self._bounce = bounce
        self._peer_requests: dict = {}
        self._peer_requests_timestamps: dict[int, float] = {}  # unique_rid -> insert time
        self._peer_requests_lock = threading.Lock()
        self._messenger = ZMQMessenger(mode="ROUTER")
        self._control = _ControlPlane("native-sender")
        self._thread_local = threading.local()  # per-thread DEALER cache for worker threads
        self._sessions = {}  # unique_rid -> TxSession
        self._sessions_lock = threading.Lock()
        # An acknowledged pre-cancel is live protocol state, not bounded
        # history: evicting it can let a delayed session write into memory the
        # receiver reclaimed after our ACK.
        self._pre_cancelled_rids: dict[int, None] = {}
        self._pre_cancelled_operations: dict[tuple[int, str, int], None] = {}
        self._cancelled_operation_tombstones: OrderedDict[tuple[int, str, int], None] = (
            OrderedDict()
        )
        self._closed_rids: OrderedDict[int, None] = OrderedDict()
        self._peer_capabilities: dict[str, _NativeProtocolCapabilities] = {}
        self._shutdown = False
        self._ingress_lock = threading.Lock()
        self._instance_rank = self._registrar.self_rank_info.instance_rank
        # Guards concurrent add() from the listener thread.
        self._loaded_remote_agents: set[str] = set()
        self._loaded_remote_agents_lock = threading.Lock()
        self._num_threads = KV_TRANSFER_NUM_THREADS
        self._send_task_queues: List[queue.Queue] = [
            queue.Queue(maxsize=_CONTROL_QUEUE_LIMIT) for _ in range(self._num_threads)
        ]
        self._stalled_operations: list[WriteMeta] = []
        self._stalled_session_owners: list["TxSession"] = []
        self._ambiguous_operations: list[WriteMeta] = []
        self._stalled_operations_lock = threading.Lock()
        self._protocol_error: Optional[RuntimeError] = None
        self._worker_threads: List[threading.Thread] = [
            threading.Thread(target=self._process_task_queue, args=(i,), daemon=True)
            for i in range(self._num_threads)
        ]

        self._start_listener()
        for t in self._worker_threads:
            t.start()
        logger.info(
            f"Sender init end with endpoint: {self._messenger.endpoint},"
            f" {self._num_threads} worker thread(s)"
        )

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def _add_req_info(self, unique_rid: int, instance_rank: int, req_info: RecvReqInfo):
        with self._peer_requests_lock:
            if unique_rid not in self._peer_requests:
                self._peer_requests[unique_rid] = {}
                self._peer_requests_timestamps[unique_rid] = time.monotonic()
            existing = self._peer_requests[unique_rid].get(instance_rank)
            if existing is not None and existing.request_epoch != req_info.request_epoch:
                error = RuntimeError(
                    "native-transfer request incarnation changed for an active "
                    f"operation: rid={unique_rid} receiver_rank={instance_rank} "
                    f"old_epoch={existing.request_epoch} new_epoch={req_info.request_epoch}"
                )
                self._protocol_error = error
                raise error
            self._peer_requests[unique_rid][instance_rank] = req_info

    def _is_req_ready(self, unique_rid: int, expected_count: int) -> bool:
        with self._peer_requests_lock:
            requests = self._peer_requests.get(unique_rid)
            if not requests:
                return False
            return len(requests) == expected_count

    def _get_req_info(self, unique_rid: Optional[int]) -> Optional[dict]:
        with self._peer_requests_lock:
            return self._peer_requests.get(unique_rid)

    def _get_first_req_info(self, unique_rid: Optional[int]) -> Optional[RecvReqInfo]:
        with self._peer_requests_lock:
            reqs = self._peer_requests.get(unique_rid)
            if not reqs:
                return None
            return next(iter(reqs.values()))

    def _remove_req_info(self, unique_rid: int):
        with self._peer_requests_lock:
            self._peer_requests.pop(unique_rid, None)
            self._peer_requests_timestamps.pop(unique_rid, None)

    def sweep_stale_req_infos(self):
        """Evict RecvReqInfo entries that have no matching TxSession and exceed the TTL.

        Called opportunistically from the listener thread when a new REQUEST_DATA
        arrives. With gen-first ADP broadcast, non-assigned DP ranks accumulate
        entries that are never consumed; this sweep prevents unbounded growth.
        """
        now = time.monotonic()
        with self._peer_requests_lock:
            stale_rids = [
                rid
                for rid, ts in self._peer_requests_timestamps.items()
                if now - ts > self._STALE_REQ_INFO_TTL_S
            ]
        if not stale_rids:
            return
        for rid in stale_rids:
            with self._sessions_lock, self._peer_requests_lock:
                if rid not in self._sessions and rid in self._peer_requests:
                    self._peer_requests.pop(rid, None)
                    self._peer_requests_timestamps.pop(rid, None)
                    logger.debug(f"Swept stale RecvReqInfo for rid={rid}")

    def setup_session(self, tx_session: "TxSession"):
        unique_rid = tx_session.disagg_request_id
        pre_cancel = False
        with self._ingress_lock:
            if self._shutdown:
                raise RuntimeError("Cannot create a TxSession after Sender shutdown started")
            if self._protocol_error is not None:
                raise RuntimeError(
                    "Sender is in fail-stop protocol state"
                ) from self._protocol_error
            with self._sessions_lock:
                self._sessions[unique_rid] = weakref.ref(tx_session)
                if unique_rid in self._pre_cancelled_rids:
                    pre_cancel = True
                    self._pre_cancelled_rids.pop(unique_rid, None)
                self._closed_rids.pop(unique_rid, None)
        if pre_cancel:
            tx_session.cancel()
            return

        cancelled_operations: list[tuple[str, int]] = []
        with self._ingress_lock, self._sessions_lock:
            req_info_map = dict(self._get_req_info(unique_rid) or {})
            for req_info_item in req_info_map.values():
                peer_ri_item = self._registrar.get_peer_rank_info(
                    req_info_item.instance_name, req_info_item.instance_rank
                )
                tx_session.register_request_operation(
                    peer_ri_item.self_endpoint, req_info_item.request_epoch
                )
                if req_info_item.request_epoch is not None:
                    operation = (
                        unique_rid,
                        peer_ri_item.self_endpoint,
                        req_info_item.request_epoch,
                    )
                    if operation in self._pre_cancelled_operations:
                        self._pre_cancelled_operations.pop(operation, None)
                        self._remember_cancelled_operation_unlocked(operation)
                        cancelled_operations.append(
                            (peer_ri_item.self_endpoint, req_info_item.request_epoch)
                        )

        if cancelled_operations:
            # A request can fan out to multiple receiver endpoints. Preserve
            # every exact pre-cancel obligation; acknowledging only the last
            # endpoint would strand the other receiver's drain wait forever.
            for endpoint, request_epoch in cancelled_operations:
                tx_session.cancel(
                    ack_endpoint=endpoint,
                    request_epoch=request_epoch,
                )
            return

        req_info = next(iter(req_info_map.values()), None)

        if req_info:
            peer_ri = self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            )
            expected_count = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
            if self._is_req_ready(unique_rid, expected_count):
                with tx_session.lock:
                    tx_session.receiver_ready = True
        return

    @classmethod
    def _remember_tombstone(cls, tombstones: OrderedDict, key) -> None:
        tombstones.pop(key, None)
        tombstones[key] = None
        while len(tombstones) > cls._TOMBSTONE_LIMIT:
            tombstones.popitem(last=False)

    def _remember_pre_cancelled_unlocked(self, unique_rid: int) -> None:
        if (
            unique_rid not in self._pre_cancelled_rids
            and len(self._pre_cancelled_rids) >= _LIVE_PROTOCOL_STATE_LIMIT
        ):
            self._protocol_error = RuntimeError(
                "Sender pre-cancel state reached its safety limit; refusing to evict live state"
            )
            raise self._protocol_error
        self._pre_cancelled_rids[unique_rid] = None

    def _remember_pre_cancelled_operation_unlocked(
        self, unique_rid: int, receiver_endpoint: str, request_epoch: int
    ) -> None:
        key = (unique_rid, receiver_endpoint, request_epoch)
        if (
            key not in self._pre_cancelled_operations
            and len(self._pre_cancelled_operations) >= _LIVE_PROTOCOL_STATE_LIMIT
        ):
            self._protocol_error = RuntimeError(
                "Sender v2 pre-cancel state reached its safety limit; "
                "refusing to evict live request incarnations"
            )
            raise self._protocol_error
        self._pre_cancelled_operations[key] = None

    def _remember_closed_unlocked(self, unique_rid: int) -> None:
        self._remember_tombstone(self._closed_rids, unique_rid)

    def _remember_cancelled_operation_unlocked(self, operation: tuple[int, str, int]) -> None:
        self._remember_tombstone(self._cancelled_operation_tombstones, operation)

    def _get_session(self, unique_rid: Optional[int]) -> Optional["TxSession"]:
        session_ref = self._sessions.get(unique_rid)
        if session_ref is None:
            return None
        session = session_ref()
        if session is None:
            logger.warning(f"TxSession {unique_rid} has been garbage collected")
            return None
        return session

    def _enqueue(self, write_meta: WriteMeta):
        # Route by (unique_rid, peer_rank) so that:
        # - Same peer's slices stay ordered on one thread (is_last_slice correctness)
        # - Different peers can run on different threads (better load balancing)
        thread_idx = hash((write_meta.unique_rid, write_meta.peer_rank)) % self._num_threads
        with self._ingress_lock:
            if self._shutdown:
                raise RuntimeError("Sender is shutting down; transfer enqueue rejected")
            try:
                self._send_task_queues[thread_idx].put_nowait(write_meta)
            except queue.Full as error:
                raise RuntimeError(
                    "native transfer work queue reached its safety limit; "
                    "rejecting new transfer work"
                ) from error

    def _get_or_connect_thread_dealer(self, endpoint: Optional[str]) -> ZMQMessenger:
        """Get or create a per-thread DEALER socket via threading.local().
        Each worker thread gets its own cache so there is no cross-thread
        access to the same ZMQ socket."""
        if endpoint is None:
            raise ValueError("Sender: peer endpoint is None; peer may not have registered yet")
        dealers = getattr(self._thread_local, "dealers", None)
        if dealers is None:
            dealers = {}
            self._thread_local.dealers = dealers
        if endpoint not in dealers:
            dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return dealers[endpoint]

    def _send_write_result(
        self,
        write_meta: WriteMeta,
        result: AgentResult,
        *,
        tail: Optional[list] = None,
    ) -> None:
        """Publish exactly one terminal response from the owning worker thread."""
        if write_meta.terminal_sent or write_meta.terminal_queued:
            return
        if write_meta.meta_type == WriteMetaType.KV:
            if write_meta.slice_id is None:
                raise RuntimeError("KV WriteMeta is missing slice_id")
            message = _make_kv_result_msg(
                self._instance_rank,
                write_meta.unique_rid,
                write_meta.slice_id,
                write_meta.is_last_slice if result == AgentResult.SUCCESS else True,
                result,
                tail=tail,
                request_epoch=write_meta.request_epoch,
                sender_endpoint=self.endpoint,
            )
        else:
            message = [
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                result.value.encode("ascii"),
            ]
            if write_meta.request_epoch is not None:
                message.extend(
                    [
                        self.endpoint.encode("utf-8"),
                        str(write_meta.request_epoch).encode("ascii"),
                    ]
                )
        try:
            self._get_or_connect_thread_dealer(write_meta.peer_endpoint).send(message)
            write_meta.terminal_sent = True
        except Exception as error:
            # The RMA is terminal but the receiver still owns one lifetime
            # credit. Move only this rare error path to the durable control
            # owner; its callback holds the session strongly and retires the
            # credit only after ZeroMQ accepts the terminal notification.
            logger.warning(
                "Terminal result send failed for request %s peer_rank=%s; "
                "moving it to the durable control queue: %s",
                write_meta.unique_rid,
                write_meta.peer_rank,
                error,
            )
            write_meta.terminal_queued = True

            def terminal_sent() -> None:
                write_meta.terminal_sent = True
                self._retire_write_meta_operation(write_meta)
                with self._stalled_operations_lock:
                    if write_meta in self._stalled_operations:
                        self._stalled_operations.remove(write_meta)

            try:
                self._control.send(
                    write_meta.peer_endpoint,
                    message,
                    retry=True,
                    wait=False,
                    on_sent=terminal_sent,
                )
            except Exception:
                write_meta.terminal_queued = False
                raise

    def _fail_write_meta(self, write_meta: WriteMeta, error: Exception) -> None:
        write_meta.task.fail(error)
        owner = write_meta.operation_owner
        if owner is not None:
            owner.set_exception(str(error))
        self._send_write_result(write_meta, AgentResult.FAILED)

    @staticmethod
    def _retire_write_meta_operation(write_meta: WriteMeta) -> None:
        owner = write_meta.operation_owner
        if owner is None:
            return
        with write_meta.lifetime_lock:
            if write_meta.operation_retired:
                return
            owner.retire_operation()
            write_meta.operation_retired = True

    def _process_task_queue(self, thread_idx: int):
        device_id = self._device_id
        torch.cuda.set_device(device_id)
        CUASSERT(cudart.cudaSetDevice(device_id))

        task_queue = self._send_task_queues[thread_idx]
        try:
            while True:
                write_meta = task_queue.get()
                if write_meta is None:
                    break
                try:
                    if write_meta.meta_type == WriteMetaType.AUX:
                        logger.debug(
                            f"_process_task_queue[{thread_idx}]: delivering aux task to agent: {write_meta}"
                        )
                        self._deliver_aux_to_agent(write_meta)
                    else:
                        self._deliver_kv_to_agent(write_meta)
                except _AmbiguousTransferError as error:
                    # submit()/wait() raised after the agent may have accepted
                    # the RMA. Publishing FAILED or retiring the lifetime
                    # credit would falsely claim drain. Retain everything and
                    # force teardown to stop before memory invalidation.
                    logger.critical(
                        "Ambiguous transfer state for request %s: %s",
                        write_meta.unique_rid,
                        error,
                    )
                    write_meta.task.fail(error)
                    owner = write_meta.operation_owner
                    if owner is not None:
                        owner.set_exception(str(error))
                    with self._stalled_operations_lock:
                        if write_meta not in self._ambiguous_operations:
                            self._ambiguous_operations.append(write_meta)
                    continue
                except Exception as e:
                    logger.error(
                        f"_process_task_queue[{thread_idx}]: unhandled exception for "
                        f"unique_rid={write_meta.unique_rid}: {e}"
                    )
                    try:
                        self._fail_write_meta(write_meta, e)
                    except Exception as notify_error:
                        logger.error(
                            "Unable to publish terminal failure for request %s: %s",
                            write_meta.unique_rid,
                            notify_error,
                        )
                finally:
                    owner = write_meta.operation_owner
                    if owner is not None and write_meta.terminal_sent:
                        try:
                            self._retire_write_meta_operation(write_meta)
                        except Exception as retire_error:
                            logger.error(
                                "Unable to retire terminal operation for request %s: %s",
                                write_meta.unique_rid,
                                retire_error,
                            )
                            with self._stalled_operations_lock:
                                if write_meta not in self._stalled_operations:
                                    self._stalled_operations.append(write_meta)
                    elif owner is not None:
                        # Serialize with the durable callback's removal. It may
                        # have published the terminal result between the first
                        # flag check above and this lock acquisition.
                        with self._stalled_operations_lock:
                            if (
                                not write_meta.terminal_sent
                                and write_meta not in self._stalled_operations
                            ):
                                # No terminal notification has retired this
                                # credit. Retain it strongly and fail closed.
                                self._stalled_operations.append(write_meta)
        finally:
            # Clean up this thread's DEALER sockets. threading.local storage
            # is only accessible from the owning thread, so shutdown must
            # happen here rather than in Sender.shutdown().
            dealers = getattr(self._thread_local, "dealers", None)
            if dealers:
                for endpoint, dealer in dealers.items():
                    try:
                        dealer.stop()
                    except Exception as e:
                        logger.warning(
                            f"_process_task_queue[{thread_idx}]: failed to stop dealer "
                            f"for endpoint {endpoint}: {e}"
                        )
                dealers.clear()

    @staticmethod
    @nvtx_range("_make_agent_request")
    def _make_agent_request(write_meta: WriteMeta, device_id: int) -> "TransferRequest":
        if not (write_meta.src_ptrs.size == write_meta.dst_ptrs.size == write_meta.sizes.size):
            raise ValueError(
                f"Pointer/size mismatch for unique_rid={write_meta.unique_rid}: "
                f"{write_meta.src_ptrs.size=}, "
                f"{write_meta.dst_ptrs.size=}, "
                f"{write_meta.sizes.size=}"
            )
        n = write_meta.src_ptrs.size
        if write_meta.meta_type == WriteMetaType.AUX:
            src_dev, dst_dev, mem_type = 0, 0, MemoryType.DRAM
        else:
            if write_meta.dst_device_id is None:
                raise RuntimeError(
                    f"_make_agent_request: dst_device_id is None for KV transfer "
                    f"unique_rid={write_meta.unique_rid}"
                )
            src_dev, dst_dev, mem_type = device_id, write_meta.dst_device_id, MemoryType.VRAM

        if n == 0:
            src_memory_descs = MemoryDescs(mem_type, [])
            dst_memory_descs = MemoryDescs(mem_type, [])
        else:
            src_memory_descs = MemoryDescs.from_arrays_uniform_device(
                mem_type, write_meta.src_ptrs, write_meta.sizes, src_dev
            )
            dst_memory_descs = MemoryDescs.from_arrays_uniform_device(
                mem_type, write_meta.dst_ptrs, write_meta.sizes, dst_dev
            )

        # NOTE: TransferRequest moves (not copies) src/dst MemoryDescs internally.
        # After this call, src_memory_descs and dst_memory_descs are in a moved-from
        # state and must NOT be accessed again.
        return TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, write_meta.peer_name, None
        )

    def _submit_and_wait(self, request, write_meta: WriteMeta) -> tuple[bool, object]:
        """Return a known terminal result or raise without claiming quiescence.

        Either submit or wait may throw after the agent accepted the request.
        Such an exception is ambiguous by contract: no terminal peer result is
        sent and registered memory remains retained through fail-stop teardown.
        """
        try:
            status = self._agent.submit_transfer_requests(request)
        except Exception as error:
            raise _AmbiguousTransferError(
                f"transfer submission may have been accepted for request "
                f"{write_meta.unique_rid} peer_rank={write_meta.peer_rank}"
            ) from error
        try:
            return bool(status.wait()), status
        except Exception as error:
            raise _AmbiguousTransferError(
                f"transfer completion is unknown for request {write_meta.unique_rid} "
                f"peer_rank={write_meta.peer_rank}"
            ) from error

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, write_meta: WriteMeta):
        assert write_meta.src_ptrs.size == write_meta.dst_ptrs.size == write_meta.sizes.size, (
            f"WriteMeta ptr/size mismatch for unique_rid={write_meta.unique_rid}"
        )

        session = write_meta.operation_owner
        if session is None:
            with self._sessions_lock:
                session = self._get_session(write_meta.unique_rid)
        if session is None:
            msg = (
                f"_deliver_kv_to_agent: TxSession {write_meta.unique_rid} not found or already GC'd"
            )
            logger.error(msg)
            write_meta.task.fail(RuntimeError(msg))
            self._send_write_result(write_meta, AgentResult.FAILED)
            return
        assert write_meta.slice_id is not None
        task = session.kv_tasks[write_meta.slice_id]
        timer = task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)
        # The session owns the INIT-to-TRANSFERRING transition. Its permanent
        # seal prevents a queued task from starting after a terminal vote.
        if not session.try_mark_transferring(task):
            status = session.status
            logger.warning(
                f"_deliver_kv_to_agent: session {write_meta.unique_rid} already "
                f"sealed or in {status.value} state; sending FAILED to receiver"
            )
            # Task may have been enqueued after cancel() already iterated kv_tasks,
            # so its future was never set by cancel(). Set it here as a fallback.
            if not task.is_done:
                task.fail(
                    RuntimeError(
                        f"session {write_meta.unique_rid} sealed or {status.value}, "
                        "transfer aborted"
                    )
                )
            self._send_write_result(write_meta, AgentResult.FAILED)
            return

        from .bounce import build_send_request, encode_result_tail

        agent_result = AgentResult.SUCCESS
        send_slot_id = None
        if write_meta.src_ptrs.size > 0:
            try:
                request, send_slot_id = build_send_request(
                    self._bounce,
                    write_meta,
                    lambda: Sender._make_agent_request(write_meta, device_id=self._device_id),
                )
            except Exception as e:
                # Don't let a gather fault escape: without a result the receiver would hang and its
                # region leak. Tell the receiver it failed and fail the local task instead.
                logger.error(
                    f"_deliver_kv_to_agent: failed to build the KV send request for "
                    f"{write_meta.unique_rid} slice={write_meta.slice_id}: {e}"
                )
                task.fail(RuntimeError(f"build_send_request failed: {e}"))
                self._send_write_result(write_meta, AgentResult.FAILED)
                return
            if timer:
                timer.record_transfer_start(write_meta.peer_rank)
            terminal_known = False
            try:
                completed, status = self._submit_and_wait(request, write_meta)
                terminal_known = True
                if not completed:
                    agent_result = AgentResult.FAILED
                    last_status = getattr(status, "last_status_str", lambda: "<no detail>")()
                    agent_name = getattr(self._agent, "name", "<?>")
                    detail = (
                        f"KV transfer agent failed: "
                        f"unique_rid={write_meta.unique_rid} "
                        f"slice={write_meta.slice_id} "
                        f"peer_rank={write_meta.peer_rank} "
                        f"peer_endpoint={write_meta.peer_endpoint} "
                        f"op={getattr(request, 'op', '?')} "
                        f"remote={getattr(request, 'remote_name', '?')} "
                        f"src_size={int(write_meta.src_ptrs.size)} "
                        f"dst_size={int(write_meta.dst_ptrs.size)} "
                        f"nixl_status={last_status} agent={agent_name}"
                    )
                    logger.error(detail)
                    task.fail(RuntimeError(detail))
            finally:
                if send_slot_id is not None and terminal_known:
                    self._bounce.release_send(send_slot_id)
        if timer:
            timer.record_transfer_end(write_meta.peer_rank)

        ## TODO: just last slice need to send task state?
        tail = (
            encode_result_tail(write_meta)
            if send_slot_id is not None and agent_result == AgentResult.SUCCESS
            else None
        )
        self._send_write_result(write_meta, agent_result, tail=tail)

        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)

        with task.lock:
            task.transferred_count += 1
            count = task.transferred_count

        if count > write_meta.expected_transfers:
            task.fail(
                RuntimeError(
                    f"KV slice {write_meta.slice_id} received more than "
                    f"{write_meta.expected_transfers} transfers"
                )
            )
            session.set_exception(
                f"KV slice {write_meta.slice_id} received more than {write_meta.expected_transfers} transfers"
            )
        elif count == write_meta.expected_transfers:
            if task.is_done:
                task.status = TaskStatus.ERROR
                session.set_exception(
                    f"KV slice {write_meta.slice_id} task already resolved on completion"
                )
            else:
                task.complete()

        logger.debug(
            f"deliver_kv_to_agent completed: unique_rid={write_meta.unique_rid}, "
            f"slice_id={write_meta.slice_id}, agent_result={agent_result}"
        )

    @nvtx_range("_deliver_aux_to_agent")
    def _deliver_aux_to_agent(self, write_meta: WriteMeta):
        session = write_meta.operation_owner or self._get_session(write_meta.unique_rid)
        if session is None:
            msg = f"_deliver_aux_to_agent: TxSession {write_meta.unique_rid} not found or already GC'd"
            logger.error(msg)
            write_meta.task.fail(RuntimeError(msg))
            self._send_write_result(write_meta, AgentResult.FAILED)
            return
        aux_task = session.aux_task
        assert aux_task is not None, f"aux_task is None for session {write_meta.unique_rid}"
        timer = aux_task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)

        if not session.try_mark_transferring(aux_task):
            status = session.status
            logger.warning(
                f"_deliver_aux_to_agent: session {write_meta.unique_rid} already "
                f"sealed or in {status.value} state; sending FAILED to receiver"
            )
            if not aux_task.is_done:
                aux_task.fail(
                    RuntimeError(
                        f"session {write_meta.unique_rid} sealed or {status.value}, "
                        "aux transfer aborted"
                    )
                )
            self._send_write_result(write_meta, AgentResult.FAILED)
            return

        agent_result = AgentResult.SUCCESS
        if write_meta.src_ptrs.size > 0:
            request = Sender._make_agent_request(write_meta, device_id=self._device_id)
            if timer:
                timer.record_transfer_start(write_meta.peer_rank)
            completed, _status = self._submit_and_wait(request, write_meta)
            if not completed:
                agent_result = AgentResult.FAILED
                aux_task.fail(
                    RuntimeError(f"aux transfer agent request failed for {write_meta.unique_rid}")
                )
                session.set_exception("aux transfer agent request failed")
            if timer:
                timer.record_transfer_end(write_meta.peer_rank)

        self._send_write_result(write_meta, agent_result)

        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        aux_task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)

        with aux_task.lock:
            aux_task._transfer_count += 1
            count = aux_task._transfer_count

        if count == write_meta.expected_transfers:
            if aux_task.is_done:
                aux_task.status = TaskStatus.ERROR
                session.set_exception("aux task already resolved on completion")
            else:
                aux_task.complete()
        elif count > write_meta.expected_transfers:
            aux_task.fail(
                RuntimeError(
                    f"aux task received more than {write_meta.expected_transfers} transfers"
                )
            )
            session.set_exception(
                f"aux task received more than {write_meta.expected_transfers} transfers"
            )

    @staticmethod
    def _align_kv_blocks(
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
        src_token_start: int,
        dst_token_start: int,
        tokens_per_block: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align src/dst block arrays using explicit token-start positions.

        Both src_token_start and dst_token_start must be block-aligned
        (multiples of tokens_per_block), which is always true for prefix-cache
        boundaries in current KV cache managers.

        Returns the (src, dst) sub-arrays that cover the shared token overlap.
        Returns a pair of empty arrays when there is no overlap (i.e. this
        context slice is entirely within generation's already-cached prefix).

        This handles four cases without special-casing:
          1. No prefix cache on either side  → identity (start_token == 0 both)
          2. Context prefix cache (src starts later than 0)  → trim dst head
          3. Generation prefix cache (dst starts later than 0)  → trim src head
          4. Chunked context (each slice has its own token_range)  → correct
             overlap even when the slice is entirely before dst_token_start
        """
        overlap_start = max(src_token_start, dst_token_start)
        src_skip = (overlap_start - src_token_start) // tokens_per_block
        dst_skip = (overlap_start - dst_token_start) // tokens_per_block
        n_transfer = min(src_block_ids.size - src_skip, dst_block_ids.size - dst_skip)
        if n_transfer <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return (
            src_block_ids[src_skip : src_skip + n_transfer],
            dst_block_ids[dst_skip : dst_skip + n_transfer],
        )

    @staticmethod
    def _beam0_block_count(block_ids: np.ndarray, total_blocks: int, beam_width: int) -> int:
        """Return the number of beam-0 blocks in a packed 1-D beam layout."""
        if beam_width <= 1 or block_ids.size <= total_blocks:
            return block_ids.size
        return max(0, block_ids.size - (beam_width - 1))

    @nvtx_range("_build_kv_write_meta")
    def _build_kv_write_meta(self, task: KVSendTask, req_info: RecvReqInfo) -> WriteMeta:
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        timer = task._perf_timer
        if timer:
            timer.record_prepare_args_start(peer_ri.instance_rank)
        targets = self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank)
        expected_transfers = len(targets.ranks)

        # Aggregate fragment pointers from all matching pool pairs.
        # Each pool pair produces one or more region pairs (rp), each containing
        # a numpy array of src/dst pointers and a uniform bytes_per_region.
        src_frag_parts: list[np.ndarray] = []
        dst_frag_parts: list[np.ndarray] = []
        # Instead of calling np.full() per region pair to build a size array and
        # then np.concatenate() all of them, we record (count, bytes_per_region)
        # tuples and construct the final sizes array with a single np.repeat().
        # For 48k+ items this avoids many small allocations in the hot loop.
        size_specs: list[tuple[int, int]] = []
        dst_device_id = peer_ri.device_id
        extractor = self._registrar.self_extractor
        peer_extractor = self._registrar.peer_extractor(
            peer_ri.instance_name, peer_ri.instance_rank
        )
        if self._registrar.should_send_kv(targets, peer_ri):
            pool_mapping = self._registrar.get_pool_mapping(peer_ri)
            dst_block_ids_per_groups = req_info.block_ids_per_layer_groups
            src_block_ids_per_groups = task._slice.block_ids_per_layer_groups

            # Aggregate fragments from all matching pools using numpy concatenation
            for (self_lg, self_pi), (peer_lg, peer_pi) in pool_mapping.items():
                src_block_ids = src_block_ids_per_groups[self_lg]
                dst_block_ids = dst_block_ids_per_groups[peer_lg]

                # Both sides trim block lists to ceil(prompt_len / tpb) in
                # _create_kv_slice, so dst must never exceed src. A smaller dst
                # (generation prefix-cache reuse) is handled via dst_start below.
                block_diff = dst_block_ids.size - src_block_ids.size
                if block_diff > 0:
                    raise ValueError(
                        f"src/dst block count mismatch: {src_block_ids.size} vs "
                        f"{dst_block_ids.size} (dst must not exceed src)"
                    )
                tpb = extractor.page_table.tokens_per_block
                token_range = task._slice.token_range
                lg_info = extractor.page_table.layer_groups[self_lg]
                window_size = getattr(lg_info, "sliding_window_size", None)

                # Block lists are the suffix of [..., slice_end); cached prefix
                # is implicit in their size. token_start = (total_blocks - n) * tpb.
                slice_end = token_range.end if token_range is not None else 0
                total_blocks = (slice_end + tpb - 1) // tpb
                src_beam0_blocks = Sender._beam0_block_count(
                    src_block_ids, total_blocks, task._beam_width
                )
                dst_beam0_blocks = Sender._beam0_block_count(
                    dst_block_ids, total_blocks, task._beam_width
                )
                assert src_beam0_blocks <= total_blocks, (
                    f"src beam-0 block list ({src_beam0_blocks}) exceeds total slice "
                    f"blocks ({total_blocks}); slice_end={slice_end}, tpb={tpb}"
                )
                assert dst_beam0_blocks <= total_blocks, (
                    f"dst beam-0 block list ({dst_beam0_blocks}) exceeds total slice "
                    f"blocks ({total_blocks}); slice_end={slice_end}, tpb={tpb}"
                )
                src_start = (total_blocks - src_beam0_blocks) * tpb
                dst_start = (total_blocks - dst_beam0_blocks) * tpb
                if req_info.dst_start_token is not None:
                    dst_start = max(dst_start, req_info.dst_start_token)
                if window_size is not None:
                    # SWA stale_end uses the request prompt_len (not slice_end —
                    # they differ for non-final slices). prompt_len must be plumbed
                    # via the session; falling back to slice_end is wrong on
                    # non-final slices.
                    assert task._prompt_len is not None, (
                        "SWA layer requires session.prompt_len; "
                        "set TxSession(prompt_len=request.prompt_len)."
                    )
                    stale_end = max(0, (task._prompt_len + 1 - window_size) // tpb)
                    src_start = max(stale_end * tpb, src_start)
                    dst_start = max(stale_end * tpb, dst_start)
                src_block_ids, dst_block_ids = Sender._align_kv_blocks(
                    src_block_ids,
                    dst_block_ids,
                    src_token_start=src_start,
                    dst_token_start=dst_start,
                    tokens_per_block=tpb,
                )

                src_region = extractor.extract(
                    src_block_ids, layer_group_id=self_lg, pool_idx=self_pi
                )
                dst_region = peer_extractor.extract(
                    dst_block_ids, layer_group_id=peer_lg, pool_idx=peer_pi
                )
                mapper = self._registrar.get_kv_map(peer_ri, (self_lg, self_pi), (peer_lg, peer_pi))
                region_pair = mapper.map(src_region, dst_region)
                region_pairs = region_pair if isinstance(region_pair, list) else [region_pair]
                for rp in region_pairs:
                    src_frag_parts.append(rp.src.memory.ptrs)
                    dst_frag_parts.append(rp.dst.memory.ptrs)
                    size_specs.append((rp.src.memory.ptrs.size, rp.src.memory.bytes_per_region))

        if src_frag_parts:
            src_frags = np.concatenate(src_frag_parts)
            dst_frags = np.concatenate(dst_frag_parts)
            # Build the kv_sizes array in one shot: np.repeat expands each
            # bytes_per_region value by its count, e.g.:
            #   values=[4096, 8192], counts=[100, 200]
            #   → [4096]*100 ++ [8192]*200
            counts, values = zip(*size_specs)
            kv_sizes = np.repeat(np.array(values, dtype=np.int64), counts)
        else:
            src_frags = np.array([], dtype=np.int64)
            dst_frags = np.array([], dtype=np.int64)
            kv_sizes = np.array([], dtype=np.int64)

        # handle mamba fragments
        m_src, m_dst, m_sizes = MambaPolicy.collect_frags(
            self_page_table=extractor.page_table,
            peer_page_table=peer_extractor.page_table,
            src_slot=task._slice.mamba_state_index,
            dst_slot=req_info.mamba_state_index,
            self_ri=self._registrar.self_rank_info,
            peer_ri=peer_ri,
        )
        if m_src:
            src_frags = np.concatenate([src_frags, np.array(m_src, dtype=np.int64)])
            dst_frags = np.concatenate([dst_frags, np.array(m_dst, dtype=np.int64)])
            kv_sizes = np.concatenate([kv_sizes, np.array(m_sizes, dtype=np.int64)])

        if timer:
            timer.record_prepare_args_end(peer_ri.instance_rank)
            timer.record_transfer_sizes(peer_ri.instance_rank, int(kv_sizes.sum()), dst_frags.size)

        return WriteMeta(
            task=task,
            src_ptrs=src_frags,
            dst_ptrs=dst_frags,
            sizes=kv_sizes,
            dst_device_id=dst_device_id,
            expected_transfers=expected_transfers,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_rank=peer_ri.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=task._unique_rid,
            slice_id=task.slice_id,
            is_last_slice=task._slice.is_last_slice,
            bounce_dst_base=req_info.bounce_dst_base,
            request_epoch=req_info.request_epoch,
        )

    def _build_aux_write_meta(self, task: AuxSendTask, req_info: RecvReqInfo) -> WriteMeta:
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        timer = task._perf_timer
        if timer:
            timer.record_prepare_args_start(peer_ri.instance_rank)
        expected_transfers = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)

        src_ptrs = np.array([], dtype=np.int64)
        dst_ptrs = np.array([], dtype=np.int64)
        sizes = np.array([], dtype=np.int64)
        if self._registrar.should_send_aux(peer_ri):
            src_aux_meta = self._registrar.self_rank_info.aux_meta
            peer_aux_meta = peer_ri.aux_meta
            assert src_aux_meta is not None
            assert peer_aux_meta is not None
            peer_slot = req_info.aux_slot
            assert peer_slot is not None, f"aux_slot is None for request {req_info.unique_rid}"
            assert task._slot is not None
            src_ptrs = src_aux_meta.ptrs + src_aux_meta.item_sizes * task._slot
            dst_ptrs = peer_aux_meta.ptrs + peer_aux_meta.item_sizes * peer_slot
            sizes = src_aux_meta.item_sizes.astype(np.int64, copy=False)

        if timer:
            timer.record_prepare_args_end(peer_ri.instance_rank)
            timer.record_transfer_sizes(
                peer_ri.instance_rank, int(sizes.sum()) if sizes.size > 0 else 0, src_ptrs.size
            )

        return WriteMeta(
            task=task,
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            sizes=sizes,
            expected_transfers=expected_transfers,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_rank=req_info.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=task._unique_rid,
            meta_type=WriteMetaType.AUX,
            request_epoch=req_info.request_epoch,
        )

    def dispatch_task(
        self,
        task: KVSendTask | AuxSendTask,
        req_info_snapshot: Optional[dict] = None,
        operation_owner: Optional["TxSession"] = None,
    ):
        # req_info_snapshot may be pre-fetched under session.lock by the caller to keep the
        # critical section small.  When not provided, we fetch it here (legacy / standalone path).
        if req_info_snapshot is None:
            req_info_snapshot = dict(self._get_req_info(task._unique_rid) or {})
        write_metas: list[WriteMeta] = []
        try:
            for info in req_info_snapshot.values():
                if operation_owner is not None:
                    peer_ri = self._registrar.get_peer_rank_info(
                        info.instance_name, info.instance_rank
                    )
                    operation_owner.register_request_operation(
                        peer_ri.self_endpoint, info.request_epoch
                    )
                if task._perf_timer is not None:
                    task._perf_timer.record_task_start(info.instance_rank)
                if isinstance(task, KVSendTask):
                    trans_meta = self._build_kv_write_meta(task, info)
                else:
                    trans_meta = self._build_aux_write_meta(task, info)
                if task._perf_timer is not None:
                    task._perf_timer.record_push_start(trans_meta.peer_rank)
                write_metas.append(trans_meta)
        except Exception as error:
            task.fail(error)
            if operation_owner is not None:
                operation_owner.set_exception(str(error))
            infos = list(req_info_snapshot.values())
            if operation_owner is not None:
                operation_owner.finish_failed_dispatch(len(infos))
            notification_failed = False
            for info in infos:
                on_sent = operation_owner.retire_operation if operation_owner is not None else None
                if isinstance(task, KVSendTask):
                    queued = self._send_failed_result_to_receiver(
                        info, retry=True, wait=False, on_sent=on_sent
                    )
                else:
                    queued = self._send_failed_aux_result_to_receiver(
                        info, retry=True, wait=False, on_sent=on_sent
                    )
                notification_failed = notification_failed or not queued
            if operation_owner is not None and notification_failed:
                # A terminal notification has no durable queue owner. Keep the
                # session strongly reachable and fail closed so its transfer
                # memory cannot be reclaimed while the receiver may still be
                # waiting or writing.
                with self._stalled_operations_lock:
                    self._stalled_session_owners.append(operation_owner)
            return

        if operation_owner is not None:
            operation_owner.finish_dispatch(write_metas)
        for write_meta in write_metas:
            try:
                self._enqueue(write_meta)
            except Exception as error:
                self._fail_write_meta(write_meta, error)
                if write_meta.operation_owner is not None and write_meta.terminal_sent:
                    write_meta.operation_owner.retire_operation()
                elif write_meta.operation_owner is not None:
                    with self._stalled_operations_lock:
                        self._stalled_operations.append(write_meta)

    def _start_listener(self):
        def handle_message(messages: list[bytes]):
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.REQUEST_DATA:
                    try:
                        self._respond_with_kv(send_id, msg)
                    except Exception as e:
                        logger.error(f"Sender: error handling REQUEST_DATA: {e}")
                case MessageType.REGISTER_RANK_INFO:
                    try:
                        self._register_peer_rank(send_id, msg)
                    except Exception as e:
                        logger.error(f"Sender: error handling REGISTER_RANK_INFO: {e}")
                case MessageType.CANCEL_SESSION:
                    try:
                        self._handle_cancel_session(msg)
                    except Exception as e:
                        logger.error(f"Sender: error handling CANCEL_SESSION: {e}")
                case _:
                    logger.error(f"Sender received unknown message type: {msg[0]}")

        self._messenger.start_listener(handle_message)

    def _register_peer_rank(self, _send_id: bytes, message: list[bytes]):
        # Skip late messages so we don't race shutdown's invalidate loop.
        if self._shutdown:
            return
        torch.cuda.set_device(self._device_id)
        CUASSERT(cudart.cudaSetDevice(self._device_id))
        ri: RankInfo = RankInfo.from_bytes(message[1])
        capabilities = _decode_protocol_capabilities(message[2] if len(message) > 2 else None)

        self._registrar.register(ri.instance_name, ri.instance_rank, ri)
        self._peer_capabilities[ri.self_endpoint] = capabilities

        agent_name = ri.instance_name + str(ri.instance_rank)
        logger.debug(f"Loading remote transfer agent descriptor for peer '{agent_name}'")
        self._agent.load_remote_agent(
            ri.instance_name + str(ri.instance_rank),
            ri.transfer_engine_info,
        )
        with self._loaded_remote_agents_lock:
            self._loaded_remote_agents.add(agent_name)
        logger.debug(
            f"Completed handling REGISTER_RANK_INFO for instance='{ri.instance_name}', rank={ri.instance_rank}"
        )

    def _handle_cancel_session(self, message: list[bytes]):
        unique_rid = int(message[1])
        ack_endpoint = message[2].decode("utf-8") if len(message) > 2 else None
        request_epoch = int(message[3]) if len(message) > 3 else None
        session = None
        resend_ack = False
        with self._ingress_lock, self._sessions_lock:
            if request_epoch is not None:
                if ack_endpoint is None:
                    raise RuntimeError("v2 cancel is missing its receiver endpoint")
                operation = (unique_rid, ack_endpoint, request_epoch)
                if operation in self._cancelled_operation_tombstones:
                    resend_ack = True
                else:
                    session_ref = self._sessions.get(unique_rid)
                    session = session_ref() if session_ref is not None else None
                    if session is not None and session.has_request_operation(
                        ack_endpoint, request_epoch
                    ):
                        pass
                    elif session is not None and session.has_request_endpoint(ack_endpoint):
                        error = RuntimeError(
                            "stale native-transfer cancel does not match the active "
                            f"request incarnation: rid={unique_rid} endpoint={ack_endpoint} "
                            f"epoch={request_epoch}"
                        )
                        self._protocol_error = error
                        raise error
                    else:
                        session = None
                        self._remember_pre_cancelled_operation_unlocked(
                            unique_rid, ack_endpoint, request_epoch
                        )
                        resend_ack = True
            else:
                session_ref = self._sessions.get(unique_rid)
                if session_ref is None:
                    if unique_rid not in self._closed_rids:
                        self._remember_pre_cancelled_unlocked(unique_rid)
                else:
                    session = session_ref()
                    if session is None and unique_rid not in self._closed_rids:
                        self._remember_pre_cancelled_unlocked(unique_rid)
        if session is not None:
            session.cancel(ack_endpoint=ack_endpoint, request_epoch=request_epoch)
        elif resend_ack and ack_endpoint is not None:
            self.send_cancel_ack(
                ack_endpoint,
                unique_rid,
                request_epoch=request_epoch,
                from_worker=False,
            )

    @nvtx_range("_respond_with_kv")
    def _respond_with_kv(self, _send_id: bytes, message: list[bytes]):
        # _sessions_lock prevents a race between session lookup and req_info save.
        # session.lock serializes _enqueue calls from both paths.
        info: RecvReqInfo = RecvReqInfo.from_bytes(message[1])
        peer_ri = self._registrar.get_peer_rank_info(info.instance_name, info.instance_rank)
        if self._shutdown:
            self._send_failed_result_to_receiver(info, retry=True, wait=False)
            return
        capabilities = self._peer_capabilities.get(
            peer_ri.self_endpoint, _LEGACY_PROTOCOL_CAPABILITIES
        )
        if info.request_epoch is not None and not _supports_native_protocol_v2(capabilities):
            logger.error(
                "Rejecting v2 request %s from incompatible peer %s (version=%s)",
                info.unique_rid,
                peer_ri.self_endpoint,
                capabilities.version,
            )
            self._send_failed_result_to_receiver(info, retry=True, wait=False)
            return
        if _requires_drain_ack_protocol() and not _supports_native_protocol_v2(capabilities):
            logger.error(
                "Rejecting request %s from legacy native-transfer peer %s: "
                "asynchronous consensus requires drain-ACK protocol version %s",
                info.unique_rid,
                peer_ri.self_endpoint,
                _NATIVE_PROTOCOL_VERSION,
            )
            self._send_failed_result_to_receiver(info, retry=True, wait=False)
            return
        cancelled_before_request = False
        with self._ingress_lock, self._sessions_lock:
            session = self._get_session(info.unique_rid)
            if info.request_epoch is not None:
                operation = (
                    info.unique_rid,
                    peer_ri.self_endpoint,
                    info.request_epoch,
                )
                if operation in self._pre_cancelled_operations:
                    self._pre_cancelled_operations.pop(operation, None)
                    self._remember_cancelled_operation_unlocked(operation)
                    cancelled_before_request = True
                elif operation in self._cancelled_operation_tombstones:
                    cancelled_before_request = True
                elif session is not None:
                    session.register_request_operation(peer_ri.self_endpoint, info.request_epoch)
            if session is None:
                if (
                    cancelled_before_request
                    or info.unique_rid in self._closed_rids
                    or info.unique_rid in self._pre_cancelled_rids
                ):
                    cancelled_before_request = True
                else:
                    self._save_peer_req_info(info)
                session = None
        if cancelled_before_request:
            self._send_failed_result_to_receiver(info)
            return
        if session is None:
            return
        dispatch_tasks: list[SendTaskBase] = []
        with session.lock:
            self._save_peer_req_info(info)
            tasks = list(session.kv_tasks)
            # No tasks: no worker will send KV_AGENT_RESULT FAILED to the receiver.
            # Send it directly to unblock the receiver's TRANSFERRING task future;
            # CANCEL_SESSION alone would leave it stuck indefinitely.
            if not tasks and session.status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                self._send_failed_result_to_receiver(info)
                return
            for task in tasks:
                if session.begin_dispatch_unlocked():
                    dispatch_tasks.append(task)
                else:
                    self._send_failed_result_to_receiver(info)
        for task in dispatch_tasks:
            self.dispatch_task(task, {info.instance_rank: info}, operation_owner=session)

    def _send_failed_result_to_receiver(
        self,
        info: RecvReqInfo,
        *,
        retry: bool = False,
        wait: bool = True,
        on_sent: Optional[Callable[[], None]] = None,
    ) -> bool:
        try:
            peer_ri = self._registrar.get_peer_rank_info(info.instance_name, info.instance_rank)
            slice_id = info.slice_id if info.slice_id is not None else 0
            self._send_control_message(
                peer_ri.self_endpoint,
                _make_kv_result_msg(
                    self._instance_rank,
                    info.unique_rid,
                    slice_id,
                    True,
                    AgentResult.FAILED,
                    request_epoch=info.request_epoch,
                    sender_endpoint=self.endpoint,
                ),
                retry=retry,
                wait=wait,
                on_sent=on_sent,
            )
            return True
        except Exception as e:
            logger.warning(
                f"_respond_with_kv: failed to abort receiver for rid={info.unique_rid}: {e}"
            )
            return False

    def _send_failed_aux_result_to_receiver(
        self,
        info: RecvReqInfo,
        *,
        retry: bool = False,
        wait: bool = True,
        on_sent: Optional[Callable[[], None]] = None,
    ) -> bool:
        try:
            peer_ri = self._registrar.get_peer_rank_info(info.instance_name, info.instance_rank)
            message = [
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(info.unique_rid).encode("ascii"),
                AgentResult.FAILED.value.encode("ascii"),
            ]
            if info.request_epoch is not None:
                message.extend(
                    [
                        self.endpoint.encode("utf-8"),
                        str(info.request_epoch).encode("ascii"),
                    ]
                )
            self._send_control_message(
                peer_ri.self_endpoint,
                message,
                retry=retry,
                wait=wait,
                on_sent=on_sent,
            )
            return True
        except Exception as error:
            logger.warning(f"Failed to abort aux receiver for rid={info.unique_rid}: {error}")
            return False

    def _send_control_message(
        self,
        endpoint: Optional[str],
        message: list[bytes],
        *,
        retry: bool = False,
        wait: bool = True,
        on_sent: Optional[Callable[[], None]] = None,
        repeat_until: Optional[Callable[[], bool]] = None,
    ) -> _ControlSend:
        return self._control.send(
            endpoint,
            message,
            retry=retry,
            wait=wait,
            on_sent=on_sent,
            repeat_until=repeat_until,
        )

    def _save_peer_req_info(self, peer_transfer_req_info: RecvReqInfo):
        req_info = peer_transfer_req_info
        self._add_req_info(req_info.unique_rid, req_info.instance_rank, req_info)
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        expected_transfers = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
        if self._is_req_ready(req_info.unique_rid, expected_transfers):
            session = self._get_session(req_info.unique_rid)
            if session is not None and not session.receiver_ready:
                session.receiver_ready = True

    def has_all_peer_req_infos(self, unique_rid: int) -> bool:
        req_info = self._get_first_req_info(unique_rid)
        if req_info:
            return self._has_all_peer_req_infos(req_info)
        return False

    def _has_all_peer_req_infos(self, req_info: RecvReqInfo) -> bool:
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        expected_transfers = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
        return self._is_req_ready(req_info.unique_rid, expected_transfers)

    def clear_session(
        self,
        unique_rid: int,
        drained_operations: Optional[set[tuple[str, Optional[int]]]] = None,
    ):
        with self._sessions_lock:
            self._sessions.pop(unique_rid, None)
            self._remember_closed_unlocked(unique_rid)
            for endpoint, request_epoch in drained_operations or ():
                if request_epoch is not None:
                    self._remember_cancelled_operation_unlocked(
                        (unique_rid, endpoint, request_epoch)
                    )
        self._remove_req_info(unique_rid)

    def send_cancel_to_receivers(self, unique_rid: int) -> None:
        """Notify all receivers involved in this session to cancel."""
        # Snapshot under the lock to avoid RuntimeError if the listener thread
        # mutates the dict via _add_req_info() concurrently.
        with self._peer_requests_lock:
            req_info_map = self._peer_requests.get(unique_rid)
            req_infos = list(req_info_map.values()) if req_info_map else []
        for req_info in req_infos:
            try:
                peer_ri = self._registrar.get_peer_rank_info(
                    req_info.instance_name, req_info.instance_rank
                )
                self._send_control_message(
                    peer_ri.self_endpoint,
                    [
                        MessageType.CANCEL_SESSION,
                        str(unique_rid).encode("ascii"),
                        self.endpoint.encode("utf-8"),
                        *(
                            [str(req_info.request_epoch).encode("ascii")]
                            if req_info.request_epoch is not None
                            else []
                        ),
                    ],
                    retry=True,
                    wait=False,
                )
            except Exception as e:
                logger.warning(f"send_cancel_to_receivers: failed for rid={unique_rid}: {e}")

    def send_cancel_ack(
        self,
        endpoint: str,
        unique_rid: int,
        *,
        request_epoch: Optional[int] = None,
        from_worker: bool,
    ) -> bool:
        """Acknowledge that all sender-side writes for a cancelled session drained."""
        try:
            message = [
                MessageType.CANCEL_SESSION_ACK,
                str(unique_rid).encode("ascii"),
                self.endpoint.encode("utf-8"),
            ]
            if request_epoch is not None:
                message.append(str(request_epoch).encode("ascii"))
            self._send_control_message(endpoint, message, retry=True, wait=False)
            return True
        except Exception as error:
            logger.warning(f"Failed to acknowledge cancel for rid={unique_rid}: {error}")
            return False

    def shutdown(self):
        with self._ingress_lock:
            if self._shutdown:
                return
            self._shutdown = True
            for q in self._send_task_queues:
                q.put(None)
        for t in self._worker_threads:
            t.join(timeout=5)
        live_workers = [t for t in self._worker_threads if t.is_alive()]
        if live_workers:
            logger.warning(
                f"Sender shutdown is waiting for {len(live_workers)} transfer worker(s) "
                "to drain before invalidating remote agents"
            )
            # Registered transfer memory and remote-agent descriptors must
            # outlive every in-flight worker. Returning after a timed join
            # would race memory teardown, so shutdown deliberately fails
            # closed and waits for the transfer agent's own bounded wait.
        for t in live_workers:
            t.join()
        with self._stalled_operations_lock:
            if self._ambiguous_operations:
                raise RuntimeError(
                    "Sender has transfer-agent operations with unknown terminal state; "
                    "retaining control sockets, remote agents, and registered memory"
                )
        self._control.flush()
        # Keep cancellation/ACK progress alive until every worker and queued
        # terminal notification has drained. Stop ingress only at that point.
        self._messenger.stop()
        self._control.flush()
        with self._stalled_operations_lock:
            if self._stalled_operations or self._stalled_session_owners:
                logger.error(
                    "Sender shutdown retained %s operation(s) and %s session owner(s) "
                    "whose terminal peer response could not be published",
                    len(self._stalled_operations),
                    len(self._stalled_session_owners),
                )

        # Snapshot under lock as defense in depth.
        with self._loaded_remote_agents_lock:
            loaded_agents = list(self._loaded_remote_agents)
            self._loaded_remote_agents.clear()
        # Invalidate all loaded remote agents to release fabric/POSIX FD resources.
        for agent_name in loaded_agents:
            try:
                self._agent.invalidate_remote_agent(agent_name)
            except Exception as e:
                logger.warning(
                    f"Failed to invalidate remote agent '{agent_name}' during shutdown: {e}"
                )
        self._control.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Sender.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()


class TxSession(TxSessionBase):
    def __init__(
        self,
        request_id: int,
        params: DisaggregatedParams,
        sender: Sender,
        aux_buffer: Optional[AuxBuffer] = None,
        timeout_s: Optional[float] = None,
        prompt_len: Optional[int] = None,
        beam_width: int = 1,
    ):
        super().__init__(
            sender,
            SessionArgsBase(params, prompt_len=prompt_len, beam_width=beam_width),
        )
        self._timeout_s = timeout_s
        self._need_aux = params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        self._sender: Sender  # narrow base class type for Pylance
        self.request_id = request_id
        self._aux_buffer = aux_buffer
        self.aux_slot = aux_buffer.alloc_slot().id if aux_buffer is not None else None
        self.receiver_ready: bool = False
        self.kv_tasks = []
        self.aux_task = None
        self.lock = threading.Lock()

        self._exception: Optional[Exception] = None
        self._closed = False
        self._sealed = False
        self._terminal_status: Optional[SessionStatus] = None
        self._terminal_snapshot: Optional[SessionStatus] = None
        self._outstanding_operations = 0
        self._close_requested = False
        self._cancel_ack_endpoints: set[str] = set()
        self._cancel_acked_endpoints: set[str] = set()
        self._request_operations: set[tuple[str, Optional[int]]] = set()
        self._cancel_ack_operations: set[tuple[str, Optional[int]]] = set()
        self._cancel_acked_operations: set[tuple[str, Optional[int]]] = set()
        self._cancel_notified = False
        # Must be last: makes session visible to listener thread,
        # so all attributes above must be initialized first.
        self._sender.setup_session(self)

    @property
    def disagg_request_id(self) -> int:
        params = self._base_args.params
        if params.disagg_request_id is not None:
            return params.disagg_request_id
        # ctx_request_id is set on gen-side requests to the ctx server's request ID,
        # which matches the key the ctx TxSession registered under.  Fall back to
        # the local request_id only when neither field is available.
        if params.ctx_request_id is not None:
            return params.ctx_request_id
        return self.request_id

    @property
    def status(self) -> SessionStatus:
        if self._terminal_snapshot is not None:
            return self._terminal_snapshot
        if self._terminal_status is not None:
            return self._terminal_status
        if self._exception is not None or any(
            task.status == TaskStatus.ERROR for task in self.kv_tasks
        ):
            return SessionStatus.ERROR
        if self.aux_task is not None and self.aux_task.status == TaskStatus.ERROR:
            return SessionStatus.ERROR
        kv_all_transferred = bool(self.kv_tasks) and all(
            t.status == TaskStatus.TRANSFERRED for t in self.kv_tasks
        )
        if kv_all_transferred:
            if self.aux_task is not None and self.aux_task.status == TaskStatus.TRANSFERRED:
                return SessionStatus.FULLY_TRANSFERRED
            return SessionStatus.KV_TRANSFERRED
        if self.kv_tasks and any(t.status == TaskStatus.TRANSFERRING for t in self.kv_tasks):
            return SessionStatus.TRANSFERRING
        return SessionStatus.READY if self.receiver_ready else SessionStatus.INIT

    def send(self, slice: KVSlice) -> None:
        with self.lock:
            params = self._base_args.params
            slice_id = len(self.kv_tasks)
            task = KVSendTask(
                slice,
                params,
                slice_id,
                prompt_len=self._base_args.prompt_len,
                beam_width=self._base_args.beam_width,
            )
            task._unique_rid = self.disagg_request_id
            self.kv_tasks.append(task)
            if self._sealed or self._closed:
                task.fail(
                    RuntimeError(
                        f"TxSession {self.disagg_request_id} is sealed; KV transfer rejected"
                    )
                )
                return
            req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
            self.begin_dispatch_unlocked()
        self._sender.dispatch_task(task, req_info_snapshot, operation_owner=self)

    def send_aux(self) -> AuxSendTask:
        with self.lock:
            params = self._base_args.params
            task = AuxSendTask(params, self.aux_slot)
            task._unique_rid = self.disagg_request_id
            self.aux_task = task
            if self._sealed or self._closed:
                task.fail(
                    RuntimeError(
                        f"TxSession {self.disagg_request_id} is sealed; aux transfer rejected"
                    )
                )
                return task
            req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
            self.begin_dispatch_unlocked()
        self._sender.dispatch_task(task, req_info_snapshot, operation_owner=self)
        return task

    def begin_dispatch_unlocked(self) -> bool:
        """Reserve a lifetime credit before preparing peer write metadata."""
        if self._sealed or self._closed:
            return False
        self._outstanding_operations += 1
        return True

    def register_request_operation(
        self, receiver_endpoint: str, request_epoch: Optional[int]
    ) -> None:
        """Bind a receiver endpoint to one immutable request incarnation."""
        operation = (receiver_endpoint, request_epoch)
        with self.lock:
            for known_endpoint, known_epoch in self._request_operations:
                if (
                    known_endpoint == receiver_endpoint
                    and known_epoch != request_epoch
                    and (known_epoch is not None or request_epoch is not None)
                ):
                    raise RuntimeError(
                        "receiver endpoint changed request incarnation inside an active "
                        f"TxSession: rid={self.disagg_request_id} endpoint={receiver_endpoint} "
                        f"old_epoch={known_epoch} new_epoch={request_epoch}"
                    )
            self._request_operations.add(operation)

    def has_request_operation(self, receiver_endpoint: str, request_epoch: Optional[int]) -> bool:
        with self.lock:
            return (receiver_endpoint, request_epoch) in self._request_operations

    def has_request_endpoint(self, receiver_endpoint: str) -> bool:
        with self.lock:
            return any(
                endpoint == receiver_endpoint
                for endpoint, _request_epoch in self._request_operations
            )

    def finish_dispatch(self, write_metas: list[WriteMeta]) -> None:
        """Replace one preparation credit with its queued peer obligations."""
        ack_endpoints: list[tuple[str, Optional[int]]] = []
        finalize_close = False
        with self.lock:
            if self._outstanding_operations <= 0:
                raise RuntimeError(f"TxSession {self.disagg_request_id} dispatch credit underflow")
            self._outstanding_operations -= 1
            for write_meta in write_metas:
                write_meta.operation_owner = self
            self._outstanding_operations += len(write_metas)
            ack_endpoints, finalize_close = self._post_drain_actions_unlocked()
        self._run_post_drain_actions(ack_endpoints, finalize_close, from_worker=False)

    def finish_failed_dispatch(self, notification_count: int) -> None:
        """Replace one preparation credit with durable failure notifications.

        Each notification callback retires one credit only after the control
        owner has accepted the terminal message. This keeps a strong session
        owner across transient ZMQ failures.
        """
        ack_endpoints: list[tuple[str, Optional[int]]] = []
        finalize_close = False
        with self.lock:
            if self._outstanding_operations <= 0:
                raise RuntimeError(f"TxSession {self.disagg_request_id} dispatch credit underflow")
            self._outstanding_operations -= 1
            self._outstanding_operations += notification_count
            ack_endpoints, finalize_close = self._post_drain_actions_unlocked()
        self._run_post_drain_actions(ack_endpoints, finalize_close, from_worker=False)

    def retire_operation(self) -> None:
        """Retire one peer response obligation after its terminal result was sent."""
        ack_endpoints: list[tuple[str, Optional[int]]] = []
        finalize_close = False
        with self.lock:
            if self._outstanding_operations <= 0:
                raise RuntimeError(f"TxSession {self.disagg_request_id} operation credit underflow")
            self._outstanding_operations -= 1
            ack_endpoints, finalize_close = self._post_drain_actions_unlocked()
        self._run_post_drain_actions(ack_endpoints, finalize_close, from_worker=True)

    def _post_drain_actions_unlocked(
        self,
    ) -> tuple[list[tuple[str, Optional[int]]], bool]:
        if self._outstanding_operations != 0:
            return [], False
        ack_operations: list[tuple[str, Optional[int]]] = []
        if self._terminal_status == SessionStatus.CANCELLED:
            if self._cancel_ack_operations:
                ack_operations = list(self._cancel_ack_operations)
                self._cancel_ack_operations.clear()
                self._cancel_acked_operations.update(ack_operations)
            elif self._cancel_ack_endpoints:
                # Legacy drain-ACK compatibility.
                ack_operations = [(endpoint, None) for endpoint in self._cancel_ack_endpoints]
                self._cancel_ack_endpoints.clear()
                self._cancel_acked_endpoints.update(endpoint for endpoint, _epoch in ack_operations)
        return ack_operations, self._close_requested and not self._closed

    def _run_post_drain_actions(
        self,
        ack_operations: list[tuple[str, Optional[int]]],
        finalize_close: bool,
        *,
        from_worker: bool,
    ) -> None:
        for endpoint, request_epoch in ack_operations:
            if not self._sender.send_cancel_ack(
                endpoint,
                self.disagg_request_id,
                request_epoch=request_epoch,
                from_worker=from_worker,
            ):
                # Queue rejection can only happen during teardown. Restore the
                # pending obligation so a duplicate cancel can retry rather
                # than silently treating an unsent ACK as durable.
                with self.lock:
                    operation = (endpoint, request_epoch)
                    if request_epoch is None:
                        self._cancel_acked_endpoints.discard(endpoint)
                        self._cancel_ack_endpoints.add(endpoint)
                    else:
                        self._cancel_acked_operations.discard(operation)
                        self._cancel_ack_operations.add(operation)
        if finalize_close:
            self._finalize_close()

    def pack_aux(self, request: LlmRequest) -> None:
        """Fill the aux buffer slot with token data from the given request."""
        assert self._aux_buffer is not None, "No aux_buffer set for this session"
        assert self.aux_slot is not None, "No aux_slot set for this session"
        self._aux_buffer.fill_slot(self.aux_slot, request)

    def is_completed(self) -> bool:
        """Non-blocking check: has the transfer completed successfully?"""
        status = self.status
        if self._need_aux:
            return status == SessionStatus.FULLY_TRANSFERRED
        return status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)

    def has_failed(self) -> bool:
        """Non-blocking check: has the transfer failed or been cancelled?"""
        if self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
            return True
        if any(task.status == TaskStatus.ERROR for task in self.kv_tasks):
            return True
        return self.aux_task is not None and self.aux_task.status == TaskStatus.ERROR

    def cancel(
        self,
        ack_endpoint: Optional[str] = None,
        request_epoch: Optional[int] = None,
    ) -> None:
        """Cancel the session and notify the remote receiver.

        Safe to call multiple times. TRANSFERRING tasks keep running (mid-write).
        Only INIT tasks have their events signalled immediately.
        The lock serializes with _deliver_kv_to_agent() so has_transferring_tasks()
        is accurate the moment this returns.
        """
        ack_operations: list[tuple[str, Optional[int]]] = []
        notify_receivers = False
        with self.lock:
            if ack_endpoint is not None:
                operation = (ack_endpoint, request_epoch)
                if request_epoch is None:
                    if ack_endpoint not in self._cancel_acked_endpoints:
                        self._cancel_ack_endpoints.add(ack_endpoint)
                    elif self._outstanding_operations == 0:
                        # CANCEL is retransmitted until an ACK arrives. Always
                        # resend an idempotent ACK after drain; queue acceptance
                        # is not proof that the previous message was delivered.
                        ack_operations.append(operation)
                elif operation not in self._request_operations:
                    raise RuntimeError(
                        "cancel request does not match an active Tx operation: "
                        f"rid={self.disagg_request_id} endpoint={ack_endpoint} "
                        f"epoch={request_epoch}"
                    )
                elif operation not in self._cancel_acked_operations:
                    self._cancel_ack_operations.add(operation)
                elif self._outstanding_operations == 0:
                    ack_operations.append(operation)
            if self._terminal_snapshot is not None:
                if self._outstanding_operations == 0:
                    v2_ack_operations = list(self._cancel_ack_operations)
                    ack_operations.extend(v2_ack_operations)
                    self._cancel_ack_operations.clear()
                    self._cancel_acked_operations.update(v2_ack_operations)
                    legacy_ack_operations = [
                        (endpoint, None) for endpoint in self._cancel_ack_endpoints
                    ]
                    ack_operations.extend(legacy_ack_operations)
                    self._cancel_ack_endpoints.clear()
                    self._cancel_acked_endpoints.update(
                        endpoint for endpoint, _epoch in legacy_ack_operations
                    )
            elif self._terminal_status == SessionStatus.CANCELLED:
                newly_ready, _ = self._post_drain_actions_unlocked()
                ack_operations.extend(newly_ready)
            else:
                self._sealed = True
                self._terminal_status = SessionStatus.CANCELLED
                exc = RuntimeError(f"TxSession {self.disagg_request_id} cancelled")
                for task in self.kv_tasks:
                    if task.status == TaskStatus.INIT:
                        task.fail(exc)
                if self.aux_task is not None and self.aux_task.status == TaskStatus.INIT:
                    self.aux_task.fail(exc)
                newly_ready, _ = self._post_drain_actions_unlocked()
                ack_operations.extend(newly_ready)
            if self._terminal_snapshot is None and not self._cancel_notified:
                self._cancel_notified = True
                notify_receivers = True
        # Send outside the lock to avoid holding it during I/O.
        if notify_receivers:
            self._sender.send_cancel_to_receivers(self.disagg_request_id)
        self._run_post_drain_actions(ack_operations, False, from_worker=False)

    def has_transferring_tasks(self) -> bool:
        """True if any KV or aux task is currently mid-write (TRANSFERRING).

        cancel_request() must return False while this is True.
        """
        with self.lock:
            return self._has_transferring_tasks_unlocked()

    def _has_transferring_tasks_unlocked(self) -> bool:
        return self._outstanding_operations != 0

    def seal_and_check_quiescent(self) -> bool:
        """Atomically prevent new transfers and check whether active writes drained.

        Sender workers use the same lock for INIT-to-TRANSFERRING transitions,
        so no task can start after this method reports quiescence.
        """
        with self.lock:
            self._sealed = True
            return not self._has_transferring_tasks_unlocked()

    def seal_and_snapshot_terminal(self) -> Optional[SessionStatus]:
        """Atomically snapshot a drained terminal state without freezing pending work."""
        with self.lock:
            if self._terminal_snapshot is not None:
                return self._terminal_snapshot
            if self._outstanding_operations != 0:
                return None
            status = self.status
            terminal = status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
            terminal = terminal or (
                not self._need_aux
                and status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)
            )
            terminal = terminal or (self._need_aux and status == SessionStatus.FULLY_TRANSFERRED)
            if not terminal:
                return None
            self._sealed = True
            self._terminal_snapshot = status
            return status

    def try_mark_transferring(self, task: SendTaskBase) -> bool:
        """Start a queued task unless the session has crossed its seal boundary."""
        with self.lock:
            if (
                self._sealed
                or self._closed
                or self._terminal_status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
                or task.status not in (TaskStatus.INIT, TaskStatus.TRANSFERRING)
            ):
                return False
            task.status = TaskStatus.TRANSFERRING
            return True

    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        """Poll or block until KV (and optionally aux) transfer finishes.

        With blocking=True (default): waits up to _timeout_s for each task.
        With blocking=False: polls non-blockingly; returns None if any KV task
        or aux is not yet done.
        """
        if self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
            return WaitResult.FAILED
        if not self.kv_tasks:
            return None
        if not blocking:
            has_pending = False
            for task in self.kv_tasks:
                if task.status == TaskStatus.ERROR:
                    return WaitResult.FAILED
                if task.status != TaskStatus.TRANSFERRED:
                    has_pending = True
            if has_pending:
                return None
            if self._need_aux:
                if self.aux_task is None:
                    return None
                if self.aux_task.status == TaskStatus.ERROR:
                    return WaitResult.FAILED
                if self.aux_task.status != TaskStatus.TRANSFERRED:
                    return None
            return WaitResult.COMPLETED

        for task in self.kv_tasks:
            if not task.wait(timeout=self._timeout_s):
                return WaitResult.TIMEOUT
            if task.status == TaskStatus.ERROR:
                return WaitResult.FAILED
        if self._need_aux and self.aux_task is not None:
            if not self.aux_task.wait(timeout=self._timeout_s):
                return WaitResult.TIMEOUT
            if self.aux_task.status == TaskStatus.ERROR:
                return WaitResult.FAILED
        return WaitResult.COMPLETED

    def set_exception(self, reason: str = ""):
        msg = f"TxSession {self.disagg_request_id} exception"
        if reason:
            msg += f": {reason}"
        with self.lock:
            if self._terminal_snapshot is not None:
                return
            if self._terminal_status == SessionStatus.CANCELLED:
                return
            self._exception = RuntimeError(msg)
            self._sealed = True
            self._terminal_status = SessionStatus.ERROR
            for task in self.kv_tasks:
                if task.status == TaskStatus.INIT:
                    task.fail(self._exception)
            if self.aux_task is not None and self.aux_task.status == TaskStatus.INIT:
                self.aux_task.fail(self._exception)

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        finalize = False
        with self.lock:
            if getattr(self, "_closed", False):
                return
            self._sealed = True
            self._close_requested = True
            finalize = self._outstanding_operations == 0
        if finalize:
            self._finalize_close()

    def _finalize_close(self) -> None:
        with self.lock:
            if self._closed or self._outstanding_operations != 0:
                return
            self._closed = True
        if self._aux_buffer is not None and self.aux_slot is not None:
            self._aux_buffer.free_slot(self.aux_slot)
            self.aux_slot = None
        # Unregister from Sender; keep fields alive for in-flight worker threads.
        if self._sender is not None:
            self._sender.clear_session(self.disagg_request_id, set(self._request_operations))

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.warning(f"TxSession.__del__: exception during close: {e}")


class KVRecvTask:
    def __init__(
        self,
        unique_rid: Optional[int],
        kv_slice: KVSlice,
        slice_id: int,
        params: DisaggregatedParams,
        aux_slot: Optional[int],
    ):
        self._event = threading.Event()
        self.slice_id = slice_id
        self.status = TaskStatus.INIT
        self.expected_transfers = 0
        self.last_slice_count = 0

        self._unique_rid = unique_rid
        self._kv_slice = kv_slice
        self._params = params
        self._exception: Optional[Exception] = None
        self._aux_slot = aux_slot
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

    def fail(self, exc: Exception) -> None:
        self._exception = exc
        self.status = TaskStatus.ERROR
        self._event.set()

    def complete(self) -> None:
        self.status = TaskStatus.TRANSFERRED
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until terminal state. Returns True if done, False on timeout."""
        return self._event.wait(timeout=timeout)

    @property
    def is_done(self) -> bool:
        return self._event.is_set()

    def print_perf_info(self, peer_rank: int, instance_name: str, instance_rank: int):
        if self._perf_timer is None:
            return
        assert self._unique_rid is not None
        perf_log_manager.log_recv_task_perf(
            self._unique_rid,
            peer_rank,
            instance_name,
            instance_rank,
            self._perf_timer,
        )


class Receiver(ReceiverBase):
    _TOMBSTONE_LIMIT = 65_536
    _SHUTDOWN_DRAIN_LOG_INTERVAL_S = 5.0

    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        agent: BaseTransferAgent,
        bounce=None,
    ):
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = threading.Event()
        self._shutdown_thread: Optional[threading.Thread] = None
        self._shutdown_error: Optional[BaseException] = None
        self._registrar = peer_registrar
        self._agent = agent
        self._bounce = bounce
        self._control = _ControlPlane("native-receiver")
        self._sender_ep_instance_map = {}
        self._sender_info_capabilities: dict[str, _NativeProtocolCapabilities] = {}
        self._sender_endpoint_capabilities: dict[str, _NativeProtocolCapabilities] = {}

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._sessions = {}  # unique_rid -> RxSession
        self._sessions_lock = threading.Lock()
        self._sessions_drained = threading.Condition(self._sessions_lock)
        self._draining_sessions: dict[int, RxSession] = {}
        # A pre-cancel is live protocol state: evicting it can let a delayed
        # request start after cancellation was already acknowledged upstream.
        self._pre_cancelled_rids: dict[int, None] = {}
        self._closed_rids: OrderedDict[int, None] = OrderedDict()
        self._closed_operations: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._protocol_error: Optional[RuntimeError] = None

        self._start_listener()
        logger.info(f"Receiver init with endpoint: {self._messenger.endpoint}")

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def shutdown(self) -> Optional[threading.Event]:
        if self._is_progress_thread():
            return self._defer_shutdown()
        run_shutdown = False
        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                self._raise_shutdown_error()
                return
            if not self._shutdown:
                self._shutdown = True
                run_shutdown = True

        if run_shutdown:
            self._run_shutdown()
        else:
            self._shutdown_complete.wait()
        self._raise_shutdown_error()
        return None

    def _is_progress_thread(self) -> bool:
        """Whether blocking here would prevent an Rx operation from draining."""
        current = threading.current_thread()
        listener_thread = getattr(self._messenger, "_listener_thread", None)
        scatter_thread = getattr(self._bounce, "_scatter_thread", None)
        return current is listener_thread or current is scatter_thread

    def _defer_shutdown(self) -> threading.Event:
        """Let the current listener/scatter callback retire its own drain credit."""
        with self._shutdown_lock:
            if self._shutdown_complete.is_set() or self._shutdown:
                return self._shutdown_complete
            self._shutdown = True
            self._shutdown_thread = threading.Thread(
                target=self._run_shutdown,
                name="trtllm-rx-shutdown",
                daemon=True,
            )
            shutdown_thread = self._shutdown_thread
        logger.warning(
            "Receiver shutdown was requested from an Rx progress thread; "
            "deferring teardown until the callback returns and all sessions drain"
        )
        shutdown_thread.start()
        return self._shutdown_complete

    def _run_shutdown(self) -> None:
        try:
            self._close_and_drain_sessions()
            # The listener and control dealers are themselves part of the
            # receive progress engine. Stop them only after no session can
            # still need a terminal result or cancel ACK.
            self._messenger.stop()
            self._control.flush()
            self._control.shutdown()
        except BaseException as error:
            self._shutdown_error = error
            logger.exception("Receiver shutdown failed before receive resources were quiescent")
        finally:
            self._shutdown_complete.set()

    def _close_and_drain_sessions(self) -> None:
        # Closing every still-live session makes it either finalize immediately
        # or retain itself strongly in _draining_sessions. Snapshot first: a
        # finalizer calls clear_session(), which takes the same condition lock.
        with self._sessions_drained:
            sessions = []
            for session_ref in self._sessions.values():
                session = session_ref()
                if session is not None:
                    sessions.append(session)
        for session in sessions:
            session.close()

        with self._sessions_drained:
            while self._draining_sessions:
                draining_rids = list(self._draining_sessions)
                logger.warning(
                    "Receiver shutdown is waiting for %d session(s) to drain before "
                    "stopping its listener or releasing transfer memory: %s",
                    len(draining_rids),
                    draining_rids,
                )
                self._sessions_drained.wait(timeout=self._SHUTDOWN_DRAIN_LOG_INTERVAL_S)

    def _raise_shutdown_error(self) -> None:
        if self._shutdown_error is not None:
            raise RuntimeError(
                "Receiver shutdown did not complete safely"
            ) from self._shutdown_error

    def clear_session(self, unique_rid: int, request_epoch: Optional[int] = None):
        with self._sessions_drained:
            self._sessions.pop(unique_rid, None)
            self._draining_sessions.pop(unique_rid, None)
            self._remember_tombstone(self._closed_rids, unique_rid)
            if request_epoch is not None:
                self._remember_tombstone(self._closed_operations, (unique_rid, request_epoch))
            self._sessions_drained.notify_all()

    def retain_draining_session(self, session: "RxSession") -> None:
        with self._sessions_drained:
            self._draining_sessions[session.disagg_request_id] = session
            self._sessions_drained.notify_all()

    def setup_session(self, rx_session: RxSessionBase):
        pre_cancel = False
        with self._sessions_lock:
            if self._shutdown:
                raise RuntimeError("Cannot create an RxSession after Receiver shutdown started")
            if self._protocol_error is not None:
                raise RuntimeError(
                    "Receiver is in fail-stop protocol state"
                ) from self._protocol_error
            self._sessions[rx_session.disagg_request_id] = weakref.ref(rx_session)
            if rx_session.disagg_request_id in self._pre_cancelled_rids:
                pre_cancel = True
                self._pre_cancelled_rids.pop(rx_session.disagg_request_id, None)
            self._closed_rids.pop(rx_session.disagg_request_id, None)
        if pre_cancel:
            rx_session.cancel()

    @classmethod
    def _remember_tombstone(cls, tombstones: OrderedDict, key) -> None:
        tombstones.pop(key, None)
        tombstones[key] = None
        while len(tombstones) > cls._TOMBSTONE_LIMIT:
            tombstones.popitem(last=False)

    def _fail_protocol(self, error: RuntimeError) -> None:
        with self._sessions_lock:
            if self._protocol_error is None:
                self._protocol_error = error
        logger.error("Native receiver entered fail-stop protocol state: %s", error)

    def _get_session(self, unique_rid: Optional[int]) -> Optional["RxSession"]:
        with self._sessions_lock:
            session_ref = self._sessions.get(unique_rid)
        if session_ref is None:
            return None
        session = session_ref()
        if session is None:
            logger.warning(f"RxSession {unique_rid} has been garbage collected")
            return None
        return session

    def _build_recv_req_info(self, task: KVRecvTask) -> RecvReqInfo:
        self_ri = self._registrar.self_rank_info
        assert task._params.ctx_request_id is not None, (
            f"ctx_request_id is None for task unique_rid={task._unique_rid}"
        )
        assert task._unique_rid is not None, "KVRecvTask unique_rid is None"
        # Receiver's cached prefix is implicit in block_ids size; sender derives dst_start.
        return RecvReqInfo(
            sender_req_id=task._params.ctx_request_id,
            instance_name=self_ri.instance_name,
            instance_rank=self_ri.instance_rank,
            block_ids_per_layer_groups=task._kv_slice.block_ids_per_layer_groups,
            unique_rid=task._unique_rid,
            dst_start_token=None,
            aux_slot=task._aux_slot,
            mamba_state_index=task._kv_slice.mamba_state_index,
            slice_id=task.slice_id,
        )

    def _build_bounce_destination_plan(
        self, receiver_req: RecvReqInfo, peer_ri: RankInfo
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive this sender rank's exact local destination sequence.

        The receiver-side registrar builds the reverse KV mapping (local GEN
        cache to the peer CTX rank). Its source fragments are therefore the
        exact local byte ranges that the same peer writes in the forward
        direction. Dummy peer slots provide only matching cardinality; their
        addresses do not affect the returned local source fragments.
        """
        extractor = self._registrar.self_extractor
        peer_extractor = self._registrar.peer_extractor(
            peer_ri.instance_name, peer_ri.instance_rank
        )
        dst_parts: list[np.ndarray] = []
        size_specs: list[tuple[int, int]] = []
        for (self_lg, self_pi), (peer_lg, peer_pi) in self._registrar.get_pool_mapping(
            peer_ri
        ).items():
            block_ids = np.asarray(receiver_req.block_ids_per_layer_groups[self_lg], dtype=np.int64)
            block_ids = block_ids[block_ids >= 0]
            if block_ids.size == 0:
                continue
            self_region = extractor.extract(block_ids, layer_group_id=self_lg, pool_idx=self_pi)
            peer_region = peer_extractor.extract(
                np.zeros(block_ids.size, dtype=np.int64),
                layer_group_id=peer_lg,
                pool_idx=peer_pi,
            )
            mapper = self._registrar.get_kv_map(peer_ri, (self_lg, self_pi), (peer_lg, peer_pi))
            region_pair = mapper.map(self_region, peer_region)
            region_pairs = region_pair if isinstance(region_pair, list) else [region_pair]
            for pair in region_pairs:
                dst_parts.append(pair.src.memory.ptrs)
                size_specs.append((pair.src.memory.ptrs.size, pair.src.memory.bytes_per_region))

        if not dst_parts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        dst_ptrs = np.concatenate(dst_parts)
        counts, values = zip(*size_specs)
        sizes = np.repeat(np.array(values, dtype=np.int64), counts)
        order = sorted(range(dst_ptrs.size), key=lambda index: int(dst_ptrs[index]))
        return (
            np.asarray([dst_ptrs[index] for index in order], dtype=np.int64),
            np.asarray([sizes[index] for index in order], dtype=np.int64),
        )

    @staticmethod
    def _fanin_bounce_safe(overlap, peer_ri) -> bool:
        """Whether multi-writer bounce's equal total//num_writers split is valid for this overlap.
        The split assumes every writer contributes the same size, which holds when:
          * duplicate_head_factor == 1 -- else some ranks don't send KV (should_send_kv) yet still
            count in expected_transfers, so the live writers overflow their slots;
          * the PP layer split is even -- a single PP stage (overlap_pp_size <= 1) is trivially fine;
            for PP fan-in, every overlapping stage must hold the same number of layers
            (peer_ri.layer_num_per_pp all-equal) or per-writer sizes differ. If that full per-stage
            list isn't available (shorter than the fan-in degree), be conservative and fall back.
        Otherwise fall back to the per-fragment path (correct, just not coalesced).
        Equal layer count means equal bytes only when the per-block sizes match; reserve() rejects
        the mismatched case, so this only needs the count to split evenly."""
        if overlap.duplicate_head_factor != 1:
            return False
        if overlap.overlap_pp_size > 1:
            lpp = getattr(peer_ri, "layer_num_per_pp", None)
            if not lpp or len(lpp) < overlap.overlap_pp_size or len(set(lpp)) != 1:
                return False
        return True

    def dispatch_task(self, task: KVRecvTask):
        params = task._params
        logger.debug(
            f"Receiver.dispatch_task: unique_rid={task._unique_rid}, ctx_dp_rank={params.ctx_dp_rank}"
        )
        receiver_req = self._build_recv_req_info(task)
        sender_dp_rank = params.ctx_dp_rank
        peer_infos: RankInfo = self._get_sender_info(params)

        if sender_dp_rank is None and _requires_drain_ack_protocol():
            raise RuntimeError(
                "asynchronous native-transfer consensus does not support ADP "
                "broadcast because no single writer cohort is known before dispatch"
            )

        if sender_dp_rank is not None:
            # Normal path: ctx_dp_rank is known, send to overlapping ranks.
            peer_overlap = self._registrar.get_peer_overlap(peer_infos, sender_dp_rank)
        else:
            # Gen-first with ADP: ctx_dp_rank unknown — broadcast REQUEST_DATA
            # to ALL ctx sender ranks so every DP group receives it.
            # get_peer_overlap returns ranks for one DP group (topology is
            # symmetric), so use dp_rank=0 as representative.
            dp_size = peer_infos.dp_size
            dp0_overlap = self._registrar.get_peer_overlap(peer_infos, 0)
            # Union of overlapping ranks across all DP groups for broadcast (deduplicated)
            all_ranks_set: set[int] = set(dp0_overlap.ranks)
            for dp in range(1, dp_size):
                all_ranks_set.update(self._registrar.get_peer_overlap(peer_infos, dp).ranks)
            all_ranks = list(all_ranks_set)
            logger.debug(
                f"Receiver.dispatch_task: ADP broadcast path, dp_size={dp_size}, "
                f"all_ranks={all_ranks}"
            )
            peer_overlap = type(dp0_overlap)(ranks=all_ranks)

        # In gen-first ADP broadcast, peer_overlap contains the union of all DP
        # groups, but expected_transfers should reflect per-DP-group count since
        # only one DP group will actually process the context request.
        if sender_dp_rank is not None:
            task.expected_transfers = len(peer_overlap.ranks)
        else:
            task.expected_transfers = len(dp0_overlap.ranks)
        # TP fan-in splits ONE region equally, so allow it only for a uniform writer set:
        # _fanin_bounce_safe() (TP-by-head / even-PP), and never under ADP broadcast (sender_dp_rank
        # None), where the real writer count exceeds expected_transfers and would overflow the slot.
        topo_overlap = peer_overlap if sender_dp_rank is not None else dp0_overlap
        allow_bounce = task.expected_transfers == 1 or (
            sender_dp_rank is not None and self._fanin_bounce_safe(topo_overlap, peer_infos)
        )
        expected_destination_plans = None
        if allow_bounce and self._bounce.enabled:
            try:
                expected_destination_plans = {
                    rank: self._build_bounce_destination_plan(
                        receiver_req,
                        self._registrar.get_peer_rank_info(peer_infos.instance_name, rank),
                    )
                    for rank in peer_overlap.ranks
                }
            except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
                logger.warning(
                    "KV bounce cannot derive an exact receiver-owned destination plan; "
                    "using per-fragment transfer: %s",
                    error,
                )
                allow_bounce = False
        bounced = allow_bounce and self._bounce.reserve(
            receiver_req,
            task.expected_transfers,
            expected_destination_plans=expected_destination_plans,
        )
        session = self._get_session(task._unique_rid)
        if session is None:
            self._bounce.release_idle_reservation((receiver_req.unique_rid, receiver_req.slice_id))
            raise RuntimeError(
                f"dispatch_task: RxSession {task._unique_rid} not found; "
                "session may have been closed before dispatch"
            )
        endpoint_by_rank = {rank: peer_infos.sender_endpoints[rank] for rank in peer_overlap.ranks}
        sender_endpoints = set(endpoint_by_rank.values())
        # Do not mix ACK-authoritative and legacy result-authoritative cleanup
        # inside one fan-out. A rolling deployment therefore stays entirely
        # on the legacy path until every participating endpoint advertises
        # drain ACK support.
        ack_capable_endpoints = (
            sender_endpoints
            if sender_endpoints
            and all(
                _supports_native_protocol_v2(
                    self._sender_endpoint_capabilities.get(endpoint, _LEGACY_PROTOCOL_CAPABILITIES)
                )
                for endpoint in sender_endpoints
            )
            else set()
        )
        v2_enabled = bool(ack_capable_endpoints) and sender_dp_rank is not None
        receiver_req.request_epoch = session.request_epoch if v2_enabled else None
        allowed_rank_cohorts = None
        if sender_dp_rank is None:
            allowed_rank_cohorts = tuple(
                frozenset(self._registrar.get_peer_overlap(peer_infos, dp).ranks)
                for dp in range(peer_infos.dp_size)
            )
        if not session.mark_transferring(
            task.slice_id,
            endpoint_by_rank,
            ack_capable_endpoints,
            request_epoch=receiver_req.request_epoch,
            bounced=bounced,
            allowed_rank_cohorts=allowed_rank_cohorts,
        ):
            # A terminal path sealed the session after receive() reserved the
            # task but before any REQUEST_DATA message was sent.
            self._bounce.release_idle_reservation((receiver_req.unique_rid, receiver_req.slice_id))
            if not task.is_done:
                task.fail(
                    RuntimeError(
                        f"RxSession {receiver_req.unique_rid} sealed before transfer started"
                    )
                )
            return
        # Fan-in: each sender gets its own sub-region base (writers must not overwrite); else serialize once.
        fanin_bounce = bounced and task.expected_transfers > 1
        key = (receiver_req.unique_rid, receiver_req.slice_id)
        receiver_req_bytes = None
        sent_ranks: set[int] = set()
        try:
            if bounced:
                # Install the lifetime callback before advertising any address.
                # Result callbacks are too late for cancel-before-result and
                # malformed-result paths; both must still retire this slice's
                # settlement credit after exact drain proof.
                self._bounce.set_completion_callback(
                    key, session._make_bounce_settlement_callback(task)
                )
            if not fanin_bounce:
                if bounced:
                    only_rank = peer_overlap.ranks[0]
                    receiver_req.bounce_dst_base = self._bounce.bind_writer(key, only_rank, 0)
                receiver_req_bytes = receiver_req.to_bytes()
            for i, rank in enumerate(peer_overlap.ranks):
                if task._perf_timer is not None:
                    task._perf_timer.record_task_start(rank)
                if fanin_bounce:
                    receiver_req.bounce_dst_base = self._bounce.bind_writer(key, rank, i)
                    receiver_req_bytes = receiver_req.to_bytes()
                endpoint = endpoint_by_rank[rank]
                assert receiver_req_bytes is not None
                self._request_sender_data(endpoint, receiver_req_bytes)
                sent_ranks.add(rank)
        except Exception as error:
            logger.error(
                "REQUEST_DATA fan-out failed for request %s slice=%s after %s/%s endpoint(s): %s",
                receiver_req.unique_rid,
                receiver_req.slice_id,
                len(sent_ranks),
                len(sender_endpoints),
                error,
            )
            (
                cancel_endpoints,
                cancel_operations,
                orphaned_bounces,
            ) = session.fail_partial_dispatch(
                task.slice_id,
                sent_ranks,
                error,
            )
            for orphaned_key in orphaned_bounces:
                try:
                    if orphaned_key == key and not sent_ranks:
                        self._bounce.release_idle_reservation(orphaned_key)
                        session._retire_bounce_settlement(task.slice_id)
                    else:
                        self._bounce.orphan_reservation(orphaned_key)
                except Exception as bounce_error:
                    logger.error(
                        "Failed to retain partial-dispatch bounce reservation %s: %s",
                        orphaned_key,
                        bounce_error,
                    )
            if cancel_endpoints:
                self.send_cancel_to_senders(
                    receiver_req.unique_rid,
                    cancel_endpoints,
                    cancel_operations,
                )
        return

    @staticmethod
    def _extract_info_endpoint(params: DisaggregatedParams) -> Optional[str]:
        ep = params.ctx_info_endpoint
        if isinstance(ep, list):
            return ep[0] if ep else None
        return ep  # str (backward compat)

    def _should_register_peer(self, params: DisaggregatedParams) -> bool:
        endpoint = self._extract_info_endpoint(params)
        return endpoint not in self._sender_ep_instance_map

    def _send_control_message(
        self,
        endpoint: Optional[str],
        message: list[bytes],
        *,
        retry: bool = False,
        wait: bool = True,
        on_sent: Optional[Callable[[], None]] = None,
        repeat_until: Optional[Callable[[], bool]] = None,
    ) -> _ControlSend:
        return self._control.send(
            endpoint,
            message,
            retry=retry,
            wait=wait,
            on_sent=on_sent,
            repeat_until=repeat_until,
        )

    def _get_sender_info(self, params: DisaggregatedParams) -> RankInfo:
        info_endpoint = self._extract_info_endpoint(params)
        if self._should_register_peer(params):
            logger.info(f"Registering peer in first request to endpoint '{info_endpoint}'")
            messenger = ZMQMessenger(mode="DEALER", endpoint=info_endpoint)
            try:
                messenger.send([MessageType.REQUEST_INSTANCE_INFO])
                message = messenger.receive()
                sender_info = RankInfo.from_bytes(message[0])
                capabilities = _decode_protocol_capabilities(
                    message[1] if len(message) > 1 else None
                )
            finally:
                messenger.stop()

            if _requires_drain_ack_protocol() and not _supports_native_protocol_v2(capabilities):
                raise RuntimeError(
                    "asynchronous Python-transceiver consensus requires native-transfer "
                    f"protocol version {_NATIVE_PROTOCOL_VERSION}, but the context peer at "
                    f"{info_endpoint} advertised legacy protocol version {capabilities.version}"
                )

            for endpoint in sender_info.sender_endpoints:
                rank_info = self._registrar.self_rank_info
                self._send_control_message(
                    endpoint,
                    [
                        MessageType.REGISTER_RANK_INFO,
                        rank_info.to_bytes(),
                        _encode_protocol_capabilities(),
                    ],
                )
                self._sender_endpoint_capabilities[endpoint] = capabilities

            self._sender_ep_instance_map[info_endpoint] = sender_info
            self._sender_info_capabilities[info_endpoint] = capabilities
            return sender_info

        else:
            return self._sender_ep_instance_map[info_endpoint]

    def send_cancel_to_senders(
        self,
        unique_rid: int,
        sender_endpoints: set[str],
        cancel_operations: set[tuple[str, int]],
    ) -> None:
        """Notify senders and retransmit v2 CANCEL until its exact ACK arrives."""
        errors: list[Exception] = []
        for endpoint in sender_endpoints:
            try:
                message = [MessageType.CANCEL_SESSION, str(unique_rid).encode("ascii")]
                epochs = {
                    epoch
                    for operation_endpoint, epoch in cancel_operations
                    if operation_endpoint == endpoint
                }
                if len(epochs) > 1:
                    raise RuntimeError(
                        f"multiple active request epochs for rid={unique_rid} endpoint={endpoint}"
                    )
                if epochs:
                    request_epoch = next(iter(epochs))
                    message.extend(
                        [
                            self.endpoint.encode("utf-8"),
                            str(request_epoch).encode("ascii"),
                        ]
                    )

                    def acked(
                        endpoint=endpoint,
                        request_epoch=request_epoch,
                        unique_rid=unique_rid,
                    ) -> bool:
                        session = self._get_session(unique_rid)
                        return session is None or not session.is_cancel_pending(
                            endpoint, request_epoch
                        )

                    self._send_control_message(
                        endpoint,
                        message,
                        retry=True,
                        wait=False,
                        repeat_until=acked,
                    )
                else:
                    self._send_control_message(endpoint, message, retry=True, wait=False)
            except Exception as e:
                errors.append(e)
                logger.error(f"send_cancel_to_senders: failed for rid={unique_rid}: {e}")
        if errors:
            error = RuntimeError(
                f"failed to enqueue cancellation for request {unique_rid}; retaining transfer state"
            )
            self._fail_protocol(error)
            raise error from errors[0]

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.KV_AGENT_RESULT:
                    try:
                        self._process_kv_agent_result(send_id, msg)
                    except Exception as e:
                        logger.error(f"Receiver: error handling KV_AGENT_RESULT: {e}")
                case MessageType.AUX_AGENT_RESULT:
                    try:
                        self._process_aux_agent_result(send_id, msg)
                    except Exception as e:
                        logger.error(f"Receiver: error handling AUX_AGENT_RESULT: {e}")
                case MessageType.CANCEL_SESSION:
                    try:
                        self._handle_cancel_session(msg)
                    except Exception as e:
                        logger.error(f"Receiver: error handling CANCEL_SESSION: {e}")
                case MessageType.CANCEL_SESSION_ACK:
                    try:
                        self._handle_cancel_session_ack(msg)
                    except Exception as e:
                        logger.error(f"Receiver: error handling CANCEL_SESSION_ACK: {e}")
                case _:
                    logger.error(f"Receiver received unknown message type: {msg[0]}")
            return True

        self._messenger.start_listener(handle_message)

    def _handle_cancel_session(self, message: list[bytes]):
        unique_rid = int(message[1])
        sender_endpoint = message[2].decode("utf-8") if len(message) > 2 else None
        request_epoch = int(message[3]) if len(message) > 3 else None
        session = None
        with self._sessions_lock:
            session_ref = self._sessions.get(unique_rid)
            if session_ref is None:
                if unique_rid not in self._closed_rids:
                    if len(self._pre_cancelled_rids) >= _LIVE_PROTOCOL_STATE_LIMIT:
                        error = RuntimeError(
                            "Receiver pre-cancel state reached its safety limit; "
                            "refusing to evict live state"
                        )
                        self._protocol_error = error
                        raise error
                    self._pre_cancelled_rids[unique_rid] = None
            else:
                session = session_ref()
                if session is None:
                    if unique_rid not in self._closed_rids:
                        if len(self._pre_cancelled_rids) >= _LIVE_PROTOCOL_STATE_LIMIT:
                            error = RuntimeError(
                                "Receiver pre-cancel state reached its safety limit; "
                                "refusing to evict live state"
                            )
                            self._protocol_error = error
                            raise error
                        self._pre_cancelled_rids[unique_rid] = None
        if session is not None:
            if request_epoch is not None:
                session.validate_remote_cancel(sender_endpoint, request_epoch)
            session.cancel()

    def _handle_cancel_session_ack(self, message: list[bytes]) -> None:
        unique_rid = int(message[1])
        sender_endpoint = message[2].decode("utf-8")
        request_epoch = int(message[3]) if len(message) > 3 else None
        session = self._get_session(unique_rid)
        if session is None:
            if request_epoch is not None:
                with self._sessions_lock:
                    if (unique_rid, request_epoch) in self._closed_operations:
                        return
                self._fail_protocol(
                    RuntimeError(
                        f"cancel ACK for unknown request incarnation rid={unique_rid} "
                        f"epoch={request_epoch}"
                    )
                )
            return
        session.process_cancel_ack(sender_endpoint, request_epoch)

    def _process_kv_agent_result(self, _send_id: bytes, message: list[bytes]):
        if message[0] != MessageType.KV_AGENT_RESULT:
            logger.error(
                f"_process_kv_agent_result: unexpected msg_type={message[0]!r}, expected KV_AGENT_RESULT"
            )
            return
        sender_endpoint = None
        request_epoch = None
        tail_index = 2
        if len(message[1]) == _KV_RESULT_PREFIX_V2.size:
            (
                peer_rank,
                unique_rid,
                request_epoch,
                sender_slice_id,
                is_last_slice,
                status_code,
            ) = _KV_RESULT_PREFIX_V2.unpack(message[1])
            if len(message) < 3:
                raise RuntimeError("v2 KV result is missing sender endpoint")
            sender_endpoint = message[2].decode("utf-8")
            tail_index = 3
        elif len(message[1]) == _KV_RESULT_PREFIX.size:
            peer_rank, unique_rid, sender_slice_id, is_last_slice, status_code = (
                _KV_RESULT_PREFIX.unpack(message[1])
            )
        else:
            raise RuntimeError(f"invalid KV result prefix size {len(message[1])}")
        from .bounce import decode_result_tail

        dst_ptrs, sizes, src_base = decode_result_tail(message, tail_index=tail_index)
        session = self._get_session(unique_rid)
        if session is None:
            if request_epoch is not None:
                with self._sessions_lock:
                    if (unique_rid, request_epoch) in self._closed_operations:
                        return
                self._fail_protocol(
                    RuntimeError(
                        f"KV result for unknown request incarnation rid={unique_rid} "
                        f"epoch={request_epoch}"
                    )
                )
            logger.warning(
                f"_process_kv_agent_result: session {unique_rid} not found (already closed?), dropping status"
            )
            return
        session.process_kv_agent_result(
            peer_rank,
            sender_slice_id,
            is_last_slice,
            _AGENT_RESULT_BY_CODE[status_code],
            dst_ptrs=dst_ptrs,
            sizes=sizes,
            src_base=src_base,
            sender_endpoint=sender_endpoint,
            request_epoch=request_epoch,
        )

    def _process_aux_agent_result(self, _send_id: bytes, message: list[bytes]):
        if len(message) not in (4, 6):
            raise RuntimeError(f"invalid AUX result frame count {len(message)}")
        peer_rank = int(message[1])
        unique_rid = int(message[2])
        status = message[3].decode("utf-8")
        sender_endpoint = message[4].decode("utf-8") if len(message) == 6 else None
        request_epoch = int(message[5]) if len(message) == 6 else None
        session = self._get_session(unique_rid)
        if session is None:
            if request_epoch is not None:
                with self._sessions_lock:
                    if (unique_rid, request_epoch) in self._closed_operations:
                        return
                self._fail_protocol(
                    RuntimeError(
                        f"AUX result for unknown request incarnation rid={unique_rid} "
                        f"epoch={request_epoch}"
                    )
                )
            logger.warning(
                f"_process_aux_agent_result: session {unique_rid} not found (already closed?), dropping status"
            )
            return
        session.process_aux_agent_result(
            peer_rank,
            AgentResult(status),
            sender_endpoint=sender_endpoint,
            request_epoch=request_epoch,
        )

    def _request_sender_data(self, endpoint: str, receiver_info_bytes: bytes):
        # receiver_info serialized once and reused for every peer rank (block-table msgpack isn't free at fan-out).
        logger.debug("Sending data request to endpoint '%s'", endpoint)
        self._send_control_message(endpoint, [MessageType.REQUEST_DATA, receiver_info_bytes])

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Receiver.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()


class RxSession(RxSessionBase):
    def __init__(
        self,
        request_id: int,
        params: DisaggregatedParams,
        receiver: Receiver,
        aux_buffer: Optional[AuxBuffer] = None,
        timeout_s: Optional[float] = None,
        prompt_len: Optional[int] = None,
        beam_width: int = 1,
    ):
        super().__init__(
            receiver,
            SessionArgsBase(params, prompt_len=prompt_len, beam_width=beam_width),
        )
        self._timeout_s = timeout_s
        self._need_aux = params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        self._receiver: Receiver  # narrow base class type for Pylance
        self.request_id = request_id
        self._aux_buffer = aux_buffer
        self.aux_slot = aux_buffer.alloc_slot().id if aux_buffer is not None else None
        self._exception: Optional[Exception] = None
        self._closed = False
        self._terminal_status: Optional[SessionStatus] = None
        self._kv_tasks: list[KVRecvTask] = []
        self._aux_count = 0
        self._aux_status: TaskStatus = TaskStatus.INIT
        self._sender_endpoints: set[str] = set()
        self.request_epoch = secrets.randbits(63) or 1
        self.lock = threading.Lock()
        self._sealed = False
        self._outstanding_operations = 0
        self._expected_operations: dict[tuple, _ExpectedReceiveOperation] = {}
        self._retired_operation_keys: set[tuple] = set()
        self._pending_scatter_callbacks = 0
        self._pending_bounce_settlements: set[int] = set()
        self._aux_obligations_reserved = False
        self._aux_obligation_slice_id: Optional[int] = None
        self._cancel_pending_endpoints: set[str] = set()
        self._cancel_pending_operations: set[tuple[str, int]] = set()
        self._ack_capable_endpoints: set[str] = set()
        self._v2_enabled: Optional[bool] = None
        self._legacy_rank_cohorts: dict[int, tuple[frozenset[int], ...]] = {}
        self._legacy_bound_cohorts: dict[int, frozenset[int]] = {}
        self._close_requested = False
        self._receiver.setup_session(self)

    @property
    def disagg_request_id(self) -> int:
        params = self._base_args.params
        if params.disagg_request_id is not None:
            return params.disagg_request_id
        # ctx_request_id is set on gen-side requests to the ctx server's request ID,
        # which matches the key the ctx TxSession registered under.  Fall back to
        # the local request_id only when neither field is available.
        if params.ctx_request_id is not None:
            return params.ctx_request_id
        return self.request_id

    @property
    def status(self) -> SessionStatus:
        if self._terminal_status is not None:
            return self._terminal_status
        if self._exception is not None or any(t.status == TaskStatus.ERROR for t in self._kv_tasks):
            return SessionStatus.ERROR
        if self._kv_tasks:
            kv_all_transferred = all(t.status == TaskStatus.TRANSFERRED for t in self._kv_tasks)
            if kv_all_transferred and self._aux_status == TaskStatus.TRANSFERRED:
                return SessionStatus.FULLY_TRANSFERRED
            if kv_all_transferred:
                return SessionStatus.KV_TRANSFERRED
            if any(t.status == TaskStatus.TRANSFERRING for t in self._kv_tasks):
                return SessionStatus.TRANSFERRING
        return SessionStatus.INIT

    def mark_transferring(
        self,
        slice_id: int,
        endpoint_by_rank: dict[int, str] | set[str],
        ack_capable_endpoints: Optional[set[str]] = None,
        *,
        request_epoch: Optional[int] = None,
        bounced: bool = False,
        allowed_rank_cohorts: Optional[tuple[frozenset[int], ...]] = None,
    ) -> bool:
        """Start a receive task and bind every advertised remote operation."""
        with self.lock:
            task = self._kv_tasks[slice_id]
            if (
                self._sealed
                or self._closed
                or self._terminal_status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
                or task.status != TaskStatus.INIT
            ):
                return False
            if isinstance(endpoint_by_rank, set):
                # Compatibility for legacy callers/tests that predate rank
                # identity binding. Production dispatch always passes a map.
                sender_endpoints = set(endpoint_by_rank)
                rank_endpoints: dict[int, str] = {}
            else:
                rank_endpoints = dict(endpoint_by_rank)
                sender_endpoints = set(rank_endpoints.values())
            v2_enabled = request_epoch is not None
            if self._v2_enabled is None:
                self._v2_enabled = v2_enabled
            elif self._v2_enabled != v2_enabled:
                raise RuntimeError(
                    f"RxSession {self.disagg_request_id} cannot mix native protocol versions"
                )
            prior_all_ack_capable = (
                not self._sender_endpoints or self._ack_capable_endpoints == self._sender_endpoints
            )
            this_all_ack_capable = bool(sender_endpoints) and (
                ack_capable_endpoints == sender_endpoints
            )
            self._sender_endpoints.update(sender_endpoints)
            if prior_all_ack_capable and this_all_ack_capable:
                self._ack_capable_endpoints.update(sender_endpoints)
            else:
                # A mixed-version session must use legacy terminal results for
                # every peer; otherwise one ACK could retire another peer's
                # still-active write.
                self._ack_capable_endpoints.clear()
            task.status = TaskStatus.TRANSFERRING
            if allowed_rank_cohorts is not None:
                self._legacy_rank_cohorts[slice_id] = allowed_rank_cohorts
                # The concrete ADP cohort is selected by the first terminal
                # result. Credits remain count-based until that immutable bind.
                self._outstanding_operations += task.expected_transfers
            else:
                for rank, endpoint in rank_endpoints.items():
                    operation_key = ("kv", slice_id, rank)
                    if operation_key in self._expected_operations:
                        raise RuntimeError(
                            f"duplicate receive operation {operation_key} for "
                            f"request {self.disagg_request_id}"
                        )
                    self._expected_operations[operation_key] = _ExpectedReceiveOperation(
                        endpoint, request_epoch
                    )
                self._outstanding_operations += (
                    len(rank_endpoints) if rank_endpoints else task.expected_transfers
                )
            if self._need_aux and not self._aux_obligations_reserved:
                if allowed_rank_cohorts is None:
                    for rank, endpoint in rank_endpoints.items():
                        self._expected_operations[("aux", rank)] = _ExpectedReceiveOperation(
                            endpoint, request_epoch
                        )
                    self._outstanding_operations += (
                        len(rank_endpoints) if rank_endpoints else task.expected_transfers
                    )
                else:
                    self._outstanding_operations += task.expected_transfers
                self._aux_obligations_reserved = True
                self._aux_obligation_slice_id = slice_id
            if bounced:
                self._pending_bounce_settlements.add(slice_id)
                self._pending_scatter_callbacks = len(self._pending_bounce_settlements)
            return True

    def _bind_legacy_cohort_unlocked(self, slice_id: int, peer_rank: int) -> None:
        cohorts = self._legacy_rank_cohorts.get(slice_id)
        if cohorts is None:
            return
        bound = self._legacy_bound_cohorts.get(slice_id)
        if bound is None:
            matches = [cohort for cohort in cohorts if peer_rank in cohort]
            if len(matches) != 1:
                raise RuntimeError(
                    f"rank {peer_rank} does not identify one allowed ADP cohort "
                    f"for request {self.disagg_request_id} slice={slice_id}"
                )
            bound = matches[0]
            other_bound_cohorts = set(self._legacy_bound_cohorts.values())
            if other_bound_cohorts and other_bound_cohorts != {bound}:
                raise RuntimeError(
                    f"rank {peer_rank} selects a different ADP writer cohort "
                    f"for request {self.disagg_request_id} slice={slice_id}"
                )
            self._legacy_bound_cohorts[slice_id] = bound
        if peer_rank not in bound:
            raise RuntimeError(
                f"rank {peer_rank} is outside the bound ADP writer cohort "
                f"for request {self.disagg_request_id} slice={slice_id}"
            )

    def _validate_receive_operation_unlocked(
        self,
        operation_key: tuple,
        *,
        sender_endpoint: Optional[str],
        request_epoch: Optional[int],
    ) -> str:
        """Return ``accept``/``duplicate`` or raise without retiring credit."""
        if operation_key in self._retired_operation_keys:
            return "duplicate"
        kind = operation_key[0]
        if kind == "kv" and operation_key[1] in self._legacy_rank_cohorts:
            self._bind_legacy_cohort_unlocked(operation_key[1], operation_key[2])
            return "accept"
        if kind == "aux" and self._legacy_rank_cohorts:
            # Aux may arrive before KV. Bind it through the same immutable
            # cohort selection instead of accepting it count-only and letting
            # a later KV result select a different DP group.
            if self._aux_obligation_slice_id is None:
                raise RuntimeError(
                    f"aux result has no owning KV slice for request {self.disagg_request_id}"
                )
            self._bind_legacy_cohort_unlocked(self._aux_obligation_slice_id, operation_key[1])
            return "accept"
        expected = self._expected_operations.get(operation_key)
        if expected is None:
            # Only compatibility/unit-test sessions created without production
            # rank maps may use count-based legacy retirement.
            if not self._expected_operations and request_epoch is None:
                return "accept"
            raise RuntimeError(
                f"unexpected native-transfer result operation {operation_key} "
                f"for request {self.disagg_request_id}"
            )
        if expected.request_epoch != request_epoch:
            raise RuntimeError(
                f"native-transfer result epoch mismatch for {operation_key}: "
                f"expected {expected.request_epoch}, got {request_epoch}"
            )
        if request_epoch is not None and expected.sender_endpoint != sender_endpoint:
            raise RuntimeError(
                f"native-transfer result source mismatch for {operation_key}: "
                f"expected {expected.sender_endpoint}, got {sender_endpoint}"
            )
        return "accept"

    def _fail_protocol_unlocked(self, error: RuntimeError) -> set[tuple[str, int]]:
        self._sealed = True
        self._exception = error
        if self._terminal_status != SessionStatus.CANCELLED:
            self._terminal_status = SessionStatus.ERROR
        targets = {
            (expected.sender_endpoint, expected.request_epoch)
            for key, expected in self._expected_operations.items()
            if key not in self._retired_operation_keys and expected.request_epoch is not None
        }
        self._cancel_pending_operations.update(targets)
        self._cancel_pending_endpoints.update(endpoint for endpoint, _epoch in targets)
        return targets

    def fail_partial_dispatch(
        self,
        slice_id: int,
        sent_ranks: set[int],
        error: Exception,
    ) -> tuple[set[str], set[tuple[str, int]], list[tuple[int, int]]]:
        """Seal a partially published fan-out without inventing remote quiescence.

        Endpoints which never received REQUEST_DATA cannot write, so their
        reserved receive credits are retired locally. Endpoints which did
        receive it remain protected by either the negotiated drain ACK or
        their legacy terminal result. In the ambiguous legacy ADP-broadcast
        case, cleanup deliberately remains retained rather than guessing.
        """
        with self.lock:
            task = self._kv_tasks[slice_id]
            self._sealed = True
            if self._terminal_status != SessionStatus.CANCELLED:
                self._terminal_status = SessionStatus.ERROR
            self._exception = RuntimeError(
                f"REQUEST_DATA fan-out failed for request {self.disagg_request_id}: {error}"
            )
            task.fail(self._exception)

            if self._expected_operations:
                never_dispatched_keys = {
                    key
                    for key in self._expected_operations
                    if (key[0] == "kv" and key[1] == slice_id and key[2] not in sent_ranks)
                    or (
                        key[0] == "aux"
                        and self._aux_obligation_slice_id == slice_id
                        and key[1] not in sent_ranks
                    )
                }
                for operation_key in never_dispatched_keys:
                    self._retire_receive_operation_unlocked(operation_key)
                sent_operations = {
                    expected
                    for key, expected in self._expected_operations.items()
                    if key not in self._retired_operation_keys
                    and (
                        (key[0] == "kv" and key[1] == slice_id and key[2] in sent_ranks)
                        or (
                            key[0] == "aux"
                            and self._aux_obligation_slice_id == slice_id
                            and key[1] in sent_ranks
                        )
                    )
                }
                cancel_endpoints = {operation.sender_endpoint for operation in sent_operations}
                cancel_operations = {
                    (operation.sender_endpoint, operation.request_epoch)
                    for operation in sent_operations
                    if operation.request_epoch is not None
                }
            else:
                max_remote_writers = min(task.expected_transfers, len(sent_ranks))
                never_dispatched = task.expected_transfers - max_remote_writers
                retired = never_dispatched
                if self._need_aux and self._aux_obligation_slice_id == slice_id:
                    retired += never_dispatched
                if retired > self._outstanding_operations:
                    raise RuntimeError(
                        f"RxSession {self.disagg_request_id} partial-dispatch credit underflow"
                    )
                self._outstanding_operations -= retired
                cancel_endpoints = set(self._sender_endpoints)
                cancel_operations = set()
            self._cancel_pending_operations.update(cancel_operations)
            self._cancel_pending_endpoints.update(
                endpoint for endpoint, _epoch in cancel_operations
            )
            orphaned = [
                (self.disagg_request_id, pending_task.slice_id)
                for pending_task in self._kv_tasks
                if pending_task.status == TaskStatus.TRANSFERRING or pending_task is task
            ]
            return cancel_endpoints, cancel_operations, orphaned

    def receive(self, slice: KVSlice) -> None:
        with self.lock:
            params = self._base_args.params
            slice_id = len(self._kv_tasks)
            task = KVRecvTask(
                self.disagg_request_id,
                slice,
                slice_id,
                params,
                aux_slot=self.aux_slot,
            )
            self._kv_tasks.append(task)
            if self._sealed or self._closed:
                task.fail(
                    RuntimeError(f"RxSession {self.disagg_request_id} is sealed; receive rejected")
                )
                return
        self._receiver.dispatch_task(task)

    def process_kv_agent_result(
        self,
        peer_rank: int,
        sender_slice_id: int,
        is_last_slice: bool,
        status: AgentResult,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        sender_endpoint: Optional[str] = None,
        request_epoch: Optional[int] = None,
    ):
        operation_key = ("kv", sender_slice_id, peer_rank)
        bounce_action: Optional[Callable[[], None]] = None
        protocol_error: Optional[RuntimeError] = None
        cancel_operations: set[tuple[str, int]] = set()
        task: Optional[KVRecvTask] = None
        with self.lock:
            try:
                validation = self._validate_receive_operation_unlocked(
                    operation_key,
                    sender_endpoint=sender_endpoint,
                    request_epoch=request_epoch,
                )
                if validation == "duplicate":
                    return
                if not 0 <= sender_slice_id < len(self._kv_tasks):
                    raise RuntimeError(
                        f"result names invalid slice {sender_slice_id}; receiver has "
                        f"{len(self._kv_tasks)} slice(s) for request {self.request_id}"
                    )
            except RuntimeError as error:
                protocol_error = error
                cancel_operations = self._fail_protocol_unlocked(error)
            if protocol_error is None:
                task = self._kv_tasks[sender_slice_id]
                rid_slice = (self.disagg_request_id, task.slice_id)
                complete_task = False
                if status == AgentResult.SUCCESS and is_last_slice:
                    task.last_slice_count += 1
                    complete_task = task.last_slice_count == task.expected_transfers
                elif status == AgentResult.FAILED:
                    detail = (
                        f"KV transfer failed for request {self.request_id} "
                        f"slice={sender_slice_id} peer_rank={peer_rank} "
                        f"is_last_slice={is_last_slice} (reported by remote agent; "
                        "see sender-side log for nixl_status)"
                    )
                    logger.error(detail)
                    if self._terminal_status != SessionStatus.CANCELLED:
                        task.fail(RuntimeError(detail))
                        self._terminal_status = SessionStatus.ERROR
                else:
                    if status != AgentResult.SUCCESS:
                        protocol_error = RuntimeError(
                            f"unknown KV result status {status!r} for request {self.request_id}"
                        )
                        cancel_operations = self._fail_protocol_unlocked(protocol_error)

                if protocol_error is None:
                    on_done = None
                    if complete_task:
                        on_done = self._make_kv_settlement_callback(
                            task,
                            peer_rank=peer_rank,
                            complete_task=complete_task,
                            settle_bounce=False,
                        )
                    if status == AgentResult.SUCCESS:
                        from .bounce import scatter_write_result

                        def record_success() -> None:
                            scatter_write_result(
                                self._receiver._bounce,
                                rid_slice,
                                peer_rank,
                                dst_ptrs,
                                sizes,
                                src_base,
                                on_done,
                            )

                        bounce_action = record_success
                    else:

                        def record_failure() -> None:
                            self._receiver._bounce.record_failure(
                                rid_slice, peer_rank, on_done=on_done
                            )

                        bounce_action = record_failure

        if protocol_error is not None:
            self._publish_protocol_failure(protocol_error, cancel_operations)
            return

        try:
            if bounce_action is not None:
                bounce_action()
        except Exception as error:
            with self.lock:
                protocol_error = RuntimeError(
                    f"bounce result validation/settlement failed for request "
                    f"{self.request_id} slice={sender_slice_id} rank={peer_rank}: {error}"
                )
                if task is not None:
                    task.fail(protocol_error)
                cancel_operations = self._fail_protocol_unlocked(protocol_error)
            self._publish_protocol_failure(protocol_error, cancel_operations)
            return

        with self.lock:
            self._retire_receive_operation_unlocked(operation_key)
            finalize_close = self._should_finalize_close_unlocked()
        if finalize_close:
            self._finalize_close()

    def process_aux_agent_result(
        self,
        peer_rank: int,
        status: AgentResult,
        *,
        sender_endpoint: Optional[str] = None,
        request_epoch: Optional[int] = None,
    ):
        # Aux is session-level (not per-slice); expected_transfers is identical
        # across all kv_tasks, so any task provides the right count.
        operation_key = ("aux", peer_rank)
        finalize_close = False
        protocol_error: Optional[RuntimeError] = None
        cancel_operations: set[tuple[str, int]] = set()
        with self.lock:
            try:
                validation = self._validate_receive_operation_unlocked(
                    operation_key,
                    sender_endpoint=sender_endpoint,
                    request_epoch=request_epoch,
                )
            except RuntimeError as error:
                protocol_error = error
                cancel_operations = self._fail_protocol_unlocked(error)
                validation = "reject"
            if validation == "duplicate":
                return
            if protocol_error is None and not self._kv_tasks:
                protocol_error = RuntimeError(
                    f"aux result arrived before KV dispatch for request {self.request_id}"
                )
                cancel_operations = self._fail_protocol_unlocked(protocol_error)
            if protocol_error is None and self._terminal_status != SessionStatus.CANCELLED:
                task = self._kv_tasks[0]
                if status == AgentResult.SUCCESS:
                    self._aux_count += 1

                    if self._aux_count == task.expected_transfers:
                        self._aux_status = TaskStatus.TRANSFERRED
                    elif self._aux_count > task.expected_transfers:
                        self._aux_status = TaskStatus.ERROR
                        self._exception = RuntimeError(
                            f"Session {self.request_id} received too many aux transfers"
                        )
                        if self._terminal_status is None:
                            self._terminal_status = SessionStatus.ERROR
                        logger.error(str(self._exception))
                elif status == AgentResult.FAILED:
                    self._aux_status = TaskStatus.ERROR
                    self._exception = RuntimeError(f"Session {self.request_id} aux transfer failed")
                    if self._terminal_status is None:
                        self._terminal_status = SessionStatus.ERROR
                else:
                    protocol_error = RuntimeError(
                        f"unknown aux result status {status!r} for request {self.request_id}"
                    )
                    cancel_operations = self._fail_protocol_unlocked(protocol_error)
            if protocol_error is None:
                self._retire_receive_operation_unlocked(operation_key)
                finalize_close = self._should_finalize_close_unlocked()
        if protocol_error is not None:
            self._publish_protocol_failure(protocol_error, cancel_operations)
            return
        if finalize_close:
            self._finalize_close()

    def _make_kv_settlement_callback(
        self,
        task: KVRecvTask,
        *,
        peer_rank: int,
        complete_task: bool,
        settle_bounce: bool,
    ) -> Callable[[bool], None]:
        request_id = self.request_id
        sender_slice_id = task.slice_id
        ri = self._receiver._registrar.self_rank_info
        instance_name, instance_rank = ri.instance_name, ri.instance_rank

        def settled(success: bool) -> None:
            try:
                if not success:
                    if task.status != TaskStatus.ERROR:
                        task.fail(
                            RuntimeError(
                                f"KV bounce settlement failed for request {request_id} "
                                f"slice={sender_slice_id}"
                            )
                        )
                    return
                if not complete_task or task.status == TaskStatus.ERROR:
                    return
                try:
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_end(peer_rank)
                    task.print_perf_info(peer_rank, instance_name, instance_rank)
                except Exception as error:
                    logger.warning(
                        f"KV transfer perf logging failed for request {request_id} "
                        f"slice={sender_slice_id}: {error}"
                    )
                task.complete()
                logger.debug(
                    f"KV transfer complete for request {request_id} slice={sender_slice_id}"
                )
            finally:
                if settle_bounce:
                    self._retire_bounce_settlement(sender_slice_id)

        return settled

    def _make_bounce_settlement_callback(self, task: KVRecvTask) -> Callable[[bool], None]:
        """Retire one slice's arena credit on every proven settlement path."""
        request_id = self.request_id
        slice_id = task.slice_id

        def settled(success: bool) -> None:
            try:
                if not success and task.status != TaskStatus.ERROR:
                    task.fail(
                        RuntimeError(
                            f"KV bounce settlement failed for request {request_id} slice={slice_id}"
                        )
                    )
            finally:
                self._retire_bounce_settlement(slice_id)

        return settled

    def _retire_bounce_settlement(self, slice_id: int) -> None:
        finalize_close = False
        with self.lock:
            if slice_id not in self._pending_bounce_settlements:
                return
            self._pending_bounce_settlements.remove(slice_id)
            self._pending_scatter_callbacks = len(self._pending_bounce_settlements)
            finalize_close = self._should_finalize_close_unlocked()
        if finalize_close:
            self._finalize_close()

    def _publish_protocol_failure(
        self,
        error: RuntimeError,
        cancel_operations: set[tuple[str, int]],
    ) -> None:
        self._receiver._fail_protocol(error)
        with self.lock:
            bounced_slices = list(self._pending_bounce_settlements)
        for slice_id in bounced_slices:
            try:
                self._receiver._bounce.orphan_reservation((self.disagg_request_id, slice_id))
            except Exception as bounce_error:
                logger.error(
                    "Failed to retain bounce reservation after protocol error for "
                    "request %s slice=%s: %s",
                    self.request_id,
                    slice_id,
                    bounce_error,
                )
        if cancel_operations:
            endpoints = {endpoint for endpoint, _epoch in cancel_operations}
            try:
                self._receiver.send_cancel_to_senders(
                    self.disagg_request_id, endpoints, cancel_operations
                )
            except Exception:
                # send_cancel_to_senders already records a receiver-wide
                # fail-stop state. Lifetime credits intentionally remain live.
                pass

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def unpack_aux(self, request: LlmRequest) -> None:
        """Read token data from the aux buffer slot into the given request."""
        assert self._aux_buffer is not None, "No aux_buffer set for this session"
        assert self.aux_slot is not None, "No aux_slot set for this session"
        first_gen_tokens, draft_tokens, (prompt_tokens, cached_tokens) = (
            self._aux_buffer.get_slot_data(self.aux_slot)
        )
        request.py_first_gen_tokens = first_gen_tokens  # type: ignore[attr-defined]
        request.py_draft_tokens = draft_tokens  # type: ignore[attr-defined]
        if request.py_disaggregated_params is not None:
            request.py_disaggregated_params.ctx_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": cached_tokens,
                },
            }

    def is_completed(self) -> bool:
        """Non-blocking check: has the transfer completed successfully?"""
        status = self.status
        if self._need_aux:
            return status == SessionStatus.FULLY_TRANSFERRED
        return status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)

    def has_failed(self) -> bool:
        """Non-blocking check: has the transfer failed or been cancelled?"""
        return self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)

    def _retire_receive_operation_unlocked(self, operation_key: tuple) -> None:
        if operation_key in self._retired_operation_keys:
            return
        if self._outstanding_operations <= 0:
            raise RuntimeError(f"RxSession {self.disagg_request_id} operation credit underflow")
        self._retired_operation_keys.add(operation_key)
        self._outstanding_operations -= 1

    def _retire_scatter_callback(self) -> None:
        finalize_close = False
        with self.lock:
            if self._pending_scatter_callbacks <= 0:
                raise RuntimeError(f"RxSession {self.disagg_request_id} scatter credit underflow")
            self._pending_scatter_callbacks -= 1
            finalize_close = self._should_finalize_close_unlocked()
        if finalize_close:
            self._finalize_close()

    def process_cancel_ack(self, sender_endpoint: str, request_epoch: Optional[int] = None) -> None:
        finalize_close = False
        drained_bounces: list[tuple[int, int]] = []
        protocol_error: Optional[RuntimeError] = None
        with self.lock:
            if request_epoch is None:
                if self._v2_enabled:
                    protocol_error = RuntimeError(
                        f"legacy cancel ACK received for v2 request {self.disagg_request_id}"
                    )
                elif sender_endpoint not in self._cancel_pending_endpoints:
                    return
                else:
                    self._cancel_pending_endpoints.discard(sender_endpoint)
                    if (
                        self._terminal_status in (SessionStatus.CANCELLED, SessionStatus.ERROR)
                        and not self._cancel_pending_endpoints
                    ):
                        self._outstanding_operations = 0
                        drained_bounces = [
                            (self.disagg_request_id, task.slice_id) for task in self._kv_tasks
                        ]
            else:
                operation = (sender_endpoint, request_epoch)
                if operation not in self._cancel_pending_operations:
                    known = any(
                        expected.sender_endpoint == sender_endpoint
                        and expected.request_epoch == request_epoch
                        for expected in self._expected_operations.values()
                    )
                    if known:
                        return
                    protocol_error = RuntimeError(
                        "cancel ACK does not match an active request operation: "
                        f"rid={self.disagg_request_id} endpoint={sender_endpoint} "
                        f"epoch={request_epoch}"
                    )
                else:
                    self._cancel_pending_operations.discard(operation)
                    for operation_key, expected in self._expected_operations.items():
                        if (
                            operation_key not in self._retired_operation_keys
                            and expected.sender_endpoint == sender_endpoint
                            and expected.request_epoch == request_epoch
                        ):
                            self._retire_receive_operation_unlocked(operation_key)
                    if not any(
                        endpoint == sender_endpoint
                        for endpoint, _epoch in self._cancel_pending_operations
                    ):
                        self._cancel_pending_endpoints.discard(sender_endpoint)
                    for slice_id in list(self._pending_bounce_settlements):
                        has_live_writer = any(
                            key[0] == "kv"
                            and key[1] == slice_id
                            and key not in self._retired_operation_keys
                            for key in self._expected_operations
                        )
                        if not has_live_writer:
                            drained_bounces.append((self.disagg_request_id, slice_id))
            if protocol_error is not None:
                self._fail_protocol_unlocked(protocol_error)
        for rid_slice in drained_bounces:
            self._receiver._bounce.confirm_drained(rid_slice)
        if protocol_error is not None:
            self._publish_protocol_failure(protocol_error, set())
            return
        with self.lock:
            finalize_close = self._should_finalize_close_unlocked()
        if finalize_close:
            self._finalize_close()

    def is_cancel_pending(self, sender_endpoint: str, request_epoch: int) -> bool:
        with self.lock:
            return (sender_endpoint, request_epoch) in self._cancel_pending_operations

    def validate_remote_cancel(self, sender_endpoint: Optional[str], request_epoch: int) -> None:
        with self.lock:
            if any(
                expected.sender_endpoint == sender_endpoint
                and expected.request_epoch == request_epoch
                for expected in self._expected_operations.values()
            ):
                return
            error = RuntimeError(
                "remote cancel does not match an active request incarnation: "
                f"rid={self.disagg_request_id} endpoint={sender_endpoint} "
                f"epoch={request_epoch}"
            )
            self._fail_protocol_unlocked(error)
        raise error

    def _is_drained_unlocked(self) -> bool:
        return (
            self._outstanding_operations == 0
            and self._pending_scatter_callbacks == 0
            and not self._cancel_pending_endpoints
            and not self._cancel_pending_operations
        )

    def _should_finalize_close_unlocked(self) -> bool:
        return self._close_requested and not self._closed and self._is_drained_unlocked()

    def cancel(self) -> None:
        """Cancel the session and notify the remote sender.

        Safe to call multiple times. TRANSFERRING tasks keep running (mid-write).
        Only INIT tasks have their events signalled immediately.
        The lock serializes with process_kv_agent_result() / process_aux_agent_result().
        """
        sender_endpoints: set[str]
        orphaned: list[tuple[int, int]] = []
        with self.lock:
            if self._terminal_status == SessionStatus.CANCELLED:
                return
            self._sealed = True
            self._terminal_status = SessionStatus.CANCELLED
            sender_endpoints = set(self._sender_endpoints)
            cancel_operations = {
                (expected.sender_endpoint, expected.request_epoch)
                for operation_key, expected in self._expected_operations.items()
                if operation_key not in self._retired_operation_keys
                and expected.request_epoch is not None
            }
            self._cancel_pending_operations.update(cancel_operations)
            self._cancel_pending_endpoints.update(
                endpoint for endpoint, _epoch in cancel_operations
            )
            exc = RuntimeError(f"RxSession {self.disagg_request_id} cancelled")
            for task in self._kv_tasks:
                rid_slice = (self.disagg_request_id, task.slice_id)
                if task.status == TaskStatus.INIT:
                    # INIT = reserved but no write in flight, so freeing its bounce reservation here
                    # is safe.
                    self._receiver._bounce.release_idle_reservation(rid_slice)
                    task.fail(exc)
                elif task.status == TaskStatus.TRANSFERRING:
                    orphaned.append(rid_slice)
        for rid_slice in orphaned:
            self._receiver._bounce.orphan_reservation(rid_slice)
        # Send outside the lock to avoid holding it during I/O.
        self._receiver.send_cancel_to_senders(
            self.disagg_request_id,
            sender_endpoints,
            cancel_operations,
        )

    def has_transferring_tasks(self) -> bool:
        """True if any KV task is currently mid-write (TRANSFERRING).

        cancel_request() must return False while this is True.
        """
        with self.lock:
            return self._has_transferring_tasks_unlocked()

    def _has_transferring_tasks_unlocked(self) -> bool:
        return not self._is_drained_unlocked()

    def seal_and_check_quiescent(self) -> bool:
        """Atomically prevent new receive work and check whether writes drained."""
        with self.lock:
            self._sealed = True
            return not self._has_transferring_tasks_unlocked()

    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]:
        """Poll or block until transfer completes.

        With blocking=False (default): polls non-blockingly; returns None if
        any KV task or aux is not yet done — caller should re-poll next cycle.
        With blocking=True: waits up to _timeout_s for each task.
        Returns WaitResult.COMPLETED on full success, WaitResult.FAILED on error/timeout.
        """
        if not blocking:
            # Use task.status instead of task.wait(timeout=0): task.complete()
            # sets status before event, so a GIL switch between the two steps
            # can cause asymmetric TP-rank completion and an allgather deadlock.
            for task in self._kv_tasks:
                if task.status == TaskStatus.TRANSFERRED:
                    continue
                if task.status == TaskStatus.ERROR:
                    return WaitResult.FAILED
                return None  # task not yet done; re-poll next cycle
        else:
            timeout = self._timeout_s
            for task in self._kv_tasks:
                done = task.wait(timeout=timeout)
                if not done:
                    return WaitResult.FAILED  # timeout
                if task.status == TaskStatus.ERROR:
                    return WaitResult.FAILED
        if self._need_aux:
            while True:
                status = self.status
                if status == SessionStatus.FULLY_TRANSFERRED:
                    return WaitResult.COMPLETED
                elif status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                    return WaitResult.FAILED
                if not blocking:
                    return None  # KV done, aux still in flight; re-poll next cycle
                time.sleep(0.001)
        return WaitResult.COMPLETED

    def close(self):
        with self.lock:
            if self._closed:
                return
            # Publish the strong drain reference while the closed-state check
            # is still serialized with finalization. Otherwise two concurrent
            # close() calls can let the second reinsert an already-cleared
            # session into Receiver._draining_sessions forever.
            self._receiver.retain_draining_session(self)
            self._sealed = True
            self._close_requested = True
            finalize_close = self._should_finalize_close_unlocked()
        if not finalize_close:
            return
        self._finalize_close()

    def _finalize_close(self) -> None:
        with self.lock:
            if self._closed or not self._is_drained_unlocked():
                return
            self._closed = True
        if self._aux_buffer is not None and self.aux_slot is not None:
            self._aux_buffer.free_slot(self.aux_slot)
            self.aux_slot = None
        # Unregister from Receiver; keep fields alive for in-flight listener messages.
        if self._receiver is not None:
            # Reclaim any bounce region still live at teardown (closed mid-transfer) so it isn't
            # leaked; a no-op for finished or non-bounce transfers.
            for task in self._kv_tasks:
                self._receiver._bounce.orphan_reservation((self.disagg_request_id, task.slice_id))
            self._receiver.clear_session(
                self.disagg_request_id,
                self.request_epoch if self._v2_enabled else None,
            )

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.warning(f"RxSession.__del__: exception during close: {e}")


class RankInfoServer:
    def __init__(self, rank_info: RankInfo, addr: Optional[str] = None, port: Optional[int] = None):
        self._rank_info = rank_info
        self._shutdown = False  # must be set before _start_listener() so __del__ is safe
        if addr is None and port is None:
            endpoint = f"tcp://{get_local_ip()}:*"
        else:
            endpoint = f"tcp://{addr}:{port}"
        self._messenger = ZMQMessenger(mode="ROUTER", endpoint=endpoint)
        self._start_listener()

    @property
    def endpoint(self) -> str:
        return self._messenger.endpoint

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        logger.debug("RankInfoServer.shutdown() called")
        self._messenger.stop()

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.REQUEST_INSTANCE_INFO:
                    try:
                        self._handle_rank_info_request(send_id, msg)
                    except Exception as e:
                        logger.error(f"RankInfoServer: error handling REQUEST_INSTANCE_INFO: {e}")
                case _:
                    logger.error(f"Instance info server received unknown message type: {msg[0]}")
            return True

        self._messenger.start_listener(handle_message)

    def _handle_rank_info_request(self, send_id: bytes, _message: list[bytes]):
        # The extra frame is backward compatible: legacy receivers consume
        # the RankInfo frame and ignore the remainder. New receivers treat a
        # missing frame as protocol v1 and keep the legacy retention path.
        self._messenger.send([send_id, self._rank_info.to_bytes(), _encode_protocol_capabilities()])

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"RankInfoServer.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()


def _create_nixl_agent(name: str) -> NixlTransferAgent:
    num_threads = int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "8"))
    kwargs = {}
    if "TRTLLM_NIXL_SPLIT_BATCH_SIZE" in os.environ:
        kwargs["split_batch_size"] = int(os.environ["TRTLLM_NIXL_SPLIT_BATCH_SIZE"])
    return NixlTransferAgent(name, True, num_threads=num_threads, **kwargs)


def _make_aux_buffer(
    kvm: KVCacheManager, max_slots: int, max_draft_len: Optional[int] = None
) -> Optional[AuxBuffer]:
    if max_slots <= 0:
        return None
    if max_draft_len is None:
        max_draft_len = max(0, int(getattr(kvm, "max_draft_len", 0)))
    return AuxBuffer(
        max_slot_num=max_slots,
        beam_width=max(1, int(getattr(kvm, "max_beam_width", 1))),
        max_draft_len=max_draft_len,
        device="cpu",
    )


@dataclass
class TransferWorkerConfig:
    kv_cache_manager: KVCacheManager
    device_id: int
    instance_name: str
    max_concurrent_sessions: int = 0
    max_draft_len: Optional[int] = None
    tx_timeout_s: Optional[float] = None
    rx_timeout_s: Optional[float] = None
    bounce: Optional["Config"] = None


class TransferWorker:
    def __init__(self, config: TransferWorkerConfig):
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = threading.Event()
        self._shutdown_thread: Optional[threading.Thread] = None
        self._shutdown_error: Optional[BaseException] = None
        self._config = config
        kvm = config.kv_cache_manager
        self._aux_buffer = _make_aux_buffer(
            kvm, config.max_concurrent_sessions, config.max_draft_len
        )
        self._rank_info = RankInfo.from_kv_cache_manager(
            config.instance_name,
            kvm,
            config.device_id,
            self._aux_buffer.meta if self._aux_buffer is not None else None,
        )
        self._setup_peer_infrastructure(kvm)
        self._setup_transfer_engine()

    def populate_instance_and_rank_info(self, endpoints: list[str], layer_num_per_pp: list[int]):
        assert self._rank_info is not None
        self._rank_info.sender_endpoints = endpoints
        self._rank_info.layer_num_per_pp = layer_num_per_pp

    def create_tx_session(self, request: LlmRequest) -> TxSession:
        params = request.py_disaggregated_params
        assert params is not None
        return TxSession(
            request_id=request.py_request_id,
            params=params,
            sender=self._sender,
            aux_buffer=self._aux_buffer,
            timeout_s=self._config.tx_timeout_s,
            prompt_len=request.prompt_len,
            beam_width=request.py_beam_width,
        )

    def create_rx_session(self, request: LlmRequest) -> RxSession:
        params = request.py_disaggregated_params
        assert params is not None
        return RxSession(
            request_id=request.py_request_id,
            params=params,
            receiver=self._receiver,
            aux_buffer=self._aux_buffer,
            timeout_s=self._config.rx_timeout_s,
            prompt_len=request.prompt_len,
            beam_width=request.py_beam_width,
        )

    def has_all_peer_req_infos_for_send(self, unique_rid: int) -> bool:
        return self._sender.has_all_peer_req_infos(unique_rid)

    def sweep_stale_req_infos(self):
        """Forward to Sender to evict orphaned RecvReqInfo from ADP broadcast."""
        self._sender.sweep_stale_req_infos()

    def _setup_peer_infrastructure(self, kvm: KVCacheManager):
        self._rank_info_server = RankInfoServer(self._rank_info) if kvm.mapping.rank == 0 else None
        self._kv_extractor = KVRegionExtractorV1(kvm)
        self._peer_registrar = PeerRegistrar(self._rank_info, self._kv_extractor)

    def _setup_transfer_engine(self):
        torch.cuda.set_device(self._config.device_id)
        CUASSERT(cudart.cudaSetDevice(self._config.device_id))
        self._agent = _create_nixl_agent(
            self._rank_info.instance_name + str(self._rank_info.instance_rank)
        )
        self._registered_mem: list = []
        try:
            self._register_kv_cache()
            if self._aux_buffer is not None:
                self._register_aux_buffer()
            from .bounce import create_bounce

            self._bounce = create_bounce(
                self._agent,
                self._config.bounce,
                device_id=self._config.device_id,
                page_table=self._rank_info.page_table,
            )
            # The bounce object owns its own NIXL descriptors (deregistered by Transport.close in
            # shutdown), so they are NOT added to _registered_mem — single-source to avoid a
            # double deregister.
            self._sender = Sender(self._peer_registrar, self._agent, bounce=self._bounce)
            self._receiver = Receiver(self._peer_registrar, self._agent, bounce=self._bounce)
            self._rank_info.transfer_engine_info = bytes(self._agent.get_local_agent_desc())
            self._rank_info.self_endpoint = self._receiver.endpoint
        except Exception:
            # shutdown()'s getattr guards handle whichever attrs got set before the failure.
            try:
                self.shutdown()
            except Exception as e:
                logger.warning(f"TransferWorker init-failure cleanup: {e}")
            raise

    def _register_kv_cache(self):
        assert self._rank_info.page_table is not None
        memory_descs = get_unique_pool_memory_descs(
            self._rank_info.page_table, self._rank_info.device_id
        )
        if memory_descs:
            reg_memory_desc = RegMemoryDescs("VRAM", memory_descs)
            self._agent.register_memory(reg_memory_desc)
            logger.debug(f"Registered KV cache memory with transfer agent: {memory_descs}")
            self._registered_mem.append(reg_memory_desc)

    def _register_aux_buffer(self):
        assert self._aux_buffer is not None
        aux_meta = self._aux_buffer.meta
        ptr_num = len(aux_meta.ptrs)
        ptr_descs = []
        for i in range(ptr_num):
            ptr_descs.append((aux_meta.ptrs[i], aux_meta.size[i], 0, f"aux_buffer_ptr_{i}"))
        reg_memory_desc = RegMemoryDescs("DRAM", ptr_descs)
        self._agent.register_memory(reg_memory_desc)
        logger.debug(f"Registered auxiliary buffer memory with transfer agent: {reg_memory_desc}")
        self._registered_mem.append(reg_memory_desc)

    @property
    def rank_info_server_endpoint(self) -> Optional[str]:
        return self._rank_info_server.endpoint if self._rank_info_server is not None else None

    @property
    def sender_endpoint(self) -> str:
        return self._sender.endpoint

    @property
    def page_table(self):
        assert self._rank_info is not None
        return self._rank_info.page_table

    def shutdown(self) -> Optional[threading.Event]:
        if self._is_progress_thread():
            return self._defer_shutdown()
        run_shutdown = False
        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                self._raise_shutdown_error()
                return
            if not self._shutdown:
                self._shutdown = True
                run_shutdown = True

        if run_shutdown:
            self._run_shutdown()
        else:
            self._shutdown_complete.wait()
        self._raise_shutdown_error()
        return None

    def _is_progress_thread(self) -> bool:
        """Whether teardown would join or wait on the current internal thread."""
        current = threading.current_thread()
        receiver = getattr(self, "_receiver", None)
        receiver_messenger = getattr(receiver, "_messenger", None)
        if current is getattr(receiver_messenger, "_listener_thread", None):
            return True
        bounce = getattr(self, "_bounce", None)
        if current is getattr(bounce, "_scatter_thread", None):
            return True
        sender = getattr(self, "_sender", None)
        sender_messenger = getattr(sender, "_messenger", None)
        if current is getattr(sender_messenger, "_listener_thread", None):
            return True
        if current in getattr(sender, "_worker_threads", ()):
            return True
        rank_info_server = getattr(self, "_rank_info_server", None)
        rank_info_messenger = getattr(rank_info_server, "_messenger", None)
        return current is getattr(rank_info_messenger, "_listener_thread", None)

    def _defer_shutdown(self) -> threading.Event:
        """Allow the current progress callback to return before teardown waits on it."""
        with self._shutdown_lock:
            if self._shutdown_complete.is_set() or self._shutdown:
                return self._shutdown_complete
            self._shutdown = True
            self._shutdown_thread = threading.Thread(
                target=self._run_shutdown,
                name="trtllm-transfer-worker-shutdown",
                daemon=True,
            )
            shutdown_thread = self._shutdown_thread
        logger.warning(
            "TransferWorker shutdown was requested from an internal progress thread; "
            "deferring teardown until the callback returns"
        )
        shutdown_thread.start()
        return self._shutdown_complete

    def _run_shutdown(self) -> None:
        try:
            self._shutdown_resources()
        except BaseException as error:
            self._shutdown_error = error
            logger.exception("TransferWorker shutdown stopped before memory teardown was safe")
        finally:
            self._shutdown_complete.set()

    def _raise_shutdown_error(self) -> None:
        if self._shutdown_error is not None:
            raise RuntimeError(
                "TransferWorker shutdown did not complete safely"
            ) from self._shutdown_error

    def _shutdown_resources(self) -> None:
        # Use getattr guards: __init__ may have failed partway, leaving some
        # attributes unset.  Without them, __del__ -> shutdown() raises
        # AttributeError and ZMQ resources from already-created sub-objects
        # are never cleaned up.
        rank_info_server = getattr(self, "_rank_info_server", None)
        if rank_info_server is not None:
            rank_info_server.shutdown()
        sender = getattr(self, "_sender", None)
        if sender is not None:
            sender.shutdown()
        receiver = getattr(self, "_receiver", None)
        if receiver is not None:
            receiver.shutdown()
        # Close the bounce transport (stops its scatter thread, deregisters its own descriptors and
        # frees the VMM buffers) once the receiver listener is stopped so no new scatter is enqueued.
        # No-op for the non-bounced path (NoBounceTransport.close); without this the daemon thread + fabric
        # buffers leak until process exit.
        bounce = getattr(self, "_bounce", None)
        if bounce is not None:
            # A close failure means the scatter thread may still touch its VMM
            # region. Propagate immediately and retain every later registered
            # memory object instead of invalidating memory under a live thread.
            bounce.close()
        # Deregister NIXL memory before agent.shutdown so pinned GPU memory is released
        # (e.g. when the KV cache manager is recreated after profiling).
        agent = getattr(self, "_agent", None)
        if agent is not None:
            registered = getattr(self, "_registered_mem", [])
            while registered:
                desc = registered.pop(0)
                try:
                    agent.deregister_memory(desc)
                except Exception as e:
                    logger.warning(f"TransferWorker.shutdown: deregister_memory failed: {e}")
            try:
                agent.shutdown()
            except Exception as e:
                logger.warning(f"TransferWorker.shutdown: agent.shutdown error: {e}")
            self._agent = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"TransferWorker.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()
