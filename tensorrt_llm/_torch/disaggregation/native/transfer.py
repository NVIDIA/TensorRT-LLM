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
import struct
import threading
import time
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

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
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger, decode_message
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer, perf_log_manager
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import (
    LifecycleAction,
    LifecycleUpdate,
    PhysicalState,
    RecvTransferRegistry,
    WriterMode,
    WriterResult,
)
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, MambaLayerGroup
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

# Standalone TransferWorker users do not have KvCacheTransceiverV2's owner map.
# Retain a worker whose shutdown cannot prove drain so its manager, listeners,
# registrations, and endpoint-owned sessions cannot be destructed underneath
# a late one-sided write.
_NON_DRAINED_TRANSFER_WORKERS: set["TransferWorker"] = set()


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

    def to_bytes(self) -> bytes:
        return msgpack.packb(
            {
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
        )

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


class SendOperationState(Enum):
    PENDING = "PENDING"
    IN_DOUBT = "IN_DOUBT"
    TERMINAL = "TERMINAL"


SendOperationKey = tuple[str, int]


@dataclass
class SendOperationRecord:
    """One exact sender operation admitted for a task and receiver rank."""

    state: SendOperationState
    terminal_message: Optional[tuple[bytes, ...]] = None
    result_delivered: bool = False


UnboundTerminalKey = tuple[WriteMetaType, str, int, Optional[int]]
PreSessionTerminalKey = tuple[int, WriteMetaType, str, int, Optional[int]]


@dataclass
class UnboundTerminalResult:
    """A no-access result created before its channel task exists."""

    req_info: RecvReqInfo
    message: tuple[bytes, ...]
    result_delivered: bool = False


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
    # Bind queued work to the exact session that authorized it. Looking a
    # session up again by request ID can attach stale work to a replacement.
    session: Optional["TxSession"] = None
    source_access_enrolled: bool = False
    operation_key: Optional[SendOperationKey] = None


class MessageType:
    TERMINATION = b"TERMINATION"
    KV_AGENT_RESULT = b"KV_AGENT_RESULT"
    REQUEST_DATA = b"REQUEST_DATA"
    REQUEST_INSTANCE_INFO = b"REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = b"REGISTER_RANK_INFO"
    AUX_AGENT_RESULT = b"AUX_AGENT_RESULT"
    CANCEL_SESSION = b"CANCEL_SESSION"


class TaskStatus(Enum):
    INIT = "INIT"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ERROR = "ERROR"


class AgentResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TransferSourceInDoubtError(RuntimeError):
    """A transfer-agent exception did not prove source access quiescent."""


@dataclass
class SendTransferContext:
    """Own one sender operation and every object needed to keep it safe.

    A context is normally lexical to a worker invocation. If submit/wait
    raises without terminal evidence, Sender retains the context so the exact
    session/source owner, transfer request/status handle, and bounce lease
    cannot be destroyed independently.
    """

    write_meta: WriteMeta
    session: "TxSession"
    source_owner: object | None = None
    transfer_request: object | None = None
    transfer_status: object | None = None
    bounce_slot_id: Optional[int] = None
    result_delivered: bool = False


@dataclass(frozen=True)
class _PendingLifecycleDelivery:
    """A registry transition retained until its session consumer acknowledges it."""

    update: LifecycleUpdate
    peer_rank: Optional[int]
    succeeded: bool


# KV_AGENT_RESULT prefix in one struct frame (was 5 ascii frames serialized/parsed under the
# GIL per slice per writer): instance_rank, unique_rid, slice_id, is_last, status. The optional
# bounce tail follows at message[2:].
_KV_RESULT_PREFIX = struct.Struct("<qqq?B")
_AGENT_RESULT_CODE = {AgentResult.SUCCESS: 0, AgentResult.FAILED: 1}
_AGENT_RESULT_BY_CODE = {0: AgentResult.SUCCESS, 1: AgentResult.FAILED}


def _make_kv_result_msg(
    instance_rank, unique_rid, slice_id, is_last_slice, agent_result, tail=None
):
    """Build a KV_AGENT_RESULT message. ALL result sends (success AND failed/cancelled) must go
    through this single binary frame so the receiver's _KV_RESULT_PREFIX.unpack never hits a stale
    ascii payload (which would fail to decode and leave the RX task stuck forever)."""
    msg = [
        MessageType.KV_AGENT_RESULT,
        _KV_RESULT_PREFIX.pack(
            int(instance_rank),
            int(unique_rid),
            int(slice_id),
            bool(is_last_slice),
            _AGENT_RESULT_CODE[agent_result],
        ),
    ]
    if tail:
        msg += tail
    return msg


class SendTaskBase:
    def __init__(self, params: DisaggregatedParams, session: Optional["TxSession"] = None):
        self.status = TaskStatus.INIT
        self._event = threading.Event()
        self._exception: Optional[Exception] = None
        self.lock = threading.Lock()
        self._params = params
        self._unique_rid: Optional[int] = params.disagg_request_id
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None
        self._session = session
        # Count every queued, active, or in-doubt operation independently. A
        # task-level ERROR is only a logical result; sibling operations may
        # still be reading the same source allocation.
        self._source_access_count = 0
        # The task itself scopes the channel (KV slice or session-level AUX).
        # A receiver instance/rank may therefore authorize at most one physical
        # operation for this task. Terminal frames are cached for idempotent
        # replay instead of launching a second write.
        self._send_operations: dict[SendOperationKey, SendOperationRecord] = {}

    def begin_source_access(self) -> None:
        with self.lock:
            self._source_access_count += 1

    def finish_source_access(self) -> None:
        with self.lock:
            if self._source_access_count <= 0:
                raise RuntimeError("source access ownership underflow")
            self._source_access_count -= 1

    @property
    def source_access_active(self) -> bool:
        with self.lock:
            return self._source_access_count > 0

    def admit_operation(
        self,
        key: SendOperationKey,
        *,
        allow_source_access: bool,
        no_access_message: tuple[bytes, ...],
    ) -> tuple[bool, SendOperationState, Optional[tuple[bytes, ...]], bool]:
        """Admit one exact physical operation or return its existing state.

        The caller holds the owning session's lifecycle lock. The task lock
        makes terminal updates from a worker atomic with duplicate admission.
        """
        with self.lock:
            existing = self._send_operations.get(key)
            if existing is not None:
                return (
                    False,
                    existing.state,
                    existing.terminal_message,
                    existing.result_delivered,
                )
            if allow_source_access:
                self._send_operations[key] = SendOperationRecord(SendOperationState.PENDING)
                self._source_access_count += 1
                return True, SendOperationState.PENDING, None, False
            self._send_operations[key] = SendOperationRecord(
                SendOperationState.TERMINAL, no_access_message
            )
            return True, SendOperationState.TERMINAL, no_access_message, False

    def operation_snapshot(
        self, key: SendOperationKey
    ) -> Optional[tuple[SendOperationState, Optional[tuple[bytes, ...]], bool]]:
        with self.lock:
            record = self._send_operations.get(key)
            if record is None:
                return None
            return record.state, record.terminal_message, record.result_delivered

    def mark_operation_in_doubt(self, key: Optional[SendOperationKey]) -> None:
        if key is None:
            return
        with self.lock:
            record = self._send_operations.get(key)
            if record is None:
                raise RuntimeError(f"sender operation {key} was not admitted")
            if record.state is not SendOperationState.TERMINAL:
                record.state = SendOperationState.IN_DOUBT

    def cache_terminal_message(
        self, key: Optional[SendOperationKey], message: list[bytes] | tuple[bytes, ...]
    ) -> None:
        if key is None:
            return
        cached = tuple(message)
        with self.lock:
            record = self._send_operations.get(key)
            if record is None:
                raise RuntimeError(f"sender operation {key} was not admitted")
            if record.state is SendOperationState.TERMINAL:
                if record.terminal_message != cached:
                    raise RuntimeError(f"conflicting terminal result for sender operation {key}")
                return
            record.state = SendOperationState.TERMINAL
            record.terminal_message = cached

    def mark_operation_result_delivered(self, key: Optional[SendOperationKey]) -> None:
        if key is None:
            return
        with self.lock:
            record = self._send_operations.get(key)
            if record is None or record.state is not SendOperationState.TERMINAL:
                raise RuntimeError(f"sender operation {key} has no terminal result to deliver")
            record.result_delivered = True

    def pending_terminal_messages(
        self,
    ) -> list[tuple[SendOperationKey, tuple[bytes, ...]]]:
        with self.lock:
            return [
                (key, record.terminal_message)
                for key, record in self._send_operations.items()
                if record.state is SendOperationState.TERMINAL
                and record.terminal_message is not None
                and not record.result_delivered
            ]

    @property
    def has_pending_result_delivery(self) -> bool:
        with self.lock:
            return any(
                record.state is SendOperationState.TERMINAL
                and record.terminal_message is not None
                and not record.result_delivered
                for record in self._send_operations.values()
            )

    def mark_transferring(self) -> bool:
        """Start physical work without reviving a terminal task."""
        with self.lock:
            if self.status is not TaskStatus.INIT:
                return self.status is TaskStatus.TRANSFERRING
            self.status = TaskStatus.TRANSFERRING
            return True

    def fail(self, exc: Exception) -> None:
        with self.lock:
            self._exception = exc
            self.status = TaskStatus.ERROR
            self._event.set()

    def complete(self) -> bool:
        with self.lock:
            if self.status is TaskStatus.ERROR:
                return False
            self.status = TaskStatus.TRANSFERRED
            self._event.set()
            return True

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
    def __init__(
        self,
        params: DisaggregatedParams,
        slot: Optional[int],
        session: Optional["TxSession"] = None,
    ):
        super().__init__(params, session)
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
        session: Optional["TxSession"] = None,
    ):
        super().__init__(params, session)
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
    _TERMINAL_RESULT_RETRY_INTERVAL_S = 0.05

    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        agent: BaseTransferAgent,
        bounce=None,
    ):
        self._shutdown_attempt_lock = threading.Lock()
        self._registrar = peer_registrar
        self._device_id = peer_registrar.self_rank_info.device_id
        self._agent = agent
        self._bounce = bounce
        self._peer_requests: dict = {}
        self._peer_requests_timestamps: dict[int, float] = {}  # unique_rid -> insert time
        self._peer_requests_lock = threading.Lock()
        self._messenger = ZMQMessenger(mode="ROUTER")
        # Control sends can originate from the listener or executor thread.
        # ZMQMessenger serializes each socket; this lock additionally makes the
        # endpoint-to-socket cache's check/create transition atomic.
        self._dealers = {}
        self._dealers_lock = threading.Lock()
        self._thread_local = threading.local()  # per-thread DEALER cache for worker threads
        # Close direct TxSession admission before shutdown can append queue
        # sentinels. Listener callbacks that were already admitted may finish
        # enqueueing before messenger.stop() joins them.
        self._operation_admission_lock = threading.RLock()
        self._dealer_admission_closed = False
        # Worker-local sockets can only be detached by their owning thread.
        # Failed closes move here so Sender remains their retry owner.
        self._failed_thread_dealers: list[ZMQMessenger] = []
        self._failed_thread_dealers_lock = threading.Lock()
        # Sender is the lifecycle root for standalone TransferWorker users.
        # Keep sessions strong until explicit, drain-checked clear_session().
        self._sessions: dict[int, TxSession] = {}
        self._sessions_lock = threading.Lock()  # Protects _sessions and _pre_cancelled_rids
        self._pre_cancelled_rids: set[int] = set()
        self._pre_session_terminal_results: dict[PreSessionTerminalKey, UnboundTerminalResult] = {}
        self._pre_session_terminal_results_lock = threading.Lock()
        self._pre_session_terminal_retry_lock = threading.Lock()
        self._next_pre_session_terminal_retry_at = 0.0
        self._in_doubt_transfers: list[SendTransferContext] = []
        self._in_doubt_transfers_lock = threading.Lock()
        self._shutdown = False
        self._listener_stopped = False
        self._shutdown_complete = False
        self._shutdown_sentinels_sent = False
        self._instance_rank = self._registrar.self_rank_info.instance_rank
        # Guards concurrent add() from the listener thread.
        self._loaded_remote_agents: set[str] = set()
        self._loaded_remote_agents_lock = threading.Lock()
        self._num_threads = KV_TRANSFER_NUM_THREADS
        self._send_task_queues: List[queue.Queue] = [
            queue.Queue() for _ in range(self._num_threads)
        ]
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

    def _add_req_info(self, unique_rid: int, req_info: RecvReqInfo):
        with self._peer_requests_lock:
            if unique_rid not in self._peer_requests:
                self._peer_requests[unique_rid] = {}
                self._peer_requests_timestamps[unique_rid] = time.monotonic()
            key = (req_info.instance_rank, req_info.slice_id)
            self._peer_requests[unique_rid][key] = req_info

    def _is_req_ready(
        self,
        unique_rid: int,
        expected_count: int,
        slice_id: Optional[int] = None,
    ) -> bool:
        with self._peer_requests_lock:
            requests = self._peer_requests.get(unique_rid)
            if not requests:
                return False
            ranks = {
                info.instance_rank
                for info in requests.values()
                if slice_id is None or info.slice_id == slice_id
            }
            return len(ranks) == expected_count

    def _get_req_info(self, unique_rid: Optional[int]) -> Optional[dict]:
        with self._peer_requests_lock:
            requests = self._peer_requests.get(unique_rid)
            return None if requests is None else dict(requests)

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

        Called from the scheduler's sender progress path. With gen-first ADP
        broadcast, non-assigned DP ranks accumulate entries that are never
        consumed; this sweep prevents unbounded growth.
        """
        # The scheduler already calls this method as its sender-side progress
        # hook. Reuse that cadence to retry pre-session terminal decisions so a
        # transient control-send failure does not wait until process shutdown.
        self._retry_pre_session_terminal_results()
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
                if (
                    rid not in self._sessions
                    and rid not in self._pre_cancelled_rids
                    and rid in self._peer_requests
                ):
                    self._peer_requests.pop(rid, None)
                    self._peer_requests_timestamps.pop(rid, None)
                    logger.debug(f"Swept stale RecvReqInfo for rid={rid}")

    def setup_session(self, tx_session: "TxSession"):
        unique_rid = tx_session.disagg_request_id
        pre_cancel = False
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            published = False
            try:
                with self._sessions_lock:
                    if self._shutdown:
                        raise RuntimeError("Sender is shutting down; new sessions are not accepted")
                    existing = self._sessions.get(unique_rid)
                    if existing is not None and existing is not tx_session:
                        raise RuntimeError(f"TxSession {unique_rid} is already registered")
                    self._sessions[unique_rid] = tx_session
                    published = True
                    if unique_rid in self._pre_cancelled_rids:
                        pre_cancel = True
                if pre_cancel:
                    tx_session.cancel()
                    return

                req_info = self._get_first_req_info(unique_rid)

                if req_info:
                    peer_ri = self._registrar.get_peer_rank_info(
                        req_info.instance_name, req_info.instance_rank
                    )
                    expected_count = len(
                        self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks
                    )
                    if self._is_req_ready(unique_rid, expected_count, req_info.slice_id):
                        with tx_session.lock:
                            tx_session.receiver_ready = True
            except Exception:
                if published:
                    with self._sessions_lock:
                        if self._sessions.get(unique_rid) is tx_session:
                            self._sessions.pop(unique_rid, None)
                raise
        return

    def _get_session(self, unique_rid: Optional[int]) -> Optional["TxSession"]:
        return self._sessions.get(unique_rid)

    def _enqueue(self, write_meta: WriteMeta):
        # Route by (unique_rid, peer_rank) so that:
        # - Same peer's slices stay ordered on one thread (is_last_slice correctness)
        # - Different peers can run on different threads (better load balancing)
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_shutdown_sentinels_sent", False):
                raise RuntimeError("Sender queues are closed for shutdown")
            thread_idx = hash((write_meta.unique_rid, write_meta.peer_rank)) % self._num_threads
            self._send_task_queues[thread_idx].put(write_meta)

    def _enqueue_owned(self, write_meta: WriteMeta) -> None:
        """Enroll source ownership before queued work can outlive its caller."""
        enrolled_here = not write_meta.source_access_enrolled
        if enrolled_here:
            write_meta.task.begin_source_access()
            write_meta.source_access_enrolled = True
        try:
            self._enqueue(write_meta)
        except Exception:
            if enrolled_here:
                write_meta.task.finish_source_access()
                write_meta.source_access_enrolled = False
            raise

    def _retain_in_doubt_transfer(self, context: SendTransferContext) -> None:
        with self._in_doubt_transfers_lock:
            self._in_doubt_transfers.append(context)

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

    def _close_thread_dealers(self, thread_idx: int, dealers: dict[str, ZMQMessenger]) -> None:
        """Detach worker-local sockets while preserving failed closes."""
        for endpoint, dealer in dealers.items():
            try:
                dealer.stop()
            except Exception as e:
                failed_lock = getattr(self, "_failed_thread_dealers_lock", None)
                if failed_lock is not None:
                    with failed_lock:
                        self._failed_thread_dealers.append(dealer)
                else:
                    self._failed_thread_dealers.append(dealer)
                logger.warning(
                    f"_process_task_queue[{thread_idx}]: failed to stop dealer "
                    f"for endpoint {endpoint}; retaining it for retry: {e}"
                )
        dealers.clear()

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
                except Exception as e:
                    logger.error(
                        f"_process_task_queue[{thread_idx}]: unhandled exception for "
                        f"unique_rid={write_meta.unique_rid}: {e}"
                    )
                    from .bounce import GatherSourceInDoubtError

                    if isinstance(e, (GatherSourceInDoubtError, TransferSourceInDoubtError)):
                        # Keep the per-operation source owner active. The
                        # retained SendTransferContext owns the exact session,
                        # request, and every handle returned before ambiguity.
                        # A gather failure can occur before the bounce slot is
                        # returned; in that case the bounce allocator owns its
                        # quarantined slot independently.
                        logger.critical(
                            f"_process_task_queue[{thread_idx}]: retaining source owner for "
                            f"unique_rid={write_meta.unique_rid} after ambiguous source access"
                        )
                    else:
                        write_meta.task.fail(e)
        finally:
            # Clean up this thread's DEALER sockets. threading.local storage
            # is only accessible from the owning thread, so shutdown must
            # happen here rather than in Sender.shutdown().
            dealers = getattr(self._thread_local, "dealers", None)
            if dealers:
                self._close_thread_dealers(thread_idx, dealers)

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

    def _failed_write_meta_message(self, write_meta: WriteMeta) -> list[bytes]:
        if write_meta.meta_type is WriteMetaType.AUX:
            return [
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                AgentResult.FAILED.value.encode("ascii"),
            ]
        return _make_kv_result_msg(
            self._instance_rank,
            write_meta.unique_rid,
            write_meta.slice_id if write_meta.slice_id is not None else 0,
            write_meta.is_last_slice,
            AgentResult.FAILED,
        )

    def _send_worker_operation_message(
        self,
        task: SendTaskBase,
        write_meta: WriteMeta,
        message: list[bytes],
    ) -> bool:
        """Cache then submit a terminal frame from the owning worker thread."""
        task.cache_terminal_message(write_meta.operation_key, message)
        try:
            self._get_or_connect_thread_dealer(write_meta.peer_endpoint).send(message)
        except Exception as e:
            logger.error(
                f"sender result delivery remains pending for rid={write_meta.unique_rid} "
                f"writer={write_meta.peer_rank}: {e}"
            )
            return False
        task.mark_operation_result_delivered(write_meta.operation_key)
        return True

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, write_meta: WriteMeta):
        task = write_meta.task
        if not isinstance(task, KVSendTask):
            raise RuntimeError("queued KV work has an invalid task type")
        if not write_meta.source_access_enrolled:
            task.begin_source_access()
            write_meta.source_access_enrolled = True

        from .bounce import GatherSourceInDoubtError, build_send_request, encode_result_tail

        session = write_meta.session
        context = None
        source_terminal = False
        timer = task._perf_timer
        try:
            if not (write_meta.src_ptrs.size == write_meta.dst_ptrs.size == write_meta.sizes.size):
                raise ValueError(
                    f"WriteMeta ptr/size mismatch for unique_rid={write_meta.unique_rid}"
                )
            if session is None:
                # Compatibility for standalone callers that construct
                # WriteMeta directly. Normal queued work is bound to an exact
                # session before admission.
                with self._sessions_lock:
                    session = self._get_session(write_meta.unique_rid)
            if session is None:
                raise RuntimeError(
                    f"_deliver_kv_to_agent: TxSession {write_meta.unique_rid} "
                    "not found or already GC'd"
                )
            if write_meta.slice_id is None:
                raise RuntimeError("queued KV work has no slice ID")
            if (
                write_meta.slice_id >= len(session.kv_tasks)
                or session.kv_tasks[write_meta.slice_id] is not task
            ):
                raise RuntimeError(
                    f"queued KV work no longer belongs to its session: "
                    f"unique_rid={write_meta.unique_rid} slice={write_meta.slice_id}"
                )

            context = SendTransferContext(
                write_meta=write_meta,
                session=session,
                source_owner=session.source_owner,
            )
            if timer:
                timer.record_push_end(write_meta.peer_rank)
            # Hold session.lock to serialize the INIT→TRANSFERRING transition
            # with cancel(). Source ownership was enrolled before queueing.
            with session.lock:
                status = session.status
                if status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                    should_abort = True
                else:
                    should_abort = not task.mark_transferring()

            if should_abort:
                logger.warning(
                    f"_deliver_kv_to_agent: session {write_meta.unique_rid} already "
                    f"in {status.value} state; sending FAILED to receiver"
                )
                task.fail(
                    RuntimeError(
                        f"session {write_meta.unique_rid} {status.value}, transfer aborted"
                    )
                )
                # The source decision is terminal before result I/O. A socket
                # error must not strand the source lease.
                source_terminal = True
                self._send_worker_operation_message(
                    task, write_meta, self._failed_write_meta_message(write_meta)
                )
                return

            agent_result = AgentResult.SUCCESS
            send_slot_id = None
            if write_meta.src_ptrs.size > 0:
                try:
                    request, send_slot_id = build_send_request(
                        self._bounce,
                        write_meta,
                        lambda: Sender._make_agent_request(write_meta, device_id=self._device_id),
                    )
                    context.transfer_request = request
                    context.bounce_slot_id = send_slot_id
                except GatherSourceInDoubtError:
                    task.mark_operation_in_doubt(write_meta.operation_key)
                    self._retain_in_doubt_transfer(context)
                    raise
                except Exception as e:
                    # Gather rollback positively fenced local source access,
                    # and no network request was submitted.
                    source_terminal = True
                    task.fail(e)
                    self._send_worker_operation_message(
                        task, write_meta, self._failed_write_meta_message(write_meta)
                    )
                    return
                transfer_terminal = False
                try:
                    if timer:
                        timer.record_transfer_start(write_meta.peer_rank)
                    transfer_status = self._agent.submit_transfer_requests(request)
                    context.transfer_status = transfer_status
                    transfer_succeeded = transfer_status.wait()
                    transfer_terminal = True
                    source_terminal = True
                    if not transfer_succeeded:
                        agent_result = AgentResult.FAILED
                        last_status = getattr(
                            transfer_status, "last_status_str", lambda: "<no detail>"
                        )()
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
                except Exception as e:
                    task.mark_operation_in_doubt(write_meta.operation_key)
                    # Retain the complete context first. Bounce quarantine is
                    # itself fallible and must never be able to lose the source
                    # owner or the backend status handle.
                    self._retain_in_doubt_transfer(context)
                    if send_slot_id is not None:
                        try:
                            self._bounce.quarantine_send(send_slot_id)
                        except Exception as quarantine_error:
                            logger.critical(
                                f"failed to quarantine send slot {send_slot_id} for "
                                f"request {write_meta.unique_rid}; retained transfer "
                                f"context remains authoritative: {quarantine_error}"
                            )
                    raise TransferSourceInDoubtError(
                        f"KV transfer-agent operation is in doubt for "
                        f"request {write_meta.unique_rid} slice={write_meta.slice_id} "
                        f"writer={write_meta.peer_rank}"
                    ) from e
                finally:
                    if transfer_terminal and send_slot_id is not None:
                        self._bounce.release_send(send_slot_id)
                        context.bounce_slot_id = None
            else:
                source_terminal = True
            if timer:
                timer.record_transfer_end(write_meta.peer_rank)

            tail = (
                encode_result_tail(write_meta)
                if send_slot_id is not None and agent_result == AgentResult.SUCCESS
                else None
            )
            result_msg = _make_kv_result_msg(
                self._instance_rank,
                write_meta.unique_rid,
                write_meta.slice_id,
                write_meta.is_last_slice,
                agent_result,
                tail=tail,
            )
            delivered = self._send_worker_operation_message(task, write_meta, result_msg)
            context.result_delivered = delivered

            if timer:
                timer.record_task_end(write_meta.peer_rank)
            ri = self._registrar.self_rank_info
            task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)

            with task.lock:
                task.transferred_count += 1
                count = task.transferred_count

            if count > write_meta.expected_transfers:
                session.set_exception(
                    f"KV slice {write_meta.slice_id} received more than "
                    f"{write_meta.expected_transfers} transfers"
                )
            elif count == write_meta.expected_transfers:
                if not task.complete():
                    session.set_exception(
                        f"KV slice {write_meta.slice_id} task failed before all writers completed"
                    )

            logger.debug(
                f"deliver_kv_to_agent completed: unique_rid={write_meta.unique_rid}, "
                f"slice_id={write_meta.slice_id}, agent_result={agent_result}"
            )
        except (GatherSourceInDoubtError, TransferSourceInDoubtError):
            raise
        except Exception as e:
            # Every fenced failure is terminal for local source access. Cache
            # the result before attempting I/O so cancellation/REQUEST_DATA can
            # replay it later.
            source_terminal = True
            task.fail(e)
            snapshot = (
                task.operation_snapshot(write_meta.operation_key)
                if write_meta.operation_key is not None
                else None
            )
            if snapshot is None or snapshot[0] is not SendOperationState.TERMINAL:
                self._send_worker_operation_message(
                    task, write_meta, self._failed_write_meta_message(write_meta)
                )
            raise
        finally:
            if source_terminal:
                task.finish_source_access()
                write_meta.source_access_enrolled = False

    @nvtx_range("_deliver_aux_to_agent")
    def _deliver_aux_to_agent(self, write_meta: WriteMeta):
        aux_task = write_meta.task
        if not isinstance(aux_task, AuxSendTask):
            raise RuntimeError("queued AUX work has an invalid task type")
        if not write_meta.source_access_enrolled:
            aux_task.begin_source_access()
            write_meta.source_access_enrolled = True

        session = write_meta.session
        context = None
        source_terminal = False
        timer = aux_task._perf_timer
        try:
            if not (write_meta.src_ptrs.size == write_meta.dst_ptrs.size == write_meta.sizes.size):
                raise ValueError(
                    f"AUX WriteMeta ptr/size mismatch for unique_rid={write_meta.unique_rid}"
                )
            if session is None:
                with self._sessions_lock:
                    session = self._get_session(write_meta.unique_rid)
            if session is None:
                raise RuntimeError(
                    f"_deliver_aux_to_agent: TxSession {write_meta.unique_rid} "
                    "not found or already GC'd"
                )
            if session.aux_task is not aux_task:
                raise RuntimeError(
                    f"queued AUX work no longer belongs to its session: "
                    f"unique_rid={write_meta.unique_rid}"
                )

            context = SendTransferContext(
                write_meta=write_meta,
                session=session,
                source_owner=session.source_owner,
            )
            if timer:
                timer.record_push_end(write_meta.peer_rank)

            with session.lock:
                status = session.status
                if status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                    should_abort = True
                else:
                    should_abort = not aux_task.mark_transferring()

            if should_abort:
                aux_task.fail(
                    RuntimeError(f"session {write_meta.unique_rid} {status.value}, AUX aborted")
                )
                source_terminal = True
                self._send_worker_operation_message(
                    aux_task, write_meta, self._failed_write_meta_message(write_meta)
                )
                return

            agent_result = AgentResult.SUCCESS
            if write_meta.src_ptrs.size > 0:
                try:
                    request = Sender._make_agent_request(write_meta, device_id=self._device_id)
                    context.transfer_request = request
                except Exception as e:
                    source_terminal = True
                    aux_task.fail(e)
                    self._send_worker_operation_message(
                        aux_task, write_meta, self._failed_write_meta_message(write_meta)
                    )
                    return
                try:
                    if timer:
                        timer.record_transfer_start(write_meta.peer_rank)
                    transfer_status = self._agent.submit_transfer_requests(request)
                    context.transfer_status = transfer_status
                    transfer_succeeded = transfer_status.wait()
                    source_terminal = True
                except Exception as e:
                    aux_task.mark_operation_in_doubt(write_meta.operation_key)
                    self._retain_in_doubt_transfer(context)
                    raise TransferSourceInDoubtError(
                        f"AUX transfer-agent operation is in doubt for "
                        f"request {write_meta.unique_rid} writer={write_meta.peer_rank}"
                    ) from e
                if not transfer_succeeded:
                    agent_result = AgentResult.FAILED
                    session.set_exception("aux transfer agent request failed")
                if timer:
                    timer.record_transfer_end(write_meta.peer_rank)
            else:
                source_terminal = True

            result_msg = [
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                agent_result.value.encode("ascii"),
            ]
            delivered = self._send_worker_operation_message(aux_task, write_meta, result_msg)
            context.result_delivered = delivered

            if timer:
                timer.record_task_end(write_meta.peer_rank)
            ri = self._registrar.self_rank_info
            aux_task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)

            with aux_task.lock:
                aux_task._transfer_count += 1
                count = aux_task._transfer_count

            if count == write_meta.expected_transfers:
                if not aux_task.complete():
                    session.set_exception("aux task failed before all writers completed")
            elif count > write_meta.expected_transfers:
                session.set_exception(
                    f"aux task received more than {write_meta.expected_transfers} transfers"
                )
        except TransferSourceInDoubtError:
            raise
        except Exception as e:
            source_terminal = True
            aux_task.fail(e)
            snapshot = (
                aux_task.operation_snapshot(write_meta.operation_key)
                if write_meta.operation_key is not None
                else None
            )
            if snapshot is None or snapshot[0] is not SendOperationState.TERMINAL:
                self._send_worker_operation_message(
                    aux_task, write_meta, self._failed_write_meta_message(write_meta)
                )
            raise
        finally:
            if source_terminal:
                aux_task.finish_source_access()
                write_meta.source_access_enrolled = False

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
            session=task._session,
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
            session=task._session,
        )

    @staticmethod
    def _operation_key(req_info: RecvReqInfo) -> SendOperationKey:
        return req_info.instance_name, req_info.instance_rank

    def _failed_operation_message(
        self, task: KVSendTask | AuxSendTask, req_info: RecvReqInfo
    ) -> tuple[bytes, ...]:
        if isinstance(task, KVSendTask):
            return tuple(
                _make_kv_result_msg(
                    self._instance_rank,
                    req_info.unique_rid,
                    task.slice_id,
                    task._slice.is_last_slice,
                    AgentResult.FAILED,
                )
            )
        return (
            MessageType.AUX_AGENT_RESULT,
            str(self._instance_rank).encode("ascii"),
            str(req_info.unique_rid).encode("ascii"),
            AgentResult.FAILED.value.encode("ascii"),
        )

    def _send_operation_message(
        self, req_info: RecvReqInfo, message: tuple[bytes, ...] | list[bytes]
    ) -> bool:
        """Best-effort delivery for a cached terminal result.

        The result remains in the operation ledger after an I/O failure, so an
        identical request/cancel replay can retry it without launching work.
        """
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                return False
            try:
                peer_ri = self._registrar.get_peer_rank_info(
                    req_info.instance_name, req_info.instance_rank
                )
                self._get_or_connect_dealer(peer_ri.self_endpoint).send(list(message))
                return True
            except Exception as e:
                logger.warning(
                    f"failed to deliver cached sender result for rid={req_info.unique_rid} "
                    f"receiver={req_info.instance_name}/{req_info.instance_rank}: {e}"
                )
                return False

    def _send_unbound_terminal_result(
        self,
        session: "TxSession",
        req_info: RecvReqInfo,
        meta_type: WriteMetaType,
        message: tuple[bytes, ...],
    ) -> None:
        """Cache and send a terminal decision made before a task exists."""
        key = session.cache_unbound_terminal_result(meta_type, req_info, message)
        if self._send_operation_message(req_info, message):
            session.mark_unbound_terminal_result_delivered(key)

    @staticmethod
    def _pre_session_terminal_key(
        meta_type: WriteMetaType, req_info: RecvReqInfo
    ) -> PreSessionTerminalKey:
        slice_id = req_info.slice_id if meta_type is WriteMetaType.KV else None
        return (
            req_info.unique_rid,
            meta_type,
            req_info.instance_name,
            req_info.instance_rank,
            slice_id,
        )

    def _send_pre_session_terminal_result(
        self,
        req_info: RecvReqInfo,
        meta_type: WriteMetaType,
        message: tuple[bytes, ...],
    ) -> None:
        """Retain a no-access result even when no TxSession exists yet."""
        key = self._pre_session_terminal_key(meta_type, req_info)
        results_lock = self._pre_session_terminal_results_lock
        with results_lock:
            record = self._pre_session_terminal_results.get(key)
            if record is None:
                record = UnboundTerminalResult(req_info=req_info, message=message)
                self._pre_session_terminal_results[key] = record
            elif record.message != message:
                raise RuntimeError(f"conflicting pre-session terminal result for {key}")
        if self._send_operation_message(req_info, message):
            with results_lock:
                current = self._pre_session_terminal_results.get(key)
                if current is record:
                    current.result_delivered = True

    def _retry_pre_session_terminal_results(self, *, force: bool = False) -> None:
        results_lock = getattr(self, "_pre_session_terminal_results_lock", None)
        if results_lock is None:
            return
        retry_lock = getattr(self, "_pre_session_terminal_retry_lock", None)
        if retry_lock is None or not retry_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            next_retry_at = getattr(self, "_next_pre_session_terminal_retry_at", 0.0)
            if not force and now < next_retry_at:
                return
            self._next_pre_session_terminal_retry_at = now + self._TERMINAL_RESULT_RETRY_INTERVAL_S
            with results_lock:
                pending = [
                    (key, record)
                    for key, record in self._pre_session_terminal_results.items()
                    if not record.result_delivered
                ]
            for key, record in pending:
                if self._send_operation_message(record.req_info, record.message):
                    with results_lock:
                        current = self._pre_session_terminal_results.get(key)
                        if current is record:
                            current.result_delivered = True
        finally:
            retry_lock.release()

    def _has_pending_pre_session_terminal_results(self) -> bool:
        results_lock = getattr(self, "_pre_session_terminal_results_lock", None)
        if results_lock is None:
            return False
        with results_lock:
            return any(
                not record.result_delivered
                for record in self._pre_session_terminal_results.values()
            )

    def retry_terminal_results(self, session: "TxSession", *, force: bool = False) -> None:
        """Retry each pending terminal frame at most once per polling interval."""
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                return
            retry_lock = getattr(session, "_terminal_retry_lock", None)
            if retry_lock is None or not retry_lock.acquire(blocking=False):
                return
            try:
                now = time.monotonic()
                if not force and now < session._next_terminal_retry_at:
                    return
                session._next_terminal_retry_at = now + self._TERMINAL_RESULT_RETRY_INTERVAL_S
                with session.lock:
                    tasks: list[SendTaskBase] = list(session.kv_tasks)
                    if session.aux_task is not None:
                        tasks.append(session.aux_task)
                    unbound = [
                        (key, record.req_info, record.message)
                        for key, record in session._unbound_terminal_results.items()
                        if not record.result_delivered
                    ]

                req_infos = list((self._get_req_info(session.disagg_request_id) or {}).values())
                info_by_operation = {
                    self._operation_key(req_info): req_info for req_info in req_infos
                }
                for task in tasks:
                    for key, message in task.pending_terminal_messages():
                        req_info = info_by_operation.get(key)
                        if req_info is None:
                            continue
                        if self._send_operation_message(req_info, message):
                            task.mark_operation_result_delivered(key)
                for key, req_info, message in unbound:
                    if self._send_operation_message(req_info, message):
                        session.mark_unbound_terminal_result_delivered(key)
            finally:
                retry_lock.release()

    def _dispatch_operation(
        self,
        task: KVSendTask | AuxSendTask,
        req_info: RecvReqInfo,
        *,
        force_no_access: bool = False,
    ) -> Optional[Exception]:
        """Claim and dispatch one exact task/receiver operation at most once."""
        session = task._session
        if session is None:
            # Compatibility for standalone callers. Production tasks are bound
            # to an exact TxSession and use the operation ledger below.
            try:
                trans_meta = (
                    self._build_kv_write_meta(task, req_info)
                    if isinstance(task, KVSendTask)
                    else self._build_aux_write_meta(task, req_info)
                )
                self._enqueue_owned(trans_meta)
                return None
            except Exception as e:
                task.fail(e)
                self._send_operation_message(
                    req_info, self._failed_operation_message(task, req_info)
                )
                return e

        key = self._operation_key(req_info)
        failed_message = self._failed_operation_message(task, req_info)
        with session.lock:
            if isinstance(task, KVSendTask):
                belongs_to_session = (
                    0 <= task.slice_id < len(session.kv_tasks)
                    and session.kv_tasks[task.slice_id] is task
                )
            else:
                belongs_to_session = session.aux_task is task
            if not belongs_to_session:
                return RuntimeError(
                    f"sender task no longer belongs to session {session.disagg_request_id}"
                )
            allow_source_access = (
                not force_no_access
                and getattr(session, "_accepting_operations", True)
                and not getattr(session, "_closed", False)
                and session.status not in (SessionStatus.ERROR, SessionStatus.CANCELLED)
            )
            newly_recorded, state, cached_message, _result_delivered = task.admit_operation(
                key,
                allow_source_access=allow_source_access,
                no_access_message=failed_message,
            )

        if not newly_recorded:
            # A duplicate REQUEST_DATA is also an acknowledgement retry. Replay
            # the exact cached decision even if an earlier send returned
            # successfully: DEALER send completion is not remote receipt.
            if (
                state is SendOperationState.TERMINAL
                and cached_message is not None
                and self._send_operation_message(req_info, cached_message)
            ):
                task.mark_operation_result_delivered(key)
            return None
        if state is SendOperationState.TERMINAL:
            assert cached_message is not None
            if self._send_operation_message(req_info, cached_message):
                task.mark_operation_result_delivered(key)
            return None

        trans_meta = None
        try:
            if task._perf_timer is not None:
                task._perf_timer.record_task_start(req_info.instance_rank)
            trans_meta = (
                self._build_kv_write_meta(task, req_info)
                if isinstance(task, KVSendTask)
                else self._build_aux_write_meta(task, req_info)
            )
            trans_meta.operation_key = key
            trans_meta.source_access_enrolled = True
            if task._perf_timer is not None:
                task._perf_timer.record_push_start(trans_meta.peer_rank)
            self._enqueue_owned(trans_meta)
            return None
        except Exception as e:
            # Descriptor construction and queue admission have no asynchronous
            # source accessor. Settle this exact writer and continue dispatching
            # the rest of the cohort.
            task.fail(e)
            try:
                task.cache_terminal_message(key, failed_message)
            finally:
                task.finish_source_access()
                if trans_meta is not None:
                    trans_meta.source_access_enrolled = False
            if self._send_operation_message(req_info, failed_message):
                task.mark_operation_result_delivered(key)
            return e

    def dispatch_task(
        self,
        task: KVSendTask | AuxSendTask,
        req_info_snapshot: Optional[dict] = None,
    ):
        # req_info_snapshot may be pre-fetched under session.lock by the caller to keep the
        # critical section small.  When not provided, we fetch it here (legacy / standalone path).
        if req_info_snapshot is None:
            req_info_snapshot = dict(self._get_req_info(task._unique_rid) or {})
        infos = list(req_info_snapshot.values())
        if isinstance(task, KVSendTask):
            infos = [
                info for info in infos if info.slice_id is None or info.slice_id == task.slice_id
            ]
        else:
            # AUX is session-level. Multiple KV slices can carry the same AUX
            # target, so dispatch at most once per receiver rank.
            by_rank = {}
            for info in infos:
                by_rank.setdefault(info.instance_rank, info)
            infos = list(by_rank.values())
        errors = []
        for info in infos:
            error = self._dispatch_operation(task, info)
            if error is not None:
                errors.append(error)
        if errors:
            raise errors[0]

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

        self._registrar.register(ri.instance_name, ri.instance_rank, ri)

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
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                return
            unique_rid = int(message[1])
            with self._sessions_lock:
                # Keep the tombstone durable. A cancel can arrive before both the
                # TxSession and REQUEST_DATA; either late arrival must still close
                # without accessing the published destination.
                self._pre_cancelled_rids.add(unique_rid)
                session = self._sessions.get(unique_rid)
                if session is None:
                    with self._peer_requests_lock:
                        req_infos = list(self._peer_requests.get(unique_rid, {}).values())
                else:
                    req_infos = []
            if session is not None:
                session.cancel()
                return
            aux_receivers: set[SendOperationKey] = set()
            for info in req_infos:
                self._send_failed_result_to_receiver(info)
                operation_key = self._operation_key(info)
                if info.aux_slot is not None and operation_key not in aux_receivers:
                    self._send_aux_failed_result_to_receiver(info)
                    aux_receivers.add(operation_key)

    def cancel_session(self, session: "TxSession") -> None:
        """Durably seal a local sender session and settle every unused channel.

        This is invoked for both local and remote cancellation. Installing the
        tombstone before result/control I/O makes a later REQUEST_DATA replay
        the same no-access decision after the TxSession has been removed.
        """
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_shutdown_complete", False) or getattr(
                self, "_dealer_admission_closed", False
            ):
                return
            unique_rid = session.disagg_request_id
            with self._sessions_lock:
                self._pre_cancelled_rids.add(unique_rid)
                with self._peer_requests_lock:
                    req_infos = list(self._peer_requests.get(unique_rid, {}).values())

            aux_receivers: set[SendOperationKey] = set()
            for info in req_infos:
                settle_aux = self._operation_key(info) not in aux_receivers
                self._settle_cancelled_info(session, info, settle_aux=settle_aux)
                if info.aux_slot is not None:
                    aux_receivers.add(self._operation_key(info))
            self.send_cancel_to_receivers(unique_rid)

    def _settle_cancelled_info(
        self,
        session: "TxSession",
        info: RecvReqInfo,
        *,
        settle_aux: bool,
        force_no_access: bool = False,
    ) -> None:
        """Replay terminal state or emit no-access for one advertised target."""
        with session.lock:
            kv_tasks = [
                task
                for task in session.kv_tasks
                if info.slice_id is None or task.slice_id == info.slice_id
            ]
            aux_task = session.aux_task

        if kv_tasks:
            for task in kv_tasks:
                error = self._dispatch_operation(task, info, force_no_access=force_no_access)
                if error is not None:
                    logger.error(
                        f"failed to settle cancelled KV operation for rid={info.unique_rid}: "
                        f"{error}"
                    )
        else:
            slice_id = info.slice_id if info.slice_id is not None else 0
            self._send_unbound_terminal_result(
                session,
                info,
                WriteMetaType.KV,
                tuple(
                    _make_kv_result_msg(
                        self._instance_rank,
                        info.unique_rid,
                        slice_id,
                        True,
                        AgentResult.FAILED,
                    )
                ),
            )

        if settle_aux and info.aux_slot is not None:
            if aux_task is None:
                self._send_unbound_terminal_result(
                    session,
                    info,
                    WriteMetaType.AUX,
                    (
                        MessageType.AUX_AGENT_RESULT,
                        str(self._instance_rank).encode("ascii"),
                        str(info.unique_rid).encode("ascii"),
                        AgentResult.FAILED.value.encode("ascii"),
                    ),
                )
            else:
                error = self._dispatch_operation(aux_task, info, force_no_access=force_no_access)
                if error is not None:
                    logger.error(
                        f"failed to settle cancelled AUX operation for rid={info.unique_rid}: "
                        f"{error}"
                    )

    @nvtx_range("_respond_with_kv")
    def _respond_with_kv(self, _send_id: bytes, message: list[bytes]):
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                return
            self._respond_with_kv_admitted(_send_id, message)

    def _respond_with_kv_admitted(self, _send_id: bytes, message: list[bytes]):
        # _sessions_lock prevents a race between session lookup and req_info save.
        # session.lock serializes _enqueue calls from both paths.
        info: RecvReqInfo = RecvReqInfo.from_bytes(message[1])
        with self._sessions_lock:
            shutting_down = getattr(self, "_shutdown", False)
            if shutting_down:
                # A request admitted after shutdown began is a durable no-access
                # decision. Keep the tombstone even if no local session exists.
                self._pre_cancelled_rids.add(info.unique_rid)
            pre_cancelled = info.unique_rid in self._pre_cancelled_rids
            session = self._get_session(info.unique_rid)
            if shutting_down:
                try:
                    self._save_peer_req_info(info)
                except Exception as e:
                    # Readiness computation is irrelevant after admission has
                    # closed. Preserve the address-bearing request so the
                    # durable no-access result can still be retried.
                    self._add_req_info(info.unique_rid, info)
                    logger.warning(
                        f"Sender shutdown could not compute readiness for late "
                        f"request {info.unique_rid}: {e}"
                    )
            else:
                self._save_peer_req_info(info)
        if session is None:
            if pre_cancelled:
                self._send_failed_result_to_receiver(info)
                if info.aux_slot is not None:
                    self._send_aux_failed_result_to_receiver(info)
            return
        if shutting_down:
            self._settle_cancelled_info(
                session,
                info,
                settle_aux=True,
                force_no_access=True,
            )
            return
        if pre_cancelled:
            # setup_session() normally performs this transition. Handle the
            # constructor/listener race directly so no operation can slip in
            # between tombstone observation and session cancellation.
            session.cancel()
            return
        with session.lock:
            tasks = [
                task
                for task in session.kv_tasks
                if info.slice_id is None or task.slice_id == info.slice_id
            ]
            terminal = session.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
            aux_task = session.aux_task
        if terminal:
            self._settle_cancelled_info(session, info, settle_aux=True)
            return
        for task in tasks:
            error = self._dispatch_operation(task, info)
            if error is not None:
                logger.error(f"failed to dispatch KV operation for rid={info.unique_rid}: {error}")
        if aux_task is not None and info.aux_slot is not None:
            error = self._dispatch_operation(aux_task, info)
            if error is not None:
                logger.error(f"failed to dispatch AUX operation for rid={info.unique_rid}: {error}")

    def _send_failed_result_to_receiver(self, info: RecvReqInfo):
        slice_id = info.slice_id if info.slice_id is not None else 0
        self._send_pre_session_terminal_result(
            info,
            WriteMetaType.KV,
            tuple(
                _make_kv_result_msg(
                    self._instance_rank,
                    info.unique_rid,
                    slice_id,
                    True,  # is_last_slice
                    AgentResult.FAILED,
                )
            ),
        )

    def _send_aux_failed_result_to_receiver(self, info: RecvReqInfo) -> None:
        """Report a terminal AUX failure when cancellation prevented access."""
        self._send_pre_session_terminal_result(
            info,
            WriteMetaType.AUX,
            (
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(info.unique_rid).encode("ascii"),
                AgentResult.FAILED.value.encode("ascii"),
            ),
        )

    def _get_or_connect_dealer(self, endpoint: Optional[str]):
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                raise RuntimeError("Sender control dealers are closed for shutdown")
            if endpoint is None:
                raise ValueError("Sender: peer endpoint is None; peer may not have registered yet")
            lock = getattr(self, "_dealers_lock", None)
            if lock is None:
                if endpoint not in self._dealers:
                    self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
                return self._dealers[endpoint]
            with lock:
                if endpoint not in self._dealers:
                    self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
                return self._dealers[endpoint]

    def _save_peer_req_info(self, peer_transfer_req_info: RecvReqInfo):
        req_info = peer_transfer_req_info
        self._add_req_info(req_info.unique_rid, req_info)
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        expected_transfers = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
        if self._is_req_ready(req_info.unique_rid, expected_transfers, req_info.slice_id):
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
        return self._is_req_ready(req_info.unique_rid, expected_transfers, req_info.slice_id)

    def clear_session(self, unique_rid: int):
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            with self._sessions_lock:
                if unique_rid in self._sessions:
                    del self._sessions[unique_rid]
            self._remove_req_info(unique_rid)

    def send_cancel_to_receivers(self, unique_rid: int) -> None:
        """Notify all receivers involved in this session to cancel."""
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self, "_dealer_admission_closed", False):
                return
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
                    self._get_or_connect_dealer(peer_ri.self_endpoint).send(
                        [MessageType.CANCEL_SESSION, str(unique_rid).encode("ascii")]
                    )
                except Exception as e:
                    logger.warning(f"send_cancel_to_receivers: failed for rid={unique_rid}: {e}")

    def shutdown(self) -> bool:
        """Stop local send work without invalidating resources under live workers."""
        attempt_lock = getattr(self, "_shutdown_attempt_lock", None)
        if attempt_lock is None:
            attempt_lock = threading.Lock()
            self._shutdown_attempt_lock = attempt_lock
        with attempt_lock:
            return self._shutdown_locked()

    def _shutdown_locked(self) -> bool:
        if getattr(self, "_shutdown_complete", False):
            return True
        cancel_failed_session_ids: set[int] = set()
        admission_lock = getattr(self, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            # TxSession creation/send/cancel shares this gate, so shutdown seals
            # every registered session before listener and worker teardown.
            self._shutdown = True
            sessions_lock = getattr(self, "_sessions_lock", None)
            if sessions_lock is None:
                sessions = []
            else:
                with sessions_lock:
                    sessions = list(self._sessions.values())
            for session in sessions:
                try:
                    session.cancel()
                except Exception as e:
                    cancel_failed_session_ids.add(id(session))
                    logger.error(
                        f"Sender.shutdown: failed to seal TxSession "
                        f"{session.disagg_request_id}: {e}"
                    )
        if not getattr(self, "_listener_stopped", False):
            # Quiesce listener before invalidate to avoid set/map mutation
            # races.  Do not advance teardown if stop fails; retry must call it
            # again rather than treating the listener as quiescent.
            try:
                self._messenger.stop()
            except Exception as e:
                logger.error(f"Sender.shutdown: listener stop failed: {e}")
                return False
            self._listener_stopped = True
        if not getattr(self, "_shutdown_sentinels_sent", False):
            admission_lock = getattr(self, "_operation_admission_lock", None)
            with admission_lock if admission_lock is not None else nullcontext():
                for q in self._send_task_queues:
                    q.put(None)
                self._shutdown_sentinels_sent = True
        for t in self._worker_threads:
            t.join(timeout=5)
        workers_alive = any(t.is_alive() for t in self._worker_threads)
        progress_failed = bool(cancel_failed_session_ids) or workers_alive
        if workers_alive:
            logger.error(
                "Sender.shutdown: worker threads are still active; retaining remote agents"
            )

        # Worker-local DEALER sockets are detached by their owning threads.
        # A failed close is transferred to this Sender and remains here until
        # a retry positively closes it.
        failed_thread_lock = getattr(self, "_failed_thread_dealers_lock", None)
        if failed_thread_lock is None:
            failed_thread_dealers = list(getattr(self, "_failed_thread_dealers", ()))
        else:
            with failed_thread_lock:
                failed_thread_dealers = list(self._failed_thread_dealers)
        thread_dealer_failed = False
        for dealer in failed_thread_dealers:
            try:
                dealer.stop()
                if failed_thread_lock is None:
                    self._failed_thread_dealers = [
                        item for item in self._failed_thread_dealers if item is not dealer
                    ]
                else:
                    with failed_thread_lock:
                        self._failed_thread_dealers = [
                            item for item in self._failed_thread_dealers if item is not dealer
                        ]
            except Exception as e:
                thread_dealer_failed = True
                logger.warning(f"Failed to retry worker-local dealer shutdown: {e}")
        progress_failed = progress_failed or thread_dealer_failed

        in_doubt_lock = getattr(self, "_in_doubt_transfers_lock", None)
        in_doubt_count = 0
        if in_doubt_lock is not None:
            with in_doubt_lock:
                in_doubt_count = len(self._in_doubt_transfers)
            if in_doubt_count:
                logger.critical(
                    f"Sender.shutdown: retaining agent and source owners for "
                    f"{in_doubt_count} in-doubt transfer(s)"
                )
                progress_failed = True
        with admission_lock if admission_lock is not None else nullcontext():
            sessions_lock = getattr(self, "_sessions_lock", None)
            if sessions_lock is None:
                sessions = []
            else:
                with sessions_lock:
                    sessions = list(self._sessions.values())
            retry_failed_session_ids: set[int] = set()
            for session in sessions:
                try:
                    self.retry_terminal_results(session, force=True)
                except Exception as e:
                    retry_failed_session_ids.add(id(session))
                    progress_failed = True
                    logger.error(
                        f"Sender.shutdown: failed to retry terminal results for "
                        f"TxSession {session.disagg_request_id}: {e}"
                    )
            try:
                self._retry_pre_session_terminal_results(force=True)
            except Exception as e:
                progress_failed = True
                logger.error(f"Sender.shutdown: failed to retry pre-session terminal results: {e}")
            active_session_ids: set[int] = set()
            for session in sessions:
                try:
                    active = session.has_transferring_tasks()
                except Exception as e:
                    active = True
                    logger.error(
                        f"Sender.shutdown: failed to inspect TxSession "
                        f"{session.disagg_request_id}: {e}"
                    )
                if active:
                    active_session_ids.add(id(session))
                    progress_failed = True
                    logger.critical(
                        f"Sender.shutdown: retaining agent for active source request "
                        f"{session.disagg_request_id}"
                    )
            pre_session_results_pending = self._has_pending_pre_session_terminal_results()
            if pre_session_results_pending:
                progress_failed = True
                logger.critical(
                    "Sender.shutdown: retaining control dealers for pending pre-session results"
                )
            for session in sessions:
                session_id = id(session)
                if session_id in (
                    cancel_failed_session_ids | retry_failed_session_ids | active_session_ids
                ):
                    continue
                try:
                    session.close()
                except Exception as e:
                    progress_failed = True
                    logger.error(
                        f"Sender.shutdown: failed to retire TxSession "
                        f"{session.disagg_request_id}: {e}"
                    )
                    continue
                # TxSession.close() normally unregisters itself. Keep shutdown
                # aggregation correct for compatible session implementations
                # that only release their own resources.
                self.clear_session(session.disagg_request_id)
            if not progress_failed:
                # No later listener callback or session mutation can create
                # another control socket/result obligation after this gate.
                self._dealer_admission_closed = True

        if progress_failed:
            return False

        invalidation_failed = False
        with self._loaded_remote_agents_lock:
            loaded_agents = list(self._loaded_remote_agents)
        # Invalidate all loaded remote agents to release fabric/POSIX FD resources.
        for agent_name in loaded_agents:
            try:
                self._agent.invalidate_remote_agent(agent_name)
                with self._loaded_remote_agents_lock:
                    self._loaded_remote_agents.discard(agent_name)
            except Exception as e:
                invalidation_failed = True
                logger.warning(
                    f"Failed to invalidate remote agent '{agent_name}' during shutdown: {e}"
                )
        if invalidation_failed:
            return False
        dealer_failed = False
        dealers_lock = getattr(self, "_dealers_lock", None)
        if dealers_lock is None:
            dealer_items = list(self._dealers.items())
        else:
            with dealers_lock:
                dealer_items = list(self._dealers.items())
        for endpoint, dealer in dealer_items:
            try:
                dealer.stop()
                if dealers_lock is None:
                    if self._dealers.get(endpoint) is dealer:
                        self._dealers.pop(endpoint, None)
                    else:
                        dealer_failed = True
                else:
                    with dealers_lock:
                        if self._dealers.get(endpoint) is dealer:
                            self._dealers.pop(endpoint, None)
                        else:
                            dealer_failed = True
            except Exception as e:
                dealer_failed = True
                logger.warning(f"Failed to stop dealer during Sender shutdown: {e}")
        if dealer_failed:
            return False
        self._shutdown_complete = True
        return True

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Sender.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        try:
            drained = self.shutdown()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error(f"Sender context cleanup failed while propagating an exception: {e}")
            return False
        if not drained:
            if exc_type is None:
                raise RuntimeError("Sender context exited before transfer resources drained")
            logger.error("Sender context retained resources while propagating an exception")
        return False


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
        source_owner: Optional[LlmRequest] = None,
    ):
        # Keep the allocator-level request owner alive until every source
        # operation is terminal. Enroll it before fallible initialization,
        # then clear the parameter so propagated tracebacks are ownership-
        # neutral.
        self._source_owner = source_owner
        source_owner = None
        self._aux_buffer = aux_buffer
        self.aux_slot = None
        try:
            super().__init__(
                sender,
                SessionArgsBase(params, prompt_len=prompt_len, beam_width=beam_width),
            )
            self._timeout_s = timeout_s
            self._need_aux = params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
            self._sender: Sender  # narrow base class type for Pylance
            self.request_id = request_id
            self.aux_slot = aux_buffer.alloc_slot().id if aux_buffer is not None else None
            # AUX contents become immutable once an AuxSendTask is published.
            # This prevents fill_slot() from racing a worker/NIXL read of the
            # same slot.
            self._aux_packed = False
            self.receiver_ready: bool = False
            self.kv_tasks = []
            self.aux_task = None
            self.lock = threading.Lock()
            self._close_lock = threading.Lock()

            self._exception: Optional[Exception] = None
            self._closed = False
            self._accepting_operations = True
            self._terminal_status: Optional[SessionStatus] = None
            # Cancellation can precede task creation. Keep those no-access
            # frames in the session lifecycle so a transient send failure
            # cannot be discarded by close().
            self._unbound_terminal_results: dict[UnboundTerminalKey, UnboundTerminalResult] = {}
            self._terminal_retry_lock = threading.Lock()
            self._next_terminal_retry_at = 0.0
            # Must be last: makes session visible to listener thread, so all
            # attributes above must be initialized first.
            self._sender.setup_session(self)
        except Exception:
            self._source_owner = None
            if self._aux_buffer is not None and self.aux_slot is not None:
                self._aux_buffer.free_slot(self.aux_slot)
                self.aux_slot = None
            self._closed = True
            raise

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
    def source_owner(self) -> Optional[LlmRequest]:
        return self._source_owner

    @staticmethod
    def _unbound_terminal_key(
        meta_type: WriteMetaType, req_info: RecvReqInfo
    ) -> UnboundTerminalKey:
        slice_id = req_info.slice_id if meta_type is WriteMetaType.KV else None
        return meta_type, req_info.instance_name, req_info.instance_rank, slice_id

    def cache_unbound_terminal_result(
        self,
        meta_type: WriteMetaType,
        req_info: RecvReqInfo,
        message: tuple[bytes, ...],
    ) -> UnboundTerminalKey:
        key = self._unbound_terminal_key(meta_type, req_info)
        with self.lock:
            record = self._unbound_terminal_results.get(key)
            if record is None:
                self._unbound_terminal_results[key] = UnboundTerminalResult(
                    req_info=req_info,
                    message=message,
                )
            elif record.message != message:
                raise RuntimeError(f"conflicting unbound terminal result for {key}")
        return key

    def mark_unbound_terminal_result_delivered(self, key: UnboundTerminalKey) -> None:
        with self.lock:
            record = self._unbound_terminal_results.get(key)
            if record is None:
                raise RuntimeError(f"unbound terminal result {key} was not recorded")
            record.result_delivered = True

    def _retry_terminal_results(self) -> None:
        sender = getattr(self, "_sender", None)
        retry = getattr(sender, "retry_terminal_results", None)
        if callable(retry):
            retry(self)

    @property
    def status(self) -> SessionStatus:
        if self._terminal_status is not None:
            return self._terminal_status
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
        admission_lock = getattr(self._sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self._sender, "_shutdown", False):
                raise RuntimeError("Sender is shutting down; new KV operations are not accepted")
            with self.lock:
                if not self._accepting_operations or self._closed:
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} is closed to new KV operations"
                    )
                params = self._base_args.params
                slice_id = len(self.kv_tasks)
                task = KVSendTask(
                    slice,
                    params,
                    slice_id,
                    prompt_len=self._base_args.prompt_len,
                    beam_width=self._base_args.beam_width,
                    session=self,
                )
                task._unique_rid = self.disagg_request_id
                self.kv_tasks.append(task)
                req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
            self._sender.dispatch_task(task, req_info_snapshot)

    def send_aux(self) -> AuxSendTask:
        admission_lock = getattr(self._sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            if getattr(self._sender, "_shutdown", False):
                raise RuntimeError("Sender is shutting down; new AUX operations are not accepted")
            with self.lock:
                if not self._accepting_operations or self._closed:
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} is closed to new AUX operations"
                    )
                if self.aux_task is not None:
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} already has an AUX operation"
                    )
                if not getattr(self, "_aux_packed", False):
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} AUX data must be packed before send"
                    )
                params = self._base_args.params
                task = AuxSendTask(params, self.aux_slot, session=self)
                task._unique_rid = self.disagg_request_id
                self.aux_task = task
                req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
            self._sender.dispatch_task(task, req_info_snapshot)
            return task

    def pack_aux(self, request: LlmRequest) -> None:
        """Fill the aux buffer slot with token data from the given request."""
        admission_lock = getattr(self._sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            with self.lock:
                if (
                    not self._accepting_operations
                    or self._closed
                    or self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
                    or getattr(self._sender, "_shutdown", False)
                ):
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} is closed to AUX packing"
                    )
                if self.aux_task is not None:
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} AUX data is frozen after send"
                    )
                if getattr(self, "_aux_packed", False):
                    raise RuntimeError(
                        f"TxSession {self.disagg_request_id} AUX data is already packed"
                    )
                aux_buffer = self._aux_buffer
                aux_slot = self.aux_slot
            assert aux_buffer is not None, "No aux_buffer set for this session"
            assert aux_slot is not None, "No aux_slot set for this session"
            aux_buffer.fill_slot(aux_slot, request)
            with self.lock:
                self._aux_packed = True

    def is_completed(self) -> bool:
        """Non-blocking check: has the transfer completed successfully?"""
        self._retry_terminal_results()
        if self.has_transferring_tasks():
            return False
        status = self.status
        if self._need_aux:
            return status == SessionStatus.FULLY_TRANSFERRED
        return status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)

    def has_failed(self) -> bool:
        """Non-blocking check: has the transfer failed or been cancelled?"""
        self._retry_terminal_results()
        # A logical failure is not safe request-cleanup evidence while a local
        # gather/NIXL worker may still read source KV pages.
        if self.has_transferring_tasks():
            return False
        if self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
            return True
        if any(task.status == TaskStatus.ERROR for task in self.kv_tasks):
            return True
        return self.aux_task is not None and self.aux_task.status == TaskStatus.ERROR

    def cancel(self) -> None:
        """Cancel the session and notify the remote receiver.

        Safe to call multiple times. Active physical work keeps running, but
        every unfinished task latches logical failure immediately. Source
        access and pending-result ledgers remain authoritative for retirement.
        The lock serializes with _deliver_kv_to_agent() so has_transferring_tasks()
        is accurate the moment this returns.
        """
        admission_lock = getattr(self._sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            self._latch_terminal_failure(
                SessionStatus.CANCELLED,
                RuntimeError(f"TxSession {self.disagg_request_id} cancelled"),
                record_exception=False,
            )
            # Tombstone, settle unused channels, and perform I/O outside the
            # session lock. Repeated cancel retries cached terminal decisions.
            self._sender.cancel_session(self)

    def has_transferring_tasks(self) -> bool:
        """True while source access or terminal-result delivery is outstanding.

        cancel_request() must return False while this is True.
        """
        bound_pending = any(
            getattr(t, "source_access_active", False)
            or t.status == TaskStatus.TRANSFERRING
            or getattr(t, "has_pending_result_delivery", False)
            for t in self.kv_tasks
        ) or (
            self.aux_task is not None
            and (
                getattr(self.aux_task, "source_access_active", False)
                or self.aux_task.status == TaskStatus.TRANSFERRING
                or getattr(self.aux_task, "has_pending_result_delivery", False)
            )
        )
        unbound_results = getattr(self, "_unbound_terminal_results", {})
        lock = getattr(self, "lock", None)
        if lock is None:
            unbound_pending = any(
                not record.result_delivered for record in unbound_results.values()
            )
        else:
            with lock:
                unbound_pending = any(
                    not record.result_delivered for record in unbound_results.values()
                )
        return bound_pending or unbound_pending

    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        """Poll or block until KV (and optionally aux) transfer finishes.

        With blocking=True (default): waits up to _timeout_s for each task.
        With blocking=False: polls non-blockingly; returns None if any KV task
        or aux is not yet done.
        """
        self._retry_terminal_results()
        terminal_failure = self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)
        if terminal_failure and not self.has_transferring_tasks():
            return WaitResult.FAILED
        if not self.kv_tasks:
            return None
        if not blocking:
            if self.has_transferring_tasks():
                return None
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
            return WaitResult.FAILED if terminal_failure else WaitResult.COMPLETED

        for task in self.kv_tasks:
            if not task.wait(timeout=self._timeout_s):
                return WaitResult.TIMEOUT
            if task.status == TaskStatus.ERROR:
                return None if self.has_transferring_tasks() else WaitResult.FAILED
        if self._need_aux and self.aux_task is not None:
            if not self.aux_task.wait(timeout=self._timeout_s):
                return WaitResult.TIMEOUT
            if self.aux_task.status == TaskStatus.ERROR:
                return None if self.has_transferring_tasks() else WaitResult.FAILED
        if self.has_transferring_tasks():
            return None
        return WaitResult.FAILED if terminal_failure else WaitResult.COMPLETED

    def _latch_terminal_failure(
        self,
        status: SessionStatus,
        exception: Exception,
        *,
        record_exception: bool,
    ) -> bool:
        """Atomically publish the first terminal failure and fail open tasks."""
        if status not in (SessionStatus.ERROR, SessionStatus.CANCELLED):
            raise ValueError(f"invalid terminal failure status {status}")
        with self.lock:
            self._accepting_operations = False
            if self._terminal_status is not None:
                return False
            self._terminal_status = status
            if record_exception:
                self._exception = exception
            for task in self.kv_tasks:
                if not task.is_done:
                    task.fail(exception)
            if self.aux_task is not None and not self.aux_task.is_done:
                self.aux_task.fail(exception)
            return True

    def set_exception(self, reason: str = ""):
        msg = f"TxSession {self.disagg_request_id} exception"
        if reason:
            msg += f": {reason}"
        sender = getattr(self, "_sender", None)
        admission_lock = getattr(sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            self._latch_terminal_failure(
                SessionStatus.ERROR,
                RuntimeError(msg),
                record_exception=True,
            )
            # ERROR seals admission just like cancellation. Install the durable
            # tombstone outside the session lock so late REQUEST_DATA is settled.
            if sender is not None:
                sender.cancel_session(self)

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        # Constructor rollback marks the partially initialized object closed.
        # Check that terminal marker before touching locks or retry ledgers that
        # may not have been created yet; __del__ can run after any init failure.
        if getattr(self, "_closed", False):
            return
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = threading.Lock()
            self._close_lock = close_lock
        sender = getattr(self, "_sender", None)
        admission_lock = getattr(sender, "_operation_admission_lock", None)
        with admission_lock if admission_lock is not None else nullcontext():
            with close_lock:
                self._close_locked()

    def _close_locked(self):
        if getattr(self, "_closed", False):
            return
        self._retry_terminal_results()
        with self.lock:
            if getattr(self, "_closed", False):
                return
            self._accepting_operations = False
            source_tasks = list(self.kv_tasks)
            if self.aux_task is not None:
                source_tasks.append(self.aux_task)
            if any(
                getattr(task, "source_access_active", False)
                or task.status in (TaskStatus.INIT, TaskStatus.TRANSFERRING)
                or getattr(task, "has_pending_result_delivery", False)
                for task in source_tasks
            ) or any(
                not record.result_delivered
                for record in getattr(self, "_unbound_terminal_results", {}).values()
            ):
                raise RuntimeError(
                    f"cannot close TxSession {self.disagg_request_id}: "
                    "source work is pending or active"
                )
        if self._aux_buffer is not None and self.aux_slot is not None:
            self._aux_buffer.free_slot(self.aux_slot)
            self.aux_slot = None
        # Unregister from Sender; keep fields alive for in-flight worker threads.
        if self._sender is not None:
            self._sender.clear_session(self.disagg_request_id)
        with self.lock:
            self._source_owner = None
            self._closed = True

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
        # The exact-writer registry is enabled only when ctx_dp_rank selects one
        # deterministic sender cohort. ADP broadcast remains on the legacy path
        # until the protocol selects its writer cohort before address publication.
        self.lifecycle_managed = False
        # Legacy ADP broadcast cannot name the selected writer cohort before
        # publication. Keep a conservative endpoint-local drain record so a
        # first failure cannot release destination memory while sibling writers
        # may still be active.
        self.publication_started = False
        self._legacy_results: dict[int, AgentResult] = {}
        self._legacy_result_conflict = False
        self._legacy_backend_quiesced = False
        # One exact set for deterministic ctx_dp_rank, or one candidate set per
        # DP group for ADP broadcast.  A legacy target is drained only when the
        # observed terminal identities equal one complete candidate cohort.
        self._valid_writer_cohorts: tuple[frozenset[int], ...] = ()
        self._exposed_writer_ranks: set[int] = set()
        self._publication_closed = False

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

    @property
    def legacy_resources_drained(self) -> bool:
        """Whether an unregistered ADP receive can no longer be accessed.

        This is a compatibility path, not the exact-writer ownership contract.
        It preserves the existing ADP assumption that exactly one advertised
        candidate cohort performs the transfer. Full attention-DP containment
        requires a negotiated writer-selection protocol before publication.
        """
        if self.lifecycle_managed or not self.publication_started or self._legacy_backend_quiesced:
            return True
        return (
            self._publication_closed
            and not self._legacy_result_conflict
            and len(self.complete_writer_cohorts(self._legacy_results)) == 1
        )

    def set_valid_writer_cohorts(self, cohorts) -> None:
        """Install the exact writer identities that may form one transfer."""
        normalized: list[frozenset[int]] = []
        for cohort in cohorts:
            ranks = frozenset(int(rank) for rank in cohort)
            if not ranks:
                raise ValueError("writer cohort must not be empty")
            if any(rank < 0 for rank in ranks):
                raise ValueError("writer cohort ranks must be non-negative")
            if ranks not in normalized:
                normalized.append(ranks)
        if not normalized:
            raise ValueError("at least one writer cohort is required")
        self._valid_writer_cohorts = tuple(normalized)
        self.expected_transfers = len(normalized[0])

    def writer_identities_valid(self, peer_rank: int, existing_ranks) -> bool:
        """Whether ``peer_rank`` was one of the broadcast address recipients."""
        del existing_ranks
        return int(peer_rank) in self._exposed_writer_ranks

    def is_complete_writer_cohort(self, ranks) -> bool:
        """Whether results identify exactly one candidate cohort."""
        if not self._publication_closed:
            return False
        identities = frozenset(ranks)
        return any(cohort == identities for cohort in self._valid_writer_cohorts)

    def complete_writer_cohorts(self, ranks) -> tuple[frozenset[int], ...]:
        """Return the candidate cohort exactly named by the result identities."""
        if not self._publication_closed:
            return ()
        identities = frozenset(ranks)
        return tuple(cohort for cohort in self._valid_writer_cohorts if cohort == identities)

    def mark_writer_exposed(self, peer_rank: int) -> None:
        """Record that this request may have carried KV and AUX addresses."""
        self.publication_started = True
        self._exposed_writer_ranks.add(int(peer_rank))

    def close_publication(self) -> None:
        """Prove that dispatch will advertise no additional writer targets."""
        self._publication_closed = True

    def record_legacy_result(self, peer_rank: int, status: AgentResult) -> bool:
        """Record a legacy terminal result; return whether it is a new result."""
        previous = self._legacy_results.get(peer_rank)
        if previous is not None:
            if previous is not status:
                self._legacy_result_conflict = True
            return False
        self._legacy_results[peer_rank] = status
        if not self.writer_identities_valid(peer_rank, self._legacy_results):
            self._legacy_result_conflict = True
        return True

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
    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        agent: BaseTransferAgent,
        bounce=None,
    ):
        self._registrar = peer_registrar
        self._agent = agent
        self._bounce = bounce
        self._dealers = {}
        self._dealers_lock = threading.Lock()
        self._dealer_admission_open = True
        self._shutdown_attempt_lock = threading.Lock()
        self._sender_ep_instance_map = {}

        self._messenger = ZMQMessenger(mode="ROUTER")
        # Strong endpoint ownership is intentional. A receive session carrying
        # an AUX/legacy target must not disappear while a remote writer may
        # still access it; clear_session removes it only after drain proof.
        self._sessions: dict[int, RxSession] = {}
        self._sessions_lock = threading.Lock()  # Protects _sessions and _pre_cancelled_rids
        self._pre_cancelled_rids: set[int] = set()
        self._recv_registry = RecvTransferRegistry()
        # Bounce settlement retries must replay the transition that performed
        # physical finalization. Calling finish_bounce_scatter() again only
        # returns a duplicate update because its one-shot actions have already
        # been consumed by the registry.
        self._bounce_lifecycle_delivery_lock = threading.Lock()
        self._pending_bounce_lifecycle_deliveries: dict[
            tuple[int, int], _PendingLifecycleDelivery
        ] = {}
        self._pending_bounce_logical_failure_deliveries: dict[
            tuple[int, int], _PendingLifecycleDelivery
        ] = {}
        self._shutdown_started = False
        self._listener_stopped = False
        self._shutdown = False

        self._start_listener()
        logger.info(f"Receiver init with endpoint: {self._messenger.endpoint}")

    @property
    def endpoint(self):
        return self._messenger.endpoint

    @property
    def transfers_drained(self) -> bool:
        """Whether all managed, legacy, and AUX receive targets are retired."""
        try:
            if not self._bounce.retry_settlements():
                return False
        except Exception as e:
            logger.error(f"failed to retry receiver bounce settlements: {e}")
            return False
        if not self._recv_registry.is_drained():
            return False
        with self._sessions_lock:
            entries = list(self._sessions.values())
        for entry in entries:
            session = entry() if isinstance(entry, weakref.ReferenceType) else entry
            if session is None:
                # A lost legacy/AUX owner is not positive quiescence evidence.
                return False
            if session.has_untracked_receive_activity():
                return False
        return True

    def begin_shutdown(self) -> None:
        """Close receive admission while keeping result routing alive.

        TransferWorker calls this before quiescing the backend. Published
        writers continue to own their targets until either their terminal
        result arrives or backend-wide quiescence is proven.
        """
        errors: list[tuple[str, Exception]] = []
        with self._sessions_lock:
            # Session enrollment and the shutdown snapshot share this lock, so
            # no ADP compatibility session can appear after admission closes.
            self._shutdown_started = True
            entries = list(self._sessions.values())
        for entry in entries:
            session = entry() if isinstance(entry, weakref.ReferenceType) else entry
            if session is not None:
                try:
                    seal_admission = getattr(session, "seal_receive_admission", None)
                    if seal_admission is not None:
                        seal_admission()
                    session.cancel()
                except Exception as e:
                    errors.append((f"session {session.disagg_request_id}", e))
                    logger.error(
                        f"Receiver.begin_shutdown: cancellation failed for "
                        f"request {session.disagg_request_id}: {e}"
                    )
        try:
            updates = self._recv_registry.begin_shutdown()
        except Exception as e:
            updates = ()
            errors.append(("receive registry", e))
            logger.error(f"Receiver.begin_shutdown: registry cancellation failed: {e}")
        for update in updates:
            try:
                if update.physical_state is not PhysicalState.DRAINED:
                    self._bounce.mark_logical_failure(
                        update.key,
                        on_done=lambda succeeded, key=update.key: self._finish_bounce(
                            key, succeeded
                        ),
                    )
            except Exception as e:
                errors.append((f"receive context {update.key}", e))
                logger.error(
                    f"Receiver.begin_shutdown: physical cancellation failed for "
                    f"request {update.key[0]} slice={update.key[1]}: {e}"
                )
            try:
                self._handle_lifecycle_update(update)
            except Exception as e:
                errors.append((f"receive notification {update.key}", e))
                logger.error(
                    f"Receiver.begin_shutdown: lifecycle notification failed for "
                    f"request {update.key[0]} slice={update.key[1]}: {e}"
                )
        if errors:
            detail = "; ".join(f"{owner}: {error}" for owner, error in errors)
            raise RuntimeError(
                f"Receiver.begin_shutdown encountered {len(errors)} error(s): {detail}"
            ) from errors[0][1]

    def mark_backend_quiesced(self) -> None:
        """Apply an externally proven remote/global writer fence.

        A local ``BaseTransferAgent.shutdown()`` is explicitly insufficient:
        callers must first establish that peer processes and the fabric can no
        longer submit one-sided writes to any advertised target.
        """
        with self._sessions_lock:
            sessions = tuple(self._sessions.values())
        for entry in sessions:
            session = entry() if isinstance(entry, weakref.ReferenceType) else entry
            if session is not None:
                session.mark_backend_quiesced()
        # Mark registry writers quiescent first.  Bounce settlement then calls
        # back into that registry, after which request cleanup may proceed.
        for update in self._recv_registry.mark_backend_quiesced():
            if LifecycleAction.RELEASE_BOUNCE in update.actions:
                self._bounce.mark_backend_quiesced(
                    update.key,
                    on_done=lambda succeeded, key=update.key: self._finish_bounce(key, succeeded),
                )
            self._handle_lifecycle_update(update)

    def shutdown(self) -> bool:
        attempt_lock = getattr(self, "_shutdown_attempt_lock", None)
        if attempt_lock is None:
            attempt_lock = threading.Lock()
            self._shutdown_attempt_lock = attempt_lock
        with attempt_lock:
            return self._shutdown_serialized()

    def _shutdown_serialized(self) -> bool:
        if getattr(self, "_shutdown", False):
            return True
        self.begin_shutdown()
        if not self.transfers_drained:
            return False
        # Admission is sealed and every admitted dispatch has exited. Close
        # dealer admission and take the final snapshot in one transaction so a
        # late control path cannot create an unowned socket after this point.
        dealers_lock = getattr(self, "_dealers_lock", None)
        if dealers_lock is None:
            self._dealer_admission_open = False
            dealers = list(self._dealers.items())
        else:
            with dealers_lock:
                self._dealer_admission_open = False
                dealers = list(self._dealers.items())
        # No writer remains, so closing result ingress cannot strand ownership.
        if not getattr(self, "_listener_stopped", False):
            try:
                self._messenger.stop()
            except Exception as e:
                logger.error(f"Receiver.shutdown: listener stop failed: {e}")
                return False
            self._listener_stopped = True
        failed = False
        for endpoint, dealer in dealers:
            try:
                dealer.stop()
                if dealers_lock is None:
                    current = self._dealers.get(endpoint)
                    if current is dealer:
                        self._dealers.pop(endpoint, None)
                    elif current is not None:
                        failed = True
                else:
                    with dealers_lock:
                        current = self._dealers.get(endpoint)
                        if current is dealer:
                            self._dealers.pop(endpoint, None)
                        elif current is not None:
                            failed = True
            except Exception as e:
                failed = True
                logger.warning(f"Failed to stop dealer during Receiver shutdown: {e}")
        if failed:
            return False
        self._shutdown = True
        return True

    def clear_session(self, unique_rid: int):
        with self._sessions_lock:
            entry = self._sessions.get(unique_rid)
        session = entry() if isinstance(entry, weakref.ReferenceType) else entry
        # resources_drained() can settle bounce work and route a callback back
        # through Receiver._get_session(). Never call it while holding the
        # session-map lock.
        if session is not None and not session.resources_drained():
            raise RuntimeError(
                f"cannot close RxSession {unique_rid}: "
                "legacy or auxiliary receive resources are not drained"
            )
        with self._sessions_lock:
            if self._sessions.get(unique_rid) is not entry:
                raise RuntimeError(
                    f"cannot close RxSession {unique_rid}: session ownership changed"
                )
            if not self._recv_registry.retire_request_if_drained(unique_rid):
                raise RuntimeError(
                    f"cannot close RxSession {unique_rid}: "
                    "receive transfer resources are not drained"
                )
            self._sessions.pop(unique_rid, None)

    def setup_session(self, rx_session: RxSessionBase):
        unique_rid = rx_session.disagg_request_id
        pre_cancel = False
        with self._sessions_lock:
            if getattr(self, "_shutdown_started", False):
                raise RuntimeError("Receiver is shutting down; new sessions are not accepted")
            existing = self._sessions.get(unique_rid)
            if existing is not None and existing is not rx_session:
                raise RuntimeError(f"RxSession {unique_rid} is already registered")
            self._sessions[unique_rid] = rx_session
            # Without generation-safe replay, cancellation tombstones are
            # durable.  Applying one to this session must not make a later
            # replacement with the same request ID admissible.
            pre_cancel = unique_rid in self._pre_cancelled_rids
        if pre_cancel:
            try:
                rx_session.cancel()
            except Exception:
                # Session enrollment and pre-cancellation are one transaction.
                # Constructor rollback can now release pre-publication local
                # resources, while the durable tombstone protects a retry.
                with self._sessions_lock:
                    if self._sessions.get(unique_rid) is rx_session:
                        self._sessions.pop(unique_rid, None)
                raise

    def record_cancel_tombstone(self, unique_rid: int) -> None:
        """Durably prevent a cancelled request ID from being re-admitted."""
        with self._sessions_lock:
            self._pre_cancelled_rids.add(unique_rid)

    def _get_session(self, unique_rid: Optional[int]) -> Optional["RxSession"]:
        with self._sessions_lock:
            entry = self._sessions.get(unique_rid)
        if entry is None:
            return None
        session = entry() if isinstance(entry, weakref.ReferenceType) else entry
        if session is None:
            logger.warning(f"RxSession {unique_rid} has been garbage collected")
            return None
        return session

    def _handle_lifecycle_update(
        self, update: LifecycleUpdate, peer_rank: Optional[int] = None
    ) -> None:
        """Optionally notify the session after the registry owns the transition."""
        session = self._get_session(update.key[0])
        if session is not None:
            session.process_lifecycle_update(update, peer_rank=peer_rank)

    def _finish_bounce(
        self, key: tuple[int, int], succeeded: bool, peer_rank: Optional[int] = None
    ) -> None:
        """Finalize bounce ownership and durably deliver its lifecycle update.

        The bounce transport retries this callback when consumer delivery
        raises. The registry transition itself is one-shot, so retain the
        original update and replay it until ``process_lifecycle_update``
        returns successfully.
        """
        with self._bounce_lifecycle_delivery_lock:
            delivery = self._pending_bounce_lifecycle_deliveries.get(key)
            if delivery is None:
                delivery = _PendingLifecycleDelivery(
                    update=self._recv_registry.finish_bounce_scatter(key, succeeded),
                    peer_rank=peer_rank,
                    succeeded=succeeded,
                )
                self._pending_bounce_lifecycle_deliveries[key] = delivery
            elif delivery.succeeded != succeeded:
                raise RuntimeError(f"conflicting bounce settlement retry for {key}")

        self._handle_lifecycle_update(delivery.update, peer_rank=delivery.peer_rank)

        with self._bounce_lifecycle_delivery_lock:
            if self._pending_bounce_lifecycle_deliveries.get(key) is delivery:
                self._pending_bounce_lifecycle_deliveries.pop(key, None)

    def _fail_bounce_logically(self, key: tuple[int, int]) -> None:
        """Report local scatter failure without claiming physical settlement.

        The transport retains the bounce slot and destination owner because a
        CUDA error is not a positive stream fence.  Keep the one-shot registry
        update replayable until its optional session consumer accepts it.
        """
        with self._bounce_lifecycle_delivery_lock:
            deliveries = getattr(self, "_pending_bounce_logical_failure_deliveries", None)
            if deliveries is None:
                deliveries = {}
                self._pending_bounce_logical_failure_deliveries = deliveries
            delivery = deliveries.get(key)
            if delivery is None:
                delivery = _PendingLifecycleDelivery(
                    update=self._recv_registry.fail_context(
                        key,
                        "bounce scatter failed without a positive local CUDA fence",
                    ),
                    peer_rank=None,
                    succeeded=False,
                )
                deliveries[key] = delivery

        self._handle_lifecycle_update(delivery.update)

        with self._bounce_lifecycle_delivery_lock:
            if deliveries.get(key) is delivery:
                deliveries.pop(key, None)

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

    @staticmethod
    def _fanin_bounce_safe(overlap) -> bool:
        """Whether equal-size fan-in bounce is safe for this overlap.

        Exact writer ranks do not change the equal-split requirement. TP head
        duplication is unsafe in either direction. PP fan-in is always direct:
        uniform peer PP-stage layer counts do not prove that this receiver's
        local layer interval intersects each writer by an equal number of bytes.
        """
        return (
            overlap.overlap_pp_size == 1
            and overlap.duplicate_head_factor == 1
            and overlap.peer_duplicate_head_factor == 1
        )

    def _single_writer_bounce_exact(self, peer_info: RankInfo) -> bool:
        """Whether the receiver can prove that one writer maps every reserved byte.

        ``IdentityMapper`` proves positional mapping, not physical extent
        equality.  Bounce admission additionally requires identical block
        geometry and a byte-equal bijection over every attention pool view so
        the sender's coalesced write cannot exceed the receiver-owned slot.
        """
        self_ri = self._registrar.self_rank_info
        local_layers = getattr(self_ri, "layer_num_per_pp", None)
        peer_layers = getattr(peer_info, "layer_num_per_pp", None)
        if local_layers is not None and peer_layers is not None:
            if sum(local_layers) != sum(peer_layers):
                return False

        get_pool_mapping = getattr(self._registrar, "get_pool_mapping", None)
        get_kv_map = getattr(self._registrar, "get_kv_map", None)
        page_table = getattr(self_ri, "page_table", None)
        peer_page_table = getattr(peer_info, "page_table", None)
        # If topology metadata is absent, the exact byte contribution is
        # unknowable. Fail closed to the direct descriptor path.
        if (
            not callable(get_pool_mapping)
            or not callable(get_kv_map)
            or page_table is None
            or peer_page_table is None
            or page_table.tokens_per_block != peer_page_table.tokens_per_block
        ):
            return False

        from ..resource.utils import get_physical_pool
        from .mixers.attention.peer import IdentityMapper

        try:
            mapping = get_pool_mapping(peer_info)
            expected_local_pool_keys = {
                (layer_group_id, pool_view_id)
                for layer_group_id, layer_group in enumerate(page_table.layer_groups)
                if isinstance(layer_group, AttentionLayerGroup)
                for pool_view_id, _pool_view in enumerate(layer_group.pool_views)
            }
            expected_peer_pool_keys = {
                (layer_group_id, pool_view_id)
                for layer_group_id, layer_group in enumerate(peer_page_table.layer_groups)
                if isinstance(layer_group, AttentionLayerGroup)
                for pool_view_id, _pool_view in enumerate(layer_group.pool_views)
            }
            mapped_peer_pool_keys = tuple(mapping.values())
            if (
                set(mapping) != expected_local_pool_keys
                or len(mapped_peer_pool_keys) != len(set(mapped_peer_pool_keys))
                or set(mapped_peer_pool_keys) != expected_peer_pool_keys
            ):
                return False

            for self_key, peer_key in mapping.items():
                if not isinstance(get_kv_map(peer_info, self_key, peer_key), IdentityMapper):
                    return False
                self_lg, self_pool_view_id = self_key
                peer_lg, peer_pool_view_id = peer_key
                self_pool_view = page_table.layer_groups[self_lg].pool_views[self_pool_view_id]
                peer_pool_view = peer_page_table.layer_groups[peer_lg].pool_views[peer_pool_view_id]
                self_pool = get_physical_pool(page_table, self_lg, self_pool_view.pool_idx)
                peer_pool = get_physical_pool(peer_page_table, peer_lg, peer_pool_view.pool_idx)
                if self_pool.slot_bytes != peer_pool.slot_bytes:
                    return False
            return True
        except Exception as e:
            logger.warning_once(
                f"Receive bounce disabled because the exact single-writer byte mapping "
                f"could not be proven: {e}",
                key="kv-bounce-single-writer-plan-unproven",
            )
            return False

    def _destination_intervals(self, task: KVRecvTask) -> set[tuple[int, int]]:
        """Build trusted allocation ranges for bounce-tail validation.

        The sender may choose a mapped subset of these blocks, but it may not
        direct local scatter outside the destination request's current KV and
        Mamba slots.  This manifest is derived locally before publication and
        never from the result frame.
        """
        extractor = self._registrar.self_extractor
        page_table = extractor.page_table
        intervals: set[tuple[int, int]] = set()
        for layer_group_id, block_ids in enumerate(task._kv_slice.block_ids_per_layer_groups):
            if layer_group_id >= len(page_table.layer_groups):
                raise ValueError(
                    f"destination layer group {layer_group_id} is absent from the page table"
                )
            layer_group = page_table.layer_groups[layer_group_id]
            if not isinstance(layer_group, AttentionLayerGroup):
                continue
            for pool_view_idx, _pool_view in enumerate(layer_group.pool_views):
                region = extractor.extract(
                    np.asarray(block_ids, dtype=np.int64),
                    layer_group_id=layer_group_id,
                    pool_idx=pool_view_idx,
                )
                size = int(region.memory.bytes_per_region)
                intervals.update((int(ptr), size) for ptr in region.memory.ptrs)

        mamba_slot = task._kv_slice.mamba_state_index
        if mamba_slot is not None:
            for layer_group in page_table.layer_groups:
                if not isinstance(layer_group, MambaLayerGroup):
                    continue
                for pool in (layer_group.conv_states, layer_group.ssm_states):
                    if pool is None:
                        continue
                    slot = int(mamba_slot)
                    if slot < 0 or slot >= pool.num_slots:
                        raise ValueError(
                            f"destination Mamba slot {slot} is outside [0, {pool.num_slots})"
                        )
                    intervals.add(
                        (int(pool.base_address + slot * pool.slot_bytes), int(pool.slot_bytes))
                    )

        return intervals

    def dispatch_task(self, task: KVRecvTask):
        params = task._params
        logger.debug(
            f"Receiver.dispatch_task: unique_rid={task._unique_rid}, ctx_dp_rank={params.ctx_dp_rank}"
        )
        receiver_req = self._build_recv_req_info(task)
        sender_dp_rank = params.ctx_dp_rank
        peer_infos: RankInfo = self._get_sender_info(params)

        if sender_dp_rank is not None:
            # Normal path: ctx_dp_rank is known, send to overlapping ranks.
            peer_overlap = self._registrar.get_peer_overlap(peer_infos, sender_dp_rank)
            valid_writer_cohorts = (tuple(peer_overlap.ranks),)
        else:
            # Gen-first with ADP: ctx_dp_rank unknown — broadcast REQUEST_DATA
            # to ALL ctx sender ranks so every DP group receives it.
            # get_peer_overlap returns ranks for one DP group (topology is
            # symmetric), so use dp_rank=0 as representative.
            dp_size = peer_infos.dp_size
            dp_overlaps = [
                self._registrar.get_peer_overlap(peer_infos, dp) for dp in range(dp_size)
            ]
            dp0_overlap = dp_overlaps[0]
            # Union of overlapping ranks across all DP groups for broadcast (deduplicated)
            all_ranks_set: set[int] = set()
            for overlap in dp_overlaps:
                all_ranks_set.update(overlap.ranks)
            all_ranks = sorted(all_ranks_set)
            logger.debug(
                f"Receiver.dispatch_task: ADP broadcast path, dp_size={dp_size}, "
                f"all_ranks={all_ranks}"
            )
            peer_overlap = type(dp0_overlap)(ranks=all_ranks)
            valid_writer_cohorts = tuple(tuple(overlap.ranks) for overlap in dp_overlaps)

        # In gen-first ADP broadcast, peer_overlap contains the union of all DP
        # groups, but expected_transfers should reflect per-DP-group count since
        # only one DP group will actually process the context request.
        task.set_valid_writer_cohorts(valid_writer_cohorts)
        # The transport currently accepts only one writer, and only when the
        # receiver can prove that its complete slot layout maps byte-for-byte.
        # Fan-in remains direct until byte-accurate per-writer extents are
        # authorized before any address is published. ADP broadcast also
        # remains direct because it does not select the exact writer cohort.
        allow_bounce = (
            self._bounce.enabled
            and sender_dp_rank is not None
            # The bound-buffer layout currently covers attention pool views;
            # Mamba state bytes remain on the direct descriptor path.
            and task._kv_slice.mamba_state_index is None
            and peer_overlap.overlap_pp_size == 1
            and task.expected_transfers == 1
            and self._single_writer_bounce_exact(peer_infos)
        )
        session = self._get_session(task._unique_rid)
        if session is None:
            raise RuntimeError(
                f"dispatch_task: RxSession {task._unique_rid} not found; "
                "session may have been closed before dispatch"
            )

        writer_ranks = tuple(int(rank) for rank in peer_overlap.ranks)
        sender_endpoints = {peer_infos.sender_endpoints[rank] for rank in peer_overlap.ranks}
        key = (receiver_req.unique_rid, receiver_req.slice_id)
        # Allocator waits are deliberately outside the publication/cancel gate.
        # If cancellation wins while reserve blocks, the reservation is still
        # unexposed and is released after the gate is acquired.
        bounced = allow_bounce and self._bounce.reserve(
            receiver_req,
            writer_ranks,
            destination_intervals_factory=lambda: self._destination_intervals(task),
        )

        # This lock is the publication gate shared with RxSession.cancel().
        with session.lock:
            session._sender_endpoints.update(sender_endpoints)
            if session._terminal_status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                dispatch_cancelled = True
                if bounced:
                    self._bounce.release_idle_reservation(key)
            else:
                dispatch_cancelled = False
                if sender_dp_rank is not None:
                    try:
                        update = self._recv_registry.prepare(
                            key,
                            writer_ranks,
                            has_bounce_slot=bounced,
                        )
                    except Exception:
                        if bounced:
                            self._bounce.release_idle_reservation(key)
                        raise
                    if not update.accepted:
                        if bounced:
                            self._bounce.release_idle_reservation(key)
                        task.fail(
                            RuntimeError(
                                f"receive lifecycle admission failed for {key}: {update.reason}"
                            )
                        )
                        if session._terminal_status is None:
                            session._terminal_status = SessionStatus.ERROR
                        task.close_publication()
                        session._close_aux_publication_locked()
                        return
                    task.lifecycle_managed = True
                task.status = TaskStatus.TRANSFERRING

        if dispatch_cancelled:
            # Cancellation may have happened before sender endpoints were
            # known. No target was reserved or published, so forwarding the
            # cancel now is sufficient and cleanup remains immediately safe.
            if task.status is not TaskStatus.ERROR:
                task.fail(RuntimeError(f"RxSession {receiver_req.unique_rid} cancelled"))
            session.close_task_publication(task)
            self.send_cancel_to_senders(receiver_req.unique_rid, sender_endpoints)
            return

        # Fan-in: each sender gets its own sub-region base (writers must not overwrite); else serialize once.
        fanin_bounce = bounced and task.expected_transfers > 1
        receiver_req_bytes = None if fanin_bounce else receiver_req.to_bytes()

        for rank in writer_ranks:
            if task.lifecycle_managed:
                # Mark POSSIBLY_EXPOSED in both ownership layers while holding
                # the same gate as cancellation. Once this returns, even a ZMQ
                # send exception is ambiguous and the target must be retained.
                with session.lock:
                    aux_publication_blocked = not session._aux_writer_exposure_allowed_locked()
                    if aux_publication_blocked:
                        publication = self._recv_registry.fail_context(
                            key, "session AUX publication gate is already closed"
                        )
                        bounce_allowed = False
                    else:
                        publication = self._recv_registry.begin_publication(key, rank)
                        if publication.publication_allowed and bounced:
                            bounce_allowed = self._bounce.mark_writer_exposed(key, rank)
                        else:
                            bounce_allowed = publication.publication_allowed
                        if publication.publication_allowed and bounce_allowed:
                            task.mark_writer_exposed(rank)
                            if not session._mark_aux_writer_exposed_locked(rank):
                                aux_publication_blocked = True
                                publication = self._recv_registry.fail_context(
                                    key, "session AUX publication gate closed during exposure"
                                )
                if aux_publication_blocked:
                    try:
                        if bounced:
                            self._bounce.mark_logical_failure(
                                key,
                                on_done=lambda succeeded, key=key, rank=rank: self._finish_bounce(
                                    key, succeeded, rank
                                ),
                            )
                    finally:
                        try:
                            self._handle_lifecycle_update(publication, peer_rank=rank)
                        finally:
                            session.close_task_publication(task)
                            self.send_cancel_to_senders(receiver_req.unique_rid, sender_endpoints)
                    return
                if not publication.publication_allowed:
                    continue
                if not bounce_allowed:
                    # No address was sent.  Settle both ownership layers with
                    # the same explicit no-access proof and let the bounce
                    # callback acknowledge physical slot release.
                    update = self._recv_registry.mark_never_published(key, rank)
                    self._bounce.record_no_access(
                        key,
                        rank,
                        succeeded=False,
                        on_done=lambda succeeded, key=key, rank=rank: self._finish_bounce(
                            key, succeeded, rank
                        ),
                    )
                    self._handle_lifecycle_update(update, peer_rank=rank)
                    session.close_task_publication(task)
                    return
            else:
                # ADP broadcast has no exact selected-writer context. The
                # session lock is still the publication/cancellation gate:
                # once this flag is set, cleanup must await the complete
                # legacy terminal-result cohort.
                with session.lock:
                    if (
                        session._terminal_status
                        in (
                            SessionStatus.ERROR,
                            SessionStatus.CANCELLED,
                        )
                        or not session._aux_writer_exposure_allowed_locked()
                    ):
                        publication_allowed = False
                    else:
                        task.mark_writer_exposed(rank)
                        publication_allowed = session._mark_aux_writer_exposed_locked(rank)
                if not publication_allowed:
                    if task.status is not TaskStatus.ERROR:
                        task.fail(
                            RuntimeError(
                                f"AUX publication gate closed for request "
                                f"{receiver_req.unique_rid} slice={receiver_req.slice_id}"
                            )
                        )
                    session.close_task_publication(task)
                    self.send_cancel_to_senders(receiver_req.unique_rid, sender_endpoints)
                    return

            if task._perf_timer is not None:
                task._perf_timer.record_task_start(rank)
            if fanin_bounce:
                receiver_req.bounce_dst_base = self._bounce.writer_base(key, rank)
                receiver_req_bytes = receiver_req.to_bytes()
            try:
                self._request_sender_data(peer_infos.sender_endpoints[rank], receiver_req_bytes)
            except Exception as e:
                if not task.lifecycle_managed:
                    session.close_task_publication(task)
                    raise
                self._recv_registry.mark_publication_ambiguous(key, rank)
                update = self._recv_registry.fail_context(
                    key, f"target publication to writer {rank} failed: {e}"
                )
                if bounced:
                    self._bounce.mark_logical_failure(
                        key,
                        on_done=lambda succeeded, key=key, rank=rank: self._finish_bounce(
                            key, succeeded, rank
                        ),
                    )
                self._handle_lifecycle_update(update, peer_rank=rank)
                logger.error(
                    f"Receiver failed to publish target for request {receiver_req.unique_rid} "
                    f"slice={receiver_req.slice_id} writer={rank}: {e}"
                )
                # Keep the request in the transceiver maps so its destination
                # blocks cannot be reused while this send remains ambiguous.
                session.close_task_publication(task)
                return
            if task.lifecycle_managed:
                publication = self._recv_registry.mark_published(key, rank)
                if publication.conflict:
                    if bounced:
                        self._bounce.mark_protocol_conflict(
                            key,
                            on_done=lambda succeeded, key=key, rank=rank: self._finish_bounce(
                                key, succeeded, rank
                            ),
                        )
                    self._handle_lifecycle_update(publication, peer_rank=rank)
                    session.close_task_publication(task)
                    return
        session.close_task_publication(task)
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

    def _get_or_connect_dealer(self, endpoint: Optional[str]):
        if endpoint is None:
            raise ValueError("Receiver: peer endpoint is None; peer may not have registered yet")
        with self._dealers_lock:
            if not getattr(self, "_dealer_admission_open", True):
                raise RuntimeError("Receiver is shutting down; dealer admission is closed")
            if endpoint not in self._dealers:
                self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
            return self._dealers[endpoint]

    def _get_sender_info(self, params: DisaggregatedParams) -> RankInfo:
        info_endpoint = self._extract_info_endpoint(params)
        with self._dealers_lock:
            if not getattr(self, "_dealer_admission_open", True):
                raise RuntimeError("Receiver is shutting down; sender discovery is closed")
        if self._should_register_peer(params):
            logger.info(f"Registering peer in first request to endpoint '{info_endpoint}'")
            messenger = ZMQMessenger(mode="DEALER", endpoint=info_endpoint)
            try:
                messenger.send([MessageType.REQUEST_INSTANCE_INFO])
                message = messenger.receive()
                sender_info = RankInfo.from_bytes(message[0])
            finally:
                messenger.stop()

            for endpoint in sender_info.sender_endpoints:
                dealer = self._get_or_connect_dealer(endpoint)
                rank_info = self._registrar.self_rank_info
                dealer.send([MessageType.REGISTER_RANK_INFO, rank_info.to_bytes()])

            self._sender_ep_instance_map[info_endpoint] = sender_info
            return sender_info

        else:
            return self._sender_ep_instance_map[info_endpoint]

    def send_cancel_to_senders(self, unique_rid: int, sender_endpoints: set[str]) -> None:
        """Notify all senders involved in this session to cancel."""
        for endpoint in sender_endpoints:
            try:
                self._get_or_connect_dealer(endpoint).send(
                    [MessageType.CANCEL_SESSION, str(unique_rid).encode("ascii")]
                )
            except Exception as e:
                logger.warning(f"send_cancel_to_senders: failed for rid={unique_rid}: {e}")

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
                case _:
                    logger.error(f"Receiver received unknown message type: {msg[0]}")
            return True

        self._messenger.start_listener(handle_message)

    def _handle_cancel_session(self, message: list[bytes]):
        unique_rid = int(message[1])
        session = None
        with self._sessions_lock:
            # Retain the tombstone even when a live session consumes this
            # cancellation. Request IDs are not generation-safe, so deleting it
            # would let a replacement session escape a delayed replay.
            self._pre_cancelled_rids.add(unique_rid)
            entry = self._sessions.get(unique_rid)
            if entry is not None:
                session = entry() if isinstance(entry, weakref.ReferenceType) else entry
        if session is not None:
            session.cancel()

    def _process_kv_agent_result(self, _send_id: bytes, message: list[bytes]):
        if message[0] != MessageType.KV_AGENT_RESULT:
            logger.error(
                f"_process_kv_agent_result: unexpected msg_type={message[0]!r}, expected KV_AGENT_RESULT"
            )
            return
        peer_rank, unique_rid, sender_slice_id, is_last_slice, status_code = (
            _KV_RESULT_PREFIX.unpack(message[1])
        )
        if sender_slice_id < 0:
            logger.error(
                f"KV result references invalid negative slice {sender_slice_id} "
                f"for request {unique_rid}; dropping status"
            )
            return
        from .bounce import decode_result_tail

        key = (unique_rid, sender_slice_id)
        target_mode = self._recv_registry.target_mode(key)
        if target_mode is not None:
            try:
                agent_result = _AGENT_RESULT_BY_CODE[status_code]
            except KeyError:
                # A newer sender may define a non-terminal state.  Without
                # negotiated enum semantics this frame is not quiescence
                # evidence, so fail logically but retain the writer target.
                update = self._recv_registry.record_protocol_conflict(
                    key,
                    peer_rank,
                    f"unknown writer result code {status_code} from rank {peer_rank}",
                )
                if target_mode is WriterMode.BOUNCE:
                    self._bounce.mark_protocol_conflict(
                        key,
                        on_done=lambda succeeded, key=key, peer_rank=peer_rank: self._finish_bounce(
                            key, succeeded, peer_rank
                        ),
                    )
                logger.error(
                    f"writer {peer_rank} returned unknown result code {status_code} "
                    f"for request {unique_rid} slice={sender_slice_id}"
                )
                self._handle_lifecycle_update(update, peer_rank=peer_rank)
                return

            try:
                dst_ptrs, sizes, src_base = decode_result_tail(message)
            except Exception as e:
                logger.error(
                    f"malformed KV result tail for request {unique_rid} "
                    f"slice={sender_slice_id} writer={peer_rank}: {e}"
                )
                dst_ptrs, sizes, src_base = None, None, None
                agent_result = AgentResult.FAILED

            expected_bounce = target_mode is WriterMode.BOUNCE
            tail_present = len(message) > 2
            complete_tail = (
                len(message) == 5
                and dst_ptrs is not None
                and sizes is not None
                and src_base is not None
            )
            malformed_bounce_success = (
                expected_bounce
                and agent_result is AgentResult.SUCCESS
                and tail_present
                and not complete_tail
            )
            if malformed_bounce_success:
                logger.error(
                    f"bounce writer {peer_rank} returned a malformed result tail "
                    f"for request {unique_rid} slice={sender_slice_id}"
                )
                agent_result = AgentResult.FAILED

            if expected_bounce:
                # Sender bounce allocation is per writer and may fall back to
                # the direct descriptor path. A complete success tail proves
                # BOUNCE; no-tail success is the backward-compatible DIRECT
                # fallback. Failure is conservatively BOUNCE because it may
                # have accessed the advertised slot.
                mode = (
                    WriterMode.BOUNCE
                    if complete_tail or agent_result is AgentResult.FAILED
                    else WriterMode.DIRECT
                )
            else:
                # Any tail on a direct-only target is a mode conflict.
                mode = WriterMode.BOUNCE if tail_present else WriterMode.DIRECT
            writer_result = (
                WriterResult.SUCCESS if agent_result is AgentResult.SUCCESS else WriterResult.FAILED
            )
            update = self._recv_registry.record_result(key, peer_rank, writer_result, mode)
            if not update.accepted:
                if update.conflict:
                    if expected_bounce:
                        self._bounce.mark_protocol_conflict(
                            key,
                            on_done=lambda succeeded, key=key, peer_rank=peer_rank: (
                                self._finish_bounce(key, succeeded, peer_rank)
                            ),
                        )
                    logger.error(
                        f"receive lifecycle conflict for request {unique_rid} "
                        f"slice={sender_slice_id} writer={peer_rank}: {update.reason}"
                    )
                    self._handle_lifecycle_update(update, peer_rank=peer_rank)
                return

            if expected_bounce:

                def on_done(
                    succeeded: bool,
                    key: tuple[int, int] = key,
                    peer_rank: int = peer_rank,
                ) -> None:
                    self._finish_bounce(key, succeeded, peer_rank)

                if mode is WriterMode.DIRECT:
                    self._bounce.record_no_access(
                        key,
                        peer_rank,
                        succeeded=agent_result is AgentResult.SUCCESS,
                        on_done=on_done,
                    )
                elif agent_result is AgentResult.SUCCESS:
                    self._bounce.record_result(
                        key,
                        peer_rank,
                        dst_ptrs,
                        sizes,
                        src_base,
                        on_done,
                        on_logical_failure=lambda key=key: self._fail_bounce_logically(key),
                    )
                else:
                    self._bounce.record_failure(key, peer_rank, on_done=on_done)

            # This optional notification intentionally happens after the
            # registry and bounce transport have consumed the terminal result.
            # A closed/GC'd RxSession can no longer strand physical ownership.
            self._handle_lifecycle_update(update, peer_rank=peer_rank)
            return

        session = self._get_session(unique_rid)
        if session is None:
            logger.warning(
                f"_process_kv_agent_result: session {unique_rid} not found (already closed?), dropping status"
            )
            return
        try:
            agent_result = _AGENT_RESULT_BY_CODE[status_code]
        except KeyError:
            session.process_kv_protocol_conflict(
                peer_rank,
                sender_slice_id,
                f"unknown writer result code {status_code}",
            )
            return
        try:
            dst_ptrs, sizes, src_base = decode_result_tail(message)
        except Exception as e:
            session.process_kv_protocol_conflict(
                peer_rank,
                sender_slice_id,
                f"malformed writer result tail: {e}",
            )
            return
        session.process_kv_agent_result(
            peer_rank,
            sender_slice_id,
            is_last_slice,
            agent_result,
            dst_ptrs=dst_ptrs,
            sizes=sizes,
            src_base=src_base,
        )

    def _process_aux_agent_result(self, _send_id: bytes, message: list[bytes]):
        _msg_type, peer_rank, unique_rid, status = decode_message(message)
        peer_rank = int(peer_rank)
        unique_rid = int(unique_rid)
        session = self._get_session(unique_rid)
        if session is None:
            logger.warning(
                f"_process_aux_agent_result: session {unique_rid} not found (already closed?), dropping status"
            )
            return
        try:
            agent_result = AgentResult(status)
        except ValueError:
            # An unknown value is not terminal evidence.  Fail the request
            # logically but keep the AUX lease until this exact writer later
            # reports a known terminal result or a global fence retires it.
            session.process_aux_protocol_conflict(
                peer_rank, f"unknown AUX writer result {status!r}"
            )
            return
        session.process_aux_agent_result(peer_rank, agent_result)

    def _request_sender_data(self, endpoint: str, receiver_info_bytes: bytes):
        # receiver_info serialized once and reused for every peer rank (block-table msgpack isn't free at fan-out).
        logger.debug("Sending data request to endpoint '%s'", endpoint)
        messenger = self._get_or_connect_dealer(endpoint)
        messenger.send([MessageType.REQUEST_DATA, receiver_info_bytes])

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Receiver.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        try:
            drained = self.shutdown()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error(f"Receiver context cleanup failed while propagating an exception: {e}")
            return False
        if not drained:
            if exc_type is None:
                raise RuntimeError("Receiver context exited before transfer resources drained")
            logger.error("Receiver context retained resources while propagating an exception")
        return False


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
        destination_owner: object | None = None,
    ):
        # Retain the request that owns its KV allocation until every accessor
        # and scatter is drained. Enroll it before fallible initialization,
        # then clear the parameter so propagated tracebacks are ownership-
        # neutral.
        self._destination_owner = destination_owner
        destination_owner = None
        self._aux_buffer = aux_buffer
        self.aux_slot = None
        try:
            super().__init__(
                receiver,
                SessionArgsBase(params, prompt_len=prompt_len, beam_width=beam_width),
            )
            self._timeout_s = timeout_s
            self._need_aux = params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
            self._receiver: Receiver  # narrow base class type for Pylance
            self.request_id = request_id
            # Receive dispatch owns the address-publication order for every
            # slice in this session. Keep it separate from ``lock`` so
            # cancellation can still close the publication gate while a
            # bounce reservation blocks.
            self._receive_lock = threading.Lock()
            self._close_lock = threading.Lock()
            self.lock = threading.Lock()
            self.aux_slot = aux_buffer.alloc_slot().id if aux_buffer is not None else None
            self._exception: Optional[Exception] = None
            self._closed = False
            self._accepting_receives = True
            self._active_receive_dispatches = 0
            self._last_slice_admitted = False
            self._receiver_retired = False
            self._terminal_status: Optional[SessionStatus] = None
            self._kv_tasks: list[KVRecvTask] = []
            self._aux_results: dict[int, AgentResult] = {}
            # AUX storage is session-scoped and its address can be repeated in
            # every KV-slice publication. Keep one exposure ledger across
            # slices; a later suppressed publication cannot revoke an earlier
            # exposure.
            self._aux_exposed_writer_ranks: set[int] = set()
            self._aux_publication_closed = not self._need_aux
            self._aux_result_conflict = False
            self._aux_drained = not self._need_aux
            self._aux_status: TaskStatus = TaskStatus.INIT
            self._sender_endpoints: set[str] = set()
            self._selected_writer_cohort: Optional[frozenset[int]] = None
            self._receiver.setup_session(self)
        except Exception:
            self._destination_owner = None
            if self._aux_buffer is not None and self.aux_slot is not None:
                self._aux_buffer.free_slot(self.aux_slot)
                self.aux_slot = None
            self._accepting_receives = False
            self._closed = True
            raise

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

    def mark_transferring(self, slice_id: int):
        with self.lock:
            self._kv_tasks[slice_id].status = TaskStatus.TRANSFERRING

    def _mark_adp_cohort_conflict_locked(
        self, task: KVRecvTask, *, channel: str, identities: frozenset[int]
    ) -> None:
        """Fail one compatibility channel without using ambiguity as drain proof."""
        detail = (
            f"ADP {channel.upper()} writer identities {sorted(identities)} conflict with "
            f"the selected/candidate cohorts for request {self.disagg_request_id} "
            f"slice={task.slice_id}"
        )
        exc = RuntimeError(detail)
        if channel == "kv":
            task._legacy_result_conflict = True
            task.fail(exc)
            if self._need_aux:
                # Cohort selection is session-wide. Evidence that a different
                # cohort handled KV also invalidates an earlier AUX drain based
                # on that selection, because every publication carried the AUX
                # address for the same candidate writers.
                self._aux_result_conflict = True
                self._aux_drained = False
                self._aux_status = TaskStatus.ERROR
                self._exception = exc
        elif channel == "aux":
            self._aux_result_conflict = True
            self._aux_drained = False
            self._aux_status = TaskStatus.ERROR
            self._exception = exc
            # Conversely, an AUX result outside the selected cohort disproves
            # session-wide cohort selection. Keep every legacy KV target from
            # that compatibility session fail-closed.
            for kv_task in self._kv_tasks:
                if kv_task.lifecycle_managed or not kv_task.publication_started:
                    continue
                kv_task._legacy_result_conflict = True
                kv_task.fail(exc)
        else:
            raise ValueError(f"unknown ADP result channel {channel!r}")
        if self._terminal_status is None:
            self._terminal_status = SessionStatus.ERROR
        logger.error(detail)

    def _select_adp_cohort_locked(
        self,
        task: KVRecvTask,
        results: dict[int, AgentResult],
        *,
        channel: str,
    ) -> None:
        """Select one cohort only while all observations remain compatible."""
        identities = frozenset(results)
        if not identities:
            return
        selected = self._selected_writer_cohort
        if selected is not None:
            if not identities <= selected:
                self._mark_adp_cohort_conflict_locked(task, channel=channel, identities=identities)
            return

        compatible = tuple(cohort for cohort in task._valid_writer_cohorts if identities <= cohort)
        complete = tuple(cohort for cohort in compatible if cohort == identities)
        if len(complete) == 1 and len(compatible) == 1:
            self._selected_writer_cohort = complete[0]
            return
        if not compatible:
            self._mark_adp_cohort_conflict_locked(task, channel=channel, identities=identities)

    def _legacy_channel_succeeded_locked(
        self, task: KVRecvTask, results: dict[int, AgentResult]
    ) -> bool:
        """Validate success from every member of the selected legacy cohort."""
        selected = self._selected_writer_cohort
        if selected is None:
            return False
        return all(results.get(rank) is AgentResult.SUCCESS for rank in selected)

    def _advance_legacy_kv_locked(self, task: KVRecvTask) -> None:
        if task.lifecycle_managed or not task.legacy_resources_drained or task.is_done:
            return
        if not task._legacy_result_conflict and self._legacy_channel_succeeded_locked(
            task, task._legacy_results
        ):
            task.complete()
            return
        detail = (
            f"ADP KV writer outcome did not match the selected cohort for "
            f"request {self.disagg_request_id} slice={task.slice_id}"
        )
        task.fail(RuntimeError(detail))
        if self._terminal_status is None:
            self._terminal_status = SessionStatus.ERROR

    def _advance_aux_locked(self, task: KVRecvTask) -> None:
        if not self._need_aux or self._aux_drained:
            return
        selected = self._selected_writer_cohort
        # ADP publishes to candidate cohorts but, by the compatibility
        # contract, exactly one cohort performs the transfer. Once selected,
        # only that cohort can be a physical accessor. Before selection, every
        # actually published rank remains a candidate accessor.
        # A protocol conflict invalidates cohort selection as physical
        # quiescence evidence.  Keep every identity that may have observed the
        # slot in the drain ledger until it reports terminal or a backend-wide
        # fence retires it.
        if self._aux_result_conflict or selected is None:
            required = self._aux_exposed_writer_ranks
        else:
            required = selected
        all_terminal = self._aux_publication_closed and required <= self._aux_results.keys()
        if not all_terminal:
            return
        self._aux_drained = True
        if (
            selected is not None
            and not self._aux_result_conflict
            and self._legacy_channel_succeeded_locked(task, self._aux_results)
        ):
            self._aux_status = TaskStatus.TRANSFERRED
            return
        self._aux_status = TaskStatus.ERROR
        self._exception = RuntimeError(
            f"ADP AUX writer outcome did not match the selected cohort for "
            f"request {self.disagg_request_id}"
        )
        if self._terminal_status is None:
            self._terminal_status = SessionStatus.ERROR

    def _aux_writer_exposure_allowed_locked(self) -> bool:
        """Whether another publication may carry the session AUX address."""
        return not (self._need_aux and self.aux_slot is not None and self._aux_publication_closed)

    def _mark_aux_writer_exposed_locked(self, peer_rank: int) -> bool:
        """Record one possible AUX accessor, rejecting exposure after closure."""
        if not self._aux_writer_exposure_allowed_locked():
            return False
        if self._need_aux and self.aux_slot is not None:
            self._aux_exposed_writer_ranks.add(int(peer_rank))
        return True

    def _close_aux_publication_locked(self) -> None:
        """Prove no later KV slice can expose the session-level AUX slot."""
        if not self._need_aux or self._aux_publication_closed:
            return
        self._aux_publication_closed = True
        if not self._kv_tasks:
            self._aux_drained = True
            self._aux_status = TaskStatus.ERROR
            return
        cohort_task = self._kv_tasks[0]
        self._select_adp_cohort_locked(cohort_task, self._aux_results, channel="aux")
        self._advance_aux_locked(cohort_task)

    def close_task_publication(self, task: KVRecvTask) -> None:
        """Close a dispatch plan and consume results that raced its final send."""
        with self.lock:
            task.close_publication()
            self._select_adp_cohort_locked(task, task._legacy_results, channel="kv")
            self._advance_legacy_kv_locked(task)

            if self._need_aux and self._kv_tasks:
                if task._kv_slice.is_last_slice or self._terminal_status in (
                    SessionStatus.ERROR,
                    SessionStatus.CANCELLED,
                ):
                    self._close_aux_publication_locked()
                cohort_task = self._kv_tasks[0]
                self._select_adp_cohort_locked(cohort_task, self._aux_results, channel="aux")
                self._advance_aux_locked(cohort_task)

    def receive(self, slice: KVSlice) -> None:
        # Serializing through dispatch is required because AUX storage and its
        # publication ledger are session-wide.  A later last slice must not
        # close/free the slot while an earlier slice can still expose it.
        with self._receive_lock:
            with self.lock:
                if not self._accepting_receives:
                    raise RuntimeError(
                        f"RxSession {self.disagg_request_id} is closing; "
                        "new receive slices are not accepted"
                    )
                if self._last_slice_admitted:
                    raise RuntimeError(
                        f"RxSession {self.disagg_request_id} already admitted its last slice"
                    )
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
                self._active_receive_dispatches += 1
                if slice.is_last_slice:
                    self._last_slice_admitted = True
            try:
                self._receiver.dispatch_task(task)
            except Exception as dispatch_error:
                # Admission and dispatch form one transaction for standalone
                # callers too.  Cancellation settles any context prepared
                # before the exception, while preserving every possibly
                # exposed target until terminal evidence arrives.
                cleanup_errors: list[Exception] = []
                try:
                    self.cancel()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                with self.lock:
                    task.close_publication()
                    task.fail(dispatch_error)
                    self._exception = dispatch_error
                    self._terminal_status = SessionStatus.ERROR
                    self._close_aux_publication_locked()
                if cleanup_errors:
                    logger.error(
                        f"RxSession {self.disagg_request_id} retained ownership after dispatch "
                        f"cleanup failed: {cleanup_errors[0]}"
                    )
                raise
            finally:
                with self.lock:
                    self._active_receive_dispatches -= 1

    def process_lifecycle_update(
        self, update: LifecycleUpdate, *, peer_rank: Optional[int] = None
    ) -> None:
        """Apply only the consumer-visible part of a registry transition.

        The registry and bounce transport have already consumed the physical
        transition before this method runs. Consequently losing this session
        cannot lose writer accounting, scatter completion, or slot ownership.
        """
        slice_id = update.key[1]
        with self.lock:
            if not 0 <= slice_id < len(self._kv_tasks):
                logger.error(
                    f"receive lifecycle update references missing slice {slice_id} "
                    f"for request {self.disagg_request_id}"
                )
                return
            task = self._kv_tasks[slice_id]
            actions = set(update.actions)
            if LifecycleAction.NOTIFY_CANCELLED in actions:
                if task.status is not TaskStatus.ERROR:
                    task.fail(RuntimeError(f"RxSession {self.disagg_request_id} cancelled"))
                self._terminal_status = SessionStatus.CANCELLED
                self._close_aux_publication_locked()
                return
            if LifecycleAction.NOTIFY_FAILURE in actions:
                detail = (
                    f"KV receive lifecycle failed for request {self.disagg_request_id} "
                    f"slice={slice_id}"
                )
                if update.reason:
                    detail += f": {update.reason}"
                task.fail(RuntimeError(detail))
                if self._terminal_status is None:
                    self._terminal_status = SessionStatus.ERROR
                self._close_aux_publication_locked()
                logger.error(detail)
                return
            if LifecycleAction.NOTIFY_SUCCESS not in actions:
                return
            if task.status is TaskStatus.ERROR:
                return

            if peer_rank is not None:
                try:
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_end(peer_rank)
                    ri = self._receiver._registrar.self_rank_info
                    task.print_perf_info(peer_rank, ri.instance_name, ri.instance_rank)
                except Exception as e:
                    logger.warning(
                        f"KV transfer perf logging failed for request {self.disagg_request_id} "
                        f"slice={slice_id}: {e}"
                    )
            task.complete()
            logger.debug(
                f"KV transfer complete for request {self.disagg_request_id} slice={slice_id}"
            )

    def process_kv_protocol_conflict(
        self, peer_rank: int, sender_slice_id: int, reason: str
    ) -> None:
        """Fail logically without treating an unrecognized frame as terminal."""
        with self.lock:
            if not 0 <= sender_slice_id < len(self._kv_tasks):
                logger.error(
                    f"KV protocol conflict references missing slice {sender_slice_id} "
                    f"for request {self.disagg_request_id}: {reason}"
                )
                return
            task = self._kv_tasks[sender_slice_id]
            # A frame for this identity is itself evidence that the peer may
            # have observed a target; never use the no-publication fast path.
            task.mark_writer_exposed(peer_rank)
            task._legacy_result_conflict = True
            detail = (
                f"KV protocol conflict for request {self.disagg_request_id} "
                f"slice={sender_slice_id} writer={peer_rank}: {reason}"
            )
            exc = RuntimeError(detail)
            task.fail(exc)
            self._exception = exc
            if self._terminal_status is None:
                self._terminal_status = SessionStatus.ERROR
            # This frame is remote evidence, not a new local publication.
            # Preserve the possible accessor even if the publication gate has
            # already closed so AUX drain remains fail-closed.
            if self._need_aux and self.aux_slot is not None:
                self._aux_exposed_writer_ranks.add(int(peer_rank))
            self._close_aux_publication_locked()
            logger.error(detail)

    def process_kv_agent_result(
        self,
        peer_rank: int,
        sender_slice_id: int,
        is_last_slice: bool,
        status: AgentResult,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
    ):
        del is_last_slice, dst_ptrs, sizes, src_base
        with self.lock:
            if not 0 <= sender_slice_id < len(self._kv_tasks):
                logger.error(
                    f"Receiver got invalid slice_id={sender_slice_id} from sender but only has "
                    f"{len(self._kv_tasks)} receive task(s) for request {self.request_id}; "
                    "dropping status"
                )
                return
            task = self._kv_tasks[sender_slice_id]
            if task.lifecycle_managed:
                # Managed results are consumed by Receiver's registry before
                # optional session notification. Reaching the compatibility
                # handler means the context was already retired; a late
                # duplicate must not be counted as a legacy writer.
                logger.debug(
                    f"Ignoring late managed KV result for request {self.disagg_request_id} "
                    f"slice={sender_slice_id} writer={peer_rank}"
                )
                return
            is_new = task.record_legacy_result(peer_rank, status)
            if task._legacy_result_conflict:
                detail = (
                    f"conflicting legacy KV result identities for request {self.request_id} "
                    f"slice={sender_slice_id} writer={peer_rank}"
                )
                task.fail(RuntimeError(detail))
                if self._terminal_status is None:
                    self._terminal_status = SessionStatus.ERROR
                self._close_aux_publication_locked()
                logger.error(detail)
                return
            if not is_new:
                return
            if status not in (AgentResult.SUCCESS, AgentResult.FAILED):
                raise ValueError(
                    f"Session {self.request_id} received unknown task status: {status.value}"
                )
            self._select_adp_cohort_locked(task, task._legacy_results, channel="kv")
            if task._legacy_result_conflict:
                self._close_aux_publication_locked()
                return
            self._advance_legacy_kv_locked(task)

    def process_aux_agent_result(self, peer_rank: int, status: AgentResult):
        # Aux is session-level (not per-slice); expected_transfers is identical
        # across all kv_tasks, so any task provides the right count.
        with self.lock:
            if not self._kv_tasks:
                logger.warning(
                    f"Aux result received before any KV tasks for request {self.request_id}"
                )
                return
            task = self._kv_tasks[0]
            # A terminal response proves this writer can no longer access the
            # aux slot. Count identities, not success-only responses, so one
            # failure cannot release the slot while siblings remain active.
            previous = self._aux_results.get(peer_rank)
            if previous is not None:
                if previous is not status:
                    self._aux_result_conflict = True
                    self._exception = RuntimeError(
                        f"Session {self.request_id} received conflicting aux results "
                        f"from writer {peer_rank}"
                    )
                    self._aux_status = TaskStatus.ERROR
                    if self._terminal_status is None:
                        self._terminal_status = SessionStatus.ERROR
                    self._close_aux_publication_locked()
                    logger.error(str(self._exception))
                return
            identity_valid = peer_rank in self._aux_exposed_writer_ranks
            if not identity_valid:
                # The frame itself is evidence that this identity participated
                # in some publication generation. Retain it in the physical
                # ledger even while latching the protocol conflict.
                self._aux_exposed_writer_ranks.add(peer_rank)
            self._aux_results[peer_rank] = status
            if not identity_valid:
                self._aux_result_conflict = True
                self._exception = RuntimeError(
                    f"Session {self.request_id} received an unexpected "
                    f"aux writer identity {peer_rank}"
                )
                if self._terminal_status is None:
                    self._terminal_status = SessionStatus.ERROR
                self._close_aux_publication_locked()

            if status not in (AgentResult.SUCCESS, AgentResult.FAILED):
                raise ValueError(
                    f"Session {self.request_id} received unknown aux send status: {status}"
                )
            if self._aux_result_conflict:
                self._aux_status = TaskStatus.ERROR
                if self._terminal_status is None:
                    self._terminal_status = SessionStatus.ERROR
                logger.error(str(self._exception))
            self._select_adp_cohort_locked(task, self._aux_results, channel="aux")
            self._advance_aux_locked(task)

    def process_aux_protocol_conflict(self, peer_rank: int, reason: str) -> None:
        """Latch an AUX protocol failure without treating it as quiescence."""
        with self.lock:
            self._aux_exposed_writer_ranks.add(peer_rank)
            self._aux_result_conflict = True
            self._aux_status = TaskStatus.ERROR
            self._exception = RuntimeError(
                f"AUX protocol conflict for request {self.disagg_request_id} "
                f"writer={peer_rank}: {reason}"
            )
            if self._terminal_status is None:
                self._terminal_status = SessionStatus.ERROR
            self._close_aux_publication_locked()
            logger.error(str(self._exception))

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

    def mark_backend_quiesced(self) -> None:
        """Retire compatibility targets after an external remote/global fence."""
        with self.lock:
            exc = RuntimeError(
                f"RxSession {self.disagg_request_id} ended without complete terminal results "
                "before remote/global quiescence"
            )
            for task in self._kv_tasks:
                if task.lifecycle_managed or task.legacy_resources_drained:
                    continue
                task._legacy_backend_quiesced = True
                task.fail(exc)
            if self._need_aux and not self._aux_drained:
                self._aux_drained = True
                self._aux_status = TaskStatus.ERROR
                self._exception = exc
            if any(task.status is TaskStatus.ERROR for task in self._kv_tasks) or (
                self._need_aux and self._aux_status is TaskStatus.ERROR
            ):
                if self._terminal_status is None:
                    self._terminal_status = SessionStatus.ERROR

    def has_untracked_receive_activity(self) -> bool:
        """Whether legacy KV or AUX memory can still have remote accessors."""
        with self.lock:
            if getattr(self, "_active_receive_dispatches", 0):
                return True
            if any(
                not task.lifecycle_managed and not task.legacy_resources_drained
                for task in self._kv_tasks
            ):
                return True
            return self._need_aux and not self._aux_drained

    def seal_receive_admission(self) -> None:
        """Prevent queued or future receive calls from entering dispatch."""
        with self.lock:
            self._accepting_receives = False

    def resources_drained(self) -> bool:
        """Whether request cleanup can release every receive-side target."""
        try:
            if not self._receiver._bounce.retry_settlements(self.disagg_request_id):
                return False
        except Exception as e:
            logger.error(
                f"failed to retry bounce settlement for request {self.disagg_request_id}: {e}"
            )
            return False
        if not self._receiver._recv_registry.is_request_drained(self.disagg_request_id):
            return False
        return not self.has_untracked_receive_activity()

    def is_completed(self) -> bool:
        """Non-blocking check: has the transfer completed successfully?"""
        if not self.resources_drained():
            return False
        status = self.status
        if self._need_aux:
            return status == SessionStatus.FULLY_TRANSFERRED
        return status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)

    def has_failed(self) -> bool:
        """Whether failure is safe to expose to request cleanup.

        Logical failure is latched immediately, but without an allocator KV
        lease this method must remain false until every registry-owned writer
        and bounce scatter is physically drained.
        """
        if not self.resources_drained():
            return False
        return self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)

    def cancel(self) -> None:
        """Cancel the session and notify the remote sender.

        Safe to call multiple times. The lock is also the address-publication
        gate, so cancellation either closes an unexposed writer or observes it
        as possibly exposed and retains its resources until terminal evidence.
        """
        errors: list[tuple[str, Exception]] = []
        with self.lock:
            self._accepting_receives = False
            try:
                self._receiver.record_cancel_tombstone(self.disagg_request_id)
            except Exception as e:
                errors.append(("receiver cancellation tombstone", e))
            self._terminal_status = SessionStatus.CANCELLED
            exc = RuntimeError(f"RxSession {self.disagg_request_id} cancelled")
            try:
                updates = self._receiver._recv_registry.cancel_request(self.disagg_request_id)
            except Exception as e:
                updates = ()
                errors.append(("receive registry", e))
            for task in self._kv_tasks:
                if task.lifecycle_managed:
                    if task.status is not TaskStatus.TRANSFERRED:
                        task.fail(exc)
                else:
                    # Legacy ADP broadcast has no exact writer context. Bounce
                    # is disabled for that path, but keep the compatibility
                    # cleanup for standalone legacy users. Retry release even
                    # after logical failure because the first attempt may have
                    # raised before retiring an idle reservation.
                    try:
                        self._receiver._bounce.release_idle_reservation(
                            (self.disagg_request_id, task.slice_id)
                        )
                    except Exception as e:
                        errors.append((f"legacy slice {task.slice_id}", e))
                    if task.status is not TaskStatus.TRANSFERRED:
                        task.fail(exc)
            self._close_aux_publication_locked()
            sender_endpoints = set(self._sender_endpoints)
        for update in updates:
            if update.physical_state is PhysicalState.DRAINED:
                continue
            try:
                self._receiver._bounce.mark_logical_failure(
                    update.key,
                    on_done=lambda succeeded, key=update.key: self._receiver._finish_bounce(
                        key, succeeded
                    ),
                )
            except Exception as e:
                errors.append((f"receive context {update.key}", e))
        # Send outside the lock to avoid holding it during I/O.
        try:
            self._receiver.send_cancel_to_senders(self.disagg_request_id, sender_endpoints)
        except Exception as e:
            errors.append(("sender cancellation", e))
        if errors:
            detail = "; ".join(f"{owner}: {error}" for owner, error in errors)
            raise RuntimeError(
                f"RxSession {self.disagg_request_id} cancellation encountered "
                f"{len(errors)} error(s): {detail}"
            ) from errors[0][1]

    def has_transferring_tasks(self) -> bool:
        """True while request cleanup could race a receive-side accessor.

        cancel_request() must return False while this is True.
        """
        return not self.resources_drained()

    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]:
        """Poll or block until transfer completes.

        With blocking=False (default): polls non-blockingly; returns None if
        any KV task or aux is not yet done — caller should re-poll next cycle.
        With blocking=True: the configured timeout latches failure and asks the
        sender to cancel, but this call remains fail-closed until published
        targets drain. A future allocator-backed KV lease will allow logical
        timeout to return without retaining the whole request.
        """
        if not self.resources_drained():
            if not blocking:
                return None
            deadline = None if self._timeout_s is None else time.monotonic() + self._timeout_s
            timeout_latched = False
            while not self.resources_drained():
                if deadline is not None and time.monotonic() >= deadline and not timeout_latched:
                    timeout_latched = True
                    with self.lock:
                        # Timeout is a local cancellation decision.  Seal
                        # admission and retain the request-id tombstone before
                        # sending cancellation so a delayed result or replay
                        # cannot attach to a replacement session.
                        self._accepting_receives = False
                        self._receiver.record_cancel_tombstone(self.disagg_request_id)
                        timeout_updates = self._receiver._recv_registry.timeout_request(
                            self.disagg_request_id
                        )
                        exc = TimeoutError(
                            f"RxSession {self.disagg_request_id} timed out while receive "
                            "resources were still active"
                        )
                        for task in self._kv_tasks:
                            if task.status is not TaskStatus.TRANSFERRED:
                                task.fail(exc)
                        if self._need_aux and not self._aux_drained:
                            self._aux_status = TaskStatus.ERROR
                            self._exception = exc
                            self._close_aux_publication_locked()
                        if self._terminal_status is None:
                            self._terminal_status = SessionStatus.ERROR
                        sender_endpoints = set(self._sender_endpoints)
                    for update in timeout_updates:
                        self._receiver._bounce.mark_logical_failure(
                            update.key,
                            on_done=lambda succeeded, key=update.key: self._receiver._finish_bounce(
                                key, succeeded
                            ),
                        )
                    self._receiver.send_cancel_to_senders(self.disagg_request_id, sender_endpoints)
                    logger.warning(str(exc))
                time.sleep(0.001)

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
        if getattr(self, "_closed", False):
            return
        with self._close_lock:
            if self._closed:
                return
            # Close admission before waiting on dispatch so a queued receive
            # cannot overtake this close attempt when the active dispatch
            # releases ``_receive_lock``.
            with self.lock:
                self._accepting_receives = False
            # Wait for an admitted receive to finish dispatch before evaluating
            # drain state. This gives retirement a stable publication frontier
            # and also serializes all AUX release.
            with self._receive_lock:
                with self.lock:
                    publication_pending = any(
                        not task._publication_closed for task in self._kv_tasks
                    )
                    terminal_gate_closed = self._terminal_status in (
                        SessionStatus.ERROR,
                        SessionStatus.CANCELLED,
                    )
                if publication_pending and not terminal_gate_closed:
                    raise RuntimeError(
                        f"cannot close RxSession {self.disagg_request_id}: "
                        "target publication is still pending"
                    )
                # clear_session is the fail-closed retirement gate. It raises
                # rather than dropping the only Python-level destination owner
                # while a direct writer is still active.
                if self._receiver is not None and not self._receiver_retired:
                    self._receiver.clear_session(self.disagg_request_id)
                    self._receiver_retired = True
                if self._aux_buffer is not None and self.aux_slot is not None:
                    self._aux_buffer.free_slot(self.aux_slot)
                    self.aux_slot = None
                self._destination_owner = None
                self._closed = True

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
        self._shutdown_started = False
        if addr is None and port is None:
            endpoint = f"tcp://{get_local_ip()}:*"
        else:
            endpoint = f"tcp://{addr}:{port}"
        self._messenger = ZMQMessenger(mode="ROUTER", endpoint=endpoint)
        self._start_listener()

    @property
    def endpoint(self) -> str:
        return self._messenger.endpoint

    def shutdown(self) -> bool:
        if self._shutdown:
            return True
        self._shutdown_started = True
        logger.debug("RankInfoServer.shutdown() called")
        try:
            self._messenger.stop()
        except Exception as e:
            logger.error(f"RankInfoServer.shutdown: listener stop failed: {e}")
            return False
        self._shutdown = True
        return True

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
        self._messenger.send([send_id, self._rank_info.to_bytes()])

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"RankInfoServer.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        try:
            stopped = self.shutdown()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error(
                f"RankInfoServer context cleanup failed while propagating an exception: {e}"
            )
            return False
        if not stopped:
            if exc_type is None:
                raise RuntimeError("RankInfoServer context exited before its listener stopped")
            logger.error("RankInfoServer retained its listener while propagating an exception")
        return False


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
        self._shutdown_attempt_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown = False
        self._session_admission_lock = threading.Lock()
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
        with self._session_admission_lock:
            if self._shutdown_started:
                raise RuntimeError("TransferWorker is shutting down; new sessions are not accepted")
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
                source_owner=request,
            )

    def create_rx_session(self, request: LlmRequest) -> RxSession:
        with self._session_admission_lock:
            if self._shutdown_started:
                raise RuntimeError("TransferWorker is shutting down; new sessions are not accepted")
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
                destination_owner=request,
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

    def _retain_for_shutdown_retry(self) -> bool:
        _NON_DRAINED_TRANSFER_WORKERS.add(self)
        return False

    def shutdown(self) -> bool:
        """Try to tear down transport resources in ownership-safe order.

        A local agent shutdown is not proof that remote processes can no
        longer submit one-sided writes. If receive results have not provided
        terminal evidence, this method returns ``False`` and intentionally
        keeps listeners, registrations, and mappings alive for a later retry.
        """
        attempt_lock = getattr(self, "_shutdown_attempt_lock", None)
        if attempt_lock is None:
            attempt_lock = threading.Lock()
            self._shutdown_attempt_lock = attempt_lock
        with attempt_lock:
            return self._shutdown_serialized()

    def _shutdown_serialized(self) -> bool:
        if getattr(self, "_shutdown", False):
            _NON_DRAINED_TRANSFER_WORKERS.discard(self)
            return True
        # Install the strong retry owner before the first fallible nested
        # teardown. This also covers exceptions reached from __del__/__exit__.
        _NON_DRAINED_TRANSFER_WORKERS.add(self)
        try:
            return self._shutdown_impl()
        except Exception as e:
            logger.error(f"TransferWorker.shutdown: retaining resources after failure: {e}")
            return False

    def _shutdown_impl(self) -> bool:
        if getattr(self, "_agent_shutdown_failed", False):
            return self._retain_for_shutdown_retry()
        admission_lock = getattr(self, "_session_admission_lock", None)
        if admission_lock is None:
            self._shutdown_started = True
        else:
            with admission_lock:
                self._shutdown_started = True

        # Use getattr guards because __init__ may have failed partway.
        rank_info_server = getattr(self, "_rank_info_server", None)
        if rank_info_server is not None and not rank_info_server.shutdown():
            return self._retain_for_shutdown_retry()

        receiver = getattr(self, "_receiver", None)
        if receiver is not None:
            # Close admission while preserving result ingress.
            receiver.begin_shutdown()

        sender = getattr(self, "_sender", None)
        if sender is not None and not sender.shutdown():
            return self._retain_for_shutdown_retry()

        # Sender work is locally terminal, but remote writers are independent.
        # Keep the receiver and every mapping live until their exact terminal
        # results (including AUX/legacy paths) have drained.
        if receiver is not None and not receiver.transfers_drained:
            logger.warning(
                "TransferWorker.shutdown: receive transfers remain active; "
                "retaining listeners and registered memory"
            )
            return self._retain_for_shutdown_retry()
        if receiver is not None and not receiver.shutdown():
            return self._retain_for_shutdown_retry()

        bounce = getattr(self, "_bounce", None)
        if bounce is not None:
            try:
                bounce.close()
            except Exception as e:
                logger.error(f"TransferWorker.shutdown: retaining unsafe/live bounce mapping: {e}")
                return self._retain_for_shutdown_retry()

        agent = getattr(self, "_agent", None)
        if agent is not None:
            registered = getattr(self, "_registered_mem", [])
            for desc in list(registered):
                try:
                    agent.deregister_memory(desc)
                except Exception as e:
                    logger.error(
                        f"TransferWorker.shutdown: retaining registered memory after "
                        f"deregister failure: {e}"
                    )
                    return self._retain_for_shutdown_retry()
                registered.remove(desc)
            try:
                agent.shutdown()
            except Exception as e:
                # Some bindings clear their local handle before propagating an
                # error. Latch this outcome so a later no-op cannot be mistaken
                # for successful quiescence/cleanup.
                self._agent_shutdown_failed = True
                logger.error(f"TransferWorker.shutdown: agent shutdown failed: {e}")
                return self._retain_for_shutdown_retry()
            self._agent = None
        self._shutdown = True
        _NON_DRAINED_TRANSFER_WORKERS.discard(self)
        return True

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"TransferWorker.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        try:
            drained = self.shutdown()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error(
                f"TransferWorker context cleanup failed while propagating an exception: {e}"
            )
            return False
        if not drained:
            if exc_type is None:
                raise RuntimeError("TransferWorker context exited before resources drained")
            logger.error("TransferWorker retained resources while propagating an exception")
        return False
