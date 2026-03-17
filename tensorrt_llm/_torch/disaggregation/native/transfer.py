from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

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
    MemoryDesc,
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
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer, perf_log_manager
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.runtime.generation import CUASSERT

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType

# Number of worker threads for KV transfer queues (default: 1)
KV_TRANSFER_NUM_THREADS = int(os.environ.get("TRTLLM_KV_TRANSFER_NUM_THREADS", "1"))


@dataclass
class RecvReqInfo:
    sender_req_id: int
    instance_name: str
    instance_rank: int
    block_ids_per_layer_groups: list[
        np.ndarray
    ]  # Block IDs per layer group, each np.ndarray(dtype=np.int64)
    unique_rid: int
    start_token_idx: Optional[int] = None
    aux_slot: Optional[int] = None

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
                "start_token_idx": self.start_token_idx,
                "aux_slot": self.aux_slot,
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


@dataclass
class WriteMeta:
    task_future: concurrent.futures.Future
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


class MessageType:
    TERMINATION = b"TERMINATION"
    KV_AGENT_RESULT = b"KV_AGENT_RESULT"
    REQUEST_DATA = b"REQUEST_DATA"
    REQUEST_INSTANCE_INFO = b"REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = b"REGISTER_RANK_INFO"
    AUX_AGENT_RESULT = b"AUX_AGENT_RESULT"


class TaskStatus(Enum):
    INIT = "INIT"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ERROR = "ERROR"


class AgentResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class SendTaskBase:
    def __init__(self, params: DisaggregatedParams):
        self.status = TaskStatus.INIT
        self.future = concurrent.futures.Future()
        self._params = params
        assert params.disagg_request_id is not None
        self._unique_rid: int = params.disagg_request_id
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

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
    ):
        super().__init__(params)
        self.slice_id = slice_id
        self.transferred_count = 0
        self._slice = kv_slice


class Sender(SenderBase):
    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        agent: BaseTransferAgent,
    ):
        self._registrar = peer_registrar
        self._device_id = peer_registrar.self_rank_info.device_id
        self._agent = agent
        self._peer_requests: dict = {}
        self._peer_requests_lock = threading.Lock()
        self._messenger = ZMQMessenger(mode="ROUTER")
        self._dealers = {}
        self._sessions = {}  # unique_rid -> TxSession
        self._sessions_lock = threading.Lock()  # Protects _sessions access
        self._shutdown = False
        self._instance_rank = self._registrar.self_rank_info.instance_rank
        self._loaded_remote_agents: set[str] = set()
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

    def _add_req_info(self, unique_rid: int, instance_rank: int, req_info: RecvReqInfo):
        with self._peer_requests_lock:
            if unique_rid not in self._peer_requests:
                self._peer_requests[unique_rid] = {}
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

    def setup_session(self, tx_session: "TxSession"):
        unique_rid = tx_session.disagg_request_id
        with self._sessions_lock:
            self._sessions[unique_rid] = weakref.ref(tx_session)

        req_info = self._get_first_req_info(unique_rid)

        if req_info:
            peer_ri = self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            )
            expected_count = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
            if self._is_req_ready(unique_rid, expected_count):
                tx_session.receiver_ready = True
        return

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
        # Distribute tasks to threads by unique_rid to ensure same session's tasks
        # are processed by the same thread in order
        thread_idx = write_meta.unique_rid % self._num_threads
        self._send_task_queues[thread_idx].put(write_meta)

    def _process_task_queue(self, thread_idx: int):
        device_id = self._device_id
        torch.cuda.set_device(device_id)
        CUASSERT(cudart.cudaSetDevice(device_id))

        task_queue = self._send_task_queues[thread_idx]
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
                if not write_meta.task_future.done():
                    write_meta.task_future.set_exception(e)

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

        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, write_meta.peer_name, None
        )
        return request

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, write_meta: WriteMeta):
        assert write_meta.src_ptrs.size == write_meta.dst_ptrs.size == write_meta.sizes.size, (
            f"WriteMeta ptr/size mismatch for unique_rid={write_meta.unique_rid}"
        )

        session = self._get_session(write_meta.unique_rid)
        if session is None:
            msg = (
                f"_deliver_kv_to_agent: TxSession {write_meta.unique_rid} not found or already GC'd"
            )
            logger.error(msg)
            if not write_meta.task_future.done():
                write_meta.task_future.set_exception(RuntimeError(msg))
            return
        assert write_meta.slice_id is not None
        task = session.kv_tasks[write_meta.slice_id]
        timer = task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)
        if session.status == SessionStatus.ERROR:
            logger.warning(
                f"_deliver_kv_to_agent: session {write_meta.unique_rid} already in ERROR state, skipping"
            )
            return
        task.status = TaskStatus.TRANSFERRING

        agent_result = AgentResult.SUCCESS
        if write_meta.src_ptrs.size > 0:
            request = Sender._make_agent_request(write_meta, device_id=self._device_id)
            if timer:
                timer.record_transfer_start(write_meta.peer_rank)
            if not self._agent.submit_transfer_requests(request).wait():
                agent_result = AgentResult.FAILED
                if not write_meta.task_future.done():
                    write_meta.task_future.set_exception(
                        RuntimeError(f"KV transfer failed for request {write_meta.unique_rid}")
                    )
                task.status = TaskStatus.ERROR
        if timer:
            timer.record_transfer_end(write_meta.peer_rank)

        ## TODO: just last slice need to send task state?
        self._get_or_connect_dealer(write_meta.peer_endpoint).send(
            [
                MessageType.KV_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                str(write_meta.slice_id).encode("ascii"),
                str(write_meta.is_last_slice).encode("ascii"),
                agent_result.value.encode("ascii"),
            ]
        )

        task.transferred_count += 1
        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)
        if task.transferred_count > write_meta.expected_transfers:
            session.set_exception(
                f"KV slice {write_meta.slice_id} received more than {write_meta.expected_transfers} transfers"
            )
        elif task.transferred_count == write_meta.expected_transfers:
            if write_meta.task_future.done():
                task.status = TaskStatus.ERROR
                session.set_exception(
                    f"KV slice {write_meta.slice_id} future already resolved on completion"
                )
            else:
                write_meta.task_future.set_result(AgentResult.SUCCESS)
                task.status = TaskStatus.TRANSFERRED

        logger.debug(
            f"deliver_kv_to_agent completed: unique_rid={write_meta.unique_rid}, "
            f"slice_id={write_meta.slice_id}, agent_result={agent_result}"
        )

    @nvtx_range("_deliver_aux_to_agent")
    def _deliver_aux_to_agent(self, write_meta: WriteMeta):
        session = self._get_session(write_meta.unique_rid)
        if session is None:
            msg = f"_deliver_aux_to_agent: TxSession {write_meta.unique_rid} not found or already GC'd"
            logger.error(msg)
            if not write_meta.task_future.done():
                write_meta.task_future.set_exception(RuntimeError(msg))
            return
        aux_task = session.aux_task
        assert aux_task is not None, f"aux_task is None for session {write_meta.unique_rid}"
        timer = aux_task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)

        agent_result = AgentResult.SUCCESS
        if write_meta.src_ptrs.size > 0:
            request = Sender._make_agent_request(write_meta, device_id=self._device_id)
            if timer:
                timer.record_transfer_start(write_meta.peer_rank)
            if not self._agent.submit_transfer_requests(request).wait():
                agent_result = AgentResult.FAILED
                session.set_exception("aux transfer agent request failed")
            if timer:
                timer.record_transfer_end(write_meta.peer_rank)

        self._get_or_connect_dealer(write_meta.peer_endpoint).send(
            [
                MessageType.AUX_AGENT_RESULT,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                agent_result.value.encode("ascii"),
            ]
        )

        aux_task._transfer_count += 1
        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        aux_task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)
        if aux_task._transfer_count == write_meta.expected_transfers:
            if aux_task.future.done():
                aux_task.status = TaskStatus.ERROR
                session.set_exception("aux future already resolved on completion")
            else:
                aux_task.future.set_result(AgentResult.SUCCESS)
                aux_task.status = TaskStatus.TRANSFERRED
        elif aux_task._transfer_count > write_meta.expected_transfers:
            session.set_exception(
                f"aux task received more than {write_meta.expected_transfers} transfers"
            )

    @staticmethod
    def _filter_kv_blocks(
        src_block_ids: np.ndarray, dst_block_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # TODO: filter the kv block_ids according to the peer_overlap
        return src_block_ids, dst_block_ids

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
        dst_device_id = None
        if self._registrar.should_send_kv(targets, peer_ri):
            dst_device_id = peer_ri.device_id
            extractor = self._registrar.self_extractor
            peer_extractor = self._registrar.peer_extractor(
                peer_ri.instance_name, peer_ri.instance_rank
            )
            pool_mapping = self._registrar.get_pool_mapping(peer_ri)
            dst_block_ids_per_groups = req_info.block_ids_per_layer_groups
            src_block_ids_per_groups = task._slice.block_ids_per_layer_groups

            # Aggregate fragments from all matching pools using numpy concatenation
            for (self_lg, self_pi), (peer_lg, peer_pi) in pool_mapping.items():
                src_block_ids = src_block_ids_per_groups[self_lg]
                dst_block_ids = dst_block_ids_per_groups[peer_lg]

                if src_block_ids.size + 1 == dst_block_ids.size:
                    # FIXME: this is a temporary solution, need to be fixed for the draft tokens
                    logger.warning(
                        "src_block_num is one less than dst_block_num, maybe it is due to draft tokens,"
                        " remove the last block from dst_block_ids "
                    )
                    dst_block_ids = dst_block_ids[:-1]
                src_block_ids, dst_block_ids = Sender._filter_kv_blocks(
                    src_block_ids, dst_block_ids
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

        if timer:
            timer.record_prepare_args_end(peer_ri.instance_rank)
            timer.record_transfer_sizes(peer_ri.instance_rank, int(kv_sizes.sum()), dst_frags.size)

        return WriteMeta(
            task_future=task.future,
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
            task_future=task.future,
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            sizes=sizes,
            expected_transfers=expected_transfers,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_rank=req_info.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=task._unique_rid,
            meta_type=WriteMetaType.AUX,
        )

    def dispatch_task(
        self,
        task: KVSendTask | AuxSendTask,
        req_info_snapshot: Optional[dict] = None,
    ):
        # req_info_snapshot may be pre-fetched under session.lock by the caller to keep the
        # critical section small.  When not provided, we fetch it here (legacy / standalone path).
        if req_info_snapshot is None:
            req_info_snapshot = dict(self._get_req_info(task._unique_rid) or {})
        for info in req_info_snapshot.values():
            if task._perf_timer is not None:
                task._perf_timer.record_task_start(info.instance_rank)
            if isinstance(task, KVSendTask):
                trans_meta = self._build_kv_write_meta(task, info)
            else:
                trans_meta = self._build_aux_write_meta(task, info)
            if task._perf_timer is not None:
                task._perf_timer.record_push_start(trans_meta.peer_rank)
            self._enqueue(trans_meta)

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
                case _:
                    logger.error(f"Sender received unknown message type: {msg[0]}")

        self._messenger.start_listener(handle_message)

    def _register_peer_rank(self, _send_id: bytes, message: list[bytes]):
        ri: RankInfo = RankInfo.from_bytes(message[1])

        self._registrar.register(ri.instance_name, ri.instance_rank, ri)

        agent_name = ri.instance_name + str(ri.instance_rank)
        logger.debug(f"Loading remote transfer agent descriptor for peer '{agent_name}'")
        self._agent.load_remote_agent(
            ri.instance_name + str(ri.instance_rank),
            ri.transfer_engine_info,
        )
        self._loaded_remote_agents.add(agent_name)
        logger.debug(
            f"Completed handling REGISTER_RANK_INFO for instance='{ri.instance_name}', rank={ri.instance_rank}"
        )

    @nvtx_range("_respond_with_kv")
    def _respond_with_kv(self, _send_id: bytes, message: list[bytes]):
        # A session's KV send may race with incoming req_infos from multiple gen ranks.
        # _sessions_lock guards against session insertion between _save_peer_req_info and
        # _get_session; session.lock serializes _enqueue calls from both paths.
        info: RecvReqInfo = RecvReqInfo.from_bytes(message[1])
        with self._sessions_lock:
            session = self._get_session(info.unique_rid)
            if session is None:
                self._save_peer_req_info(info)
                return
        with session.lock:
            self._save_peer_req_info(info)
            tasks = list(session.kv_tasks)
        for task in tasks:
            if task._perf_timer is not None:
                task._perf_timer.record_task_start(info.instance_rank)
            trans_meta = self._build_kv_write_meta(task, info)
            if task._perf_timer is not None:
                task._perf_timer.record_push_start(trans_meta.peer_rank)
            self._enqueue(trans_meta)

    def _get_or_connect_dealer(self, endpoint: Optional[str]):
        if endpoint is None:
            raise ValueError("Sender: peer endpoint is None; peer may not have registered yet")
        if endpoint not in self._dealers:
            self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._dealers[endpoint]

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

    def clear_session(self, unique_rid: int):
        with self._sessions_lock:
            if unique_rid in self._sessions:
                del self._sessions[unique_rid]
        self._remove_req_info(unique_rid)

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True

        for q in self._send_task_queues:
            q.put(None)
        for t in self._worker_threads:
            t.join(timeout=5)
        # Invalidate all loaded remote agents to release fabric/POSIX FD resources
        for agent_name in self._loaded_remote_agents:
            try:
                self._agent.invalidate_remote_agent(agent_name)
            except Exception as e:
                logger.warning(
                    f"Failed to invalidate remote agent '{agent_name}' during shutdown: {e}"
                )
        self._loaded_remote_agents.clear()
        for dealer in self._dealers.values():
            try:
                dealer.stop()
            except Exception as e:
                logger.warning(f"Failed to stop dealer during Sender shutdown: {e}")
        self._dealers.clear()
        self._messenger.stop()

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
    ):
        super().__init__(sender, SessionArgsBase(params))
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
        # Must be last: makes session visible to listener thread,
        # so all attributes above must be initialized first.
        self._sender.setup_session(self)

    @property
    def status(self) -> SessionStatus:
        if self._exception is not None or any(t.status == TaskStatus.ERROR for t in self.kv_tasks):
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

    def send(self, slice: KVSlice) -> concurrent.futures.Future:
        with self.lock:
            params = self._base_args.params
            slice_id = len(self.kv_tasks)
            task = KVSendTask(slice, params, slice_id)
            self.kv_tasks.append(task)
            req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
        self._sender.dispatch_task(task, req_info_snapshot)
        return task.future

    def send_aux(self) -> AuxSendTask:
        with self.lock:
            params = self._base_args.params
            task = AuxSendTask(params, self.aux_slot)
            self.aux_task = task
            req_info_snapshot = dict(self._sender._get_req_info(task._unique_rid) or {})
        self._sender.dispatch_task(task, req_info_snapshot)
        return task

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
        """Non-blocking check: has the transfer failed?"""
        return self.status == SessionStatus.ERROR

    def wait_complete(self) -> Optional[WaitResult]:
        """Block until KV (and optionally aux) transfer finishes.

        Returns WaitResult.COMPLETED, WaitResult.FAILED, or WaitResult.TIMEOUT.
        """
        try:
            for task in self.kv_tasks:
                kv_status = task.future.result(timeout=self._timeout_s)
                if kv_status != AgentResult.SUCCESS:
                    return WaitResult.FAILED
            if self._need_aux and self.aux_task is not None:
                aux_status = self.aux_task.future.result(timeout=self._timeout_s)
                if aux_status != AgentResult.SUCCESS:
                    return WaitResult.FAILED
            return WaitResult.COMPLETED
        except TimeoutError:
            return WaitResult.TIMEOUT
        except Exception:
            return WaitResult.FAILED

    def set_exception(self, reason: str = ""):
        msg = f"TxSession {self.disagg_request_id} exception"
        if reason:
            msg += f": {reason}"
        self._exception = RuntimeError(msg)
        for task in self.kv_tasks:
            if not task.future.done():
                task.future.set_exception(self._exception)
        if self.aux_task is not None and not self.aux_task.future.done():
            self.aux_task.future.set_exception(self._exception)

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._aux_buffer is not None and self.aux_slot is not None:
            self._aux_buffer.free_slot(self.aux_slot)
            self.aux_slot = None
        # Unregister from Sender; do not null out fields — worker threads
        # may still access kv_tasks/aux_task/_sender for in-flight transfers.
        if self._sender is not None:
            self._sender.clear_session(self.disagg_request_id)

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
        self.future = concurrent.futures.Future()
        self.slice_id = slice_id
        self.status = TaskStatus.INIT
        self.expected_transfers = 0
        self.last_slice_count = 0

        self._unique_rid = unique_rid
        self._kv_slice = kv_slice
        self._params = params
        self._exception = None
        self._aux_slot = aux_slot
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

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
    ):
        self._registrar = peer_registrar
        self._agent = agent
        self._dealers = {}
        self._sender_ep_instance_map = {}

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._sessions = {}  # unique_rid -> RxSession
        self._sessions_lock = threading.Lock()
        self._shutdown = False

        self._start_listener()
        logger.info(f"Receiver init with endpoint: {self._messenger.endpoint}")

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def shutdown(self):
        if getattr(self, "_shutdown", False):
            return
        self._shutdown = True
        for dealer in self._dealers.values():
            try:
                dealer.stop()
            except Exception as e:
                logger.warning(f"Failed to stop dealer during Receiver shutdown: {e}")
        self._dealers.clear()
        self._messenger.stop()

    def clear_session(self, unique_rid: int):
        with self._sessions_lock:
            self._sessions.pop(unique_rid, None)

    def setup_session(self, rx_session: RxSessionBase):
        with self._sessions_lock:
            self._sessions[rx_session.disagg_request_id] = weakref.ref(rx_session)

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
        return RecvReqInfo(
            sender_req_id=task._params.ctx_request_id,
            instance_name=self_ri.instance_name,
            instance_rank=self_ri.instance_rank,
            block_ids_per_layer_groups=task._kv_slice.block_ids_per_layer_groups,
            unique_rid=task._unique_rid,
            aux_slot=task._aux_slot,
        )

    def dispatch_task(self, task: KVRecvTask):
        params = task._params
        logger.debug(f"Preparing async data transfer request for disagg_params={params}")
        receiver_req = self._build_recv_req_info(task)
        sender_dp_rank = params.ctx_dp_rank
        if sender_dp_rank is None:
            raise ValueError(
                f"ctx_dp_rank is None for request {task._unique_rid}; "
                "disaggregated params may be missing context rank info"
            )
        peer_infos: RankInfo = self._get_sender_info(params)
        peer_overlap = self._registrar.get_peer_overlap(peer_infos, sender_dp_rank)
        task.expected_transfers = len(peer_overlap.ranks)
        session = self._get_session(task._unique_rid)
        if session is None:
            raise RuntimeError(
                f"dispatch_task: RxSession {task._unique_rid} not found; "
                "session may have been closed before dispatch"
            )
        session.mark_transferring(task.slice_id)
        for rank in peer_overlap.ranks:
            if task._perf_timer is not None:
                task._perf_timer.record_task_start(rank)
            self._request_sender_data(peer_infos.sender_endpoints[rank], receiver_req)
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
        if endpoint not in self._dealers:
            self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._dealers[endpoint]

    def _get_sender_info(self, params: DisaggregatedParams) -> RankInfo:
        info_endpoint = self._extract_info_endpoint(params)
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
                case _:
                    logger.error(f"Receiver received unknown message type: {msg[0]}")
            return True

        self._messenger.start_listener(handle_message)

    def _process_kv_agent_result(self, _send_id: bytes, message: list[bytes]):
        msg_type, peer_rank, unique_rid, slice_id_str, is_last_slice_str, status = decode_message(
            message
        )
        peer_rank = int(peer_rank)
        unique_rid = int(unique_rid)
        slice_id = int(slice_id_str)
        if msg_type.encode("ascii") != MessageType.KV_AGENT_RESULT:
            logger.error(
                f"_process_kv_agent_result: unexpected msg_type={msg_type!r}, expected KV_AGENT_RESULT"
            )
            return
        session = self._get_session(unique_rid)
        if session is None:
            logger.warning(
                f"_process_kv_agent_result: session {unique_rid} not found (already closed?), dropping status"
            )
            return
        session.process_kv_agent_result(
            peer_rank, slice_id, is_last_slice_str == "True", AgentResult(status)
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
        session.process_aux_agent_result(peer_rank, AgentResult(status))

    def _request_sender_data(self, endpoint: str, receiver_info: RecvReqInfo):
        logger.debug(
            f"Sending data request to endpoint '{endpoint}' with request info: {receiver_info}"
        )
        messenger = self._get_or_connect_dealer(endpoint)
        messenger.send([MessageType.REQUEST_DATA, receiver_info.to_bytes()])

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
    ):
        super().__init__(receiver, SessionArgsBase(params))
        self._timeout_s = timeout_s
        self._need_aux = params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        self._receiver: Receiver  # narrow base class type for Pylance
        self.request_id = request_id
        self._aux_buffer = aux_buffer
        self.aux_slot = aux_buffer.alloc_slot().id if aux_buffer is not None else None
        self._exception: Optional[Exception] = None
        self._closed = False
        self._kv_tasks: list[KVRecvTask] = []
        self._aux_count = 0
        self._aux_status: TaskStatus = TaskStatus.INIT
        self._receiver.setup_session(self)

    @property
    def status(self) -> SessionStatus:
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
        self._kv_tasks[slice_id].status = TaskStatus.TRANSFERRING

    def receive(self, slice: KVSlice) -> concurrent.futures.Future:
        params = self._base_args.params
        slice_id = len(self._kv_tasks)
        task = KVRecvTask(
            params.disagg_request_id,
            slice,
            slice_id,
            params,
            aux_slot=self.aux_slot,
        )
        self._kv_tasks.append(task)
        self._receiver.dispatch_task(task)
        return task.future

    def process_kv_agent_result(
        self, peer_rank: int, slice_id: int, is_last_slice: bool, status: AgentResult
    ):
        task = self._kv_tasks[slice_id]
        if status == AgentResult.SUCCESS:
            if is_last_slice:
                task.last_slice_count += 1
                if task.last_slice_count == task.expected_transfers:
                    if not task.future.done():
                        task.future.set_result(AgentResult.SUCCESS)
                    task.status = TaskStatus.TRANSFERRED

                    logger.debug(
                        f"KV transfer complete for request {self.request_id} slice {slice_id}"
                    )
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_end(peer_rank)
                    ri = self._receiver._registrar.self_rank_info
                    task.print_perf_info(peer_rank, ri.instance_name, ri.instance_rank)
        elif status == AgentResult.FAILED:
            if not task.future.done():
                task.future.set_exception(
                    RuntimeError(
                        f"KV transfer failed for request {self.request_id} slice {slice_id}"
                    )
                )
            task.status = TaskStatus.ERROR
        else:
            raise ValueError(
                f"Session {self.request_id} received unknown task status: {status.value}"
            )

    def process_aux_agent_result(self, _peer_rank: int, status: AgentResult):
        # Aux is session-level (not per-slice); expected_transfers is identical
        # across all kv_tasks, so any task provides the right count.
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
                logger.error(str(self._exception))
        elif status == AgentResult.FAILED:
            self._aux_status = TaskStatus.ERROR
            self._exception = RuntimeError(f"Session {self.request_id} aux transfer failed")
        else:
            raise ValueError(
                f"Session {self.request_id} received unknown aux send status: {status}"
            )

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def unpack_aux(self, request: LlmRequest) -> None:
        """Read token data from the aux buffer slot into the given request."""
        assert self._aux_buffer is not None, "No aux_buffer set for this session"
        assert self.aux_slot is not None, "No aux_slot set for this session"
        first_gen_tokens, draft_tokens = self._aux_buffer.get_slot_tokens(self.aux_slot)
        request.py_first_gen_tokens = first_gen_tokens  # type: ignore[attr-defined]
        request.py_draft_tokens = draft_tokens  # type: ignore[attr-defined]

    def is_completed(self) -> bool:
        """Non-blocking check: has the transfer completed successfully?"""
        status = self.status
        if self._need_aux:
            return status == SessionStatus.FULLY_TRANSFERRED
        return status in (SessionStatus.KV_TRANSFERRED, SessionStatus.FULLY_TRANSFERRED)

    def has_failed(self) -> bool:
        """Non-blocking check: has the transfer failed?"""
        return self.status == SessionStatus.ERROR

    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]:
        """Block until transfer completes.

        With blocking=False (default): returns None if KV is done but transfer
        not fully complete — caller should re-poll next cycle.
        With blocking=True: spins until fully complete.
        Returns WaitResult.COMPLETED on full success, WaitResult.FAILED on error.
        """
        try:
            for task in self._kv_tasks:
                kv_status = task.future.result()
                if kv_status != AgentResult.SUCCESS:
                    return WaitResult.FAILED
            if self._need_aux:
                while True:
                    status = self.status
                    if status == SessionStatus.FULLY_TRANSFERRED:
                        return WaitResult.COMPLETED
                    elif status == SessionStatus.ERROR:
                        return WaitResult.FAILED
                    if not blocking:
                        return None  # KV done, aux still in flight; re-poll next cycle
                    time.sleep(0.001)
            return WaitResult.COMPLETED
        except TimeoutError:
            return WaitResult.FAILED
        except Exception:
            return WaitResult.FAILED

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._aux_buffer is not None and self.aux_slot is not None:
            self._aux_buffer.free_slot(self.aux_slot)
            self.aux_slot = None
        # Unregister from Receiver; do not null out fields — listener thread
        # may still access _kv_tasks/_receiver for in-flight status messages.
        if self._receiver is not None:
            self._receiver.clear_session(self.disagg_request_id)

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
        self._messenger.send([send_id, self._rank_info.to_bytes()])

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


def _deregister_registered_memory(transfer_agent, registered_memorys):
    try:
        if transfer_agent is None or not registered_memorys:
            return
        while registered_memorys:
            register_memory = registered_memorys[0]
            try:
                logger.info(f"Deregistering transfer memory: {register_memory}")
                transfer_agent.deregister_memory(register_memory)
            except Exception:
                logger.error("deregister memory failed in finalizer")
            registered_memorys.pop(0)
    except Exception:
        logger.error("unexpected error in _deregister_registered_memory finalizer")


@dataclass
class TransferWorkerConfig:
    kv_cache_manager: KVCacheManager
    device_id: int
    instance_name: str
    max_concurrent_sessions: int = 0
    max_draft_len: Optional[int] = None
    tx_timeout_s: Optional[float] = None
    rx_timeout_s: Optional[float] = None


class TransferWorker:
    def __init__(self, config: TransferWorkerConfig):
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
        )

    def has_all_peer_req_infos_for_send(self, unique_rid: int) -> bool:
        return self._sender.has_all_peer_req_infos(unique_rid)

    def _setup_peer_infrastructure(self, kvm: KVCacheManager):
        self._rank_info_server = RankInfoServer(self._rank_info) if kvm.mapping.rank == 0 else None
        self._kv_extractor = KVRegionExtractorV1(kvm)
        self._peer_registrar = PeerRegistrar(self._rank_info, self._kv_extractor)

    def _setup_transfer_engine(self):
        self._agent = _create_nixl_agent(
            self._rank_info.instance_name + str(self._rank_info.instance_rank)
        )
        self._registered_mem: list = []
        self._finalizer = weakref.finalize(
            self, _deregister_registered_memory, self._agent, self._registered_mem
        )
        try:
            self._register_kv_cache()
            if self._aux_buffer is not None:
                self._register_aux_buffer()
            self._sender = Sender(self._peer_registrar, self._agent)
            self._receiver = Receiver(self._peer_registrar, self._agent)
            self._rank_info.transfer_engine_info = bytes(self._agent.get_local_agent_desc())
            self._rank_info.self_endpoint = self._receiver.endpoint
        except Exception:
            self._finalizer()
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

    def shutdown(self):
        if getattr(self, "_shutdown", False):
            return
        self._shutdown = True
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
        # Deregister NIXL memory before shutting down components, so that
        # pinned GPU memory is released and can be re-allocated (e.g. when
        # the KV cache manager is recreated after profiling).
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer()

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"TransferWorker.__del__: exception during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()
