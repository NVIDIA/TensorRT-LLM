import concurrent
import os
import queue
import threading
import weakref
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

import msgpack
import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm.bindings
from tensorrt_llm import Mapping, logger
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
    RxSessionBase,
    SessionArgsBase,
    SessionStatus,
    TaskIdType,
    TxSessionBase,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger, decode_message
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer, perf_log_manager
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.runtime.generation import CUASSERT

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType

# Environment variable to control the number of threads for task queue processing
# Default is 1 (single-threaded, original behavior)
KV_TRANSFER_NUM_THREADS = int(os.environ.get("TRTLLM_KV_TRANSFER_NUM_THREADS", "1"))


@dataclass
class RecvReqInfo:
    sender_req_id: int
    instance_name: str
    instance_rank: int
    block_ids_per_layer_groups: list[list[int]]  # Block IDs per layer group
    unique_rid: int
    start_token_idx: Optional[int] = None
    aux_slot: Optional[int] = None

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> "RecvReqInfo":
        return cls(**msgpack.unpackb(data, raw=False))


@dataclass
class ReadMeta:
    unique_rid: int
    slice_id: int
    target_ranks: Optional[List[int]] = None


@dataclass
class WriteMeta:
    future_for_task: concurrent.futures.Future
    expected_transfers: int
    peer_name: str
    peer_rank: int
    peer_endpoint: str
    unique_rid: int
    src_ptrs: List[int]
    dst_ptrs: List[int]
    sizes: List[int]
    dst_device_id: Optional[int] = None
    slice_id: Optional[int] = None
    is_last_slice: bool = False

    @property
    def is_aux(self) -> bool:
        """True for aux (token) transfers, False for KV (cache) transfers."""
        return self.slice_id is None


class MessageType:
    TERMINATION = b"TERMINATION"
    TASK_STATUS = b"TASK_STATUS"
    PEER_INFO = b"PEER_INFO"
    REQUEST_DATA = b"REQUEST_DATA"
    REQUEST_INSTANCE_INFO = b"REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = b"REGISTER_RANK_INFO"
    AUX_SEND_STATUS = b"AUX_SEND_STATUS"


class TaskStatus(Enum):
    INIT = "INIT"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ERROR = "ERROR"


class SyncStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AuxSendTask:
    def __init__(self, params: DisaggregatedParams, slot: int):
        self._params = params
        self._unique_rid = params.disagg_request_id
        self._slot = slot
        self._status = TaskStatus.INIT
        self._future = concurrent.futures.Future()
        self._transferred_count = 0
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    def print_perf_info(self, peer_rank: int, instance_name: str, instance_rank: int):
        perf_log_manager.log_task_perf(
            "AuxSendTask",
            self._unique_rid,
            peer_rank,
            instance_name,
            instance_rank,
            self._perf_timer,
        )


class KVSendTask:
    def __init__(
        self,
        kv_slice: KVSlice,
        params: DisaggregatedParams,
        slice_id: int,
    ):
        self._future = concurrent.futures.Future()
        self._expected_transfers = 0
        self._slice = kv_slice
        self._params = params
        self._unique_rid = params.disagg_request_id
        self._slice_id = slice_id
        self._status = TaskStatus.INIT
        self._transferred_count = 0
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def transferred_count(self) -> int:
        return self._transferred_count

    @transferred_count.setter
    def transferred_count(self, v: int):
        self._transferred_count = v

    def print_perf_info(self, peer_rank: int, instance_name: str, instance_rank: int):
        perf_log_manager.log_task_perf(
            "KVSendTask",
            self._unique_rid,
            peer_rank,
            instance_name,
            instance_rank,
            self._perf_timer,
        )


@dataclass
class ReqInfoManager:
    # unique_rid -> instance_rank -> RecvReqInfo
    _peer_requests: dict[str, dict[int, RecvReqInfo]] = field(default_factory=dict)
    _lock = threading.Lock()

    def add_req_info(self, unique_rid: str, instance_rank: int, req_info: RecvReqInfo):
        with self._lock:
            if unique_rid not in self._peer_requests:
                self._peer_requests[unique_rid] = {}
            self._peer_requests[unique_rid][instance_rank] = req_info

    def is_ready(self, unique_rid: str, expected_count: int) -> bool:
        with self._lock:
            requests = self._peer_requests.get(unique_rid)
            if not requests:
                return False
            return len(requests) == expected_count

    def get_req_info(self, unique_rid: str) -> Optional[dict[int, RecvReqInfo]]:
        with self._lock:
            return self._peer_requests.get(unique_rid)

    def get_first_req_info(self, unique_rid: str) -> Optional[RecvReqInfo]:
        with self._lock:
            reqs = self._peer_requests.get(unique_rid)
            if not reqs:
                return None
            return next(iter(reqs.values()))

    def remove_req_info(self, unique_rid: str):
        with self._lock:
            if unique_rid in self._peer_requests:
                del self._peer_requests[unique_rid]


class Sender:
    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._registrar = peer_registrar
        self._device_id = device_id
        self._agent = agent
        self._peer_reqs = ReqInfoManager()
        self._messenger = ZMQMessenger(mode="ROUTER")
        self._dealers = {}
        self._tx_sessions = {}  # unique_rid -> TxSession
        self._sessions_lock = threading.Lock()  # Protects _tx_sessions access
        self._closed = False
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

    def setup_session(self, tx_session: TxSessionBase):
        unique_rid = tx_session.disagg_request_id
        with self._sessions_lock:
            self._tx_sessions[unique_rid] = weakref.ref(tx_session)

        req_info = self._peer_reqs.get_first_req_info(unique_rid)

        if req_info:
            peer_ri = self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            )
            expected_count = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
            if self._peer_reqs.is_ready(unique_rid, expected_count):
                tx_session.set_peer_ready()
        return

    def _get_tx_session(self, unique_rid: int) -> "TxSession":
        session_ref = self._tx_sessions.get(unique_rid)
        if session_ref is None:
            return None
        session = session_ref()
        if session is None:
            logger.warning(f"TxSession {unique_rid} has been garbage collected")
            return None
        return session

    def submit_task(self, write_meta: WriteMeta):
        # Distribute tasks to threads by unique_rid to ensure same session's tasks
        # are processed by the same thread in order
        thread_idx = write_meta.unique_rid % self._num_threads
        self._send_task_queues[thread_idx].put(write_meta)

    def _process_task_queue(self, thread_idx: int):
        """Process tasks from the queue assigned to this thread.

        Args:
            thread_idx: Index of the worker thread (0 to num_threads-1)
        """
        device_id = self._device_id
        torch.cuda.set_device(device_id)
        CUASSERT(cudart.cudaSetDevice(device_id))

        task_queue = self._send_task_queues[thread_idx]
        while True:
            write_meta = task_queue.get()
            if write_meta is None:
                break
            if write_meta.is_aux:
                logger.debug(
                    f"_process_task_queue[{thread_idx}]: delivering aux task to agent: {write_meta}"
                )
                self._deliver_aux_to_agent(write_meta)
            else:
                self._deliver_kv_to_agent(write_meta)

    @staticmethod
    @nvtx_range("_make_agent_request")
    def _make_agent_request(write_meta: WriteMeta, device_id: int):
        if write_meta.is_aux:
            src_list = [
                (src_ptr, size, 0) for src_ptr, size in zip(write_meta.src_ptrs, write_meta.sizes)
            ]
            dst_list = [
                (dst_ptr, size, 0) for dst_ptr, size in zip(write_meta.dst_ptrs, write_meta.sizes)
            ]
            src_mem_type = MemoryType.DRAM
            dst_mem_type = MemoryType.DRAM
        else:
            assert write_meta.dst_device_id is not None
            src_list = [
                (src_ptr, size, device_id)
                for src_ptr, size in zip(write_meta.src_ptrs, write_meta.sizes)
            ]
            dst_list = [
                (dst_ptr, size, write_meta.dst_device_id)
                for dst_ptr, size in zip(write_meta.dst_ptrs, write_meta.sizes)
            ]
            src_mem_type = MemoryType.VRAM
            dst_mem_type = MemoryType.VRAM
        peer_name = write_meta.peer_name

        # Use C++ MemoryDescs directly with batch constructor (list of tuples)
        src_memory_descs = MemoryDescs(src_mem_type, src_list)
        dst_memory_descs = MemoryDescs(dst_mem_type, dst_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, peer_name, None
        )
        return request, src_list, dst_list

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, write_meta: WriteMeta):
        assert (
            write_meta.src_ptrs is not None
            and write_meta.dst_ptrs is not None
            and write_meta.sizes is not None
        )
        assert len(write_meta.src_ptrs) == len(write_meta.dst_ptrs)
        assert len(write_meta.sizes) == len(write_meta.src_ptrs)

        session = self._get_tx_session(write_meta.unique_rid)
        assert session is not None
        task = session.kv_tasks[write_meta.slice_id]
        timer = task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)
        assert session.status != SessionStatus.ERROR
        task.status = TaskStatus.TRANSFERRING

        skip_send = len(write_meta.src_ptrs) == 0
        logger.debug(f"Submitting transfer request to transfer agent, skip_send={skip_send}")
        sync_status = SyncStatus.SUCCESS
        if timer:
            timer.record_transfer_start(write_meta.peer_rank)
        if not skip_send:
            request, _, _ = Sender._make_agent_request(write_meta, device_id=self._device_id)
            if not self._agent.submit_transfer_requests(request).wait():
                sync_status = SyncStatus.FAILED
                write_meta.future_for_task.set_exception(RuntimeError("Transfer failed"))
                task.status = TaskStatus.ERROR
        if timer:
            timer.record_transfer_end(write_meta.peer_rank)

        ## TODO: just last slice need to send task state?
        self._get_or_connect_dealer(write_meta.peer_endpoint).send(
            [
                MessageType.TASK_STATUS,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                str(write_meta.slice_id).encode("ascii"),
                str(write_meta.is_last_slice).encode("ascii"),
                sync_status.value.encode("ascii"),
            ]
        )

        task.transferred_count += 1
        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)
        if task.transferred_count > write_meta.expected_transfers:
            write_meta.future_for_task.set_exception(
                RuntimeError(
                    f"Session {write_meta.unique_rid} has more than {write_meta.expected_transfers} transfers"
                )
            )
            # TODO: set exception for the session ?
            session.mark_error()
        elif task.transferred_count == write_meta.expected_transfers:
            # TODO avoid set_result if tranfser failed since it has been set exception
            write_meta.future_for_task.set_result(sync_status)
            task.status = TaskStatus.TRANSFERRED

        logger.debug(
            f"deliver_kv_to_agent completed: unique_rid={write_meta.unique_rid}, "
            f"slice_id={write_meta.slice_id}, sync_status={sync_status}"
        )

    @nvtx_range("_deliver_aux_to_agent")
    def _deliver_aux_to_agent(self, write_meta: WriteMeta):
        assert write_meta.src_ptrs is not None
        session = self._get_tx_session(write_meta.unique_rid)
        assert session is not None
        assert isinstance(session, TxSession)
        aux_task = session.aux_task
        timer = aux_task._perf_timer
        if timer:
            timer.record_push_end(write_meta.peer_rank)

        skip_send = len(write_meta.src_ptrs) == 0
        sync_status = SyncStatus.SUCCESS
        if not skip_send:
            request, _, _ = Sender._make_agent_request(write_meta, device_id=self._device_id)
            if timer:
                timer.record_transfer_start(write_meta.peer_rank)
            if not self._agent.submit_transfer_requests(request).wait():
                sync_status = SyncStatus.FAILED
                write_meta.future_for_task.set_exception(RuntimeError("Transfer failed"))
                session.mark_error()
            if timer:
                timer.record_transfer_end(write_meta.peer_rank)

        self._get_or_connect_dealer(write_meta.peer_endpoint).send(
            [
                MessageType.AUX_SEND_STATUS,
                str(self._instance_rank).encode("ascii"),
                str(write_meta.unique_rid).encode("ascii"),
                sync_status.value.encode("ascii"),
            ]
        )

        aux_task._transferred_count += 1
        if timer:
            timer.record_task_end(write_meta.peer_rank)
        ri = self._registrar.self_rank_info
        aux_task.print_perf_info(write_meta.peer_rank, ri.instance_name, ri.instance_rank)
        if aux_task._transferred_count == write_meta.expected_transfers:
            aux_task.future.set_result(sync_status)
            aux_task.status = TaskStatus.TRANSFERRED
        elif aux_task._transferred_count > write_meta.expected_transfers:
            aux_task.future.set_exception(
                RuntimeError(
                    f"Session {write_meta.unique_rid} has more than {write_meta.expected_transfers} transfers"
                )
            )
            session.mark_error()

    def _should_send_kv(self, peer_overlap: PeerOverlap, peer_rank_info: RankInfo) -> bool:
        dup_head_factor = peer_overlap.duplicate_head_factor
        if dup_head_factor <= 1:
            return True
        self_ri = self._registrar.self_rank_info
        self_tp_rank_in_dp_group = self_ri.tp_rank % self_ri.tp_size_per_dp_group
        return (peer_rank_info.dp_rank % dup_head_factor) == (
            self_tp_rank_in_dp_group % dup_head_factor
        )

    def _should_send_aux(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated
        self_ri = self._registrar.self_rank_info
        self_tp_rank_in_dp_group = self_ri.tp_rank % self_ri.tp_size_per_dp_group

        should_send_in_tp = False
        if self_ri.tp_size_per_dp_group <= peer_rank_info.tp_size_per_dp_group:
            should_send_in_tp = True
        else:
            ratio = self_ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group
            should_send_in_tp = self_tp_rank_in_dp_group % ratio == 0

        # Compute peer pp_rank's global layer range from layer_num_per_pp
        self_layer_num_per_pp = self_ri.layer_num_per_pp
        peer_layer_num_per_pp = peer_rank_info.layer_num_per_pp
        peer_start_layer = sum(peer_layer_num_per_pp[: peer_rank_info.pp_rank])
        peer_end_layer = peer_start_layer + peer_layer_num_per_pp[peer_rank_info.pp_rank]

        # Find the first self pp_rank whose global layers overlap with peer's pp_rank.
        # All tp/pp ranks have the same aux data, so we only select one self pp_rank
        # to send to the peer; pick the first overlapping one to avoid duplication.
        first_matching_self_pp_rank = None
        self_layer_offset = 0
        for p in range(self_ri.pp_size):
            self_start = self_layer_offset
            self_end = self_start + self_layer_num_per_pp[p]
            if self_start < peer_end_layer and self_end > peer_start_layer:
                first_matching_self_pp_rank = p
                break
            self_layer_offset += self_layer_num_per_pp[p]

        should_send_in_pp = first_matching_self_pp_rank == self_ri.pp_rank
        return should_send_in_tp and should_send_in_pp

    @staticmethod
    def _filter_kv_blocks(src_block_ids, dst_block_ids) -> tuple[list[int], list[int]]:
        # TODO: filter the kv block_ids according to the peer_overlap
        return src_block_ids, dst_block_ids

    @nvtx_range("_build_kv_write_meta")
    def _build_kv_write_meta(self, task: KVSendTask, req_info: RecvReqInfo) -> WriteMeta:
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        if task._perf_timer is not None:
            task._perf_timer.record_prepare_args_start(peer_ri.instance_rank)
        targets = self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank)
        expected_transfers = len(targets.ranks)
        if task._expected_transfers == 0:
            task._expected_transfers = expected_transfers
        if not self._should_send_kv(targets, peer_ri):
            if task._perf_timer is not None:
                task._perf_timer.record_prepare_args_end(peer_ri.instance_rank)
                task._perf_timer.record_transfer_sizes(peer_ri.instance_rank, 0, 0)
            return WriteMeta(
                future_for_task=task._future,
                src_ptrs=[],
                dst_ptrs=[],
                sizes=[],
                expected_transfers=expected_transfers,
                peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
                peer_rank=peer_ri.instance_rank,
                peer_endpoint=peer_ri.self_endpoint,
                unique_rid=task._unique_rid,
                slice_id=task._slice_id,
                is_last_slice=task._slice.is_last_slice,
            )

        dst_device_id = peer_ri.device_id
        dst_block_ids_per_groups = req_info.block_ids_per_layer_groups
        src_block_ids_per_groups = task._slice.block_ids_per_layer_groups

        extractor = self._registrar.self_extractor
        peer_extractor = self._registrar.peer_extractor(
            peer_ri.instance_name, peer_ri.instance_rank
        )

        # Get pool mapping: (self_lg, self_pi) -> (peer_lg, peer_pi)
        pool_mapping = self._registrar.get_pool_mapping(peer_ri)

        # Aggregate fragments from all matching pools
        src_frags: List[int] = []
        dst_frags: List[int] = []
        kv_sizes: List[int] = []

        for (self_lg, self_pi), (peer_lg, peer_pi) in pool_mapping.items():
            src_block_ids = src_block_ids_per_groups[self_lg]
            dst_block_ids = dst_block_ids_per_groups[peer_lg]

            if len(src_block_ids) + 1 == len(dst_block_ids):
                # FIXME: this is a temporary solution, need to be fixed for the draft tokens
                logger.warning(
                    "src_block_num is one less than dst_block_num, maybe it is due to draft tokens,"
                    " remove the last block from dst_block_ids "
                )
                dst_block_ids = dst_block_ids[:-1]
            src_block_ids, dst_block_ids = Sender._filter_kv_blocks(src_block_ids, dst_block_ids)

            src_region = extractor.extract(src_block_ids, layer_group_id=self_lg, pool_idx=self_pi)
            dst_region = peer_extractor.extract(
                dst_block_ids, layer_group_id=peer_lg, pool_idx=peer_pi
            )
            mapper = self._registrar.get_kv_map(peer_ri, (self_lg, self_pi), (peer_lg, peer_pi))
            region_pair = mapper.map(src_region, dst_region)
            region_pairs = region_pair if isinstance(region_pair, list) else [region_pair]
            for rp in region_pairs:
                src_frags.extend(rp.src.memory.ptrs)
                dst_frags.extend(rp.dst.memory.ptrs)
                frag_size = rp.src.memory.bytes_per_region
                kv_sizes.extend([frag_size] * len(rp.src.memory.ptrs))

        if task._perf_timer is not None:
            transfer_total_size = sum(kv_sizes)
            task._perf_timer.record_prepare_args_end(peer_ri.instance_rank)
            task._perf_timer.record_transfer_sizes(
                peer_ri.instance_rank, transfer_total_size, len(dst_frags)
            )

        return WriteMeta(
            future_for_task=task._future,
            src_ptrs=src_frags,
            dst_ptrs=dst_frags,
            sizes=kv_sizes,
            dst_device_id=dst_device_id,
            expected_transfers=expected_transfers,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_rank=peer_ri.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=task._unique_rid,
            slice_id=task._slice_id,
            is_last_slice=task._slice.is_last_slice,
        )

    def _build_aux_write_meta(self, task: AuxSendTask, req_info: RecvReqInfo) -> WriteMeta:
        peer_rank_info = self._registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        )
        if task._perf_timer is not None:
            task._perf_timer.record_prepare_args_start(peer_rank_info.instance_rank)
        expected_transfers = len(
            self._registrar.get_peer_overlap(peer_rank_info, peer_rank_info.dp_rank).ranks
        )
        if not self._should_send_aux(peer_rank_info):
            if task._perf_timer is not None:
                task._perf_timer.record_prepare_args_end(peer_rank_info.instance_rank)
                task._perf_timer.record_transfer_sizes(
                    req_info.instance_name + str(req_info.instance_rank), 0, 0
                )
            return WriteMeta(
                future_for_task=task._future,
                src_ptrs=[],
                dst_ptrs=[],
                sizes=[],
                expected_transfers=expected_transfers,
                peer_name=req_info.instance_name + str(req_info.instance_rank),
                peer_rank=req_info.instance_rank,
                peer_endpoint=self._registrar.get_peer_rank_info(
                    req_info.instance_name, req_info.instance_rank
                ).self_endpoint,
                unique_rid=task._unique_rid,
            )
        peer_aux_meta = self._registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        ).aux_meta

        peer_slot = req_info.aux_slot
        src_aux_meta = self._registrar.self_rank_info.aux_meta

        src_ptrs = [
            ptr + item_size * task._slot
            for ptr, item_size in zip(src_aux_meta.ptrs, src_aux_meta.item_sizes)
        ]
        dst_ptrs = [
            ptr + item_size * peer_slot
            for ptr, item_size in zip(peer_aux_meta.ptrs, peer_aux_meta.item_sizes)
        ]
        size = [item_size for item_size in src_aux_meta.item_sizes]

        if task._perf_timer is not None:
            task._perf_timer.record_prepare_args_end(peer_rank_info.instance_rank)
            task._perf_timer.record_transfer_sizes(req_info.instance_rank, sum(size), len(src_ptrs))
        return WriteMeta(
            future_for_task=task._future,
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            sizes=size,
            expected_transfers=expected_transfers,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_rank=req_info.instance_rank,
            peer_endpoint=self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            ).self_endpoint,
            unique_rid=task._unique_rid,
        )

    def dispatch_task(self, task: KVSendTask | AuxSendTask):
        req_info_dict = self._peer_reqs.get_req_info(task._unique_rid)

        if req_info_dict:
            for info in req_info_dict.values():
                if task._perf_timer is not None:
                    task._perf_timer.record_task_start(info.instance_rank)
                if isinstance(task, KVSendTask):
                    trans_meta = self._build_kv_write_meta(task, info)
                else:
                    trans_meta = self._build_aux_write_meta(task, info)
                if task._perf_timer is not None:
                    task._perf_timer.record_push_start(trans_meta.peer_rank)
                self.submit_task(trans_meta)

    def _start_listener(self):
        def handle_message(messages: list[bytes]):
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.REQUEST_DATA:
                    self._respond_with_kv(send_id, msg)
                case MessageType.REGISTER_RANK_INFO:
                    self._register_peer_rank(send_id, msg)
                case _:
                    raise ValueError(f"Sender received unknown message type: {msg[0]}")

        self._messenger.start_listener(handle_message)

    def _register_peer_rank(self, send_id: bytes, message: list[bytes]):
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
    def _respond_with_kv(self, send_id: bytes, message: list[bytes]):
        # each context send task may respond to multiple req_infos which is from different gen ranks.
        # we support that tx_session send is called before or after getting the req_info.
        #  which means we need to support three cases:
        # 1. the tx_session send is called before getting the all req_infos.
        # 2. the tx_session send is called after getting the all req_infos.
        # 3. the tx_session send is called after getting some req_infos and before getting the other req_infos.
        # response_with_kv and function tx_session send both may trigger the submit task,
        # use session.lock to serialize the submit task in both functions to avoid duplication.
        # use self._sessions_lock to ensure the session will not be inserted between
        # self._save_peer_req_info and get_tx_session
        info: RecvReqInfo = RecvReqInfo.from_bytes(message[1])
        with self._sessions_lock:
            session = self._get_tx_session(info.unique_rid)
            if session is None:
                self._save_peer_req_info(info)
                return
        with session.lock:
            self._save_peer_req_info(info)

            tasks = session.kv_tasks
            if tasks:
                for task in tasks:
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_start(info.instance_rank)
                    trans_meta = self._build_kv_write_meta(task, info)
                    if task._perf_timer is not None:
                        task._perf_timer.record_push_start(trans_meta.peer_rank)
                    self.submit_task(trans_meta)

    def _get_or_connect_dealer(self, endpoint: str):
        if endpoint is None:
            raise ValueError("endpoint is None")
        if endpoint not in self._dealers:
            self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._dealers[endpoint]

    def _save_peer_req_info(self, peer_transfer_req_info: RecvReqInfo):
        req_info = peer_transfer_req_info
        self._peer_reqs.add_req_info(req_info.unique_rid, req_info.instance_rank, req_info)
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        expected_transfers = len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)
        if self._peer_reqs.is_ready(req_info.unique_rid, expected_transfers):
            if req_info.unique_rid in self._tx_sessions:
                session = self._get_tx_session(req_info.unique_rid)
                if not session.peer_ready:
                    session.set_peer_ready()

    def clear_session(self, unique_rid: int):
        """Clear session-related resources from Sender.

        Args:
            unique_rid: The unique request ID of the session to clear
        """
        with self._sessions_lock:
            if unique_rid in self._tx_sessions:
                del self._tx_sessions[unique_rid]
        self._peer_reqs.remove_req_info(unique_rid)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True

        # Stop all worker threads by sending None to each queue
        if hasattr(self, "_send_task_queues"):
            for q in self._send_task_queues:
                q.put(None)
            if hasattr(self, "_worker_threads"):
                for t in self._worker_threads:
                    t.join(timeout=5)
                # Invalidate all loaded remote agents to release fabric/POSIX FD resources
        if hasattr(self, "_loaded_remote_agents"):
            for agent_name in self._loaded_remote_agents:
                try:
                    self._agent.invalidate_remote_agent(agent_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to invalidate remote agent '{agent_name}' during shutdown: {e}"
                    )
            self._loaded_remote_agents.clear()
        self._messenger.stop()

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Exception in Sender.__del__: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class TxSession(TxSessionBase):
    def __init__(self, request_id: int, params: DisaggregatedParams, sender: Sender, aux_slot: int):
        super().__init__(sender, SessionArgsBase(params))
        self.request_id = request_id
        self.aux_slot = aux_slot
        self._peer_ready: bool = False
        self._has_error = False
        self._exception = None
        self._closed = False
        self._kv_tasks = []
        self._aux_task = None
        self._lock = threading.Lock()
        # setup_session registers this session, making it visible to the
        # listener thread.  All instance attributes that the listener may
        # access (_kv_tasks, _lock, etc.) must be initialised BEFORE this call
        # to avoid a race where the listener sees the session before its
        # attributes are ready.
        self._sender.setup_session(self)

    @property
    def peer_ready(self) -> bool:
        return self._peer_ready

    @property
    def status(self) -> SessionStatus:
        if self._has_error or any(t.status == TaskStatus.ERROR for t in self._kv_tasks):
            return SessionStatus.ERROR
        kv_all_transferred = bool(self._kv_tasks) and all(
            t.status == TaskStatus.TRANSFERRED for t in self._kv_tasks
        )
        if kv_all_transferred:
            if self._aux_task is not None and self._aux_task.status == TaskStatus.TRANSFERRED:
                return SessionStatus.FULLY_TRANSFERRED
            return SessionStatus.KV_TRANSFERRED
        if self._kv_tasks and any(t.status == TaskStatus.TRANSFERRING for t in self._kv_tasks):
            return SessionStatus.TRANSFERRING
        return SessionStatus.READY if self._peer_ready else SessionStatus.INIT

    def send(self, slice: KVSlice) -> TaskIdType:
        with self._lock:
            params = self._base_args.params
            slice_id = len(self._kv_tasks)
            task = KVSendTask(slice, params, slice_id)
            self._kv_tasks.append(task)
            self._sender.dispatch_task(task)
            return task.slice_id

    def send_aux(self) -> AuxSendTask:
        with self._lock:
            params = self._base_args.params
            task = AuxSendTask(params, self.aux_slot)
            self._aux_task = task
            self._sender.dispatch_task(task)
            return task

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def kv_tasks(self) -> list:
        return self._kv_tasks

    @property
    def aux_task(self) -> Optional["AuxSendTask"]:
        return self._aux_task

    def get_kv_task_future(self, task_id: TaskIdType) -> concurrent.futures.Future:
        return self._kv_tasks[task_id].future

    def set_peer_ready(self):
        self._peer_ready = True

    def mark_error(self):
        self._has_error = True

    def poll_task(self, id: TaskIdType) -> SessionStatus:
        if self._kv_tasks[id].status == TaskStatus.ERROR:
            return SessionStatus.ERROR
        return self.status

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Clear session from Sender's lookup dict.
        # Do NOT null out _kv_tasks/_aux_task/_sender here:
        # worker threads may still hold a strong reference to this session
        # and need access to these fields to finish in-flight transfers.
        # Resources are freed when the session object is garbage collected.
        if self._sender is not None:
            self._sender.clear_session(self.disagg_request_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class KVRecvTask:
    def __init__(
        self,
        unique_rid: int,
        kv_slice: KVSlice,
        slice_id: int,
        params: DisaggregatedParams,
        aux_slot: int,
    ):
        self._unique_rid = unique_rid
        self._kv_slice = kv_slice
        self._slice_id = slice_id
        self._params = params
        self._status = TaskStatus.INIT
        self._exception = None
        self._future = concurrent.futures.Future()
        self._expected_transfers = 0
        self._aux_slot = aux_slot
        self._perf_timer = PerfTimer() if perf_log_manager.enabled else None

    @property
    def status(self) -> TaskStatus:
        return self._status

    @status.setter
    def status(self, s: TaskStatus):
        self._status = s

    @property
    def future(self) -> concurrent.futures.Future:
        return self._future

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def expected_transfers(self) -> int:
        return self._expected_transfers

    @expected_transfers.setter
    def expected_transfers(self, v: int):
        self._expected_transfers = v

    def print_perf_info(self, peer_rank: int, instance_name: str, instance_rank: int):
        perf_log_manager.log_recv_task_perf(
            self._unique_rid,
            peer_rank,
            instance_name,
            instance_rank,
            self._perf_timer,
        )


class Receiver:
    def __init__(
        self,
        peer_registrar: PeerRegistrar,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._registrar = peer_registrar
        self._device_id = device_id
        self._agent = agent
        self._dealers = {}
        self._sender_ep_instance_map = {}

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._start_listener()
        logger.info(f"Receiver init with endpoint: {self._messenger.endpoint}")

        self._rx_sessions = {}  # unique_rid -> RxSession
        self._closed = False

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._messenger.stop()

    def clear_session(self, unique_rid: int):
        """Clear session-related resources from Receiver.

        Args:
            unique_rid: The unique request ID of the session to clear
        """
        self._rx_sessions.pop(unique_rid, None)

    def setup_session(self, rx_session: RxSessionBase):
        self._rx_sessions[rx_session.disagg_request_id] = weakref.ref(rx_session)

    def _get_rx_session(self, unique_rid: int) -> RxSessionBase:
        session_ref = self._rx_sessions.get(unique_rid)
        if session_ref is None:
            return None
        session = session_ref()
        if session is None:
            logger.warning(f"RxSession {unique_rid} has been garbage collected")
            return None
        return session

    def _build_recv_req_info(self, task: KVRecvTask) -> RecvReqInfo:
        self_ri = self._registrar.self_rank_info
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
            raise ValueError("sender_dp_rank is None")
        peer_infos: RankInfo = self._get_sender_info(params)
        peer_overlap = self._registrar.get_peer_overlap(peer_infos, sender_dp_rank)
        task._expected_transfers = len(peer_overlap.ranks)
        session = self._get_rx_session(task._unique_rid)
        session._kv_tasks[task._slice_id].status = TaskStatus.TRANSFERRING
        for rank in peer_overlap.ranks:
            if task._perf_timer is not None:
                task._perf_timer.record_task_start(rank)
            self._request_sender_data(peer_infos.sender_endpoints[rank], receiver_req)
        return

    def _need_register_peer_in_first_request(self, params: DisaggregatedParams) -> bool:
        return params.ctx_info_endpoint not in self._sender_ep_instance_map

    def _get_or_connect_dealer(self, endpoint: str):
        if endpoint is None:
            raise ValueError("endpoint is None")
        if endpoint not in self._dealers:
            self._dealers[endpoint] = ZMQMessenger(mode="DEALER", endpoint=endpoint)
        return self._dealers[endpoint]

    def _get_sender_info(self, params: DisaggregatedParams) -> RankInfo:
        if self._need_register_peer_in_first_request(params):
            logger.info(
                f"Registering peer in first request to endpoint '{params.ctx_info_endpoint}'"
            )
            messenger = ZMQMessenger(mode="DEALER", endpoint=params.ctx_info_endpoint)
            messenger.send([MessageType.REQUEST_INSTANCE_INFO])
            message = messenger.receive()
            sender_info = RankInfo.from_bytes(message[0])
            messenger.stop()

            for endpoint in sender_info.sender_endpoints:
                dealer = self._get_or_connect_dealer(endpoint)
                rank_info = self._registrar.self_rank_info
                dealer.send([MessageType.REGISTER_RANK_INFO, rank_info.to_bytes()])

            self._sender_ep_instance_map[params.ctx_info_endpoint] = sender_info
            return sender_info

        else:
            return self._sender_ep_instance_map[params.ctx_info_endpoint]

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.TASK_STATUS:
                    self._process_kv_task_status(send_id, msg)
                case MessageType.AUX_SEND_STATUS:
                    self._process_aux_state(send_id, msg)
                case _:
                    raise ValueError(f"Receiver received unknown message type: {msg[0]}")

        self._messenger.start_listener(handle_message)

    def _process_kv_task_status(self, send_id: bytes, message: list[bytes]):
        msg_type, peer_rank, unique_rid, _, is_last_slice_str, status = decode_message(message)
        peer_rank = int(peer_rank)
        unique_rid = int(unique_rid)
        assert msg_type.encode("ascii") == MessageType.TASK_STATUS
        session = self._get_rx_session(unique_rid)
        session.process_kv_task_status(peer_rank, is_last_slice_str == "True", SyncStatus(status))

    def _process_aux_state(self, send_id: bytes, message: list[bytes]):
        msg_type, peer_rank, unique_rid, status = decode_message(message)
        peer_rank = int(peer_rank)
        unique_rid = int(unique_rid)
        session = self._get_rx_session(unique_rid)
        session.process_aux_state(peer_rank, SyncStatus(status))

    def _request_sender_data(self, endpoint: str, receiver_info: RecvReqInfo):
        logger.debug(
            f"Sending data request to endpoint '{endpoint}' with request info: {receiver_info}"
        )
        messenger = self._get_or_connect_dealer(endpoint)
        messenger.send([MessageType.REQUEST_DATA, receiver_info.to_bytes()])

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class RxSession(RxSessionBase):
    def __init__(
        self,
        request_id: int,
        params: DisaggregatedParams,
        receiver: Receiver,
        aux_slot: int,
    ):
        super().__init__(receiver, SessionArgsBase(params))
        self.request_id = request_id
        self.aux_slot = aux_slot
        self._has_error = False
        self._exception = None
        self._closed = False
        self._kv_tasks = []
        self._last_slice_counts = 0
        self._aux_counts = 0
        self._aux_done = False
        self._receiver.setup_session(self)

    @property
    def status(self) -> SessionStatus:
        if self._has_error or (self._kv_tasks and self._kv_tasks[0].status == TaskStatus.ERROR):
            return SessionStatus.ERROR
        if self._kv_tasks:
            task_status = self._kv_tasks[0].status
            if task_status == TaskStatus.TRANSFERRED and self._aux_done:
                return SessionStatus.FULLY_TRANSFERRED
            if task_status == TaskStatus.TRANSFERRED:
                return SessionStatus.KV_TRANSFERRED
            if task_status == TaskStatus.TRANSFERRING:
                return SessionStatus.TRANSFERRING
        return SessionStatus.INIT

    def receive(self, slice: KVSlice) -> TaskIdType:
        assert len(self._kv_tasks) == 0, "RxSession only supports a single KV slice"
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
        return task.slice_id

    def process_kv_task_status(self, peer_rank: int, is_last_slice: bool, status: SyncStatus):
        task = self._kv_tasks[0]  # receive task slice only support slice 0
        if status == SyncStatus.SUCCESS:
            if is_last_slice:
                self._last_slice_counts += 1
                if self._last_slice_counts == task.expected_transfers:
                    task.future.set_result(SyncStatus.SUCCESS)
                    task.status = TaskStatus.TRANSFERRED

                    logger.debug("Task state handled successfully")
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_end(peer_rank)
                    ri = self._receiver._registrar.self_rank_info
                    task.print_perf_info(peer_rank, ri.instance_name, ri.instance_rank)
        elif status == SyncStatus.FAILED:
            task.future.set_exception(RuntimeError(f"Task state: {status.value}"))
            task.status = TaskStatus.ERROR
        else:
            raise ValueError(f"Session received unknown task status: {status}")

    def process_aux_state(self, peer_rank: int, status: SyncStatus):
        task = self._kv_tasks[0]  # receive task slice only support slice 0
        if status == SyncStatus.SUCCESS:
            self._aux_counts += 1

            if self._aux_counts == task.expected_transfers:
                self._aux_done = True
            elif self._aux_counts > task.expected_transfers:
                logger.error(
                    f"Session {self.request_id} has more than {task.expected_transfers} transfers"
                )
                self._has_error = True
        elif status == SyncStatus.FAILED:
            self._has_error = True
        else:
            raise ValueError(
                f"Session {self.request_id} received unknown aux send status: {status}"
            )

    def get_kv_task_future(self, task_id: TaskIdType) -> concurrent.futures.Future:
        return self._kv_tasks[task_id].future

    def poll_task(self, id: TaskIdType) -> SessionStatus:
        if self._kv_tasks[id].status == TaskStatus.ERROR:
            return SessionStatus.ERROR
        return self.status

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Clear session from Receiver's lookup dict.
        # Do NOT null out _kv_tasks/_receiver or reset counters here:
        # the listener thread may still hold a strong reference to this session
        # and need access to these fields to finish processing in-flight status messages.
        # Resources are freed when the session object is garbage collected.
        if self._receiver is not None:
            self._receiver.clear_session(self.disagg_request_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RankInfoServer:
    def __init__(self, rank_info: RankInfo, addr: Optional[str] = None, port: Optional[int] = None):
        self._rank_info = rank_info
        if addr is None and port is None:
            endpoint = f"tcp://{get_local_ip()}:*"
        else:
            endpoint = f"tcp://{addr}:{port}"
        self._messenger = ZMQMessenger(mode="ROUTER", endpoint=endpoint)
        self._start_listener()
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._messenger.endpoint

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
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
                    self._process_request_rank_info(send_id, msg)
                case _:
                    raise ValueError(
                        f"Instance info server received unknown message type: {msg[0]}"
                    )

        self._messenger.start_listener(handle_message)

    def _process_request_rank_info(self, send_id: bytes, _message: list[bytes]):
        self._messenger.send([send_id, self._rank_info.to_bytes()])

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def _deregister_registered_memory(transfer_agent, registered_memorys):
    try:
        if transfer_agent is None or not registered_memorys:
            return
        while registered_memorys:
            register_memory = registered_memorys[0]
            try:
                logger.info(f" transfer worker deregister memory {register_memory} ")
                transfer_agent.deregister_memory(register_memory)
            except Exception:
                logger.error("deregister memory failed in finalizer")
            registered_memorys.pop(0)
    except Exception:
        logger.error("unexpected error in _deregister_registered_memory finalizer")


class TransferWorker:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        mapping: Mapping,
        device_id: int,
        instance_name: str,
        aux_buffer: Optional[AuxBuffer] = None,
    ):
        self._mapping = mapping

        self._rank_info: RankInfo = None
        self._kv_cache_manager = kv_cache_manager
        self._aux_buffer = aux_buffer
        self._device_id = device_id
        self._finalizer = None

        self.init_rank_info(instance_name)
        is_leader = self._mapping.rank == 0
        if is_leader:
            self._rank_info_server = RankInfoServer(self._rank_info)
        else:
            self._rank_info_server = None
        self._kv_extractor = KVRegionExtractorV1(self._kv_cache_manager)
        self._peer_registrar = PeerRegistrar(self._rank_info, self._kv_extractor)

        # NixlTransferAgent configuration from environment variables
        # num_threads: number of dedicated threads for large batch transfers
        # split_batch_size: batch size threshold to use dedicated threads (default 1024)
        nixl_num_threads = int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "8"))
        nixl_agent_kwargs = {}
        if "TRTLLM_NIXL_SPLIT_BATCH_SIZE" in os.environ:
            nixl_agent_kwargs["split_batch_size"] = int(os.environ["TRTLLM_NIXL_SPLIT_BATCH_SIZE"])

        self._agent = NixlTransferAgent(
            self._rank_info.instance_name + str(self._rank_info.instance_rank),
            True,
            num_threads=nixl_num_threads,
            **nixl_agent_kwargs,
        )
        self._registered_mem = []
        self._register_kv_cache()
        if self._aux_buffer is not None:
            self._register_aux_buffer()

        self._sender = Sender(self._peer_registrar, device_id, self._agent)
        self._receiver = Receiver(self._peer_registrar, device_id, self._agent)
        self._rank_info.transfer_engine_info = bytes(self._agent.get_local_agent_desc())
        # self._rank_info.endpoint = self._sender.endpoint
        self._rank_info.self_endpoint = self._receiver.endpoint

        reg_snapshot = list(self._registered_mem) if self._registered_mem is not None else []
        self._finalizer = weakref.finalize(
            self, _deregister_registered_memory, self._agent, reg_snapshot
        )

    def populate_instance_and_rank_info(self, endpoints: list[str], layer_num_per_pp: list[int]):
        self._rank_info.sender_endpoints = endpoints
        self._rank_info.layer_num_per_pp = layer_num_per_pp
        self._rank_info.layer_num_per_pp = layer_num_per_pp

    def create_tx_session(self, request: LlmRequest) -> TxSession:
        """
        Create a txSession for the request.
        """
        if self._aux_buffer is not None:
            aux_slot = self._aux_buffer.alloc_slot()
        else:
            aux_slot = None
        return TxSession(
            request_id=request.py_request_id,
            params=request.py_disaggregated_params,
            sender=self._sender,
            aux_slot=aux_slot,
        )

    def create_rx_session(self, request: LlmRequest) -> RxSession:
        """
        Create a rxSession for the request.
        """
        if self._aux_buffer is not None:
            aux_slot = self._aux_buffer.alloc_slot()
        else:
            aux_slot = None
        return RxSession(
            request_id=request.py_request_id,
            params=request.py_disaggregated_params,
            receiver=self._receiver,
            aux_slot=aux_slot,
        )

    def clear_session(self, session: TxSession | RxSession):
        aux_slot = session.aux_slot
        if self._aux_buffer is not None:
            self._aux_buffer.free_slot(aux_slot)

    def init_rank_info(self, instance_name):
        m = self._mapping
        kvm = self._kv_cache_manager
        enable_attention_dp = m.enable_attention_dp

        self._rank_info = RankInfo(
            instance_name=instance_name,
            instance_rank=m.rank,
            tp_size=m.tp_size,
            tp_rank=m.tp_rank,
            pp_size=m.pp_size,
            pp_rank=m.pp_rank,
            dp_size=m.tp_size if enable_attention_dp else m.dp_size,
            dp_rank=m.tp_rank if enable_attention_dp else 0,
            cp_size=m.cp_size,
            cp_rank=m.cp_rank,
            device_id=self._device_id,
            layer_num_per_pp=[len(kvm.pp_layers)],
            sender_endpoints=[],
            server_endpoint="",
            self_endpoint="",
            transfer_engine_info=bytes(),
            attention=AttentionInfo(
                kv_heads_per_rank=kvm.num_kv_heads_per_layer[0],
                tokens_per_block=kvm.tokens_per_block,
                dims_per_head=kvm.head_dim,
                element_bytes=get_size_in_bytes(1, kvm.dtype),
                enable_attention_dp=enable_attention_dp,
                is_mla=kvm.kv_factor == 1,
            ),
            aux_meta=self._aux_buffer.meta if self._aux_buffer is not None else None,
            # Build page table from manager (supports V1 and V2)
            page_table=build_page_table_from_manager(kvm),
        )

    def _register_kv_cache(self):
        # Get pool information from page_table (works for V1 and V2)
        page_table = self._rank_info.page_table
        memory_descs = []

        # Deduplicate pools (different layer_groups may share the same pool)
        unique_pools: dict[tuple[int, int], int] = {}  # (ptr, size) -> counter
        pool_counter = 0

        for lg_idx, lg in enumerate(page_table.layer_groups):
            for pv in lg.pool_views:
                pool = get_physical_pool(page_table, lg_idx, pv.pool_idx)
                pool_key = (pool.base_address, get_pool_bytes(pool))
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1

        for (pool_ptr, pool_size), idx in unique_pools.items():
            memory_desc = (
                pool_ptr,
                pool_size,
                self._device_id,
                f"kv_cache_memory_pool{idx}",
            )
            memory_descs.append(memory_desc)

        if memory_descs:
            reg_memory_desc = RegMemoryDescs("VRAM", memory_descs)
            self._agent.register_memory(reg_memory_desc)
            logger.debug("Registered KV cache memory with transfer agent: %s", memory_descs)
            self._registered_mem.append(reg_memory_desc)

    def _register_aux_buffer(self):
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
        return self._rank_info.page_table

    # pack the aux data to the meta buffer

    def pack_aux(self, tx_session: TxSession, request: LlmRequest):
        self._aux_buffer.fill_slot(tx_session.aux_slot, request)

    def unpack_aux(self, rx_session: RxSession, request: LlmRequest):
        first_gen_tokens, draft_tokens = self._aux_buffer.get_slot_tokens(rx_session.aux_slot)

        # TODO: not first gen ,but add_tokens?
        request.py_first_gen_tokens = first_gen_tokens
        request.py_draft_tokens = draft_tokens
        return request

    def shutdown(self):
        if self._rank_info_server is not None:
            self._rank_info_server.shutdown()
        if self._sender is not None:
            self._sender.shutdown()
        if self._receiver is not None:
            self._receiver.shutdown()
        # Deregister NIXL memory before shutting down components, so that
        # pinned GPU memory is released and can be re-allocated (e.g. when
        # the KV cache manager is recreated after profiling).
        if self._finalizer is not None:
            self._finalizer()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
