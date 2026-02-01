import concurrent
import os
import queue
import threading
import weakref
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

import msgpack

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
from tensorrt_llm._torch.disaggregation.base.kv_transfer import (
    KVSlice,
    RxSessionBase,
    SessionArgsBase,
    SessionState,
    SessionStatus,
    TaskIdType,
    TxSessionBase,
)
from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.kv_mapper import (
    InstanceInfo,
    KVMapperFactory,
    PeerOverlapTargets,
    PeerRegistrar,
    RankInfo,
)
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger, decode_message
from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer, perf_log_manager
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
from tensorrt_llm.disaggregated_params import DisaggregatedParams

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
    block_ids: list[int]
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
    target_ranks: List[int] = None


@dataclass
class WriteMeta:
    future_for_task: concurrent.futures.Future

    expected_transfers: int
    peer_name: str
    peer_rank: int
    src_kv_ptrs: List[int] = None
    dst_kv_ptrs: List[int] = None
    kv_sizes: List[int] = None
    dst_device_id: int = None
    src_aux_ptrs: List[int] = None
    dst_aux_ptrs: List[int] = None
    aux_sizes: List[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    unique_rid: Optional[int] = None
    slice_id: Optional[int] = None
    is_last_slice: Optional[bool] = False
    is_only_aux: Optional[bool] = False


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
    AUX_TRANSFERRED = "AUX_TRANSFERRED"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"


class AuxSendTask:
    def __init__(self, params: DisaggregatedParams, slot: int, mapper: KVMapperFactory):
        self._params = params
        self._unique_rid = params.disagg_request_id
        self._slot = slot
        self._kv_map = mapper
        self._registrar = self._kv_map.peer_registrar
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

    def _create_write_meta(self, req_info: RecvReqInfo) -> WriteMeta:
        peer_rank_info = self._registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        )
        if self._perf_timer is not None:
            self._perf_timer.record_prepare_args_start(peer_rank_info.instance_rank)
        expected_transfers = len(
            self._kv_map.get_peer_overlap_targets(peer_rank_info, peer_rank_info.dp_rank).ranks
        )
        if not self._should_write(peer_rank_info):
            if self._perf_timer is not None:
                self._perf_timer.record_prepare_args_end(peer_rank_info.instance_rank)
                self._perf_timer.record_transfer_sizes(
                    req_info.instance_name + str(req_info.instance_rank), 0, 0
                )
            return WriteMeta(
                future_for_task=self._future,
                src_aux_ptrs=[],
                dst_aux_ptrs=[],
                aux_sizes=[],
                expected_transfers=expected_transfers,
                is_only_aux=True,
                peer_name=req_info.instance_name + str(req_info.instance_rank),
                peer_rank=req_info.instance_rank,
                peer_endpoint=self._registrar.get_peer_rank_info(
                    req_info.instance_name, req_info.instance_rank
                ).self_endpoint,
                unique_rid=self._unique_rid,
            )
        peer_aux_meta = self._registrar.get_peer_rank_info(
            req_info.instance_name, req_info.instance_rank
        ).aux_meta

        peer_slot = req_info.aux_slot

        src_aux_meta = self._registrar.rank_info.aux_meta

        src_ptrs = [
            ptr + item_size * self._slot
            for ptr, item_size in zip(src_aux_meta.ptrs, src_aux_meta.item_sizes)
        ]
        dst_ptrs = [
            ptr + item_size * peer_slot
            for ptr, item_size in zip(peer_aux_meta.ptrs, peer_aux_meta.item_sizes)
        ]
        size = [item_size for item_size in src_aux_meta.item_sizes]

        if self._perf_timer is not None:
            self._perf_timer.record_prepare_args_end(peer_rank_info.instance_rank)
            self._perf_timer.record_transfer_sizes(req_info.instance_rank, sum(size), len(src_ptrs))
        return WriteMeta(
            future_for_task=self._future,
            src_aux_ptrs=src_ptrs,
            dst_aux_ptrs=dst_ptrs,
            aux_sizes=size,
            expected_transfers=expected_transfers,
            is_only_aux=True,
            peer_name=req_info.instance_name + str(req_info.instance_rank),
            peer_rank=req_info.instance_rank,
            peer_endpoint=self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            ).self_endpoint,
            unique_rid=self._unique_rid,
        )

    def _should_write(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated
        self_ri = self._registrar.rank_info
        self_tp_size_per_dp_group = (
            self_ri.tp_size // self_ri.dp_size if self_ri.enable_attention_dp else self_ri.tp_size
        )
        self_tp_rank_in_dp_group = self_ri.tp_rank % self_tp_size_per_dp_group

        peer_tp_size_per_dp_group = (
            peer_rank_info.tp_size // peer_rank_info.dp_size
            if peer_rank_info.enable_attention_dp
            else peer_rank_info.tp_size
        )
        should_send_in_tp = False
        if self_tp_size_per_dp_group <= peer_tp_size_per_dp_group:
            should_send_in_tp = True

        else:
            ratio = self_tp_size_per_dp_group // peer_tp_size_per_dp_group
            should_send_in_tp = self_tp_rank_in_dp_group % ratio == 0

        should_send_in_pp = False
        self_pp_size = self_ri.pp_size
        peer_pp_size = peer_rank_info.pp_size
        # FIXME:  if pp_layer_num // pp_size !=0, how to handle it?
        if self_pp_size <= peer_pp_size:
            should_send_in_pp = True
        else:
            ratio = self_pp_size // peer_pp_size
            should_send_in_pp = self_ri.pp_rank % ratio == 0
        return should_send_in_tp and should_send_in_pp

    def print_perf_info(self, peer_rank: int):
        if not perf_log_manager.enabled:
            return

        transfer_size = self._perf_timer.get_transfer_size(peer_rank)
        avg_segment_size = self._perf_timer.get_average_segment_size(peer_rank)
        entry_count = self._perf_timer.get_transfer_entry_count(peer_rank)
        prepare_args_latency_ms = self._perf_timer.get_prepare_args_latency(peer_rank) * 1000
        queue_latency_ms = self._perf_timer.get_queue_latency(peer_rank) * 1000
        transfer_latency_ms = self._perf_timer.get_transfer_latency(peer_rank) * 1000
        task_latency_ms = self._perf_timer.get_task_latency(peer_rank) * 1000
        throughput_mbs = self._perf_timer.get_transfer_throughput(peer_rank)

        ri = self._registrar.rank_info
        csv_line = (
            f"AuxSendTask,{self._unique_rid},{peer_rank},"
            f"{transfer_size},{avg_segment_size},{entry_count},"
            f"{prepare_args_latency_ms:.3f},{queue_latency_ms:.3f},"
            f"{transfer_latency_ms:.3f},{task_latency_ms:.3f},{throughput_mbs:.2f}"
        )
        info_msg = (
            f"AuxSendTask.print_perf_info: unique_rid={self._unique_rid}, peer_rank={peer_rank}, "
            f"transfer_size={transfer_size} byte, avg_segment_size={avg_segment_size} byte, "
            f"entry_count={entry_count}, prepare_args_latency={prepare_args_latency_ms:.3f} ms, "
            f"queue_latency={queue_latency_ms:.3f} ms, transfer_latency={transfer_latency_ms:.3f} ms, "
            f"task_latency={task_latency_ms:.3f} ms, throughput={throughput_mbs:.2f} MB/s"
        )
        perf_log_manager.log(ri.instance_name, ri.instance_rank, csv_line, info_msg)


class KVSendTask:
    def __init__(
        self,
        kv_slice: KVSlice,
        params: DisaggregatedParams,
        slice_id: int,
        mapper: KVMapperFactory,
    ):
        self._kv_map = mapper
        self._registrar = self._kv_map.peer_registrar
        self._future = concurrent.futures.Future()
        self._first_transfer = False
        self._extraction_count = 0
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

    @nvtx_range("_create_write_meta")
    def _create_write_meta(self, req_info: RecvReqInfo) -> WriteMeta:
        assert self.is_active(), "KVSendTask is not active"
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        if self._perf_timer is not None:
            self._perf_timer.record_prepare_args_start(peer_ri.instance_rank)
        targets = self._kv_map.get_peer_overlap_targets(peer_ri, peer_ri.dp_rank)
        expected_transfers = len(targets.ranks)
        if not self._first_transfer:
            self._first_transfer = True
            self._expected_transfers = expected_transfers
        self._extraction_count = self._extraction_count + 1
        if not self._should_write(targets, peer_ri):
            if self._perf_timer is not None:
                self._perf_timer.record_prepare_args_end(peer_ri.instance_rank)
                self._perf_timer.record_transfer_sizes(peer_ri.instance_rank, 0, 0)
            return WriteMeta(
                future_for_task=self._future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expected_transfers=expected_transfers,
                peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
                peer_rank=peer_ri.instance_rank,
                peer_endpoint=peer_ri.self_endpoint,
                unique_rid=self._unique_rid,
                slice_id=self._slice_id,
                is_last_slice=self._slice.is_last_slice,
            )

        dst_device_id = peer_ri.device_id
        dst_block_ids = req_info.block_ids
        src_block_ids = self._slice.block_ids

        if len(src_block_ids) + 1 == len(dst_block_ids):
            # FIXME: this is a temporary solution, need to be fixed for the draft tokens
            logger.warning(
                "src_block_num is one less than dst_block_num, maybe it is due to draft tokens,"
                " remove the last block from dst_block_ids "
            )
            dst_block_ids = dst_block_ids[:-1]
        src_block_ids, dst_block_ids = self._filter_kv_blocks(src_block_ids, dst_block_ids)

        extractor = self._kv_map.self_extractor
        src_block_ptrs = extractor.block_ptrs(src_block_ids)
        self_block_size = extractor.kv_pool_attrs.block_bytes[0]
        peer_extractor = self._kv_map.peer_extractor(peer_ri.instance_name, peer_ri.instance_rank)
        dst_block_ptrs = peer_extractor.block_ptrs(dst_block_ids)
        dst_block_size = peer_extractor.kv_pool_attrs.block_bytes[0]
        logger.debug(
            f"KVSendTask._create_write_meta: extracted KV block pointers -> "
            f"src_blocks={src_block_ptrs}, src_block_size={self_block_size}, "
            f"dst_blocks={dst_block_ptrs}, dst_block_size={dst_block_size}"
        )
        mapper = self._kv_map.get_kv_map(peer_ri)
        (
            src_frags,
            dst_frags,
            dst_blocks_size,
        ) = mapper(src_block_ptrs, self_block_size, dst_block_ptrs, dst_block_size)

        src_blocks_size = dst_blocks_size
        logger.debug(
            f"KVSendTask._create_write_meta: mapped KV pointers for transfer -> "
            f"src_frags={src_frags}, src_blocks_size={src_blocks_size}, "
            f"dst_frags={dst_frags}, dst_blocks_size={dst_blocks_size}"
        )

        if self._perf_timer is not None:
            self._perf_timer.record_prepare_args_end(peer_ri.instance_rank)
            self._perf_timer.record_transfer_sizes(
                peer_ri.instance_rank, src_blocks_size * len(src_frags), len(dst_frags)
            )
        return WriteMeta(
            future_for_task=self._future,
            src_kv_ptrs=src_frags,
            dst_kv_ptrs=dst_frags,
            kv_sizes=[src_blocks_size] * len(src_frags),
            dst_device_id=dst_device_id,
            expected_transfers=expected_transfers,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_rank=peer_ri.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=self._unique_rid,
            slice_id=self._slice_id,
            is_last_slice=self._slice.is_last_slice,
        )

    def is_active(self) -> bool:
        if self._first_transfer:
            return self._extraction_count < self._expected_transfers
        else:
            return True

    def _should_write(
        self, peer_overlap_targets: PeerOverlapTargets, peer_rank_info: RankInfo
    ) -> bool:
        dup_head_factor = peer_overlap_targets.duplicate_head_factor
        if dup_head_factor <= 1:
            return True
        peer_ri = peer_rank_info
        peer_dp_rank = peer_ri.dp_rank if peer_ri.enable_attention_dp else 0
        self_ri = self._registrar.rank_info
        self_tp_size_per_dp_group = (
            self_ri.tp_size // self_ri.dp_size if self_ri.enable_attention_dp else self_ri.tp_size
        )
        self_tp_rank_in_dp_group = self_ri.tp_rank % self_tp_size_per_dp_group
        return (peer_dp_rank % dup_head_factor) == (self_tp_rank_in_dp_group % dup_head_factor)

    def _filter_kv_blocks(self, src_block_ids, dst_block_ids) -> tuple[list[int], list[int]]:
        # TODO: filter the kv block_ids according to the peer_overlap_targets
        return src_block_ids, dst_block_ids

    def print_perf_info(self, peer_rank: int):
        if not perf_log_manager.enabled:
            return

        transfer_size = self._perf_timer.get_transfer_size(peer_rank)
        avg_segment_size = self._perf_timer.get_average_segment_size(peer_rank)
        entry_count = self._perf_timer.get_transfer_entry_count(peer_rank)
        prepare_args_latency_ms = self._perf_timer.get_prepare_args_latency(peer_rank) * 1000
        queue_latency_ms = self._perf_timer.get_queue_latency(peer_rank) * 1000
        transfer_latency_ms = self._perf_timer.get_transfer_latency(peer_rank) * 1000
        task_latency_ms = self._perf_timer.get_task_latency(peer_rank) * 1000
        throughput_mbs = self._perf_timer.get_transfer_throughput(peer_rank)

        ri = self._registrar.rank_info
        csv_line = (
            f"KVSendTask,{self._unique_rid},{peer_rank},"
            f"{transfer_size},{avg_segment_size},{entry_count},"
            f"{prepare_args_latency_ms:.3f},{queue_latency_ms:.3f},"
            f"{transfer_latency_ms:.3f},{task_latency_ms:.3f},{throughput_mbs:.2f}"
        )
        info_msg = (
            f"KVSendTask.print_perf_info: unique_rid={self._unique_rid}, peer_rank={peer_rank}, "
            f"transfer_size={transfer_size} byte, avg_segment_size={avg_segment_size} byte, "
            f"entry_count={entry_count}, prepare_args_latency={prepare_args_latency_ms:.3f} ms, "
            f"queue_latency={queue_latency_ms:.3f} ms, transfer_latency={transfer_latency_ms:.3f} ms, "
            f"task_latency={task_latency_ms:.3f} ms, throughput={throughput_mbs:.2f} MB/s"
        )
        perf_log_manager.log(ri.instance_name, ri.instance_rank, csv_line, info_msg)


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
        mapper: KVMapperFactory,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._kv_map = mapper
        self._registrar = self._kv_map.peer_registrar
        self._device_id = device_id
        self._agent = agent
        self._peer_reqs = ReqInfoManager()

        self._messenger = ZMQMessenger(mode="ROUTER")
        self._start_listener()

        self._dealers = {}
        self._tx_sessions = {}  # unique_rid -> TxSession
        self._sessions_lock = threading.Lock()  # Protects _tx_sessions access
        logger.info(f" Sender init end with endpoint: {self._messenger.endpoint}")
        self._closed = False
        self._instance_rank = self._kv_map._ri.instance_rank

        # Multi-threaded task queue support
        self._num_threads = KV_TRANSFER_NUM_THREADS
        self._send_task_queues: List[queue.Queue] = [
            queue.Queue() for _ in range(self._num_threads)
        ]
        self._worker_threads: List[threading.Thread] = [
            threading.Thread(target=self._process_task_queue, args=(i,), daemon=True)
            for i in range(self._num_threads)
        ]
        for t in self._worker_threads:
            t.start()
        logger.info(f"Sender started with {self._num_threads} worker thread(s)")

    @property
    def endpoint(self):
        return self._messenger.endpoint

    def setup_session(self, tx_session: TxSessionBase):
        unique_rid = tx_session._base_args.params.disagg_request_id
        with self._sessions_lock:
            self._tx_sessions[unique_rid] = weakref.ref(tx_session)

        req_info = self._peer_reqs.get_first_req_info(unique_rid)

        if req_info:
            peer_ri = self._registrar.get_peer_rank_info(
                req_info.instance_name, req_info.instance_rank
            )
            expected_count = len(
                self._kv_map.get_peer_overlap_targets(peer_ri, req_info.instance_rank).ranks
            )
            if self._peer_reqs.is_ready(unique_rid, expected_count):
                tx_session.state.status = SessionStatus.READY
        return

    def _get_tx_session(self, unique_rid: int) -> TxSessionBase:
        if unique_rid not in self._tx_sessions:
            return None
        session_ref = self._tx_sessions[unique_rid]
        if session_ref is None:
            raise RuntimeError(f"TxSession {unique_rid} reference is None")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"TxSession {unique_rid} has been garbage collected")
        return session

    def submit_task(self, agent_args: WriteMeta):
        # Distribute tasks to threads by unique_rid to ensure same session's tasks
        # are processed by the same thread in order
        thread_idx = agent_args.unique_rid % self._num_threads
        self._send_task_queues[thread_idx].put(agent_args)

    def _process_task_queue(self, thread_idx: int):
        """Process tasks from the queue assigned to this thread.

        Args:
            thread_idx: Index of the worker thread (0 to num_threads-1)
        """
        task_queue = self._send_task_queues[thread_idx]
        while True:
            agent_args = task_queue.get()
            if agent_args is None:
                break
            if agent_args.is_only_aux:
                logger.debug(
                    f"_process_task_queue[{thread_idx}]: delivering aux task to agent: {agent_args}"
                )
                self._deliver_aux_to_agent(agent_args)
            else:
                self._deliver_kv_to_agent(agent_args)

    @staticmethod
    def _make_agent_request(agent_args: WriteMeta, is_aux: bool, device_id: int):
        if is_aux:
            assert agent_args.src_aux_ptrs is not None and agent_args.dst_aux_ptrs is not None
            src_list = [
                (src_ptr, size, 0)
                for src_ptr, size in zip(agent_args.src_aux_ptrs, agent_args.aux_sizes)
            ]
            dst_list = [
                (dst_ptr, size, 0)
                for dst_ptr, size in zip(agent_args.dst_aux_ptrs, agent_args.aux_sizes)
            ]
            src_mem_type = MemoryType.DRAM
            dst_mem_type = MemoryType.DRAM
            peer_name = agent_args.peer_name
        else:
            assert agent_args.src_kv_ptrs is not None and agent_args.dst_kv_ptrs is not None
            src_list = [
                (src_ptr, size, device_id)
                for src_ptr, size in zip(agent_args.src_kv_ptrs, agent_args.kv_sizes)
            ]
            dst_list = [
                (dst_ptr, size, agent_args.dst_device_id)
                for dst_ptr, size in zip(agent_args.dst_kv_ptrs, agent_args.kv_sizes)
            ]
            src_mem_type = MemoryType.VRAM
            dst_mem_type = MemoryType.VRAM
            peer_name = agent_args.peer_name

        # Use C++ MemoryDescs directly with batch constructor (list of tuples)
        src_memory_descs = MemoryDescs(src_mem_type, src_list)
        dst_memory_descs = MemoryDescs(dst_mem_type, dst_list)
        request = TransferRequest(
            TransferOp.WRITE, src_memory_descs, dst_memory_descs, peer_name, None
        )
        return request, src_list, dst_list

    @nvtx_range("_deliver_kv_to_agent")
    def _deliver_kv_to_agent(self, agent_args: WriteMeta):
        assert len(agent_args.src_kv_ptrs) == len(agent_args.dst_kv_ptrs)
        assert len(agent_args.kv_sizes) == len(agent_args.src_kv_ptrs)
        assert agent_args.is_only_aux is False, "agent_args.is_only_aux should be False"

        unique_rid = agent_args.unique_rid
        slice_id = agent_args.slice_id
        peer_endpoint = agent_args.peer_endpoint
        session = self._get_tx_session(unique_rid)
        assert session is not None
        task = session._kv_tasks[slice_id]
        if task._perf_timer is not None:
            task._perf_timer.record_push_end(agent_args.peer_rank)
        assert session.state.status != SessionStatus.ERROR
        session.state.status = SessionStatus.TRANSFERRING
        task.status = TaskStatus.TRANSFERRING
        request, src_kv_list, _ = Sender._make_agent_request(
            agent_args, is_aux=False, device_id=self._device_id
        )

        skip_send = len(src_kv_list) == 0
        logger.debug(f"Submitting transfer request to transfer agent: {request}")
        agent_handler = None
        if task._perf_timer is not None:
            task._perf_timer.record_transfer_start(agent_args.peer_rank)
        if not skip_send:
            agent_handler = self._agent.submit_transfer_requests(request)

        sync_status = "SUCCESS"
        if not skip_send and not agent_handler.wait():
            sync_status = "FAILED"
            agent_args.future_for_task.set_exception(RuntimeError("Transfer failed"))
            task.status = TaskStatus.ERROR
            session.state.status = SessionStatus.ERROR
        if task._perf_timer is not None:
            task._perf_timer.record_transfer_end(agent_args.peer_rank)

        messenger = self._get_or_connect_dealer(peer_endpoint)

        ## TODO: just last slice need to send task state?
        messenger.send(
            [
                MessageType.TASK_STATUS,
                str(self._instance_rank).encode("ascii"),
                str(unique_rid).encode("ascii"),
                str(slice_id).encode("ascii"),
                str(agent_args.is_last_slice).encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )

        curr = task.transferred_count
        task.transferred_count = curr + 1
        if task._perf_timer is not None:
            task._perf_timer.record_task_end(agent_args.peer_rank)
        task.print_perf_info(agent_args.peer_rank)
        if task.transferred_count > agent_args.expected_transfers:
            agent_args.future_for_task.set_exception(
                RuntimeError(
                    f"Session {unique_rid} has more than {agent_args.expected_transfers} transfers"
                )
            )
            # TODO: set exception for the session ?
            session.state.status = SessionStatus.ERROR
        elif task.transferred_count == agent_args.expected_transfers:
            # TODO avoid set_result if tranfser failed since it has been set exception
            agent_args.future_for_task.set_result(sync_status)
            task.status = TaskStatus.TRANSFERRED
            session.state.finished_tasks.append(slice_id)
            if agent_args.is_last_slice:
                session.state.status = SessionStatus.TRANSFERRED

        logger.debug(
            f"deliver_kv_to_agent completed: unique_rid={agent_args.unique_rid}, "
            f"slice_id={slice_id}, sync_status={sync_status}"
        )

    @nvtx_range("submit_send_aux_to_agent")
    def _deliver_aux_to_agent(self, agent_args: WriteMeta):
        # TODO: submit the aux data task to the transfer agent
        assert agent_args.is_only_aux is True
        # assert agent_args.src_aux_ptrs is not None

        session = self._get_tx_session(agent_args.unique_rid)
        assert session is not None
        if session._aux_task._perf_timer is not None:
            session._aux_task._perf_timer.record_push_end(agent_args.peer_rank)
        skip_send = len(agent_args.src_aux_ptrs) == 0
        sync_status = "SUCCESS"
        agent_handler = None
        if not skip_send:
            request, _, _ = Sender._make_agent_request(
                agent_args, is_aux=True, device_id=self._device_id
            )
            agent_handler = self._agent.submit_transfer_requests(request)

            if session._aux_task._perf_timer is not None:
                session._aux_task._perf_timer.record_transfer_start(agent_args.peer_rank)
            if not agent_handler.wait():
                sync_status = "FAILED"
                agent_args.future_for_task.set_exception(RuntimeError("Transfer failed"))
                session.state.status = SessionStatus.ERROR

            if session._aux_task._perf_timer is not None:
                session._aux_task._perf_timer.record_transfer_end(agent_args.peer_rank)
        messenger = self._get_or_connect_dealer(agent_args.peer_endpoint)
        messenger.send(
            [
                MessageType.AUX_SEND_STATUS,
                str(self._instance_rank).encode("ascii"),
                str(agent_args.unique_rid).encode("ascii"),
                sync_status.encode("ascii"),
            ]
        )

        aux_task = session._aux_task
        aux_task._transferred_count += 1
        if aux_task._perf_timer is not None:
            aux_task._perf_timer.record_task_end(agent_args.peer_rank)
        aux_task.print_perf_info(agent_args.peer_rank)
        if aux_task._transferred_count == agent_args.expected_transfers:
            aux_task.future.set_result(sync_status)
            aux_task.status = TaskStatus.AUX_TRANSFERRED
            session.state.status = SessionStatus.AUX_TRANSFERRED
        elif aux_task._transferred_count > agent_args.expected_transfers:
            aux_task.future.set_exception(
                RuntimeError(
                    f"Session {agent_args.unique_rid} has more than {agent_args.expected_transfers} transfers"
                )
            )
            session.state.status = SessionStatus.ERROR

    def dispatch_task(self, task: KVSendTask | AuxSendTask):
        req_info_dict = self._peer_reqs.get_req_info(task._unique_rid)

        if req_info_dict:
            for info in req_info_dict.values():
                if task._perf_timer is not None:
                    task._perf_timer.record_task_start(info.instance_rank)
                trans_meta = task._create_write_meta(info)
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
        with session._lock:
            self._save_peer_req_info(info)

            tasks = session._kv_tasks
            if tasks:
                for task in tasks:
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_start(info.instance_rank)
                    trans_meta = task._create_write_meta(info)
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
        expected_transfers = len(
            self._kv_map.get_peer_overlap_targets(peer_ri, req_info.instance_rank).ranks
        )
        if self._peer_reqs.is_ready(req_info.unique_rid, expected_transfers):
            if req_info.unique_rid in self._tx_sessions:
                session = self._get_tx_session(req_info.unique_rid)
                if session.state.status == SessionStatus.INIT:
                    session.state.status = SessionStatus.READY

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
        self._state = SessionState(status=SessionStatus.INIT, finished_tasks=[])
        self._exception = None
        self._sender.setup_session(self)
        self._closed = False
        self._kv_tasks = []
        self._aux_task = None
        self._lock = threading.Lock()

    @property
    def state(self) -> SessionState:
        return self._state

    @state.setter
    def state(self, s: SessionState):
        self._state = s

    def send(self, slice: KVSlice) -> TaskIdType:
        with self._lock:
            params = self._base_args.params
            slice_id = len(self._kv_tasks)
            task = KVSendTask(slice, params, slice_id, self._sender._kv_map)
            self._kv_tasks.append(task)
            self._sender.dispatch_task(task)
            return task.slice_id

    def send_aux(self) -> AuxSendTask:
        with self._lock:
            params = self._base_args.params
            slot = self.aux_slot
            task = AuxSendTask(params, slot, self._sender._kv_map)

            self._aux_task = task
            self._sender.dispatch_task(task)
            return task

    def poll_task(self, id: TaskIdType) -> SessionStatus:
        return self._kv_tasks[id].state

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Clear session resources from Sender
        if self._sender is not None:
            unique_rid = self._base_args.params.disagg_request_id
            self._sender.clear_session(unique_rid)
        self._kv_tasks = None
        self._aux_task = None
        self._sender = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in TxSession.__del__: {e}")


class KVRecvTask:
    def __init__(
        self,
        unique_rid: int,
        kv_slice: KVSlice,
        slice_id: int,
        params: DisaggregatedParams,
        mapper: KVMapperFactory,
        aux_slot: int,
    ):
        self._unique_rid = unique_rid
        self._kv_slice = kv_slice
        self._slice_id = slice_id
        self._params = params
        self._kv_map = mapper
        self._registrar = self._kv_map.peer_registrar
        self._status = TaskStatus.INIT
        self._exception = None
        self._future = concurrent.futures.Future()
        self._first_transfer = False
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

    def _create_req_info(self) -> RecvReqInfo:
        return RecvReqInfo(
            sender_req_id=self._params.ctx_request_id,
            instance_name=self._registrar.rank_info.instance_name,
            instance_rank=self._registrar.rank_info.instance_rank,
            block_ids=self._kv_slice.block_ids,
            unique_rid=self._unique_rid,
            aux_slot=self._aux_slot,
        )

    def _make_read_meta(self, peer_ii, peer_dp_rank) -> ReadMeta:
        peer_overlap_targets = self._kv_map.get_peer_overlap_targets(peer_ii, peer_dp_rank)
        if not self._first_transfer:
            self._first_transfer = True
            self._expected_transfers = len(peer_overlap_targets.ranks)
        return ReadMeta(
            slice_id=self._slice_id,
            unique_rid=self._unique_rid,
            target_ranks=peer_overlap_targets.ranks,
        )

    def print_perf_info(self, peer_rank: int):
        if not perf_log_manager.enabled:
            return

        task_latency_ms = self._perf_timer.get_task_latency(peer_rank) * 1000

        ri = self._registrar.rank_info
        # CSV: task_type,unique_rid,peer_rank,size,avg_seg,count,prepare_args,queue,transfer,task,throughput
        csv_line = f"KVRecvTask,{self._unique_rid},{peer_rank},,,,,,,{task_latency_ms:.3f},"
        info_msg = (
            f"KVRecvTask.print_perf_info: unique_rid={self._unique_rid}, "
            f"peer_rank={peer_rank}, task_latency={task_latency_ms:.3f} ms"
        )
        perf_log_manager.log(ri.instance_name, ri.instance_rank, csv_line, info_msg)


class Receiver:
    def __init__(
        self,
        mapper: KVMapperFactory,
        device_id: int,
        agent: BaseTransferAgent,
    ):
        self._kv_map = mapper
        self._registrar = self._kv_map.peer_registrar
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
        if unique_rid in self._rx_sessions:
            del self._rx_sessions[unique_rid]

    def setup_session(self, rx_session: RxSessionBase):
        self._rx_sessions[rx_session._base_args.params.disagg_request_id] = weakref.ref(rx_session)

    def _get_rx_session(self, unique_rid: int) -> RxSessionBase:
        if unique_rid not in self._rx_sessions:
            raise RuntimeError(f"RxSession {unique_rid} not found")
        session_ref = self._rx_sessions[unique_rid]
        if session_ref is None:
            raise RuntimeError(f"RxSession {unique_rid} reference is None")
        session = session_ref()
        if session is None:
            raise RuntimeError(f"RxSession {unique_rid} has been garbage collected")
        return session

    def dispatch_task(self, task: KVRecvTask):
        params = task._params
        logger.debug(f"Preparing async data transfer request for disagg_params={params}")
        receiver_req = task._create_req_info()
        sender_dp_rank = params.ctx_dp_rank
        if sender_dp_rank is None:
            raise ValueError("sender_dp_rank is None")
        peer_infos: InstanceInfo = self._get_sender_info(params)
        agent_args = task._make_read_meta(peer_infos, sender_dp_rank)
        session = self._get_rx_session(agent_args.unique_rid)
        session._kv_tasks[agent_args.slice_id].status = TaskStatus.TRANSFERRING
        for rank in agent_args.target_ranks:
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

    def _get_sender_info(self, params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(params):
            logger.info(
                f"Registering peer in first request to endpoint '{params.ctx_info_endpoint}'"
            )
            messenger = ZMQMessenger(mode="DEALER", endpoint=params.ctx_info_endpoint)
            messenger.send([MessageType.REQUEST_INSTANCE_INFO])
            message = messenger.receive()
            sender_info = InstanceInfo.from_bytes(message[0])
            messenger.stop()

            for endpoint in sender_info.sender_endpoints:
                dealer = self._get_or_connect_dealer(endpoint)
                rank_info = self._registrar.rank_info
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
        session.process_kv_task_status(peer_rank, is_last_slice_str == "True", status)

    def _process_aux_state(self, send_id: bytes, message: list[bytes]):
        msg_type, peer_rank, unique_rid, status = decode_message(message)
        peer_rank = int(peer_rank)
        unique_rid = int(unique_rid)
        session = self._get_rx_session(unique_rid)
        session.process_aux_state(peer_rank, status)

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
            logger.error(f"Exception in Receiver.__del__: {e}")

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
        self._state = SessionState(status=SessionStatus.INIT, finished_tasks=[])
        self._exception = None
        self._receiver.setup_session(self)
        self._closed = False
        self._kv_tasks = []
        self._last_slice_counts = 0
        self._aux_counts = 0

    @property
    def state(self) -> SessionState:
        return self._state

    @state.setter
    def state(self, s: SessionState):
        self._state = s

    def receive(self, slice: KVSlice) -> TaskIdType:
        params = self._base_args.params
        slice_id = len(self._kv_tasks)
        task = KVRecvTask(
            params.disagg_request_id,
            slice,
            slice_id,
            params,
            self._receiver._kv_map,
            aux_slot=self.aux_slot,
        )
        self._kv_tasks.append(task)
        self._receiver.dispatch_task(task)
        return task.slice_id

    def process_kv_task_status(self, peer_rank: int, is_last_slice: bool, status: str):
        task = self._kv_tasks[0]  # receive task slice only support slice 0
        if status == "SUCCESS":
            if is_last_slice:
                self._last_slice_counts += 1
                if self._last_slice_counts == task.expected_transfers:
                    task.future.set_result("SUCCESS")
                    task.status = TaskStatus.TRANSFERRED
                    self.state.status = SessionStatus.TRANSFERRED
                    self.state.finished_tasks.append(0)

                    logger.debug("Task state handled successfully")
                    if task._perf_timer is not None:
                        task._perf_timer.record_task_end(peer_rank)
                    task.print_perf_info(peer_rank)
        elif status == "FAILED":
            task.future.set_exception(RuntimeError(f"Task state: {status}"))
            task.status = TaskStatus.ERROR
            self.state.status = SessionStatus.ERROR
        else:
            raise ValueError(f"Session received unknown task status: {status}")

    def process_aux_state(self, peer_rank: int, status: str):
        task = self._kv_tasks[0]  # receive task slice only support slice 0
        if status == "SUCCESS":
            self._aux_counts += 1

            if self._aux_counts == task.expected_transfers:
                task.status = TaskStatus.AUX_TRANSFERRED
                self.state.status = SessionStatus.AUX_TRANSFERRED
            elif self._aux_counts > task.expected_transfers:
                logger.error(
                    f"Session {self.request_id} has more than {task.expected_transfers} transfers"
                )
                self.state.status = SessionStatus.ERROR
        elif status == "FAILED":
            self.state.status = SessionStatus.ERROR
        else:
            raise ValueError(
                f"Session {self.request_id} received unknown aux send status: {status}"
            )

    def poll_task(self, id: TaskIdType) -> SessionStatus:
        return self._kv_tasks[id].state

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Clear session resources from Receiver
        if self._receiver is not None:
            unique_rid = self._base_args.params.disagg_request_id
            self._receiver.clear_session(unique_rid)
        self._receiver = None
        self._kv_tasks = None
        self._last_slice_counts = 0
        self._aux_counts = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in RxSession.__del__: {e}")


class InstanceInfoServer:
    def __init__(self, instance_info: InstanceInfo, addr: str = None, port: int = None):
        self._instance_info = instance_info
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
        logger.debug("InstanceInfoServer.shutdown() called")
        self._messenger.stop()

    def _start_listener(self):
        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            msg = messages[1:]
            match msg[0]:
                case MessageType.TERMINATION:
                    return False
                case MessageType.REQUEST_INSTANCE_INFO:
                    self._process_request_instance_info(send_id, msg)
                case _:
                    raise ValueError(
                        f"Instance info server received unknown message type: {msg[0]}"
                    )

        self._messenger.start_listener(handle_message)

    def _process_request_instance_info(self, send_id: bytes, message: list[bytes]):
        self._messenger.send([send_id, self._instance_info.to_bytes()])

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Exception in InstanceInfoServer.__del__: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def _deregister_registered_memory(transfer_agent, registered_memorys):
    try:
        if transfer_agent is None or not registered_memorys:
            return
        for register_memory in registered_memorys:
            try:
                transfer_agent.deregister_memory(register_memory)
            except Exception:
                logger.error("deregister memory failed in finalizer")
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

        self._instance_info: InstanceInfo = None
        self._rank_info: RankInfo = None
        self._kv_cache_manager = kv_cache_manager
        self._aux_buffer = aux_buffer
        self._device_id = device_id
        self._finalizer = None

        self.init_instance_info(instance_name)
        is_leader = self._mapping.rank == 0
        if is_leader:
            self._instance_info_server = InstanceInfoServer(self._instance_info)
        else:
            self._instance_info_server = None
        self._peer_registrar = PeerRegistrar(self._rank_info, self._instance_info)
        self._peer_kv_map = KVMapperFactory(self._peer_registrar, self._kv_cache_manager)
        self._agent = NixlTransferAgent(
            self._rank_info.instance_name + str(self._rank_info.instance_rank), True, num_threads=8
        )
        self._registered_mem = []
        self._register_kv_cache()
        if self._aux_buffer is not None:
            self._register_aux_buffer()

        self._sender = Sender(self._peer_kv_map, device_id, self._agent)
        self._receiver = Receiver(self._peer_kv_map, device_id, self._agent)
        self._rank_info.transfer_engine_info = bytes(self._agent.get_local_agent_desc())
        # self._rank_info.endpoint = self._sender.endpoint
        self._rank_info.self_endpoint = self._receiver.endpoint

        reg_snapshot = list(self._registered_mem) if self._registered_mem is not None else []
        self._finalizer = weakref.finalize(
            self, _deregister_registered_memory, self._agent, reg_snapshot
        )

    def populate_instance_and_rank_info(self, endpoints: list[str], layer_num_per_pp: list[int]):
        self._instance_info.sender_endpoints = endpoints
        self._instance_info.layer_num_per_pp = layer_num_per_pp
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

    def init_instance_info(self, instance_name):
        rank = self._mapping.rank

        tp_size = self._mapping.tp_size
        pp_size = self._mapping.pp_size
        dp_size = self._mapping.dp_size
        cp_size = self._mapping.cp_size
        tp_rank = self._mapping.tp_rank
        pp_rank = self._mapping.pp_rank
        enable_attention_dp = self._mapping.enable_attention_dp
        dp_rank = 0
        if enable_attention_dp:
            dp_size = self._mapping.tp_size
            dp_rank = tp_rank
        cp_rank = self._mapping.cp_rank
        is_mla = self._kv_cache_manager.kv_factor == 1
        self._kv_cache_manager.kv_factor
        heads_num_per_rank = self._kv_cache_manager.num_kv_heads_per_layer[0]
        tokens_per_block = self._kv_cache_manager.tokens_per_block
        dims_per_head = self._kv_cache_manager.head_dim
        element_bytes = get_size_in_bytes(1, self._kv_cache_manager.dtype)
        layer_num_per_pp = [len(self._kv_cache_manager.pp_layers)]
        sender_endpoints = []
        self._instance_info = InstanceInfo(
            instance_name=instance_name,
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            cp_size=cp_size,
            kv_heads_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_bytes=element_bytes,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            sender_endpoints=sender_endpoints,
        )
        self._rank_info = RankInfo(
            instance_name=instance_name,
            instance_rank=rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device_id=self._device_id,
            kv_heads_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_bytes=element_bytes,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            kv_ptrs=[self._kv_cache_manager.get_unique_primary_pool().data_ptr()],
            aux_ptrs=[],
            server_endpoint="",
            self_endpoint="",
            transfer_engine_info=bytes(),
            aux_meta=self._aux_buffer.meta if self._aux_buffer is not None else None,
        )

    def _register_kv_cache(self):
        memory_pool = self._kv_cache_manager.get_unique_primary_pool()
        memory_desc = (
            memory_pool.data_ptr(),
            memory_pool.numel() * memory_pool.element_size(),
            self._device_id,
            "kv_cache_memory",
        )
        reg_memory_desc = RegMemoryDescs("VRAM", [memory_desc])
        self._agent.register_memory(reg_memory_desc)
        logger.debug(f"Registered KV cache memory with transfer agent: {memory_desc}")
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
        if self._instance_info_server is not None:
            self._instance_info_server.shutdown()
        if self._sender is not None:
            self._sender.shutdown()
        if self._receiver is not None:
            self._receiver.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Exception in TransferWorker.__del__: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
