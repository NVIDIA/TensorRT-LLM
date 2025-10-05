import math
import pickle  # nosec B403
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (_object_to_tensor,
                                                _tensor_to_object)

try:
    from mpi4py import MPI
except Exception:
    MPI = None  # deferred; functions will error if used when ENABLE_MULTI_DEVICE is True

from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_comm,
                                 mpi_disabled, mpi_isend, mpi_isend_object,
                                 mpi_recv, mpi_recv_object, mpi_send,
                                 mpi_send_object, torch_pybind11_abi)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.internal.process_group import init_pg
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm import ray_stub as ray


class Distributed(ABC):

    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size

    @property
    def has_tp(self):
        return self.mapping.has_tp()

    @property
    def has_pp(self):
        return self.mapping.has_pp()

    @property
    def cp_size(self):
        return self.mapping.cp_size

    @property
    def pp_size(self):
        return self.mapping.pp_size

    @property
    def tp_size(self):
        return self.mapping.tp_size

    @property
    def cp_rank(self):
        return self.mapping.cp_rank

    @property
    def tp_rank(self):
        return self.mapping.tp_rank

    @property
    def pp_rank(self):
        return self.mapping.pp_rank

    @property
    def is_last_pp_rank(self):
        return self.mapping.is_last_pp_rank()

    @property
    def is_second_last_pp_rank(self):
        return self.mapping.is_second_last_pp_rank()

    @property
    def is_first_pp_rank(self):
        return self.mapping.is_first_pp_rank()

    @property
    def next_pp_rank(self):
        return self.mapping.next_pp_rank()

    @property
    def prev_pp_rank(self):
        return self.mapping.prev_pp_rank()

    @property
    def has_cp_ulysses(self):
        return self.mapping.has_cp_ulysses()

    @property
    def has_cp_helix(self):
        return self.mapping.has_cp_helix()

    @property
    def cp_config(self):
        return self.mapping.cp_config

    @abstractmethod
    def broadcast(self, obj, root=0):
        pass

    @abstractmethod
    def allgather(self, obj, root=0):
        pass


def safe_broadcast(comm, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
    """
    Safely broadcasts potentially large objects by splitting into fixed-size chunks,
    using raw-byte MPI.Bcast to avoid pickle5's out-of-band buffer allocations.

    Args:
        comm: communicator to broadcast
        obj: Python object to broadcast
        root: Rank of the broadcasting process
        chunk_size: Maximum size of each chunk in bytes (default: 4MB)

    Returns:
        The broadcasted object on all ranks
    """
    if not ENABLE_MULTI_DEVICE:
        return obj
    if ENABLE_MULTI_DEVICE and MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    rank = comm.Get_rank()

    # ---- Serialization phase (root only) ----
    # Header layout: [ok_flag, total_size, num_chunks] as int64
    header = np.zeros(3, dtype=np.int64)
    if rank == root:
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            total_size = len(serialized)
            num_chunks = math.ceil(total_size /
                                   chunk_size) if total_size > 0 else 0
            header[:] = (1, total_size, num_chunks)
        except Exception as e:
            # Signal failure to all ranks, then raise
            header[:] = (0, 0, 0)
            comm.Bcast([header, MPI.INT64_T], root=root)
            raise RuntimeError(f"Serialization failed: {str(e)}") from e
    else:
        serialized = None  # not used on non-root before Bcast

    # ---- Metadata broadcast (Bcast the fixed-size header) ----
    comm.Bcast([header, MPI.INT64_T], root=root)
    ok_flag, total_size, num_chunks = int(header[0]), int(header[1]), int(
        header[2])
    if not ok_flag:
        raise RuntimeError("Root rank failed during serialization")

    # ---- Allocate receive buffer (non-root) or build a view (root) ----
    # We broadcast raw bytes chunk by chunk.
    if rank == root:
        src_view = memoryview(serialized)
        dst_buf = None
        dst_view = None
    else:
        # Pre-allocate a contiguous byte buffer to receive the payload
        dst_buf = bytearray(total_size)
        dst_view = memoryview(dst_buf)
        src_view = None  # not used on non-root

    # ---- Chunked raw-byte broadcast with MPI.Bcast ----
    # Each round sends exactly `cur` bytes of the global payload.
    offset = 0
    for i in range(num_chunks):
        cur = min(chunk_size, total_size - offset)
        if cur <= 0:
            break  # safety guard for zero-size payloads

        if rank == root:
            # Root sends a slice of the source view
            part = src_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)
        else:
            # Non-root receives directly into the destination view
            part = dst_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)

        offset += cur

    # ---- Reconstruction and deserialization ----
    # Validate the received byte count and unpickle.
    if rank == root:
        # Root already has `serialized`
        if len(serialized) != total_size:
            raise RuntimeError(
                f"Data size mismatch at root: expected {total_size}, got {len(serialized)}"
            )
        try:
            return pickle.loads(serialized)  # nosec B301
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e
    else:
        if len(dst_buf) != total_size:
            raise RuntimeError(
                f"Data size mismatch at rank {rank}: expected {total_size}, got {len(dst_buf)}"
            )
        try:
            return pickle.loads(dst_buf)  # nosec B301
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e


def safe_gather(comm, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
    """
    Safely gather potentially large objects by splitting into fixed-size chunks,
    using raw-byte MPI.Gatherv. This variant uses Allgather on lengths so every
    rank can compute sizes/displacements/total locally, removing extra broadcasts.

    Args:
        comm: communicator to gather
        obj: Python object to gather
        root: Rank that receives the gathered objects
        chunk_size: Per-round max bytes each rank contributes (default: 4MB)

    Returns:
        On root: list of deserialized objects (len == comm.size)
        On non-root: None
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if ENABLE_MULTI_DEVICE and MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    rank = comm.Get_rank()
    size = comm.Get_size()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # -- Serialize locally --
    try:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        my_n = np.int64(len(payload))
    except Exception as e:
        # Keep collectives aligned: every rank must call Allgather exactly once
        _ = comm.allgather(int(-1))
        raise RuntimeError(f"Rank {rank} serialization failed: {e}") from e

    # -- Allgather lengths so all ranks know sizes and can compute displacements --
    # We allgather just the int64 length to minimize traffic.
    lengths = np.array(comm.allgather(int(my_n)),
                       dtype=np.int64)  # shape (size,)
    if (lengths < 0).any():
        raise RuntimeError(f"Serialization failed on at least one rank")
    # Every rank computes displacements & total locally and identically:
    displs = np.zeros(size, dtype=np.int64)
    if size > 1:
        displs[1:] = np.cumsum(lengths[:-1])
    total = int(lengths.sum())

    # -- Prepare buffers --
    sendbuf_full = np.frombuffer(payload, dtype=np.uint8, count=len(payload))
    if rank == root:
        recvbuf = np.empty(total,
                           dtype=np.uint8)  # single contiguous receive buffer
    else:
        recvbuf = None

    # -- Chunked Gatherv loop --
    # IMPORTANT: All ranks must execute the same number of Gatherv rounds.
    # Using a deterministic schedule based only on (lengths, chunk_size):
    #   num_rounds = ceil(max(lengths)/chunk_size)
    max_len = int(lengths.max()) if size > 0 else 0
    num_rounds = (max_len + chunk_size - 1) // chunk_size if max_len > 0 else 0

    for r in range(num_rounds):
        # Each rank contributes up to chunk_size bytes from its remaining payload
        # this round. Round-local offset is r * chunk_size.
        round_offs = r * chunk_size
        # Per-rank count this round:
        #   count = max(0, min(chunk, length - round_offs))
        remaining = lengths - round_offs
        remaining = np.maximum(remaining, 0)
        counts64 = np.minimum(remaining, chunk_size).astype(np.int64)

        # Target displacements this round are base displs + round_offs (where count>0)
        round_displs64 = displs + np.minimum(np.maximum(lengths, 0), round_offs)

        # Many MPI impls expect 32-bit ints for counts/displs in Gatherv
        counts32 = counts64.astype(np.int32)
        displs32 = round_displs64.astype(np.int32)

        # Local slice to send this round (may be zero-length)
        send_start = min(round_offs, int(my_n))
        send_len = int(counts32[rank])
        send_part = sendbuf_full[send_start:send_start + send_len]

        if rank == root:
            comm.Gatherv([send_part, MPI.BYTE],
                         [recvbuf, counts32, displs32, MPI.BYTE],
                         root=root)
        else:
            comm.Gatherv([send_part, MPI.BYTE], None, root=root)

    # Note: ranks with zero data (my_n == 0) still participate in every Gatherv
    # round with count=0. This is required to keep the collectives matched.

    # -- Reconstruct on root --
    if rank == root:
        out = []
        for i in range(size):
            sz = int(lengths[i])
            if sz == 0:
                # Deserialize a canonical empty/None. Adjust to your needs.
                out.append(None)  # None
                continue
            start = int(displs[i])
            blob = recvbuf[start:start + sz].tobytes()
            try:
                out.append(pickle.loads(blob))  # nosec B301
            except Exception as e:
                raise RuntimeError(
                    f"Deserialization failed for rank {i}: {e}") from e
        return out

    return None


class MPIDist(Distributed):
    tp_comm: MPI.Comm

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self.create_tp_comm()
        self.create_pp_comm()
        self.create_cp_comm()

    def broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = mpi_comm()
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    def allgather(self, obj):
        return mpi_allgather(obj)

    def barrier(self):
        mpi_barrier()

    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        return mpi_isend(buf, dest, tag)

    def send(self, buf: np.ndarray, dest, tag=0):
        # blocking send numpy buffer
        mpi_send(buf, dest, tag)

    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        return mpi_recv(buf, src, tag)

    def send_object(self, obj, dest, tag=0):
        mpi_send_object(obj, dest, tag)

    def isend_object(self, obj, dest, tag=0):
        return mpi_isend_object(obj, dest, tag)

    def recv_object(self, src, tag=0):
        return mpi_recv_object(src, tag)

    def create_tp_comm(self):
        new_group = mpi_comm().group.Incl(self.mapping.tp_group)
        self.tp_comm = mpi_comm().Create_group(new_group)

    def create_pp_comm(self):
        new_group = mpi_comm().group.Incl(self.mapping.pp_group)
        self.pp_comm = mpi_comm().Create_group(new_group)

    def create_cp_comm(self):
        new_group = mpi_comm().group.Incl(self.mapping.cp_group)
        self.cp_comm = mpi_comm().Create_group(new_group)

    def cp_allgather(self, obj):
        return self.cp_comm.allgather(obj)

    def tp_allgather(self, obj):
        return self.tp_comm.allgather(obj)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_gather(comm, obj, root=root, chunk_size=chunk_size)

    def tp_broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    def pp_allgather(self, obj):
        return self.pp_comm.allgather(obj)

    def pp_gather(self, obj):
        return self.pp_comm.gather(obj)

    def pp_broadcast(self, obj, root=0):
        return self.pp_comm.bcast(obj, root)


class MultiHandleWrapper:
    """
    Wrapper that encapsulates multiple handles and provides a single wait() interface
    to unify the API between MPIDist and TorchDist.
    """

    def __init__(self, handles):
        self.handles = handles if isinstance(handles, list) else [handles]

    def wait(self):
        for handle in self.handles:
            try:
                handle.wait()
            except Exception as e:
                raise RuntimeError(f"Asynchronous operation failed: {e}") from e


class TorchDist(Distributed):

    @property
    def rank(self):
        return torch.distributed.get_rank()

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        assert dist.is_initialized(
        ), "torch.distributed should be initialized before TorchDist"

        self.cluster_info = None

        from tensorrt_llm._utils import set_torch_comm
        set_torch_comm(self)  # Set as global instance
        mapping.build_mesh()

        self.setup_local_comm()
        self.default_store = torch.distributed.distributed_c10d._get_default_store(
        )

        init_pg(torch.distributed.group.WORLD, self.local_comm,
                torch_pybind11_abi())

    def setup_local_comm(self):
        self._get_cluster_info()

        # node IP -> list of ranks
        ip_to_ranks = {}
        for rank, (node_ip, _) in enumerate(self.cluster_info):
            ip_to_ranks.setdefault(node_ip, []).append(int(rank))

        self.local_comm = None
        for ranks in ip_to_ranks.values():
            # All global ranks from the default process group to participate in the call,
            # even if some ranks are not part of the new process group being created
            pg = dist.new_group(ranks=ranks, backend='cuda:nccl,cpu:gloo')
            if int(self.rank) in ranks:
                logger.debug(
                    f"[Rank {self.rank}] Done setting local comm. ip_to_ranks: {ip_to_ranks}"
                )
                self.local_comm = pg

    def _get_cluster_info(self):
        if self.cluster_info is not None:
            return self.cluster_info

        if ray.is_initialized():
            node_ip = ray.util.get_node_ip_address()
        else:
            raise RuntimeError("Ray is not initialized")

        gpu_index = [int(id) for id in ray.get_gpu_ids()]

        assert len(gpu_index) == 1

        # Gather node ip
        node_list = [None] * torch.distributed.get_world_size()

        torch.distributed.all_gather_object(node_list, node_ip)

        # Gather gpu index
        gpu_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gpu_list, gpu_index[0])

        # Gather rank
        rank_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_list,
                                            torch.distributed.get_rank())

        rank_info_list = [None] * torch.distributed.get_world_size()
        for i in range(len(rank_list)):
            rank_info_list[rank_list[i]] = (node_list[i], gpu_list[i])

        self.cluster_info = rank_info_list

        logger.debug(f"Cluster info: {self.cluster_info}")
        return self.cluster_info

    @staticmethod
    def log_op(func, enable_log=False):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_log:
                logger.debug(
                    f"{func.__name__} enter: {args[1:]}, {kwargs}, rank: {torch.distributed.get_rank()}"
                )
            ret = func(*args, **kwargs)

            if enable_log:
                logger.debug(f"{func.__name__} exit: {ret}")
            return ret

        return wrapper

    @log_op
    def broadcast(self, obj, root=0):
        assert not (self.mapping.has_cp_ulysses() and self.mapping.has_tp()
                    ), 'Unsupported mix of Ulysses CP and TP.'

        if mpi_disabled():
            if isinstance(obj, torch.Tensor):
                dist.broadcast(obj, src=root)
                return obj
            else:
                obj_list = [obj]
                dist.broadcast_object_list(obj_list, src=root)
                return obj_list[0]

        if self.mapping.has_cp_ulysses():
            self.broadcast_cp(obj, root)
        elif self.mapping.has_tp():
            self.broadcast_tp(obj, root)

    @log_op
    def allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj) for _ in range(self.world_size)
            ]
            dist.all_gather(output_list, obj)
            return output_list
        else:
            obj_list = [None] * self.world_size
            dist.all_gather_object(obj_list, obj)
            return obj_list

    @log_op
    def barrier(self):
        dist.barrier()

    @log_op
    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        tensor = torch.from_numpy(buf)
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def send(self, buf: np.ndarray, dest, tag=0):
        raise NotImplementedError(
            "blocking send is not implemented for TorchDist")

    @log_op
    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        tensor = torch.empty_like(torch.from_numpy(buf))
        dist.recv(tensor, src=src, tag=tag)
        return tensor.numpy()

    @log_op
    def isend_tensor(self, tensor: torch.Tensor, dest, tag=0):
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def recv_tensor(self, tensor: torch.Tensor, src, tag=0):
        dist.recv(tensor, src=src, tag=tag)
        return tensor

    @log_op
    def recv_object(self, src, tag=0):
        size_tensor = torch.tensor([0], dtype=torch.int32)
        torch.distributed.recv(size_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        bytes_size = size_tensor.item()
        recv_tensor = torch.empty(bytes_size, dtype=torch.uint8)
        torch.distributed.recv(recv_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        return _tensor_to_object(recv_tensor, bytes_size,
                                 torch.distributed.group.WORLD)

    @log_op
    def send_object(self, obj, dest, tag=0):
        raise NotImplementedError(
            "send_object is not implemented for TorchDist")

    @log_op
    def isend_object(self, obj, dest, tag=0):
        input_tensor, local_size = _object_to_tensor(
            obj, torch.device("cpu"), torch.distributed.group.WORLD)

        # Send object size
        works = []
        works.append(
            torch.distributed.isend(torch.tensor([local_size],
                                                 dtype=torch.int32),
                                    dst=dest,
                                    tag=tag))
        works.append(torch.distributed.isend(input_tensor, dst=dest, tag=tag))
        return MultiHandleWrapper(works)

    @log_op
    def recv_object_from_isend(self, src, tag):
        size_tensor = torch.tensor([0], dtype=torch.int32)
        torch.distributed.recv(size_tensor, src=src, tag=tag)
        bytes_size = size_tensor.item()
        recv_tensor = torch.empty(bytes_size, dtype=torch.uint8)
        torch.distributed.recv(recv_tensor, src=src, tag=tag)
        return _tensor_to_object(recv_tensor, bytes_size,
                                 torch.distributed.group.WORLD)

    @log_op
    def allreduce(self,
                  obj: int | float | torch.Tensor,
                  op=torch.distributed.ReduceOp.SUM):
        is_base_type = isinstance(obj, int) or isinstance(obj, float)
        if is_base_type:
            obj = torch.tensor(obj)

        dist.all_reduce(obj, op=op)

        if is_base_type:
            obj = obj.item()

        return obj

    @log_op
    def tp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.tp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.tp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.tp_group_pg)
            return output_list

    @log_op
    def tp_gather(self, obj, dst=0):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.tp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.tp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.tp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.tp_group_pg)
            return output_list

    @log_op
    def tp_broadcast(self, obj, root=0):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.tp_group_pg)
            return obj
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.tp_group_pg,
                device=torch.device("cpu"))
            return ret[0]

    @log_op
    def pp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.pp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.pp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_gather(self, obj, dst=0):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.pp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.pp_group_pg)
            return output_list
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.pp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_broadcast(self, obj, root=0):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.pp_group_pg)
            return obj
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.pp_group_pg,
                device=torch.device("cpu"))
            return ret[0]


# TODO: rename to PPCommNCCL
class PPComm:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
            self.mapping.world_size,
            self.mapping.rank,
        )

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()
        self.nccl_comm.send(tensor, dest)

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()
        self.nccl_comm.recv(tensor, src)


class PPCommTorch:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.pg = self.mapping.pp_group_pg
        self.pg_group = self.mapping.pp_group

    def _global_to_local_rank(self, global_rank: int):
        assert global_rank in self.pg_group
        return self.pg_group.index(global_rank)

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()

        self.pg.send([tensor], self._global_to_local_rank(dest), tag=0).wait()

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()

        self.pg.recv([tensor], self._global_to_local_rank(src), tag=0).wait()


_pp_comm = None


def init_pp_comm(mapping):
    """Initialize PPComm once at startup"""
    global _pp_comm
    if mpi_disabled():
        _pp_comm = PPCommTorch(mapping)
    else:
        _pp_comm = PPComm(mapping)


@TorchDist.log_op
def pp_recv(tensor):
    """Receive tensors from previous pp rank."""
    _pp_comm.recv(tensor)


@TorchDist.log_op
def pp_send(tensor):
    """Send tensors to next pp rank."""
    _pp_comm.send(tensor)
