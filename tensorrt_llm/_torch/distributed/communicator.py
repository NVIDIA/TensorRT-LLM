import math
import os
import pickle  # nosec B403
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_comm,
                                 mpi_isend, mpi_isend_object, mpi_recv,
                                 mpi_recv_object, mpi_send, mpi_send_object)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.mapping import Mapping


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
    def has_cp(self):
        return self.mapping.has_cp()

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
        _ = comm.allgather((False, ))
        raise RuntimeError(f"Rank {rank} serialization failed: {e}") from e

    # -- Allgather lengths so all ranks know sizes and can compute displacements --
    # We allgather just the int64 length to minimize traffic.
    lengths = np.array(comm.allgather(int(my_n)),
                       dtype=np.int64)  # shape (size,)
    # Optional sanity: if someone signaled failure above, handle here (not used).
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
                out.append(pickle.loads(blob))
            except Exception as e:
                raise RuntimeError(
                    f"Deserialization failed for rank {i}: {e}") from e
        return out

    return None


class MPIDist(Distributed):

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self.create_tp_comm()

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

    def tp_allgather(self, obj):
        return self.tp_comm.allgather(obj)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_gather(comm, obj, root=root, chunk_size=chunk_size)

    def tp_broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.tp_comm
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)


class TorchDist(Distributed):

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        if not dist.is_initialized():
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            # TODO: fix the constant default port
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=mapping.world_size,
                                    rank=mapping.rank)
        self.device_tp_group = dist.new_group(mapping.tp_group, backend="nccl")
        self.cpu_tp_group = dist.new_group(mapping.tp_group, backend="gloo")
        self.device_cp_group = dist.new_group(mapping.cp_group, backend="nccl")
        self.cpu_cp_group = dist.new_group(mapping.cp_group, backend="gloo")

    def broadcast_tp(self, obj, root=0):
        if root not in self.mapping.tp_group:
            return obj
        elif self.rank == root:
            torch.distributed.broadcast_object_list([obj],
                                                    src=root,
                                                    group=self.cpu_tp_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=root,
                                                    group=self.cpu_tp_group)
            return recv[0]

    def broadcast_cp(self, obj, root=0):
        if root not in self.mapping.cp_group:
            return obj
        elif self.rank == root:
            torch.distributed.broadcast_object_list([obj],
                                                    src=root,
                                                    group=self.cpu_cp_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=root,
                                                    group=self.cpu_cp_group)
            return recv[0]

    def broadcast(self, obj, root=0):
        assert not (self.mapping.has_cp()
                    and self.mapping.has_tp()), 'unsupport mix cp and tp now'
        if self.mapping.has_cp():
            self.broadcast_cp(obj, root)
        elif self.mapping.has_tp():
            self.broadcast_tp(obj, root)
        else:
            pass


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


_pp_comm = None


def init_pp_comm(mapping):
    """Initialize PPComm once at startup"""
    global _pp_comm
    _pp_comm = PPComm(mapping)


def pp_recv(tensor):
    """Receive tensors from previous pp rank."""
    _pp_comm.recv(tensor)


def pp_send(tensor):
    """Send tensors to next pp rank."""
    _pp_comm.send(tensor)
