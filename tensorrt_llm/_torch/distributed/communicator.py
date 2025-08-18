import os
import socket
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, ValuesView

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.utils.cpp_extension
from mpi4py import MPI
from torch.distributed.distributed_c10d import (_object_to_tensor,
                                                _tensor_to_object)

from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_broadcast,
                                 mpi_comm, mpi_isend, mpi_isend_object,
                                 mpi_recv, mpi_recv_object, mpi_send,
                                 mpi_send_object)
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


class MPIDist(Distributed):
    tp_comm: MPI.Comm

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self.create_tp_comm()

    def broadcast(self, obj, root=0):
        return mpi_broadcast(obj, root)

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

    def tp_gather(self, obj):
        return self.tp_comm.gather(obj)

    def tp_broadcast(self, obj, root=0):
        return self.tp_comm.bcast(obj, root)


class TorchDist(Distributed):

    @property
    def rank(self):
        return torch.distributed.get_rank()

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        assert dist.is_initialized()
        # print("Legacy Mapping:", mapping.to_dict())
        self.mapping.dist = self  # Mapping handles device mesh creation
        self.cluster_info = None

        if self.rank == 0:
            print(f"DeviceMesh: {self.mapping.device_mesh}")

        self.setup_local_comm()
        self.default_store = torch.distributed.distributed_c10d._get_default_store(
        )

        # We are using a pybind11 version incompatible with PyTorch,
        # preventing us from passing torch objects to C++.
        # This broker library compiles with pybind11 headers shipped by PyTorch
        # to receive and forward these objects to core library.
        # TODO: align pybind11 version.
        src_path = os.path.dirname(os.path.abspath(__file__))
        pg_util_path = os.path.normpath(f"{src_path}/../../libs")
        pg_broker = torch.utils.cpp_extension.load(
            name="pg_broker",
            sources=[f"{src_path}/pgBroker.cpp"],
            extra_ldflags=[f"-L{pg_util_path}", "-lpg_utils"])
        pg_broker.init_pg(torch.distributed.group.WORLD, self.local_comm)
        pg_broker.init_store(self.default_store)

        # pybind11_abi = f"{torch._C._PYBIND11_COMPILER_TYPE}{torch._C._PYBIND11_STDLIB}{torch._C._PYBIND11_BUILD_ABI}"
        # pg_utils_bindings.init_pg(
        #     torch.distributed.group.WORLD.boxed(), self.local_comm.boxed())
        # pg_utils_bindings.init_store(self.default_store, pybind11_abi)

        print(
            f"TorchDist init - done w/ set_world_and_local_pg, rank {self.rank}",
            flush=True)
        self.tp_group = self.mapping.tp_group

    def _get_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def setup_local_comm(self):
        self._get_cluster_info()

        # node IP -> list of ranks
        ip_to_ranks = {}
        for rank, (node_ip, _) in enumerate(self.cluster_info):
            ip_to_ranks.setdefault(node_ip, []).append(int(rank))

        self.local_comm = None
        for ranks in ip_to_ranks.values():
            # all global ranks from the default process group to participate in the call,
            # even if some ranks are not part of the new process group being created
            pg = dist.new_group(ranks=ranks, backend='cuda:nccl,cpu:gloo')
            if int(self.rank) in ranks:
                print(
                    f"[Rank {self.rank}] Done set local comm. ip_to_ranks: {ip_to_ranks}"
                )
                self.local_comm = pg

    def _get_cluster_info(self):
        if self.cluster_info is not None:
            return self.cluster_info

        node_ip = ray.util.get_node_ip_address()

        # ray might return str
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

        # DEBUG
        print("Cluster info: ", self.cluster_info)
        return self.cluster_info

    @staticmethod
    def log_op(func, enable_log=False):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_log:
                # Skip self arg.
                print(
                    f"{func.__name__} enter: {args[1:]}, {kwargs}, rank: {torch.distributed.get_rank()}"
                )
            ret = func(*args, **kwargs)

            if enable_log:
                print(f"{func.__name__} exit:  {ret}")
            return ret

        return wrapper

    @log_op
    def broadcast(self, obj, root=0):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root)
            return obj
        else:
            obj_list = [obj]
            dist.broadcast_object_list(obj_list, src=root)
            return obj_list[0]

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
        raise NotImplementedError("send is not implemented for TorchDist")
        # blocking send numpy buffer
        tensor = torch.from_numpy(buf)
        dist.send(tensor, dst=dest, tag=tag)

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
        return works

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
    def isend_tensor_list(self,
                          tensor_list: ValuesView[torch.Tensor],
                          dest,
                          tag=0):
        if len(tensor_list) == 0:
            return None
        elif len(tensor_list) == 1:
            return [self.isend_tensor(next(iter(tensor_list)), dest, tag)]
        return [dist.isend(torch.cat(tensor_list), dst=dest, tag=tag)]

    @log_op
    def recv_tensor_list(self,
                         tensor_list: ValuesView[torch.Tensor],
                         src,
                         tag=0):
        if len(tensor_list) == 0:
            return []

        first_tensor = next(iter(tensor_list))
        if len(tensor_list) == 1:
            return [self.recv_tensor(first_tensor, src, tag)]

        # Receive tensors
        recv_tensor = torch.empty_like(torch.cat(
            [t.to('meta') for t in tensor_list]),
                                       device=first_tensor.device)
        dist.recv(recv_tensor, src, tag)
        # Assign to tensor_list
        recv_tensor = recv_tensor.flatten()
        offset = 0
        for t in tensor_list:
            t.copy_(recv_tensor[offset:offset + t.numel()].reshape(t.shape))
            offset += t.numel()
        return tensor_list

    @log_op
    def all_reduce(self,
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

        # (TODO: joyang)  For torch dist, it is natural to use local PP rank instead of global rank.
        self.pg.send([tensor], self._global_to_local_rank(dest), tag=0).wait()

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()

        # (TODO: joyang)  For torch dist, it is natural to use local PP rank instead of global rank.
        self.pg.recv([tensor], self._global_to_local_rank(src), tag=0).wait()


_pp_comm = None


def init_pp_comm(mapping):
    """Initialize PPComm once at startup"""
    global _pp_comm
    if os.environ.get("DISABLE_MPI") == "1":
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
