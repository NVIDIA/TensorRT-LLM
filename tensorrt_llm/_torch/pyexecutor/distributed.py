import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

from ..._utils import (mpi_allgather, mpi_barrier, mpi_broadcast, mpi_comm,
                       mpi_isend, mpi_recv)
from ...mapping import Mapping


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

    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        return mpi_recv(buf, src, tag)

    def isend_tensor(self, tensor: torch.Tensor, dest, tag=0):
        return self.isend(tensor.numpy(), dest, tag)

    def recv_tensor(self, tensor: torch.Tensor, src, tag=0):
        return self.recv(tensor.numpy(), src, tag)

    def create_tp_comm(self):
        new_group = mpi_comm().group.Incl(self.mapping.tp_group)
        self.tp_comm = mpi_comm().Create_group(new_group)

    def tp_allgather(self, obj):
        return self.tp_comm.allgather(obj)

    def tp_broadcast(self, obj, root=0):
        return self.tp_comm.bcast(obj, root)


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
