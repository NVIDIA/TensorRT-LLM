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
import os
import threading

import torch
from mpi4py import MPI
from mpi4py.util import pkl5

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

# mpi4py only exports MPI_COMM_TYPE_SHARED, so we define OMPI_COMM_TYPE_HOST here
OMPI_COMM_TYPE_HOST = 9

comm = pkl5.Intracomm(MPI.COMM_WORLD)


def set_mpi_comm(new_comm):
    global comm
    comm = new_comm


thread_local_comm = threading.local()


def set_thread_local_mpi_comm(new_comm):
    thread_local_comm.value = new_comm


def mpi_comm():
    if hasattr(thread_local_comm,
               "value") and thread_local_comm.value is not None:
        return thread_local_comm.value
    return comm


local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


def local_mpi_comm():
    return local_comm


def mpi_disabled() -> bool:
    """True if TLLM_DISABLE_MPI is set to "1", False otherwise."""
    return os.environ.get("TLLM_DISABLE_MPI") == "1"


def mpi_rank():
    if mpi_disabled():
        try:
            return torch.distributed.get_rank()
        except ValueError:
            # Fallback: return 0 when MPI is absent (Ray / Slurm PMIx)
            return 0
    return mpi_comm().Get_rank() if ENABLE_MULTI_DEVICE else 0


def global_mpi_rank():
    if mpi_disabled():
        # Fallback: return 0 when MPI is absent (Ray / Slurm PMIx)
        return 0

    return MPI.COMM_WORLD.Get_rank() if ENABLE_MULTI_DEVICE else 0


def global_mpi_size():
    return MPI.COMM_WORLD.Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_world_size():
    return mpi_comm().Get_size() if ENABLE_MULTI_DEVICE else 1


def local_mpi_rank():
    if mpi_disabled():
        # For Ray/non-MPI: the device was already set during worker init
        # torch.cuda.current_device() returns the correct local device ID
        try:
            return torch.cuda.current_device()
        except ValueError:
            return 0
    return mpi_comm().Get_rank() % torch.cuda.device_count(
    ) if ENABLE_MULTI_DEVICE else 0


def local_mpi_size():
    return local_comm.Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Barrier()


def local_mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        local_comm.Barrier()


def mpi_broadcast(obj, root=0):
    return mpi_comm().bcast(obj, root) if global_mpi_size() > 1 else obj


def mpi_allgather(obj):
    return mpi_comm().allgather(obj) if ENABLE_MULTI_DEVICE else obj


def mpi_isend(buf, dest, tag=0):
    # isend in buf-like objects (e.g. numpy array)
    # return request handle if ENABLE_MULTI_DEVICE
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().Isend(buf, dest, tag=tag)
    return None


def mpi_send(buf, dest, tag=0):
    # send in buf-like objects (e.g. numpy array)
    # return request handle if ENABLE_MULTI_DEVICE
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Send(buf, dest, tag=tag)
    return None


def mpi_recv(buf, source, tag):
    # recv in buf-like object (e.g. numpy array)
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().Recv(buf, source, tag=tag)
    return None


def mpi_send_object(obj, dest, tag=0):
    if ENABLE_MULTI_DEVICE:
        mpi_comm().send(obj, dest=dest, tag=tag)


def mpi_isend_object(obj, dest, tag=0):
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().isend(obj, dest=dest, tag=tag)
    return None


def mpi_recv_object(source, tag):
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().recv(source=source, tag=tag)
    return None
