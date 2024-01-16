# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import array
import struct
from contextlib import contextmanager
from typing import List, Tuple

from cuda import cudart
from cuda.cudart import cudaError_t

from .mapping import Mapping


def _raise_if_error(error: cudaError_t):
    if error != cudaError_t.cudaSuccess:
        raise RuntimeError(error)


@contextmanager
def peer_access(mapping: Mapping):
    set_peer_access(mapping, True)
    try:
        yield
    finally:
        set_peer_access(mapping, False)


def set_peer_access(mapping: Mapping, enabled: bool = True):
    src_node = mapping.rank
    for dest_node in mapping.tp_group:
        if dest_node == src_node:
            continue

        error, result = cudart.cudaDeviceCanAccessPeer(src_node, dest_node)
        _raise_if_error(error)

        if result == 0:
            raise RuntimeError(
                f"Can't enable access between nodes {src_node} and {dest_node}")

        if enabled:
            cudart.cudaDeviceEnablePeerAccess(dest_node, 0)
        else:
            cudart.cudaDeviceDisablePeerAccess(dest_node)
        error = cudart.cudaGetLastError()[0]
        if error not in [
                cudaError_t.cudaSuccess,
                cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
                cudaError_t.cudaErrorPeerAccessNotEnabled
        ]:
            raise RuntimeError(error)


class IpcMemory():

    IPC_BARRIERS_SIZE_PER_GPU = 25 * 4  # Max all reduce blocks * sizeof(float)

    def __init__(self, mapping, size):
        self.mapping = mapping
        self.peer_ptrs, self.local_ptr = IpcMemory.open_ipc_memory(
            self.mapping, size, True)

    def __del__(self):
        IpcMemory.close_ipc_memory(self.mapping, self.peer_ptrs)

    def serialize(self) -> List[int]:
        buffer = bytes(0)
        for ptr in self.peer_ptrs:
            buffer += struct.pack("P", ptr)

        return array.array("Q", buffer).tolist()

    @staticmethod
    def open_ipc_memory(mapping: Mapping,
                        size: int,
                        set_to_zero: bool = False) -> Tuple[List[int], int]:
        """ Allocates a buffer with the given *size* on each GPU. Then, enables IPC communication between TP groups.
        Returns a list of buffer pointers, buffers[i] is a handle to the corresponding buffer residing on GPU #i.
        Call close_ipc_handle with the *buffer*.
        """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD.Split(mapping.pp_rank, mapping.tp_rank)

        error, local_ptr = cudart.cudaMalloc(size)
        _raise_if_error(error)
        if set_to_zero:
            _raise_if_error(cudart.cudaMemset(local_ptr, 0, size)[0])
        error, local_handle = cudart.cudaIpcGetMemHandle(local_ptr)
        _raise_if_error(error)

        handles_reserved = comm.allgather(local_handle.reserved)
        handles = []
        for reserved in handles_reserved:
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = reserved
            handles.append(handle)

        peer_ptrs = []
        for node, handle in enumerate(handles):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
            else:
                error, ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess)
                _raise_if_error(error)
                peer_ptrs.append(ptr)

        return peer_ptrs, local_ptr

    @staticmethod
    def close_ipc_memory(mapping: Mapping, peer_ptrs: List[int]):
        for node, ptr in enumerate(peer_ptrs):
            if node == mapping.tp_rank:
                _raise_if_error(cudart.cudaFree(ptr)[0])
            else:
                _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])
