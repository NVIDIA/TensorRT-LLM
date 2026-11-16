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
import ctypes
import struct
import sys
from typing import List, Tuple

try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cuda, cudart

from .logger import logger
from .mapping import Mapping

# Must match CUDA_IPC_HANDLE_SIZE / CU_IPC_HANDLE_SIZE.
_IPC_MEM_HANDLE_SIZE = 64


def _raise_if_error(error: cudart.cudaError_t | cuda.CUresult):
    if isinstance(error, cudart.cudaError_t):
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime API error: {repr(error)}")
    if isinstance(error, cuda.CUresult):
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Driver API error: {repr(error)}")


def _ipc_mem_handle_to_bytes(handle) -> bytes:
    """Serialize a cudaIpcMemHandle across cuda-python / cuda-bindings versions."""
    reserved = getattr(handle, "reserved", None)
    if reserved is not None:
        return bytes(reserved)
    # Some cuda-bindings builds omit the .reserved attribute on cudaIpcMemHandle_t.
    return ctypes.string_at(handle.getPtr(), _IPC_MEM_HANDLE_SIZE)


def _ipc_mem_handle_from_bytes(data: bytes):
    """Rebuild a cudaIpcMemHandle from the opaque 64-byte payload."""
    handle = cudart.cudaIpcMemHandle_t()
    if hasattr(handle, "reserved"):
        try:
            handle.reserved = data
        except TypeError:
            # Older bindings expect list[int] rather than bytes.
            handle.reserved = list(data)
    else:
        if len(data) != _IPC_MEM_HANDLE_SIZE:
            raise ValueError(
                f"Invalid CUDA IPC mem handle size: {len(data)} (expected {_IPC_MEM_HANDLE_SIZE})"
            )
        ctypes.memmove(handle.getPtr(), data, _IPC_MEM_HANDLE_SIZE)
    return handle


def can_access_peer(mapping: Mapping) -> bool:
    src_node = mapping.local_rank

    for rank in mapping.tp_group:
        dest_node = mapping.get_local_rank(rank)

        # Early exit if devices are on different nodes
        if mapping.get_node_rank(rank) != mapping.node_rank:
            logger.info(
                f"Detect inter-node TP between rank {mapping.rank} and rank {rank}, fail to access peer GPU memory"
            )
            return False

        # Skip if same device
        if dest_node == src_node:
            continue

        error, result = cudart.cudaDeviceCanAccessPeer(src_node, dest_node)
        _raise_if_error(error)

        if result == 0:
            logger.info(
                f"cudaDeviceCanAccessPeer failed for device: {src_node} peerDevice: {dest_node}"
            )
            return False

    return True


class IpcMemory:
    # WARNING: Must in sync with FLAGS_SIZE in cpp/include/tensorrt_llm/runtime/ipcUtils.h
    # (Max all reduce blocks + 1) * sizeof(int)
    IPC_BARRIERS_SIZE_PER_GPU = (24 + 1) * 4

    def __init__(self, mapping: Mapping, size: int, open_ipc: bool = True):
        self.mapping = mapping
        self.open_ipc = open_ipc and mapping.tp_size <= mapping.gpus_per_node
        self.peer_ptrs = [0] * mapping.tp_size
        self.local_ptr = 0

        if self.open_ipc:
            self.peer_ptrs, self.local_ptr = IpcMemory.open_ipc_memory(self.mapping, size, True)

    def __del__(self):
        if not sys.is_finalizing() and self.open_ipc:
            IpcMemory.close_ipc_memory(self.mapping, self.peer_ptrs)

    def serialize(self) -> List[int]:
        buffer = bytes(0)
        for ptr in self.peer_ptrs:
            buffer += struct.pack("P", ptr)

        return array.array("Q", buffer).tolist()

    @staticmethod
    def open_ipc_memory(
        mapping: Mapping, size: int, set_to_zero: bool = False
    ) -> Tuple[List[int], int]:
        """Allocates a buffer with the given *size* on each GPU. Then, enables IPC communication between TP groups.
        Returns a list of buffer pointers, buffers[i] is a handle to the corresponding buffer residing on GPU #i.
        Call close_ipc_handle with the *buffer*.
        """

        def align_size(size, alignment):
            if (size % alignment) != 0:
                size += alignment - (size % alignment)
            return size

        from tensorrt_llm._torch.distributed.communicator import Distributed

        dist = Distributed.get(mapping)

        # see allocateIpcMemory in cpp/tensorrt_llm/runtime/ipcUtils.cpp for alignment reason
        # 1 << 21 is 2MB
        aligned_size = align_size(size, 1 << 21)
        error, local_ptr = cudart.cudaMalloc(aligned_size)
        _raise_if_error(error)
        if set_to_zero:
            _raise_if_error(cudart.cudaMemset(local_ptr, 0, aligned_size)[0])
        error, local_handle = cudart.cudaIpcGetMemHandle(local_ptr)
        _raise_if_error(error)
        handles_reserved = dist.tp_allgather(_ipc_mem_handle_to_bytes(local_handle))
        handles = [_ipc_mem_handle_from_bytes(reserved) for reserved in handles_reserved]

        peer_ptrs = []
        for node, handle in enumerate(handles):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
            else:
                error, ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                _raise_if_error(error)
                peer_ptrs.append(ptr)

        return peer_ptrs, local_ptr

    @staticmethod
    def close_ipc_memory(mapping: Mapping, peer_ptrs: List[int]):
        for node, ptr in enumerate(peer_ptrs):
            if node == mapping.tp_rank:
                if ptr != 0:
                    _raise_if_error(cudart.cudaFree(ptr)[0])
            else:
                if ptr != 0:
                    _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])
