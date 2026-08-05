# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from typing import List, Optional, Tuple

try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cuda, cudart

from .logger import logger
from .mapping import Mapping


def _raise_if_error(error: cudart.cudaError_t | cuda.CUresult):
    if isinstance(error, cudart.cudaError_t):
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime API error: {error!r}")
    if isinstance(error, cuda.CUresult):
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Driver API error: {error!r}")


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
            ipc_memory = IpcMemory.open_ipc_memory(self.mapping, size, True)
            if ipc_memory is None:
                # CUDA IPC is unusable on this system. Keep the pointers null so that
                # callers fall back to the non-IPC (NCCL) path instead of failing hard.
                self.open_ipc = False
            else:
                self.peer_ptrs, self.local_ptr = ipc_memory

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
    ) -> Optional[Tuple[List[int], int]]:
        """Allocates a buffer with the given *size* on each GPU. Then, enables IPC communication between TP groups.
        Returns a list of buffer pointers, buffers[i] is a handle to the corresponding buffer residing on GPU #i.
        Call close_ipc_handle with the *buffer*.

        Returns None when CUDA IPC is not usable. `cudaDeviceCanAccessPeer` only reports that the
        GPUs can address each other's memory, it does not guarantee that IPC handles can be
        exported/imported: that additionally fails on GPUs without CUDA IPC support and when the
        ranks do not share an IPC namespace. The outcome is agreed upon by the whole TP group, so
        that either every rank gets IPC buffers or none of them does.
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
        ipc_error = None if error == cudart.cudaError_t.cudaSuccess else error
        if ipc_error is not None:
            # Exchange a dummy handle to keep the collective below symmetric.
            local_handle = cudart.cudaIpcMemHandle_t()
        handles_reserved = dist.tp_allgather(local_handle.reserved)

        handles = []
        for reserved in handles_reserved:
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = reserved
            handles.append(handle)

        peer_ptrs = []
        opened_ptrs = []
        for node, handle in enumerate(handles):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
            elif ipc_error is None:
                error, ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                if error != cudart.cudaError_t.cudaSuccess:
                    ipc_error = error
                    continue
                peer_ptrs.append(ptr)
                opened_ptrs.append(ptr)

        if not all(dist.tp_allgather(ipc_error is None)):
            if ipc_error is not None:
                logger.warning(
                    f"CUDA IPC is not usable on this system: {ipc_error!r}. "
                    "Custom all-reduce kernels are disabled, falling back to NCCL."
                )
            for ptr in opened_ptrs:
                _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])
            _raise_if_error(cudart.cudaFree(local_ptr)[0])
            return None

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
