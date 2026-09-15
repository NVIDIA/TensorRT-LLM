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
        # Set only when IPC was expected to work (P2P reported and the TP group fits in
        # one node) but the handle exchange failed anyway. Buffers that never requested
        # IPC in the first place -- inter-node TP for instance -- keep `open_ipc` False
        # with this flag unset, so that callers can tell the two cases apart.
        self.ipc_failed = False
        self.peer_ptrs = [0] * mapping.tp_size
        self.local_ptr = 0

        if self.open_ipc:
            ipc_memory = IpcMemory.open_ipc_memory(self.mapping, size, True)
            if ipc_memory is None:
                # CUDA IPC is unusable on this system. Keep the pointers null so that
                # callers fall back to the non-IPC (NCCL) path instead of failing hard.
                self.open_ipc = False
                self.ipc_failed = True
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

        def disable_ipc(reason: str, opened_ptrs: List[int]) -> None:
            # Every rank of the group takes this path, including the ones whose own
            # handles were fine, so log unconditionally: otherwise a rank that is
            # falling back because of a peer would degrade silently.
            logger.warning_once(
                f"CUDA IPC is not usable on this system: {reason} "
                "Custom all-reduce kernels are disabled, falling back to NCCL.",
                key="cuda-ipc-unavailable",
            )
            for ptr in opened_ptrs:
                _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])
            _raise_if_error(cudart.cudaFree(local_ptr)[0])

        # A rank that cannot export its handle contributes None instead of an
        # uninitialized `cudaIpcMemHandle_t`, so that no rank ever hands garbage to
        # `cudaIpcOpenMemHandle`. This single collective carries both the payload and
        # the agreement on whether exporting worked everywhere.
        error, local_handle = cudart.cudaIpcGetMemHandle(local_ptr)
        get_error = None if error == cudart.cudaError_t.cudaSuccess else error
        handles_reserved = dist.tp_allgather(local_handle.reserved if get_error is None else None)

        if any(reserved is None for reserved in handles_reserved):
            disable_ipc(
                f"cudaIpcGetMemHandle failed with {get_error!r}."
                if get_error is not None
                else "cudaIpcGetMemHandle failed on a peer rank of the TP group.",
                [],
            )
            return None

        peer_ptrs = []
        opened_ptrs = []
        open_error = None
        for node, reserved in enumerate(handles_reserved):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
                continue
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = reserved
            error, ptr = cudart.cudaIpcOpenMemHandle(handle, cudart.cudaIpcMemLazyEnablePeerAccess)
            if error != cudart.cudaError_t.cudaSuccess:
                open_error = error
                break
            peer_ptrs.append(ptr)
            opened_ptrs.append(ptr)

        if not all(dist.tp_allgather(open_error is None)):
            disable_ipc(
                f"cudaIpcOpenMemHandle failed with {open_error!r}."
                if open_error is not None
                else "cudaIpcOpenMemHandle failed on a peer rank of the TP group.",
                opened_ptrs,
            )
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
