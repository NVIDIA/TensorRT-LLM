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
import ctypes
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

# Must match CUDA_IPC_HANDLE_SIZE / CU_IPC_HANDLE_SIZE.
_IPC_MEM_HANDLE_SIZE = 64


def _raise_if_error(error: cudart.cudaError_t | cuda.CUresult):
    if isinstance(error, cudart.cudaError_t):
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime API error: {error!r}")
    if isinstance(error, cuda.CUresult):
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Driver API error: {error!r}")


# cudaError_t names meaning "this system cannot do CUDA IPC between these GPUs", as
# opposed to a misuse of the API. Only these fall back to NCCL, see
# IpcMemory.open_ipc_memory. github.com/NVIDIA/TensorRT-LLM/issues/16899 is
# cudaErrorInvalidDevice from cudaIpcOpenMemHandle on GPUs that report P2P.
_IPC_CAPABILITY_ERROR_NAMES = (
    "cudaErrorInvalidDevice",
    "cudaErrorNotSupported",
    "cudaErrorPeerAccessUnsupported",
)


def _ipc_capability_errors() -> set:
    return {
        int(getattr(cudart.cudaError_t, name))
        for name in _IPC_CAPABILITY_ERROR_NAMES
        if hasattr(cudart.cudaError_t, name)
    }


def _cuda_error_name(code: int) -> str:
    try:
        return repr(cudart.cudaError_t(code))
    except ValueError:
        return f"cudaError_t({code})"


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
        ranks do not share an IPC namespace. Only those capability errors fall back, any other
        IPC error still raises. The outcome is agreed upon by the whole TP group, so that either
        every rank gets IPC buffers, or none of them does, or all of them raise.
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

        def release(opened_ptrs: List[int]) -> None:
            for ptr in opened_ptrs:
                _raise_if_error(cudart.cudaIpcCloseMemHandle(ptr)[0])
            _raise_if_error(cudart.cudaFree(local_ptr)[0])

        def agree(api: str, errors: List[Optional[int]], opened_ptrs: List[int]) -> bool:
            """Group-wide verdict on one IPC step, from every rank's error code.

            Returns True when every rank succeeded. When the only failures are
            capability errors, i.e. this system cannot do CUDA IPC at all, releases
            this rank's buffers and returns False so the caller falls back to NCCL.
            Any other error is a real bug (a corrupted handle for instance) and
            raises, so it is not hidden behind a silent NCCL fallback. Every rank
            sees the same codes, so every rank takes the same branch.
            """
            failed = [(rank, code) for rank, code in enumerate(errors) if code is not None]
            if not failed:
                return True
            release(opened_ptrs)
            failures = ", ".join(f"rank {rank}: {_cuda_error_name(code)}" for rank, code in failed)
            if not all(code in _ipc_capability_errors() for _, code in failed):
                raise RuntimeError(f"{api} failed on the TP group ({failures})")
            # Every rank of the group takes this path, including the ones whose own
            # call was fine, so log unconditionally: otherwise a rank that is
            # falling back because of a peer would degrade silently.
            local_error = errors[mapping.tp_rank]
            reason = (
                f"{api} failed with {_cuda_error_name(local_error)}."
                if local_error is not None
                else f"{api} failed on a peer rank of the TP group ({failures})."
            )
            logger.warning_once(
                f"CUDA IPC is not usable on this system: {reason} "
                "Custom all-reduce kernels are disabled, falling back to NCCL.",
                key="cuda-ipc-unavailable",
            )
            return False

        # A rank that cannot export its handle contributes its error code instead of
        # an uninitialized `cudaIpcMemHandle_t`, so that no rank ever hands garbage to
        # `cudaIpcOpenMemHandle`. This single collective carries both the payload and
        # what every rank needs to agree on the outcome of exporting.
        error, local_handle = cudart.cudaIpcGetMemHandle(local_ptr)
        handles_reserved = dist.tp_allgather(
            _ipc_mem_handle_to_bytes(local_handle)
            if error == cudart.cudaError_t.cudaSuccess
            else int(error)
        )
        if not agree(
            "cudaIpcGetMemHandle",
            [reserved if isinstance(reserved, int) else None for reserved in handles_reserved],
            [],
        ):
            return None

        peer_ptrs = []
        opened_ptrs = []
        open_error = None
        for node, reserved in enumerate(handles_reserved):
            if node == mapping.tp_rank:
                peer_ptrs.append(local_ptr)
                continue
            handle = _ipc_mem_handle_from_bytes(reserved)
            error, ptr = cudart.cudaIpcOpenMemHandle(handle, cudart.cudaIpcMemLazyEnablePeerAccess)
            if error != cudart.cudaError_t.cudaSuccess:
                open_error = int(error)
                break
            peer_ptrs.append(ptr)
            opened_ptrs.append(ptr)

        if not agree("cudaIpcOpenMemHandle", dist.tp_allgather(open_error), opened_ptrs):
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
