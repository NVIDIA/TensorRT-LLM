# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank row exchange on the copy engines: each rank pushes a tensor into the peer's receive
buffer over NVLink with `cudaMemcpyAsync` and the pair signals with stream memory operations, so a
GEMM it overlaps keeps every SM.

Protocol (`exchange`), on the caller's stream, every flag in the memory of the rank that waits on
it, every value a constant (so a CUDA graph replays it):

  1. write peer.ready = 1      my receive buffer is free (everything before on this stream is done)
  2. wait  my.ready == 1; write my.ready = 0      the peer's buffer is free
  3. copy  my rows -> the peer's receive buffer   (copy engine)
  4. write peer.data = 1       (the write's default memory barrier orders it after the copy)
  5. wait  my.data == 1; write my.data = 0        the peer's rows are in my buffer

A rank's step-1 signal for exchange k is issued after its step 5 of exchange k - 1, which follows
the peer's reset of `ready` for k - 1 (step 2 precedes the peer's step 4); likewise for `data`, so no
signal is lost or seen twice. The result is a view of the receive buffer, valid until the next
exchange's step 1. Both ranks must call `exchange` identically (same calls, order and sizes).
"""

from __future__ import annotations

import torch

#: Bytes kept for the two flags after the data region (one 128-byte line each).
_FLAG_BYTES = 256


class _CudaArray:
    """`__cuda_array_interface__` over a raw device allocation (uint8), for `torch.as_tensor`."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


class CopyEngineExchange:
    """`exchange(x)` -> the peer's `x` (a view of this rank's receive buffer, same shape and
    dtype). `capacity` bytes a call at most. Built on the attention-TP group of a two-rank
    `mapping` (IPC memory inside a node); `create` returns None where that does not hold."""

    def __init__(self, mapping, capacity: int):
        from cuda.bindings import driver, runtime

        from tensorrt_llm._ipc_utils import IpcMemory

        assert mapping.tp_size == 2
        self._driver, self._runtime = driver, runtime
        self.capacity = capacity
        self._ipc = IpcMemory(mapping, capacity + _FLAG_BYTES, open_ipc=True)
        rank = mapping.tp_rank
        local, peer = self._ipc.peer_ptrs[rank], self._ipc.peer_ptrs[1 - rank]
        assert local and peer, "IPC memory did not open"
        self._peer = peer
        self._my_ready, self._my_data = local + capacity, local + capacity + 128
        self._peer_ready, self._peer_data = peer + capacity, peer + capacity + 128
        self._recv = torch.as_tensor(_CudaArray(local, capacity), device=torch.device("cuda"))

    @classmethod
    def create(cls, mapping, capacity: int):
        """The exchange over `mapping`'s attention-TP pair, or None (not two ranks, ranks on
        different nodes, or IPC memory unavailable on either rank). A collective of the pair:
        both ranks build it or neither."""
        if mapping.tp_size != 2 or mapping.tp_size > mapping.gpus_per_node:
            return None
        from tensorrt_llm._torch.distributed.communicator import Distributed

        try:
            exchange = cls(mapping, capacity)
        except Exception:  # noqa: BLE001 -- keep NCCL where the copy engines cannot be reached
            exchange = None
        if not all(Distributed.get(mapping).tp_allgather(exchange is not None)):
            return None
        return exchange

    def _write(self, stream, addr: int, value: int) -> None:
        d = self._driver
        (err,) = d.cuStreamWriteValue32(
            stream,
            d.CUdeviceptr(addr),
            value,
            d.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        assert err == d.CUresult.CUDA_SUCCESS, err

    def _wait(self, stream, addr: int, value: int) -> None:
        d = self._driver
        (err,) = d.cuStreamWaitValue32(
            stream,
            d.CUdeviceptr(addr),
            value,
            d.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        assert err == d.CUresult.CUDA_SUCCESS, err

    def exchange(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and x.is_cuda
        nbytes = x.numel() * x.element_size()
        assert nbytes <= self.capacity, (nbytes, self.capacity)
        stream = self._driver.CUstream(torch.cuda.current_stream().cuda_stream)
        r = self._runtime
        self._write(stream, self._peer_ready, 1)
        self._wait(stream, self._my_ready, 1)
        self._write(stream, self._my_ready, 0)
        (err,) = r.cudaMemcpyAsync(
            self._peer,
            x.data_ptr(),
            nbytes,
            r.cudaMemcpyKind.cudaMemcpyDefault,
            r.cudaStream_t(torch.cuda.current_stream().cuda_stream),
        )
        assert err == r.cudaError_t.cudaSuccess, err
        self._write(stream, self._peer_data, 1)
        self._wait(stream, self._my_data, 1)
        self._write(stream, self._my_data, 0)
        return self._recv[:nbytes].view(x.dtype).view(x.shape)
