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

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

#: Bytes kept for the two flags after the data region (one 128-byte line each).
_FLAG_BYTES = 256
HIDDEN = 2560
#: Words of one QSA selection row: the target's `token_topk + compress_ratio - 1`.
_SELECTION_WIDTH = 2051


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
    """The pair's buffers and flags: `exchange(x)` -> the peer's `x` (`ce_exchange`, a view of
    this rank's receive buffer, same shape and dtype). `capacity` bytes a call at most. Built on
    the attention-TP group of a two-rank `mapping` (IPC memory inside a node); `create` returns
    None where that does not hold."""

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
        except (RuntimeError, OSError, MemoryError) as error:
            logger.warning(f"copy-engine exchange unavailable ({error}); the pair runs NCCL")
            exchange = None
        if not all(Distributed.get(mapping).tp_allgather(exchange is not None)):
            return None
        return exchange

    def exchange(self, x: torch.Tensor) -> torch.Tensor:
        """`ce_exchange(self, x)`."""
        return ce_exchange(self, x)


def _write(driver, stream, addr: int, value: int) -> None:
    (err,) = driver.cuStreamWriteValue32(
        stream,
        driver.CUdeviceptr(addr),
        value,
        driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err


def _wait(driver, stream, addr: int, value: int) -> None:
    (err,) = driver.cuStreamWaitValue32(
        stream,
        driver.CUdeviceptr(addr),
        value,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err


class CeExchange(OpWrapper):
    """The peer's `x` over `pair` (a `CopyEngineExchange`), by the module docstring's protocol on
    the caller's stream: a view of this rank's receive buffer, same shape and dtype, valid until
    the next exchange. Both ranks call it identically."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"sp2: {why}", spec=dict(world=2, rows=rows, width=width, dtype=dtype))
        for dtype, width, what in (
            (torch.bfloat16, HIDDEN, "bf16 hidden rows"),
            (torch.int32, _SELECTION_WIDTH, "int32 QSA selection rows"),
        )
        for rows, why in (
            (8192, f"half of a 16384-token step's {what}, the buffer's largest exchange"),
            (4096, f"half of an 8192-token step's {what}, also replayed from a CUDA graph chain"),
            (1024, f"half of a 2048-token chunk's {what}"),
            (77, f"an odd row count of {what}"),
        )
    )

    note = """
    Certified byte-exact against the peer's input, eagerly and replayed from a CUDA graph chain
    with new inputs. Not certified: pairs spanning nodes, groups other than two ranks.
    """

    def __call__(self, pair: CopyEngineExchange, x: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and x.is_cuda
        nbytes = x.numel() * x.element_size()
        assert nbytes <= pair.capacity, (nbytes, pair.capacity)
        d, r = pair._driver, pair._runtime
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        _write(d, stream, pair._peer_ready, 1)
        _wait(d, stream, pair._my_ready, 1)
        _write(d, stream, pair._my_ready, 0)
        (err,) = r.cudaMemcpyAsync(
            pair._peer,
            x.data_ptr(),
            nbytes,
            r.cudaMemcpyKind.cudaMemcpyDefault,
            r.cudaStream_t(torch.cuda.current_stream().cuda_stream),
        )
        assert err == r.cudaError_t.cudaSuccess, err
        _write(d, stream, pair._peer_data, 1)
        _wait(d, stream, pair._my_data, 1)
        _write(d, stream, pair._my_data, 0)
        return pair._recv[:nbytes].view(x.dtype).view(x.shape)

    def reference(self, xs_in_rank_order: Sequence[torch.Tensor], rank: int) -> torch.Tensor:
        """The other rank's `x`. The documented collective exception to mirroring `__call__`."""
        return xs_in_rank_order[1 - rank]


ce_exchange = CeExchange()
