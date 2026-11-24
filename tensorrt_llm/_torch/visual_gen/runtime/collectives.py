# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
from torch import nn

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping, MpiTopology

__all__ = [
    "AllReduceStrategy",
    "AsyncAllToAll",
    "BufferKind",
    "TensorParallelSubgroup",
    "TensorParallelSum",
]


class TensorParallelSubgroup(MpiTopology):
    """A tensor-parallel mapping for the first `size` ranks and their explicit process group."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, size: int, rank: int, group: dist.ProcessGroup | None):
        super().__init__(world_size=size, rank=rank, tp_size=size, gpus_per_node=size)
        self.process_group = group

    @property
    def tp_group_pg(self) -> dist.ProcessGroup:
        if self.process_group is None:
            raise RuntimeError("Tensor-parallel subgroup is not initialized.")
        return self.process_group

    @property
    def tp_group_name(self) -> str:
        return self.tp_group_pg.group_name


class TensorParallelSum(nn.Module):
    """Sum across a TP mapping and supply its GEMM output allocation policy."""

    def __init__(self, mapping: Mapping, strategy: AllReduceStrategy, dtype: torch.dtype | None):
        super().__init__()
        self.mapping = mapping
        self._process_group = None
        self._engine = None
        if mapping.tp_size > 1:
            if isinstance(mapping, TensorParallelSubgroup) and mpi_disabled():
                self._process_group = mapping.tp_group_pg
            else:
                self._engine = AllReduce(mapping=mapping, strategy=strategy, dtype=dtype)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._process_group is not None:
            result = tensor.contiguous().clone()
            dist.all_reduce(result, group=self._process_group)
            return result
        if self._engine is not None:
            return self._engine(tensor)
        return tensor

    def output_allocation(self) -> tuple[int, list[int] | None]:
        if self._engine is not None and self._engine.uses_nccl_symmetric_memory_window():
            return int(BufferKind.NCCL_WINDOW), self.mapping.tp_group
        return int(BufferKind.DEFAULT), None


class AsyncAllToAll:
    """Push sequence shards on a dedicated stream and join a batch of transfers."""

    def __init__(self, group: dist.ProcessGroup, stream: torch.cuda.Stream):
        self._group = group.boxed()
        self._stream = stream

    @torch.compiler.disable(recursive=False)
    def push(self, tensor: torch.Tensor) -> torch.Tensor:
        recv, ticket = torch.ops.trtllm.ulysses_a2a_async_prepare(tensor, self._group)
        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(self._stream):
            ready.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(ticket, self._group)
        return recv

    @torch.compiler.disable(recursive=False)
    def wait(self, transfers: int) -> None:
        with torch.cuda.stream(self._stream):
            for _ in range(transfers):
                torch.ops.trtllm.ulysses_a2a_async_barrier(self._group, 60000)
            done = torch.cuda.Event()
            done.record()
        torch.cuda.current_stream().wait_event(done)
