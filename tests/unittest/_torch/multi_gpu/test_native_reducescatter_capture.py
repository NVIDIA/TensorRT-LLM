# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native MPI operator's cold and warm CUDA graph paths."""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# MPIPoolExecutor retains its management thread after first use.
pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _capture_first_reducescatter(_):
    world_size = tensorrt_llm.mpi_world_size()
    rank = tensorrt_llm.mpi_rank()
    assert world_size == 8
    torch.cuda.set_device(rank)
    group = list(range(world_size))
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        # Initialize only the base communicator. In particular, do not warm
        # ReduceScatter: that would hide its cold eligibility synchronization.
        local = torch.full((1,), rank, dtype=torch.bfloat16, device="cuda")
        gathered = torch.ops.trtllm.allgather(local, None, group)
        torch.testing.assert_close(
            gathered,
            torch.arange(world_size, dtype=torch.bfloat16, device="cuda"),
            rtol=0,
            atol=0,
        )

        # 24 MiB full input and 3 MiB output fall inside the prefill window.
        # Distinct destination blocks catch incorrect slicing as well as sums.
        data = torch.empty((2048, 6144), dtype=torch.bfloat16, device="cuda")
        for destination, chunk in enumerate(data.chunk(world_size)):
            chunk.fill_(rank + 1 + destination)
        expected = torch.full_like(data[:256], 36 + 8 * rank)
    stream.synchronize()

    def capture_and_check():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = torch.ops.trtllm.reducescatter(data, None, group)
        with torch.cuda.stream(stream):
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            # A replay must read the current input, not a capture-time result.
            data.add_(1)
            expected.add_(world_size)
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        stream.synchronize()

    print(f"rank={rank} pid={os.getpid()} phase=first-capture", flush=True)
    capture_and_check()

    # The capture fallback must leave eligibility retryable. On SM103 this
    # eager call initializes the prefill communicator; other GPUs cache bypass.
    print(f"rank={rank} pid={os.getpid()} phase=eager", flush=True)
    with torch.cuda.stream(stream):
        eager = torch.ops.trtllm.reducescatter(data, None, group)
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    stream.synchronize()
    print(f"rank={rank} pid={os.getpid()} phase=warm-capture", flush=True)
    capture_and_check()
    return rank


def test_first_native_reducescatter_inside_cuda_graph():
    if torch.cuda.device_count() < 8:
        pytest.skip("requires eight local GPUs")
    # A fresh executor is essential: a module-scoped pool could inherit a
    # cached prefill communicator from another ReduceScatter test.
    with MPIPoolExecutor(max_workers=8) as executor:
        ranks = list(executor.map(_capture_first_reducescatter, range(8)))
    assert sorted(ranks) == list(range(8))
