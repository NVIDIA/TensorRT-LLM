# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An NCCL window buffer handed to a CUDA graph capture must not be recycled.

The defect this pins: `NCCLWindowAllocator::requestBuffer` will reuse a pooled
window buffer while a graph is capturing (only *registration* is skipped during
capture, and the best-fit reuse branch runs before that check). The captured
kernel then holds that address for the life of the graph, but
`createNCCLWindowTensor`'s deleter calls `releaseBuffer` as soon as the returned
tensor dies -- which, for an all-reduce whose output is an intermediate, is at
the end of the capture. The buffer goes back on the free list, the next
requester is given it, and every replay of the graph writes into memory that now
belongs to someone else.

On a real model that surfaces as a non-finite residual and a decode collapse to
token 0. This test does not go looking for that: it asserts the aliasing
directly, which is deterministic where the numerical symptom is not.

  1. capture a graph whose all-reduce output is consumed by a following op, so
     nothing holds the window tensor afterwards;
  2. ask the same allocator for a window buffer of that size;
  3. write a sentinel into it;
  4. replay the graph;
  5. the sentinel must still be there.

Measured before the fix: clobbered on 10/10 replays at 2 KiB, 12 KiB and 96 KiB.
After suppressing the recycling: 0/10 at every size.
"""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceParams,
                                             AllReduceStrategy)
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

_FAILED_PREFIX = "FAILED:"
_SENTINEL = 7.0
_REPLAYS = 10


def run_single_rank(tp_size, single_rank_func, *args):
    """Run one rank and return True or a FAILED: string.

    Exceptions are stringified rather than raised: mpi4py would otherwise have
    to pickle a traceback whose frames hold torch.ops references.
    """
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        with torch.inference_mode():
            single_rank_func(tp_size, rank, *args)
    except Exception as exc:  # noqa: BLE001
        import traceback
        return f"{_FAILED_PREFIX} rank {rank}: {exc!r}\n{traceback.format_exc()}"
    return True


def _run_capture_reuse(tp_size, rank, tokens, hidden, second_graph=False):
    from tensorrt_llm.bindings.internal.thop import BufferKind

    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_SYMMETRIC,
                          dtype=torch.bfloat16)
    params = AllReduceParams()

    def once(inp):
        out = allreduce(inp, all_reduce_params=params)
        return out[0] if isinstance(out, (list, tuple)) else out

    x = torch.full((tokens, hidden), float(rank + 1), dtype=torch.bfloat16, device="cuda")

    static_in = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        once(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # `* 1.0` is load-bearing: it makes the all-reduce output an
        # intermediate, so its window tensor is dropped at the end of the
        # capture. Returning it directly keeps the buffer inUse and the defect
        # cannot occur -- which is why several earlier attempts at this test
        # passed against a broken build.
        _ = once(static_in) * 1.0
    torch.cuda.synchronize()

    if second_graph:
        # The shape a real model has: graph A's intermediate all-reduce output is
        # released at the end of A's capture, graph B's capture is handed the
        # same buffer and KEEPS it as its output, and replaying A corrupts B.
        # Measured to clobber 10/10 replays, identically to the plain-allocation
        # victim -- the defect does not care which kind of consumer inherits the
        # buffer.
        graph_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_b):
            victim = once(static_in)
        torch.cuda.synchronize()
    else:
        victim, kind = torch.ops.trtllm.allocate_output(
            x, int(BufferKind.NCCL_WINDOW), list(mapping.tp_group), None, None)
        if int(kind) != int(BufferKind.NCCL_WINDOW):
            pytest.skip("NCCL window allocation unavailable on this platform")
    victim.fill_(_SENTINEL)
    torch.cuda.synchronize()

    for i in range(_REPLAYS):
        static_in.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        clobbered = int((victim.float() != _SENTINEL).sum().item())
        assert clobbered == 0, (
            f"replay {i}: the captured all-reduce wrote into a window buffer the "
            f"allocator had already handed to another owner "
            f"({clobbered} of {victim.numel()} elements changed). A buffer given "
            f"out while cudaStreamIsCapturing() must not be returned to the free "
            f"pool when its tensor dies.")


# Already launched under a multi-rank MPI job (srun --ntasks=N --mpi=pmix)? Then
# run in place. `MPIPoolExecutor` spawns its workers with MPI_Comm_spawn, which
# does not work on every launcher -- on this cluster it hangs after collection --
# and a test that cannot be executed where the bug lives is not much of a test.
_LAUNCHED_MULTI_RANK = MPI.COMM_WORLD.Get_size() > 1


_SIZES = [
    pytest.param(1, 1024, id="tokens1_hidden1024"),
    pytest.param(1, 6144, id="tokens1_hidden6144"),
    pytest.param(8, 6144, id="tokens8_hidden6144"),
]


# Requesting the mpi_pool_executor fixture at all is what hangs under a launcher
# where MPI_Comm_spawn does not work, so the two modes are separate tests rather
# than one test with a branch: a skipped test never touches the fixture.
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.skipif(not _LAUNCHED_MULTI_RANK, reason="not launched under multi-rank MPI")
@pytest.mark.parametrize("tokens,hidden", _SIZES)
def test_capture_owned_window_buffer_is_not_recycled_inplace(tokens, hidden):
    """For `srun --ntasks=N --mpi=pmix python -m pytest ...`."""
    result = run_single_rank(MPI.COMM_WORLD.Get_size(), _run_capture_reuse, tokens, hidden)
    assert result is True, result


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.skipif(_LAUNCHED_MULTI_RANK, reason="already multi-rank; see the in-place test")
@pytest.mark.parametrize("tokens,hidden", _SIZES)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_capture_owned_window_buffer_is_not_recycled(tokens, hidden, mpi_pool_executor):
    """For CI, which drives these tests through an MPIPoolExecutor."""
    tp_size = mpi_pool_executor.num_workers
    results = list(
        mpi_pool_executor.map(
            run_single_rank,
            *zip(*[(tp_size, _run_capture_reuse, tokens, hidden)] * tp_size),
        )
    )
    for r in results:
        assert r is True, r


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.skipif(not _LAUNCHED_MULTI_RANK, reason="not launched under multi-rank MPI")
@pytest.mark.parametrize("tokens,hidden", _SIZES)
def test_capture_owned_buffer_is_not_given_to_another_graph(tokens, hidden):
    """The cross-graph form, which is what a real model actually does."""
    result = run_single_rank(MPI.COMM_WORLD.Get_size(), _run_capture_reuse, tokens,
                             hidden, True)
    assert result is True, result
