# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Multicast vocab all-gather of the vocab-parallel draft head: the NCCL all-gather + FP32 widening, bit for
bit, eager and in CUDA graphs, over mixed row counts on one workspace."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_hopper

import tensorrt_llm
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.distributed.nvls_allgather import NVLSVocabGather
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

SHARD = 1024


def _local(rank, rows, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed * 1000 + rank)
    return (torch.randn(rows, SHARD, device="cuda", generator=gen) * 8).to(torch.bfloat16)


def _run_rank(world_size, row_counts):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    from tensorrt_llm._torch.distributed import nvls_allgather

    nvls_allgather._MIN_WORLD_SIZE = world_size
    try:
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
        gather = NVLSVocabGather.get(mapping, SHARD)
        assert gather is not None

        def ref(x):
            return allgather(x, mapping, dim=-1).float()

        # Eager, mixed row counts; one row holds the sentinel bits (a NaN) and +-0.
        for i, rows in enumerate(row_counts):
            x = _local(rank, rows, i + 1)
            if i == 0:
                x.view(torch.int16)[0, 3] = -1
                x[0, 5] = -0.0
            assert gather.can_gather(x)
            out = gather(x)
            expect = ref(x)
            if i == 0:
                for k in range(world_size):
                    assert torch.isnan(out[0, k * SHARD + 3])
                    assert torch.signbit(out[0, k * SHARD + 5])
                    out[0, k * SHARD + 3] = 0
                    expect[0, k * SHARD + 3] = 0
            assert torch.equal(out.view(torch.int32), expect.view(torch.int32)), rows

        # CUDA graph over the same shapes, replayed with new inputs.
        inputs = [
            torch.zeros(rows, SHARD, device="cuda", dtype=torch.bfloat16) for rows in row_counts
        ]
        for x in inputs:
            gather(x)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [gather(x) for x in inputs]
        for rep in range(3):
            for i, x in enumerate(inputs):
                x.copy_(_local(rank, x.shape[0], 100 + 10 * rep + i))
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for x, out in zip(inputs, outs):
                assert torch.equal(out.view(torch.int32), ref(x).view(torch.int32))
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_hopper
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_nvls_vocab_gather(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_rank, *zip(*[(world_size, [1, 8, 3, 128, 16])] * world_size)
    )
    for r in results:
        assert r is True
