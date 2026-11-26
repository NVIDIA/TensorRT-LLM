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
"""One-launch peer-memory all-gather: bit-identical to concatenating the shards along the last dim.

The tensor-parallel group is emulated on one GPU: every rank gets its own zero-filled receive
buffer, state and CUDA stream, and the ranks' launches run concurrently.
"""

import pytest
import torch

from tensorrt_llm._torch.distributed.ll_allgather import LowLatencyAllGather

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _EmulatedGroup:
    def __init__(self, nranks, max_shard_bytes=1 << 16):
        size = LowLatencyAllGather.buffer_bytes(nranks, max_shard_bytes)
        self.buffers = [
            torch.zeros(size // 8, dtype=torch.int64, device="cuda") for _ in range(nranks)
        ]
        ptrs = [b.data_ptr() for b in self.buffers]
        self.gathers = [
            LowLatencyAllGather(r, nranks, ptrs, max_shard_bytes) for r in range(nranks)
        ]
        self.streams = [torch.cuda.Stream() for _ in range(nranks)]

    def gather(self, shards):
        torch.cuda.synchronize()
        outs = []
        for gather, stream, shard in zip(self.gathers, self.streams, shards):
            with torch.cuda.stream(stream):
                outs.append(gather.gather(shard))
        torch.cuda.synchronize()
        return outs


def _shards(nranks, rows, width, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shards = []
    for _ in range(nranks):
        x = torch.randn(rows, width, generator=gen).to(dtype)
        # Bit patterns a value-level copy could disturb: signed zeros, infinities, a NaN payload.
        x[0, :4] = torch.tensor([-0.0, 0.0, float("inf"), float("-inf")]).to(dtype)
        x.view(torch.int16 if x.element_size() == 2 else torch.int32)[-1, -1] = -1
        shards.append(x.cuda())
    return shards


@pytest.mark.parametrize("nranks", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("rows,width", [(13, 256), (48, 1536)])
def test_matches_concatenation(nranks, dtype, rows, width):
    group = _EmulatedGroup(nranks, max_shard_bytes=48 * 1536 * 4)
    shards = _shards(nranks, rows, width, dtype, seed=rows * width + nranks)
    want = torch.cat(shards, dim=-1)
    # Consecutive launches alternate slots; stale packets of the previous one are never read.
    for _ in range(3):
        for got in group.gather(shards):
            assert got.shape == want.shape and got.dtype == want.dtype
            assert torch.equal(got.view(torch.int8), want.view(torch.int8))


@pytest.mark.parametrize("nranks,rows,width,max_programs", [(8, 6, 1536, None), (4, 5, 1100, 16)])
def test_pair_matches_two_gathers(nranks, rows, width, max_programs):
    """``gather_pair`` returns both concatenations from one launch, interleaved with single gathers
    on the same buffers."""
    group = _EmulatedGroup(nranks)
    first = _shards(nranks, rows, width, torch.bfloat16, seed=rows + nranks)
    second = _shards(nranks, rows, width, torch.bfloat16, seed=rows + nranks + 100)
    want = (torch.cat(first, dim=-1), torch.cat(second, dim=-1))
    for _ in range(3):
        torch.cuda.synchronize()
        outs = []
        for gather, stream, x0, x1 in zip(group.gathers, group.streams, first, second):
            with torch.cuda.stream(stream):
                outs.append(gather.gather_pair(x0, x1, max_programs=max_programs))
        torch.cuda.synchronize()
        for got in outs:
            for g, w in zip(got, want):
                assert torch.equal(g.view(torch.int16), w.view(torch.int16))
        for got in group.gather(first):
            assert torch.equal(got.view(torch.int16), want[0].view(torch.int16))


@pytest.mark.parametrize("pair", [False, True], ids=["gather", "gather_pair"])
def test_cuda_graph_replay(pair):
    group = _EmulatedGroup(2)
    # Rank r gathers shards[r]; gather_pair also gathers shards[2 + r].
    shards = [torch.zeros(6, 1536, dtype=torch.bfloat16, device="cuda") for _ in range(4)]

    def launch(rank):
        gather = group.gathers[rank]
        if pair:
            return gather.gather_pair(shards[rank], shards[2 + rank])
        return (gather.gather(shards[rank]),)

    torch.cuda.synchronize()
    for rank, stream in enumerate(group.streams):  # Compile outside capture.
        with torch.cuda.stream(stream):
            launch(rank)
    torch.cuda.synchronize()
    graphs, outs = [], []
    for rank, stream in enumerate(group.streams):
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outs.append(launch(rank))
            graphs.append(graph)
    for seed in range(4):
        for shard, fresh in zip(shards, _shards(4, 6, 1536, torch.bfloat16, seed)):
            shard.copy_(fresh)
        torch.cuda.synchronize()
        for graph, stream in zip(graphs, group.streams):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        want = (torch.cat(shards[:2], dim=-1), torch.cat(shards[2:], dim=-1))
        for out in outs:
            for got, w in zip(out, want):
                assert torch.equal(got.view(torch.int16), w.view(torch.int16))
