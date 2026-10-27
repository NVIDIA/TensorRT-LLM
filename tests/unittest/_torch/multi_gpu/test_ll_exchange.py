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
"""The one-launch TP exchanges (distributed/ll_exchange.py) against the same reduction over every rank's
inputs: key max / gather and greedy argmax of TPKeyExchange, and TPGroupExchange's subgroup all-gather,
eager and replayed from a CUDA graph (the epochs keep advancing across launches)."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.ll_exchange import TPGroupExchange, TPKeyExchange
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="needs SM90+ (PDL)",
    ),
]


def _mapping(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    MPI.COMM_WORLD.barrier()
    return Mapping(world_size=tp, tp_size=tp, rank=rank), rank


def _keys(m, seed, r):
    g = torch.Generator(device="cuda").manual_seed(seed * 31 + r)
    return torch.randint(-(2**62), 2**62, (m,), generator=g, device="cuda", dtype=torch.int64)


def _key_exchange(tp):
    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    try:
        # Consecutive launches of one channel with changing row counts (alternating slots).
        for step, m in enumerate((1, 64, 7, 512, 33)):
            all_keys = torch.stack([_keys(m, step, r) for r in range(tp)])
            gather = torch.empty_like(all_keys)
            max_out, _ = ex.exchange(all_keys[rank].clone(), gather_out=gather)
            torch.cuda.synchronize()
            assert torch.equal(max_out, all_keys.max(0).values), ("max", m)
            assert torch.equal(gather, all_keys), ("gather", m)

        # Greedy argmax over a vocabulary sharded by rank: ties go to the lowest global index.
        m, vocab = 16, 1000
        g = torch.Generator(device="cuda").manual_seed(5)
        logits = torch.randn((m, tp * vocab), generator=g, device="cuda")
        logits[0, :] = 0.0
        logits[0, 3::vocab] = 1.0  # one maximum per shard, all equal: token 3
        logits[1, vocab - 1] = logits[1, 2 * vocab - 1] = 50.0  # tied across the first two shards
        shard = logits[:, rank * vocab : (rank + 1) * vocab]
        vals, lidx = shard.max(-1)
        lidx = lidx.to(torch.int32)
        for _ in range(3):
            tokens = ex.argmax_tokens(vals, lidx, rank * vocab)
            torch.cuda.synchronize()
            ref = logits.argmax(-1).to(torch.int32)
            ref[0] = 3
            ref[1] = vocab - 1
            assert torch.equal(tokens, ref), (tokens.tolist(), ref.tolist())

        # CUDA-graph replays with fresh inputs copied into the captured buffers.
        m = 24
        keys = _keys(m, 100, rank)
        max_buf = torch.empty_like(keys)
        ex.exchange(keys, max_out=max_buf)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ex.exchange(keys, max_out=max_buf)
        for step in range(5):
            all_keys = torch.stack([_keys(m, 200 + step, r) for r in range(tp)])
            keys.copy_(all_keys[rank])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(max_buf, all_keys.max(0).values), ("graph", step)
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _group_allgather(tp):
    mapping, rank = _mapping(tp)
    group_size = 2
    ex = TPGroupExchange.create(mapping, group_size, max_bytes=64 * 4 * 256 * 2, max_merge_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    try:
        members = range(rank // group_size * group_size, (rank // group_size + 1) * group_size)
        for step, rows in enumerate((1, 9, 64, 3)):

            def x_of(r):
                g = torch.Generator(device="cuda").manual_seed(step * 17 + r)
                return torch.randn((rows, 4, 256), generator=g, device="cuda").to(torch.bfloat16)

            out = ex.allgather(x_of(rank))
            torch.cuda.synchronize()
            ref = torch.cat([x_of(r).reshape(rows, -1) for r in members], dim=-1)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), ("allgather", rows)
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _worker(fn, tp):
    try:
        with torch.inference_mode():
            fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True


CASES = {"key_exchange": _key_exchange, "group_allgather": _group_allgather}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_ll_exchange(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)
