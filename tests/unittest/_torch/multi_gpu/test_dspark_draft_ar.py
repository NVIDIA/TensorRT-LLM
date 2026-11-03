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
"""DSpark drafter all-reduce site: the owner-multicast embedding site is bitwise the vocab-sharded
lookup + all-reduce, eagerly and under CUDA-graph replay."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168


def _differ(ref, out):
    return int((ref.view(torch.int16) != out.view(torch.int16)).sum())


def _run_rank_embed(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.modules.embedding import Embedding
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.speculative.dspark_ar_sites import draft_embed_site
    from tensorrt_llm._torch.speculative.spec_tail_kernels import vocab_shard_embedding
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        vocab = 1000 * world_size - 3  # a shorter last shard
        embedding = Embedding(
            vocab,
            HIDDEN,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        ).cuda()
        gen = torch.Generator(device="cuda").manual_seed(9)
        full = torch.randn(vocab, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        full.view(torch.int16)[:, ::7] = -32768  # -0 lanes: +0 after the all-reduce
        start, end = embedding.vocab_start_index, embedding.vocab_end_index
        embedding.weight.data[: end - start].copy_(full[start:end])
        bad = 0
        for n in (1, 2, 9, 17, 129):
            for dtype in (torch.int64, torch.int32):
                ids = torch.randint(0, vocab, (n,), device="cuda", generator=gen).to(dtype)
                ids[0] = vocab - 1
                if n > 2:
                    ids[1] = -1  # outside the vocab: a zero row
                    ids[2] = vocab
                expected = vocab_shard_embedding(embedding, ids)
                out = draft_embed_site(embedding, ids)
                assert out is not None
                bad += _differ(expected, out)
        # CUDA graph replay.
        ids = torch.randint(0, vocab, (9,), device="cuda", generator=gen)
        draft_embed_site(embedding, ids)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = draft_embed_site(embedding, ids)
        for rep in range(3):
            ids.copy_(torch.randint(0, vocab, (9,), device="cuda", generator=gen))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            bad += _differ(vocab_shard_embedding(embedding, ids), out)
        del graph
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


_needs = pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability()[0] != 10
    or os.environ.get("TLLM_TEST_SKIP_MNNVL") == "1",
    reason="needs two SM100-family GPUs with MNNVL",
)
_WORLD_SIZES = [2, 4] if torch.cuda.device_count() >= 4 else [2]


@_needs
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_dspark_draft_embed_site(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    for r in mpi_pool_executor.map(_run_rank_embed, *zip(*[(world,)] * world)):
        assert r is True
