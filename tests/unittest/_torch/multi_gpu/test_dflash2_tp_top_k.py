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
"""Real-collective test for the DFlash 2 TP shard reduction.

``tests/unittest/_torch/speculative/hw_agnostic/test_dflash2_semantics.py``
replays ``DFlashWorker._dflash2_global_top_k`` against a mocked ``allgather``.
What that cannot show is the two properties the reduction borrows from the
collective itself: rank-order concatenation along ``dim=-1`` (so a gathered
id's shard offset is the one it was tagged with) and the interleaved
``(id, value)`` pair parity surviving the concatenation (each rank contributes
an even ``2 * top_k`` chunk). This runs the same reduction across MPI ranks
over NCCL and checks that every rank lands on the unsharded top-k.
"""

import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)

NUM_GENS, DRAFT_LEN, VOCAB = 3, 5, 512


def run_single_rank(tensor_parallel_size, single_rank_forward_func, *args):
    """Wrapper used by MPIPoolExecutor; matches test_allgather.py."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(tensor_parallel_size, rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _full_logits(top_k: int, tp_size: int, skew: str) -> torch.Tensor:
    """Block logits every rank agrees on (seeded on the host), optionally
    arranged so the global top-k concentrates in one shard.

    ``skew="one_shard"`` plants the winners inside the last rank's shard so
    that rank 0 contributes nothing to the result and the gathered order has
    to be read from the ids, not the concatenation position.
    """
    generator = torch.Generator().manual_seed(2024)
    logits = torch.randn(NUM_GENS, DRAFT_LEN, VOCAB, generator=generator)
    if skew == "one_shard":
        shard = VOCAB // tp_size
        planted = torch.arange(VOCAB - shard, VOCAB - shard + top_k)
        logits[..., planted] += 100.0
    return logits


@torch.inference_mode()
def run_global_top_k(tp_size: int, tp_rank: int, top_k: int, skew: str):
    mapping = Mapping(world_size=tp_size, rank=tp_rank, tp_size=tp_size)
    worker = DFlashWorker.__new__(DFlashWorker)
    worker.mapping = mapping
    worker._d2t = None

    full_logits = _full_logits(top_k, tp_size, skew).to("cuda")
    shard = VOCAB // tp_size
    local_logits = full_logits[..., tp_rank * shard : (tp_rank + 1) * shard].contiguous()
    spec_metadata = SimpleNamespace(draft_vocab_size=VOCAB, vocab_size=VOCAB)

    candidate_ids, unary_logits, block_logits = worker._dflash2_global_top_k(
        local_logits, spec_metadata, top_k=top_k, full_vocab=VOCAB
    )

    expected_unary, expected_ids = torch.topk(full_logits, top_k, dim=-1)
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_unary)
    assert candidate_ids.dtype == torch.long
    assert block_logits.shape == (NUM_GENS, DRAFT_LEN, VOCAB)
    assert (block_logits == float("-inf")).all()


@pytest.mark.parametrize("skew", ["random", "one_shard"])
@pytest.mark.parametrize("top_k", [3, 4], ids=lambda k: f"top_k:{k}")
@pytest.mark.parametrize("world_size", [2], ids=lambda x: f"world:{x}")
def test_dflash2_global_top_k_matches_unsharded_across_ranks(world_size, top_k, skew):
    """Every TP rank reduces its vocab shard to the same global top-k the
    unsharded path picks. Odd and even top_k cover the pair-parity of the
    interleaved gather."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_global_top_k, top_k, skew)] * world_size),
        )
        for r in results:
            assert r is True
