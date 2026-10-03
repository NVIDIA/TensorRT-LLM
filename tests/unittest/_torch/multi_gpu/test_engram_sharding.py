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
"""Compare real head/row shard collectives with an unsharded Engram table."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams, AllReduceStrategy, allgather
from tensorrt_llm._torch.modules.engram import ShardedFp8MultiHeadEmbedding
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MPIPoolExecutor retains a worker thread after its first use.
pytestmark = pytest.mark.threadleak(enabled=False)

D = 128
BLOCK_SIZE = 32
NUM_TOKENS = 29

# 24 heads selects head sharding at TP4/8; 18 exercises the row-sharding fallback.
HEAD_COUNTS = (24, 18)


def _buckets(num_heads: int) -> list:
    """Unequal buckets expose incorrect head offsets."""
    return [16 * (1 + (7 * j) % 11) for j in range(num_heads)]


def _checkpoint(num_heads: int, seed: int = 11) -> dict:
    """Generate identical finite FP8 weights and UE8M0 scales on every rank."""
    gen = torch.Generator().manual_seed(seed)
    rows = sum(_buckets(num_heads))
    magnitude = torch.randint(0, 127, (rows, D), dtype=torch.uint8, generator=gen)
    sign = torch.randint(0, 2, (rows, D), dtype=torch.uint8, generator=gen) << 7
    scale = torch.randint(120, 135, (rows, D // BLOCK_SIZE), dtype=torch.uint8, generator=gen)
    return {
        "weight": (magnitude | sign).view(torch.float8_e4m3fn),
        "scale": scale.view(torch.float8_e8m0fnu),
    }


def _indices(num_heads: int, seed: int = 5) -> torch.Tensor:
    """Generate bucket-local indices matching the hash provider's bounds."""
    gen = torch.Generator().manual_seed(seed)
    cols = [
        torch.randint(0, n, (NUM_TOKENS, 1), dtype=torch.long, generator=gen)
        for n in _buckets(num_heads)
    ]
    return torch.cat(cols, dim=1).cuda()


def _table(num_heads: int, tp_size: int, tp_rank: int) -> ShardedFp8MultiHeadEmbedding:
    with torch.device("cuda"):
        table = ShardedFp8MultiHeadEmbedding(
            list_of_N=_buckets(num_heads),
            D=D,
            block_size=BLOCK_SIZE,
            dtype=torch.bfloat16,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
    table.load_weights(_checkpoint(num_heads))
    assert table.weight.device.type == table.scale.device.type == "cpu"
    assert table.weight.is_pinned() and table.scale.is_pinned()
    assert table.offsets.is_cuda
    return table


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


@torch.inference_mode()
def run_engram_shard_roundtrip(tp_size: int, tp_rank: int, num_heads: int):
    """Shard, look up, combine -- and land back on the unsharded answer."""
    mapping = Mapping(world_size=tp_size, rank=tp_rank, tp_size=tp_size)
    table = _table(num_heads, tp_size, tp_rank)
    indices = _indices(num_heads)

    expect_head_cut = num_heads % tp_size == 0
    assert table.shard_heads == expect_head_cut, (
        f"{num_heads} heads over {tp_size} ranks should have taken the "
        f"{'head' if expect_head_cut else 'row'} cut"
    )

    # Reassemble the flattened head-major lookup as the model does at consumption.
    local = table(indices).flatten(start_dim=-2)
    if table.shard_heads:
        assert local.shape == (NUM_TOKENS, num_heads // tp_size * D), (
            f"a head-sharded rank must contribute only its own columns, got {tuple(local.shape)}"
        )
        combined = allgather(local, mapping, dim=-1)
    else:
        assert local.shape == (NUM_TOKENS, num_heads * D), (
            f"a row-sharded rank must contribute full width, got {tuple(local.shape)}"
        )
        # Pin NCCL to test row-shard reconstruction independently of tactic selection.
        combined = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)(
            local, all_reduce_params=AllReduceParams(enable_allreduce=True)
        )

    reference_table = _table(num_heads, tp_size=1, tp_rank=0)
    reference = reference_table(indices).flatten(start_dim=-2)
    # Keep host shards alive until their asynchronous UVA reads finish.
    torch.cuda.synchronize()
    assert reference.count_nonzero() > 0, "degenerate reference table"
    torch.testing.assert_close(combined, reference, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS, ids=lambda n: f"heads:{n}")
@pytest.mark.parametrize("world_size", [4, 8], ids=lambda w: f"world:{w}")
def test_engram_shard_roundtrip(world_size, num_heads):
    """Head all-gather and row all-reduce must exactly reconstruct the full lookup."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_engram_shard_roundtrip, num_heads)] * world_size),
        )
        for r in results:
            assert r is True
