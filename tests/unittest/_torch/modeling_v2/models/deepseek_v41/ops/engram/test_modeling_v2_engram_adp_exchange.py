# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank test of the Engram attention-DP exchange entries (`engram_adp_exchange_hashes`, `engram_adp_push`,
`engram_adp_receive`): four ranks, each holding six of the 24 heads of two FP8 tables (the family's
`ShardedFp8MultiHeadEmbedding`, head-sharded), exchange their hash rows, push their heads of every group token and
receive all heads of their own tokens. Gated bit for bit: the gathered hashes against every rank's hash rows, and the
received rows against the unsharded table's lookup of the rank's own hashes. Eager over three rounds (both parities of
the hash slots), with the pushes on a side stream as the targets run them, and replayed from a CUDA graph on new data.

Needs 4 visible GPUs of one tray; run under the repo's `mpi_pool_executor` fixture.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_adp_exchange import (
    engram_adp_exchange_create,
    engram_adp_exchange_hashes,
    engram_adp_push,
    engram_adp_receive,
    engram_adp_takes,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 4
        or torch.cuda.get_device_capability() != (10, 3),
        reason="needs 4 SM103 GPUs",
    ),
]

BLOCK = 32
CAPACITY = 160


def _buckets(heads, layer):
    return [53 + 7 * h + 3 * layer for h in range(heads)]


def _checkpoint_rows(buckets, dim, layer):
    gen = torch.Generator(device="cpu").manual_seed(100 + layer)
    rows = sum(buckets)
    weight = (torch.randn(rows, dim, generator=gen) * 4).to(torch.float8_e4m3fn)
    # scale codes include 0 (2^-127) and the extremes a checkpoint can hold
    scale = torch.randint(110, 140, (rows, dim // BLOCK), generator=gen, dtype=torch.uint8)
    scale[::97, 0] = 0
    return weight, scale.view(torch.float8_e8m0fnu)


def _table(buckets, dim, tp_size, tp_rank, rows):
    t = ShardedFp8MultiHeadEmbedding(
        list_of_N=buckets,
        D=dim,
        block_size=BLOCK,
        dtype=torch.bfloat16,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    t.to("cuda")
    t.load_weights({"weight": rows[0], "scale": rows[1]})
    return t


def _hashes(rank, count, buckets, layer, salt):
    gen = torch.Generator(device="cpu").manual_seed(1000 * salt + 10 * rank + layer)
    cols = [torch.randint(0, n, (count,), generator=gen) for n in buckets]
    return (
        torch.stack(cols, dim=1).to(torch.int64).cuda()
        if count
        else torch.empty(0, len(buckets), dtype=torch.int64, device="cuda")
    )


def _rank_body(world, counts, heads, dim, layers):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank, enable_attention_dp=True)
        buckets = [_buckets(heads, layer) for layer in range(layers)]
        rows = [_checkpoint_rows(buckets[layer], dim, layer) for layer in range(layers)]
        tables = [_table(buckets[layer], dim, world, rank, rows[layer]) for layer in range(layers)]
        assert all(t.shard_heads for t in tables)
        exchange = engram_adp_exchange_create(mapping, 0, world, CAPACITY, heads, dim, layers)
        assert exchange is not None, "no peer memory between the four GPUs"
        exchange.reference_tables = [
            _table(buckets[layer], dim, 1, 0, rows[layer]) for layer in range(layers)
        ]
        assert engram_adp_takes(exchange, list(counts)) and not engram_adp_takes(
            exchange, [CAPACITY + 1] * world
        )
        side = torch.cuda.Stream()

        def step(own):
            round_ = engram_adp_exchange_hashes(exchange, own, list(counts))
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for layer in range(layers):
                    engram_adp_push(exchange, round_, layer, tables[layer])
            torch.cuda.current_stream().wait_stream(side)
            return round_, [engram_adp_receive(exchange, round_, layer) for layer in range(layers)]

        def check(round_, got, salt):
            exchange.reference_hashes = [
                tuple(_hashes(r, counts[r], buckets[layer], layer, salt) for layer in range(layers))
                for r in range(world)
            ]
            want = engram_adp_exchange_hashes.reference(exchange, round_.hashes, list(counts))
            for r in range(world):
                assert torch.equal(round_.gathered[r, :, : counts[r]], want[r, :, : counts[r]]), (
                    f"hashes of rank {r}"
                )
            for layer in range(layers):
                ref = engram_adp_receive.reference(exchange, round_, layer)
                assert (
                    got[layer].shape == (counts[rank], heads * dim)
                    and got[layer].dtype == torch.bfloat16
                )
                assert torch.equal(got[layer], ref), f"rows of layer {layer}"

        for salt in range(3):
            own = [
                _hashes(rank, counts[rank], buckets[layer], layer, salt) for layer in range(layers)
            ]
            engram_adp_exchange_hashes.is_valid(exchange, own, list(counts))
            round_, got = step(own)
            for layer in range(layers):
                engram_adp_push.is_valid(exchange, round_, layer, tables[layer])
                engram_adp_receive.is_valid(exchange, round_, layer)
            torch.cuda.synchronize()
            check(round_, got, salt)

        # CUDA graph: static hash buffers, refilled before each replay.
        static = [_hashes(rank, counts[rank], buckets[layer], layer, 3) for layer in range(layers)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            round_, got = step(static)
        for salt in range(4, 7):
            for layer in range(layers):
                static[layer].copy_(_hashes(rank, counts[rank], buckets[layer], layer, salt))
            graph.replay()
            torch.cuda.synchronize()
            check(round_, got, salt)
    except Exception:
        traceback.print_exc()
        raise
    return True


_WORLDS = {c.spec["world"] for c in engram_adp_exchange_hashes.CELLS}
assert _WORLDS == {4}, _WORLDS


@pytest.mark.parametrize(
    "cell",
    engram_adp_exchange_hashes.CELLS,
    ids=[c.why[:40] for c in engram_adp_exchange_hashes.CELLS],
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_certified_cells(mpi_pool_executor, cell):
    """Hashes, pushes and receives of each cell (the push and receive entries share the hash entry's CELLS)."""
    s = cell.spec
    world = mpi_pool_executor.num_workers
    assert world == s["world"]
    args = (world, tuple(s["counts"]), s["heads"], s["dim"], s["layers"])
    results = mpi_pool_executor.map(_rank_body, *zip(*[args] * world))
    for r in results:
        assert r is True
