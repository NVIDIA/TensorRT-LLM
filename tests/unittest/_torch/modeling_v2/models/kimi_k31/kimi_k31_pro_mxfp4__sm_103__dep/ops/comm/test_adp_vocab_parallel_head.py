# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Kimi K3.1 attention-DP vocab-parallel LM head entries on one GB300 tray.

Each cell runs on four MPI ranks with their own row counts, on the exchange path it names (the NVLink exchange
kernels on the group's reserved workspace, or NCCL with the workspace set aside): every rank regenerates every
rank's rows and the head from the cell's seed, calls the entry with its rows and checks it bit for bit against the
replicated head. A sequence test runs both entries back to back at changing row counts, eagerly and replayed from one
CUDA graph (the exchanges' call counter lives on the device). A one-GPU sweep checks the property the entries rest on
at the DEP16 shapes: the shard GEMM over the 16 ranks' padded rows gives the replicated GEMM's columns bit for bit,
for every padded row count of the bar range.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm.adp_vocab_parallel_head import (  # noqa: E501
    DEFAULT_MAX_ROWS,
    k31_adp_vocab_logits,
    k31_adp_vocab_top32,
    reserve_exchange,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
VOCAB = 163840
HIDDEN = 7168
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (0.02 * torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
    rows = [
        (torch.randn(n, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
        for n in spec["rows"]
    ]
    return weight, rows


def _exact(got, ref) -> bool:
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    return all(
        g.dtype == r.dtype
        and g.shape == r.shape
        and torch.equal(
            g.view(torch.int32) if g.dtype == torch.float32 else g,
            r.view(torch.int32) if r.dtype == torch.float32 else r,
        )
        for g, r in zip(got, ref)
    )


def _setup(world: int):
    """This rank's mapping, with the group's exchange workspace reserved (collective: every rank calls it)."""
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    reserve_exchange(mapping, DEFAULT_MAX_ROWS)
    return rank, mapping


def _rank_cell(spec: dict, seed: int, which: str) -> dict:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm import (
        adp_vocab_parallel_head as head,
    )

    rank, mapping = _setup(spec["world"])
    weight, rows = _inputs(spec, seed)
    op = k31_adp_vocab_logits if which == "logits" else k31_adp_vocab_top32
    # An NCCL cell runs as a group without a workspace does: the reserved one is set aside for the call.
    aside = dict(head._EXCHANGES) if spec["exchange"] == "nccl" else {}
    for key in aside:
        del head._EXCHANGES[key]
    try:
        with torch.inference_mode():
            got = op(rows[rank], weight, mapping, spec["rows"])
            ref = op.reference(rows, weight, mapping, spec["rows"])
            torch.cuda.synchronize()
            exact = _exact(got, ref)
    finally:
        head._EXCHANGES.update(aside)
    return dict(rank=rank, exact=exact)


def _rank_sequence(seed: int) -> dict:
    """Both entries back to back at changing row counts on the exchange kernels: eagerly, then captured once in a
    CUDA graph and replayed on new rows (the call counter advances on the device)."""
    rank, mapping = _setup(WORLD)
    steps = [("logits", (40, 40, 40, 40)), ("top32", (35, 35, 35, 35)), ("logits", (12, 30, 2, 30))]
    bad = []
    with torch.inference_mode():
        for i, (which, counts) in enumerate(steps):
            op = k31_adp_vocab_logits if which == "logits" else k31_adp_vocab_top32
            weight, rows = _inputs(dict(rows=counts), seed + i)
            if not _exact(
                op(rows[rank], weight, mapping, counts), op.reference(rows, weight, mapping, counts)
            ):
                bad.append(f"eager {i} {which}")
        counts = (20, 20, 20, 20)
        weight, rows = _inputs(dict(rows=counts), seed + 7)
        x = rows[rank].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            k31_adp_vocab_logits(x, weight, mapping, counts)
            k31_adp_vocab_top32(x, weight, mapping, counts)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = k31_adp_vocab_logits(x, weight, mapping, counts)
            unary, ids = k31_adp_vocab_top32(x, weight, mapping, counts)
        for rep in range(3):
            _, rows = _inputs(dict(rows=counts), seed + 100 + rep)
            x.copy_(rows[rank])
            graph.replay()
            torch.cuda.synchronize()
            if not _exact(logits, k31_adp_vocab_logits.reference(rows, weight, mapping, counts)):
                bad.append(f"graph replay {rep} logits")
            if not _exact(
                (unary, ids), k31_adp_vocab_top32.reference(rows, weight, mapping, counts)
            ):
                bad.append(f"graph replay {rep} top32")
    return dict(rank=rank, bad=bad)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "which, cell",
    [("logits", c) for c in k31_adp_vocab_logits.CELLS]
    + [("top32", c) for c in k31_adp_vocab_top32.CELLS],
    ids=[c.why[:56] for c in k31_adp_vocab_logits.CELLS + k31_adp_vocab_top32.CELLS],
)
def test_certified_cells(which, cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed, which)] * WORLD))
    for r in results:
        assert r["exact"], f"rank {r['rank']}: {which} differ from the replicated head"


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_exchange_sequence_and_graph_replay(mpi_pool_executor) -> None:
    results = mpi_pool_executor.map(_rank_sequence, *zip(*[(5,)] * WORLD))
    for r in results:
        assert not r["bad"], f"rank {r['rank']}: {r['bad']}"


def test_dep16_shard_gemm_is_the_replicated_columns() -> None:
    """At world 16: F.linear(16 ranks' rows padded to R, shard) == F.linear(own rows, head)[:, shard], every padded
    count R up to 320 (64 requests x 5 rows) and owner row counts from 2 to R, three shards."""
    world = 16
    shard = VOCAB // world
    gen = torch.Generator(device="cuda").manual_seed(11)
    weight = (0.02 * torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
    bad = []
    with torch.inference_mode():
        for padded in range(2, 321):
            xall = torch.randn(world * padded, HIDDEN, generator=gen, device="cuda").to(
                torch.bfloat16
            )
            parts = [F.linear(xall, weight[r * shard : (r + 1) * shard]) for r in (0, 7, 15)]
            for owner, rows in ((0, padded), (9, max(2, padded - 3)), (15, 2)):
                own = xall[owner * padded : owner * padded + rows]
                full = F.linear(own, weight)
                for r, part in zip((0, 7, 15), parts):
                    got = part[owner * padded : owner * padded + rows]
                    if not torch.equal(got, full[:, r * shard : (r + 1) * shard]):
                        bad.append((padded, owner, rows, r))
    assert not bad, f"shard GEMM columns differ from the replicated GEMM at {bad[:8]}"


def test_one_row_is_refused() -> None:
    from tensorrt_llm.mapping import Mapping

    weight = torch.zeros(64, 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="one-row"):
        k31_adp_vocab_logits.is_valid(
            torch.zeros(1, 8, dtype=torch.bfloat16, device="cuda"),
            weight,
            Mapping(world_size=1, tp_size=1, rank=0),
            [1],
        )
