# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moe_front entry on one GB300 tray (four MPI ranks).

Each cell runs on four ranks: every rank builds the same unsharded weights from the cell's seed, packs its slice
(``front_weight``), calls the entry on the shared input through the TP group's workspace and gates the result
against the entry's reference (the replicated router GEMM + latent_down + ``kimi_k3_noaux_tc_mxfp8_quant`` [+ the
shared gate_up + SiTU]). The top-16 sets must match, the weights agree where the order does, at most 1% of the
MXFP8 bytes / scales may differ (the latent's last bit), the SiTU within BF16-level error; every rank must get the
same bytes.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import os
import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_front import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    moe_front,
)

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
#: The routed latent width (Kimi K3.1).
LATENT = 3584
GATE_CAP, LINEAR_CAP = 4.0, 25.0
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3),)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


def _weights(seed: int, world: int, shared_cols: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    down = (torch.randn(LATENT, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = (torch.rand(NUM_EXPERTS, device="cuda", generator=gen) - 0.5) * 0.19
    shards = None
    if shared_cols:
        shards = [
            (torch.randn(2 * shared_cols, HIDDEN, device="cuda", generator=gen) * 0.02).to(
                torch.bfloat16
            )
            for _ in range(world)
        ]
    return down, router, bias, shards


def _rank_cell(spec: dict, seed: int) -> dict:
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe import (
        moe_front as op_mod,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    world, rows, shared_cols = spec["world"], spec["rows"], spec["shared_cols"]
    registry = vars(
        sys.modules.setdefault("_k31_front_ws", types.ModuleType("_k31_front_ws"))
    ).setdefault("by_world", {})
    if world not in registry:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank, gpus_per_node=world)
        registry[world] = op_mod.front_workspace(mapping)
    ws = registry[world]
    down, router, bias, shards = _weights(seed, world, shared_cols)
    shard = shards[rank] if shards else None
    weight = op_mod.front_weight(down, router, rank, world, shard)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    x = torch.randn(rows, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    args = (x, weight, bias, spec["scale"], ws, shared_cols, GATE_CAP, LINEAR_CAP)
    with validating(moe_front):
        got = moe_front(*args)
    again = moe_front(*args)
    ref = moe_front.reference(x, down, router, bias, spec["scale"], shard, GATE_CAP, LINEAR_CAP)
    torch.cuda.synchronize()
    ids, weights, codes, scales, shared = got
    r_ids, r_weights, r_codes, r_scales, r_shared = ref
    set_diff = sum(len(set(ids[t].tolist()) ^ set(r_ids[t].tolist())) for t in range(rows))
    same = ids == r_ids
    weight_bad = int(((weights.float() - r_weights.float()).abs() > 1e-2)[same].sum())
    soft = int((codes.view(torch.uint8) != r_codes.view(torch.uint8)).sum())
    soft += int((scales.reshape(-1) != r_scales.reshape(-1)).sum())
    situ_err = 0.0
    if shared_cols:
        situ_err = (shared.float() - r_shared.float()).abs().max().item() / max(
            r_shared.float().abs().max().item(), 1e-6
        )
    blob = torch.cat([t.reshape(-1).view(torch.uint8) for t in got[:4]])
    digest = int(zlib.crc32(blob.cpu().numpy().tobytes()))
    repeat = all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(got[:4], again[:4])
    )
    return dict(
        rank=rank,
        set_diff=set_diff,
        weight_bad=weight_bad,
        soft=soft,
        elems=codes.numel(),
        situ_err=situ_err,
        hash=digest,
        repeat=repeat,
        shapes=(tuple(ids.shape), tuple(weights.shape), tuple(codes.shape), tuple(scales.shape)),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", moe_front.CELLS, ids=[c.why[:48] for c in moe_front.CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)))
    rows = spec["rows"]
    for r in results:
        assert r["shapes"] == ((rows, 16), (rows, 16), (rows, LATENT), (rows, LATENT // 32)), r
        assert r["set_diff"] == 0 and r["weight_bad"] == 0, r
        assert r["soft"] <= r["elems"] // 100, r
        assert r["situ_err"] <= 2e-2, r
        assert r["repeat"], r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different bytes"
    print(f"{cell.why}: MXFP8 bytes / scales off the replicated path: {results[0]['soft']}")


def _rank_runner_sort(world: int, local: int, tile: int, seed: int) -> dict:
    """One rank: ``moe_front_runner_sort`` for the expert window of this rank's EP group (``local`` experts,
    EP groups interleaved over the TP ranks as the engine maps MoE TP x EP) against ``moe_front`` on the same
    input and the standalone runner sort of its ids, at every row count the front's sort serves."""
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.runner_sort import (  # noqa: E501
        runner_sort,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe import (
        moe_front as op_mod,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    registry = vars(
        sys.modules.setdefault("_k31_front_ws", types.ModuleType("_k31_front_ws"))
    ).setdefault("by_world", {})
    if world not in registry:
        mapping = Mapping(
            world_size=world, tp_size=world, rank=rank, gpus_per_node=torch.cuda.device_count()
        )
        registry[world] = op_mod.front_workspace(mapping)
    ws = registry[world]
    down, router, bias, _ = _weights(seed, world, 0)
    weight = op_mod.front_weight(down, router, rank, world, None)
    offset = (rank % (NUM_EXPERTS // local)) * local
    bad = []
    for rows in (1, 2, 8, 9, 16, 17, 20, 24, 30, 32):
        gen = torch.Generator(device="cuda").manual_seed(seed + rows)
        x = torch.randn(rows, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        assert op_mod.runner_sort_fits(rows, local, tile)
        got = op_mod.moe_front_runner_sort(
            x, weight, bias, 2.5, ws, 0, GATE_CAP, LINEAR_CAP, tile, offset, local
        )
        plain = moe_front(x, weight, bias, 2.5, ws, 0, GATE_CAP, LINEAR_CAP)
        ref = runner_sort(got[0], offset, local, tile)
        torch.cuda.synchronize()
        e2p, p2t, total, t2e, t2l, nt = got[5]
        r_e2p, r_p2t, r_total, r_t2e, r_t2l, r_nt = ref
        tiles, padded = int(r_nt[0]), int(r_total[0])
        same = all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(got[:4], plain[:4])
        )
        same &= torch.equal(e2p, r_e2p) and int(nt[0]) == tiles and int(total[0]) == padded
        same &= torch.equal(t2e[:tiles], r_t2e[:tiles]) and torch.equal(t2l[:tiles], r_t2l[:tiles])
        rows_of_tiles = r_p2t[:padded] >= 0  # the tiles' padding rows are never read
        same &= torch.equal(p2t[:padded][rows_of_tiles], r_p2t[:padded][rows_of_tiles])
        if not same:
            bad.append(rows)
    return dict(rank=rank, offset=offset, bad=bad)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "local", [112, 224], ids=["ep8-112-local-experts", "ep4-224-local-experts"]
)
def test_runner_sort_is_the_standalone_sort(local, mpi_pool_executor) -> None:
    """The front's runner sort (the routed experts' prerouted layout at <= 32 rows) is the standalone runner sort
    of the ids it returns, for the EP8 window (112 experts: TEP8, TEP32) and the EP4 window (224 experts: TEP8 MoE
    TP2 x EP4, TEP16), and leaves the front's other outputs as they are. The sort reads only the gathered ids, the
    same on every width, so one tray's four ranks cover the deployed widths."""
    results = list(mpi_pool_executor.map(_rank_runner_sort, *zip(*[(WORLD, local, 8, 23)] * WORLD)))
    for r in results:
        assert not r["bad"], r


def test_guard_refuses_too_many_rows() -> None:
    ws = dict(world=WORLD)
    x = torch.zeros(33, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(1024, HIDDEN, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="M <="):
        moe_front.is_valid(x, w, bias, 1.0, ws, 0, 1.0, 1.0)


def test_guard_refuses_shared_columns_past_16_rows() -> None:
    ws = dict(world=WORLD)
    x = torch.zeros(24, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(4096, HIDDEN, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="shared columns"):
        moe_front.is_valid(x, w, bias, 1.0, ws, 1536, 1.0, 1.0)
