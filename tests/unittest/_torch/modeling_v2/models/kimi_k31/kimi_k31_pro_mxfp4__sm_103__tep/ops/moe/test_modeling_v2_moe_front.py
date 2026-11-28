# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moe_front entry on one GB300 tray (four MPI ranks).

Each cell (a sample of the entry's, ``sample_cells``) runs on four ranks: every rank builds the same unsharded
weights from the cell's seed, packs its slice (``front_weight``), calls the entry on the shared input through the TP
group's workspace and gates the result against the entry's reference (the replicated router GEMM + latent_down +
``kimi_k3_noaux_tc_mxfp8_quant`` [+ the shared gate_up + SiTU]). The top-16 sets must match, the weights agree where
the order does, at most 1% of the MXFP8 bytes / scales may differ (the latent's last bit), the SiTU within BF16-level
error.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import sys
import types
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_front import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    moe_front,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)
from _validating import validating  # noqa: E402 — needs the path declared above

pickle_by_value(__name__)

WORLD = 4
#: The routed latent width (Kimi K3.1).
LATENT = 3584
GATE_CAP, LINEAR_CAP = 4.0, 25.0
pytestmark = on_ranks(WORLD, 103)


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
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe import (
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
    return dict(
        rank=rank,
        set_diff=set_diff,
        weight_bad=weight_bad,
        soft=soft,
        elems=codes.numel(),
        situ_err=situ_err,
        shapes=(tuple(ids.shape), tuple(weights.shape), tuple(codes.shape), tuple(scales.shape)),
    )


CELLS = sample_cells(moe_front.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, world=WORLD)
    rows = spec["rows"]
    for r in results:
        assert r["shapes"] == ((rows, 16), (rows, 16), (rows, LATENT), (rows, LATENT // 32)), r
        assert r["set_diff"] == 0 and r["weight_bad"] == 0, r
        assert r["soft"] <= r["elems"] // 100, r
        assert r["situ_err"] <= 2e-2, r
