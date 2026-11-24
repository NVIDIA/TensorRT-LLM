# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the fused latent-MoE tail's normalized latent is bitwise the MNNVL all-reduce RMSNorm's, TP 8 split.

The unfused path reduces the routed latent through the engine ``AllReduce`` module, which on a group
spanning GB300 trays runs the MNNVL all-reduce with the RMSNorm fused in. A tail built with
``mnnvl_norm`` must give the same bits. Each cell runs on four MPI ranks (one tray) with the MNNVL
strategy requested explicitly, and the MNNVL one-shot threshold set so that rows <= 18 take the
one-shot kernel and longer rows the two-shot kernel, the split a TP 8 group has by default (the TP 16
split is ``test_latent_moe_tail_mnnvl_norm.py``; the one-shot threshold is read once per process). The
rows cover the tail's one-shot collective against both MNNVL kernels (1..32) and its two-shot
collective up to the TP 8 dispatch limit (33..128).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import os
import pickle
import sys
import types

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.latent_moe_tail import (  # noqa: E501
    EPS,
    HIDDEN,
    LATENT,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
# The one-shot / two-shot split of a TP 8 group at the default 1 MiB threshold: 18 rows.
ONESHOT_ROWS = 18
THRESHOLD_BYTES = ONESHOT_ROWS * LATENT * WORLD * 2
ROWS = [*range(1, 33), 33, 36, 37, 38, 48, 64, 96, 128]

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not IS_CUTLASS_DSL_AVAILABLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3),)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


def _rank_rows(rows: list, seed: int) -> dict:
    """One rank: the MNNVL all-reduce RMSNorm and the tail's latent for every row count."""
    # Before the first MNNVL call: the native op reads the threshold once.
    os.environ["TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES"] = str(THRESHOLD_BYTES)
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_latent import (  # noqa: E501
        KimiK3LatentMoETail,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (  # noqa: E501
        allreduce_module,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
    state = vars(
        sys.modules.setdefault("_k31_mnnvl_norm_tp8", types.ModuleType("_k31_mnnvl_norm_tp8"))
    )
    if "module" not in state:
        state["module"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        state["tail"] = KimiK3LatentMoETail.get(
            mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=EPS, mnnvl_norm=True
        )
        state["plain"] = KimiK3LatentMoETail.get(
            mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=EPS, mnnvl_norm=False
        )
    module, tail, plain = state["module"], state["tail"], state["plain"]
    assert module.mnnvl_allreduce is not None, "the MNNVL strategy did not build the MNNVL path"

    gen = torch.Generator(device="cuda").manual_seed(seed)
    rms_weight = (1 + 0.1 * torch.randn(LATENT, generator=gen, device="cuda")).to(torch.bfloat16)
    up_weight = (torch.randn(HIDDEN, LATENT, generator=gen, device="cuda") * LATENT**-0.5).to(
        torch.bfloat16
    )
    report = {}
    kernels = {}
    with torch.inference_mode():
        # Which MNNVL kernels the threshold sends the last one-shot and the first two-shot row count to.
        for m in (ONESHOT_ROWS, ONESHOT_ROWS + 1):
            probe = torch.randn(m, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                allreduce_module(probe, module, int(AllReduceFusionOp.RMS_NORM), rms_weight, EPS)
                torch.cuda.synchronize()
            kernels[m] = sorted(
                {
                    name
                    for e in prof.events()
                    for name in ("oneshotAllreduceFusionKernel", "rmsNormLamport")
                    if name in e.name
                }
            )
        for m in rows:
            routed = [
                torch.randn(m, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
                for _ in range(WORLD)
            ]
            shared = [
                (torch.randn(m, HIDDEN, generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
                for _ in range(WORLD)
            ]
            general = allreduce_module(
                routed[rank], module, int(AllReduceFusionOp.RMS_NORM), rms_weight, EPS
            ).clone()
            mine = (routed[rank], shared[rank], rms_weight, up_weight)
            fused = tail._forward(*mine, stages=1).clone()
            again = tail._forward(*mine, stages=1).clone()
            unmatched = plain._forward(*mine, stages=1).clone()
            torch.cuda.synchronize()
            bits = general.view(torch.int16)
            report[m] = dict(
                differ=int((fused.view(torch.int16) != bits).sum()),
                repeat_equal=bool(torch.equal(fused.view(torch.int16), again.view(torch.int16))),
                plain_differ=int((unmatched.view(torch.int16) != bits).sum()),
            )
    return dict(rank=rank, rows=report, kernels=kernels)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("seed", [3, 17])
def test_latent_is_bitwise_to_mnnvl_allreduce_rmsnorm(seed, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_rows, *zip(*[(ROWS, seed)] * WORLD)))
    for r in sorted(results, key=lambda r: r["rank"]):
        assert r["kernels"] == {
            ONESHOT_ROWS: ["oneshotAllreduceFusionKernel"],
            ONESHOT_ROWS + 1: ["rmsNormLamport"],
        }, r["kernels"]
        for m, row in r["rows"].items():
            print(
                f"rank {r['rank']} rows {m}: {row['differ']} elements differ "
                f"(tail without mnnvl_norm: {row['plain_differ']})"
            )
            assert row["repeat_equal"], f"rank {r['rank']} rows {m}: calls disagree"
            assert row["differ"] == 0, f"rank {r['rank']} rows {m}: {row}"
