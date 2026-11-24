# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the two-shot-only fused latent-MoE tail is bitwise the unfused MoE tail.

At attention TP 32 a second tail instance (``twoshot_only``) serves 33..192 rows. Its output must be the
unfused path's at those rows bit for bit: the routed partial (finalized, or the routed experts' unfinalized
rows finalized by the routed-expert finalize kernel) through the engine ``AllReduce`` with the RMSNorm fused in
(MNNVL two-shot), the cuBLAS latent up projection (``k31_fused_a_gemm``), the shared partial through the
engine ``AllReduce`` (MNNVL two-shot), and their sum added to the residual (``k31_moe_residual_add``).

The cell runs a one-tray stand-in on four MPI ranks: the instance's rows at TP 4, the MNNVL one-shot threshold
set so that these rows take the two-shot kernels (as at TP 32), and the 32-rank kernel's polled sums (two
ranks per wait instead of 16). The 16 + 16 rank chunks of a 32-rank group need eight trays.

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
# Rows above 9 take the MNNVL two-shot kernels (the TP 16 split; every row of the instance is two-shot).
THRESHOLD_BYTES = 9 * LATENT * WORLD * 2
ROWS = [33, 40, 64, 65, 100, 128, 129, 192]
CASES = ["random", "unfinalized", "cancel", "zeros"]

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


def _partials(rank: int, m: int, case: str, gen: torch.Generator, common: torch.Generator):
    """This rank's (routed, shared) partials."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_routed_output import (  # noqa: E501
        UnfinalizedRoutedOutput,
    )

    shared = (torch.randn(m, HIDDEN, generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
    if case == "random":
        return torch.randn(m, LATENT, generator=gen, device="cuda").to(torch.bfloat16), shared
    if case == "unfinalized":
        rows = m * 16
        y = torch.randn(rows, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
        weights = (torch.rand(m, 16, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        local = torch.rand(m, 16, generator=gen, device="cuda") < 0.25
        local[::5] = False  # rows with no local expert
        perm = torch.randperm(rows, generator=gen, device="cuda").view(m, 16).to(torch.int32)
        e2p = torch.where(local, perm, torch.full_like(perm, -1))
        return UnfinalizedRoutedOutput(y, weights.contiguous(), e2p.contiguous()), shared
    if case == "cancel":
        # Ranks 0..1 and 2..3 hold opposite rows (every other row also a small per-rank term).
        sign = 1.0 if rank < WORLD // 2 else -1.0
        x = torch.randn(m, LATENT, generator=common, device="cuda") * 100
        xs = torch.randn(m, HIDDEN, generator=common, device="cuda") * 100
        routed = x * sign
        routed[1::2] += torch.randn(m, LATENT, generator=gen, device="cuda")[1::2]
        return routed.to(torch.bfloat16), (xs * sign).to(torch.bfloat16)
    routed = torch.randn(m, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
    routed[::2] = 0
    shared[::3] = 0
    return routed, shared


def _rank_rows(rows: list, seed: int) -> dict:
    """One rank: the unfused tail and the two-shot-only tail for every row count and case."""
    # Before the first MNNVL call: the native op reads the threshold once.
    os.environ["TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES"] = str(THRESHOLD_BYTES)
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E501
        run_moe_finalize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_tail_latent as lmt,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_tail_twoshot_collective as tsc,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_latent import (  # noqa: E501
        KimiK3LatentMoETail,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_routed_output import (  # noqa: E501
        UnfinalizedRoutedOutput,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (  # noqa: E501
        allreduce_module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (  # noqa: E501
        k31_fused_a_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (  # noqa: E501
        k31_moe_residual_add,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
    state = vars(sys.modules.setdefault("_k31_twoshot_only", types.ModuleType("_k31_twoshot_only")))
    if "tail" not in state:
        state["routed_ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        state["shared_ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        # The TP 32 instance's rows and polled sums on this tray; restored once the instance is built (its
        # compiled kernels keep the poll width in their cache key).
        poll = tsc.POLL_RANKS
        lmt._TWOSHOT_ONLY_ROWS[WORLD] = lmt._TWOSHOT_ONLY_ROWS[32]
        tsc.POLL_RANKS = 2
        try:
            state["tail"] = KimiK3LatentMoETail.get(
                mapping,
                hidden_size=HIDDEN,
                latent_size=LATENT,
                rms_eps=EPS,
                fold_finalize=True,
                twoshot_only=True,
            )
        finally:
            tsc.POLL_RANKS = poll
            del lmt._TWOSHOT_ONLY_ROWS[WORLD]
    routed_ar, shared_ar, tail = state["routed_ar"], state["shared_ar"], state["tail"]
    assert routed_ar.mnnvl_allreduce is not None, "the MNNVL strategy did not build the MNNVL path"
    assert tail._twoshot_collective_topk is not None and tail._collective is None

    gen = torch.Generator(device="cuda").manual_seed(seed * 100 + rank)
    common = torch.Generator(device="cuda").manual_seed(seed)
    rms_weight = (1 + 0.1 * torch.randn(LATENT, generator=common, device="cuda")).to(torch.bfloat16)
    up_weight = (torch.randn(HIDDEN, LATENT, generator=common, device="cuda") * LATENT**-0.5).to(
        torch.bfloat16
    )
    report = {}
    with torch.inference_mode():
        for m in rows:
            assert tail.can_run(m) and not tail.can_run(32)
            for case in CASES:
                routed, shared = _partials(rank, m, case, gen, common)
                prefix = (torch.randn(m, HIDDEN, generator=common, device="cuda") * 2).to(
                    torch.bfloat16
                )
                finalized = routed
                if isinstance(routed, UnfinalizedRoutedOutput):
                    finalized = torch.empty(m, LATENT, dtype=torch.bfloat16, device="cuda")
                    run_moe_finalize(
                        routed.gemm2_permuted,
                        routed.expanded_idx_to_permuted_idx,
                        routed.expert_weights,
                        finalized,
                    )
                latent = allreduce_module(
                    finalized, routed_ar, int(AllReduceFusionOp.RMS_NORM), rms_weight, EPS
                )
                up = k31_fused_a_gemm(latent, up_weight.t())
                shared_sum = allreduce_module(
                    shared, shared_ar, int(AllReduceFusionOp.NONE), None, None
                )
                layer = k31_moe_residual_add(prefix, up, shared_sum)
                moe = up + shared_sum

                out = tail(routed, shared, rms_weight, up_weight).clone()
                mailbox = tail.forward_to_mailbox(routed, shared, rms_weight, up_weight)
                settled = prefix + tail.copy_out(mailbox, m)
                tail.push_shared(shared)
                pushed = tail(routed, shared, rms_weight, up_weight, shared_pushed=True).clone()
                torch.cuda.synchronize()
                report[(m, case)] = dict(
                    output=int((out.view(torch.int16) != moe.view(torch.int16)).sum()),
                    layer=int((settled.view(torch.int16) != layer.view(torch.int16)).sum()),
                    pushed=int((pushed.view(torch.int16) != moe.view(torch.int16)).sum()),
                )
    return dict(rank=rank, rows=report)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("seed", [5, 23])
def test_twoshot_only_tail_is_bitwise_to_unfused_tail(seed, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_rows, *zip(*[(ROWS, seed)] * WORLD)))
    for r in sorted(results, key=lambda r: r["rank"]):
        for (m, case), row in r["rows"].items():
            assert row == dict(output=0, layer=0, pushed=0), (
                f"rank {r['rank']} rows {m} {case}: {row}"
            )
