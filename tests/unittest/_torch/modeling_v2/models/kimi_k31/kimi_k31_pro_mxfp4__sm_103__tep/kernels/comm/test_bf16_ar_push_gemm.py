# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the tep target's push GEMM (``bf16_ar_push_gemm``) and its TP32 o_proj op on one GB300 tray
(four MPI ranks).

``bf16_ar_push_gemm`` claims that its push epilogue + the family's Lamport finish
(``lamport_ar``) equal the engine all-reduce of FlashInfer's split-K GEMM at the same tactic; the
``oproj_allreduce_push`` op claims cuBLAS (``F.linear``) + the engine all-reduce for the TP32 o_proj
rows above 48. Both are checked bit for bit here, over row counts that change the transport and the
GEMM tactic, interleaved with engine calls on the same workspace (the rotation state is shared). The
ranks report hashes, so every rank must hold the same tensor.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import types

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the push chain kernels are CuTe DSL kernels", allow_module_level=True)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
HIDDEN = 7168
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or torch.cuda.device_count() < WORLD,
        reason="one GB300 tray: sm_100 / sm_103 and four GPUs",
    ),
]


def _module(world: int):
    """The engine AllReduce module (MNNVL forced, one tray), cached per worker process."""
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_ar_push_kernels", types.ModuleType("_k31_ar_push")))
    if "ar" not in reg:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank, gpus_per_node=4)
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    assert ar.mnnvl_allreduce is not None, "MNNVL was not built on this tray"
    return rank, ar


def _digest(t: torch.Tensor) -> int:
    bits = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    w = torch.arange(1, bits.numel() + 1, device=t.device, dtype=torch.int64) % 65521
    return int((bits * w).sum())


def _rank_push_gemm(rows: int, k: int, seed: int) -> dict:
    """Push GEMM + finish vs the engine all-reduce of FlashInfer's split-K GEMM at the same tactic."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        lamport_ar as lam,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.comm import (  # noqa: E501
        bf16_ar_push_gemm as pg,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        oproj_allreduce_push as opm,
    )

    rank, ar = _module(WORLD)
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    x[0, :8] = 0
    w = (torch.randn(HIDDEN, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    tactic = pg.tactic_for(rows, HIDDEN, k)
    assert tactic is not None, (rows, k)
    # The test drives both transports: grow for both, then take the (final) view.
    opm.lamport_workspace(ar, rows, HIDDEN, True)
    ws = opm.lamport_workspace(ar, rows, HIDDEN, False)
    ref = None
    try:
        import flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk as fsk

        fsk._MAX_M = 256
        if tactic.mma_n in (8, 16, 32):  # the tiles FlashInfer's split-K GEMM takes
            part = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
            fsk.run_splitk_dense(
                x,
                w.t().contiguous(),
                None,
                part,
                True,
                fsk.SplitKTactic(tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages),
            )
            ref = ar(part)
    except ImportError:
        pass
    outs = []
    for one_shot in (True, False):
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        pg.run_bf16_gemm_ar_push(x, w, ws, one_shot, out)
        cpp = lam.one_shot_for(rows, HIDDEN, WORLD, opm.DEFAULT_ONE_SHOT_MAX_BYTES)
        outs.append(lam.run_lamport_finish(ws, out, one_shot, cpp_one_shot=cpp))
        ar(x @ w.t())  # engine call in between
    torch.cuda.synchronize()
    fails = []
    if not torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)):
        fails.append("transports_differ")
    if ref is not None and not torch.equal(outs[0].view(torch.int16), ref.view(torch.int16)):
        fails.append("vs_splitk_engine")
    cublas = ar(x @ w.t())
    torch.cuda.synchronize()
    err = float((outs[0].float() - cublas.float()).abs().max())
    return dict(
        rank=rank,
        fails=fails,
        hash=_digest(outs[0]),
        checked_splitk=ref is not None,
        err=err,
    )


def _rank_tp32_wide(rows: int, seed: int) -> dict:
    """TP32 o_proj rows above 48: push GEMM + finish vs cuBLAS (``F.linear``) + the engine all-reduce, and the GEMM
    alone (only rank 0 feeds rows, so the all-reduce returns its BF16 partial, ``-0`` as ``+0``)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        oproj_allreduce_push as opm,
    )

    rank, ar = _module(WORLD)
    k = 384
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    common = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    x[0, :8] = 0
    x[-1, 8:16] = -0.0
    w = (torch.randn(HIDDEN, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    x0 = (torch.randn(rows, k, generator=common, device="cuda") * 8).to(torch.bfloat16)
    x0[::3] = 0
    w0 = (torch.randn(HIDDEN, k, generator=common, device="cuda") * k**-0.5).to(torch.bfloat16)
    w0[:, k // 2 :] = -w0[:, : k // 2]
    x0[1::3, k // 2 :] = x0[1::3, : k // 2]  # exact cancellation inside the K sum
    fails = []
    for i in range(2):
        out = opm.oproj_allreduce_push(x, w, ar)
        ar(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        ref = ar(torch.nn.functional.linear(x, w))
        torch.cuda.synchronize()
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            fails.append(("vs_cublas_engine", i))
    got = opm.oproj_allreduce_push(x0 if rank == 0 else torch.zeros_like(x0), w0, ar)
    want = torch.nn.functional.linear(x0, w0)
    want[want == 0] = 0.0
    torch.cuda.synchronize()
    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
        fails.append("gemm_vs_cublas")
    return dict(rank=rank, fails=fails, served=opm.push_serves(x, w, ar), hash=_digest(out))


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("rows", [49, 56, 57, 64, 72, 80, 96, 128, 129, 160, 190, 192])
def test_tp32_wide_rows_are_cublas_plus_engine_allreduce(rows, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_tp32_wide, *zip(*[(rows, 3000 + rows)] * WORLD)))
    for r in results:
        assert r["served"] and not r["fails"], r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "rows,k",
    [
        (1, 384),
        (2, 384),
        (4, 384),
        (8, 384),
        (16, 384),
        (32, 384),
        (48, 384),
        (64, 384),
        (128, 384),
        (192, 384),
        (1, 768),
        (6, 768),
        (16, 768),
        (24, 768),
    ]
    + [(32, 768), (40, 768), (48, 768), (72, 768), (80, 768), (128, 768), (192, 768)]
    + [(8, 1536), (16, 1536), (72, 1536)]
    + [(8, 256), (72, 256), (1, 512), (24, 512), (72, 512), (8, 896), (48, 896), (80, 896)],
)
def test_push_gemm_is_the_engine_allreduce_of_split_k(rows, k, mpi_pool_executor) -> None:
    results = list(
        mpi_pool_executor.map(_rank_push_gemm, *zip(*[(rows, k, 2000 + rows + k)] * WORLD))
    )
    for r in results:
        assert not r["fails"], r
        # Within bf16 rounding of cuBLAS + engine AR (the GEMM's K order may differ).
        assert r["err"] < 0.1, r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_tactic_table_bounds() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.comm import (  # noqa: E501
        bf16_ar_push_gemm as pg,
    )

    assert pg.tactic_for(192, HIDDEN, 384) is not None and pg.tactic_for(193, HIDDEN, 384) is None
    assert pg.tactic_for(192, HIDDEN, 768) is not None and pg.tactic_for(193, HIDDEN, 768) is None
    assert pg.tactic_for(72, HIDDEN, 1536) is not None and pg.tactic_for(73, HIDDEN, 1536) is None
    for k in (256, 512, 896, 1792):
        assert pg.tactic_for(80, HIDDEN, k) is not None and pg.tactic_for(81, HIDDEN, k) is None
    # K = 448 is not a whole number of K tiles: tuned up to 32 rows, never with a K split.
    assert pg.tactic_for(32, HIDDEN, 448) is not None and pg.tactic_for(33, HIDDEN, 448) is None
    assert all(pg.tactic_for(m, HIDDEN, 448).split_k == 1 for m in range(1, 33))
    assert pg.tactic_for(8, HIDDEN, 3072) is None
