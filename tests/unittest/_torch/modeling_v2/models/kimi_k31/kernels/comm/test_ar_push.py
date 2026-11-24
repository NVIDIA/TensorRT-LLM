# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's MNNVL Lamport all-reduce kernels on one GB300 tray (four MPI ranks).

``lamport_ar`` claims bitwise equality with the engine MNNVL all-reduce
(``trtllm::mnnvl_fusion_allreduce``) of the same bf16 input, whichever transport carries it and
whichever engine kernel (one-shot / two-shot) it reproduces. It is checked bit for bit here, over
row counts that change the transport and the owner pattern of the two-shot rows, interleaved with
engine calls on the same workspace (the rotation state is shared). The ranks report hashes, so
every rank must hold the same tensor. The push GEMM that feeds these kernels in the tep target is
tested with that target (``kimi_k31_pro_mxfp4__sm_103__tep/kernels/comm/test_bf16_ar_push_gemm.py``).

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


def _rank_lamport(rows: int, seed: int) -> dict:
    """Push kernel + finish vs the engine all-reduce of the same input, every transport pairing."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        lamport_ar as lam,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        oproj_allreduce_push as opm,
    )

    rank, ar = _module(WORLD)
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    x[0, :16] = 0
    x[-1, 16:32] = -0.0  # -0 must travel as +0
    # The test drives both transports: grow for both, then take the (final) view.
    opm.lamport_workspace(ar, rows, HIDDEN, True)
    ws = opm.lamport_workspace(ar, rows, HIDDEN, False)
    engine = ar(x)
    fails = []
    for one_shot in (True, False):
        for cpp_one_shot in (True, False):
            lam.run_lamport_push(ws, x, one_shot)
            out = lam.run_lamport_finish(
                ws, torch.empty_like(x), one_shot, cpp_one_shot=cpp_one_shot
            )
            ref = ar(x)  # an engine call on the same workspace in between
            torch.cuda.synchronize()
            want_engine = cpp_one_shot == lam.one_shot_for(
                rows, HIDDEN, WORLD, opm.DEFAULT_ONE_SHOT_MAX_BYTES
            )
            if want_engine and not torch.equal(out.view(torch.int16), engine.view(torch.int16)):
                fails.append(("vs_engine", one_shot, cpp_one_shot))
            if not torch.equal(ref.view(torch.int16), engine.view(torch.int16)):
                fails.append(("engine_not_repeatable", one_shot, cpp_one_shot))
    # Two-shot finish without the gather, then the gather kernel.
    lam.run_lamport_push(ws, x, False)
    lam.run_lamport_finish_in_place(
        ws,
        rows,
        HIDDEN,
        cpp_one_shot=lam.one_shot_for(rows, HIDDEN, WORLD, opm.DEFAULT_ONE_SHOT_MAX_BYTES),
    )
    gathered = lam.run_lamport_gather(ws, torch.empty_like(x))
    ar(x)
    torch.cuda.synchronize()
    if not torch.equal(gathered.view(torch.int16), engine.view(torch.int16)):
        fails.append(("in_place_gather",))
    return dict(rank=rank, fails=fails, hash=_digest(engine))


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 9, 33, 72])
def test_lamport_push_finish_is_the_engine_allreduce(rows, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_lamport, *zip(*[(rows, 1000 + rows)] * WORLD)))
    for r in results:
        assert not r["fails"], r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"
