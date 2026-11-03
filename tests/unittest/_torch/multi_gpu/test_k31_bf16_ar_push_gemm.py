# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 split-K o_proj GEMM with the MNNVL all-reduce push epilogue (k31_bf16_ar_push_gemm) on 8 GPUs:
push GEMM + the Lamport finish kernel gives the bits of the FlashInfer split-K GEMM at the same tactic followed
by the C++ MNNVL all-reduce (tactics with token tiles FlashInfer has, up to 32), and is deterministic and within
BF16 rounding of cuBLAS + the C++ all-reduce at every tuned row count, with C++ all-reduces interleaved on the
workspace."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

N = 7168
TOKEN_COUNTS = (1, 7, 14, 16, 21, 28, 32, 40, 48, 64, 72)


@torch.inference_mode()
def _check(tp_size: int):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        import flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk as fsk

        from tensorrt_llm._torch.kernels.blackwell import k31_bf16_ar_push_gemm as pg

        k = 12288 // tp_size
        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(10 + rank)
        w = (torch.randn(N, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        wt = w.t().contiguous()
        ar(torch.zeros(max(TOKEN_COUNTS), N, dtype=torch.bfloat16, device="cuda"))
        assert pg.tactic_for(129, N, k) is None
        for m in TOKEN_COUNTS:
            x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            x[0, :8] = 0
            tactic = pg.tactic_for(m, N, k)
            if tactic is None:
                continue
            assert pg.can_push(x, w)
            ref = None
            if tactic.mma_n <= 32:
                ref_partial = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
                fsk.run_splitk_dense(
                    x,
                    wt,
                    None,
                    ref_partial,
                    True,
                    fsk.SplitKTactic(tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages),
                )
                ref = ar(ref_partial)
            cublas = ar(x @ w.t())
            first = None
            for _ in range(2):
                target = ar.lamport_push_target(m, N, force=True)
                pg.run_bf16_gemm_ar_push(x, w, target.workspace, target.one_shot, target.output)
                target.pushed = True
                got = ar.finish_lamport_push(target)
                ar(cublas)
                torch.cuda.synchronize()
                if ref is not None:
                    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), m
                if first is None:
                    first = got.clone()
                assert torch.equal(got.view(torch.int16), first.view(torch.int16)), m
            torch.testing.assert_close(first, cublas, atol=6e-2, rtol=2e-2)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(
    torch.cuda.device_count() < 8 or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs 8 SM100/SM103 GPUs with NVLink multicast",
)
@pytest.mark.parametrize("mpi_pool_executor", [8], indirect=True)
def test_k31_bf16_ar_push_gemm_matches_gemm_plus_allreduce(mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_check, *zip(*[(tp_size,)] * tp_size))
    assert all(results)
