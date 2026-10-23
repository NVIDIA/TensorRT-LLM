# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check NCCL determinism across decode/prefill shapes and graph replays."""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def run_deterministic_allreduce_rank(world, dtype, fusion_op, with_bias):
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    previous = os.environ.get("FORCE_DETERMINISTIC")
    os.environ["FORCE_DETERMINISTIC"] = "1"
    try:
        generator = torch.Generator().manual_seed(732)
        hidden = 4096
        # Independent rank contributions with cancellation and unequal magnitudes.
        rank_rows = torch.randn(world, hidden, generator=generator).to(dtype)
        rank_rows[0] *= 8
        rank_rows[-1] = -rank_rows[0] + rank_rows[-1] / 8
        reference_sum = rank_rows.double().sum(0).to(dtype).cuda()
        local_row = rank_rows[rank].cuda()
        residual_row = torch.randn(hidden, generator=generator).to(dtype).cuda()
        weight = torch.randn(hidden, generator=generator).to(dtype).cuda()
        bias = (
            torch.randn(hidden, generator=generator).to(dtype).cuda() * 0.01 if with_bias else None
        )
        op = AllReduce(
            Mapping(world_size=world, rank=rank, tp_size=world),
            strategy=AllReduceStrategy.NCCL,
        )

        def params(residual):
            return AllReduceParams(
                fusion_op=fusion_op,
                residual=residual,
                norm_weight=weight,
                bias=bias,
                eps=1e-6,
            )

        def as_tuple(result):
            return result if isinstance(result, tuple) else (result,)

        expected = None
        for tokens in (1, 17, 257, 1025):
            storage = torch.empty(tokens, 2 * hidden, device="cuda", dtype=dtype)
            x = storage[:, ::2]
            x.copy_(local_row)
            residual = residual_row.repeat(tokens, 1)
            actual = as_tuple(op(x, all_reduce_params=params(residual)))
            if expected is None:
                expected = tuple(t[0].clone() for t in actual)
            for value, first in zip(actual, expected, strict=True):
                torch.testing.assert_close(value[-1], first, rtol=0, atol=0)

            # Use FP64 summation as an independent oracle, with explicit stores
            # at the collective, bias and residual boundaries.
            summed = reference_sum
            if fusion_op == AllReduceFusionOp.NONE:
                reference = (summed,)
            else:
                if bias is not None:
                    summed = (summed.double() + bias.double()).to(dtype)
                if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                    summed = (summed.double() + residual_row.double()).to(dtype)
                normalized = summed.double()
                normalized *= torch.rsqrt(normalized.square().mean() + 1e-6)
                normalized = (normalized * weight.double()).to(dtype)
                reference = (
                    (normalized, summed)
                    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
                    else (normalized,)
                )
            tolerance = 0.008 if dtype == torch.bfloat16 else 0.001
            if dtype == torch.float32:
                tolerance = 2e-6
            for value, ref in zip(actual, reference, strict=True):
                torch.testing.assert_close(value[-1], ref, rtol=tolerance, atol=tolerance)

        x = local_row.repeat(17, 1)
        residual = residual_row.repeat(17, 1)
        op(x, all_reduce_params=params(residual))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = as_tuple(op(x, all_reduce_params=params(residual)))
        # Replay must consume new buffers, including the fused residual input.
        x.mul_(0.5)
        residual.mul_(0.25)
        graph.replay()
        updated = as_tuple(op(x, all_reduce_params=params(residual)))
        for value, ref in zip(captured, updated, strict=True):
            torch.testing.assert_close(value, ref, rtol=0, atol=0)
        return True
    finally:
        if previous is None:
            os.environ.pop("FORCE_DETERMINISTIC", None)
        else:
            os.environ["FORCE_DETERMINISTIC"] = previous


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "fusion_op",
    [
        AllReduceFusionOp.NONE,
        AllReduceFusionOp.RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_nccl_deterministic_phase_and_graph(mpi_pool_executor, dtype, fusion_op, with_bias):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        run_deterministic_allreduce_rank,
        *zip(*[(world, dtype, fusion_op, with_bias)] * world),
    )
    assert all(results)
