# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi K3 latent-MoE tail vs the unfused TP tail (MNNVL AllReduce +
RMSNorm, full up-projection, shared AllReduce, add) on 8 GPUs."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN, LATENT, EPS = 7168, 3584, 1e-6
TOKEN_COUNTS = (1, 3, 5, 6, 7, 28, 112, 128)


@torch.inference_mode()
def _tail_forward(tp_size: int, routed: torch.Tensor, shared: torch.Tensor, gamma, up_weight):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        from tensorrt_llm._torch.modules.kimi_k3_moe_tail import KimiK3LatentMoETail, math_reference

        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
        gamma, up_weight = gamma.cuda(), up_weight.cuda()
        tail = KimiK3LatentMoETail.get(mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=EPS)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
        norm = AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=gamma, eps=EPS)
        for m in TOKEN_COUNTS:
            y = routed[rank, :m].cuda()
            s = shared[rank, :m].cuda()
            # Twice: the second call exercises the rotated Lamport buffers.
            for _ in range(2):
                fused = tail(y, s, gamma, up_weight)
            unfused = ar(y, all_reduce_params=norm) @ up_weight.t() + ar(s)
            reference = math_reference(
                routed[:, :m].cuda(), shared[:, :m].cuda(), gamma, up_weight, EPS
            )
            torch.testing.assert_close(fused.float(), reference, atol=6e-2, rtol=2e-2)
            torch.testing.assert_close(fused, unfused, atol=3.2e-2, rtol=1e-2)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(
    torch.cuda.device_count() < 8 or torch.cuda.get_device_capability()[0] != 10,
    reason="needs 8 SM10x GPUs with NVLink multicast",
)
@pytest.mark.parametrize("mpi_pool_executor", [8], indirect=True)
def test_kimi_k3_fused_moe_tail_matches_unfused(mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    torch.manual_seed(0)
    max_m = max(TOKEN_COUNTS)
    routed = (torch.randn(tp_size, max_m, LATENT) * 0.5).to(torch.bfloat16)
    shared = (torch.randn(tp_size, max_m, HIDDEN) * 0.05).to(torch.bfloat16)
    gamma = (1.0 + 0.1 * torch.randn(LATENT)).to(torch.bfloat16)
    up_weight = (torch.randn(HIDDEN, LATENT) * 0.02).to(torch.bfloat16)
    results = mpi_pool_executor.map(
        _tail_forward,
        *zip(*[(tp_size, routed, shared, gamma, up_weight)] * tp_size),
    )
    assert all(results)
