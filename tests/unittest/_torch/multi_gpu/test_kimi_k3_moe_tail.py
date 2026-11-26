# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi K3 latent-MoE tail vs the unfused TP tail (MNNVL AllReduce +
RMSNorm, full up-projection, shared AllReduce, add) on 8 GPUs, and its
one-kernel form for up to 8 tokens vs its collective and up-projection."""

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

HIDDEN, LATENT, EPS, TOP_K = 7168, 3584, 1e-6, 16
TOKEN_COUNTS = (1, 3, 5, 6, 7, 28, 112, 128)
# The one-kernel tail serves up to 8 tokens; 12 stays on the collective and the up-projection.
ONE_KERNEL_MAX_TOKENS = 8
ONE_KERNEL_TOKEN_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 12)
ONE_KERNEL_ROUNDS = 4
# Against the FP32 reference, and between two kernel paths of the same computation.
REFERENCE_TOLERANCE = {"atol": 6e-2, "rtol": 2e-2}
PATH_TOLERANCE = {"atol": 3.2e-2, "rtol": 1e-2}


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
            torch.testing.assert_close(fused.float(), reference, **REFERENCE_TOLERANCE)
            torch.testing.assert_close(fused, unfused, **PATH_TOLERANCE)
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


def _unfinalized_case(tp_size: int, m: int, seed: int, split: bool):
    """Every rank's routed-expert rows before the top-k sum, its route map, its shared partial
    and the routed partial the finalize would produce, plus the routing weights (the same on
    every rank). ``split``: each route is owned by half of the ranks, as with expert parallelism."""
    g = torch.Generator().manual_seed(seed)
    weights = (torch.rand(m, TOP_K, generator=g) * 0.2 + 0.01).to(torch.bfloat16)
    group = torch.randint(0, 2, (m, TOP_K), generator=g)
    ranks = []
    for r in range(tp_size):
        owned = group == r // (tp_size // 2) if split else torch.ones(m, TOP_K, dtype=torch.bool)
        num_rows = int(owned.sum())
        index = torch.full((m, TOP_K), -1, dtype=torch.int32)
        index[owned] = torch.randperm(num_rows, generator=g).to(torch.int32)
        rows = (torch.randn(max(num_rows, 1), LATENT, generator=g) * 0.5).to(torch.bfloat16)
        partial = torch.zeros(m, LATENT, dtype=torch.float32)
        for slot in range(TOP_K):  # FP32 in slot order, one BF16 rounding, as the finalize does
            picked = rows[index[:, slot].clamp(min=0).long()].float()
            term = picked * weights[:, slot : slot + 1].float()
            partial = torch.where(owned[:, slot : slot + 1], partial + term, partial)
        shared = (torch.randn(m, HIDDEN, generator=g) * 0.05).to(torch.bfloat16)
        ranks.append((rows, index, shared, partial.to(torch.bfloat16)))
    return weights, ranks


@torch.inference_mode()
def _one_kernel_tail_forward(tp_size: int, gamma, up_weight):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        from tensorrt_llm._torch.modules.kimi_k3_moe_tail import (
            KimiK3LatentMoETail,
            UnfinalizedRoutedOutput,
            math_reference,
        )

        mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
        gamma, up_weight = gamma.cuda(), up_weight.cuda()
        tail = KimiK3LatentMoETail.get(mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=EPS)
        one_kernel = tail._wave
        assert one_kernel is not None, "the one-kernel tail was not built"
        for it in range(ONE_KERNEL_ROUNDS):
            for m in ONE_KERNEL_TOKEN_COUNTS:
                served = m <= ONE_KERNEL_MAX_TOKENS
                assert one_kernel.supports(m) == served
                # Two different inputs back to back, with only the copy kernel between the two
                # launches: the second one must find the first one's slot advanced and re-armed,
                # across token counts and across launches of the other kernels.
                cases = [
                    _unfinalized_case(tp_size, m, 1000 * it + 10 * m + i, split=it % 2 == 0)
                    for i in range(2)
                ]
                calls = []
                for weights, ranks in cases:
                    rows, index, shared, _ = ranks[rank]
                    routed = UnfinalizedRoutedOutput(rows.cuda(), weights.cuda(), index.cuda())
                    calls.append((routed, shared.cuda()))
                before = one_kernel._state.tolist()
                for routed, shared in calls:
                    one_kernel_out = tail(routed, shared, gamma, up_weight)
                one_kernel_out = one_kernel_out.clone()
                # (slot, token count) of the last one-kernel launch: it ran exactly when served.
                after = [(before[0] + len(calls)) % 3, m, 0, 0] if served else before
                assert one_kernel._state.tolist() == after
                # The second input through the collective and the up-projection.
                tail._wave = None
                try:
                    two_kernels_out = tail(*calls[-1], gamma, up_weight).clone()
                finally:
                    tail._wave = one_kernel
                ranks = cases[-1][1]
                reference = math_reference(
                    torch.stack([r[3] for r in ranks]).cuda(),
                    torch.stack([r[2] for r in ranks]).cuda(),
                    gamma,
                    up_weight,
                    EPS,
                )
                torch.testing.assert_close(one_kernel_out.float(), reference, **REFERENCE_TOLERANCE)
                torch.testing.assert_close(one_kernel_out, two_kernels_out, **PATH_TOLERANCE)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _one_kernel_tail_requested() -> bool:
    """Whether the tail builds its one-kernel form here (8 SM10x GPUs; on by default on SM103)."""
    if torch.cuda.device_count() < 8 or torch.cuda.get_device_capability()[0] != 10:
        return False
    from tensorrt_llm._torch.modules.kimi_k3_moe_tail.latent_moe_tail import (
        fused_tail_wave_requested,
    )

    return fused_tail_wave_requested(8, HIDDEN, LATENT, torch.device("cuda", 0))


@pytest.mark.skipif(
    not _one_kernel_tail_requested(),
    reason="needs 8 SM10x GPUs with NVLink multicast and the one-kernel tail on "
    "(the default on SM103; KIMI_K3_FUSED_MOE_TAIL_WAVE=1 elsewhere)",
)
@pytest.mark.parametrize("mpi_pool_executor", [8], indirect=True)
def test_kimi_k3_fused_moe_tail_one_kernel_matches_two_kernels(mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    torch.manual_seed(0)
    gamma = (1.0 + 0.1 * torch.randn(LATENT)).to(torch.bfloat16)
    up_weight = (torch.randn(HIDDEN, LATENT) * 0.02).to(torch.bfloat16)
    results = mpi_pool_executor.map(
        _one_kernel_tail_forward,
        *zip(*[(tp_size, gamma, up_weight)] * tp_size),
    )
    assert all(results)
