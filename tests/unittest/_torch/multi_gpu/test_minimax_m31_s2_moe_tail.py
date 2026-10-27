# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""trtllm::minimax_m31_s2_moe_tail (TP2W_S2_TAIL) on 2 ranks, bit for bit the chain it fuses:
m3_moe_finalize -> AllReduce(ONESHOT, RESIDUAL_RMS_NORM) -> mxfp8_quantize(norm_out, swizzled). Every output is
compared: norm_out, residual_out, the MXFP8 bytes, the valid rows' scales; the pad rows' scales are 0."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN, TOPK, EPS = 6144, 4, 1e-6
ROWS = (1, 8, 48, 64, 88, 112, 128, 136, 176, 224)


def _sf_valid_mask(rows, dev):
    cols = HIDDEN // 32
    padded = (rows + 127) // 128 * 128
    m = torch.zeros(padded * cols, dtype=torch.bool, device=dev)
    r = torch.arange(padded, device=dev).view(-1, 1).expand(padded, cols)
    c = torch.arange(cols, device=dev).view(1, -1).expand(padded, cols)
    off = (
        (r // 128) * (cols // 4 * 512)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (c % 4)
    )
    m[off.reshape(-1)] = (r < rows).reshape(-1)
    return m


def _run_rank(world):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
            m3_moe_finalize,
        )

        # getattr keeps "ops" out of this function's code names: cloudpickle would otherwise pickle the torch.ops
        # module object (a ModuleType subclass it cannot reduce) along with the function.
        trtllm_ops = getattr(torch, "ops").trtllm
        op = trtllm_ops.minimax_m31_s2_moe_tail
        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
        g0 = torch.Generator(device=dev).manual_seed(7)
        w_norm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g0, device=dev)).to(torch.bfloat16)
        w_norm[::97] = -w_norm[::97]
        bad = []
        for rows in ROWS:
            for sdt in (torch.bfloat16, torch.float32):
                for seed in (0, 1, 2):
                    g = torch.Generator(device=dev).manual_seed(1000 * seed + 17 * rank + rows)
                    n_perm = rows * TOPK
                    fc2 = (
                        torch.randn(((n_perm + 127) // 128 * 128, HIDDEN), generator=g, device=dev)
                        * 0.5
                    )
                    if seed == 2:  # 2^+-12 per-row stress
                        fc2 = fc2 * torch.pow(
                            2.0,
                            torch.randint(
                                -12, 13, (fc2.shape[0], 1), generator=g, device=dev
                            ).float(),
                        )
                    fc2 = fc2.to(torch.bfloat16)
                    perm = (
                        torch.randperm(n_perm, generator=g, device=dev)
                        .to(torch.int32)
                        .view(rows, TOPK)
                    )
                    if seed == 1:  # dropped slots
                        drop = torch.rand((rows, TOPK), generator=g, device=dev) < 0.02
                        perm = torch.where(drop, torch.full_like(perm, -1), perm)
                    perm = perm.contiguous()
                    scales = torch.softmax(
                        torch.randn((rows, TOPK), generator=g, device=dev), -1
                    ).to(sdt)
                    shared = (torch.randn((rows, HIDDEN), generator=g, device=dev) * 0.5).to(
                        torch.bfloat16
                    )
                    residual = (torch.randn((rows, HIDDEN), generator=g, device=dev) * 2).to(
                        torch.bfloat16
                    )
                    o = m3_moe_finalize(fc2, scales.contiguous(), perm, shared)
                    n, r = ar(
                        o,
                        all_reduce_params=AllReduceParams(
                            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                            residual=residual,
                            norm_weight=w_norm,
                            eps=EPS,
                            trigger_completion_at_end=False,
                        ),
                    )
                    q, sf = trtllm_ops.mxfp8_quantize(n, True)
                    b = op(
                        fc2,
                        scales.contiguous(),
                        perm,
                        shared,
                        residual,
                        w_norm,
                        ar.workspace,
                        rank,
                        world,
                        EPS,
                        False,
                    )
                    torch.cuda.synchronize()
                    valid = _sf_valid_mask(rows, dev)
                    diffs = (
                        int((n.view(torch.int16) != b[0].view(torch.int16)).sum()),
                        int((r.view(torch.int16) != b[1].view(torch.int16)).sum()),
                        int((q.view(torch.uint8) != b[2].view(torch.uint8)).sum()),
                        int((sf[valid] != b[3][valid]).sum()),
                        int((b[3][~valid] != 0).sum()),
                    )
                    if any(diffs):
                        bad.append((rows, str(sdt), seed, diffs))
        assert not bad, bad
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.cuda.get_device_capability()[0] < 10,
    reason="needs 2 GPUs of SM100 or newer",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_s2_moe_tail_is_the_fused_chain_bit_for_bit(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    for r in mpi_pool_executor.map(_run_rank, *zip(*[(world,)] * world)):
        assert r is True
