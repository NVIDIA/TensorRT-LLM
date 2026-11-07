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
"""Pushed MNNVL all-reduce with the ``RESIDUAL_RMS_NORM`` finish (``AllReduce.push_target`` /
``finish_push``) and the BF16 split-K GEMM push epilogue: bitwise the C++ MNNVL fused all-reduce
of the same partials, with C++ calls interleaved on the workspace."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from _torch.helpers import bf16_bits_differ as _differ
from _torch.helpers import mnnvl_torch_push as _torch_push
from _torch.helpers import mpi_worker_env
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
EPS = 1e-6


def _input(m, h, seed, rank, special=False):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    x = (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    if special:
        bits = x.view(torch.int16)
        r = torch.rand(m, h, device="cuda", generator=gen)
        bits[r < 0.05] = -32768  # -0
        bits[(r >= 0.05) & (r < 0.08)] = 0
        den = (r >= 0.08) & (r < 0.1)
        bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
            torch.int16
        )
    return x


@mpi_worker_env(TLLM_TEST_MNNVL="1", TRTLLM_FORCE_MNNVL_AR="1")
def _run_rank(world_size):
    from tensorrt_llm._torch.distributed import (
        AllReduce,
        AllReduceFusionOp,
        AllReduceParams,
        AllReduceStrategy,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm._torch.kernels.blackwell import bf16_splitk_push_gemm as push_gemm
    from tensorrt_llm._torch.kernels.blackwell.mnnvl_pushed_allreduce_norm import norm_finish_config
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert allreduce.mnnvl_allreduce is not None
        gen = torch.Generator(device="cuda").manual_seed(5)
        gamma = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
        bad = 0

        def params(residual):
            return AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=gamma,
                eps=EPS,
            )

        # A pushed input finished with the residual + RMSNorm against the C++ fused all-reduce,
        # one-shot and two-shot sizes, with C++ calls in between.
        transports = set()
        for m in (1, 7, 8, 14, 16, 21, 28, 32, 56, 64, 112, 128):
            for special in (False, True):
                x = _input(m, HIDDEN, 7 * m + special, rank, special)
                res = _input(m, HIDDEN, 11 * m + special, 0, special)
                ref_out, ref_res = allreduce(x, all_reduce_params=params(res))
                target = allreduce.push_target(m, HIDDEN, all_reduce_params=params(res))
                # Pushed exactly where the finish kernel has a single-load launch shape.
                served = norm_finish_config(m, HIDDEN, x.device)[0] > 0
                assert (target is not None) == served, m
                if not served:
                    continue
                transports.add(target.one_shot)
                assert target.norm_params is not None
                # The fusion composes only with a finish kernel that writes both outputs.
                for option in ({"in_place": True}, {"consumer_finishes": True}):
                    assert (
                        allreduce.push_target(m, HIDDEN, all_reduce_params=params(res), **option)
                        is None
                    )
                workspace = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
                _torch_push(workspace, target, x, rank)
                out, res_out = allreduce.finish_push(target)
                bad += _differ(ref_out, out) + _differ(ref_res, res_out)
                again_out, again_res = allreduce(x, all_reduce_params=params(res))
                bad += _differ(ref_out, again_out) + _differ(ref_res, again_res)
        assert transports == {True, False}, "both transports must be finished with the fusion"

        # The BF16 split-K GEMM's push epilogue against its own store-only variant followed by
        # the C++ fused all-reduce (the drafter's o_proj and down_proj shapes).
        for k in (1024, 1792):
            wgen = torch.Generator(device="cuda").manual_seed(13 + rank)
            w = (torch.randn(HIDDEN, k, device="cuda", generator=wgen) / k**0.5).to(torch.bfloat16)
            for m in (1, 7, 14, 28, 56, 64):
                tactic = push_gemm.default_push_tactic(m, HIDDEN, k)
                assert tactic is not None
                x = _input(m, k, 3 * m, rank)
                res = _input(m, HIDDEN, 5 * m, 0)
                plain = push_gemm.run_gemm(x, w, torch.empty(m, HIDDEN, **_bf16()), tactic)
                ref_out, ref_res = allreduce(plain, all_reduce_params=params(res))
                target = allreduce.push_target(m, HIDDEN, all_reduce_params=params(res))
                assert target is not None
                push_gemm.run_gemm_push(x, w, target.view, target.one_shot, tactic)
                out, res_out = allreduce.finish_push(target)
                bad += _differ(ref_out, out) + _differ(ref_res, res_out)
        torch.cuda.synchronize()
        assert bad == 0, f"{bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _bf16():
    return {"dtype": torch.bfloat16, "device": "cuda"}


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_pushed_allreduce_residual_rms_norm_matches_cpp(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
