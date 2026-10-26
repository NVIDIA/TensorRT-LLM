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
"""DSpark draft prep under TP: the forked back-fill projection on its own all-reduce workspace equals
the inline one."""

import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.models.modeling_dspark import DSv4DSparkDraftModel, _private_allreduce
from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN, CAPTURES, KV = 1024, 3, 3 * 512


def _fp8(gen, n, k):
    """E4M3 weights with power-of-two 32x32 UE8M0 scales, as the checkpoint stores them."""
    w = torch.randn(n, k, generator=gen, device="cuda") * 0.02
    tiles = w.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3))
    exponent = torch.ceil(torch.log2(tiles / 448.0))
    scale = torch.exp2(exponent).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (w / scale).to(torch.float8_e4m3fn), torch.exp2(exponent).to(torch.float8_e8m0fnu)


@torch.inference_mode()
def _run(tp_size: int, strategy: str) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        if strategy == "MNNVL":
            # The MNNVL protocol on one node (fabric memory replaced by POSIX handles).
            os.environ["TLLM_TEST_MNNVL"] = "1"
        mapping = Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=rank,
            gpus_per_node=torch.cuda.device_count(),
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        main_proj, main_scale = _fp8(gen, HIDDEN, CAPTURES * HIDDEN)
        wkv, wkv_scale = _fp8(gen, KV, HIDDEN)

        model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
        nn.Module.__init__(model)
        model.model_config = SimpleNamespace(mapping=mapping)
        stage0 = nn.Module()
        stage0.main_norm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16)
        stage0.main_norm.weight.copy_(
            1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda").bfloat16()
        )
        stage0.self_attn = nn.Module()
        stage0.self_attn.o_b_proj = nn.Module()
        shared = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy[strategy], dtype=torch.bfloat16
        )
        stage0.self_attn.o_b_proj.all_reduce = shared
        model.mtp_layers = nn.ModuleList([stage0]).cuda()
        model._cache_main_proj_shard(
            {"mtp.0.main_proj.weight": main_proj.cpu(), "mtp.0.main_proj.scale": main_scale.cpu()},
            32,
            torch.device("cuda"),
        )
        assert model._main_proj_shard is not None
        rows_scale = wkv_scale.view(torch.uint8).repeat_interleave(32, 0)
        model._backfill_wkv_fp8 = (wkv, rows_scale.contiguous())
        model.prep_fork_allreduce = _private_allreduce(shared, 64, HIDDEN)
        assert model.prep_fork_allreduce is not None

        def error(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm()).item()

        captured = torch.randn(40, CAPTURES * HIDDEN, generator=gen, device="cuda").bfloat16()
        rows = torch.arange(12, device="cuda").view(2, 6)
        same_kernel = shared.mnnvl_allreduce is not None

        def inline():
            main_x = model.project_captured_rows(captured, rows)
            kv = mxfp8_lowm_linear(main_x[:, :5].reshape(10, -1), *model._backfill_wkv_fp8)
            return main_x, kv.view(2, 5, -1)

        def matches_inline(forked):
            want = inline()
            if same_kernel:
                return all(torch.equal(a, b) for a, b in zip(forked, want))
            return all(error(a, b) < 1e-2 for a, b in zip(forked, want))

        # The fork on its own workspace, eager and replayed from one graph next to the shared
        # all-reduce on another stream; neither all-reduce disturbs the other.
        results = [matches_inline(model.project_backfill_rows(captured, rows))]
        side = torch.cuda.Stream()
        other = torch.randn(6, HIDDEN, generator=gen, device="cuda").bfloat16()
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=torch.zeros_like(other),
            norm_weight=stage0.main_norm.weight,
            eps=1e-6,
        )

        def concurrent():
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                forked = model.project_backfill_rows(captured, rows)
            reduced = [shared(other, all_reduce_params=params)[0] for _ in range(3)]
            torch.cuda.current_stream().wait_stream(side)
            return forked, reduced

        concurrent()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forked, reduced = concurrent()
        for _ in range(5):
            captured.copy_(torch.randn(captured.shape, generator=gen, device="cuda").bfloat16())
            other.copy_(torch.randn(other.shape, generator=gen, device="cuda").bfloat16())
            graph.replay()
            eager = model.project_backfill_rows(captured, rows)
            results.append(all(torch.equal(a, b) for a, b in zip(forked, eager)))
            results.append(matches_inline(forked))
            want = shared(other, all_reduce_params=params)[0]
            results.append(all(torch.equal(r, want) for r in reduced))
        torch.cuda.synchronize()
        assert all(results), results
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or get_sm_version() not in (100, 103),
    reason="needs 4 SM100 or SM103 GPUs",
)
@pytest.mark.parametrize("strategy", ["AUTO", "MNNVL"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_forked_backfill_projection_matches_inline(mpi_pool_executor, strategy):
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(tp_size, strategy)] * tp_size))
    for result in results:
        assert result is True
