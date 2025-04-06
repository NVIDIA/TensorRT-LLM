# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pickle
import random
import sys
import traceback
from time import sleep

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, DeepseekAllReduce,
                                             ParallelConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    residual, weight, ar_version):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, tensor_parallel_size, rank, residual,
                                 weight, ar_version)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def ar_cuda_graph(x: list[torch.Tensor], tensor_parallel_size: int,
                  tensor_parallel_rank: int, residual: torch.Tensor,
                  weight: torch.Tensor, ar_version: str):
    x = [i.cuda() for i in x]
    x0 = x[0]

    all_reduce = AllReduce(
        ParallelConfig(tensor_parallel_size=tensor_parallel_size,
                       tensor_parallel_rank=tensor_parallel_rank))
    graph = torch.cuda.CUDAGraph()

    # Warm up
    for _ in range(2):
        output = all_reduce(
            x0.chunk(tensor_parallel_size, dim=0)[tensor_parallel_rank])

    with torch.cuda.graph(graph):
        output = all_reduce(
            x0.chunk(tensor_parallel_size, dim=0)[tensor_parallel_rank])

    def ref_func(x):
        x0, x1 = x.chunk(2, dim=0)
        return x0 + x1

    for i in x:
        x0.copy_(i)
        graph.replay()
        torch.testing.assert_close(output, ref_func(i))


@torch.inference_mode()
def ar_cuda_graph_residual_norm(x: list[torch.Tensor],
                                tensor_parallel_size: int,
                                tensor_parallel_rank: int,
                                residual: torch.Tensor, weight: torch.Tensor,
                                ar_version: str):
    x = [i.cuda() for i in x]
    x0 = x[0]
    residual = residual.cuda()
    weight = weight.cuda()
    all_reduce = AllReduce(
        ParallelConfig(tensor_parallel_size=tensor_parallel_size,
                       tensor_parallel_rank=tensor_parallel_rank)
    ) if ar_version == "v1" else DeepseekAllReduce(
        ParallelConfig(tensor_parallel_size=tensor_parallel_size,
                       tensor_parallel_rank=tensor_parallel_rank))
    graph = torch.cuda.CUDAGraph()

    if ar_version == "v1":
        params = {
            "input":
            x0.chunk(tensor_parallel_size, dim=0)[tensor_parallel_rank],
            "all_reduce_params":
            AllReduceParams(fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                            residual=residual,
                            norm_weight=weight,
                            eps=1e-6)
        }
    else:
        params = {
            "hidden_states":
            x0.chunk(tensor_parallel_size, dim=0)[tensor_parallel_rank],
            "reduce_fusion_inputs": [residual, weight],
            "eps":
            1e-6,
            "fusion_op":
            AllReduceFusionOp.RESIDUAL_RMS_NORM
        }

    print("Here")

    # Warm up
    for _ in range(2):
        output = all_reduce(**params)

    with torch.cuda.graph(graph):
        output = all_reduce(**params)

    print(output)

    def ref_func(x):
        x0, x1 = x.chunk(2, dim=0)
        x = x0 + x1
        res = x + residual
        x = rms_norm(res, weight)
        return x, res

    for i in x:
        x0.copy_(i)
        sleep(random.random() * 5 * (tensor_parallel_rank + 1))
        graph.replay()
        torch_out, torch_res = ref_func(i)
        torch.testing.assert_close(output[0], torch_out)
        torch.testing.assert_close(output[1], torch_res)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("hidden_size", [16, 256], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("ar_version", ["v1", "v2"], ids=["ar_v1", "ar_v2"])
@pytest.mark.parametrize("fused_add_norm", [True, False],
                         ids=["fused_add_norm", "unfused_add_norm"])
def test_ar_cuda_graph(hidden_size, fused_add_norm, ar_version):
    if ar_version == "v2" and not fused_add_norm:
        pytest.skip("AR v2 does not support unfused add norm")
    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = 2
    test_round = 20
    x = [
        torch.randn((hidden_size * 2, ), dtype=dtype) for _ in range(test_round)
    ]
    residual = torch.randn((hidden_size, ), dtype=dtype)
    weight = torch.randn((hidden_size, ), dtype=dtype)
    test_func = ar_cuda_graph if not fused_add_norm else ar_cuda_graph_residual_norm
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, test_func, x, residual, weight,
                    ar_version)] * 2),
        )
        for r in results:
            assert r is True


if __name__ == "__main__":
    test_ar_cuda_graph(256, True, "v1")
