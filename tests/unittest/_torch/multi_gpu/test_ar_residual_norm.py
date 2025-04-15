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
import pickle
import random
import sys
import traceback
from time import sleep

import cloudpickle
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping

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
                    residual, weights, hidden_size, dtype, fused_add_norm):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, residual, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights,
                                 fused_add_norm)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
        x: torch.Tensor, residual: torch.Tensor, hidden_size: int,
        dtype: torch.dtype, tensor_parallel_size: int,
        tensor_parallel_rank: int, weights: torch.Tensor, fused_add_norm: bool):
    backend = Backend()
    x = x.cuda()
    residual = residual.cuda()
    norm_weight = torch.randn((hidden_size, ), dtype=dtype, device="cuda")
    eps = 1e-5

    l0 = Linear(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=False,
        dtype=dtype,
        tensor_parallel_mode=TensorParallelMode.ROW,
        mapping=Mapping(world_size=tensor_parallel_size,
                        tp_size=tensor_parallel_size,
                        rank=tensor_parallel_rank),
    ).cuda()
    norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).cuda()

    # Since all the modules here are provided by TRT-LLM,
    # so it has to be fullgraph compatible
    @torch.compile(backend=backend, fullgraph=True)
    def func(input, residual):
        xs = torch.chunk(input, 2, dim=-1)
        x = l0(xs[tensor_parallel_rank])
        if fused_add_norm:
            x, inter_output = norm(x, residual)
        else:
            inter_output = x + residual
            x = norm(inter_output)
        # Plus one here to trick torch compile that this is not an in-place update function
        # because we may transform it to an out-of-place function
        return x, inter_output + 1

    l0.load_weights([dict(weight=weights[0])])
    norm.weight.data.copy_(norm_weight)

    final_output, inter_output = func(x.clone(), residual.clone())

    assert backend.match_count[0] == 1, "Pattern matching failed"

    # torch run
    l0_torch = nn.Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype)
    l0_torch.weight.data.copy_(weights[0])
    l0_torch.cuda()

    def ref_func(input, residual):
        torch_output = l0_torch.forward(input)
        torch_inter_output = torch_output + residual
        torch_final_output = rms_norm(torch_inter_output, norm_weight, eps)
        return torch_final_output, torch_inter_output + 1

    seq_len = x.shape[0]

    for i in range(1, seq_len + 1):
        sleep(random.randint(0, 10) / 10 * tensor_parallel_rank)
        final_output, inter_output = func(x[:i], residual[:i])
        torch_final_output, torch_inter_output = ref_func(x[:i], residual[:i])

        torch.testing.assert_close(
            torch_final_output,
            final_output,
            rtol=0.05,
            atol=0.15,
        )
        torch.testing.assert_close(
            torch_inter_output,
            inter_output,
            rtol=0.05,
            atol=0.15,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("seq_len", [8, 32], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [16, 256], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("fused_add_norm", [True, False],
                         ids=["fused_add_norm", "unfused_add_norm"])
def test_row_linear_residual_norm_fusion(seq_len, hidden_size, fused_add_norm):
    pytest.skip(
        "Skip for now, waiting for proper fix for this issue: https://nvbugspro.nvidia.com/bug/5060957"
    )
    torch.manual_seed(42)
    dtype = torch.bfloat16
    tensor_parallel_size = 2
    x = torch.randn((seq_len, hidden_size), dtype=dtype)
    residual = torch.randn_like(x)
    l0_weight = torch.randn((hidden_size, hidden_size), dtype=dtype)
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size,
                    row_linear_residual_norm_fusion_forward, x, residual,
                    [l0_weight], hidden_size, dtype, fused_add_norm)] * 2),
        )
        for r in results:
            assert r is True


if __name__ == "__main__":
    test_row_linear_residual_norm_fusion(256, True)
