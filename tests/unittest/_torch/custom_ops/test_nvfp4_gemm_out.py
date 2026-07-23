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

import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.compilation.backend import Backend


def _make_operands(dtype: torch.dtype):
    torch.manual_seed(0)
    activation = torch.randn((4, 128), dtype=dtype, device="cuda")
    weight = torch.randn((192, 128), dtype=dtype, device="cuda")
    activation_scale = (448 * 6) / activation.abs().max().float()
    weight_scale = (448 * 6) / weight.abs().max().float()
    activation_fp4, activation_sf = torch.ops.trtllm.fp4_quantize(
        activation, activation_scale, 16, False)
    weight_fp4, weight_sf = torch.ops.trtllm.fp4_quantize(
        weight, weight_scale, 16, False)
    alpha = torch.reciprocal(activation_scale * weight_scale).reshape(1)
    return activation_fp4, weight_fp4, activation_sf, weight_sf, alpha


@skip_pre_blackwell
def test_nvfp4_gemm_out_compile_and_cuda_graph():
    operands = _make_operands(torch.bfloat16)

    with torch.inference_mode(), autotune():
        reference = torch.ops.trtllm.nvfp4_gemm(
            *operands,
            output_dtype=torch.bfloat16,
            allowed_backends="cutlass",
        )
        eager_base = reference.new_empty((8, reference.size(1)))
        torch.ops.trtllm.nvfp4_gemm_out(*operands, eager_base)
        eager_output = eager_base.narrow(0, 0, reference.size(0))

    torch.testing.assert_close(eager_output,
                               reference,
                               rtol=1e-2,
                               atol=0.15)

    def run_out(act_fp4, weight, act_sf, weight_sf, alpha, output):
        torch.ops.trtllm.nvfp4_gemm_out(act_fp4, weight, act_sf, weight_sf,
                                        alpha, output)
        return output.narrow(0, 0, act_fp4.size(0))

    compiled_base = reference.new_empty((8, reference.size(1)))
    compiled = torch.compile(
        run_out, backend=Backend(enable_inductor=False), fullgraph=True)
    with torch.inference_mode():
        compiled_result = compiled(*operands, compiled_base)
    assert compiled_result.data_ptr() == compiled_base.data_ptr()
    torch.testing.assert_close(compiled_result,
                               reference,
                               rtol=1e-2,
                               atol=0.15)

    graph_base = reference.new_empty((8, reference.size(1)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.nvfp4_gemm_out(*operands, graph_base)
    graph.replay()
    torch.cuda.synchronize()
    graph_output = graph_base.narrow(0, 0, reference.size(0))
    torch.testing.assert_close(graph_output,
                               reference,
                               rtol=1e-2,
                               atol=0.15)
