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
"""Numerical and mutable CUDA graph checks for the Blackwell router GEMM."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 33, 512, 1025])
def test_bf16x3_router_gemm_mutable_graph(tokens: int) -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Validated on GB300")
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.bf16x3_router_gemm import bf16x3_router_gemm

    torch.manual_seed(20260919)
    inputs = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(256, 8192, device="cuda", dtype=torch.float32) / 8192**0.5
    with torch.inference_mode():
        bf16x3_router_gemm(inputs, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = bf16x3_router_gemm(inputs, weights)
        for seed in range(4):
            torch.manual_seed(seed)
            inputs.normal_()
            weights.normal_(std=8192**-0.5)
            if seed == 1:
                inputs.zero_()
            elif seed == 2:
                weights[:, ::2].neg_()
            graph.replay()
            # FP64 is independent of TF32 settings and split-K reduction order.
            expected = (inputs.double() @ weights.double().T).float()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
            assert actual.dtype == torch.float32
            assert actual.shape == (tokens, 256)
