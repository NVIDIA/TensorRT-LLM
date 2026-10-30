# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize(
    "k_n",
    [(7168, 2112), (8192, 8192), (8192, 57344), (28672, 8192)],
)
@pytest.mark.parametrize(
    "m",
    [1, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_cublas_mm(dtype, m, k_n):
    k, n = k_n
    torch.random.manual_seed(0)
    shape_x = (m, k)
    shape_w = (n, k)
    x = torch.randn(shape_x, device="cuda").to(dtype)
    w = torch.randn(shape_w, device="cuda").to(dtype)
    output = torch.ops.trtllm.cublas_mm(
        x,
        w.t(),
        bias=None,
        out_dtype=None,
    )
    ref = torch.matmul(x, w.t())
    np.testing.assert_allclose(ref.float().cpu(), output.float().cpu(), atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "k_n",
    [(7168, 256)],
)
@pytest.mark.parametrize(
    "m",
    [1, 8, 512, 1024],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_cublas_mm_out_fp32(dtype, m, k_n):
    k, n = k_n
    torch.random.manual_seed(44)
    shape_x = (m, k)
    shape_w = (n, k)
    x = torch.randn(shape_x, device="cuda").to(dtype)
    w = torch.randn(shape_w, device="cuda").to(dtype)
    output = torch.ops.trtllm.cublas_mm(
        x,
        w.t(),
        bias=None,
        out_dtype=torch.float32,
    )
    ref = F.linear(x.float(), w.float())
    np.testing.assert_allclose(ref.float().cpu(), output.float().cpu(), atol=0.01, rtol=0.01)


@pytest.mark.parametrize("m", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_cublas_mm_tuned_graph(m, use_bias):
    from tensorrt_llm._torch.autotuner import autotune

    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("BF16 tactic tuning requires SM121")
    torch.manual_seed(19)
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(384, device="cuda", dtype=x.dtype) if use_bias else None
    with autotune():
        output = torch.ops.trtllm.cublas_mm_tuned(x, w.t(), bias, None)
    ref = F.linear(x.float(), w.float(), bias.float() if use_bias else None)
    torch.testing.assert_close(output.float(), ref, atol=0.25, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops.trtllm.cublas_mm_tuned(x, w.t(), bias, None)
    graph.replay()
    torch.testing.assert_close(captured, output)


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("use_bias", [False, True])
def test_cublas_mm_tactic_zero_matches_cublas_mm(m, use_bias):
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("BF16 tactics are only used on SM121")
    torch.manual_seed(23)
    # The SM121 table has no entry for this shape, so today's choice is cuBLASLt's own.
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(384, device="cuda", dtype=x.dtype) if use_bias else None
    expected = torch.ops.trtllm.cublas_mm(x, w.t(), bias, None)
    actual = torch.ops.trtllm.cublas_mm_tactic(x, w.t(), bias, None, 0, None, 0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["unaligned", "m17"])
def test_cublas_mm_tuned_fallback(case):
    from tensorrt_llm._torch.autotuner import autotune

    torch.manual_seed(29)
    if case == "unaligned":
        # A 16-byte offset keeps cuBLASLt's own alignment but fails the 256-byte guard.
        x = torch.randn(2 * 256 + 8, device="cuda", dtype=torch.bfloat16)[8:].view(2, 256)
    else:
        x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    expected = torch.ops.trtllm.cublas_mm(x, w.t(), None, None)
    with autotune():
        actual = torch.ops.trtllm.cublas_mm_tuned(x, w.t(), None, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    test_cublas_mm(torch.float16, 12, (8192, 10240))
