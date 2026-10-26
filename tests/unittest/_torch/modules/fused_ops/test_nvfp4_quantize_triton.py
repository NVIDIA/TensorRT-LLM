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
"""Bitwise test: Triton NVFP4 quantize (modules/fused_ops/nvfp4_quantize_triton)
against torch.ops.trtllm.fp4_quantize, both scale-factor layouts."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
from tensorrt_llm._torch.modules.fused_ops import nvfp4_quantize_triton as nq
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100 or nq.triton is None,
    reason="NVFP4 quantization needs SM100+ and Triton",
)


def _inputs(m: int, k: int):
    torch.manual_seed(m * 131 + k)
    x = (torch.randn(m, k, device="cuda") * torch.rand(m, 1, device="cuda") * 4).to(torch.bfloat16)
    x[0, :16] = 0  # an all-zero block: SF 0, values 0
    if m > 2:
        x[2, 5] = 1e4  # an outlier that saturates its block's values
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1).to(torch.float32)
    return x, gs


@pytest.mark.parametrize("swizzled", [True, False])
@pytest.mark.parametrize("k", [512, 2048, 7168, 16384])
@pytest.mark.parametrize("m", [1, 3, 127, 128, 129, 1000, 1024])
def test_bitwise_vs_fp4_quantize(m, k, swizzled):
    x, gs = _inputs(m, k)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, swizzled)
    q, sf = nq.nvfp4_quantize_triton(x, gs, swizzled)
    assert q.shape == q_ref.shape
    assert sf.numel() == sf_ref.numel()
    assert torch.equal(q, q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref.view(torch.uint8).flatten())


def test_dispatch(monkeypatch):
    x, gs = _inputs(64, 7168)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    for env in ("0", "1"):
        monkeypatch.setenv(nq.NVFP4_QUANTIZE_TRITON_ENV, env)
        q, sf = nq.fp4_quantize(x, gs, 16, False, False)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(sf.view(torch.uint8).flatten(), sf_ref.view(torch.uint8).flatten())
    # Unsupported inputs (non-contiguous, K % 64 != 0) take the C++ op.
    monkeypatch.setenv(nq.NVFP4_QUANTIZE_TRITON_ENV, "1")
    assert not nq.nvfp4_quantize_triton_supported(x[:, :7104], gs)
    assert not nq.nvfp4_quantize_triton_supported(x.t(), gs)
