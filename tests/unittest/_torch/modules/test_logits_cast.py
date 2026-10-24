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

"""The vectorized logits widening is bit-identical to ``Tensor.float()``."""

import pytest
import torch

from tensorrt_llm._torch.modules import logits_cast
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _special_values(dtype):
    finfo = torch.finfo(dtype)
    return torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            finfo.max,
            finfo.min,
            finfo.tiny,
            finfo.tiny / 4,
            -finfo.tiny / 4,
            1.0,
            -1.0,
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 163840), (7, 1001), (240, 163840), (3, 5, 4099)])
def test_widen_matches_float_bitwise(dtype, shape):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 30).to(dtype)
    flat = x.view(-1)
    special = _special_values(dtype).cuda()
    flat[: special.numel()] = special
    out = logits_cast.widen_logits_to_fp32(x)
    ref = x.float()
    assert out.dtype == torch.float32 and out.shape == ref.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_non_contiguous_and_fp32_take_tensor_float():
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert torch.equal(logits_cast.widen_logits_to_fp32(x), x.float())
    y = torch.randn(4, 8, device="cuda")
    assert logits_cast.widen_logits_to_fp32(y) is not None
    assert torch.equal(logits_cast.widen_logits_to_fp32(y), y)


def test_logits_processor_flag(monkeypatch):
    torch.manual_seed(1)
    head = torch.nn.Linear(512, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
    monkeypatch.delenv(logits_cast.FAST_LOGITS_CAST_ENV, raising=False)
    ref = LogitsProcessor()(hidden, head, None, True)
    monkeypatch.setenv(logits_cast.FAST_LOGITS_CAST_ENV, "1")
    out = LogitsProcessor()(hidden, head, None, True)
    assert out.dtype == torch.float32
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_capturable_in_cuda_graph():
    x = torch.randn(16, 163840, device="cuda", dtype=torch.bfloat16)
    logits_cast.widen_logits_to_fp32(x)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = logits_cast.widen_logits_to_fp32(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, x.float())
