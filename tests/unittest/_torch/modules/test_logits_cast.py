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
import triton

from tensorrt_llm._torch.modules import logits_cast, logits_processor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def kernel_launches(monkeypatch):
    """Record the grid of each widening kernel launch."""
    launches = []
    kernel = logits_cast._widen_to_fp32_kernel

    class RecordingKernel:
        def __getitem__(self, grid):
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(logits_cast, "_widen_to_fp32_kernel", RecordingKernel())
    return launches


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
# A partial last block in 2D, full blocks only, and a 3D tensor that spans many blocks.
@pytest.mark.parametrize("shape", [(7, 1001), (4, 4096), (3, 5, 4099)])
def test_widen_matches_float_bitwise(dtype, shape, kernel_launches):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 30).to(dtype)
    flat = x.view(-1)
    special = _special_values(dtype).cuda()
    flat[: special.numel()] = special
    out = logits_cast.widen_logits_to_fp32(x)
    ref = x.float()
    assert len(kernel_launches) == 1
    assert out.dtype == torch.float32 and out.shape == ref.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_unsupported_inputs_take_tensor_float(kernel_launches, monkeypatch):
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert torch.equal(logits_cast.widen_logits_to_fp32(x), x.float())
    y = torch.randn(4, 8, device="cuda")
    assert torch.equal(logits_cast.widen_logits_to_fp32(y), y)
    z = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    logits_cast.widen_logits_to_fp32(z).sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
    w = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert torch.equal(logits_cast.widen_logits_to_fp32(w), w.float())
    assert not kernel_launches


def test_logits_processor_takes_kernel(kernel_launches):
    torch.manual_seed(1)
    head = torch.nn.Linear(512, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
    # The executor runs the model under inference mode, so the logits take the kernel.
    with torch.inference_mode():
        out = logits_processor.LogitsProcessor()(hidden, head, None, True)
        ref = head(hidden).float()
    assert len(kernel_launches) == 1
    assert out.dtype == torch.float32
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


# A vocab that is a multiple of 16, and one that is not, so numel is odd for
# 7 rows and a multiple of 16 for 16 rows.
@pytest.mark.parametrize("vocab", [163840, 1001])
def test_capturable_in_cuda_graph(vocab, monkeypatch):
    # Compile for one row count outside the capture; the captured row counts reuse it.
    logits_cast.widen_logits_to_fp32(torch.randn(1, vocab, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    compiles = []
    monkeypatch.setattr(
        triton.knobs.runtime,
        "jit_post_compile_hook",
        lambda **kwargs: compiles.append(kwargs["repr"]),
    )
    for rows in (7, 16):
        x = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = logits_cast.widen_logits_to_fp32(x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, x.float())
    assert not compiles
