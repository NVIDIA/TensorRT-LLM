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
"""LayerNorm module dtype handling.

The module upcasts activations to FP32 for the normalization while its
parameters stay in the model dtype. aten layer_norm rejects mixed
input/weight dtypes (both the CUDA kernel and fake-tensor tracing), so with
bf16/fp16 parameters the forward used to raise — first observed as a dynamo
fake-tensor failure in the Qwen3-VL vision tower (whose LayerNorms take the
text dtype). These tests pin that every parameter dtype works, eagerly and
under torch.compile, and matches the FP32 reference.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


HIDDEN = 1152
EPS = 1e-6


def _reference(x, weight, bias, residual=None):
    x = x.float()
    if residual is not None:
        x = x + residual.float()
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight.float(), bias.float(), EPS)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("with_residual", [False, True], ids=["plain", "residual"])
@requires_cuda
def test_layer_norm_param_dtypes(dtype, with_residual):
    torch.manual_seed(0)
    module = LayerNorm(hidden_size=HIDDEN, eps=EPS, dtype=dtype, device="cuda")
    with torch.no_grad():
        torch.nn.init.normal_(module.weight)
        torch.nn.init.normal_(module.bias)
    x = torch.randn(4, HIDDEN, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None

    if with_residual:
        out, residual_out = module(x, residual=residual)
        assert residual_out.dtype == dtype
    else:
        out = module(x)
    assert out.dtype == dtype

    ref = _reference(x, module.weight, module.bias, residual).to(dtype)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_cuda
def test_layer_norm_bf16_params_under_compile():
    """The vision-tower shape of the original failure: bf16 parameters,
    forward traced by dynamo (fullgraph, as inside a compiled model)."""
    torch.manual_seed(0)
    module = LayerNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    compiled = torch.compile(module.forward, fullgraph=True, dynamic=True, backend="eager")
    out, residual_out = compiled(x, residual=residual)
    ref = _reference(x, module.weight, module.bias, residual).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    assert residual_out.dtype == torch.bfloat16
