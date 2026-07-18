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
"""Parity test: fused Triton FP8 dynamic-quant (modules/fused_ops/
fp8_dynamic_quant) vs the reference CUDA op
``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor``.

The fused path replaces only the amax->scale half with a two-stage Triton
reduction and reuses the apply-only CUDA op, so for finite inputs both the
scale and the quantized bytes must be *bitwise-identical* to the reference op
(the scale is the ``cublas_scaled_mm`` scale_a, so anything but bit-parity
would perturb the GEMM). A separate case guards the non-contiguous path.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm torch ops)
from tensorrt_llm._torch.modules.fused_ops.fp8_dynamic_quant import (
    fp8_dynamic_quant_scale,
    fp8_dynamic_quantize,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused FP8 dynamic-quant kernels require CUDA"
)


def _reference(x: torch.Tensor):
    """The op this kernel replaces: returns (qfp8, fp32 scale flattened to [1]).

    The op returns the scale in the input dtype with shape ``(1, 1)``; the fused
    helper returns fp32 ``(1,)``. Flatten + widen so parity is compared on value,
    not layout.
    """
    q, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    return q, s.to(torch.float32).reshape(-1)


# bf16/fp16 are the activation dtypes this FP8 dynamic-quant path actually sees;
# for both, the fused helper is bitwise-identical to the reference op.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1), (1, 128), (64, 2048), (17, 4096), (256, 512)])
def test_bitwise_parity_vs_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda") * 12.0

    ref_q, ref_s = _reference(x)
    q, s = fp8_dynamic_quantize(x)

    # Scale: bitwise-identical fp32 value.
    assert torch.equal(s.reshape(-1), ref_s), (s, ref_s)
    # Quantized payload: bitwise-identical bytes (reinterpret fp8 as int8).
    assert torch.equal(q.view(torch.int8), ref_q.view(torch.int8))


@pytest.mark.parametrize("shape", [(1, 128), (64, 2048), (256, 512)])
def test_fp32_scale_near_parity(shape):
    # fp32 inputs are not a real path here (activations are bf16/fp16), and the
    # op stores an fp32 scale with no dtype rounding, so the last fp32 bit of a
    # multiply-by-reciprocal (this kernel) vs a divide (the op) is exposed. The
    # scales agree to well within an fp32 ULP; bf16/fp16 rounding masks this
    # difference entirely (hence exact parity above).
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cuda") * 12.0
    _, ref_s = _reference(x)
    s = fp8_dynamic_quant_scale(x)
    assert torch.allclose(s.reshape(-1), ref_s, rtol=1e-5, atol=0.0), (s, ref_s)


@pytest.mark.parametrize("scale_mul", [1e-6, 1.0, 1e4])
def test_scale_parity_across_magnitudes(scale_mul):
    # Tiny magnitudes exercise the MIN_SCALING_FACTOR clamp; large ones exercise
    # amax well above the e4m3 max (448).
    torch.manual_seed(0)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16) * scale_mul
    _, ref_s = _reference(x)
    s = fp8_dynamic_quant_scale(x)
    assert torch.equal(s.reshape(-1), ref_s), (scale_mul, s, ref_s)


def test_noncontiguous_input():
    # A strided view must reduce over its LOGICAL elements: equal to the result
    # on a contiguous copy (guards the `.contiguous()` handling), and equal to
    # the reference op on the same logical values.
    torch.manual_seed(0)
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 8.0
    view = x[:, ::2]
    assert not view.is_contiguous()

    s_view = fp8_dynamic_quant_scale(view)
    s_copy = fp8_dynamic_quant_scale(view.contiguous())
    assert torch.equal(s_view, s_copy)

    _, ref_s = _reference(view.contiguous())
    assert torch.equal(s_view.reshape(-1), ref_s)
