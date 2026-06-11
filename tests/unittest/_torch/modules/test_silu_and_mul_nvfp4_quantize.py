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
"""Unit tests for the fused SwiGLU + NVFP4-quantize kernel.

``silu_and_mul_nvfp4_quantize`` folds the standalone NVFP4 input-quantize
(FP16/BF16 -> NVFP4) into the SwiGLU activation that produces it, at the
GatedMLP ``gate_up_proj -> swiglu -> down_proj`` site (dense MLP + shared
experts on DeepSeek-V3.2 / Kimi-K2.5). It targets small problem sizes where the
constrained GEMM+SwiGLU+quant fusion cannot run.

Input is ``[M, 2N]`` (gate half in ``[0, N)``, up half in ``[N, 2N)``); output
is the NVFP4-quantized ``[M, N]`` activation plus its swizzled scale factors.

Accuracy is checked against the unfused reference the fusion replaces: the
Triton ``silu_and_mul`` (sigmoid(a) * a * b) followed by
``torch.ops.trtllm.fp4_quantize``. The fused kernel computes the gate in
registers while already reading the tensor to quantize it, so a handful of
values right at quantization boundaries can fall on the other side of a step due
to FMA-vs-separate rounding; we require a >= 99% match on the packed FP4 bytes,
mirroring ``test_fused_rmsnorm_fp4_quantize.py``.
"""

import pytest
import torch

from tests.unittest.utils.util import getSMVersion


def silu_and_mul_nvfp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(
        torch.ops.trtllm, "silu_and_mul_nvfp4_quantize")


def fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm,
                                                    "fp4_quantize")


skip_unless_swiglu = pytest.mark.skipif(
    getSMVersion() < 100 or not silu_and_mul_nvfp4_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm "
    "silu_and_mul_nvfp4_quantize + fp4_quantize ops",
)


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU matching the Triton silu_and_mul_kernel
    (tensorrt_llm/_torch/modules/swiglu.py:42): sigmoid(a) * a * b, computed in
    fp32 then cast back to the input dtype."""
    d = x.shape[-1] // 2
    a = x[:, :d].to(torch.float32)
    b = x[:, d:].to(torch.float32)
    out = torch.sigmoid(a) * a * b
    return out.to(x.dtype)


def fp4_quantize_ref(act: torch.Tensor, sf_scale: torch.Tensor):
    """Unfused baseline: the standalone NVFP4 quantize the fusion replaces."""
    return torch.ops.trtllm.fp4_quantize(
        act.contiguous(),
        sf_scale,
        16,
        False,  # use_ue8m0
        True,  # is_sf_swizzled_layout
    )


def make_sf_scale(act: torch.Tensor) -> torch.Tensor:
    """Per-tensor SF scale (amax / (6 * 448)), the same scalar fed to both the
    fused op and fp4_quantize -- both forward it to cvt_warp_fp16_to_fp4."""
    return (act.abs().amax().float() / (6.0 * 448.0)).to(act.device).view(1)


def assert_fp4_match(fp4_fused: torch.Tensor, fp4_ref: torch.Tensor, ctx: str):
    assert fp4_fused.shape == fp4_ref.shape, (
        f"shape mismatch {tuple(fp4_fused.shape)} vs {tuple(fp4_ref.shape)} "
        f"({ctx})")
    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= 0.99, (
        f"FP4 packed match rate {match_rate:.4f} < 0.99 ({ctx})")


@skip_unless_swiglu
@pytest.mark.parametrize("m", [1, 4, 17, 64, 256])
@pytest.mark.parametrize("n", [16, 128, 512, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_nvfp4_quantize_vs_separate(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Input is [M, 2N]: gate half then up half.
    x = torch.randn(m, 2 * n, dtype=dtype, device=device)

    act_ref = silu_and_mul_ref(x)
    sf_scale = make_sf_scale(act_ref)
    fp4_ref, _ = fp4_quantize_ref(act_ref, sf_scale)

    quant_out, scale_out = torch.ops.trtllm.silu_and_mul_nvfp4_quantize(
        x, sf_scale, 16)

    assert quant_out.shape == (m, n // 2), "output must be packed [M, N/2]"
    assert_fp4_match(quant_out, fp4_ref, f"swiglu m={m} n={n} {dtype}")


@skip_unless_swiglu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_nvfp4_quantize_does_not_mutate_input(dtype):
    """The kernel only reads the input; it must not write back into it."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    m, n = 32, 512

    x = torch.randn(m, 2 * n, dtype=dtype, device=device)
    x_before = x.clone()
    sf_scale = make_sf_scale(silu_and_mul_ref(x))

    torch.ops.trtllm.silu_and_mul_nvfp4_quantize(x, sf_scale, 16)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
