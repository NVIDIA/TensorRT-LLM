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
"""Parity tests: the FP8-emitting variants of kernels/fused_ops/rmsnorm_residual_add.

Each variant is checked against the unfused chain it replaces: flashinfer_rmsnorm
(+ aten residual add / scalar mul) followed by static_quantize_e4m3_per_tensor of
the bf16 result, i.e. the quantize the consuming FP8 Linear would run. The only
tolerated difference is the fp32 reduction order in the sums of squares.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_residual_add import (
    rmsnorm_fp8_quant,
    rmsnorm_residual_add,
    rmsnorm_residual_add_scale,
)


def _rmsnorm(x, w):
    return torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6)


def _static_quant(x, qs):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, qs)[0]


def _mismatch(a, b):
    return (a.view(torch.uint8) != b.view(torch.uint8)).float().mean().item()


# The three FP8-emitting decoder-layer norms, each against the unfused chain it replaces:
#   input: rmsnorm -> FP8 quant (first layer's qkv input)
#   attn:  rmsnorm + residual add, plus the pre-FFN norm -> FP8 (gate_up input)
#   tail:  rmsnorm + residual add + layer scalar, plus the next input norm -> FP8 (qkv input)
# 2560 is the Gemma4-E4B hidden size.
@pytest.mark.parametrize("variant", ["input", "attn", "tail"])
@pytest.mark.parametrize("n_tokens", [1, 333, 4096])
@pytest.mark.parametrize("hidden", [2560, 5376])
@pytest.mark.parametrize("quant_scale", [0.03, 0.003])  # 0.03: FP8 ties; 0.003 saturates
def test_rmsnorm_fp8_variants_parity(variant, n_tokens, hidden, quant_scale):
    torch.manual_seed(13)
    x = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 3
    r = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    w2 = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    sc = torch.tensor([0.987654], dtype=torch.float32, device="cuda")
    qs = torch.tensor([quant_scale], dtype=torch.float32, device="cuda")
    if variant == "input":
        n2 = rmsnorm_fp8_quant(x, w, 1e-6, qs)
        ref_n2 = own_n2 = _static_quant(_rmsnorm(x, w), qs)
    else:
        if variant == "attn":
            out, n2 = rmsnorm_residual_add(
                x, r, w, 1e-6, next_norm_weight=w2, next_norm_eps=1e-6, next_norm_quant_scale=qs
            )
            ref = r + _rmsnorm(x, w)
        else:
            out, n2 = rmsnorm_residual_add_scale(
                x, r, w, sc, 1e-6, next_norm_weight=w2, next_norm_eps=1e-6, next_norm_quant_scale=qs
            )
            ref = ((r + _rmsnorm(x, w)) * sc).to(torch.bfloat16)
        assert out.dtype == torch.bfloat16
        mm = (out.view(torch.uint16) != ref.view(torch.uint16)).float().mean().item()
        assert mm < 1e-4, f"bf16 mismatch fraction {mm}"
        # A differing element is one bf16 step of the residual or norm term off (absolute, since the
        # sum may cancel); this rejects NaN and large errors.
        torch.testing.assert_close(out, ref, atol=2**-4, rtol=2**-7)
        # The next norm reads the bf16-rounded primary output, as a standalone norm would.
        ref_n2 = _static_quant(_rmsnorm(ref, w2), qs)
        # The same chain on the kernel's own primary output, which isolates the second norm's error.
        own_n2 = _static_quant(_rmsnorm(out, w2), qs)
    assert n2.dtype == torch.float8_e4m3fn and n2.shape == x.shape
    mm = _mismatch(n2, ref_n2)
    assert mm < 1e-4, f"fp8 mismatch fraction {mm}"
    # Against its own input, a differing element is at most one E4M3 step off (relative 2^-3, or the
    # 2^-9 subnormal step); this rejects NaN and large errors.
    torch.testing.assert_close(n2.float(), own_n2.float(), atol=2**-9, rtol=2**-3)
