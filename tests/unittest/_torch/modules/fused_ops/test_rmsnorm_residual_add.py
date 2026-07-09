# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Parity tests: fused post-norm residual kernels (modules/fused_ops/
rmsnorm_residual_add) vs the unfused chains.

Covers the three variants served by the shared kernel body:
- ``rmsnorm_residual_add_scale`` (norm + residual add + fp32 scalar mul +
  bf16 cast),
- ``rmsnorm_residual_add`` (no scalar stage),
- the optional second output (the RMSNorm of the bf16-rounded primary
  output, e.g. the next decoder layer's input norm).

Each reference reproduces the exact serving-path op sequence it replaces
(flashinfer_rmsnorm + aten ops). The only tolerated difference is the fp32
reduction-order inside the sums of squares (one-step bf16 flips, measured
~5e-6 at serving shapes on SM100).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_residual_add import (
    rmsnorm_residual_add,
    rmsnorm_residual_add_scale,
)


def _reference(x, r, w, sc, eps):
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)
    s = r + n
    t = s * sc
    return t.to(torch.bfloat16)


def _mismatch(a, b):
    return (a.view(torch.uint16) != b.view(torch.uint16)).float().mean().item()


def _check(x, r, w, sc, eps):
    fused = rmsnorm_residual_add_scale(x, r, w, sc, eps)
    ref = _reference(x, r, w, sc, eps)
    mm = _mismatch(fused, ref)
    assert mm < 1e-4, f"bf16 mismatch fraction {mm}"


# 5376 is the Gemma4-31B hidden size; 228 the pinned decode batch; 7700 the
# round-3 profiled prefill token count; 333 exercises the row tail.
@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_rmsnorm_residual_add_scale_parity(n_tokens, hidden):
    torch.manual_seed(1234)
    x = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    sc = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    _check(x, r, w, sc, 1e-6)


@pytest.mark.parametrize("scalar", [0.987654, 2.5, 0.0])
def test_rmsnorm_residual_add_scale_nontrivial_scalar(scalar):
    """layer_scalar loads from the checkpoint - never assume 1.0."""
    torch.manual_seed(7)
    x = torch.randn((333, 5376), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((333, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    sc = torch.tensor([scalar], dtype=torch.float32, device="cuda")
    _check(x, r, w, sc, 1e-6)


def test_rmsnorm_residual_add_scale_non_pow2_hidden_and_strided():
    """Masked tail columns (non-power-of-2 H) + row-strided inputs."""
    torch.manual_seed(3)
    n, h = 65, 384
    xbuf = torch.randn((n, h + 128), dtype=torch.bfloat16, device="cuda")
    rbuf = torch.randn((n, h + 64), dtype=torch.bfloat16, device="cuda")
    x, r = xbuf[:, :h], rbuf[:, :h]
    w = torch.rand((h,), dtype=torch.bfloat16, device="cuda") + 0.5
    sc = torch.tensor([1.25], dtype=torch.float32, device="cuda")
    fused = rmsnorm_residual_add_scale(x, r, w, sc, 1e-6)
    ref = _reference(x.contiguous(), r.contiguous(), w, sc, 1e-6)
    mm = _mismatch(fused, ref)
    assert mm < 1e-4, f"bf16 mismatch fraction {mm}"


@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_rmsnorm_residual_add_parity(n_tokens, hidden):
    """Post-attention variant: rmsnorm(x) then the aten bf16 residual add."""
    torch.manual_seed(42)
    x = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    fused = rmsnorm_residual_add(x, r, w, 1e-6)
    ref = r + torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6)
    mm = _mismatch(fused, ref)
    assert mm < 1e-4, f"bf16 mismatch fraction {mm}"


@pytest.mark.parametrize("n_tokens", [1, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_rmsnorm_residual_add_scale_dual_norm_parity(n_tokens, hidden):
    """Tail with the secondary next-layer input-norm output.

    The primary output must match the plain tail; the secondary output must
    match a standalone flashinfer_rmsnorm applied to the (bf16-rounded)
    primary output - that is exactly the tensor the next layer's standalone
    input norm would read.
    """
    torch.manual_seed(11)
    x = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    r = torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    w2 = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    sc = torch.tensor([0.987654], dtype=torch.float32, device="cuda")
    out, n2 = rmsnorm_residual_add_scale(x, r, w, sc, 1e-6, next_norm_weight=w2, next_norm_eps=1e-6)
    ref = _reference(x, r, w, sc, 1e-6)
    mm = _mismatch(out, ref)
    assert mm < 1e-4, f"primary-output bf16 mismatch fraction {mm}"
    ref_n2 = torch.ops.trtllm.flashinfer_rmsnorm(out, w2, 1e-6)
    mm2 = _mismatch(n2, ref_n2)
    assert mm2 < 1e-4, f"secondary-norm bf16 mismatch fraction {mm2}"


if __name__ == "__main__":
    test_rmsnorm_residual_add_scale_parity(7700, 5376)
    test_rmsnorm_residual_add_scale_parity(228, 5376)
    test_rmsnorm_residual_add_scale_nontrivial_scalar(0.987654)
    test_rmsnorm_residual_add_scale_non_pow2_hidden_and_strided()
    test_rmsnorm_residual_add_parity(7700, 5376)
    test_rmsnorm_residual_add_parity(228, 5376)
    test_rmsnorm_residual_add_scale_dual_norm_parity(7700, 5376)
    test_rmsnorm_residual_add_scale_dual_norm_parity(228, 5376)
    print("ALL PARITY CHECKS PASSED")
