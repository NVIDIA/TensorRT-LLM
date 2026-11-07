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
"""Tests for the MiniMax-H3 transformer-block AdaLN kernels: norm1 (CuTe DSL) and the gated residual + norm2 (CUDA)."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.fused_adaln import (
    h3_gate_res_norm_mod,
    h3_norm_mod,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.native_ops import load_ops
from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.norm_scale_shift_cute import (
    rmsnorm_indexed_scale_shift_op,
)
from tensorrt_llm._utils import get_sm_version

_EPS = 1e-5

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _inputs(hidden, rows=1029, n_mod=6, seed=3):
    """Activations ``[rows, hidden]``, an AdaLN table ``[n_mod, 6 * hidden]`` and one table row per token."""
    torch.manual_seed(seed)
    x = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    attn = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    mod = (0.5 * torch.randn((n_mod, 6 * hidden), device="cuda")).to(torch.bfloat16)
    idx = torch.randint(0, n_mod, (rows,), device="cuda")
    return x, attn, weight, mod, idx


def _modulated_rmsnorm(x, weight, shift, scale, idx, eps):
    """FP32 ``rmsnorm(x) * weight * (1 + scale[idx]) + shift[idx]``."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return normed * (1.0 + scale.index_select(0, idx).float()) + shift.index_select(0, idx).float()


@requires_cuda
@pytest.mark.parametrize("hidden", [256, 5376])  # one warp / three warps, seven pieces per thread
def test_cute_norm_scale_shift_matches_norm_mod(hidden):
    """The CuTe norm1 op, given norm1's (shift, scale) table slices, matches ``h3_norm_mod``.

    Both round an FP32 result to BF16 (``h3_norm_mod`` also rounds the normalized value first), so
    each is within one BF16 ulp of the output scale of the FP32 formula, and within two of each other.
    """
    x, _, weight, mod, idx = _inputs(hidden)
    shift, scale = mod[:, :hidden], mod[:, hidden : 2 * hidden]
    actual = rmsnorm_indexed_scale_shift_op(x, weight, shift, scale, idx, _EPS)
    expected = _modulated_rmsnorm(x, weight, shift, scale, idx, _EPS)
    triton = h3_norm_mod(x, weight, mod, idx, _EPS, 1, 0)
    assert actual.shape == x.shape and actual.dtype == torch.bfloat16
    bound = expected.abs().max() * 2**-7
    assert (actual.float() - expected).abs().max() <= bound
    assert (actual.float() - triton.float()).abs().max() <= 2 * bound


@pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103)),
    reason="needs an SM100/SM103 GPU",
)
def test_cuda_gate_resid_norm_matches_gate_res_norm_mod():
    """The CUDA norm2 op, given (gate, shift, scale) slices in the block's order, matches ``h3_gate_res_norm_mod``.

    ``h = x + gate * attn`` is one FP32 rounding in both kernels, so it is bit-identical. ``y`` normalizes ``h``
    and rounds once (``h3_gate_res_norm_mod`` also rounds the normalized value first), so each is within one BF16
    ulp of the output scale of the FP32 formula, and within two of each other.
    """
    hidden = 5376
    x, attn, weight, mod, idx = _inputs(hidden)
    gate, shift, scale = (mod[:, c * hidden : (c + 1) * hidden] for c in (2, 3, 4))
    y, h = load_ops().gate_resid_rmsnorm_indexed_shift_scale(
        x, attn, weight, gate, shift, scale, idx, _EPS
    )
    triton_h, triton_y = h3_gate_res_norm_mod(x, attn, weight, mod, idx, _EPS, 2, 4, 3)
    assert torch.equal(h, triton_h)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    expected = _modulated_rmsnorm(h, weight, shift, scale, idx, _EPS)
    bound = expected.abs().max() * 2**-7
    assert (y.float() - expected).abs().max() <= bound
    assert (y.float() - triton_y.float()).abs().max() <= 2 * bound
