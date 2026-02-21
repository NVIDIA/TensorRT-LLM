# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the fused GDN gating custom ops (torch + triton)."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.fla import (
    gdn_gating as _gdn_gating_ops,  # noqa: F401
)


def _reference_gdn_gating(A_log, a, dt_bias, beta=1.0, threshold=20.0):
    """Pure-torch reference: g = -exp(A_log) * softplus(a + dt_bias)."""
    return -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), beta, threshold)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 8, 32])
@pytest.mark.parametrize("num_heads", [8, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestFusedGdnGating:
    """Tests for torch_fused_gdn_gating and triton_fused_gdn_gating ops."""

    def test_torch_op_matches_reference(self, batch_size, seq_len, num_heads, dtype):
        """torch_fused_gdn_gating matches the inline reference computation."""
        A_log = torch.randn(num_heads, device="cuda", dtype=dtype)
        a = torch.randn(batch_size, seq_len, num_heads, device="cuda", dtype=dtype)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=dtype)

        ref = _reference_gdn_gating(A_log, a, dt_bias)
        out = torch.ops.auto_deploy.torch_fused_gdn_gating(A_log, a, dt_bias)

        assert out.dtype == torch.float32
        assert out.shape == (batch_size, seq_len, num_heads)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_triton_op_matches_reference(self, batch_size, seq_len, num_heads, dtype):
        """triton_fused_gdn_gating matches the inline reference computation."""
        A_log = torch.randn(num_heads, device="cuda", dtype=dtype)
        a = torch.randn(batch_size, seq_len, num_heads, device="cuda", dtype=dtype)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=dtype)

        ref = _reference_gdn_gating(A_log, a, dt_bias)
        out = torch.ops.auto_deploy.triton_fused_gdn_gating(A_log, a, dt_bias)

        assert out.dtype == torch.float32
        assert out.shape == (batch_size, seq_len, num_heads)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_torch_and_triton_match(self, batch_size, seq_len, num_heads, dtype):
        """torch and triton ops produce identical results."""
        A_log = torch.randn(num_heads, device="cuda", dtype=dtype)
        a = torch.randn(batch_size, seq_len, num_heads, device="cuda", dtype=dtype)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=dtype)

        out_torch = torch.ops.auto_deploy.torch_fused_gdn_gating(A_log, a, dt_bias)
        out_triton = torch.ops.auto_deploy.triton_fused_gdn_gating(A_log, a, dt_bias)

        torch.testing.assert_close(out_torch, out_triton, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (2.0, 10.0)])
def test_softplus_parameters(beta, threshold):
    """Verify that non-default softplus beta/threshold are respected."""
    num_heads = 16
    A_log = torch.randn(num_heads, device="cuda", dtype=torch.float16)
    a = torch.randn(2, 4, num_heads, device="cuda", dtype=torch.float16)
    dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float16)

    ref = _reference_gdn_gating(A_log, a, dt_bias, beta, threshold)
    out_torch = torch.ops.auto_deploy.torch_fused_gdn_gating(A_log, a, dt_bias, beta, threshold)
    out_triton = torch.ops.auto_deploy.triton_fused_gdn_gating(A_log, a, dt_bias, beta, threshold)

    torch.testing.assert_close(out_torch, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_triton, ref, atol=1e-5, rtol=1e-5)
