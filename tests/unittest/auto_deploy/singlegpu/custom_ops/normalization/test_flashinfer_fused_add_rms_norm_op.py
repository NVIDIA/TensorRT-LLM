# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (
    flashinfer_fused_add_rms_norm,
)


def rms_norm_ref(x, weight, eps):
    """Reference implementation of RMSNorm using PyTorch ops."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_flashinfer_fused_add_rms_norm_kernel(dtype, hidden_size):
    bsz = 4
    seq_len = 128
    eps = 1e-6

    # Create inputs
    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Clone for reference
    x_ref = x.clone()
    residual_ref = residual.clone()

    residual_ref_out = x_ref + residual_ref
    x_ref_out = rms_norm_ref(residual_ref_out, weight, eps)

    # Run kernel (Our fused op)
    x_out, residual_out = flashinfer_fused_add_rms_norm(x, residual, weight, eps)

    rtol, atol = (1e-2, 1e-2)

    torch.testing.assert_close(residual_out, residual_ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_out, x_ref_out, rtol=rtol, atol=atol)

    # Verify in-place modification happened
    assert x is x_out
    assert residual is residual_out
