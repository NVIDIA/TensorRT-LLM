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

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, unswizzle_sf
from tensorrt_llm.math_utils import ceil_div, pad_up
from tests.unittest.utils.util import getSMVersion


def fused_gated_rmsnorm_quant_available():
    """Check if the fused_gated_rmsnorm_quant op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_gated_rmsnorm_quant")


skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for RMSNorm triton kernels",
)

skip_unless_nvfp4_kernel = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_gated_rmsnorm_quant_available(),
    reason="Requires SM100+ (Blackwell) and trtllm.fused_gated_rmsnorm_quant op",
)


def reference_rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None = None,
    eps: float = 1e-5,
    norm_before_gate: bool = True,
    group_size: int | None = None,
) -> torch.Tensor:
    """Reference implementation of gated RMSNorm."""

    def silu(t):
        return t * torch.sigmoid(t)

    if z is None:
        if group_size is None or group_size == x.shape[-1]:
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps)
            return weight * x_norm
        else:
            hidden_size = x.shape[-1]
            num_groups = hidden_size // group_size
            x_reshaped = x.reshape(*x.shape[:-1], num_groups, group_size)
            variance = x_reshaped.pow(2).mean(-1, keepdim=True)
            x_norm = x_reshaped * torch.rsqrt(variance + eps)
            x_norm = x_norm.reshape(*x.shape[:-1], hidden_size)
            return weight * x_norm
    else:
        z_activated = silu(z)
        if norm_before_gate:
            if group_size is None or group_size == x.shape[-1]:
                variance = x.pow(2).mean(-1, keepdim=True)
                x_norm = x * torch.rsqrt(variance + eps)
            else:
                hidden_size = x.shape[-1]
                num_groups = hidden_size // group_size
                x_reshaped = x.reshape(*x.shape[:-1], num_groups, group_size)
                variance = x_reshaped.pow(2).mean(-1, keepdim=True)
                x_norm = x_reshaped * torch.rsqrt(variance + eps)
                x_norm = x_norm.reshape(*x.shape[:-1], hidden_size)
            return weight * x_norm * z_activated
        else:
            x_gated = x * z_activated
            if group_size is None or group_size == x.shape[-1]:
                variance = x_gated.pow(2).mean(-1, keepdim=True)
                x_norm = x_gated * torch.rsqrt(variance + eps)
            else:
                hidden_size = x.shape[-1]
                num_groups = hidden_size // group_size
                x_reshaped = x_gated.reshape(*x.shape[:-1], num_groups, group_size)
                variance = x_reshaped.pow(2).mean(-1, keepdim=True)
                x_norm = x_reshaped * torch.rsqrt(variance + eps)
                x_norm = x_norm.reshape(*x.shape[:-1], hidden_size)
            return weight * x_norm


@skip_no_cuda
class TestRMSNormBasic:
    def test_basic_rmsnorm(self):
        hidden_size = 2048
        batch_size = 4
        device = "cuda"
        dtype = torch.float16

        torch.manual_seed(0)
        norm = RMSNorm(hidden_size, eps=1e-5).to(device).to(dtype)
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == dtype
        assert not torch.isnan(output).any()

    def test_gated_rmsnorm(self):
        hidden_size = 2048
        batch_size = 4
        device = "cuda"
        dtype = torch.float16

        torch.manual_seed(1)
        norm = RMSNorm(hidden_size, eps=1e-5, norm_before_gate=True).to(device).to(dtype)
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        output = norm(x, z=z)

        assert output.shape == x.shape
        assert output.dtype == dtype
        assert not torch.isnan(output).any()

    def test_compare_with_reference(self):
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16
        eps = 1e-5

        torch.manual_seed(2)
        norm = (
            RMSNorm(hidden_size, eps=eps, norm_before_gate=False, group_size=group_size)
            .to(device)
            .to(dtype)
        )
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

        output_triton = norm(x, z=z)
        output_ref = reference_rmsnorm_gated(
            x, norm.weight, z, eps=eps, norm_before_gate=False, group_size=group_size
        )

        torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)


@skip_no_cuda
class TestRMSNormNVFP4:
    @skip_unless_nvfp4_kernel
    def test_nvfp4_error_handling(self):
        hidden_size = 2048
        batch_size = 4
        device = "cuda"
        dtype = torch.float16

        norm = (
            RMSNorm(hidden_size, eps=1e-5, is_nvfp4=True, norm_before_gate=False)
            .to(device)
            .to(dtype)
        )
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="nvfp4_scale"):
            norm(x, z=z)

    @skip_unless_nvfp4_kernel
    def test_nvfp4_cuda_kernel(self):
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16

        norm = (
            RMSNorm(
                hidden_size, eps=1e-5, is_nvfp4=True, norm_before_gate=False, group_size=group_size
            )
            .to(device)
            .to(dtype)
        )
        norm.nvfp4_scale = torch.randn(1, device=device, dtype=torch.float32).abs()

        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        output = norm(x, z=z)

        assert isinstance(output, Fp4QuantizedTensor)
        assert output.fp4_tensor.dtype == torch.uint8
        assert output.is_sf_swizzled is True
        assert output.fp4_tensor.shape == (batch_size, hidden_size // 2)
        assert not torch.isnan(output.scaling_factor).any()
        assert not torch.isinf(output.scaling_factor).any()

    @skip_unless_nvfp4_kernel
    def test_nvfp4_fallback_to_triton(self):
        hidden_size = 2048
        batch_size = 4
        device = "cuda"
        dtype = torch.float16

        norm = (
            RMSNorm(hidden_size, eps=1e-5, is_nvfp4=True, norm_before_gate=True)
            .to(device)
            .to(dtype)
        )
        norm.nvfp4_scale = torch.randn(1, device=device, dtype=torch.float32).abs()
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        output = norm(x)

        assert isinstance(output, torch.Tensor)
        assert not isinstance(output, Fp4QuantizedTensor)
        assert output.shape == x.shape
        assert output.dtype == dtype


@skip_no_cuda
class TestRMSNormCUDAvsTriton:
    @skip_unless_nvfp4_kernel
    def test_cuda_triton_fp4_comparison(self):
        """Compare fused CUDA kernel (norm+FP4) vs Triton norm + separate fp4_quantize."""
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16
        eps = 1e-5
        sf_vec_size = 16

        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.empty(hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(weight, mean=1.0, std=0.1)

        # Compute sf_scale from the reference normalized output
        ref_normed = reference_rmsnorm_gated(
            x,
            weight,
            z,
            eps=eps,
            norm_before_gate=False,
            group_size=group_size,
        )
        sf_scale = (ref_normed.abs().amax().float() / (6.0 * 448.0)).view(1).to(device)

        # Path 1: Fused CUDA kernel (gated RMSNorm + FP4 quantization)
        norm_cuda = (
            RMSNorm(
                hidden_size, eps=eps, is_nvfp4=True, norm_before_gate=False, group_size=group_size
            )
            .to(device)
            .to(dtype)
        )
        norm_cuda.weight.data.copy_(weight)
        norm_cuda.nvfp4_scale = sf_scale

        # Path 2: Triton norm (full-precision) + separate fp4_quantize
        norm_triton = (
            RMSNorm(
                hidden_size, eps=eps, is_nvfp4=False, norm_before_gate=False, group_size=group_size
            )
            .to(device)
            .to(dtype)
        )
        norm_triton.weight.data.copy_(norm_cuda.weight.data)

        output_cuda_fp4 = norm_cuda(x, z=z)
        output_triton = norm_triton(x, z=z)

        assert isinstance(output_cuda_fp4, Fp4QuantizedTensor)
        assert isinstance(output_triton, torch.Tensor)

        # Quantize the Triton output with the same NVFP4 scheme
        fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
            output_triton.contiguous(),
            sf_scale,
            sf_vec_size,
            False,
            True,  # use_ue8m0=False, is_sf_swizzled_layout=True
        )

        # Compare FP4 packed values (byte-level)
        fp4_match_rate = (output_cuda_fp4.fp4_tensor == fp4_separate).float().mean().item()
        assert fp4_match_rate >= 0.99, f"FP4 packed values match rate {fp4_match_rate:.4f} < 0.99"

        # Compare scale factors: unswizzle with padded dimensions, then
        # slice the valid [batch_size, num_sf_cols] portion
        padded_rows = pad_up(batch_size, 128)
        num_sf_cols = ceil_div(hidden_size, sf_vec_size)
        padded_cols = pad_up(num_sf_cols, 4) * sf_vec_size

        cuda_sf_unswizzled = unswizzle_sf(
            output_cuda_fp4.scaling_factor, padded_rows, padded_cols, sf_vec_size
        )[:batch_size, :num_sf_cols]
        triton_sf_unswizzled = unswizzle_sf(sf_separate, padded_rows, padded_cols, sf_vec_size)[
            :batch_size, :num_sf_cols
        ]

        sf_match_rate = (cuda_sf_unswizzled == triton_sf_unswizzled).float().mean().item()
        assert sf_match_rate >= 0.99, f"Scale factor match rate {sf_match_rate:.4f} < 0.99"

    def test_triton_vs_reference(self):
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16
        eps = 1e-5

        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

        weight = torch.ones(hidden_size, device=device, dtype=dtype)
        output_ref = reference_rmsnorm_gated(
            x, weight, z, eps=eps, norm_before_gate=False, group_size=group_size
        )

        norm_triton = (
            RMSNorm(
                hidden_size, eps=eps, is_nvfp4=False, norm_before_gate=False, group_size=group_size
            )
            .to(device)
            .to(dtype)
        )
        norm_triton.weight.data.copy_(weight)
        output_triton = norm_triton(x, z=z)

        torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)
