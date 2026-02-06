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
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
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


def quantize_to_fp4_e2m1(
    tensor: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP4 E2M1 format with group-wise scaling."""
    device = tensor.device
    input_shape = tensor.shape
    hidden_size = input_shape[-1]
    batch_shape = input_shape[:-1]
    num_groups = hidden_size // group_size

    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=device,
    )

    tensor_grouped = tensor.reshape(*batch_shape, num_groups, group_size)
    group_max = tensor_grouped.abs().max(dim=-1, keepdim=True)[0]
    scaling_factors = group_max.clamp(min=1e-8)
    tensor_normalized = tensor_grouped / scaling_factors

    tensor_normalized_flat = tensor_normalized.reshape(-1, 1)
    e2m1_values_expanded = e2m1_values.unsqueeze(0)
    distances = (tensor_normalized_flat - e2m1_values_expanded).abs()
    indices = distances.argmin(dim=1)

    indices = indices.reshape(*batch_shape, num_groups, group_size)
    indices = indices.reshape(*batch_shape, hidden_size)

    indices_uint8 = indices.to(torch.uint8)
    low_nibbles = indices_uint8[:, 0::2]
    high_nibbles = indices_uint8[:, 1::2]
    packed = low_nibbles | (high_nibbles << 4)

    scaling_factors = scaling_factors.reshape(*batch_shape, num_groups)
    return packed, scaling_factors


def unpack_and_scale_fp4(
    fp4_packed: torch.Tensor, scaling_factors: torch.Tensor, target_shape: tuple, group_size: int
) -> torch.Tensor:
    """Unpack FP4 values and apply scaling to get float values."""
    device = fp4_packed.device

    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=device,
    )

    fp4_uint8 = fp4_packed.view(torch.uint8)
    low_nibble = fp4_uint8 & 0x0F
    high_nibble = (fp4_uint8 >> 4) & 0x0F

    unpacked = torch.stack([low_nibble, high_nibble], dim=-1)
    unpacked = unpacked.reshape(*target_shape)
    values = e2m1_lut[unpacked.long()]

    hidden_size = target_shape[-1]
    num_groups = hidden_size // group_size
    batch_shape = target_shape[:-1]

    values = values.reshape(*batch_shape, num_groups, group_size)
    sf = scaling_factors.reshape(*batch_shape, num_groups, 1)
    values = values * sf
    values = values.reshape(*target_shape)

    return values


@skip_no_cuda
class TestRMSNormBasic:
    def test_basic_rmsnorm(self):
        hidden_size = 2048
        batch_size = 4
        device = "cuda"
        dtype = torch.float16

        norm = RMSNorm(hidden_size, eps=1e-5).to(device).to(dtype)
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

        norm = RMSNorm(hidden_size, eps=1e-5, norm_before_gate=True).to(device).to(dtype)
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

        norm = (
            RMSNorm(hidden_size, eps=eps, norm_before_gate=False, group_size=group_size)
            .to(device)
            .to(dtype)
        )
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
        norm.nvfp4_scale = torch.randn(hidden_size, device=device, dtype=torch.float32)

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
        norm.nvfp4_scale = torch.randn(hidden_size, device=device, dtype=torch.float32)
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
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16
        eps = 1e-5

        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

        norm_cuda = (
            RMSNorm(
                hidden_size, eps=eps, is_nvfp4=True, norm_before_gate=False, group_size=group_size
            )
            .to(device)
            .to(dtype)
        )
        norm_cuda.nvfp4_scale = torch.ones(hidden_size, device=device, dtype=torch.float32)

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

        output_triton_fp4_packed, output_triton_sf = quantize_to_fp4_e2m1(
            output_triton.float(), group_size=group_size
        )

        num_groups = hidden_size // group_size
        cuda_sf_reshaped = output_cuda_fp4.scaling_factor.reshape(batch_size, num_groups)
        cuda_unpacked = unpack_and_scale_fp4(
            output_cuda_fp4.fp4_tensor, cuda_sf_reshaped, (batch_size, hidden_size), group_size
        )

        triton_sf_reshaped = output_triton_sf.reshape(batch_size, num_groups)
        triton_unpacked = unpack_and_scale_fp4(
            output_triton_fp4_packed, triton_sf_reshaped, (batch_size, hidden_size), group_size
        )

        torch.testing.assert_close(cuda_sf_reshaped, triton_sf_reshaped, rtol=0.05, atol=0.01)
        torch.testing.assert_close(cuda_unpacked, triton_unpacked, rtol=0.1, atol=0.1)

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
