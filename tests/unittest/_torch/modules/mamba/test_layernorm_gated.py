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

import sys

import pytest
import torch
import triton
from utils.util import getSMVersion

import tensorrt_llm._torch.modules.mamba.layernorm_gated as layernorm_gated
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, unswizzle_sf
from tensorrt_llm.math_utils import ceil_div, pad_up


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
    gate_activation: str = "silu",
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
        z_activated = silu(z) if gate_activation == "silu" else torch.sigmoid(z)
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


def max_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance between two same-dtype float tensors, in units in the last place."""
    int_dtype = {2: torch.int16, 4: torch.int32}[a.element_size()]
    sign_bit = 1 << (8 * a.element_size() - 1)

    def ordered(t: torch.Tensor) -> torch.Tensor:
        bits = t.view(int_dtype).to(torch.int64)
        return torch.where(bits < 0, -(bits + sign_bit), bits)

    return (ordered(a) - ordered(b)).abs().max().item()


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

    @pytest.mark.parametrize("batch_size", [32, layernorm_gated._MULTIROW_LARGE_TILE_MIN_ROWS + 32])
    @pytest.mark.parametrize("scale", [32.0, 0.13])
    def test_fp8_matches_separate_static_quantization(self, scale, batch_size):
        hidden_size = 128
        device = "cuda"
        dtype = torch.bfloat16

        torch.manual_seed(3)
        norm = RMSNorm(hidden_size, eps=1e-6, norm_before_gate=True).to(device).to(dtype)
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        fp8_scale = torch.tensor(scale, device=device, dtype=torch.float32)

        bf16_output = norm(x, z=z)
        expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(bf16_output, fp8_scale)
        norm.fp8_scale = fp8_scale
        actual = norm(x, z=z)

        assert actual.dtype == torch.float8_e4m3fn
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    @pytest.mark.parametrize("group_size", [None, 64])
    @pytest.mark.parametrize("with_gate", [False, True])
    def test_delta_weight_matches_explicit_offset(self, group_size, with_gate):
        hidden_size = 128
        device = "cuda"
        dtype = torch.bfloat16

        torch.manual_seed(4)
        stored_weight = torch.randn(hidden_size, device=device, dtype=dtype) * 0.1
        x = torch.randn(16, hidden_size, device=device, dtype=dtype)
        z = torch.randn_like(x) if with_gate else None
        plain = RMSNorm(
            hidden_size,
            eps=1e-6,
            group_size=group_size,
            weight_is_delta=False,
        ).to(device=device, dtype=dtype)
        delta = RMSNorm(
            hidden_size,
            eps=1e-6,
            group_size=group_size,
            weight_is_delta=True,
        ).to(device=device, dtype=dtype)
        plain.weight.data.copy_(stored_weight)
        delta.weight.data.copy_(stored_weight)

        plain_output = plain(x, z=z)
        delta_output = delta(x, z=z)
        plain_reference = reference_rmsnorm_gated(
            x,
            stored_weight,
            z,
            eps=1e-6,
            group_size=group_size,
        )
        delta_reference = reference_rmsnorm_gated(
            x,
            stored_weight + 1,
            z,
            eps=1e-6,
            group_size=group_size,
        )

        torch.testing.assert_close(plain_output, plain_reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(delta_output, delta_reference, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(plain_output, delta_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("scale", [32.0, 0.13])
    @pytest.mark.parametrize(
        "num_tokens,heads,N", [(64, 8, 128), (33, 4, 512), (1025, 32, 128), (1025, 32, 64)]
    )
    @pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
    def test_token_major_fp8_matches_separate_static_quantization(
        self, scale, num_tokens, heads, N, gate_activation
    ):
        """Both the multi-row (N<=256, 4-row and large-tile launches) and
        generic-fallback (N>256) paths of rms_norm_gated_token_major must
        quantize identically to norm-then-static-quantize."""
        from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

        device = "cuda"
        dtype = torch.bfloat16

        torch.manual_seed(3)
        M = num_tokens * heads
        x = torch.randn(M, N, device=device, dtype=dtype)
        weight = torch.randn(N, device=device, dtype=dtype) * 0.1 + 1.0
        wide = torch.randn(num_tokens, heads * N + 256, device=device, dtype=dtype)
        z = wide[:, 256:].view(num_tokens, heads, N)
        fp8_scale = torch.tensor(scale, device=device, dtype=torch.float32)

        bf16_output = rms_norm_gated_token_major(
            x, z, weight, 1e-6, gate_activation=gate_activation
        )
        expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(bf16_output, fp8_scale)
        actual = rms_norm_gated_token_major(
            x,
            z,
            weight,
            1e-6,
            fp8_scale=fp8_scale,
            gate_activation=gate_activation,
        )

        assert actual.dtype == torch.float8_e4m3fn
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    @pytest.mark.parametrize(
        "num_rows,N,dtype,expected_rows",
        [
            (1, 128, torch.bfloat16, 4),
            (32767, 128, torch.bfloat16, 4),
            (32768, 128, torch.bfloat16, 8),
            (32768, 128, torch.float16, 8),
            (32768, 64, torch.bfloat16, 16),
            (32768, 256, torch.bfloat16, 4),
            (32768, 128, torch.float32, 4),
            (32768, 64, torch.float32, 8),
        ],
    )
    def test_multirow_rows_per_program(self, num_rows, N, dtype, expected_rows):
        """Large launches take enough rows per program for whole 16-byte loads."""
        assert layernorm_gated._multirow_rows(num_rows, N, dtype.itemsize) == expected_rows

    @pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("N", [64, 128, 256])
    @pytest.mark.parametrize("num_tokens", [1023, 1025])
    def test_multirow_launch_shapes_match_reference(
        self, num_tokens, N, dtype, gate_activation, monkeypatch
    ):
        """Launches just below and above the large-tile threshold (32 heads:
        32736 and 32800 rows) match the reference, agree bitwise between the
        token-major and dense entry points, and differ from the 4-row launch
        only by rounding."""
        device = "cuda"
        heads = 32
        num_rows = num_tokens * heads

        torch.manual_seed(5)
        x = torch.randn(num_rows, N, device=device, dtype=dtype)
        weight = torch.rand(N, device=device, dtype=dtype) + 0.5
        wide = torch.randn(num_tokens, heads * N + 64, device=device, dtype=dtype)
        z = wide[:, 64:].view(num_tokens, heads, N)
        z_rows = z.reshape(num_rows, N)

        output = layernorm_gated.rms_norm_gated_token_major(
            x, z, weight, 1e-6, gate_activation=gate_activation
        )
        dense_output, _, _ = layernorm_gated._layer_norm_fwd(
            x,
            weight,
            None,
            1e-6,
            z=z_rows.contiguous(),
            norm_before_gate=True,
            is_rms_norm=True,
            gate_activation=gate_activation,
        )
        assert torch.equal(output, dense_output)

        expected = reference_rmsnorm_gated(
            x.float(), weight.float(), z_rows.float(), eps=1e-6, gate_activation=gate_activation
        )
        torch.testing.assert_close(output, expected.to(dtype), rtol=1e-2, atol=1e-2)

        rows = layernorm_gated._multirow_rows(num_rows, N, x.element_size())
        monkeypatch.setattr(layernorm_gated, "_MULTIROW_LARGE_TILE_MIN_ROWS", sys.maxsize)
        four_row_output = layernorm_gated.rms_norm_gated_token_major(
            x, z, weight, 1e-6, gate_activation=gate_activation
        )
        if rows == layernorm_gated._MULTIROW_ROWS:
            assert torch.equal(output, four_row_output)
        elif dtype == torch.float32:
            torch.testing.assert_close(output, four_row_output, rtol=1e-5, atol=0)
        else:
            assert max_ulp_distance(output, four_row_output) <= 1

    @pytest.mark.skipif(getSMVersion() < 90, reason="PDL requires SM90+")
    @pytest.mark.parametrize("num_tokens,heads,N", [(1, 47, 128), (7, 8, 256), (2048, 48, 128)])
    def test_token_major_pdl_launch_is_bitwise_invariant(
        self, num_tokens: int, heads: int, N: int
    ) -> None:
        """Enabling PDL must not change any output bit."""
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(11)
        num_rows = num_tokens * heads
        x = torch.randn(num_rows, N, device=device, dtype=dtype)
        weight = torch.randn(N, device=device, dtype=dtype) * 0.1 + 1.0
        wide = torch.randn(num_tokens, heads * N + 128, device=device, dtype=dtype)
        z = wide[:, 128:].view(num_tokens, heads, N)

        outputs = {}
        for launch_with_pdl in (False, True):
            output = torch.empty_like(x)
            layernorm_gated._rms_norm_gated_fwd_multirow_kernel[
                (triton.cdiv(num_rows, layernorm_gated._MULTIROW_ROWS),)
            ](
                x,
                output,
                weight,
                z,
                None,
                None,
                x.stride(0),
                output.stride(0),
                z.stride(0),
                num_rows,
                1e-6,
                N=N,
                ROWS=layernorm_gated._MULTIROW_ROWS,
                HEADS_PER_TOK=heads,
                GATE_SIGMOID=True,
                WEIGHT_IS_DELTA=False,
                LAUNCH_WITH_PDL=launch_with_pdl,
                num_warps=layernorm_gated._MULTIROW_NUM_WARPS,
                launch_pdl=launch_with_pdl,
            )
            outputs[launch_with_pdl] = output

        assert torch.equal(outputs[True], outputs[False])


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
