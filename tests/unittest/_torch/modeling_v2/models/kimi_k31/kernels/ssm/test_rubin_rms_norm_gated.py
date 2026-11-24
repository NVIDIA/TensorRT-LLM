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
"""The family's gated RMSNorm launchers (``kernels/ssm/rubin_rms_norm_gated.py``) against a PyTorch
reference: plain, gated and delta-weight norms, the FP8 output, the token-major multi-row kernel and
its PDL launch, and the engine's fused NVFP4 kernel."""

import pytest
import torch
import triton
from utils.util import getSMVersion

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_rms_norm_gated as layernorm_gated
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm.math_utils import ceil_div, pad_up


def fused_gated_rmsnorm_quant_available():
    """Check if the fused_gated_rmsnorm_quant op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_gated_rmsnorm_quant")


skip_unless_nvfp4_kernel = pytest.mark.skipif(
    not fused_gated_rmsnorm_quant_available(),
    reason="needs the trtllm.fused_gated_rmsnorm_quant op",
)

pytestmark = pytest.mark.skipif(
    getSMVersion() != 107, reason="the family's Rubin kernels: needs an SM107 GPU"
)


def _rms(t, group_size, eps):
    """``t`` over the RMS of each ``group_size`` group of its last dimension (None: all of it)."""
    groups = t.reshape(*t.shape[:-1], -1, group_size or t.shape[-1])
    return (groups * torch.rsqrt(groups.pow(2).mean(-1, keepdim=True) + eps)).reshape(t.shape)


def reference_rmsnorm_gated(x, weight, z=None, eps=1e-5, norm_before_gate=True, group_size=None):
    """Gated RMSNorm in PyTorch: ``z``'s SiLU gates the normalized rows, or the rows before the
    norm with ``norm_before_gate=False``."""
    if z is None:
        return weight * _rms(x, group_size, eps)
    gate = z * torch.sigmoid(z)
    if norm_before_gate:
        return weight * _rms(x, group_size, eps) * gate
    return weight * _rms(x * gate, group_size, eps)


def _gated_rmsnorm(
    x,
    weight,
    z=None,
    *,
    eps,
    group_size=None,
    norm_before_gate=True,
    fp8_scale=None,
    weight_is_delta=False,
):
    """RMS-normalize the 2-D rows of ``x`` with the generic launcher (no bias)."""
    y, _, _ = layernorm_gated._layer_norm_fwd(
        x,
        weight,
        None,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
        fp8_scale=fp8_scale,
        weight_is_delta=weight_is_delta,
    )
    return y


class TestRMSNormBasic:
    def test_compare_with_reference(self):
        hidden_size = 2048
        batch_size = 4
        group_size = 1024
        device = "cuda"
        dtype = torch.float16
        eps = 1e-5

        torch.manual_seed(2)
        weight = torch.empty(hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

        output_triton = _gated_rmsnorm(
            x, weight, z, eps=eps, group_size=group_size, norm_before_gate=False
        )
        output_ref = reference_rmsnorm_gated(
            x, weight, z, eps=eps, norm_before_gate=False, group_size=group_size
        )

        torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("scale", [32.0, 0.13])
    def test_fp8_matches_separate_static_quantization(self, scale):
        hidden_size = 128
        batch_size = 32
        device = "cuda"
        dtype = torch.bfloat16

        torch.manual_seed(3)
        weight = torch.empty(hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(weight, mean=1.0, std=0.1)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        z = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        fp8_scale = torch.tensor(scale, device=device, dtype=torch.float32)

        bf16_output = _gated_rmsnorm(x, weight, z, eps=1e-6, norm_before_gate=True)
        expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(bf16_output, fp8_scale)
        actual = _gated_rmsnorm(x, weight, z, eps=1e-6, norm_before_gate=True, fp8_scale=fp8_scale)

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

        plain_output = _gated_rmsnorm(
            x, stored_weight, z, eps=1e-6, group_size=group_size, weight_is_delta=False
        )
        delta_output = _gated_rmsnorm(
            x, stored_weight, z, eps=1e-6, group_size=group_size, weight_is_delta=True
        )
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
    @pytest.mark.parametrize("num_tokens,heads,N", [(64, 8, 128), (33, 4, 512)])
    @pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
    def test_token_major_fp8_matches_separate_static_quantization(
        self, scale, num_tokens, heads, N, gate_activation
    ):
        """Both the multi-row (N<=256) and generic-fallback (N>256) paths of
        rms_norm_gated_token_major must quantize identically to norm-then-
        static-quantize."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_rms_norm_gated import (
            rms_norm_gated_token_major,
        )

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

    @pytest.mark.parametrize(
        "num_tokens,heads,N", [(8192, 96, 128), (1000, 8, 256), (301, 96, 128)]
    )
    @pytest.mark.parametrize("save_rstd", [False, True])
    def test_multirow_tile_loop_is_bitwise_invariant(
        self, num_tokens: int, heads: int, N: int, save_rstd: bool
    ) -> None:
        """Looping a program over several 4-row tiles (prefill-sized launches) must not change any
        output (or rstd) bit, including the masked tail tiles."""
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(13)
        num_rows = num_tokens * heads
        x = torch.randn(num_rows, N, device=device, dtype=dtype)
        weight = torch.randn(N, device=device, dtype=dtype) * 0.1 + 1.0
        wide = torch.randn(num_tokens, heads * N + 64, device=device, dtype=dtype)
        z = wide[:, 64:].view(num_tokens, heads, N)

        outputs = {}
        for loop in (1, 4, 3):
            output = torch.empty_like(x)
            rstd = torch.empty(num_rows, device=device, dtype=torch.float32) if save_rstd else None
            layernorm_gated._rms_norm_gated_fwd_multirow_kernel[
                (triton.cdiv(num_rows, layernorm_gated._MULTIROW_ROWS * loop),)
            ](
                x,
                output,
                weight,
                z,
                rstd,
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
                LAUNCH_WITH_PDL=False,
                LOOP=loop,
                num_warps=layernorm_gated._MULTIROW_NUM_WARPS,
            )
            outputs[loop] = (output, rstd)

        for loop in (4, 3):
            assert torch.equal(outputs[loop][0], outputs[1][0])
            if save_rstd:
                assert torch.equal(outputs[loop][1], outputs[1][1])
        # The launcher picks the looped grid for this many rows and matches the single-tile launch.
        got = layernorm_gated.rms_norm_gated_token_major(x, z, weight, 1e-6, None, "sigmoid")
        assert torch.equal(got, outputs[1][0])


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

        # Path 1: the engine's fused CUDA kernel (gated RMSNorm + FP4 quantization)
        fp4_out, sf_cuda = torch.ops.trtllm.fused_gated_rmsnorm_quant(
            x, z, weight, group_size, eps, sf_scale
        )
        fp4_cuda = fp4_out.view(torch.uint8)  # 8 FP4 values packed per int32

        # Path 2: Triton norm (full-precision) + separate fp4_quantize
        output_triton = _gated_rmsnorm(
            x, weight, z, eps=eps, group_size=group_size, norm_before_gate=False
        )

        # Quantize the Triton output with the same NVFP4 scheme
        fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
            output_triton.contiguous(),
            sf_scale,
            sf_vec_size,
            False,
            True,  # use_ue8m0=False, is_sf_swizzled_layout=True
        )

        # Compare FP4 packed values (byte-level)
        assert fp4_cuda.shape == fp4_separate.shape
        fp4_match_rate = (fp4_cuda == fp4_separate).float().mean().item()
        assert fp4_match_rate >= 0.99, f"FP4 packed values match rate {fp4_match_rate:.4f} < 0.99"

        # Compare scale factors: unswizzle with padded dimensions, then
        # slice the valid [batch_size, num_sf_cols] portion
        padded_rows = pad_up(batch_size, 128)
        num_sf_cols = ceil_div(hidden_size, sf_vec_size)
        padded_cols = pad_up(num_sf_cols, 4) * sf_vec_size

        cuda_sf_unswizzled = unswizzle_sf(sf_cuda, padded_rows, padded_cols, sf_vec_size)[
            :batch_size, :num_sf_cols
        ]
        triton_sf_unswizzled = unswizzle_sf(sf_separate, padded_rows, padded_cols, sf_vec_size)[
            :batch_size, :num_sf_cols
        ]

        sf_match_rate = (cuda_sf_unswizzled == triton_sf_unswizzled).float().mean().item()
        assert sf_match_rate >= 0.99, f"Scale factor match rate {sf_match_rate:.4f} < 0.99"
