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
"""Unit tests for fused relu2 + NVFP4 quantization kernel."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.utils import gelu_tanh as _gelu_tanh_sentinel
from tensorrt_llm._torch.utils import relu2 as _relu2_sentinel
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tests.unittest.utils.util import getSMVersion


def fused_relu2_quantize_available():
    """Check if the fused_relu2_quantize op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_relu2_quantize")


def fused_gelu_tanh_quantize_available():
    """Check if the fused_gelu_tanh_quantize op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_gelu_tanh_quantize")


def fp4_quantize_available():
    """Check if the fp4_quantize op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")


skip_unless_fused_relu2_quantize = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_relu2_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm.fused_relu2_quantize op",
)

skip_unless_fused_relu2_and_fp4_quantize = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_relu2_quantize_available() or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_relu2_quantize + fp4_quantize ops",
)

skip_unless_fused_gelu_tanh_and_fp4_quantize = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_gelu_tanh_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_gelu_tanh_quantize + fp4_quantize ops",
)

# MLP-level heuristic gate tests don't run the kernel; they only check the
# boolean fast-path detection in MLP.create_weights. Skip on pre-Blackwell
# only because Linear.__init__ assumes a CUDA device, and skip if the gelu
# fused op isn't compiled into this build (the gate keys off hasattr).
skip_unless_fused_gelu_tanh_op = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_gelu_tanh_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm.fused_gelu_tanh_quantize op",
)


# FP4 E2M1 lookup table for reference implementation
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Reference relu2 activation: square(relu(x))."""
    return torch.square(F.relu(x))


def cast_to_fp4(weight: torch.Tensor) -> torch.Tensor:
    """Cast tensor values to FP4 E2M1 format (as uint8)."""
    device = weight.device

    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
    mask_shape = list(weight.shape)
    mask = mask.expand([*mask_shape, 7])

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()

    ord_val = torch.searchsorted(E2M1_BOUNDS.to(device), weight_abs, out_int32=True).to(torch.uint8)
    round_val = torch.any((weight_abs.unsqueeze(-1) == E2M1_BOUNDS.to(device)) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_val + round_val).to(torch.uint8)
    return fp4_val


def quantize_nvfp4_ref(
    input: torch.Tensor, sf_scale: torch.Tensor, sf_vec_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference NVFP4 quantization implementation.

    Args:
        input: Input tensor [M, N], already activated (e.g., after relu2)
        sf_scale: Per-tensor scaling factor (sf_scale = amax / (6 * 448))
        sf_vec_size: Block size for per-block scaling (default 16)

    Returns:
        Tuple of (fp4_packed, scale_factors)
    """
    m, n = input.shape
    assert n % sf_vec_size == 0, f"N ({n}) must be divisible by sf_vec_size ({sf_vec_size})"

    # Reshape for block-wise quantization
    input_blocked = input.view(m, n // sf_vec_size, sf_vec_size)

    # Compute per-block amax
    per_block_amax = input_blocked.abs().amax(dim=-1).float()

    # Compute per-block scale: amax / 6.0
    per_block_scale = per_block_amax / 6.0

    # Quantize per-block scale to FP8
    q_per_block_scale = per_block_scale / sf_scale
    q_per_block_scale[per_block_scale == 0] = 1.0
    q_per_block_scale_fp8 = q_per_block_scale.to(torch.float8_e4m3fn)

    # Dequantize scale for actual quantization
    scale_dequant = q_per_block_scale_fp8.float() * sf_scale

    # Scale the input
    scale_expanded = scale_dequant.unsqueeze(-1).expand_as(input_blocked)
    scaled_input = input_blocked / (scale_expanded + 1e-12)
    scaled_input = scaled_input.view(m, n)

    # Cast to FP4
    fp4_vals = cast_to_fp4(scaled_input)

    # Pack two FP4 values into one uint8
    packed = (fp4_vals[..., 1::2] << 4) | fp4_vals[..., 0::2]

    return packed, q_per_block_scale_fp8


def fused_relu2_quantize_ref(
    input: torch.Tensor, sf_scale: torch.Tensor, sf_vec_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for fused relu2 + NVFP4 quantization.

    Args:
        input: Input tensor [M, N]
        sf_scale: Per-tensor scaling factor
        sf_vec_size: Block size for per-block scaling (default 16)

    Returns:
        Tuple of (fp4_packed, scale_factors)
    """
    # Apply relu2 activation
    activated = relu2(input)
    # Quantize to NVFP4
    return quantize_nvfp4_ref(activated, sf_scale, sf_vec_size)


@skip_unless_fused_relu2_quantize
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_relu2_quantize_zeros(dtype):
    """Test fused_relu2_quantize with inputs that produce zeros after relu2."""
    device = torch.device("cuda")

    # All negative inputs -> relu2 produces all zeros
    m, n = 32, 64
    input_tensor = -torch.abs(torch.randn(m, n, dtype=dtype, device=device))
    sf_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(input_tensor, sf_scale, 16)

    assert fp4_fused.shape == (m, n // 2)
    assert (fp4_fused == 0).all(), "All negative inputs should produce zero output"


@skip_unless_fused_relu2_and_fp4_quantize
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_relu2_quantize_vs_separate_ops(m, n, dtype):
    """
    Compare fused_relu2_quantize kernel output against separate relu2 + fp4_quantize.

    This test verifies that the fused CUDA kernel produces FP4 packed values that
    closely match running relu2 activation followed by fp4_quantize separately.

    Note: Due to floating point precision differences in intermediate calculations
    (e.g., FMA vs separate mul+add), a small percentage of values at quantization
    boundaries may differ. We require >= 99% match rate.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    input_tensor = torch.randn(m, n, dtype=dtype, device=device)
    activated = relu2(input_tensor)
    sf_scale = (activated.abs().amax().float() / (6.0 * 448.0)).to(device)
    sf_scale = sf_scale.view(1)

    fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
        activated,
        sf_scale,
        16,
        False,
        True,  # use_ue8m0=False, is_sf_swizzled_layout=True
    )
    fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(
        input_tensor.contiguous(), sf_scale, 16
    )

    match_rate = (fp4_fused == fp4_separate).float().mean().item()
    assert match_rate >= 0.99, (
        f"FP4 values match rate {match_rate:.4f} < 0.99 for shape ({m}, {n}), dtype {dtype}"
    )


@skip_unless_fused_relu2_and_fp4_quantize
def test_fused_relu2_quantize_vs_separate_ops_various_sf_scales():
    """
    Test with various sf_scale values to ensure consistent behavior.
    """
    device = torch.device("cuda")
    m, n = 64, 128
    dtype = torch.bfloat16

    torch.manual_seed(123)
    input_tensor = torch.randn(m, n, dtype=dtype, device=device)
    activated = relu2(input_tensor)

    # Test with different sf_scale values
    for scale_multiplier in [0.1, 1.0, 10.0]:
        sf_scale = (
            (activated.abs().amax().float() / (6.0 * 448.0) * scale_multiplier).to(device).view(1)
        )
        fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
            activated, sf_scale, 16, False, True
        )
        fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(
            input_tensor.contiguous(), sf_scale, 16
        )

        match_rate = (fp4_fused == fp4_separate).float().mean().item()
        assert match_rate >= 0.99, (
            f"FP4 values match rate {match_rate:.4f} < 0.99 with scale_multiplier={scale_multiplier}"
        )


@skip_unless_fused_gelu_tanh_and_fp4_quantize
@pytest.mark.parametrize(
    "m,n",
    [
        # Wan 14B FFN intermediate
        (128, 13824),
        # Wan 1.3B FFN intermediate
        (4096, 8960),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gelu_tanh_quantize(m, n, dtype):
    """
    Compare fused_gelu_tanh_quantize kernel against separate F.gelu(approximate='tanh')
    + fp4_quantize for Wan 2.2 MLP intermediate shapes.

    The CUDA kernel computes gelu_tanh in fp32 then rounds back to native precision
    before the byte-for-byte identical NVFP4 quant epilogue used by fused_relu2_quantize.
    A small fraction of values that land exactly on a quantization boundary can differ
    between the fused and separate paths due to FMA vs. mul+add ordering in the
    activation/scaling pipeline; mirror the >= 99% match tolerance the relu2 tests use.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    input_tensor = torch.randn(m, n, dtype=dtype, device=device)
    activated = F.gelu(input_tensor, approximate="tanh")
    sf_scale = (activated.abs().amax().float() / (6.0 * 448.0)).to(device).view(1)

    fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
        activated,
        sf_scale,
        16,
        False,
        True,  # use_ue8m0=False, is_sf_swizzled_layout=True
    )
    fp4_fused, sf_fused = torch.ops.trtllm.fused_gelu_tanh_quantize(
        input_tensor.contiguous(), sf_scale, 16
    )

    assert fp4_fused.shape == (m, n // 2)
    assert sf_fused.shape == sf_separate.shape
    # The swizzled scale-factor tensor is consumed by the downstream NVFP4
    # GEMM. A buggy SF write would still let the FP4-byte match-rate pass
    # but corrupt dequantization in the GEMM epilogue; compare it explicitly.
    assert torch.equal(sf_fused, sf_separate), (
        "Swizzled scale-factor tensors diverge between fused_gelu_tanh_quantize "
        "and the activation-then-fp4_quantize reference path."
    )

    match_rate = (fp4_fused == fp4_separate).float().mean().item()
    assert match_rate >= 0.99, (
        f"FP4 values match rate {match_rate:.4f} < 0.99 for shape ({m}, {n}), dtype {dtype}"
    )


def _build_nvfp4_mlp(activation, force_dynamic_quantization: bool) -> MLP:
    """Construct a small MLP wired up with an NVFP4 quant_config.

    Mirrors how `WanBlock.__init__` builds its FFN: ROW-parallel down_proj
    backed by NVFP4. Returns the MLP after Linear's own create_weights() has
    populated parameters in __init__; the caller can then simulate a
    calibrated checkpoint via setattr before invoking mlp.create_weights() to
    evaluate the fast-path gate.
    """
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    config = ModelConfig(
        quant_config=quant_config,
        force_dynamic_quantization=force_dynamic_quantization,
    )
    return MLP(
        hidden_size=128,
        intermediate_size=256,
        bias=False,
        activation=activation,
        dtype=torch.bfloat16,
        config=config,
    )


@skip_unless_fused_gelu_tanh_op
def test_mlp_uses_fused_gelu_tanh_quant_on_static_nvfp4():
    """Heuristic gate test: MLP enables the fused gelu_tanh + NVFP4 fast path
    only when the activation is the gelu_tanh sentinel AND the down_proj is
    NVFP4 with a (non-None) input_scale AND force_dynamic_quantization is
    False.

    The dynamic-quant assertion is the critical regression guard for chunk 2:
    NVFP4LinearMethod._input_prepare returns a stale `module.alpha` when fed
    an Fp4QuantizedTensor under dynamic quantization, so the fused path must
    stay off until a dynamic-aware fusion is added.
    """
    # Static NVFP4 + gelu_tanh -> fused path ON.
    # NVFP4LinearMethod.create_weights (linear.py:1295) allocates a non-None
    # input_scale Parameter for any static NVFP4 layer, which the gate reads
    # via `getattr(..., 'input_scale', None) is not None`. has_static_input_scale
    # is FP8-only and not relevant to the NVFP4 gate, so we do not set it.
    mlp = _build_nvfp4_mlp(activation=_gelu_tanh_sentinel, force_dynamic_quantization=False)
    mlp.create_weights()
    assert mlp._use_fused_gelu_tanh_quant is True
    assert mlp._use_fused_relu2_quant is False

    # Dynamic NVFP4 + gelu_tanh -> fused path OFF (regression guard).
    mlp_dyn = _build_nvfp4_mlp(activation=_gelu_tanh_sentinel, force_dynamic_quantization=True)
    mlp_dyn.create_weights()
    assert mlp_dyn._use_fused_gelu_tanh_quant is False, (
        "fused gelu_tanh+NVFP4 path must stay off under dynamic quantization "
        "to avoid using a stale module.alpha (linear.py:1263-1270)"
    )

    # Note: there is no module-level "uncalibrated NVFP4" state to model in
    # this unit test. NVFP4LinearMethod.create_weights always allocates a
    # placeholder input_scale Parameter (linear.py:1295), and the only way to
    # reach input_scale=None is via a runtime mutation after create_weights
    # (e.g., linear.py:821 when a layer opts into dynamic quant at load time).
    # The gate guards against that via `getattr(down_proj, 'input_scale', None)
    # is not None` in mlp.py, which falls back to the unfused path. The
    # relu2-activation case below already verifies that the gelu_tanh path
    # stays OFF when its specific conditions are not met.

    # Static NVFP4 + relu2 -> gelu path stays OFF, relu2 path turns ON.
    # Confirms the relu2 fast path is not regressed (Nemotron-H).
    mlp_relu2 = _build_nvfp4_mlp(activation=_relu2_sentinel, force_dynamic_quantization=False)
    mlp_relu2.create_weights()
    assert mlp_relu2._use_fused_gelu_tanh_quant is False
    assert mlp_relu2._use_fused_relu2_quant is True


@pytest.mark.parametrize(
    "activation_sentinel, helper_name",
    [
        (_gelu_tanh_sentinel, "_fused_gelu_tanh_quant"),
        (_relu2_sentinel, "_fused_relu2_quant"),
    ],
    ids=["gelu_tanh", "relu2"],
)
@skip_unless_fused_gelu_tanh_and_fp4_quantize
def test_fused_helpers_preserve_3d_prefix_dims(activation_sentinel, helper_name):
    """Regression test for the B>1 shape-preservation contract that the
    PR #14773 review (CodeRabbit comment) flagged.

    The fused activation+quant helpers in `MLP` flatten ``x`` to 2D before the
    kernel call. Without re-inflating the packed FP4 output back to the
    input rank, ``NVFP4LinearMethod.apply`` would only restore the original
    shape on a torch.Tensor input (3D plain tensor), not on the
    ``Fp4QuantizedTensor`` it actually receives here -- because the unflatten
    branch in ``linear.py`` keys off ``fp4_tensor.dim() > 2``. With a 2D
    return, the downstream ``down_proj`` would silently flatten ``[B, S, H]``
    to ``[B*S, H]``; broadcasting in the residual add then only happens to
    work when B==1 (the prior integration-test regime).

    Verifies the fix preserves the prefix dims for B>1.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, seq_len = 2, 4

    mlp = _build_nvfp4_mlp(activation=activation_sentinel, force_dynamic_quantization=False)
    mlp.create_weights()
    mlp = mlp.to(device=device)

    # _build_nvfp4_mlp leaves down_proj.input_scale as the uninitialized
    # placeholder Parameter from NVFP4LinearMethod.create_weights. Populate
    # it with a real scalar so the kernel call doesn't read garbage; the
    # actual value is irrelevant for a shape-only assertion.
    with torch.no_grad():
        mlp.down_proj.input_scale.copy_(torch.tensor([1.0 / 6.0], device=device))

    helper = getattr(mlp, helper_name)
    # Mirror the path the MLP forward takes: helper consumes the `up_proj`
    # output shape, which is [B, S, intermediate_size]. The intermediate
    # size is 256 in _build_nvfp4_mlp, but it doesn't matter for the shape
    # assertion; the helper writes back to hidden_size via down_proj.
    x_3d = torch.randn(
        batch_size, seq_len, mlp.intermediate_size, dtype=torch.bfloat16, device=device
    )
    quant_out = helper(x_3d)

    assert quant_out.fp4_tensor.dim() == 3, (
        f"{helper_name} flattened a 3D input to {quant_out.fp4_tensor.shape}; "
        "downstream NVFP4LinearMethod.apply will silently collapse batch dims."
    )
    assert quant_out.fp4_tensor.shape == (
        batch_size,
        seq_len,
        mlp.intermediate_size // 2,
    ), f"unexpected packed-FP4 shape {tuple(quant_out.fp4_tensor.shape)}"
