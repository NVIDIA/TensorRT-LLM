# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/layernorm_gated.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
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

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from ...._utils import get_sm_version
from ...utils import Fp4QuantizedTensor


def fused_gated_rmsnorm_quant_shape_ok(hidden_size: int,
                                       group_size: int) -> bool:
    """True if ``torch.ops.trtllm.fused_gated_rmsnorm_quant`` supports this shape.

    Keep in sync with TORCH_CHECKs in cpp/tensorrt_llm/thop/fusedGatedRMSNormQuant.cpp.
    """
    if group_size <= 0 or hidden_size % group_size != 0:
        return False
    if group_size % 256 != 0:
        return False
    if not (256 <= group_size <= 8192):
        return False
    if hidden_size % 16 != 0:
        return False
    return True


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"OUTPUT_FP8": lambda args: args["FP8_SCALE"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    FP8_SCALE,  # static scale for an optional FP8 output
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        # Cast to int64 to avoid overflow: row * stride_z_row can exceed INT32_MAX
        # when Z is a non-contiguous slice (e.g., 131071 * 22656 = 2,969,544,576)
        Z += tl.cast(row, tl.int64) * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    if OUTPUT_FP8:
        # Match the existing two-kernel path: RMSNorm first stores to the
        # input dtype, then static quantization reloads and multiplies by the
        # rounded reciprocal of its input scale.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y *= tldevice.rcp_rn(tl.load(FP8_SCALE).to(tl.float32))
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


# Rows per program of the multi-row gated-RMSNorm kernel. At the GDN decode
# shape (thousands of 128-element rows) one row per CTA leaves the kernel
# launch-limited; 4 rows with 4 warps reproduces the single-row kernel's
# reduction order (bitwise-identical output) at ~2x the throughput.
_MULTIROW_ROWS = 4
_MULTIROW_NUM_WARPS = 4
_MULTIROW_MAX_N = 256


@triton.heuristics({"OUTPUT_FP8": lambda args: args["FP8_SCALE"] is not None})
@triton.jit
def _rms_norm_gated_fwd_multirow_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Z,  # pointer to the gate branch
    Rstd,  # pointer to the 1/std
    FP8_SCALE,  # static scale for an optional FP8 output
    stride_x_row,
    stride_y_row,
    stride_z_tok,
    M,  # number of rows in X
    eps,
    N: tl.constexpr,  # row length; power of two, whole row per program
    ROWS: tl.constexpr,  # rows per program
    HEADS_PER_TOK: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
):
    """rmsnorm(x) * silu(z), several short rows per program.

    Z is addressed token-major: row r reads z at
    (r // HEADS_PER_TOK) * stride_z_tok + (r % HEADS_PER_TOK) * N. With
    HEADS_PER_TOK == 1 and stride_z_tok == z's row stride this is a plain
    [M, N] z; with HEADS_PER_TOK == heads it reads a [num_tokens, heads, N]
    view whose (heads, N) block is contiguous per token, e.g. a column slice
    of a wider projection.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    cols = tl.arange(0, N)
    mask2d = row_mask[:, None]
    x_off = rows[:, None].to(tl.int64) * stride_x_row + cols[None, :]
    x = tl.load(X + x_off, mask=mask2d, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    w = tl.load(W + cols).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    tok = rows // HEADS_PER_TOK
    head = rows % HEADS_PER_TOK
    z_off = (tok[:, None].to(tl.int64) * stride_z_tok + head[:, None] * N +
             cols[None, :])
    z = tl.load(Z + z_off, mask=mask2d, other=0.0).to(tl.float32)
    y *= z * tl.sigmoid(z)
    if OUTPUT_FP8:
        # Match the existing two-kernel path: RMSNorm first stores to the
        # input dtype, then static quantization reloads and multiplies by the
        # rounded reciprocal of its input scale.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y *= tldevice.rcp_rn(tl.load(FP8_SCALE).to(tl.float32))
    y_off = rows[:, None].to(tl.int64) * stride_y_row + cols[None, :]
    tl.store(Y + y_off, y.to(Y.dtype.element_ty), mask=mask2d)


def _multirow_gated_rmsnorm_eligible(N, ngroups, bias, z, norm_before_gate,
                                     is_rms_norm):
    return (is_rms_norm and norm_before_gate and z is not None and bias is None
            and ngroups == 1 and N <= _MULTIROW_MAX_N and (N & (N - 1)) == 0)


def rms_norm_gated_token_major(x, z, weight, eps, out=None, fp8_scale=None):
    """rmsnorm(x) * silu(z) with z read in place from a 3D token-major view.

    x: [num_tokens * heads, N] with contiguous rows. z: [num_tokens, heads, N]
    whose (heads, N) block is contiguous per token and whose token stride is
    arbitrary (e.g. a column slice of a wider per-token projection). Falls
    back to the generic kernel on a packed copy of z when the shape is not
    eligible for the multi-row kernel.

    When fp8_scale (a scalar fp32 tensor holding the downstream static input
    scale) is given, the output is quantized to float8_e4m3fn in the same
    kernel.
    """
    M, N = x.shape
    num_tokens, heads, n_z = z.shape
    assert n_z == N and num_tokens * heads == M, (
        f"z shape {tuple(z.shape)} does not match x shape {tuple(x.shape)}")
    weight = weight.contiguous()
    eligible = (x.stride(-1) == 1 and z.stride(2) == 1 and z.stride(1) == N
                and N <= _MULTIROW_MAX_N and (N & (N - 1)) == 0)
    if not eligible:
        y, _, _ = _layer_norm_fwd(
            x,
            weight,
            None,
            eps,
            z=z.reshape(M, N),
            out=out,
            norm_before_gate=True,
            is_rms_norm=True,
            fp8_scale=fp8_scale,
        )
        return y
    if out is None:
        out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
        out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, _MULTIROW_ROWS), )
    with torch.cuda.device(x.device.index):
        _rms_norm_gated_fwd_multirow_kernel[grid](
            x,
            out,
            weight,
            z,
            rstd,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0),
            M,
            eps,
            N=N,
            ROWS=_MULTIROW_ROWS,
            HEADS_PER_TOK=heads,
            num_warps=_MULTIROW_NUM_WARPS,
        )
    return out


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    fp8_scale=None,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
        out = torch.empty_like(x, dtype=out_dtype)
    assert out.stride(-1) == 1
    if fp8_scale is not None:
        assert fp8_scale.dtype == torch.float32
        assert fp8_scale.numel() == 1
        assert out.dtype == torch.float8_e4m3fn
    mean = (torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    if _multirow_gated_rmsnorm_eligible(group_size, ngroups, bias, z,
                                        norm_before_gate, is_rms_norm):
        grid = (triton.cdiv(M, _MULTIROW_ROWS), )
        with torch.cuda.device(x.device.index):
            _rms_norm_gated_fwd_multirow_kernel[grid](
                x,
                out,
                weight,
                z,
                rstd,
                fp8_scale,
                x.stride(0),
                out.stride(0),
                z.stride(0),
                M,
                eps,
                N=group_size,
                ROWS=_MULTIROW_ROWS,
                HEADS_PER_TOK=1,
                num_warps=_MULTIROW_NUM_WARPS,
            )
        return out, mean, rstd
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
        is_nvfp4: bool = False,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size if group_size is not None else hidden_size
        self.norm_before_gate = norm_before_gate

        self.is_nvfp4 = is_nvfp4
        # nvfp4_scale will be set externally if is_nvfp4 is True
        self.nvfp4_scale: torch.Tensor | None = None
        # fp8_scale will be attached from the downstream static FP8 linear.
        self.fp8_scale: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor | Fp4QuantizedTensor:
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = self.weight.contiguous()
        bias = None
        if self.bias is not None:
            bias = self.bias.contiguous()

        fp8_scale = self.fp8_scale
        if fp8_scale is not None:
            if self.is_nvfp4:
                raise ValueError(
                    "FP8 and NVFP4 RMSNorm outputs are mutually exclusive")

        # NVFP4 quantized path - uses optimized fused CUDA kernel
        # Fuses: SiLU gating + Group RMSNorm + FP4 quantization
        if self.is_nvfp4 and z is not None and not self.norm_before_gate and \
            get_sm_version() >= 100 and \
           fused_gated_rmsnorm_quant_shape_ok(self.hidden_size, self.group_size):
            if self.nvfp4_scale is None:
                raise ValueError(
                    "RMSNormGated NVFP4 output requested but no `nvfp4_scale` is attached. "
                    "Please set module.nvfp4_scale = input_scale from the next linear layer."
                )

            sf_scale = self.nvfp4_scale.contiguous()
            fp4_out, sf_out = torch.ops.trtllm.fused_gated_rmsnorm_quant(
                x, z, weight, self.group_size, self.eps, sf_scale)

            # fp4_out is int32 with 8 FP4 values packed per int32
            fp4_u8 = fp4_out.view(torch.uint8)
            # Reshape to match expected output shape
            if len(x_shape_og) != 2:
                fp4_u8 = fp4_u8.reshape(*x_shape_og[:-1], x_shape_og[-1] // 2)

            return Fp4QuantizedTensor(fp4_u8, sf_out, is_sf_swizzled=True)

        # Original Triton kernel path
        y, _, _ = _layer_norm_fwd(
            x,
            weight,
            bias,
            self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
            fp8_scale=fp8_scale,
        )
        return y.reshape(x_shape_og)
