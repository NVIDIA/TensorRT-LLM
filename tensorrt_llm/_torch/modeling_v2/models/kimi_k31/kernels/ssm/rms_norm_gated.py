# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated RMSNorm (Triton) of the KDA output norm, token-major.

``rms_norm_gated_token_major`` normalizes each row group and multiplies by the gate; the family op
``ops/ssm/rms_norm_gated_token_major`` calls it directly.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl


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
    GATE_SIGMOID: tl.constexpr,
    WEIGHT_IS_DELTA: tl.constexpr,
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
        gate = tl.sigmoid(z)
        if not GATE_SIGMOID:
            gate *= z
        x *= gate
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
    if WEIGHT_IS_DELTA:
        w += 1.0
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        gate = tl.sigmoid(z)
        if not GATE_SIGMOID:
            gate *= z
        y *= gate
    if OUTPUT_FP8:
        # Match the existing two-kernel path: RMSNorm first stores to the
        # input dtype, then static quantization reloads and multiplies by the
        # rounded reciprocal of its input scale.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y *= tldevice.rcp_rn(tl.load(FP8_SCALE).to(tl.float32))
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


_MULTIROW_ROWS = 4


_MULTIROW_NUM_WARPS = 4


_MULTIROW_MAX_N = 256


@functools.lru_cache(maxsize=None)
def _pdl_device_policy(device_index: int) -> tuple[bool, int]:
    """Whether PDL is allowed on this device, and its SM count."""
    props = torch.cuda.get_device_properties(device_index)
    sm_version = props.major * 10 + props.minor
    allowed = sm_version >= 90 and get_env_enable_pdl()
    return allowed, props.multi_processor_count


def _multirow_pdl(grid_size: int, device_index: int) -> bool:
    """Use PDL only when the launch grid cannot cover all SMs by itself."""
    allowed, num_sms = _pdl_device_policy(device_index)
    return allowed and grid_size < num_sms


@triton.heuristics({"OUTPUT_FP8": lambda args: args["FP8_SCALE"] is not None})
@triton.heuristics({"SAVE_RSTD": lambda args: args["Rstd"] is not None})
@triton.jit
def _rms_norm_gated_fwd_multirow_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Z,  # pointer to the gate branch
    Rstd,  # optional pointer to the 1/std
    FP8_SCALE,  # static scale for an optional FP8 output
    stride_x_row,
    stride_y_row,
    stride_z_tok,
    M,  # number of rows in X
    eps,
    N: tl.constexpr,  # row length; power of two, whole row per program
    ROWS: tl.constexpr,  # rows per program
    HEADS_PER_TOK: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
    WEIGHT_IS_DELTA: tl.constexpr,
    SAVE_RSTD: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    """Gated rmsnorm(x), several short rows per program.

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
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
    x_off = rows[:, None].to(tl.int64) * stride_x_row + cols[None, :]
    x = tl.load(X + x_off, mask=mask2d, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    if SAVE_RSTD:
        tl.store(Rstd + rows, rstd, mask=row_mask)
    w = tl.load(W + cols).to(tl.float32)
    if WEIGHT_IS_DELTA:
        w += 1.0
    y = x * rstd[:, None] * w[None, :]
    tok = rows // HEADS_PER_TOK
    head = rows % HEADS_PER_TOK
    z_off = tok[:, None].to(tl.int64) * stride_z_tok + head[:, None] * N + cols[None, :]
    z = tl.load(Z + z_off, mask=mask2d, other=0.0).to(tl.float32)
    gate = tl.sigmoid(z)
    if not GATE_SIGMOID:
        gate *= z
    y *= gate
    if OUTPUT_FP8:
        # Match the existing two-kernel path: RMSNorm first stores to the
        # input dtype, then static quantization reloads and multiplies by the
        # rounded reciprocal of its input scale.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y *= tldevice.rcp_rn(tl.load(FP8_SCALE).to(tl.float32))
    y_off = rows[:, None].to(tl.int64) * stride_y_row + cols[None, :]
    tl.store(Y + y_off, y.to(Y.dtype.element_ty), mask=mask2d)
    if LAUNCH_WITH_PDL:
        # Release only after both Rstd (when present) and Y have been stored.
        tl.extra.cuda.gdc_launch_dependents()


def _multirow_gated_rmsnorm_eligible(N, ngroups, bias, z, norm_before_gate, is_rms_norm):
    return (
        is_rms_norm
        and norm_before_gate
        and z is not None
        and bias is None
        and ngroups == 1
        and N <= _MULTIROW_MAX_N
        and (N & (N - 1)) == 0
    )


def rms_norm_gated_token_major(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: torch.Tensor | None = None,
    gate_activation: str = "silu",
) -> torch.Tensor:
    """Gated rmsnorm(x) with z read from a 3D token-major view.

    x: [num_tokens * heads, N] with contiguous rows. z: [num_tokens, heads, N]
    whose (heads, N) block is contiguous per token and whose token stride is
    arbitrary (e.g. a column slice of a wider per-token projection). Falls
    back to the generic kernel on a packed copy of z when the shape is not
    eligible for the multi-row kernel.

    When fp8_scale (a scalar fp32 tensor holding the downstream static input
    scale) is given, the output is quantized to float8_e4m3fn in the same
    kernel. ``gate_activation`` is ``"silu"`` for GDN's ``z * sigmoid(z)``
    gate or ``"sigmoid"`` for KDA's ``sigmoid(z)`` gate.
    """
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")
    gate_sigmoid = gate_activation == "sigmoid"
    M, N = x.shape
    num_tokens, heads, n_z = z.shape
    assert n_z == N and num_tokens * heads == M, (
        f"z shape {tuple(z.shape)} does not match x shape {tuple(x.shape)}"
    )
    weight = weight.contiguous()
    eligible = (
        x.stride(-1) == 1
        and z.stride(2) == 1
        and z.stride(1) == N
        and N <= _MULTIROW_MAX_N
        and (N & (N - 1)) == 0
    )
    if not eligible:
        y, _, _ = _layer_norm_fwd(
            x,
            weight,
            None,
            eps,
            z=z.reshape(M, N),
            norm_before_gate=True,
            is_rms_norm=True,
            fp8_scale=fp8_scale,
            gate_activation=gate_activation,
        )
        return y
    out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
    out = torch.empty_like(x, dtype=out_dtype)
    grid_size = triton.cdiv(M, _MULTIROW_ROWS)
    device_index = x.device.index
    launch_with_pdl = _multirow_pdl(grid_size, device_index)
    with torch.cuda.device(device_index):
        _rms_norm_gated_fwd_multirow_kernel[(grid_size,)](
            x,
            out,
            weight,
            z,
            None,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0),
            M,
            eps,
            N=N,
            ROWS=_MULTIROW_ROWS,
            HEADS_PER_TOK=heads,
            GATE_SIGMOID=gate_sigmoid,
            WEIGHT_IS_DELTA=False,
            LAUNCH_WITH_PDL=launch_with_pdl,
            num_warps=_MULTIROW_NUM_WARPS,
            launch_pdl=launch_with_pdl,
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
    gate_activation="silu",
    weight_is_delta=False,
):
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")
    gate_sigmoid = gate_activation == "sigmoid"
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
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
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    if _multirow_gated_rmsnorm_eligible(
        group_size, ngroups, bias, z, norm_before_gate, is_rms_norm
    ):
        grid_size = triton.cdiv(M, _MULTIROW_ROWS)
        device_index = x.device.index
        launch_with_pdl = _multirow_pdl(grid_size, device_index)
        with torch.cuda.device(device_index):
            _rms_norm_gated_fwd_multirow_kernel[(grid_size,)](
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
                GATE_SIGMOID=gate_sigmoid,
                WEIGHT_IS_DELTA=weight_is_delta,
                LAUNCH_WITH_PDL=launch_with_pdl,
                num_warps=_MULTIROW_NUM_WARPS,
                launch_pdl=launch_with_pdl,
            )
        return out, mean, rstd
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
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
            GATE_SIGMOID=gate_sigmoid,
            WEIGHT_IS_DELTA=weight_is_delta,
            num_warps=num_warps,
        )
    return out, mean, rstd
