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

"""Triton kernels of ``trtllm::k3_rms_norm_gated_token_major``.

The same kernels as ``tensorrt_llm/_torch/modules/mamba/layernorm_gated.py``, which the built-in models keep using;
the two give the same bits.
"""

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


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
    EARLY_TRIGGER: tl.constexpr = False,
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
    if EARLY_TRIGGER:
        # Dependents read Y only after their own griddepcontrol.wait.
        tl.extra.cuda.gdc_launch_dependents()
    # The weight is a parameter (not produced in-step), so it is fetched before
    # the PDL wait; X (the producer's output) and Z (an earlier in-step kernel's
    # output) are read only after it.
    w = tl.load(W + cols).to(tl.float32)
    if WEIGHT_IS_DELTA:
        w += 1.0
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
    x_off = rows[:, None].to(tl.int64) * stride_x_row + cols[None, :]
    x = tl.load(X + x_off, mask=mask2d, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    if SAVE_RSTD:
        tl.store(Rstd + rows, rstd, mask=row_mask)
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
    if LAUNCH_WITH_PDL and not EARLY_TRIGGER:
        # Release only after both Rstd (when present) and Y have been stored.
        tl.extra.cuda.gdc_launch_dependents()
