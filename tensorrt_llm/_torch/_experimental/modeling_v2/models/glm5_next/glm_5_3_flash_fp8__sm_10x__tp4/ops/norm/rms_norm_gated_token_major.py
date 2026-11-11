# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/layernorm_gated.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated RMS norm of per-head rows, with the gate read from a token-major ``[tokens, heads, N]`` view.

The kernel, its launch configuration and its PDL policy are those Modeling V1's
``trtllm::rms_norm_gated_token_major`` (``tensorrt_llm/_torch/modules/mamba/layernorm_gated.py``) uses for
these layouts; this entry does not import it. The V1 op's generic one-row-per-program kernel for other
layouts is not part of this entry, so it takes the multi-row kernel's layouts only.

The entry's contract and reference are copied from kimi_k3_vl's
(``models/kimi_k3_vl/ops/norm/rms_norm_gated_token_major.py``), which calls the V1 op; the reference is
the same computation. qwen4_exp's entry (``models/qwen4_exp/ops/norm/rms_norm_gated_token_major.py``)
also calls the V1 op, launching the multi-row kernel itself for its prefill shapes.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Rows per program of the multi-row kernel. At the decode shape (thousands of 128-element rows) one row
# per CTA leaves the kernel launch-limited; 4 rows with 4 warps reproduces the single-row kernel's
# reduction order at about twice the throughput.
_MULTIROW_ROWS = 4
_MULTIROW_NUM_WARPS = 4
_MULTIROW_MAX_N = 256


@functools.lru_cache(maxsize=None)
def _pdl_device_policy(device_index: int) -> tuple[bool, int]:
    """Whether PDL is allowed on this device, and its SM count."""
    props = torch.cuda.get_device_properties(device_index)
    sm_version = props.major * 10 + props.minor
    allowed = sm_version >= 90 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
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
    EARLY_TRIGGER: tl.constexpr = False,
):
    """Gated rmsnorm(x), several short rows per program.

    Z is addressed token-major: row r reads z at
    (r // HEADS_PER_TOK) * stride_z_tok + (r % HEADS_PER_TOK) * N. With
    HEADS_PER_TOK == heads it reads a [num_tokens, heads, N] view whose
    (heads, N) block is contiguous per token, e.g. a column slice of a wider
    projection.
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
        # RMSNorm first rounds to the input dtype, then static quantization
        # multiplies by the rounded reciprocal of its input scale.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y *= tldevice.rcp_rn(tl.load(FP8_SCALE).to(tl.float32))
    y_off = rows[:, None].to(tl.int64) * stride_y_row + cols[None, :]
    tl.store(Y + y_off, y.to(Y.dtype.element_ty), mask=mask2d)
    if LAUNCH_WITH_PDL and not EARLY_TRIGGER:
        # Release only after both Rstd (when present) and Y have been stored.
        tl.extra.cuda.gdc_launch_dependents()


class RmsNormGatedTokenMajor(OpWrapper):
    """Return ``rmsnorm(x) * weight * gate(z)``, one launch, a new tensor.

    ``gate`` is ``sigmoid(z)`` (``"sigmoid"``, KDA's output gate) or ``z * sigmoid(z)`` (``"silu"``, GDN's).
    Row ``r = t * heads + h`` of ``x`` is gated by ``z[t, h, :]``.

    Args:
        x: ``[tokens * heads, N]`` bf16 (or fp16 / fp32) rows with a unit column stride; the row stride is free.
        z: ``[tokens, heads, N]`` gate whose ``(heads, N)`` block is dense per token; the token stride is free
            (a column slice of a wider projection).
        weight: ``[N]`` contiguous, any float dtype.
        eps: added to the mean square.
        fp8_scale: optional one-element fp32 tensor on x's device; with it the output is float8_e4m3fn.
        gate_activation: ``"sigmoid"`` or ``"silu"``.

    Returns:
        A new contiguous ``[tokens * heads, N]`` tensor in ``x.dtype`` (float8_e4m3fn with ``fp8_scale``).
    """

    ARCHS = frozenset({Arch.SM_100})
    # The op's own forms; the shapes the shipped target normalizes are in this entry's test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="silu gate (z * sigmoid(z)): the other gate form, 16 heads of 128",
            spec=dict(tokens=64, heads=16, width=128, eps=1e-5, gate_activation="silu"),
        ),
        *(
            Cell(
                why=f"row width {width}: another power-of-two row a program holds whole",
                spec=dict(tokens=48, heads=4, width=width, eps=1e-6, gate_activation="sigmoid"),
            )
            for width in (64, 256)
        ),
        Cell(
            why="6 heads: a row count that is not a multiple of the 4-row program (masked tail rows)",
            spec=dict(tokens=7, heads=6, width=128, eps=1e-5, gate_activation="sigmoid"),
        ),
        *(
            Cell(
                why=f"fp8 output at scale {scale}: rounded norm output, then static e4m3 quantization",
                spec=dict(
                    tokens=64,
                    heads=16,
                    width=128,
                    eps=1e-5,
                    gate_activation="sigmoid",
                    fp8_scale=scale,
                ),
            )
            for scale in (0.5, 3.0)
        ),
    )
    note = """
    Norm, weight and gate only; projections are the caller's. Norm before gate:
    the mean square is taken over x alone. fp32 math, one rounding to the output
    dtype. Inputs are not mutated; the output is new.

    One kernel: whole rows per program, four rows per program, so the row width
    must be a power of two of at most 256 and z's per-token (heads, N) block dense
    (is_valid). The module op's generic one-row kernel for other layouts is not
    part of this entry.

    fp8 quantization first rounds to x.dtype, then multiplies by the correctly
    rounded reciprocal of the fp32 scale and rounds to e4m3; the kernel reads only
    the first element of fp8_scale.

    PDL is used when TRTLLM_ENABLE_PDL is unset or 1, the device is SM90 or later
    and the grid has fewer programs than the device has SMs: the weight is loaded
    before the wait, dependents are released at the start. It changes
    scheduling, not results. Triton specializes on the first call per width,
    head count and dtype; warm it outside CUDA-graph capture.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: torch.Tensor | None = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        rows, width = x.shape
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn if fp8_scale is not None else x.dtype)
        grid_size = triton.cdiv(rows, _MULTIROW_ROWS)
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
                rows,
                eps,
                N=width,
                ROWS=_MULTIROW_ROWS,
                HEADS_PER_TOK=z.shape[1],
                GATE_SIGMOID=gate_activation == "sigmoid",
                WEIGHT_IS_DELTA=False,
                LAUNCH_WITH_PDL=launch_with_pdl,
                num_warps=_MULTIROW_NUM_WARPS,
                # Dependents (out_proj) prefetch their weights; they read Y only
                # after their own griddepcontrol.wait.
                EARLY_TRIGGER=_pdl_device_policy(device_index)[0],
                launch_pdl=launch_with_pdl,
            )
        return out

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: torch.Tensor | None = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        """fp32: norm before gate, the gate read token-major, one rounding to the output dtype. With
        ``fp8_scale``, the normalized value is rounded to ``x.dtype`` first, as the kernel does."""
        rows, width = x.shape
        zf = z.float().reshape(rows, width)
        xf = x.float()
        y = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        gate = torch.sigmoid(zf)
        if gate_activation == "silu":
            gate = gate * zf
        y = y * gate
        if fp8_scale is None:
            return y.to(x.dtype)
        return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: torch.Tensor | None = None,
        gate_activation: str = "silu",
    ) -> None:
        # The kernel checks nothing: every condition below is one it computes wrongly (or out of bounds)
        # rather than refuses.
        assert gate_activation in ("silu", "sigmoid"), (
            f"gate_activation must be 'silu' or 'sigmoid'; {gate_activation!r} computes the silu gate"
        )
        assert x.ndim == 2 and z.ndim == 3, "x must be [tokens * heads, N] and z [tokens, heads, N]"
        rows, width = x.shape
        assert z.shape[2] == width and z.shape[0] * z.shape[1] == rows, (
            f"z {tuple(z.shape)} must be [tokens, heads, {width}] with tokens * heads == {rows} x rows: "
            "row r reads z[r // heads, r % heads]"
        )
        assert width & (width - 1) == 0 and width <= _MULTIROW_MAX_N, (
            f"the row width must be a power of two of at most {_MULTIROW_MAX_N}, got {width}"
        )
        assert x.stride(-1) == 1, (
            "x's columns must be contiguous: the kernel steps them by one element"
        )
        assert z.stride(2) == 1 and z.stride(1) == width, (
            "z's (heads, N) block must be dense per token: the kernel reads head h at h * N"
        )
        assert weight.shape == (width,) and weight.is_contiguous(), (
            f"weight must be contiguous [{width}]"
        )
        if fp8_scale is not None:
            assert fp8_scale.numel() == 1, "fp8_scale must contain exactly one element"
            assert fp8_scale.dtype == torch.float32, "fp8_scale must have dtype torch.float32"
            assert fp8_scale.device == x.device, "fp8_scale must be on x's device"


rms_norm_gated_token_major = RmsNormGatedTokenMajor()
