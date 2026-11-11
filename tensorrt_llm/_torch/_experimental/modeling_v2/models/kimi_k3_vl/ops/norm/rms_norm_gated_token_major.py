# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated RMS norm of per-head rows, with the gate read from a token-major [tokens, heads, N] view."""

from typing import Optional

import torch

import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class RmsNormGatedTokenMajor(OpWrapper):
    """Return `rmsnorm(x) * weight * gate(z)` per row, `gate` = sigmoid ("sigmoid") or z * sigmoid(z) ("silu").

    Row `r = t * heads + h` of `x` [tokens * heads, N] is gated by `z[t, h, :]`. With `fp8_scale`
    the result is quantized to float8_e4m3fn in the same kernel. A new tensor.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 {heads} heads, T={tokens}: sigmoid gate in a full-rank column slice",
                spec=dict(
                    tokens=tokens,
                    heads=heads,
                    width=128,
                    dtype=torch.bfloat16,
                    eps=1e-5,
                    gate_activation="sigmoid",
                    extra=3 * heads * 128,
                ),
            )
            for heads in (6, 12, 24, 96)
            for tokens in (1, 7, 64, 1024, 8192)
        ),
        *(
            Cell(
                why=f"width {width}, {gate}: multi-row power-of-two row length and gate form",
                spec=dict(
                    tokens=tokens,
                    heads=heads,
                    width=width,
                    dtype=torch.bfloat16,
                    eps=1e-6,
                    gate_activation=gate,
                ),
            )
            for width in (32, 64, 128, 256)
            for tokens, heads, gate in ((48, 4, "silu"), (192, 1, "sigmoid"))
        ),
        *(
            Cell(
                why=f"generic fallback at width {width}: non-power-of-two or above 256",
                spec=dict(
                    tokens=32,
                    heads=heads,
                    width=width,
                    dtype=torch.bfloat16,
                    eps=1e-5,
                    gate_activation="sigmoid",
                ),
            )
            for width, heads in ((96, 4), (512, 2))
        ),
        Cell(
            why="head-strided gate: multi-row fallback with one head per row",
            spec=dict(
                tokens=16,
                heads=6,
                width=128,
                dtype=torch.bfloat16,
                eps=1e-5,
                gate_activation="sigmoid",
                head_strided=True,
            ),
        ),
        *(
            Cell(
                why=f"fp8 scale {scale}: rounded norm output before static quantization",
                spec=dict(
                    tokens=64,
                    heads=6,
                    width=128,
                    dtype=torch.bfloat16,
                    eps=1e-5,
                    gate_activation="sigmoid",
                    fp8_scale=scale,
                ),
            )
            for scale in (0.5, 1.0, 3.0)
        ),
    )
    note = """
    Norm, weight and gate only; projections are the caller's. Inputs are not
    mutated; output is contiguous and new. fp8 quantization first rounds to
    x.dtype, then multiplies by the correctly rounded reciprocal of the fp32
    scale and rounds to e4m3. Other accepted input dtypes are not certified here.
    fp8_scale must be None or a single-element fp32 tensor on x's CUDA device;
    the kernel reads only its first element, not a per-column scale.

    No metadata or state. Triton specializes on first call; warm outside graph
    capture, then replay rewritten inputs bit for bit. A power-of-two N<=256
    with dense last strides and dense per-token (heads,N) blocks uses the
    four-row token-major kernel; token stride is free. Otherwise z is reshaped
    (possibly copied), and the row-level multi-row or generic kernel is used.
    weight is made contiguous. Shape, gate-name and last-stride failures are
    rejected by the op.
    The multi-row kernel uses PDL when TRTLLM_ENABLE_PDL is unset or 1, SM>=9.0,
    and the grid has fewer CTAs than SMs: wait before the first load, release
    dependents after the final store. The module-level RMSNorm instead takes
    z already row-aligned with x.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        return torch.ops.trtllm.rms_norm_gated_token_major(
            x, z, weight, eps, fp8_scale, gate_activation
        )

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        """fp32 reference: norm before gate, the gate read token-major, one final rounding."""
        rows, n = x.shape
        zf = z.float().reshape(rows, n)
        xf = x.float()
        y = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        gate = torch.sigmoid(zf)
        if gate_activation == "silu":
            gate = gate * zf
        y = y * gate
        if fp8_scale is None:
            return y.to(x.dtype)
        # The output is first rounded to x.dtype, then scaled by the rounded reciprocal of the scale.
        return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> None:
        if fp8_scale is not None:
            assert fp8_scale.numel() == 1, "fp8_scale must contain exactly one element"
            assert fp8_scale.dtype == torch.float32, "fp8_scale must have dtype torch.float32"
            assert fp8_scale.device == x.device, "fp8_scale must be on x's device"


rms_norm_gated_token_major = RmsNormGatedTokenMajor()
