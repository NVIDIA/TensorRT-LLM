# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA output gate on Rubin: gated RMS norm of per-head rows, the gate read from a token-major view."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.ssm.rubin_rms_norm_gated import rms_norm_gated_token_major as _rms_norm_gated


class RubinRmsNormGatedTokenMajor(OpWrapper):
    """Return `rmsnorm(x) * weight * gate(z)` per row, `gate` = sigmoid ("sigmoid") or z * sigmoid(z) ("silu").

    Row `r = t * heads + h` of `x` [tokens * heads, N] is gated by `z[t, h, :]`. With `fp8_scale`
    the result is quantized to float8_e4m3fn in the same kernel. A new tensor. `prefetch` changes
    only which kernel prefill-sized calls run (a CuTe DSL kernel that keeps more rows in flight), not
    the result.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank, T={tokens}: sigmoid gate in the qkvg GEMM's gate columns",
                spec=dict(tokens=tokens, heads=heads, width=128, eps=1e-5, gate_columns_of=4),
            )
            for heads in (96, 12)
            for tokens in (1, 37, 512)
        ),
        *(
            Cell(
                why=f"{heads} heads, a 16k-token context chunk",
                spec=dict(tokens=16384, heads=heads, width=128, eps=1e-5, gate_columns_of=4),
            )
            for heads in (96, 12)
        ),
        *(
            Cell(
                why=f"{heads} heads, T=512: g columns of the stacked [q|k|v|g|f_a|b] GEMM output, row pitch {pitch}",
                spec=dict(tokens=512, heads=heads, width=128, eps=1e-5, stacked=True),
            )
            for heads, pitch in ((96, 49376), (12, 6288))
        ),
        Cell(
            why="dense gate (no column slice) at 12 heads",
            spec=dict(tokens=64, heads=12, width=128, eps=1e-6, gate_columns_of=1),
        ),
        Cell(
            why="FP8 output (fp8_scale); no target passes it",
            spec=dict(tokens=37, heads=12, width=128, eps=1e-5, gate_columns_of=4, fp8_scale=0.05),
        ),
        *(
            Cell(
                why=f"prefetch, {heads} heads, {tokens} tokens: the stacked GEMM output's g columns",
                spec=dict(
                    tokens=tokens, heads=heads, width=128, eps=1e-5, stacked=True, prefetch=True
                ),
            )
            for heads, tokens in (
                (96, 16384),
                (96, 2953),
                (96, 283),
                (96, 129),
                (12, 131072),
                (12, 8191),
            )
        ),
        Cell(
            why="prefetch with an FP8 output: the multi-row kernel",
            spec=dict(
                tokens=2953,
                heads=12,
                width=128,
                eps=1e-5,
                gate_columns_of=4,
                fp8_scale=0.05,
                prefetch=True,
            ),
        ),
    )
    note = """
    The KDA layer's output gate on the context path (decode and verify gate
    inside their kernels): x is the contiguous [T * H, 128] bf16 core of
    rubin_kda_prefill, z the [T, H, 128] view of the g columns of the fused
    [q | k | v | g] projection (token stride 4 * H * 128) or of the
    specialized targets' [q | k | v | g | f_a | b] GEMM output (6288 or
    49376), head blocks contiguous; weight the o_norm weight,
    gate_activation "sigmoid".
    Inputs are not mutated; the output is new and contiguous; one final
    rounding to x.dtype (within 1e-2 relative of fp32). These shapes take
    the four-row token-major Triton kernel (power-of-two N <= 256, dense
    per-token head blocks), with PDL when the grid is below one wave.
    With prefetch=True (dep16's context server, kda_gated_rmsnorm_prefetch)
    a call of prefill size (where that kernel loops over tiles, from about
    27k rows on sm_107) without fp8_scale, with bf16 rows of 128 and
    16-byte aligned rows of x and z, runs the CuTe DSL kernel
    (kernels/ssm/rubin_rms_norm_gated_token_major_cute): a persistent grid,
    two rows per warp, several pairs of rows loaded before any is computed,
    and the Triton kernel's instructions and reduction tree; bit for bit the
    same rows.
    Graph-capturable once the Triton specialization (and, with prefetch, the
    CuTe DSL variant) is warm.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        prefetch: bool = False,
    ) -> torch.Tensor:
        return _rms_norm_gated(x, z, weight, eps, fp8_scale, gate_activation, prefetch)

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        prefetch: bool = False,
    ) -> torch.Tensor:
        """fp32: norm before gate, the gate read token-major, one final rounding; `prefetch` does
        not enter it."""
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
        return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
        prefetch: bool = False,
    ) -> None:
        assert isinstance(prefetch, bool), "prefetch is a flag"
        if fp8_scale is not None:
            # The kernel reads element 0 as the per-tensor scale.
            assert fp8_scale.numel() == 1, "fp8_scale must contain exactly one element"
            assert fp8_scale.dtype == torch.float32, "fp8_scale must have dtype torch.float32"


rubin_rms_norm_gated_token_major = RubinRmsNormGatedTokenMajor()
