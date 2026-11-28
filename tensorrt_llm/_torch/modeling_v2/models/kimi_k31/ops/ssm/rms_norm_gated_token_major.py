# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA output gate: gated RMS norm of per-head rows, the gate read from a token-major [tokens, heads, N] view."""

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.rms_norm_gated import (
    rms_norm_gated_token_major as _rms_norm_gated_token_major,
)


class RmsNormGatedTokenMajor(OpWrapper):
    """Return `rmsnorm(x) * weight * gate(z)` per row, `gate` = sigmoid ("sigmoid") or z * sigmoid(z) ("silu").

    Row `r = t * heads + h` of `x` [tokens * heads, N] is gated by `z[t, h, :]`. With `fp8_scale`
    the result is quantized to float8_e4m3fn in the same kernel. A new tensor.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank, T={tokens}: sigmoid gate in the qkvg GEMM's gate columns",
                spec=dict(tokens=tokens, heads=heads, width=128, eps=1e-5, gate_columns_of=4),
            )
            for heads in (96, 48, 24, 12, 6, 3)
            for tokens in (1, 37, 512)
        ),
        *(
            Cell(
                why=f"{heads} heads, long prefill chunk T={tokens}",
                spec=dict(tokens=tokens, heads=heads, width=128, eps=1e-5, gate_columns_of=4),
            )
            for heads, tokens in ((96, 4096), (12, 8192))
        ),
        Cell(
            why="dense gate (no column slice) at 6 heads",
            spec=dict(tokens=64, heads=6, width=128, eps=1e-6, gate_columns_of=1),
        ),
        *(
            Cell(
                why=f"sm_107 context: {heads} heads, a 16k-token chunk",
                spec=dict(tokens=16384, heads=heads, width=128, eps=1e-5, gate_columns_of=4),
            )
            for heads in (96, 12)
        ),
    )
    note = """
    The KDA layer's output gate on its prefill path (the one-token decode and
    the fused verify apply the gate inside their kernels): x is the
    contiguous [T * H, 128] bf16 core of trtllm::kda_prefill, z the [T, H, 128]
    view of the fused [q | k | v | g] projection's gate columns (token stride
    4 * H * 128, head blocks contiguous), weight the bf16 o_norm weight, eps
    the layer's rms_norm_eps, gate_activation "sigmoid", no fp8 scale. Norm,
    weight and gate only. Inputs are not mutated; the output is contiguous and
    new; one final rounding to x.dtype (within 1e-2 relative of fp32). The
    target's shapes take the four-row token-major Triton kernel (power-of-two
    N <= 256, dense per-token head blocks); it uses PDL when TRTLLM_ENABLE_PDL
    is unset or 1 and the grid is below one wave. Graph-capturable after the
    Triton specialization is warm. ARCHS lists only the architectures the cells
    were driven on: sm_103 (GB300) and sm_107 (VR200).
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
        return _rms_norm_gated_token_major(x, z, weight, eps, fp8_scale, gate_activation)

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
