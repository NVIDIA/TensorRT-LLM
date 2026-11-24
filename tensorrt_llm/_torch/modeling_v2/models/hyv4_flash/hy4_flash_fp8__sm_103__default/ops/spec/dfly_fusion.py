# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2's slot-0 fusion and its per-position hidden correction plus confidence head.

Macro entries composed of `flashinfer_rmsnorm`, `cublas_mm` and torch elementwise ops. Weights are
the checkpoint's BF16 ``[out, in]`` tensors; `w.t()` is `cublas_mm`'s column-major operand.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.cublas_mm__r11 import (  # noqa: E501
    cublas_mm,
)

#: The DFly config's `rms_norm_eps`.
DEFAULT_EPS = 1e-5


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()


def _assert_last_dim_contiguous(x: torch.Tensor, name: str) -> None:
    # The rmsnorm kernel assumes `stride(-1) == 1` and reads the wrong elements otherwise.
    assert x.stride(-1) == 1, f"{name}'s last dim must be contiguous"


def _assert_weight(w: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    # `cublas_mm` takes mat_b's leading dimension from `k`, not its stride, so a `w` sliced from a
    # wider buffer is accepted and read wrong; only a dense row-major `w` is safe.
    assert w.shape == shape, f"{name} must be {shape}; got {tuple(w.shape)}"
    assert w.is_contiguous(), f"{name} must be dense row-major"


class DflyFusion(OpWrapper):
    """`slot0 = anchors + gate * eh_proj(cat(enorm(anchors), hnorm(target_hidden)))`, one row per request."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="slot-0 fusion: one draft request, the per-step decode shape",
            spec=dict(rows=1, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: one width-6 draft block",
            spec=dict(rows=6, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: the largest cuda-graph batch, 64 draft requests",
            spec=dict(rows=64, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: 64 requests x width 6, the largest block the engine materializes",
            spec=dict(rows=384, dtype=torch.bfloat16),
        ),
    )

    def __call__(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        fused = cublas_mm(
            torch.cat(
                (
                    flashinfer_rmsnorm(anchors, enorm, eps),
                    flashinfer_rmsnorm(target_hidden, hnorm, eps),
                ),
                -1,
            ),
            eh_proj.t(),
        )
        return anchors + gate * fused

    def reference(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """fp32 throughout, rounded once."""
        with true_fp32_matmul():
            fused = (
                torch.cat(
                    (
                        _rmsnorm_fp32(anchors, enorm, eps),
                        _rmsnorm_fp32(target_hidden, hnorm, eps),
                    ),
                    -1,
                )
                @ eh_proj.float().t()
            )
            out = anchors.float() + gate.float() * fused
        return out.to(anchors.dtype)

    def is_valid(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> None:
        _assert_last_dim_contiguous(anchors, "anchors")
        _assert_last_dim_contiguous(target_hidden, "target_hidden")
        hidden = enorm.numel()
        assert hnorm.shape == (hidden,), f"hnorm must be [{hidden}]; got {tuple(hnorm.shape)}"
        _assert_weight(eh_proj, (hidden, 2 * hidden), "eh_proj")
        # A [rows, 1] or [1, H] gate broadcasts into a plausible wrong fusion.
        assert gate.shape == (hidden,), f"gate must be [{hidden}]; got {tuple(gate.shape)}"


class DflyHiddenCorrection(OpWrapper):
    """`corrected = hidden + down(silu(gate(c)) * up(c))`, `c = cat(hidden_norm(hidden), embed_norm(prev))`.

    Also returns the confidence head's logit per row, `confidence_proj @ corrected + confidence_bias`
    (`[rows]`); the caller applies the sigmoid.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="hidden correction: one draft request, the per-step decode shape",
            spec=dict(rows=1, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: one width-6 draft block",
            spec=dict(rows=6, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: the largest cuda-graph batch, 64 draft requests",
            spec=dict(rows=64, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: 64 requests x width 6, the largest block the engine materializes",
            spec=dict(rows=384, dtype=torch.bfloat16),
        ),
    )

    note = """
    `gate_proj` and `up_proj` have the same shape, so a swap is not detectable here; it is a
    load-time mapping.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat(
            (
                flashinfer_rmsnorm(hidden, hidden_norm, eps),
                flashinfer_rmsnorm(previous_embed, embed_norm, eps),
            ),
            -1,
        )
        delta = cublas_mm(
            F.silu(cublas_mm(combined, gate_proj.t())) * cublas_mm(combined, up_proj.t()),
            down_proj.t(),
        )
        corrected = hidden + delta
        confidence = F.linear(corrected, confidence_proj, confidence_bias).squeeze(-1)
        return corrected, confidence

    def reference(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 throughout, rounded once."""
        with true_fp32_matmul():
            combined = torch.cat(
                (
                    _rmsnorm_fp32(hidden, hidden_norm, eps),
                    _rmsnorm_fp32(previous_embed, embed_norm, eps),
                ),
                -1,
            )
            delta = (
                F.silu(combined @ gate_proj.float().t()) * (combined @ up_proj.float().t())
            ) @ (down_proj.float().t())
            corrected = hidden.float() + delta
            confidence = corrected @ confidence_proj.float().t() + confidence_bias.float()
        return corrected.to(hidden.dtype), confidence.squeeze(-1).to(hidden.dtype)

    def is_valid(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> None:
        _assert_last_dim_contiguous(hidden, "hidden")
        _assert_last_dim_contiguous(previous_embed, "previous_embed")
        width = hidden_norm.numel()
        assert embed_norm.shape == (width,), (
            f"embed_norm must be [{width}]; got {tuple(embed_norm.shape)}"
        )
        _assert_weight(gate_proj, (width, 2 * width), "gate_proj")
        _assert_weight(up_proj, (width, 2 * width), "up_proj")
        _assert_weight(down_proj, (width, width), "down_proj")
        _assert_weight(confidence_proj, (1, width), "confidence_proj")
        assert confidence_bias.shape == (1,), (
            f"confidence_bias must be [1]; got {tuple(confidence_bias.shape)}"
        )


dfly_fusion = DflyFusion()
dfly_hidden_correction = DflyHiddenCorrection()
