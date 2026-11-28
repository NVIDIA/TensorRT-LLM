# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank's piece of the shared expert for a decode-sized step, as stock runs it (`full_tp_shared_expert_forward`):
the low-M warp-MMA kernel (`low_m_gemm.mxfp8_lowm_linear`: bf16 activations times the E4M3 weights dequantized exactly
with their per-row UE8M0 scales, fp32 accumulation) for gate | up, the clamped SwiGLU (`swiglu`, `silu_and_mul`), the
low-M kernel again for down. Stock takes this path for every bf16 step of at most 64 rows (`_SHARED_EXPERT_FULL_TP_MAX_
ROWS`), eager and graph alike (it is not gated on the decode-graph flag as the attention projections are)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

MAX_ROWS = 64  # stock's `_SHARED_EXPERT_FULL_TP_MAX_ROWS`


def shared_expert_lowm_serves(x: torch.Tensor) -> bool:
    """Whether stock runs this input on the piece path (`_full_tp_shared`):
    bf16 `[1..64, H]`, contiguous, 16-byte aligned."""
    return (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


class SharedExpertLowm(OpWrapper):
    """`y = W2 swiglu(W13 x)` of this rank's piece: `x` bf16 `[m <= 64, H]`, `w13` E4M3 `[2P, H]` (gate rows then up
    rows, each zero-padded to P) with per-row UE8M0 scales `s13` uint8 `[2P, H/32]`, `w2` E4M3 `[H, P]` with `s2` uint8
    `[H, P/32]`, `limit` the SwiGLU clamp (0: none). Returns bf16 `[m, H]`, this rank's partial sum. With a bf16 limit
    (`swiglu_out_supported`) gate | up stores the clamped SwiGLU itself: the bits of the three launches (gate_up GEMM,
    `swiglu`, down GEMM), which the test checks."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows (one request, five drafts), 384-wide piece",
            spec=dict(m=6, piece=384),
        ),
        Cell(
            why="a 14-row prompt (eager context chunk), as stock's eager step",
            spec=dict(m=14, piece=384),
        ),
        Cell(why="one row", spec=dict(m=1, piece=384)),
        Cell(why="16 rows", spec=dict(m=16, piece=384)),
        Cell(why="64 rows: stock's row limit of the piece path", spec=dict(m=64, piece=384)),
    )
    note = (
        "Exact-weight fp32 products; the intermediate and the "
        "output round once to bf16 (the reference states the band)."
    )

    def __call__(
        self,
        x: torch.Tensor,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear, swiglu_out_supported
        from tensorrt_llm._torch.modules.swiglu import swiglu

        lim = limit if limit > 0 else None
        if lim is not None and swiglu_out_supported(x.shape[0], x.shape[1], w13, lim):
            # as the built-in shared expert (#22359): gate | up stores the clamped SwiGLU itself and releases the
            # down GEMM right after its own dependency wait (two launches, the three launches' bits)
            hidden = mxfp8_lowm_linear(x, w13, s13, swiglu_out_limit=lim, release_after_wait=True)
            return mxfp8_lowm_linear(hidden, w2, s2)
        gate_up = mxfp8_lowm_linear(x, w13, s13)
        return mxfp8_lowm_linear(swiglu(gate_up, swiglu_limit=lim), w2, s2)

    def reference(
        self,
        x: torch.Tensor,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        """fp32 products of the dequantized weights; the intermediate rounded to bf16 as the kernel's output is."""

        def deq(w, s):
            return (
                w.float()
                * torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)[:, : w.shape[1]]
            )

        gu = (x.float() @ deq(w13, s13).t()).to(torch.bfloat16).float()
        gate, up = gu.chunk(2, dim=-1)
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        h = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
        return (h @ deq(w2, s2).t()).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        limit: float,
    ) -> None:
        assert shared_expert_lowm_serves(x), (
            "x is bf16 [1..64, H], contiguous and 16-byte aligned (stock's piece path)"
        )
        h = x.shape[1]
        assert w13.dtype == w2.dtype == torch.float8_e4m3fn and s13.dtype == s2.dtype == torch.uint8
        assert w13.shape[1] == h and w13.shape[0] % 16 == 0 and s13.shape == (w13.shape[0], h // 32)
        assert w2.shape == (h, w13.shape[0] // 2) and s2.shape == (h, w2.shape[1] // 32)
        assert limit >= 0


shared_expert_lowm = SharedExpertLowm()
