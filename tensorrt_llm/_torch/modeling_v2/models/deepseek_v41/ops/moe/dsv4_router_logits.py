# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE router's logits: fp32 `x @ W.T` over the checkpoint's bf16 gate, as stock's gate computes them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if TYPE_CHECKING:
    from tensorrt_llm._torch.custom_ops.router_logit_partials import (
        RouterLogitPartials,  # noqa: F401
    )


class Dsv4RouterLogits(OpWrapper):
    """fp32 `[M, E]` logits of bf16 rows `x` `[M, H]` against the bf16 gate `weight` `[E, H]`, accumulated in fp32:
    stock's gate itself (`DeepseekV4Gate.forward` -> `router_gemm_autotuned`). Its runner (`RouterGemmRunner`) has two
    tactics, 0 the engine's low-M warp-MMA kernel with bf16 weights (`low_m_gemm.bf16_lowm_linear`, up to 64 rows) and
    -1 `dsv3_router_gemm_op` (its per-shape kernel, else cuBLAS); the autotuner picks one per power-of-two row bucket up
    to 64 (more rows, and a cache miss, run -1). The two sum in different orders, so the logits follow stock's pick bit
    for bit only by asking the same tuner (GB300 tep8: stock runs tactic 0 on a 19-row prompt and -1 on 36- and 48-row
    ones, whose logits a fixed low-M kernel missed by ~1e-6 relative -- enough to move one row's routing).
    With `partials` and 12..1280 rows on sm_100 / sm_103 (65..1280 elsewhere) the result is the engine's
    `RouterLogitPartials` instead: fp32 K-chunk partials `[S, M, E]` from one strided-batched GEMM, which
    `dsv4_route_nvfp4_quant` sums (s = 0..S-1) as it loads them, so the split-K reduce launch (from 12 rows, the tuned
    gate GEMM) gives way. That is the engine's router at these sizes for DeepSeek-V4.1, which opts its MoEs into
    partials from `router_logit_partials.SMALL_M_MIN_ROWS` rows (`moe_router_partials_small_m`:
    `applies(small_m=True)`). Only the summation order changes.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows (one token and five drafts)",
            spec=dict(m=6, hidden=5120, experts=384),
        ),
        Cell(why="one row", spec=dict(m=1, hidden=5120, experts=384)),
        Cell(why="16 rows", spec=dict(m=16, hidden=5120, experts=384)),
        Cell(
            why="dep32 decode / tep8 verify of four requests: 24 rows",
            spec=dict(m=24, hidden=5120, experts=384),
        ),
        Cell(why="64 rows: the low-M kernel's edge", spec=dict(m=64, hidden=5120, experts=384)),
        Cell(
            why="65 rows without partials: the runner's tactic -1",
            spec=dict(m=65, hidden=5120, experts=384),
        ),
        Cell(
            why="a prefill chunk: the runner's tactic -1",
            spec=dict(m=2048, hidden=5120, experts=384),
        ),
        Cell(
            why="11 rows with partials: the tuned gate GEMM (below the partials' rows)",
            spec=dict(m=11, hidden=5120, experts=384, partials=True),
        ),
        Cell(
            why="12 rows: the first K-chunk partials row count",
            spec=dict(m=12, hidden=5120, experts=384, partials=True),
        ),
        Cell(
            why="dep32 decode, 48 rows: K-chunk partials",
            spec=dict(m=48, hidden=5120, experts=384, partials=True),
        ),
        Cell(why="64 rows with partials", spec=dict(m=64, hidden=5120, experts=384, partials=True)),
        Cell(
            why="dep8 decode, 160 rows: K-chunk partials",
            spec=dict(m=160, hidden=5120, experts=384, partials=True),
        ),
        Cell(
            why="dep8 decode, 1280 rows: K-chunk partials (the largest)",
            spec=dict(m=1280, hidden=5120, experts=384, partials=True),
        ),
    )
    note = (
        "Both paths accumulate in fp32 over the same bf16 operands; "
        "they differ only in summation order (gate 1e-5 relative)."
    )

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, partials: bool = False
    ) -> "torch.Tensor | RouterLogitPartials":
        if partials:
            from tensorrt_llm._torch.custom_ops import router_logit_partials as rlp

            if rlp.applies(
                x, weight, small_m=True
            ):  # DeepSeek-V4.1's opt-in (`moe_router_partials_small_m`)
                return rlp.router_logit_partials(x, weight)
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import router_gemm_autotuned

        return router_gemm_autotuned(x if x.is_contiguous() else x.contiguous(), weight)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, partials: bool = False
    ) -> torch.Tensor:
        """float64 product of the bf16 operands, rounded to fp32."""
        return (x.double() @ weight.double().t()).float()

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, partials: bool = False) -> None:
        assert (
            x.dtype == weight.dtype == torch.bfloat16
            and x.dim() == weight.dim() == 2
            and x.shape[1] == weight.shape[1]
        )
        assert weight.is_contiguous() and x.stride(-1) == 1


dsv4_router_logits = Dsv4RouterLogits()
