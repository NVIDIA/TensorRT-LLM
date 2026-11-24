# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L0's dense MLP, first half: the fused gate/up low-M split-K GEMM with the SiTU epilogue
(``kernels/gemm/rubin_ull_k3_dense0.py``'s ``SituSplitKGemmKernel``)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import rubin_ull_k3_dense0 as _k

#: The TP16 slice of the checkpoint's 33792-wide dense MLP.
INTER = 33792 // 16
#: Kimi K3.1's activation_situ_beta and activation_situ_linear_beta.
BETA, LINEAR_BETA = 4.0, 25.0


def _cell(why: str, rows: int, linear_beta: Optional[float] = LINEAR_BETA) -> Cell:
    gu, _ = _k.default_tactics(rows)
    tactic = (gu.mma_m, gu.mma_n, gu.split_k, gu.ab_stages)
    return Cell(why=why, spec=dict(rows=rows, tactic=tactic, linear_beta=linear_beta))


def gate_up_halves(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate rows and the up rows of the kernel's fused [2 I, H] weight (64-row tiles of 32 gate
    rows then their 32 up rows: ``k3_dense0.fuse_gate_up``), as two [I, H] views' copies."""
    tiles = weight.view(-1, 2, _k.GU_TILE // 2, weight.shape[1])
    return tiles[:, 0].reshape(-1, weight.shape[1]), tiles[:, 1].reshape(-1, weight.shape[1])


class Dense0SituGateUp(OpWrapper):
    """``hidden[:, :I] = situ(bf16(x @ Wg^T), bf16(x @ Wu^T))`` with
    ``situ(g, u) = beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta)`` (``u`` as is
    when ``linear_beta`` is None), the gate/up products rounded to BF16 first and the activation in
    FP32, stored BF16.

    ``x`` [M, 7168] BF16 contiguous, ``weight`` the fused [2 I, 7168] gate/up tiles
    (``k3_dense0.fuse_gate_up``), ``hidden`` BF16 [M, P >= I] with unit column stride (columns at or
    past I untouched), ``grid_out`` any contiguous BF16 [M, 2 I] (sets the launch grid, never
    written), ``tactic`` a ``k3_dense0.SplitKTactic`` with ``mma_m`` 64. PDL as ``pdl``.
    Returns ``hidden``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 verify: 8 rows, K split 4", 8),
        _cell("b2 verify: 16 rows, K split 2", 16),
        _cell("b4 verify: 32 rows", 32),
        _cell("q1 decode, 3 requests (3 rows of the 8-row tile)", 3),
        _cell("24 rows: a partial 32-row tile", 24),
        _cell("without the up soft cap (linear_beta None)", 8, None),
    )

    note = """
    The served L0 dense MLP of the ULL GEN step (TP16 slice: I = 2112), up to 32 rows; wider steps
    take the cuBLAS + Triton SiTU fallback. The kernel compiles per (tactic, PDL, beta,
    linear_beta) on first use (``FlatDense0`` precompiles the served row counts before capture).
    The weight warp streams its first stages before the grid dependency wait; ``hidden`` is only
    written after it. Not bitwise the fallback (another split and summation order). tanh and
    sigmoid use ``ex2.approx``, within a few FP32 ulps of the exact functions.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        grid_out: torch.Tensor,
        tactic,
        beta: float,
        linear_beta: Optional[float],
        pdl: bool,
    ) -> torch.Tensor:
        return _k.run_situ_gate_up(x, weight, hidden, grid_out, tactic, beta, linear_beta, pdl=pdl)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        grid_out: torch.Tensor,
        tactic,
        beta: float,
        linear_beta: Optional[float],
        pdl: bool,
    ) -> torch.Tensor:
        """The [M, I] activation: FP64 gate/up products rounded to BF16 (as the kernel rounds
        them), the activation in FP64, rounded once. ``hidden``, ``grid_out``, ``tactic`` and
        ``pdl`` are not read."""
        gate_w, up_w = gate_up_halves(weight)
        g = (x.double() @ gate_w.double().t()).to(torch.bfloat16).double()
        u = (x.double() @ up_w.double().t()).to(torch.bfloat16).double()
        situ = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        return (situ * u).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        grid_out: torch.Tensor,
        tactic,
        beta: float,
        linear_beta: Optional[float],
        pdl: bool,
    ) -> None:
        """The launcher checks the shapes and the tactic (ValueError), not the operands' layout:
        ``x`` and ``weight`` are read as dense rows from their base pointers and ``hidden`` with its
        row stride from its base."""
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("dense0_situ_gate_up: x and weight must be contiguous")
        if any(t.dtype != torch.bfloat16 for t in (x, weight, hidden, grid_out)):
            raise ValueError("dense0_situ_gate_up: BF16 operands only")
        if hidden.data_ptr() % 4 or hidden.stride(0) % 2:
            raise ValueError("dense0_situ_gate_up: hidden rows must hold aligned BF16 pairs")


dense0_situ_gate_up = Dense0SituGateUp()
