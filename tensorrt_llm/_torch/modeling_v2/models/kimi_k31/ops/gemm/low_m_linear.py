# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 decode / verify BF16 projection: a measured low-M CuTe DSL kernel where the plan names one, else cuBLAS."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

# ``plan_choice`` (a plan's kernel choice by row count, before the operand checks of ``low_m_choice``) is also
# re-exported for the targets.
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import (
    LowMPlan,
    encode_choice,
    k31_low_m_gemm,
    plan_choice,
)


def low_m_choice(plan: Optional[LowMPlan], hidden_states: torch.Tensor, weight: torch.Tensor):
    """The plan's kernel choice for ``hidden_states @ weight.T``, or None when the call stays on cuBLAS.

    None also for operands the low-M kernels cannot read: not 2D, not BF16, not contiguous, or not 32-byte
    aligned (a freshly allocated activation and a parameter always are; a column slice is not).
    """
    if (
        plan is None
        or hidden_states.dim() != 2
        or hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or not weight.is_contiguous()
        or hidden_states.data_ptr() % 32
        or weight.data_ptr() % 32
    ):
        return None
    return plan_choice(plan, hidden_states.shape[0], weight.shape[0], weight.shape[1])


def _cell(why: str, m: int, n: int, k: int, layout: str, kind: str) -> Cell:
    """``kind`` is the branch the layout's plan takes for ``[m, k] x [k, n]``; the test holds it to that."""
    return Cell(why=why, spec=dict(m=m, n=n, k=k, layout=layout, kind=kind))


class K31LowMLinear(OpWrapper):
    """``hidden_states @ weight.T``: BF16 ``[M, K] x [N, K]^T -> [M, N]``, FP32 accumulation, one launch.

    ``weight`` is a row-major ``nn.Linear`` weight (or a packed concatenation of several). ``plan`` maps
    ``(N, K)`` and an M range to one of FlashInfer's CuTe DSL low-M kernels (``kernels/gemm/low_m_gemm``:
    direct, cluster split-K, TGV); the per-layout plans are in ``ops/gemm/low_m_plans``. A shape or M the plan
    does not name runs ``F.linear`` (cuBLAS), which is what the general target runs for these projections
    today. ``pdl`` launches the CuTe DSL kernels with programmatic dependent launch (they wait on the previous
    grid before reading ``hidden_states``); it does not affect the cuBLAS branch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "TEP32 KDA [q|k|v|g] at a DFlash2 width-8 verify (8 rows): split-K",
            8,
            1536,
            7168,
            "tep32",
            "splitk",
        ),
        _cell(
            "TEP32 packed DSA hidden projection with the gate at 8 rows: split-K",
            8,
            2656,
            7168,
            "tep32",
            "splitk",
        ),
        _cell(
            "TEP32 MLA q_b at 5 rows (one width-5 verify): direct", 5, 576, 1536, "tep32", "direct"
        ),
        _cell("TEP32 indexer wq_b at 8 rows: TGV", 8, 4096, 1536, "tep32", "tgv"),
        _cell(
            "TEP16 KDA [q|k|v|g] at one decode row: chunked split-K (the general target's bits)",
            1,
            3072,
            7168,
            "tep16",
            "splitkc",
        ),
        _cell("TEP16 MLA q_b at 6 rows (one width-6 verify): TGV", 6, 1152, 1536, "tep16", "tgv"),
        _cell(
            "TEP16 KDA [q|k|v|g] at 24 rows: chunked split-K (the general target's bits)",
            24,
            3072,
            7168,
            "tep16",
            "splitkc",
        ),
        _cell("TEP16 KDA [q|k|v|g] at 72 rows (B12 x 6): TGV", 72, 3072, 7168, "tep16", "tgv"),
        _cell(
            "TEP16 shared-expert gate_up at 40 rows (B8 x 5): chunked TGV, cuBLAS's (19, 38) split",
            40,
            768,
            7168,
            "tep16_shared",
            "tgvchunk",
        ),
        _cell(
            "TEP16 shared-expert gate_up at 60 rows (B12 x 5): chunked TGV, cuBLAS's (14, 28, 42) split",
            60,
            768,
            7168,
            "tep16_shared",
            "tgvchunk",
        ),
        _cell(
            "TEP16 KDA gate pair [f_a|b] at 80 rows (B16 x 5): wide chunked split-K (the general target's bits)",
            80,
            136,
            7168,
            "tep16",
            "splitkc",
        ),
        _cell(
            "TEP16 KDA [q|k|v|g] at 2954 rows (a prefill chunk): cuBLAS",
            2954,
            3072,
            7168,
            "tep16",
            "cublas",
        ),
        _cell("DEP16 KDA [q|k|v|g] at 16 rows: TGV", 16, 49152, 7168, "dep16", "tgv"),
        _cell(
            "DEP16 packed DSA hidden projection at 112 rows (~14 requests x 8): cuBLAS",
            112,
            14560,
            7168,
            "dep16",
            "cublas",
        ),
    )

    note = """
    Numerics: every branch accumulates in FP32 and rounds once to BF16; the kernels differ from cuBLAS (and
    from each other) only in summation order, so a plan row is not in general bitwise the cuBLAS call it
    replaces. Up to 32 rows the TEP16 plan does give the general target's bits on every projection but the
    packed DSA hidden ones, and on the KDA gate pair ``[f_a | b]`` wherever it plans it (``low_m_plans``).

    Compilation: the CuTe DSL kernels JIT-compile on the first call per tactic (the direct kernel per M, N, K
    as well) and stay cached for the process; that first call must come before CUDA-graph capture, which the
    engine's warmup forward at each captured batch size does.

    Host cost: a planned call spends tens of microseconds of Python (DLPack conversion, launch), which a CUDA
    graph removes and an eager step does not. Plans are for the graph-captured generation steps.

    Hard limits: the direct and split-K kernels serve M <= 32 (plans name them only there; the chunked split-K's
    wide token tiles serve any M); K % 128 == 0 for every kernel (all K3.1 projections but the TP32
    shared-expert down projection, K 192, which no plan lists).
    """

    def __call__(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, plan: Optional[LowMPlan], pdl: bool
    ) -> torch.Tensor:
        choice = low_m_choice(plan, hidden_states, weight)
        if choice is None:
            return F.linear(hidden_states, weight)
        kind, params = encode_choice(choice)
        return k31_low_m_gemm(hidden_states, weight, kind, params, pdl)

    def reference(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, plan: Optional[LowMPlan], pdl: bool
    ) -> torch.Tensor:
        """The FP32 product rounded once to BF16 (TF32 off). ``plan`` and ``pdl`` do not change the value."""
        with true_fp32_matmul():
            return (hidden_states.float() @ weight.float().t()).to(hidden_states.dtype)

    def is_valid(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, plan: Optional[LowMPlan], pdl: bool
    ) -> None:
        # The kernels read K from the activation; a weight of another inner size is read with it.
        assert weight.dim() == 2 and hidden_states.shape[-1] == weight.shape[1], (
            "hidden_states [..., K] x weight [N, K]"
        )


k31_low_m_linear = K31LowMLinear()


def k31_planned_projection(
    x: torch.Tensor, weight: torch.Tensor, plan: Optional[LowMPlan], pdl: bool
) -> torch.Tensor:
    """``k31_low_m_linear`` over the flattened leading dims of ``x`` (a verify step's ``[B, T, hidden]``)."""
    if x.dim() == 2:
        return k31_low_m_linear(x, weight, plan, pdl)
    out = k31_low_m_linear(x.reshape(-1, x.shape[-1]), weight, plan, pdl)
    return out.view(*x.shape[:-1], weight.shape[0])
