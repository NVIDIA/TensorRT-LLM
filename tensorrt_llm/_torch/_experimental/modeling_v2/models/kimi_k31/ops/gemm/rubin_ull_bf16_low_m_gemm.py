# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A planned low-M BF16 projection on one of FlashInfer's three CuTe DSL kernels (direct, TGV,
cluster split-K), the tactic a plan row of ``kernels/gemm/rubin_ull_low_m_gemm_plans.py`` names."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _cell(why: str, rows: int, n: int, k: int, tactic: tuple) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, n=n, k=k, tactic=tactic))


class Bf16LowMGemm(OpWrapper):
    """``out = x @ weight.T`` for BF16 ``x`` [M, K] and the stored BF16 ``weight`` [N, K], FP32
    accumulation rounded once to BF16, on the kernel ``tactic`` names:

    * ``("direct", block_size, outputs_per_block)``: ``dense_bf16_gemm_direct`` (CUDA-core dot
      products, ``M`` rows per block);
    * ``("tgv", index)``: ``tgv_gemm_cute_ext`` tactic ``index``;
    * ``("splitk", mma_m, mma_n, split_k, ab_stages)``: ``dense_bf16_gemm_sm100_splitk`` (each
      cluster rank an exact K slice in FP32, reduced through DSMEM in a fixed order).

    ``out`` contiguous BF16 [M, N]; PDL as ``enable_pdl``. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The tactics the target's plans (ULL_DECODE_PLAN in the helix16 target's modeling.py,
    # KIMI_K31_DECODE_PLAN, the helix16 drafter's _DRAFT_GEMM_PLAN) give its banks at the served
    # rows; N and K are this rank's (TP16 / Helix CP16).
    CELLS: tuple[Cell, ...] = (
        _cell(
            "packed DSA [hidden | gate] bank, b4 verify (32 rows, past the warp split-K's 16)",
            32,
            3040,
            7168,
            ("splitk", 64, 32, 2, 9),
        ),
        _cell("packed DSA bank, b1 verify (8 rows)", 8, 3040, 7168, ("splitk", 128, 8, 4, 6)),
        _cell("packed DSA bank, b2 verify (16 rows)", 16, 3040, 7168, ("splitk", 64, 16, 2, 11)),
        _cell("packed DSA bank, 2 rows: the direct kernel", 2, 3040, 7168, ("direct", 128, 2)),
        _cell(
            "DSA hidden bank (no packed gate), b1 verify", 8, 2272, 7168, ("splitk", 128, 8, 4, 6)
        ),
        _cell(
            "DSA hidden bank, 28 rows: K split 4 at mma_n 32",
            28,
            2272,
            7168,
            ("splitk", 64, 32, 4, 8),
        ),
        _cell(
            "DSA gate bank (6 local heads x 128), b2 verify",
            16,
            768,
            7168,
            ("splitk", 64, 8, 4, 10),
        ),
        _cell("DSA gate bank, b4 verify", 32, 768, 7168, ("splitk", 64, 16, 4, 8)),
        _cell("DSA gate bank, q1 decode (4 rows, direct)", 4, 768, 7168, ("direct", 128, 2)),
        _cell("packed q_b bank, q1 decode row (direct)", 1, 22528, 1536, ("direct", 32, 4)),
        _cell("DSA hidden bank, 48 rows: the TGV kernel", 48, 2272, 7168, ("tgv", 18)),
        _cell(
            "DFlash2 drafter conv_proj and gate_up / 16, one request's block (8 rows)",
            8,
            1792,
            7168,
            ("splitk", 64, 8, 4, 12),
        ),
        _cell(
            "DFlash2 drafter conv_proj and gate_up / 16, two requests' blocks (16 rows)",
            16,
            1792,
            7168,
            ("splitk", 64, 16, 4, 10),
        ),
        _cell("DFlash2 drafter [q_a | kv_a], 16 rows", 16, 2112, 7168, ("splitk", 64, 16, 4, 6)),
        _cell("DFlash2 drafter q_b (6 heads), 16 rows", 16, 1152, 1536, ("splitk", 64, 16, 4, 6)),
        _cell("DFlash2 drafter q_a, 8 rows", 8, 1536, 7168, ("splitk", 64, 8, 4, 12)),
        _cell("DFlash2 drafter kv_a, 8 rows", 8, 576, 7168, ("splitk", 64, 8, 4, 12)),
        _cell("DFlash2 drafter q_b (96 heads), 8 rows", 8, 18432, 1536, ("splitk", 128, 8, 1, 6)),
        _cell("DFlash2 drafter o_proj (6 heads), 8 rows", 8, 7168, 768, ("tgv", 18)),
        _cell("DFlash2 drafter down / 16, 8 rows", 8, 7168, 896, ("tgv", 18)),
        _cell(
            "DFlash2 drafter history: the five layers' kv_a, 8 accepted rows",
            8,
            2880,
            7168,
            ("splitk", 64, 8, 2, 12),
        ),
    )

    note = """
    Each kernel compiles once per (shape class, tactic, PDL) on its first call: make it before
    CUDA-graph capture. The tactic is chosen by the caller from a measured plan; a kernel and a
    tactic that do not fit the shape raise (ValueError from the FlashInfer launcher) before any
    launch. Not bitwise cuBLAS, and the three kernels are not bitwise each other: each sums the same
    BF16 products in its own FP32 order.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        tactic: tuple,
        enable_pdl: bool,
    ) -> torch.Tensor:
        if tactic[0] == "direct":
            from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
                DirectTactic,
                run_direct_dense,
            )

            return run_direct_dense(
                x, weight.t(), out, enable_pdl, DirectTactic(*tactic[1:], x.shape[0])
            )
        if tactic[0] == "tgv":
            from flashinfer.gemm.kernels.tgv_gemm_cute_ext import run_tgv_cute_ext

            return run_tgv_cute_ext(x, weight.t(), None, out, enable_pdl, tactic[1])
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        return run_splitk_dense(x, weight.t(), None, out, enable_pdl, SplitKTactic(*tactic[1:]))

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        tactic: tuple,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """The FP64 product rounded once to BF16; ``out``, ``tactic`` and ``enable_pdl`` are not
        read."""
        return (x.double() @ weight.double().t()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        tactic: tuple,
        enable_pdl: bool,
    ) -> None:
        """The launchers check shapes, layouts and alignment themselves (ValueError). What they do
        not check: an ``out`` that shares memory with ``x`` or ``weight`` is stored to while other
        CTAs still read the operands, a silently wrong product."""
        for operand, name in ((x, "x"), (weight, "weight")):
            if _overlap(out, operand):
                raise ValueError(f"bf16_low_m_gemm: out must not share memory with {name}")


def _span(t: torch.Tensor) -> tuple[int, int]:
    """``[first, last)`` bytes ``t`` addresses."""
    if t.numel() == 0:
        return t.data_ptr(), t.data_ptr()
    last = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return t.data_ptr(), t.data_ptr() + (last + 1) * t.element_size()


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    (a0, a1), (b0, b1) = _span(a), _span(b)
    return a0 < b1 and b0 < a1


bf16_low_m_gemm = Bf16LowMGemm()
