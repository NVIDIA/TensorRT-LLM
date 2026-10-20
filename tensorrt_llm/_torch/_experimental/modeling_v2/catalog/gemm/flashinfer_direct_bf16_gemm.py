# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 matmul for one or two rows over a long K, through FlashInfer's register-prefetch kernel."""

from __future__ import annotations

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    DirectTactic,
    default_tactic,
    run_direct_dense,
)

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: The tactic each certified shape runs, `(block_size, outputs_per_block, rows_per_block)`,
#: keyed by `(M, K, N)`: the fastest of the kernel's own tactic space in CUDA-graph replay on a
#: GB200, rotating weights so no call reads them from L2. Fixed rather than autotuned so every
#: rank of a group runs the same kernel on the same shape; any other shape takes the kernel's
#: default tactic.
TACTICS: dict[tuple[int, int, int], DirectTactic] = {
    (1, 8192, 4384): DirectTactic(128, 4, 1),
    (1, 8192, 1280): DirectTactic(128, 4, 1),
    (1, 8192, 2048): DirectTactic(128, 4, 1),
    (2, 8192, 4384): DirectTactic(64, 4, 2),
    (2, 8192, 1280): DirectTactic(32, 1, 2),
    (2, 8192, 2048): DirectTactic(64, 1, 2),
}


class FlashinferDirectBf16Gemm(OpWrapper):
    """`mat_a @ mat_b` in bf16, fp32-accumulated, one launch, a new `[M, N]` out.

    Each CTA owns a few output columns and keeps the whole K extent of their
    weight rows in registers; each thread accumulates its slice of K in fp32 and
    the warps' partials are summed in shared memory in a fixed order, so the result
    depends only on the shape and the tactic. `mat_b` is read column-major, which
    is what a stored weight's `.t()` already is, as for `cublas_mm`.

    Fusion boundary: the product only. No bias, no out_dtype other than bf16.
    """

    ARCHS = frozenset({Arch.SM_100})

    # The nemotron_h BF16 targets' decode projections at one and two rows: the
    # rank's mamba in_proj (8192 -> 4384), qkv and shared-expert up projection
    # (8192 -> 1280), and fc1_latent (8192 -> 2048) -- the K = 8192 shapes where this
    # kernel beats cublas_mm at batch 1.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron_h BF16 decode {what}, T={m}: 8192 -> {n}",
            spec=dict(m=m, k=8192, n=n),
        )
        for m in (1, 2)
        for what, n in (
            ("mamba in_proj", 4384),
            ("qkv / shared up", 1280),
            ("fc1_latent", 2048),
        )
    )

    note = """
    Measured on a GB200 against cublas_mm, CUDA-graph replay, weights rotated
    over eight copies (us per call, one row): in_proj 11.2 vs 14.3, qkv 4.7 vs
    9.7, shared up 4.4 vs 10.1, fc1_latent 6.5 vs 8.7. At two rows the gap
    narrows (in_proj 13.1 vs 14.2, fc1_latent 7.6 vs 8.7); from four rows on
    cublas_mm is as fast or faster, and at eight the kernel is twice as slow --
    which is why the targets call it for one or two rows only. On the
    K = 1024 / 1280 / 2048 projections (o_proj, shared down, fc2_latent) it
    does not win at any row count.

    PDL: the caller says whether the kernel is launched as a programmatic
    dependent (`pdl`). The kernel waits for its predecessor before it reads
    anything, weights included, and releases its dependents after its stores.

    CUDA graphs: capturable. The kernel is compiled once per (shape, tactic,
    pdl) on first call, in-process, so warmup must run each captured row count
    before capture.

    Refused loudly by the kernel's own checks: M outside [1, 32], K not a
    multiple of the tactic's 8 * block_size, a row-major mat_b, a strided
    mat_a, a start that is not 32-byte aligned, any dtype but bf16.
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> torch.Tensor:
        m, k = mat_a.shape
        n = mat_b.shape[1]
        tactic = TACTICS.get((m, k, n)) or default_tactic(m, n, k)
        out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
        return run_direct_dense(mat_a, mat_b, out, pdl, tactic)

    def reference(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> torch.Tensor:
        """fp32-accumulated, TF32 off, rounded once to bf16; `pdl` changes scheduling only."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        return ref.to(mat_a.dtype)

    def is_valid(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> None:
        # Everything the kernel accepts it answers right; what it would answer wrong it
        # refuses itself (see the note), and restating those checks here would only
        # replace its error message with a worse one.
        return None


flashinfer_direct_bf16_gemm = FlashinferDirectBf16Gemm()
