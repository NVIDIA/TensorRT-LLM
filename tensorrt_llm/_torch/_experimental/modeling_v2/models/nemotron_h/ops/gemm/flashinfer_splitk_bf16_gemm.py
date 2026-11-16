# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 matmul for 3 to 32 rows, through FlashInfer's SM100 split-K tensor-core kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (  # noqa: E501
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: The tactic each tuned shape runs, `(mma_m, mma_n, split_k, ab_stages)`, keyed by `(M, K, N)`:
#: the AutoTuner's pick for Modeling V1's low-M race (TRTLLM_LOW_M_GEMM_BACKEND=auto) on a GB200 at
#: the dep16 decode shapes, replaced where a sweep of the kernel's own candidates in CUDA-graph
#: replay found one at least 3% faster (mamba out_proj at 24 and 32 rows, fc1_latent). Fixed
#: rather than autotuned so every rank of a group runs the same kernel; a row count without an
#: entry takes the next larger tuned one's tactic, and a shape without any (fc2_latent) the
#: kernel's default.
TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # mamba in_proj 8192 -> 35072
    (4, 8192, 35072): (64, 8, 2, 12),
    (8, 8192, 35072): (128, 16, 1, 5),
    (16, 8192, 35072): (128, 32, 1, 5),
    (32, 8192, 35072): (128, 32, 1, 5),
    # mamba out_proj 16384 -> 8192
    (4, 16384, 8192): (128, 8, 4, 6),
    (8, 16384, 8192): (128, 8, 2, 5),
    (16, 16384, 8192): (128, 16, 2, 5),
    (24, 16384, 8192): (128, 32, 2, 5),
    (32, 16384, 8192): (128, 32, 2, 5),
    # shared-expert up 8192 -> 10240
    (4, 8192, 10240): (128, 16, 1, 6),
    (8, 8192, 10240): (128, 8, 1, 6),
    (16, 8192, 10240): (128, 16, 1, 6),
    (32, 8192, 10240): (128, 32, 1, 5),
    # attention o_proj 8192 -> 8192
    (4, 8192, 8192): (128, 8, 2, 6),
    (8, 8192, 8192): (64, 16, 1, 9),
    (16, 8192, 8192): (64, 16, 1, 9),
    (32, 8192, 8192): (64, 32, 1, 8),
    # fc1_latent 8192 -> 2048
    (4, 8192, 2048): (64, 8, 4, 10),
    (8, 8192, 2048): (64, 8, 4, 10),
    (16, 8192, 2048): (64, 16, 4, 10),
}
_ROWS = (4, 8, 16, 24, 32)
#: The dep16 decode projections the target may route here, `(name, K, N)`; the cells run each at
#: every measured row count, whether or not TACTICS names a tactic for it.
SHAPES = (
    ("mamba in_proj", 8192, 35072),
    ("mamba out_proj", 16384, 8192),
    ("attention o_proj", 8192, 8192),
    ("shared-expert up", 8192, 10240),
    ("fc1_latent", 8192, 2048),
    ("fc2_latent", 2048, 8192),
)


def _tactic(m: int, k: int, n: int):
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic, default_tactic

    for r in _ROWS:
        if r >= m and (r, k, n) in TACTICS:
            return SplitKTactic(*TACTICS[(r, k, n)])
    return default_tactic(m, n, k)


class FlashinferSplitkBf16Gemm(OpWrapper):
    """`mat_a @ mat_b` in bf16, fp32-accumulated, a new `[M, N]` out.

    FlashInfer's SM100 CuTe DSL dense kernel: tcgen05 MMA with the rows on the narrow MMA side,
    K optionally split over CTAs and reduced in a fixed order. `mat_b` is read column-major, which
    is what a stored weight's `.t()` already is, as for `cublas_mm`.

    Fusion boundary: the product only. No bias, no out_dtype other than bf16.
    """

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"nemotron_h BF16 dep16 decode {what}, T={m}: {k} -> {n}",
            spec=dict(m=m, k=k, n=n),
        )
        for what, k, n in SHAPES
        for m in _ROWS
    )

    note = """
    Measured on a GB200 against cublas_mm, CUDA-graph replay, weights rotated so no call reads
    them from L2 (us per call at 4 / 8 / 16 / 32 rows): mamba in_proj 79.4 / 80.5 / 81.3 / 81.7
    vs 88.1 / 88.1 / 88.9 / 89.4; mamba out_proj 38.2 / 38.9 / 39.3 / 42.2 vs 46.1 / 46.7 / 48.1 /
    49.3; shared-expert up 26.3 / 26.3 / 26.7 / 28.1 vs 27.6 / 27.9 / 28.6 / 31.5; fc1_latent
    6.1 / 6.1 / 7.4 / 7.0 vs 8.4 / 8.6 / 8.2 / 9.6; fc2_latent 5.7 / 5.7 / 5.7 / 6.1 vs 6.7 / 6.4 /
    6.5 / 7.0. Attention o_proj, qkv and the shared-expert down projection run as fast on
    cublas_mm (within 2%), so the target leaves them there by default. The dep16 target calls it
    for decode steps of up to TRTLLM_NEMOTRON_ULTRA_DEP16_SPLITK_MAX_ROWS rows (off by default).

    PDL: the caller says whether the kernel is launched as a programmatic dependent (`pdl`).

    CUDA graphs: capturable. The kernel is compiled once per (shape, tactic, pdl) on first call,
    in-process, so warmup must run each captured row count before capture.
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> torch.Tensor:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense

        m, k = mat_a.shape
        n = mat_b.shape[1]
        out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
        return run_splitk_dense(mat_a, mat_b, None, out, pdl, _tactic(m, k, n))

    def reference(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> torch.Tensor:
        """fp32-accumulated, TF32 off, rounded once to bf16; `pdl` changes scheduling only."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        return ref.to(mat_a.dtype)

    def is_valid(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> None:
        return None


flashinfer_splitk_bf16_gemm = FlashinferSplitkBf16Gemm()
