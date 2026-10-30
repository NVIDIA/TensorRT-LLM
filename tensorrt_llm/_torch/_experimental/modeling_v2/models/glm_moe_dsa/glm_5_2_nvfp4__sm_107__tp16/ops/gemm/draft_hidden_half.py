# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's FP32 share of the next MTP draft step's eh_proj hidden half, hnorm included, as
one split-K GEMM."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._low_m_splitk import (
    device_limits,
    operands_overlap,
    run_splitk_hidden_half,
    splitk_default_tactic,
)

#: GLM-5.2 at tp16: hidden 6144, this rank's 6144 / 16 = 384 output rows of eh_proj's hidden
#: half, hnorm eps 1e-5.
_GLM = dict(k=6144, n=384, eps=1e-5)


class GlmDraftHiddenHalf(OpWrapper):
    """`out = rmsnorm(x) @ weight.T` (with `norm_weight`) or `out = x @ weight.T` (without), FP32
    out; returns `out`.

    eh_proj is row-parallel over `cat(enorm(embed(token)), hnorm(hidden))`, so its output is a
    token half (tabulated at load) plus `hnorm(hidden) @ W_e[:, H:].T`. This op computes this
    rank's `H / tp` outputs of that hidden half in FP32, on the draft step's hidden state before
    the draft LM head runs. `weight` is this rank's rows `[rank * H / tp, (rank + 1) * H / tp)`
    of `W_e[:, H:]` (gathered from the row-parallel shards once, at load).

    With `norm_weight` (one row of H = 6144): the kernel's B producer computes FlashInfer's
    RMSNorm of `x` itself -- the 64-thread layout, the sequential square sums, the two
    butterflies and their sum, `rsqrt.approx.ftz` of the IEEE `sum / H + eps`, `(x * rstd) *
    (w + 0)` rounded to BF16 -- so the result is bit-identical to running that RMSNorm first and
    the GEMM on its output. Without it `x` is the normalized row(s).
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP draft steps 0-5 (one request): 1 row, hnorm in the kernel, split-K 8 "
            "(48 CTAs, every K tile in its own stage)",
            spec=dict(_GLM, rows=1, in_kernel_norm=True),
        ),
        Cell(
            why="two requests per draft step: 2 normalized rows, the plain FP32-output GEMM",
            spec=dict(_GLM, rows=2, in_kernel_norm=False),
        ),
    )

    note = """
    The tactic is the shape's `splitk_default_tactic`; the in-kernel norm needs one row, H =
    6144 and a tactic whose CTA K slice fits its stages (`hnorm_b_supported`), else ValueError.
    PDL follows `TRTLLM_ENABLE_PDL` (default on): before its grid dependency wait the kernel
    reads only the weight (TMA) and the norm weight and zeroes its own B stages; it reads `x`
    after the wait.

    Loud, so not guarded (ValueError before the launch): M outside [1, 32], K not a multiple of
    128, non-BF16 x or weight, a non-FP32 or non-contiguous out, pointers not 32-byte aligned,
    the in-kernel norm where it does not serve. The first call of a (tactic, norm eps) compiles
    the kernel; make it before CUDA-graph capture.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> torch.Tensor:
        """Write `out` and return it.

        Args:
            x: [rows, H] BF16 with unit inner stride: the raw hidden state with `norm_weight`,
                else the normalized one.
            weight: [N, H] BF16, contiguous: this rank's hidden-half rows of eh_proj.
            out: [rows, N] FP32, contiguous.
            norm_weight: [H] BF16 hnorm weight for the in-kernel norm, or None.
            eps: hnorm's epsilon (with `norm_weight`), else None.
        """
        rows, k = x.shape
        sm_count, smem_capacity = device_limits(x.device.index)
        tactic = splitk_default_tactic(
            rows, weight.shape[0], k, sm_count=sm_count, smem_capacity=smem_capacity
        )
        return run_splitk_hidden_half(
            x, weight, out, get_env_enable_pdl(), tactic, norm_weight=norm_weight, norm_eps=eps
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> torch.Tensor:
        """FP32 product (TF32 off) of the BF16 normalized row(s) and the weight. The norm is
        FlashInfer's formula in FP32 -- `rstd = rsqrt(sum(x ** 2) / H + eps)`, `(x * rstd) * w`
        rounded to BF16 once -- with torch's summation order. `out` is not read."""
        xf = x.float()
        if norm_weight is not None:
            rstd = torch.rsqrt(xf.pow(2).sum(dim=-1, keepdim=True) / xf.shape[-1] + eps)
            xf = ((xf * rstd) * norm_weight.float()).to(torch.bfloat16).float()
        with true_fp32_matmul():
            return xf @ weight.float().t()

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> None:
        # `eps` without `norm_weight` is ignored: the row goes into the GEMM unnormalized.
        assert (norm_weight is None) == (eps is None), (
            "pass norm_weight and eps together: eps alone leaves x unnormalized"
        )
        # The owner CTA of a tile stores while other clusters still read x and the weight.
        for operand, name in ((x, "x"), (weight, "weight")):
            assert not operands_overlap(out, operand), f"out must not share memory with {name}"


draft_hidden_half = GlmDraftHiddenHalf()
