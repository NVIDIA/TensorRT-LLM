# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE back on SM107: MegaMoE's top-k combine reduce and the latent RMSNorm in one
kernel, from the bucket's combine plane."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe.rubin_moe_back_norm import check_operands
from ...kernels.moe.rubin_moe_back_norm import run_moe_back_norm as _launch


def _cell(why: str, rows: int, bucket: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, bucket=bucket, **spec))


class RubinMoeBackNorm(OpWrapper):
    """``out[r] = rmsnorm(sum_k scores[r, k] * plane[r, k], weight, eps)`` for ``r < rows``.

    ``plane`` BF16 ``[T, top_k, hidden]`` (each row's ``[top_k, hidden]`` dense) is MegaMoE's
    combine plane, every routed slot's FC2 output of a row; ``scores`` FP32 ``[T, top_k]`` its
    routing weights; ``weight`` BF16 ``[hidden]`` the latent norm's weight; ``out`` BF16
    ``[rows, hidden]`` dense, ``rows <= T``. One launch on the current stream; returns ``out``.

    The bits are those of the two kernels it replaces: the engine's ``TopkReduce`` over the plane
    (the BF16 combine format with FP32 scores: ``v[0] * s[0]``, then an FMA per slot in slot
    order, one rounding to BF16), then FlashInfer's RMSNorm of the BF16 row (its thread layout,
    fragment sum, butterfly and two-warp combine, ``rsqrt.approx``, ``x * rstd * (w + 0)``).

    ``pdl``: launch programmatically; each CTA loads its norm weights, waits for the grid before
    it, then releases its own dependents.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("DEP16 generation b12 k7: 96 rows of bucket 128", 96, 128),
        _cell("DEP16 generation b12 k7, heavy-tailed expert outputs", 96, 128, mode="heavy"),
        _cell("DEP16 generation b24 k7: 192 rows of bucket 256", 192, 256),
        _cell(
            "DEP16 generation b6 k7: 48 rows of bucket 64, launched without PDL", 48, 64, pdl=False
        ),
        _cell("one row of the smallest bucket", 1, 32),
        _cell("a full bucket: 128 rows", 128, 128),
        _cell("generation with context rows: 300 rows of bucket 384", 300, 384),
        _cell("more rows than twice the SMs: 512 rows, two rows per CTA", 512, 512),
        _cell("zero slots: rows whose routing weights are all 0", 64, 64, mode="zero_scores"),
    )

    note = """
    ``hidden`` must be a width FlashInfer's RMSNorm sums with 64 threads per row, two rows per
    block, in 16-byte vectors that tile the row exactly: a multiple of 512 in (3072, 6144], such as
    K3.1's latent 3584 (two rows per CTA only up to 4096); the kernel class raises otherwise. Rows
    past ``out.shape[0]`` are not read. A row whose slots all carry weight 0 normalizes to zeros.
    """

    def __call__(
        self,
        plane: torch.Tensor,
        scores: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> torch.Tensor:
        return _launch(plane, scores, weight, eps, out, pdl=pdl)

    def reference(
        self,
        plane: torch.Tensor,
        scores: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> torch.Tensor:
        """FP32 torch: the slot sum in slot order (rounded to BF16), then the RMSNorm in FP32
        (rounded to BF16). Not bit-exact: torch's FMA contraction and sum order differ."""
        rows = out.shape[0]
        p = plane[:rows].float()
        s = scores[:rows].float()
        acc = p[:, 0] * s[:, :1]
        for k in range(1, plane.shape[1]):
            acc = torch.addcmul(acc, p[:, k], s[:, k : k + 1])
        x = acc.to(torch.bfloat16).float()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x * rstd * weight.float()).to(torch.bfloat16)

    def is_valid(
        self,
        plane: torch.Tensor,
        scores: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> None:
        """The kernel reads raw rows: a misshaped, mistyped or strided operand is read wrong."""
        try:
            check_operands(plane, scores, weight, out)
        except ValueError as e:
            raise AssertionError(str(e)) from e
        assert isinstance(pdl, bool), "pdl is a launch switch"


rubin_moe_back_norm = RubinMoeBackNorm()
