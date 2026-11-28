# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 end of a MoE layer: ``prefix + (routed + shared)`` in one kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_residual_add import (
    k31_moe_residual_add as _k31_moe_residual_add,
)

_HIDDEN = 7168


def _add(rows: int, cols: int, why: str, data: str = "randn") -> Cell:
    return Cell(why=why, spec=dict(rows=rows, cols=cols, data=data))


class K31MoeResidualAdd(OpWrapper):
    """``prefix + (routed + shared)`` for BF16 ``[M, N]``, bit-identical to the two PyTorch BF16 adds the layer ran
    (``routed_out + shared_out``, then ``prefix_sum + hidden_states``): each element rounds the inner sum to BF16 and
    the outer sum to BF16 again (FP32 arithmetic, round to nearest even, denormals kept).

    ``prefix`` is the running residual (``prefix_sum``) at the end of a Kimi K3.1 MoE layer, ``routed`` the routed
    experts' output after the latent up projection and ``shared`` the shared experts' output, all ``[tokens,
    7168]``. Returns a new tensor. One CuTe DSL kernel reading each input once.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _add(1, _HIDDEN, "one decode row"),
        _add(8, _HIDDEN, "TEP32 verify step at B1 (8 rows)"),
        _add(72, _HIDDEN, "TEP16 verify step at B12 (72 rows)"),
        _add(99, _HIDDEN, "attention-DP verify step, ~14 requests x 7 rows"),
        _add(2048, _HIDDEN, "TEP8 sequence-parallel prefill: one rank's rows of a 16k step"),
        _add(8192, _HIDDEN, "DEP16 ctx step: 8192 tokens per rank (more than one wave)"),
        _add(37, _HIDDEN, "odd row count (a partial CTA)"),
        _add(3, 100, "row length off the 8-element vector (one element per step)"),
        _add(
            256,
            _HIDDEN,
            "values near overflow and denormals: rounding of both sums visible",
            data="edge",
        ),
    )

    note = """
    Bitwise to the two PyTorch BF16 adds on finite inputs (a NaN stays a NaN, payload aside). The three inputs must
    be contiguous 2D BF16 tensors of one shape (the op raises otherwise). The kernel moves 16-byte vectors when the
    four base pointers are 16-byte aligned and the element count is a multiple of 8, single elements otherwise; it is
    bitwise the same either way. Programmatic dependent launch follows TRTLLM_ENABLE_PDL (default on): the kernel
    waits for its producer before its first load and, at up to one wave of CTAs, releases its dependent once its
    loads are issued. The first call per variant compiles, so it must run outside CUDA-graph capture.
    """

    def __call__(
        self, prefix: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
    ) -> torch.Tensor:
        return _k31_moe_residual_add(prefix, routed, shared)

    def reference(
        self, prefix: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
    ) -> torch.Tensor:
        """The two PyTorch BF16 adds the op replaces."""
        return prefix + (routed + shared)

    def is_valid(self, prefix: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor) -> None:
        """Nothing to add: the op raises on every layout it would read wrongly."""
        return None


k31_moe_residual_add = K31MoeResidualAdd()
