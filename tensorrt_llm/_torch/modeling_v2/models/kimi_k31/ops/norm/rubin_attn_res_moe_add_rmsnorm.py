# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin deferred MoE residual update + attention-residual mix + RMSNorm, one kernel.

A MoE layer leaves ``residual + bf16(routed + shared)`` to the next layer's pre-attention mix,
which computes it bit for bit (the ``_mx`` entry also stores the MXFP8 form of its output: the
catalog ``mxfp8_quantize``'s, which the dense MXFP8 GEMM of the KDA projection takes). The kernel
is ``kernels/norm/rubin_attn_res.py``; ``__call__`` calls its launcher directly.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_norm_common import (
    attn_res_cells,
    bf16_add,
    check_attn_res_inputs,
    mix_rmsnorm_reference,
    mxfp8_exact_reference,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res import k3_attn_res_moe_add_rmsnorm as _launch_moe_add
    from ...kernels.norm.rubin_attn_res import k3_attn_res_moe_add_rmsnorm_mx as _launch_moe_add_mx


def moe_residual_reference(
    residual: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
) -> torch.Tensor:
    """``bf16(residual + bf16(routed + shared))``, both adds rounded to nearest."""
    return bf16_add(residual, bf16_add(routed, shared, ftz=False), ftz=False)


def _mx_cells() -> tuple[Cell, ...]:
    # The TEP8 decode step's KDA input: the catalog quantizer's codes and R128c4 scales of one
    # 128-row tile for the dense MXFP8 QKVG GEMM (b1 / b2 / b4 k7: 8 / 16 / 32 rows).
    return tuple(
        Cell(
            why=f"{rows} rows, N={n}: catalog-exact MXFP8, R128c4 scales of one tile",
            spec=dict(rows=rows, candidates=n, early_trigger=True),
        )
        for rows, n in ((8, 2), (8, 9), (32, 5))
    )


class RubinAttnResMoeAddRmsnorm(OpWrapper):
    """``(prefix, output)``: ``prefix = bf16(residual + bf16(routed + shared))`` (the previous
    MoE layer's residual update) and ``rmsnorm(mix(block, prefix)) * out_weight``. ``residual``,
    ``routed``, ``shared`` are bf16 [M, 7168]; both outputs are new bf16 [M, 7168]."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = attn_res_cells(include_eps=False)
    note = """
    The update has two round-to-nearest adds, as the Triton MoE residual add has; the mix and
    norm are rubin_attn_res_rmsnorm's (same kernel, same rounding). ``early_trigger``: see
    rubin_attn_res_rmsnorm. Inputs are read-only.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch_moe_add(
            residual,
            routed,
            shared,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two bf16 adds, then the mix and the norm as the kernel rounds them."""
        prefix = moe_residual_reference(residual, routed, shared)
        output = mix_rmsnorm_reference(
            prefix, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        return prefix, output

    def is_valid(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> None:
        check_attn_res_inputs(residual, (routed, shared), block, res_weight, rms_weight, out_weight)


class RubinAttnResMoeAddRmsnormMx(OpWrapper):
    """:class:`RubinAttnResMoeAddRmsnorm` plus the MXFP8 form ``(xq, xs)`` of its output:
    returns ``(prefix, output, xq, xs)``, xq float8_e4m3fn [M, 7168]; xs uint8 [56 * 512], the
    swizzled R128c4 scales of one 128-row tile (M <= 128)."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = _mx_cells()
    note = """
    The codes and scales are the catalog ``mxfp8_quantize``'s (no amax floor) of ``output``, bit
    for bit: ``(xq, xs)`` are the operands ``rubin_mxfp8_linear`` would make of ``output``. M <= 128
    (the launcher raises otherwise); scale rows past M are left unwritten.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_moe_add_mx(
            residual,
            routed,
            shared,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The non-``_mx`` reference plus :func:`mxfp8_exact_reference` of its output in one
        128-row scale block. The trigger does not change results."""
        prefix = moe_residual_reference(residual, routed, shared)
        output = mix_rmsnorm_reference(
            prefix, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        return (prefix, output, *mxfp8_exact_reference(output))

    def is_valid(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> None:
        check_attn_res_inputs(residual, (routed, shared), block, res_weight, rms_weight, out_weight)


rubin_attn_res_moe_add_rmsnorm = RubinAttnResMoeAddRmsnorm()
rubin_attn_res_moe_add_rmsnorm_mx = RubinAttnResMoeAddRmsnormMx()
