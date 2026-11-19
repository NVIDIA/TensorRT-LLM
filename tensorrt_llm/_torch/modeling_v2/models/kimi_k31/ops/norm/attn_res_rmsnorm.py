# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual mix + RMSNorm through the family's PDL attn_res kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _attn_res_common as _c

if IS_CUTLASS_DSL_AVAILABLE:
    # The kernel module also holds the process-wide row limits the setters below change.
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import (
        attn_res as _k3_attn_res,
    )


def _require_cute_dsl(setter: str) -> None:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise RuntimeError(
            f"{setter} configures the CuTe DSL attn_res kernel, but the CuTe DSL (nvidia-cutlass-dsl) is not available"
        )


def set_early_trigger_max_rows(rows: int) -> None:
    """Release the dependent grid right after ``griddepcontrol.wait`` in every PDL attn_res mix of up to ``rows`` rows
    (process-wide; 0, the kernel's default, never). The release point changes no result; the kernel's
    ``_EARLY_TRIGGER_MAX_M`` states the launch order that makes it safe."""
    _require_cute_dsl("set_early_trigger_max_rows")
    _k3_attn_res._EARLY_TRIGGER_MAX_M = rows


def set_reg_cap_min_rows(rows: int) -> None:
    """Take the register-capped build of the PDL attn_res mixes with five or more candidates from ``rows`` rows on
    (process-wide; 0, the kernel's default, never; ``attn_res.reg_cap``). The cap changes the register allocation
    only, so the same results."""
    _require_cute_dsl("set_reg_cap_min_rows")
    _k3_attn_res.REG_CAP_MIN_TOKENS = rows


class K31AttnResRmsnorm(OpWrapper):
    """``out = rmsnorm(attn_res([block..., layer]), out_weight)`` per row, ``[M, 7168]`` BF16.

    Per token the candidates are the ``N - 1`` snapshot rows ``block[:, m]`` and ``layer[m]``;
    each scores ``rsqrt(mean(v^2) + rms_eps) * <v, res_weight * rms_weight>``, the softmax of the
    scores mixes them (rounded to BF16) and the trailing RMSNorm normalizes the mixture (BF16) and
    scales it by ``out_weight`` (BF16). ``trtllm::attn_res_rmsnorm_fwd`` computes the same; this
    kernel sums the statistics in another order (one BF16 rounding step apart in a small fraction
    of the elements) and loads the weights and snapshot rows before ``griddepcontrol.wait``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = _c.cells(add=False)

    note = """
    The TEP and DEP targets run this kernel for BF16 [M, 7168] rows with 2..9 candidates below
    ``_K31_MIX_PERSISTENT_ROWS`` (192) rows per rank (the DEP target also below 99 rows with 7+
    candidates; ``_k31_mix_persistent``); larger batches take the persistent mix, other operands
    the engine's attn_res ops. Hidden size 7168 and 2..12 candidates are compile-time properties of
    the kernel. Launched with programmatic dependent launch unless TRTLLM_ENABLE_PDL=0.
    """

    def __call__(
        self,
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> torch.Tensor:
        return _k3_attn_res.k31_attn_res_rmsnorm(
            layer, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )

    def reference(
        self,
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> torch.Tensor:
        return _c.reference(layer, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)

    def is_valid(
        self,
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        _c.check_operands(layer, block, res_weight, rms_weight, out_weight)


k31_attn_res_rmsnorm = K31AttnResRmsnorm()
