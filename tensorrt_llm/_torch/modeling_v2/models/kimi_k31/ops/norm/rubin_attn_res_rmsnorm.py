# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin attention-residual mix + RMSNorm (and the residual add), one PDL kernel.

The kernel and its launchers are in ``kernels/norm/rubin_attn_res.py``. ``__call__`` calls the
launcher directly; nothing is registered as a torch op. What the ``rubin_attn_res_*`` entries
share (shapes, cells, references, input guard) is in ``_rubin_norm_common.py``.
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
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res import k3_attn_res_add_rmsnorm as _launch_add_rmsnorm
    from ...kernels.norm.rubin_attn_res import k3_attn_res_rmsnorm as _launch_rmsnorm

_NOTE = """
    The vectorized PDL kernel: a cluster of 4 or 8 CTAs per row; weights, snapshot rows and
    their statistics load before griddepcontrol.wait. It sums the statistics in another order
    than the C++ attn_res_rmsnorm_fwd, so the output can differ from it by one bf16 step in a
    small fraction of the elements; the updated prefix is a plain bf16 add, bit for bit.
    ``early_trigger`` (a compile-time constant) releases the dependent grid right after the
    wait instead of after the stores; the results are the same bits. The caller must order
    the next attn_res launch after this grid (an MNNVL all-reduce does). Shapes the kernel
    serves: H = 7168, 1..1024 rows, 2..12 candidates; the K3.1 Rubin targets call it up to
    ``PDL_MAX_M`` rows and use the C++ ops above that. The first call per shape
    compiles, so it runs outside CUDA graph capture.
    """


class RubinAttnResRmsnorm(OpWrapper):
    """``rmsnorm(mix(block, layer)) * out_weight`` for ``layer`` [M, 7168] and the K snapshot
    rows ``block`` [K, M, 7168] (bf16). Returns a new bf16 [M, 7168] tensor."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = attn_res_cells()
    note = _NOTE

    def __call__(
        self,
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> torch.Tensor:
        return _launch_rmsnorm(
            layer,
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
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> torch.Tensor:
        """fp32, rounding where the kernel rounds. ``early_trigger`` does not change results."""
        return mix_rmsnorm_reference(
            layer, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )

    def is_valid(
        self,
        layer: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> None:
        check_attn_res_inputs(layer, (), block, res_weight, rms_weight, out_weight)


class RubinAttnResAddRmsnorm(OpWrapper):
    """``(updated, output)``: ``updated = bf16(prefix + addend)`` and the mix + RMSNorm of
    :class:`RubinAttnResRmsnorm` over ``block`` and ``updated``. Both new bf16 [M, 7168]."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = attn_res_cells()
    note = _NOTE

    def __call__(
        self,
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch_add_rmsnorm(
            prefix,
            addend,
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
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fast-math bf16 add, then the mix and the norm as the kernel rounds them."""
        updated = bf16_add(prefix, addend, ftz=True)
        output = mix_rmsnorm_reference(
            updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        return updated, output

    def is_valid(
        self,
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
    ) -> None:
        check_attn_res_inputs(prefix, (addend,), block, res_weight, rms_weight, out_weight)


rubin_attn_res_rmsnorm = RubinAttnResRmsnorm()
rubin_attn_res_add_rmsnorm = RubinAttnResAddRmsnorm()
