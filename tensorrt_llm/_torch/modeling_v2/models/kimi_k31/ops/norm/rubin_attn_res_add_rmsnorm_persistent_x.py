# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual mix + RMSNorm on the persistent grid, with the epilogues that replace the
kernels around it on a generation step. The kernel is ``kernels/norm/rubin_attn_res_persistent_x.py`` (CuTe DSL);
``__call__`` calls its launcher directly."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_attn_res_add_rmsnorm_persistent import rubin_attn_res_add_rmsnorm_persistent

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res_persistent_x import (
        attn_res_add_rmsnorm_persistent_x as _launch,
    )
    from ...kernels.norm.rubin_attn_res_persistent_x import supported as _kernel_supported

_HIDDEN = 7168
_EPS = 1e-5


class AttnResX(NamedTuple):
    """The op's results: the updated prefix (``prefix`` itself without an addend) and the normed rows, then the
    epilogues that were asked for (None otherwise)."""

    updated: torch.Tensor  # [rows, 7168] bf16
    output: torch.Tensor  # [rows, 7168] bf16
    output_f32: Optional[torch.Tensor]  # [rows, 7168] fp32: output.float()
    codes: Optional[torch.Tensor]  # [rows, 7168] float8_e4m3fn: output's MXFP8 codes
    scales: Optional[torch.Tensor]  # [pad128(rows) * 224] uint8: their 128x4 swizzled UE8M0 scales


def persistent_x_available() -> bool:
    """Whether the CuTe DSL kernel runs on the current device (an SM 10.x GPU with the CuTe DSL); elsewhere the
    K3.1 targets keep today's calls."""
    return IS_CUTLASS_DSL_AVAILABLE and _kernel_supported()


def _cells() -> tuple[Cell, ...]:
    """The targets' generation steps above the PDL kernel's rows: DEP16's 96 and helix8's 144 rows,
    N = 2..9, each with the flag sets its target calls (helix8 has no FP32 router-input copy)."""
    cells = []
    for step, rows, pdl_rows, f32 in (
        ("DEP16 GEN b12 k7", 96, 64, True),
        ("helix8 GEN b24 k5", 144, 128, False),
    ):
        where = f"{step} ({rows} rows, above the PDL kernel's {pdl_rows})"
        for n in range(2, 10):
            for form, kw in (
                ("plain, grid cap", dict(grid_cap=True)),
                ("addend, grid cap", dict(addend=True, grid_cap=True)),
                ("MoE residual fold", dict(addend=True, addend2=True)),
                ("FP32 copy", dict(f32=True)),
                ("addend + FP32 copy", dict(addend=True, f32=True)),
                ("MoE residual fold + MXFP8 form", dict(addend=True, addend2=True, mx=True)),
            ):
                if kw.get("f32") and not f32:
                    continue
                cells.append(
                    Cell(
                        why=f"{where}, N={n}, {form}",
                        spec=dict(tokens=rows, candidates=n, hidden=_HIDDEN, eps=_EPS, **kw),
                    )
                )
    return tuple(cells)


class RubinAttnResAddRmsnormPersistentX(OpWrapper):
    """``rubin_attn_res_add_rmsnorm_persistent``'s contract with epilogues, on ``prefix`` [rows, 7168], the bank
    ``block`` [K, rows, 7168] (2 <= K + 1 <= 9) and the mix's three weights, all bf16:

    - ``addend``: ``updated = bf16(prefix + addend)``; with ``addend2`` too, ``updated = bf16(prefix +
      bf16(addend + addend2))``, the MoE residual update ``residual + bf16(routed + shared)``;
    - ``f32``: ``output_f32 = output.float()``;
    - ``mx`` (with ``addend`` and ``addend2``): ``(codes, scales)`` = ``mxfp8_quantize(output, True, 32)``;
    - ``grid_cap``: min(SMs, rows) CTAs.

    Returns :class:`AttnResX`. Flag sets the kernel is built for: none, addend, addend + addend2, f32 with or
    without addend, addend + addend2 + mx.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = _cells()
    note = """
    kernels/norm/rubin_attn_res_persistent_x.py: the CuTe DSL form of the engine's former
    attn_res_add_rmsnorm_persistent_x_fwd (attnResFwdPersistentX.cu), byte for byte: the same schedule (one CTA per
    SM, or min(SMs, rows) with grid_cap, a TMA ring of candidate rows, TMEM for the values) and the instructions
    that kernel's --use_fast_math build executes. Every output is bit for bit what
    rubin_attn_res_add_rmsnorm_persistent and the kernel each epilogue replaces make of the same inputs:
    rubin_moe_residual_add before it, rubin_upcast_fp32 or mxfp8_quantize after it. It compiles once per N and
    flag set on the first call (outside CUDA graph capture). It needs an SM 10.x GPU: elsewhere
    ``persistent_x_available()`` is False and the K3.1 targets keep today's calls.
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        addend2: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        f32: bool = False,
        mx: bool = False,
        grid_cap: bool = False,
    ) -> AttnResX:
        return AttnResX(
            *_launch(
                prefix,
                addend,
                addend2,
                block,
                res_weight,
                rms_weight,
                out_weight,
                rms_eps,
                out_eps,
                f32,
                mx,
                grid_cap,
            )
        )

    def reference(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        addend2: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        f32: bool = False,
        mx: bool = False,
        grid_cap: bool = False,
    ) -> AttnResX:
        """Today's calls: the bf16 add(s) as torch rounds them, the persistent mix's reference, ``.float()`` and
        the catalog quantizer's reference. ``grid_cap`` places CTAs only."""
        rows, hidden = prefix.shape
        k = block.shape[0]
        updated = prefix
        if addend is not None:
            add = addend
            if addend2 is not None:
                add = (addend.float() + addend2.float()).to(torch.bfloat16)
            updated = (prefix.float() + add.float()).to(torch.bfloat16)
        _, output = rubin_attn_res_add_rmsnorm_persistent.reference(
            updated.view(rows, 1, hidden),
            None,
            block.view(k, rows, 1, hidden),
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
        )
        output = output.view(rows, hidden)
        codes = scales = None
        if mx:
            from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import (
                mxfp8_quantize,
            )

            codes, scales = mxfp8_quantize.reference(output, True, 32)
        return AttnResX(updated, output, output.float() if f32 else None, codes, scales)

    def is_valid(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        addend2: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        f32: bool = False,
        mx: bool = False,
        grid_cap: bool = False,
    ) -> None:
        assert prefix.dim() == 2 and prefix.shape[1] == _HIDDEN and prefix.dtype == torch.bfloat16
        assert block.dim() == 3 and block.shape[1:] == prefix.shape and 2 <= block.shape[0] + 1 <= 9
        assert addend2 is None or addend is not None, "a second addend needs the first"
        assert not mx or (addend is not None and addend2 is not None and not f32), (
            "the MXFP8 form is built with the MoE residual fold only"
        )
        assert not (f32 and addend2 is not None), "the FP32 copy is built without the second addend"


rubin_attn_res_add_rmsnorm_persistent_x = RubinAttnResAddRmsnormPersistentX()
