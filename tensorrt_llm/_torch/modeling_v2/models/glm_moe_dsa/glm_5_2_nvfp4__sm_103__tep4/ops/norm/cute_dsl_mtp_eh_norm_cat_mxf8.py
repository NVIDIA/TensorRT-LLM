# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP module's two input RMSNorms written as one MXF8 operand: `concat(enorm(e),
hnorm(h))` quantized for the MXF8 eh_proj GEMM, no bf16 concat."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _quant
from ...kernels.norm import mtp_eh_norm_cat_mxf8 as _kernel

HIDDEN = _kernel.HIDDEN
#: The operand width: the two normalized halves side by side.
K_TOTAL = _kernel.K_TOTAL
CompiledMtpEhNormCat = _kernel.CompiledKernel
supports = _kernel.supports


def compile(device_index: int) -> CompiledMtpEhNormCat:  # noqa: A001 -- the kernel's own name
    """Compile the kernel for `device_index` and launch it once on one row, so no context step
    meets a first module load. Row count, pitches and eps are runtime values."""
    fn = _kernel.compile_norm_cat_q8(int(device_index))
    dev = torch.device("cuda", int(device_index))
    with torch.cuda.device(dev):
        x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)
        _kernel.norm_cat_q8(fn, x, w, x, w, 1e-5)
        torch.cuda.synchronize(dev)
    return fn


def operand(rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffers of a `[rows, 12288]` MXF8 operand: e4m3 bytes and the blocked e8m0 scales."""
    return (
        torch.empty(rows, K_TOTAL, dtype=torch.uint8, device=device),
        torch.empty(_quant.sf_bytes(rows, K_TOTAL), dtype=torch.uint8, device=device),
    )


class CuteDslMtpEhNormCatMxf8(OpWrapper):
    """Write `concat(rmsnorm(e, e_weight), rmsnorm(h, h_weight))` as the MXF8 operand `(q
    [T, 12288] uint8 e4m3, sf)` and return it.

    `e` and `h` are bf16 `[T, 6144]` (same T, unit inner stride, a row pitch that is a
    multiple of 8, 16-byte aligned); the weights bf16 `[6144]`. Each half is the hidden-width
    RMSNorm's bf16 output (`cute_dsl_hidden_rmsnorm`'s bytes) quantized by the activation
    quant's rule: per 32-element block a power-of-two scale and e4m3 values; the scale rows
    past T up to the next multiple of 128 get 2^0.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The MTP module's context rows that take the replicated eh_proj (above the TP4 k-split
    # cells): a partial last scale block, a served context step, the token budget.
    CELLS: tuple[Cell, ...] = (
        Cell(why="65 rows, the first context count", spec=dict(rows=65)),
        Cell(why="7310 rows, a served context step", spec=dict(rows=7310)),
        Cell(why="max_num_tokens", spec=dict(rows=8192)),
    )

    note = """
    The target takes this op on the MTP module's context rows that the TP4 k-split front end
    does not take, ahead of the MXF8 eh_proj (`cute_dsl_projection_mxf8` `mtp_eh`): no bf16
    norms, no concat copies and no activation quant (about 1.06 -> 0.54 ms per context step at
    7-8k rows on B300). The kernel checks operands on every call and raises.
    """

    def __call__(
        self,
        e: torch.Tensor,
        e_weight: torch.Tensor,
        h: torch.Tensor,
        h_weight: torch.Tensor,
        eps: float,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledMtpEhNormCat,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.norm_cat_q8(compiled, e, e_weight, h, h_weight, eps, q, sf)

    def reference(
        self,
        e: torch.Tensor,
        e_weight: torch.Tensor,
        h: torch.Tensor,
        h_weight: torch.Tensor,
        eps: float,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledMtpEhNormCat,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(q, sf)` as new tensors: fp32 RMSNorm of each half rounded to bf16, the concat,
        the activation quant's torch twin. The kernel's sum of squares runs in another order,
        so a bf16 value may differ by one ulp. `q`, `sf` and `compiled` are not used."""
        return _kernel.reference(e, e_weight, h, h_weight, eps)

    def is_valid(
        self,
        e: torch.Tensor,
        e_weight: torch.Tensor,
        h: torch.Tensor,
        h_weight: torch.Tensor,
        eps: float,
        q: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledMtpEhNormCat,
    ) -> None:
        assert supports(e, h), (
            f"e and h must be bf16 [T, {HIDDEN}] row blocks, got {tuple(e.shape)} / "
            f"{tuple(h.shape)}"
        )
        rows = int(e.shape[0])
        assert tuple(q.shape) == (rows, K_TOTAL) and q.dtype == torch.uint8, (
            f"q must be uint8 [{rows}, {K_TOTAL}]"
        )
        assert sf.numel() >= _quant.sf_bytes(rows, K_TOTAL), "sf is too short"


cute_dsl_mtp_eh_norm_cat_mxf8 = CuteDslMtpEhNormCatMxf8()
