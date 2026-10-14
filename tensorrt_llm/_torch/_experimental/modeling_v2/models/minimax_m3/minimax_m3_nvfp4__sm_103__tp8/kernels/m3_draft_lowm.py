# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter's BF16 linears on FlashInfer's fixed-tactic low-M kernels (``dispatch.DRAFT_LOWM``).

``install`` swaps the linear method of every bias-free BF16 ``Linear`` of the drafter (qkv, o_proj, gate_up, down, the
fc; not the head) for ``DraftLowMLinearMethod``: a covered row count runs ``ops/gemm/m3_draft_lowm`` at the tactic
of its (N, K) table, every other call the in-tree method. ``prebuild`` compiles every covered call before the
executor's warmup; a forward never compiles one, and a covered call it finds uncompiled is a miss.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod

from .. import dispatch
from ..ops.gemm.m3_draft_lowm import m3_draft_lowm, tactic_for

# (N, K, M, PDL) of every call prebuild compiled.
_BUILT: Set[Tuple[int, int, int, bool]] = set()


class DraftLowMLinearMethod(UnquantizedLinearMethod):
    """The in-tree BF16 linear method with the drafter's covered rows on the fixed-tactic low-M kernels."""

    def apply(self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]):
        out = draft_linear(module.weight, input, bias)
        if out is not None:
            return out
        return super().apply(module, input, bias)


def draft_linear(
    weight: torch.Tensor, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """x [M, K] @ weight^T through the low-M kernel when ``dispatch.DRAFT_LOWM`` covers it; None for the in-tree
    GEMM."""
    if bias is not None or x.dim() != 2:
        return None
    n, k = weight.shape
    m = int(x.shape[0])
    if not dispatch.DRAFT_LOWM.covers(((n, k), m)):
        dispatch.uncovered(
            dispatch.DRAFT_LOWM.path, f"the drafter's {n}x{k} linear past its tactic table's rows"
        )
        return None
    if not x.is_contiguous():
        # The first pass's fc may read a column window of the captured hidden states; the in-tree GEMM takes strides.
        dispatch.uncovered(
            dispatch.DRAFT_LOWM.path, "a strided activation (the in-tree GEMM reads it in place)"
        )
        return None
    use_pdl = get_env_enable_pdl()
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        dispatch.missed(dispatch.DRAFT_LOWM.path, m, "the activation or weight is not BF16")
        return None
    if (n, k, m, use_pdl) not in _BUILT:
        dispatch.missed(
            dispatch.DRAFT_LOWM.path,
            m,
            f"the {n}x{k} kernel at {m} rows was not compiled before warmup",
        )
        return None
    kind, tactic = tactic_for(n, k, m)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    return m3_draft_lowm(x, weight, out, kind, tactic, use_pdl)


def _drafter_linears(draft_model: nn.Module):
    for module in draft_model.modules():
        if (
            isinstance(module, Linear)
            and not isinstance(module, LMHead)
            and type(module.quant_method) is UnquantizedLinearMethod
            and getattr(module, "bias", None) is None
            and module.weight.dtype == torch.bfloat16
        ):
            yield module


def install(draft_model: nn.Module) -> int:
    """Put every covered drafter linear on ``DraftLowMLinearMethod``; returns how many were installed."""
    count = 0
    for module in _drafter_linears(draft_model):
        n, k = module.weight.shape
        if any(
            dispatch.DRAFT_LOWM.covers(((n, k), m))
            for m in range(1, dispatch.DECODE_MAX_TOKENS + 1)
        ):
            module.quant_method = DraftLowMLinearMethod()
            count += 1
    return count


def prebuild(draft_model: nn.Module) -> int:
    """Compile (and run once) every covered row count of every installed drafter linear, outside capture, on the
    current stream. Returns the number of calls compiled."""
    use_pdl = get_env_enable_pdl()
    built = 0
    shapes = {
        tuple(module.weight.shape): module.weight
        for module in draft_model.modules()
        if isinstance(module, Linear) and isinstance(module.quant_method, DraftLowMLinearMethod)
    }
    for (n, k), weight in sorted(shapes.items()):
        for nk, m in sorted(key for key in dispatch.DRAFT_LOWM.keys if key[0] == (n, k)):
            if (n, k, m, use_pdl) in _BUILT:
                continue
            kind, tactic = tactic_for(n, k, m)
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
            m3_draft_lowm(x, weight, out, kind, tactic, use_pdl)
            _BUILT.add((n, k, m, use_pdl))
            built += 1
    return built
