# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE prologue of the row-owner island (``row_owner_moe``) on the owner's rows.

A decode step of ``R`` rows runs the MoE prologue -- router GEMM + K-split sum, the fused
top-8 routing, the pool GEMM and the pre-expert RMSNorm + NVFP4 quantization -- on every
row of every TP rank. Each of these is a per-row function of its input row once three
launch choices are fixed to the full step's, so the owner of rows ``t = ranks * i + rank``
can run them on its ``R / ranks`` rows and get the full step's bits:

* router: the K-split count is part of each logit's summation order, so the owner's call
  runs ``splits(R)`` -- the full step's count -- passed per call (the served path picks it
  from the call's own row count); the staged epilogue and the row tile are row-local;
* routing: launched with the arguments of a call over all ``R`` rows (``TORCH_ORDER`` above
  128 rows, one warp from 512 rows), one program per row -- the context-parallel MoE's
  ``shard_routing`` rule;
* pool GEMM: a cuBLAS split-K-1 kernel whose per-element K order is the full step's pick
  (checked bitwise per row count and layer weight; a row count that falls off
  needs a pinned pick);
* ``rmsnorm_fp4quant``: per row (unswizzled scales).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton

from . import router_gemm as _rg


def router_logits(
    hidden: torch.Tensor, weight: torch.Tensor, scale: float, split_count: int
) -> torch.Tensor:
    """``RouterGemm`` with an explicit K-split count (the full step's, ``splits(R)``)."""
    rows = hidden.shape[0]
    partials = torch.empty(
        (split_count, rows, _rg.EXPERTS), device=hidden.device, dtype=torch.float32
    )
    _rg._gemm(_rg.row_tile(rows))(hidden, weight, partials, split_count)
    output = torch.empty((rows, _rg.EXPERTS), device=hidden.device, dtype=torch.float32)
    size = rows * _rg.EXPERTS
    splits_pow2 = triton.next_power_of_2(split_count)
    block = _rg.SUM_TILE // splits_pow2
    _rg._router_gemm_sum_kernel[(triton.cdiv(size, block),)](
        partials,
        output,
        scale,
        size,
        SPLITS=split_count,
        SPLITS_POW2=splits_pow2,
        BLOCK=block,
        num_warps=4,
        launch_pdl=True,
    )
    return output


def routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    full_rows: int,
    selected: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    renorm_eps: float = 1e-20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The fused top-8 routing on the owner's rows, launched as for ``full_rows`` rows."""
    from tensorrt_llm._torch.models import modeling_musespark_open as _builtin

    rows = logits.shape[0]
    if selected is None:
        selected = torch.empty((rows, 8), device=logits.device, dtype=torch.int32)
    if weights is None:
        weights = torch.empty((rows, 8), device=logits.device, dtype=torch.float32)
    _builtin._full_routing_kernel[(rows,)](
        logits,
        bias,
        selected,
        weights,
        renorm_eps,
        EXPERTS=256,
        TORCH_ORDER=full_rows > 128,
        num_warps=1 if full_rows >= 512 else 4,
    )
    return selected, weights


def owner_prologue(
    hidden: torch.Tensor,
    full_rows: int,
    router_weight: torch.Tensor,
    router_scale: float,
    bias: torch.Tensor,
    pool_weight: torch.Tensor,
    quant_gain: torch.Tensor,
    quant_eps: float,
    global_scale: torch.Tensor,
    side: Optional[torch.cuda.Stream] = None,
    sms: Optional[int] = None,
):
    """Router + routing on the current stream || pool + NVFP4 quant on ``side``, joined.

    Returns ``(selected [m, 8] int32, weights [m, 8] fp32, fp4 [m, 2048] uint8,
    scales [m, 256] uint8)`` for the owner's ``m`` rows of a step of ``full_rows`` rows.
    """
    from flashinfer.norm import rmsnorm_fp4quant

    main = torch.cuda.current_stream()
    if side is not None:
        side.wait_stream(main)
    logits = router_logits(hidden, router_weight, router_scale, _rg.splits(full_rows, sms))
    selected, weights = routing(logits, bias, full_rows)
    ctx = torch.cuda.stream(side) if side is not None else torch.cuda.stream(main)
    with ctx:
        pooled = F.linear(hidden, pool_weight)
        fp4, sf = rmsnorm_fp4quant(
            pooled,
            quant_gain,
            eps=quant_eps,
            global_scale=global_scale.reshape(1),
            block_size=16,
            scale_format="e4m3",
            is_sf_swizzled_layout=False,
        )
    if side is not None:
        main.wait_stream(side)
    return selected, weights, fp4.view(torch.uint8), sf.view(torch.uint8)
