# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 routing and NVFP4 quantization of the MoE input in one launch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

_SF_VEC = 16


def _sum_partials(parts: torch.Tensor) -> torch.Tensor:
    """fp32 sum of `[S, n, E]` K-chunk partials in the kernel's order, s = 0..S-1."""
    total = parts[0].clone()
    for s in range(1, parts.shape[0]):
        total += parts[s]
    return total


class Dsv4RouteNvfp4Quant(OpWrapper):
    """`(ids, weights, x_fp4, x_sf)` of the rows of `x`, from the router logits, in one CuTe-DSL launch.

    Routing is the checkpoint's: `scores = sqrt(softplus(logits))`, the `top_k` experts of `scores + selection_bias`,
    their `scores` normalised to sum 1 and scaled by `routed_scaling_factor`. The input is quantized to NVFP4 (group 16,
    e4m3 scales in row-major order, `global_scale` the one-element fp32 input scale) exactly as `fp4_quantize` does.
    `logits` `[n, E]` fp32, `selection_bias` `[E]` fp32, `x` `[n, H]` bf16; returns ids int32 `[n, top_k]`, weights
    bf16 `[n, top_k]`, x_fp4 uint8 `[n, H/2]` and x_sf uint8 `[n, H/16]`. Replaces about fourteen torch launches of the
    unfused routing plus the quantizer. `logits` may also be the engine's `RouterLogitPartials` (fp32 K-chunk partials
    `[S, n, E]`, see `dsv4_router_logits(partials=True)`): the launch routes on their sum, s = 0..S-1.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="one row (a one-token step)",
            spec=dict(n=1, hidden=5120, experts=384, top_k=6, scale=1.5),
        ),
        Cell(
            why="a 300-row step of the 384-expert bank",
            spec=dict(n=300, hidden=5120, experts=384, top_k=6, scale=1.5),
        ),
        Cell(
            why="an 8192-row MoE chunk",
            spec=dict(n=8192, hidden=5120, experts=384, top_k=6, scale=1.5),
        ),
    )
    note = (
        "Experts picked from exactly tied scores may come out in a "
        "different order than torch.topk's (the test compares sets)."
    )

    def __call__(
        self,
        logits,
        selection_bias,
        x,
        global_scale,
        top_k: int,
        routed_scaling_factor: float,
        weights_dtype: torch.dtype = torch.bfloat16,
        staged=None,
        slot_of_expert=None,
        replicas: bool = False,
        mxfp8=None,
        counters=None,
    ) -> tuple:
        """With `staged` = a MegaMoE bucket's (index, weights, NVFP4 rows, scale rows) the launch writes the routing and
        the quantized rows there (the bucket's other rows get slot -1, weight 0); `slot_of_expert` (the placement table,
        a replica table when `replicas`) makes the written ids expert slots, and `mxfp8` = (E4M3 rows, UE8M0 scales)
        also receives each row as MXFP8. The returned tensors are then the launch's own (ids stay expert ids).

        `counters` = (histogram int32 `[E]`, flag int32 `[1]`, live-row bound int32 `[1]`, row mask int8 or None, the
        launch's first row in the forward): the engine's warmup expert placement counters (`RouteCounters`). While the
        flag is set, the launch adds the expert picks of every counted row to the histogram: rows below the bound (`-1`:
        every row) whose mask byte is set (with a mask) and whose logits are not all zero. Nothing else changes."""
        from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import (
            dsv4_route_nvfp4_quant,
        )

        return dsv4_route_nvfp4_quant(
            logits,
            selection_bias,
            x,
            global_scale,
            top_k,
            routed_scaling_factor,
            weights_dtype,
            staged=staged,
            slot_of_expert=slot_of_expert,
            replicas=replicas,
            mxfp8=mxfp8,
            counters=counters,
        )[:4]

    def reference(
        self, logits, selection_bias, x, global_scale, top_k: int, routed_scaling_factor: float
    ) -> tuple:
        if hasattr(logits, "parts"):
            logits = _sum_partials(logits.parts)
        scores = torch.sqrt(F.softplus(logits.float()))
        ids = torch.topk(scores + selection_bias, top_k, dim=-1)[1]
        w = torch.gather(scores, 1, ids)
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-20)
        xq, sf = fp4_quantize.reference(x, global_scale, _SF_VEC, False, False)
        return (
            ids.to(torch.int32),
            (w * routed_scaling_factor).to(torch.bfloat16),
            xq,
            sf.view(x.shape[0], -1),
        )

    def is_valid(
        self,
        logits,
        selection_bias,
        x,
        global_scale,
        top_k: int,
        routed_scaling_factor: float,
        weights_dtype: torch.dtype = torch.bfloat16,
        staged=None,
        slot_of_expert=None,
        replicas: bool = False,
        mxfp8=None,
        counters=None,
    ) -> None:
        assert weights_dtype in (torch.bfloat16, torch.float32)
        if counters is not None:
            hist, flag, bound, mask, _ = counters
            assert hist.dtype == flag.dtype == bound.dtype == torch.int32 and hist.is_contiguous()
            assert hist.shape == (selection_bias.shape[0],) and flag.numel() == bound.numel() == 1
            assert mask is None or mask.dtype == torch.int8
        assert not replicas or slot_of_expert is not None, (
            "a replica table is a slot_of_expert table"
        )
        if hasattr(logits, "parts"):
            assert (
                logits.parts.dtype == torch.float32
                and logits.parts.dim() == 3
                and logits.parts.stride(2) == 1
            )
            logits = logits.parts[0]
        assert logits.dtype == torch.float32 and logits.dim() == 2 and logits.shape[0] == x.shape[0]
        assert selection_bias.dtype == torch.float32 and selection_bias.shape == (logits.shape[1],)
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] % 16 == 0
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
        assert 0 < top_k <= logits.shape[1]


dsv4_route_nvfp4_quant = Dsv4RouteNvfp4Quant()


class Dsv4Route(OpWrapper):
    """`dsv4_route_nvfp4_quant`'s routing alone, `(ids int32 [n, top_k], weights [n, top_k])`, for an input its
    producer already quantized (stock's `dsv4_route`, as the CuTe-DSL MoE's `try_fused_route` runs it). `weights_dtype`:
    bf16 (the TRT-LLM-gen runner's routing scales), or fp32 as the CuTe-DSL experts take them (stock's
    `routing_scales_dtype=torch.float32`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = Dsv4RouteNvfp4Quant.CELLS
    note = Dsv4RouteNvfp4Quant.note

    def __call__(
        self,
        logits,
        selection_bias,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        weights_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple:
        from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import dsv4_route

        return dsv4_route(
            logits, selection_bias, top_k, routed_scaling_factor, weights_dtype, hidden
        )[:2]

    def reference(
        self,
        logits,
        selection_bias,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        weights_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple:
        scores = torch.sqrt(F.softplus(logits.float()))
        ids = torch.topk(scores + selection_bias, top_k, dim=-1)[1]
        w = torch.gather(scores, 1, ids)
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-20)
        return ids.to(torch.int32), (w * routed_scaling_factor).to(weights_dtype)

    def is_valid(
        self,
        logits,
        selection_bias,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        weights_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        assert weights_dtype in (torch.bfloat16, torch.float32)
        assert logits.dtype == torch.float32 and logits.dim() == 2 and logits.is_contiguous()
        assert selection_bias.dtype == torch.float32 and selection_bias.shape == (logits.shape[1],)
        assert 0 < top_k <= logits.shape[1]


dsv4_route = Dsv4Route()
