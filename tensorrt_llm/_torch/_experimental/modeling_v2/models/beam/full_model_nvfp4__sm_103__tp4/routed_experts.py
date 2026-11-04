# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 sparse experts of one rank on the device: every route through one grouped GEMM call.

The expert provider of the full model's sparse layers (``component.moe``). For the normalized
input ``h`` ``[T, 5120]`` bf16 and the router's ``ids`` ``[T, 8]`` int64 and ``weights``
``[T, 8]`` fp32, with the layer's operands (``ExpertOperands``)::

    rows     = h with each token repeated once per route        [8T, 5120] bf16, row 8t + j = h[t]
    fc1, sf1 = fp4_quantize(rows, g_fc1)                        the layer's one FC1 input scale
    y        = fused_moe(fc1, ids as [8T, 1] int32, unit weights)                [8T, 5120] bf16
               per route: gate = bf16(alpha * fc1 @ W_gate), up = bf16(alpha * fc1 @ W_up),
               act = bf16(silu(gate.float()) * up.float()), quantized with the layer's one FC2
               input scale, y = bf16(alpha_down * act @ W_down)
    local    = per token, its eight routes in ascending expert id, from acc = 0:
                   acc += float(bf16(float(y) * weight))                       fp32
               bf16(acc * 2.5)                                                 [T, 5120] bf16

Each (token, route) is its own top-1 row with weight 1.0, so the operator's finalize stores the
route as is -- no sum over routes, no atomics -- and ``y`` holds exactly the unweighted per-route
expert outputs the host-enumerated reference provider (``expert.nvfp4_swiglu_expert`` under
``dispatch_combine.beam_dispatch_combine``) computes. The weighting, the order, the roundings and
the scaling of the combination are this module's, as the source model computes them
(``ordered_combine``). The operator is CUTLASS's NVFP4 grouped GEMM (``torch.ops.trtllm.fused_moe``):
its FC1 epilogue stores gate and up in bf16 and its activation rounds the SwiGLU product to bf16
before quantizing it for FC2, which is the source's rounding chain.

Nothing is read back to the host: the expert list, the per-expert row counts and the row
selection stay on the device, so a CUDA graph captures the call and the overlap scheduler never
waits on it. The operator's tactic table must be warm before a graph captures it (the engine's
autotuning warmup and eager graph warmup run it first): its first call after tuning releases the
tuner's profiling workspace.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.utils import ActivationType

from .layout import GROUP

HIDDEN_SIZE = 5120
EXPERTS_PER_TOKEN = 8
ROUTED_SCALING_FACTOR = 2.5
#: The grouped GEMM's finalize: with one route per row it stores the row unchanged either way.
USE_FUSED_FINALIZE = True

#: The kernel contract, as recorded in load and run records.
CONTRACT = {
    "provider": "torch.ops.trtllm.fused_moe (CUTLASS NVFP4 grouped GEMM), one top-1 row per "
    "(token, route) with weight 1.0, swizzled FC1 input scales, tp_size=1 on the rank's slices",
    "activation_quantizer": "torch.ops.trtllm.fp4_quantize(rows, g_fc1, sf_vec_size=16, "
    "sf_use_ue8m0=False, is_sf_swizzled_layout=True)",
    "activation": "Swiglu: bf16(silu(gate.float()) * up.float())",
    "fc1": "[up; gate] codes and 128x4-swizzled block scales of one fused bank, alpha = "
    "input_scale * weight_scale_2 (equal for gate and up)",
    "combine": "per token, ascending expert id: acc += float(bf16(float(route) * weight)), then "
    "bf16(acc * 2.5)",
}


@dataclasses.dataclass(frozen=True)
class ExpertOperands:
    """One sparse layer's expert operands on this rank, as the grouped GEMM takes them.

    ``fc1_weight`` ``[256, 2I, K/16]`` int64 (``[up; gate]`` E2M1 codes), ``fc1_block_scales``
    ``[256, 2I, K/64]`` int32 (128x4-swizzled E4M3 bytes), ``fc1_alpha`` ``[256]`` fp32,
    ``fc1_input_scale_inv`` 0-D fp32 (``g`` of the layer's one FC1 input scale),
    ``fc2_weight`` ``[256, K, I/16]`` int64, ``fc2_block_scales`` ``[256, K, I/64]`` int32,
    ``fc2_input_scale_inv`` ``[256]`` fp32 and ``fc2_alpha`` ``[256]`` fp32. Every tensor is a
    view of the layer's banks; nothing is copied.
    """

    fc1_weight: torch.Tensor
    fc1_block_scales: torch.Tensor
    fc1_alpha: torch.Tensor
    fc1_input_scale_inv: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_block_scales: torch.Tensor
    fc2_input_scale_inv: torch.Tensor
    fc2_alpha: torch.Tensor

    def quant_scales(self) -> list:
        return [
            self.fc1_input_scale_inv,
            self.fc1_block_scales,
            self.fc1_alpha,
            self.fc2_input_scale_inv,
            self.fc2_block_scales,
            self.fc2_alpha,
        ]


def ordered_combine(
    routes: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> torch.Tensor:
    """Weighted sum of each token's routes in ascending expert id: ``[T, K, H]`` -> ``[T, H]`` bf16.

    ``routes[t, j]`` is the bf16 output of expert ``ids[t, j]`` for token ``t``, unweighted;
    ``weights[t, j]`` its fp32 mixing weight. Each weighted route is rounded to bf16 once and
    added in fp32, starting from zero, in ascending expert id -- the order of the source's
    expert loop, not the router's score order -- and the sum is scaled and rounded once. The
    ids of a row must be distinct (the router's top-k are). Device-only: no value is read back.
    """
    tokens, top_k, hidden = routes.shape
    order = ids.argsort(dim=-1)
    ordered_weights = weights.gather(1, order)
    acc = torch.zeros(tokens, hidden, dtype=torch.float32, device=routes.device)
    for j in range(top_k):
        slot = order[:, j].view(tokens, 1, 1).expand(tokens, 1, hidden)
        route = routes.gather(1, slot).view(tokens, hidden)
        acc += (route.float() * ordered_weights[:, j : j + 1]).to(torch.bfloat16).float()
    return (acc * routed_scaling_factor).to(torch.bfloat16)


@torch.no_grad()
def nvfp4_routed_experts(
    h: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    operands: ExpertOperands,
    *,
    tune_max_rows: int,
    record: Optional[dict] = None,
) -> torch.Tensor:
    """The rank's MoE partial for ``h`` ``[T, 5120]`` bf16 -> ``[T, 5120]`` bf16 (before the TP sum).

    ``ids`` ``[T, 8]`` int64 and ``weights`` ``[T, 8]`` fp32 are the router's selection and
    normalized weights, used as given. ``tune_max_rows`` is the largest operator row count the
    engine can issue (eight per token of its largest step), the autotuner's bound. With
    ``record`` (a dict), the call's operator inputs and per-route outputs are stored in it.
    """
    tokens = h.shape[0]
    num_rows = tokens * EXPERTS_PER_TOKEN
    # One token's expansion reshapes to a stride-0 view; the quantizer needs dense rows.
    rows = (
        h.unsqueeze(1)
        .expand(tokens, EXPERTS_PER_TOKEN, HIDDEN_SIZE)
        .reshape(num_rows, HIDDEN_SIZE)
        .contiguous()
    )
    fc1, fc1_sf = fp4_quantize(rows, operands.fc1_input_scale_inv, GROUP, False, True)
    route_ids = ids.reshape(num_rows, 1).to(torch.int32)
    unit_weights = torch.ones(num_rows, 1, dtype=torch.float32, device=h.device)
    routes = torch.empty(num_rows, HIDDEN_SIZE, dtype=torch.bfloat16, device=h.device)
    torch.ops.trtllm.fused_moe(
        fc1,
        route_ids,
        unit_weights,
        operands.fc1_weight,
        None,
        operands.fc2_weight,
        None,
        torch.bfloat16,
        quant_scales=operands.quant_scales(),
        input_sf=fc1_sf,
        swizzled_input_sf=True,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_fused_finalize=USE_FUSED_FINALIZE,
        tune_max_num_tokens=tune_max_rows,
        activation_type=int(ActivationType.Swiglu),
        unpadded_hidden_size=HIDDEN_SIZE,
        out_tensor=routes,
    )
    routes = routes.view(tokens, EXPERTS_PER_TOKEN, HIDDEN_SIZE)
    local = ordered_combine(routes, ids, weights)
    if record is not None:
        record.update(
            {
                "route_fc1_fp4": fc1,
                "route_fc1_sf_swizzled": fc1_sf,
                "routes": routes,
                "local": local,
            }
        )
    return local


__all__ = [
    "CONTRACT",
    "EXPERTS_PER_TOKEN",
    "ExpertOperands",
    "nvfp4_routed_experts",
    "ordered_combine",
]
