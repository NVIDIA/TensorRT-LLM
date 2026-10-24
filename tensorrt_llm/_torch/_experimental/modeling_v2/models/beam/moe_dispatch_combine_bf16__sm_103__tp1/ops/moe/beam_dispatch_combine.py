# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch tokens to their selected experts and combine the weighted outputs in fp32."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Block 3's MoE: 5120 hidden, 256 experts, 8 selected per token, routed scaling factor 2.5.
_BLOCK3 = dict(hidden_size=5120, num_experts=256, top_k=8, routed_scaling_factor=2.5)


class BeamDispatchCombine(OpWrapper):
    """Run every selected expert on its tokens and sum the weighted outputs per token.

    `hidden_states [T, H]` bf16, `expert_ids [T, K]` int64 and `mixing_weights [T, K]` fp32
    are one routing call's tokens and result: slot `j` of row `t` sends token `t` to expert
    `expert_ids[t, j]` with weight `mixing_weights[t, j]`. `expert_provider(e, rows)` is the
    expert computation; it returns expert `e`'s unweighted bf16 output, one row per row of
    `rows`, in their order. The steps, as the source model computes them::

        assignments = the T*K slots in token-major order, each with its token and weight
        for e in the selected experts, ascending:
            tokens  = tokens of the assignments to e, ascending          [n] int64
            out     = expert_provider(e, hidden_states[tokens])          [n, H] bf16
            route   = (out.float() * weights[:, None]).to(bfloat16)      [n, H] each route rounded once
            acc[tokens] += route.float()                                 [T, H] fp32
        output = (acc * routed_scaling_factor).to(bfloat16)              [T, H] bf16

    The provider is called once per selected expert, with at least one row; an expert no
    token selected is never called. Returns `output`. `expert_ids`, `mixing_weights` and
    `hidden_states` are read, never written; the weights are used as given, without
    re-normalization or bias.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block 3, 1 token: eight experts of one row each", spec=dict(_BLOCK3, tokens=1)),
        Cell(
            why="block 3, 17 tokens: experts shared by several tokens",
            spec=dict(_BLOCK3, tokens=17),
        ),
        Cell(
            why="block 3, 129 tokens: almost every expert selected, the longest context",
            spec=dict(_BLOCK3, tokens=129),
        ),
    )

    note = """
    Every weighted route is rounded to bf16 before it is added, and the sum is fp32. On
    finite values that is a different answer from an fp32 route, a bf16 product, a bf16
    accumulator or a sum rounded only at the end, and from folding the scaling factor into
    the weights: each moves outputs by bf16 units. A token's eight routes are added in
    ascending expert order, one expert at a time; within an expert a token appears once,
    so no addition order is left to the hardware. Adding them in slot order, or all at
    once with one scattered add, changes the fp32 sum on elements where routes cancel.

    The selected experts and their tokens are host metadata: `unique()` and `tolist()`
    synchronize with the device twice for the expert list, and the two boolean selections
    twice per selected expert -- 512 synchronizations when 255 experts are selected. Token
    rows, weights, products, roundings, sums and the scaling run on the device of
    `hidden_states`.

    Not the fused MoE runners of the catalog: those own the expert weights and activation
    (and their quantization) as well as the combination, so they cannot call an expert
    given as a function, and they combine without the per-route bf16 rounding.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        mixing_weights: torch.Tensor,
        expert_provider: Callable[[int, torch.Tensor], torch.Tensor],
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        tokens, top_k = expert_ids.shape
        flat_ids = expert_ids.reshape(-1)
        flat_weights = mixing_weights.reshape(-1)
        token_ids = torch.arange(tokens, device=hidden_states.device).repeat_interleave(top_k)
        acc = torch.zeros(
            tokens, hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        for expert_id in flat_ids.unique().tolist():
            assigned = flat_ids == expert_id
            expert_tokens = token_ids[assigned]
            out = expert_provider(expert_id, hidden_states[expert_tokens])
            route = (out.float() * flat_weights[assigned].unsqueeze(-1)).to(torch.bfloat16)
            acc.index_add_(0, expert_tokens, route.float())
        return (acc * routed_scaling_factor).to(torch.bfloat16)

    def reference(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        mixing_weights: torch.Tensor,
        expert_provider: Callable[[int, torch.Tensor], torch.Tensor],
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        """The combination in plain torch, expert by expert over every id up to the largest.

        Each expert's (token, slot) pairs are looked up in the routing table directly, so
        nothing is shared with `__call__`'s flattening and grouping. The product of a bf16
        output and an fp32 weight is exact in fp64; rounding it to fp32 is the fp32 product,
        and that is rounded to bf16 as the source does. The bf16 routes are then added in
        fp32 in ascending expert order, row by row, and the sum is scaled and rounded once.
        Rounding each route is the source's semantics, not a loss of reference accuracy.
        """
        acc = torch.zeros(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for expert_id in range(int(expert_ids.max()) + 1):
            token_slot = (expert_ids == expert_id).nonzero()
            if token_slot.shape[0] == 0:
                continue
            token, slot = token_slot[:, 0], token_slot[:, 1]
            out = expert_provider(expert_id, hidden_states[token])
            weight = mixing_weights[token, slot].to(torch.float64)
            route = (out.to(torch.float64) * weight[:, None]).to(torch.float32).to(torch.bfloat16)
            acc[token] = acc[token] + route.to(torch.float32)
        return (acc * routed_scaling_factor).to(torch.bfloat16)


beam_dispatch_combine_op = BeamDispatchCombine()
