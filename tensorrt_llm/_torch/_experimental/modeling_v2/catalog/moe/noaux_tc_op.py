# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped top-k expert routing by biased sigmoid score, via the trtllm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class _NoauxTcOp(OpWrapper):
    """Route each token to `topk` experts by biased sigmoid score.

    Selection uses `sigmoid(router_logits) + bias`; the combine weights are
    gathered from the *unbiased* `sigmoid(router_logits)`, renormalized to sum
    to 1 and scaled by `routed_scaling_factor`. The bias steers which experts
    are chosen and never reaches the weights -- getting that backwards is the
    single easiest way to misread this op.

    Returns `(topk_weights [T, topk] in router_logits.dtype, topk_ids [T, topk]
    int32)`, both newly allocated and contiguous.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1's router is the only caller: 256 experts in 8 groups, 4 groups
    # kept, 8 experts per token, scaled by 2.5. The token counts are the ones
    # the target produces -- one per decode step, and the chunk it splits a
    # prefill into.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one token through the full 256-expert router",
            spec=dict(tokens=1, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
        Cell(
            why="prefill chunk: the batch the target routes at once",
            spec=dict(tokens=2048, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(tokens=37, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
    )

    note = """
    The kernel evaluates sigmoid as `0.5 * tanh(x/2) + 0.5`, which saturates to
    exactly 1.0 at x >= ~17 and exactly 0.0 at x <= ~-18.5. `reference` uses the
    same form; the 1/(1+exp(-x)) form disagrees in the tails.

    Ties in the selection score are broken toward the lower expert id, which
    `reference` reproduces with a stable descending sort.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.noaux_tc_op(
            router_logits, bias, n_group, topk_group, topk, routed_scaling_factor
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native torch, in the kernel's own arithmetic.

        Two details are the kernel's rather than the obvious choice, and both
        are load-bearing: sigmoid is the tanh form, and the renormalization is
        evaluated in fp64. An all-fp32 normalization disagrees on a third of the
        elements at `routed_scaling_factor = 2.5`; the fp64 one is bit-exact.
        """
        num_experts = router_logits.shape[-1]
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        choice = scores + bias.float()
        if n_group > 1:
            grouped = choice.view(-1, n_group, num_experts // n_group)
            group_score = torch.topk(grouped, k=2, dim=-1).values.sum(-1)
            keep = torch.sort(group_score, dim=-1, descending=True, stable=True).indices[
                :, :topk_group
            ]
            mask = torch.zeros_like(group_score).scatter_(-1, keep, 1.0)
            mask = mask.unsqueeze(-1).expand_as(grouped).reshape(choice.shape)
            choice = torch.where(
                mask.bool(), choice, torch.tensor(float("-inf"), device=choice.device)
            )
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
        weights = torch.gather(scores, 1, ids)
        total = weights.sum(-1, keepdim=True).double()
        weights = weights.double() / (total + 1e-20) * routed_scaling_factor
        return weights.to(router_logits.dtype), ids.to(torch.int32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
    ) -> None:
        # Three domains the op does not police. All three were observed to
        # return a plausible-looking wrong answer instead of raising: the kernel
        # addresses both tensors from data_ptr() as dense buffers and ignores
        # strides, and topk past the end of a row emits expert ids outside the
        # expert range at non-zero weight.
        assert router_logits.is_contiguous(), (
            "router_logits must be contiguous; a strided view is read as a dense "
            "[num_tokens, num_experts] buffer and silently gives wrong routing"
        )
        assert bias.is_contiguous(), (
            "bias must be contiguous; a strided view is read as a dense "
            "[num_experts] buffer and silently gives wrong routing"
        )
        assert topk <= router_logits.shape[-1], (
            f"topk ({topk}) exceeds num_experts ({router_logits.shape[-1]}); the "
            "kernel reads out of bounds and emits out-of-range expert ids"
        )


noaux_tc_op = _NoauxTcOp()
