# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped sigmoid top-k routing with a selection-only correction bias, all in fp32."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Block 3's router: 256 experts in 8 groups of 32, 4 groups kept, 8 experts per token.
_BLOCK3 = dict(hidden_size=5120, num_experts=256, n_group=8, topk_group=4, top_k=8, eps=1e-20)


class BeamGroupedTopkRouter(OpWrapper):
    """Route `hidden_states [T, hidden]` with `weight [E, hidden]` and `correction_bias [E]`.

    Every step is fp32, with TF32 and autocast off for the call::

        logits = hidden_states @ weight^T                    [T, E]
        scores = sigmoid(logits)                             unbiased
        selection_scores = scores + correction_bias          used only to choose
        group_scores = sum of the two largest selection_scores in each of n_group
                       contiguous groups                     [T, n_group]
        groups = top-topk_group group_scores, descending     [T, topk_group]
        masked_selection_scores = selection_scores, -inf outside those groups
        ids = top-top_k masked_selection_scores, descending  [T, top_k] int64
        weights = scores[ids] / (sum(scores[ids]) + eps)     [T, top_k]

    Returns every tensor above in a dict under those names, so a caller that needs only
    `ids` and `weights` and a test that checks each step observe the same call. The
    correction bias decides the selection and never reaches `weights`; no routed scaling
    factor is applied.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block 3, 1 token", spec=dict(_BLOCK3, tokens=1)),
        Cell(why="block 3, 17 tokens", spec=dict(_BLOCK3, tokens=17)),
        Cell(
            why="block 3, 129 tokens: several hundred decision margins",
            spec=dict(_BLOCK3, tokens=129),
        ),
    )

    note = """
    Ties follow `torch.topk(..., sorted=True)` and nothing else: there is no secondary
    index rule. Measured on sm_103 at these shapes, a tie straddling the 4th/5th group or
    the 8th/9th expert boundary keeps the lower index, and tied values inside the
    selection are not in index order -- a row of 256 equal scores selects groups
    [1, 0, 2, 3] and experts [7, 6, 4, 5, 1, 0, 2, 3]. That is the source model's order,
    which is why the selection is not delegated to a kernel with its own tie rule.

    The fused grouped top-k op (`noaux_tc_op`) is not this computation: it takes logits,
    uses a tanh-form sigmoid that saturates where this one does not, renormalizes in
    fp64, applies the routed scaling factor itself and breaks ties by lower expert index.
    `reference` keeps the source's `topk` selections because no other formulation
    reproduces their order on ties.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        top_k: int,
        eps: float,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", enabled=False):
            with true_fp32_matmul():
                logits = F.linear(hidden_states.float(), weight)
            scores = logits.sigmoid()
            selection_scores = scores + correction_bias
            tokens, num_experts = selection_scores.shape
            per_group = num_experts // n_group
            grouped = selection_scores.view(tokens, n_group, per_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            groups = group_scores.topk(topk_group, dim=-1, sorted=True).indices
            group_mask = torch.zeros_like(group_scores).scatter_(1, groups, 1)
            expert_mask = (
                group_mask.unsqueeze(-1)
                .expand(tokens, n_group, per_group)
                .reshape(tokens, num_experts)
            )
            masked_selection_scores = selection_scores.masked_fill(
                ~expert_mask.bool(), float("-inf")
            )
            ids = masked_selection_scores.topk(top_k, dim=-1, sorted=True).indices
            selected = scores.gather(-1, ids)
            weights = selected / (selected.sum(dim=-1, keepdim=True) + eps)
        return dict(
            logits=logits,
            scores=scores,
            selection_scores=selection_scores,
            group_scores=group_scores,
            groups=groups,
            masked_selection_scores=masked_selection_scores,
            ids=ids,
            weights=weights,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        top_k: int,
        eps: float,
    ) -> dict[str, torch.Tensor]:
        """The routing in plain torch: fp32 operands, TF32 and autocast off.

        Exclusion is written as group membership and each group's score as the first two
        entries of a descending sort; both are exact, so the result is the source routing's
        bit for bit. The two selections stay `topk(..., sorted=True)`, the source's own
        call, because the order of tied scores is whatever that call returns.
        """
        with torch.autocast("cuda", enabled=False), true_fp32_matmul():
            logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))
            scores = torch.sigmoid(logits)
            selection_scores = scores + correction_bias.to(torch.float32)
            tokens, num_experts = selection_scores.shape
            per_group = num_experts // n_group
            ranked = selection_scores.reshape(tokens, n_group, per_group).sort(
                dim=-1, descending=True
            )
            group_scores = ranked.values[..., :2].sum(dim=-1)
            groups = torch.topk(group_scores, topk_group, dim=-1, sorted=True).indices
            expert_group = torch.arange(num_experts, device=logits.device) // per_group
            eligible = (expert_group.view(1, num_experts, 1) == groups.unsqueeze(1)).any(dim=-1)
            masked_selection_scores = torch.where(
                eligible, selection_scores, torch.full_like(selection_scores, float("-inf"))
            )
            ids = torch.topk(masked_selection_scores, top_k, dim=-1, sorted=True).indices
            selected = torch.take_along_dim(scores, ids, dim=-1)
            weights = selected / (selected.sum(dim=-1, keepdim=True) + eps)
        return dict(
            logits=logits,
            scores=scores,
            selection_scores=selection_scores,
            group_scores=group_scores,
            groups=groups,
            masked_selection_scores=masked_selection_scores,
            ids=ids,
            weights=weights,
        )

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        top_k: int,
        eps: float,
    ) -> None:
        # A [1, T, hidden] batch projects fine, and the group view then folds its leading
        # dimension away -- silently when T == 1.
        assert hidden_states.dim() == 2, (
            f"hidden_states must be [tokens, hidden], got {tuple(hidden_states.shape)}"
        )
        # A length-1 bias broadcasts over every expert, and a wider one promotes every
        # selection score out of fp32; neither raises.
        assert correction_bias.shape == (weight.shape[0],), (
            f"correction_bias must be [{weight.shape[0]}], got {tuple(correction_bias.shape)}"
        )
        assert correction_bias.dtype == torch.float32, (
            f"correction_bias must be torch.float32, got {correction_bias.dtype}"
        )
        num_experts = weight.shape[0]
        assert num_experts % n_group == 0 and num_experts // n_group >= 2, (
            f"{num_experts} experts do not form {n_group} groups of at least two"
        )
        # With fewer eligible experts than top_k, the expert top-k returns excluded ones.
        assert 0 < topk_group <= n_group and 0 < top_k <= topk_group * (num_experts // n_group), (
            f"top_k={top_k} cannot be chosen from {topk_group} of {n_group} groups"
        )


beam_grouped_topk_router = BeamGroupedTopkRouter()
