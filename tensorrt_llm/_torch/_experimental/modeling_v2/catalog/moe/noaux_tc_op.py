# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-k expert routing by biased sigmoid score, grouped or single-group, via the trtllm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class NoauxTcOp(OpWrapper):
    """Route each token to `topk` experts by biased sigmoid score.

    Selection uses `sigmoid(router_logits) + bias`; the combine weights are
    gathered from the *unbiased* `sigmoid(router_logits)`, renormalized to sum
    to 1 and then scaled by `routed_scaling_factor`, so each row of weights sums
    to `routed_scaling_factor`. The bias steers which experts are chosen and
    never reaches the weights -- getting that backwards is the single easiest
    way to misread this op.

    Returns `(topk_weights [T, topk] in router_logits.dtype, topk_ids [T, topk]
    int32)`, both newly allocated and contiguous, each row in descending order
    of biased score.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # Two routers call this.
    # - deepseek-r1: 256 experts in 8 groups, 4 groups kept, 8 experts per token,
    #   scale 2.5, bf16 logits.
    # - nemotron-3 ultra: 512 experts in one group, 22 per token, scale 5.0, fp32
    #   logits, and an fp32 bias of 25 to 57 that changes per layer.
    # One group takes a different kernel path from eight, and that path switches
    # top-k implementations at 256 tokens. The token counts are the ones the
    # targets produce: one per decode step, the decode CUDA-graph ceiling, and
    # the chunk a prefill is split into.
    #
    # The ultra checkpoint's biases sit on a 0.125 grid, with one to four levels
    # per layer. The synthetic ultra cells draw them uniformly over layer 107's
    # range at full fp32 resolution, so every bit of the bias add counts.
    # Logits are drawn at the scale each layer's gate produces on unit-RMS
    # inputs: std 1.3 at layer 107 and 0.004 at layer 1.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one token through the full 256-expert router",
            spec=dict(
                tokens=1,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.bfloat16,
                logit_std=1.0,
                bias=("normal", 0.0, 1.0),
            ),
        ),
        Cell(
            why="prefill chunk: the batch the target routes at once",
            spec=dict(
                tokens=2048,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.bfloat16,
                logit_std=1.0,
                bias=("normal", 0.0, 1.0),
            ),
        ),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(
                tokens=37,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.bfloat16,
                logit_std=1.0,
                bias=("normal", 0.0, 1.0),
            ),
        ),
        Cell(
            why="nemotron-3 ultra decode, 1 token: the single-group path (512 experts, "
            "top-22) the 8-group cells never reach; fp32 logits, so fp32 weights",
            spec=dict(
                tokens=1,
                experts=512,
                n_group=1,
                topk_group=1,
                topk=22,
                scale=5.0,
                logits_dtype=torch.float32,
                logit_std=1.3,
                bias=("uniform", 37.75, 38.25),
            ),
        ),
        Cell(
            why="nemotron-3 ultra decode at its 32-row CUDA-graph ceiling: still the "
            "small-batch top-k, which serves every call of up to 256 tokens",
            spec=dict(
                tokens=32,
                experts=512,
                n_group=1,
                topk_group=1,
                topk=22,
                scale=5.0,
                logits_dtype=torch.float32,
                logit_std=1.3,
                bias=("uniform", 37.75, 38.25),
            ),
        ),
        Cell(
            why="nemotron-3 ultra prefill, 8192 tokens: above 256 tokens the kernel "
            "switches to its single-warp iterated top-k",
            spec=dict(
                tokens=8192,
                experts=512,
                n_group=1,
                topk_group=1,
                topk=22,
                scale=5.0,
                logits_dtype=torch.float32,
                logit_std=1.3,
                bias=("uniform", 37.75, 38.25),
            ),
        ),
        Cell(
            why="layer 1's checkpoint bias (25.25 / 25.375) at its own logit scale: "
            "scores crowd 0.5, so exact 22nd-place ties are routine and go to the lower id",
            spec=dict(
                tokens=8192,
                experts=512,
                n_group=1,
                topk_group=1,
                topk=22,
                scale=5.0,
                logits_dtype=torch.float32,
                logit_std=0.004,
                bias=("checkpoint", 1),
            ),
        ),
        Cell(
            why="layer 107's checkpoint bias (37.75 / 38 / 38.25) on the decode path, "
            "where the fp32 add keeps the score only to 2^-18",
            spec=dict(
                tokens=32,
                experts=512,
                n_group=1,
                topk_group=1,
                topk=22,
                scale=5.0,
                logits_dtype=torch.float32,
                logit_std=1.3,
                bias=("checkpoint", 107),
            ),
        ),
    )

    note = """
    The kernel evaluates sigmoid as `0.5 * tanh(x/2) + 0.5`, which saturates to
    exactly 1.0 at x >= ~17 and exactly 0.0 at x <= ~-18.5. `reference` uses the
    same form; the 1/(1+exp(-x)) form disagrees in the tails.

    Ties in the selection score are broken toward the lower expert id, which
    `reference` reproduces with a stable descending sort.

    Everything that decides the selection is fp32, whatever the input dtypes:
    - the sigmoid;
    - the bias add (the bias is widened first);
    - the top-k key, which keeps all 32 bits of the biased score.
    The weights' denominator is an fp32 warp sum of the selected unbiased
    scores. The division runs in fp64 and rounds once to `router_logits.dtype`,
    so fp32 logits give fp32 weights.

    A large bias costs score resolution, and only an fp32 add keeps enough.
    `sigmoid + bias` rounds to the fp32 step at the bias's magnitude: 2^-19 at
    25, 2^-18 at 38. A bf16 add would round to 1/8 or 1/4 there. At fp32
    resolution, exact ties at the top-k boundary are still routine: about 4% of
    tokens at nemotron-3 ultra's layer-1 logit scale. They go to the lower id.
    HF's `torch.topk` leaves tie order unspecified, so a comparison against HF
    can agree only up to ties.

    Single-group routing has two top-k implementations, switched at 256 tokens
    (`TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS` moves the switch). Up to the switch,
    a two-level parallel top-k serves decode-sized batches; above it, a
    single-warp iterated top-k runs. They agree bit for bit, and the cells drive
    both.
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
        """Native torch: the kernel's own arithmetic for the selection, fp64 for the weights.

        The selection matches the kernel bit for bit, ties included. That needs
        two of the kernel's choices rather than the obvious ones: the tanh form
        of sigmoid, and an fp32 bias add.

        The weights are normalized entirely in fp64, sum included, and rounded
        once, so this side is the more accurate one. The kernel sums the
        selected scores with a 5-level fp32 warp tree, and the test's band is
        that sum's rounding.
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
        weights = torch.gather(scores, 1, ids).double()
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * routed_scaling_factor
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


noaux_tc_op = NoauxTcOp()
