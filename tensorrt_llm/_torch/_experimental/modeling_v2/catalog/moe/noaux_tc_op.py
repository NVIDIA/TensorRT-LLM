# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped top-k expert routing by biased sigmoid score, via the trtllm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _step5_rows(*rows: tuple[int, str]) -> tuple[Cell, ...]:
    """One Step-5-Preview router cell per row count.

    The row count leads `why`, so each cell's test id (the first 44
    characters) stays distinct.
    """
    return tuple(
        Cell(
            why=f"step5 T={tokens}: {what}",
            spec=dict(
                tokens=tokens,
                experts=352,
                n_group=1,
                topk_group=1,
                topk=8,
                scale=3.0,
                logits_dtype=torch.float32,
            ),
        )
        for tokens, what in rows
    )


class NoauxTcOp(OpWrapper):
    """Route each token to `topk` experts by biased sigmoid score.

    Selection uses `sigmoid(router_logits) + bias`; the combine weights are
    gathered from the *unbiased* `sigmoid(router_logits)`, renormalized to sum
    to 1 and scaled by `routed_scaling_factor`. The bias steers which experts
    are chosen and never reaches the weights -- getting that backwards is the
    single easiest way to misread this op.

    Returns `(topk_weights [T, topk] in router_logits.dtype, topk_ids [T, topk]
    int32)`, both newly allocated and contiguous.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The first five cells are deepseek-r1's router: 256 experts in 8 groups,
    # 4 groups kept, 8 experts per token, scaled by 2.5, at the token counts
    # that target produces -- one per decode step, and the chunk it splits a
    # prefill into. After them, Step-5-Preview's
    # router: 352 experts in one group, 8 per token, scaled by 3.0, on fp32
    # logits, at every row count a Step-5 MoE layer routes at once.
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
        Cell(
            why="the default target's expert chunk, which its attention-DP gather fills",
            spec=dict(tokens=8192, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
        Cell(
            why="deepseek MTP router: fp32 logits, so the weights are the fp32 "
            "token_final_scales fused_moe needs",
            spec=dict(
                tokens=256,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.float32,
            ),
        ),
        *_step5_rows(
            (1, "decode: one request at tp4/tep4"),
            (37, "a token count that is not a multiple of any tile"),
            (128, "decode: max_batch_size requests at tp4/tep4"),
            (512, "dep4 decode: 4 x 128 rows after the all-gather"),
            (8192, "prefill: one max_num_tokens chunk per rank"),
            (32768, "dep4 prefill: 4 x 8192 rows after the all-gather"),
        ),
    )

    note = """
    The kernel evaluates sigmoid as `0.5 * tanh(x/2) + 0.5`, which saturates to
    exactly 1.0 at x >= ~17 and exactly 0.0 at x <= ~-18.5. `reference` uses the
    same form; the 1/(1+exp(-x)) form disagrees in the tails.

    Ties in the selection score are broken toward the lower expert id, which
    `reference` reproduces with a stable descending sort.

    ARCHS: the five deepseek-r1 cells were driven on sm_100 and sm_103. The
    Step-5 cells (`why` starting `step5`) were driven on sm_103 only and are
    uncertified on sm_100 until they run there.

    Which kernel runs. `n_group = 1` takes the ungrouped path of
    `invokeNoAuxTc` (`noAuxTcKernels.cu`): 352 experts run on the
    512-expert instantiation (256 threads, experts 352..511 padded with a
    -inf selection key, so never chosen while topk <= 352). Within it, the
    token count picks one of two top-k implementations: at most 256 tokens,
    the two-level parallel top-k (`SmallBatch`); above, the single-warp
    iterated `reduceTopK`. The crossover is read once per process from
    `TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS` (default 256); the cells run with
    it unset, so T = 1, 37 and 128 drive the first and T = 512, 8192 and
    32768 the second. On sm_103 both are bit-exact against `reference` in
    ids and in fp32 weights at every Step-5 cell; the test's tie case also
    runs T = 256 and 257, either side of the default crossover.

    Step-5 caller preconditions: `router_logits` the fp32 output of the
    router GEMM, dense [T, 352]; `bias` the checkpoint's fp32
    `moe.router_bias`, dense [352]. With fp32 logits the weights come back
    fp32 -- the `token_final_scales` fused_moe takes -- already scaled by
    `routed_scaling_factor`. The op allocates both outputs on each call, so
    under CUDA-graph capture they come from the graph's private pool and are
    overwritten on every replay.

    Hard limits: the kernel addresses logits as `token * num_experts + e` in
    int64 and launches one CTA per token, so no int32 bound is near: 2^31
    logits is 6.1M rows at 352 experts, against 32768 rows at the
    envelope's widest call (dep4 prefill, 4 x 8192 gathered rows). No index
    passes through fp32.

    Uncertified: `TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS` set to anything but
    its default; bf16 or fp16 logits at 352 experts (the target passes fp32);
    a bf16 or fp16 bias; topk other than 8 or scale other than 3.0 at 352.
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


noaux_tc_op = NoauxTcOp()
