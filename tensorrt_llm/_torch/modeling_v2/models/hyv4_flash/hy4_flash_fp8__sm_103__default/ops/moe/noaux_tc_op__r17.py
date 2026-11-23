# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped top-k expert routing by biased sigmoid score, at the Hy4-Flash router.

Local variant of ``catalog/moe/noaux_tc_op.py`` (``__r17``) owned by the
Hy4-Flash unified target: same op call, same reference and same guards, with
``CELLS`` replaced by the configurations this target calls it with. The shared
entry and its cells are unchanged; this file is what the target imports.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's router: 512 experts in one group, 16 experts per token,
#: scaling 3.998, bf16 logits and a bf16 `e_score_correction_bias` [512].
_HY4 = dict(
    experts=512,
    n_group=1,
    topk_group=1,
    topk=16,
    scale=3.998,
    logits_dtype=torch.bfloat16,
    bias_dtype=torch.bfloat16,
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

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash router's rows: one token per decode step, a 64-row block,
    # and the 5120-token prefill chunk, plus a token count that is not a
    # multiple of any tile. The kernel dispatches on the token count as well as
    # the expert count: 1, 37 and 64 take its two-level small-batch path and
    # 5120 its single-warp path, so the cells cover both.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one token through the 512-expert router",
            spec=dict(**_HY4, tokens=1),
        ),
        Cell(
            why="64-row end: the router over a short batch",
            spec=dict(**_HY4, tokens=64),
        ),
        Cell(
            why="prefill chunk: the router over the 5120-token chunk's rows",
            spec=dict(**_HY4, tokens=5120),
        ),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(**_HY4, tokens=37),
        ),
    )

    note = """
    Local variant (`__r17`) of `catalog/moe/noaux_tc_op.py`, owned by the
    Hy4-Flash unified target. The shared entry's own cells -- DeepSeek-R1's 256
    experts in 8 groups -- stay with the shared entry: this checkpoint's router
    is 512 experts in a single group.

    The op is a compiled-in C++ thop (`cpp/tensorrt_llm/thop/noAuxTcOp.cpp`,
    `TORCH_LIBRARY_FRAGMENT(trtllm, m)`), so it is in the wheel; there is no JIT
    path to certify. `invokeNoAuxTc` picks between two ungrouped kernels by
    token count (`TLLM_NOAUX_TC_SMALL_BATCH_MAX_TOKENS`, default 256): the
    two-level redux path at or below it (cells 1, 37, 64) and the single-warp
    iterated path above it (cell 5120). The two are documented bit-identical
    and both are cells here.

    The kernel evaluates sigmoid as `0.5 * tanh(x/2) + 0.5`, which saturates to
    exactly 1.0 at x >= ~17 and exactly 0.0 at x <= ~-18.5. `reference` uses the
    same form; the 1/(1+exp(-x)) form disagrees in the tails.

    Ties in the selection score are broken toward the lower expert id, which
    `reference` reproduces with a stable descending sort.

    Measured on sm_103 (GB300) against `reference`, bf16 logits and bf16 bias:
    tokens 1, 37 and 64 are bit-identical (0 differing elements of 16, 592 and
    1024); tokens 5120 differs on 1 element of 81920, by exactly one bf16 ulp
    (0.39% relative, 1.3e-5 relative RMS). That is the kernel's float32 warp
    sum against torch's float32 row sum -- ~1e-7 relative before the rounding
    -- landing on opposite sides of a bf16 midpoint. Row sums are within
    2.1e-3 relative of 3.998, the bf16 accumulation of 16 rounded weights. One
    call at 5120 tokens: 24 us.

    Not certified here: experts other than 512, n_group > 1, topk other than
    16, scales other than 3.998, fp32 or fp16 logits, an fp32 bias, and zero
    tokens (the op returns empty outputs; the kernel's own test covers that).
    `topk > 32`, `n_group > 32` and `num_experts % n_group != 0` raise loudly
    in the thop.
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
