# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ungrouped sigmoid top-8 expert routing at any token count, one CuTe DSL launch."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import moe_sigmoid_topk_swiglu_rows as _kernel

EXPERTS = _kernel.EXPERTS
TOP_K = _kernel.TOP_K

#: GLM-5.2's router: 256 experts in one group, top-8, scaled by 2.5, fp32 logits.
_GLM = dict(n_group=1, topk_group=1, topk=TOP_K, scale=2.5)


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load.
    The token count is symbolic: one handle serves every chunk."""
    return _kernel.compile_topk()


def takes(router_logits: torch.Tensor, bias: torch.Tensor, n_group: int, topk_group: int,
          topk: int) -> bool:  # fmt: skip
    """Whether the op takes this routing: one group, top-8, contiguous fp32 `[T >= 1, 256]`
    logits and a contiguous fp32 `[256]` bias. Anything else is `noaux_tc_op`'s."""
    return (
        n_group == 1
        and topk_group == 1
        and topk == TOP_K
        and router_logits.dtype is torch.float32
        and router_logits.ndim == 2
        and router_logits.shape[0] >= 1
        and router_logits.shape[1] == EXPERTS
        and router_logits.is_contiguous()
        and bias.dtype is torch.float32
        and tuple(bias.shape) == (EXPERTS,)
        and bias.is_contiguous()
    )


class CuteDslMoeSigmoidTopk(OpWrapper):
    """`noaux_tc_op` for one group: route each token to 8 of 256 experts.

    Selection by `sigmoid(router_logits) + bias`, the combine weights from the
    unbiased `sigmoid(router_logits)`, renormalized and scaled by
    `routed_scaling_factor`. Sigmoid is `0.5 * tanh(x / 2) + 0.5`, as the trtllm op
    evaluates it; ties go to the lower expert id. Returns `(topk_weights [T, 8] fp32,
    topk_ids [T, 8] int32)`, both new and contiguous, the trtllm op's dtypes.

    One warp per token, eight experts per lane: the ids are exact. The weights
    normalize in fp32 where the trtllm op normalizes in fp64, so they can differ
    from it in the last fp32 bits.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The chunks that reach it: the MTP layer's above 64 tokens (the fused decode
    # front end covers up to 64), and the trunk's when its prefill top-k did not
    # build.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the first token count past the fused decode front end",
            spec=dict(**_GLM, tokens=65),
        ),
        Cell(
            why="a count that leaves the last 8-token CTA partly empty",
            spec=dict(**_GLM, tokens=1037),
        ),
        Cell(why="a prefill chunk at max_num_tokens", spec=dict(**_GLM, tokens=8192)),
        Cell(
            why="equal selection scores: the lower expert id must win",
            spec=dict(**_GLM, tokens=64, ties=True),
        ),
    )

    note = """
    The target compiles one handle at load and takes this op wherever `takes`
    holds; every other routing goes to `noaux_tc_op` at the call site. The
    kernel checks dtypes and shapes on the host and raises.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
        compiled: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.sigmoid_topk(compiled, router_logits, bias, routed_scaling_factor)

    def reference(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
        compiled: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native torch: the tanh-form sigmoid in fp32 (the selection compares fp32
        scores, so a reference that rounds them differently picks differently on
        near-ties), a stable descending sort for the ids, and the renormalization in
        fp64. `n_group`, `topk_group` and `compiled` are ignored (one group)."""
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        choice = scores + bias.float()
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
        weights = torch.gather(scores, 1, ids).double()
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * routed_scaling_factor
        return weights.float(), ids.to(torch.int32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
        compiled: Callable,
    ) -> None:
        # The kernel is built for one group and top-8 and reads neither argument:
        # anything else returns an ungrouped top-8.
        assert (n_group, topk_group, topk) == (1, 1, TOP_K), (
            f"one group, top-{TOP_K} only; got n_group={n_group} topk_group={topk_group} "
            f"topk={topk}, which the kernel would route as ungrouped top-{TOP_K}"
        )


cute_dsl_moe_sigmoid_topk = CuteDslMoeSigmoidTopk()
