# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renormalized MoE routing: top-k over the raw router logits, then a softmax
over just those k, in one trtllm call."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _ckpt(tokens: int, output_dtype: torch.dtype = torch.bfloat16) -> dict:
    """The checkpoint's router: 512 experts, top-10, bf16 logits."""
    return dict(
        tokens=tokens,
        num_experts=512,
        topk=10,
        logits_dtype=torch.bfloat16,
        output_dtype=output_dtype,
        dist="randn",
    )


class RenormMoeRoutingOp(OpWrapper):
    """`norm_topk_prob=True` routing. Returns `(topk_ids, topk_weights)` -- IDS FIRST.

    Per token: `ids` = the `topk` largest `router_logits`, ranked in the
    logits' own dtype, by decreasing value, ties to the LOWER expert index
    (a stable descending sort -- `torch.topk` breaks ties the other way, and at
    512 bf16 experts ties are common: it disagrees on ~30% of rows);
    `weights` = an fp32 softmax over just those `topk` logits, cast to
    `output_dtype` afterwards. Equal to HF's softmax-then-top-k-then-divide up
    to rounding (5.96e-08 at 512 / top-10).

    `ids` is `[T, topk]` int32; `weights` `[T, topk]` in `output_dtype`, which
    defaults to fp32 -- not the logits' dtype. Sibling routing ops return
    `(weights, ids)`; both are `[T, topk]`, so a swapped unpack is no shape
    error.

    Fusion boundary: routing only. The router GEMM is the caller's, and so is
    expert parallelism -- every column of the row is ranked.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="_moe_block / MTPLayer._moe decode: one token, bf16 weights", spec=_ckpt(1)),
        Cell(why="7 tokens: an odd count", spec=_ckpt(7)),
        Cell(why="2048 tokens: prefill-shaped", spec=_ckpt(2048)),
        Cell(
            why="4096 tokens: thousands of rows carry an exact bf16 tie in the top-10",
            spec=_ckpt(4096),
        ),
        Cell(why="8192 tokens: max_num_tokens", spec=_ckpt(8192)),
        Cell(
            why="the default fp32 output dtype at the checkpoint router",
            spec=_ckpt(512, torch.float32),
        ),
        Cell(
            why="fp32 logits, bf16 out: the cast must follow the fp32 softmax",
            spec=dict(_ckpt(1024), logits_dtype=torch.float32, magnitude=3.0),
        ),
        Cell(
            why="fp16 logits: the third accepted input dtype",
            spec=dict(_ckpt(256), logits_dtype=torch.float16, magnitude=2.0),
        ),
        Cell(
            why="coarse logits (3 values): ties decide most of the ranking",
            spec=dict(_ckpt(64), output_dtype=torch.float32, dist="coarse"),
        ),
        Cell(
            why="topk 16: the op's ceiling",
            spec=dict(_ckpt(256), topk=16),
        ),
        Cell(why="topk 1: a single slot at weight exactly 1", spec=dict(_ckpt(256), topk=1)),
        Cell(
            why="33 experts: one past a warp's 32 lanes",
            spec=dict(_ckpt(64), num_experts=33, output_dtype=torch.float32),
        ),
        Cell(
            why="129 experts: past 128, where the block grows to 256 threads",
            spec=dict(_ckpt(64), num_experts=129, output_dtype=torch.float32),
        ),
        Cell(
            why="topk == num_experts (8 of 8): the full ranking",
            spec=dict(_ckpt(16), num_experts=8, topk=8, output_dtype=torch.float32),
        ),
    )

    note = """
    Metadata consumed: none; stateless, deterministic, runs on the ambient
    stream. The launch picks its kernel instance by
    `nextPowerOfTwo(num_experts)` and `nextPowerOfTwo(topk)` -- 512 / top-10 is
    the (512, 16) instance with a 256-thread block. PDL (`TRTLLM_ENABLE_PDL`)
    was measured bitwise inert.

    Preconditions, enforced loudly by the op: `router_logits` 2-D CUDA bf16 /
    fp16 / fp32; `output_dtype` fp32, bf16 or None (fp16 raises even for fp16
    logits); `topk <= 16`; `num_experts <= 512` -- this checkpoint sits on
    that ceiling. `topk == 0` and `num_tokens == 0` return empties.

    Silent, refused by `is_valid`: a non-contiguous `router_logits` (read as a
    dense `[T, E]` buffer from `data_ptr()`) and `topk > num_experts` (the
    surplus slots come back as expert 0 at weight 0 -- duplicated ids).

    Non-finite logits poison the whole row to NaN (`+inf`, `NaN`, an all
    `-inf` row); a `-inf` logit among finite ones is a soft mask, at weight 0
    once `topk` exceeds the finite experts.

    `torch.ops.trtllm.default_moe_routing_op` has the identical signature and
    softmaxes before the top-k: same ids, un-renormalized weights (rows sum to
    0.09-0.27 here). `RenormalizeMoeRoutingMethod`'s PyTorch fallback (used past
    512 / 16) is not bit-compatible: it uses `torch.topk` and casts before the
    softmax.

    Bit-exact against `reference` over 2040 configurations (in dtype x out
    dtype x num_experts {1..512} x topk {0..16} x tokens {0..8192}); the cells
    and the test's sweep re-drive the boundaries of that space.

    Hard limits
      num_experts  <= 512 (raises past it)
      topk         <= 16 (raises past it); > num_experts silent, guarded
      T            certified 0..8192 (16384 measured once)
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.renorm_moe_routing_op(router_logits, topk, output_dtype)

    def reference(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable descending sort in the logits' dtype, fp32 softmax over the
        selected k, then the cast."""
        order = torch.sort(router_logits, dim=-1, descending=True, stable=True).indices[:, :topk]
        selected = torch.gather(router_logits, 1, order).float()
        e = torch.exp(selected - selected.max(-1, keepdim=True).values)
        weights = e / e.sum(-1, keepdim=True)
        return order.to(torch.int32), weights.to(output_dtype or torch.float32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        # The kernel addresses router_logits from data_ptr() as dense
        # [T, num_experts]: a column slice routes on the wider buffer's bytes.
        assert router_logits.is_contiguous(), (
            "router_logits must be contiguous; a strided view is read as a dense "
            "[num_tokens, num_experts] buffer and silently gives wrong routing"
        )
        # Surplus slots come back as duplicated expert 0 at weight 0.
        assert topk <= router_logits.shape[-1], (
            f"topk ({topk}) exceeds num_experts ({router_logits.shape[-1]}); the "
            "surplus slots come back as duplicates of expert 0 at weight 0"
        )


renorm_moe_routing_op = RenormMoeRoutingOp()
