# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renormalized MoE routing: top-k over the raw router logits, then a softmax
over just those k, in one trtllm call.

This target's local variant of
``qwen3_8_flash_next_nvfp4__sm_103__tp1/ops/moe/renorm_moe_routing_op``, copied
unchanged."""

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

    # The trunk pre-routes the trtllm-gen FP4 runner with bf16 weights at every
    # member: tp4/tep4 route every row the rank holds; dep4 routes the
    # all-gathered rows of the 4 ranks, then runs its expert window in chunks of
    # at most 8192 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTPLayer._moe decode: one token, bf16 weights; the trunk's decode at batch 1",
            spec=_ckpt(1),
        ),
        Cell(
            why="the trunk's decode at the largest captured batch: 128 rows at tp4/tep4, "
            "4 x 32 gathered at dep4, bf16 weights",
            spec=_ckpt(128),
        ),
        Cell(
            why="the trunk's MTP7 verify at tep4: 128 requests x 8 rows, bf16 weights",
            spec=_ckpt(1024),
        ),
        Cell(
            why="the trunk's prefill chunk at max_num_tokens 8192, bf16 weights",
            spec=_ckpt(8192),
        ),
        Cell(
            why="dep4 prefill over the gathered rows, 4 ranks x max_num_tokens 8192: "
            "the envelope's largest call, bf16 weights",
            spec=_ckpt(32768),
        ),
        Cell(
            why="_moe_block at 16384 rows (max_num_tokens of the tuned config): fp32 weights, "
            "as the CuteDSL experts take them; thousands of rows carry an exact bf16 tie",
            spec=_ckpt(16384, torch.float32),
        ),
        Cell(
            why="coarse logits (3 values): ties decide most of the ranking",
            spec=dict(_ckpt(64), output_dtype=torch.float32, dist="coarse"),
        ),
    )

    note = """
    Source: compiled-in C++, no JIT path. `torch_ext::renorm_moe_routing_op`
    in `cpp/tensorrt_llm/thop/moe/routing/customMoeRoutingOp.cpp` (a
    `TORCH_LIBRARY_FRAGMENT(trtllm)` schema) launches `customMoeRoutingKernel`
    from `cpp/tensorrt_llm/kernels/moe/routing/customMoeRoutingKernels.cu`.

    Metadata consumed: none; stateless, deterministic, runs on the ambient
    stream. The launch picks its kernel instance by
    `nextPowerOfTwo(num_experts)` and `nextPowerOfTwo(topk)` -- 512 / top-10 is
    the (512, 16) instance with a 256-thread block. PDL (`TRTLLM_ENABLE_PDL`)
    was measured bitwise inert.

    One warp per row, `ceil(T / 8)` blocks capped at 8192
    (`invokeCustomMoeRouting`), so the row count changes only the grid: every
    certified count is under the cap (32768 rows = 4096 blocks), and the
    grid-stride loop past 65536 rows is not exercised.

    The trunk's bf16-weight cells, measured on GB300: ids equal and weights
    bit-exact at 1 / 128 / 1024 / 8192 / 32768 rows. On their randn logits the
    10th and 11th largest logits tie on 0 / 13 / 142 / 1078 / 4496 rows and
    `torch.topk`'s ids differ on 0 / 36 / 317 / 2501 / 9886 rows, so the
    ties-to-lower-index rule is exercised at every multi-row cell. The bf16
    weights of a row sum to 1 within 2.44e-3 (the rounding of ten weights).
    CUDA-graph replay at 128 rows is bit-exact against the reference on
    refilled logits (the test holds it).

    Preconditions, enforced loudly by the op: `router_logits` 2-D CUDA bf16 /
    fp16 / fp32; `output_dtype` fp32, bf16 or None (fp16 raises even for fp16
    logits); `topk <= 16`; `num_experts <= 512` -- this checkpoint sits on
    that ceiling.

    Silent, refused by `is_valid`: a non-contiguous `router_logits` (read as a
    dense `[T, E]` buffer from `data_ptr()`; the test holds this refusal) and
    `topk > num_experts` (the surplus slots come back as expert 0 at weight 0
    -- duplicated ids).

    `torch.ops.trtllm.default_moe_routing_op` has the identical signature and
    softmaxes before the top-k: same ids, un-renormalized weights (rows sum to
    0.09-0.27 here). `RenormalizeMoeRoutingMethod`'s PyTorch fallback (used past
    512 / 16) is not bit-compatible: it uses `torch.topk` and casts before the
    softmax.

    Bit-exact against `reference` over 2040 configurations at bring-up (in
    dtype x out dtype x num_experts {1..512} x topk {0..16} x tokens
    {0..8192}); the cells hold the target's router (512 / top-10, bf16 logits,
    bf16 and fp32 weights), and the test holds the weights to HF's
    `norm_topk_prob` formulation at bf16 logits.

    Hard limits
      num_experts  <= 512 (raises past it)
      topk         <= 16 (raises past it); > num_experts silent, guarded
      T            certified 1..32768 with bf16 weights (dep4's gathered
                   prefill, 4 x max_num_tokens 8192), 16384 with fp32 weights.
                   From the source, not probed: `numTokens` is narrowed to
                   int32 and the row offset `tokenId * numExperts` is uint32,
                   which wraps at 2^32 / 512 = 8388608 rows -- 256x the
                   envelope.

    Not certified: other num_experts / topk / dtype combinations, `topk == 0`
    and `num_tokens == 0` (both return empties), non-finite logits (`+inf`,
    NaN and an all `-inf` row poison the row to NaN; a `-inf` among finite
    logits is a soft mask). Row counts between the cells -- dep4's smallest
    gathered decode (4 rows), the MTP3 verify (128 x 4) -- run the same kernel
    instance on a different grid size and are not cells. Router logits from
    the checkpoint's own router GEMM were not driven; the cells use randn and
    coarse logits.
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
