# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer entered with router logits, left unfinalized: the trtllm-gen
runner's Renormalize routing + grouped FC1 GEMM + SwiGLU + NVFP4
requantization + grouped FC2 GEMM, in one call, returning the permuted FC2 rows
with the routing weights and the expanded->permuted index map instead of the
combined `[T, H]` rows. The consumer forms the top-k weighted sum itself."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .fp4_block_scale_moe_renorm_routed import ACT_SWIGLU, ROUTING_RENORMALIZE, _ckpt
from .fp4_block_scale_moe_renorm_routed import fp4_block_scale_moe_renorm_routed as _finalized


def finalize(
    rows: torch.Tensor, topk_weights: torch.Tensor, topk_rows: torch.Tensor
) -> torch.Tensor:
    """The runner's finalize, emulated: per token `sum_k w[t, k] * rows[topk_rows[t, k]]`
    as fp32 FMAs in slot order from zero (a -1 slot skipped), rounded once to the
    rows' dtype. Each fp32 FMA is formed exactly in float64 (a product of two
    fp32 values is exact there) and rounded to fp32. Returns `[T, H]`."""
    acc = torch.zeros(topk_rows.shape[0], rows.shape[1], dtype=torch.float32, device=rows.device)
    for k in range(topk_rows.shape[1]):
        idx = topk_rows[:, k].long()
        r = rows[idx.clamp(min=0)].float().double()
        w = topk_weights[:, k].float().double().unsqueeze(-1)
        step = (r * w + acc.double()).float()
        acc = torch.where((idx >= 0).unsqueeze(-1), step, acc)
    return acc.to(rows.dtype)


class Fp4BlockScaleMoeRenormPermuted(OpWrapper):
    """`fp4_block_scale_moe_renorm_routed` with `do_finalize=False`.

    Same operands, same routing, same GEMMs; the runner skips its finalize
    launch and returns three tensors:

    - `rows` `[P, H]` bf16: the FC2 output in the runner's permuted, padded row
      order (`P` is the runner's padded maximum; rows no slot maps to hold
      garbage);
    - `topk_weights` `[T, k]` bf16: the routing weights, slot order;
    - `topk_rows` `[T, k]` int32: slot `(t, k)`'s row in `rows`, or -1.

    `finalize(rows, topk_weights, topk_rows)` (this module) is bitwise the
    finalized entry's output on the same operands.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="_moe_block decode, finalize folded into the next site: one token; the block "
            "routing kernel",
            spec=_ckpt(1),
        ),
        Cell(
            why="4 rows (MTP3 verify; 2..16-row batches): finalize folded into "
            "qwen4_exp_hc_decode_rows_first",
            spec=_ckpt(4),
        ),
        Cell(
            why="128 rows: the con=128 decode batch; the cluster routing kernel",
            spec=_ckpt(128),
        ),
        Cell(why="coarse 3-level logits: ties decide most slots", spec=_ckpt(64, dist="coarse")),
    )

    note = """
    Measured against the finalized entry on identical operands: the emulated
    finalize of the three outputs is bitwise its output at every cell, so the
    routing semantics, the GEMMs and the weight layout are the finalized
    entry's (see its note). Which finalize kernel the runner would have
    launched (scalar below ~1184 blocks, vectorized above) does not change the
    arithmetic: both accumulate fp32 FMAs in slot order and round once.

    `rows` is the runner's own `[P, H]` buffer, allocated per call at the
    padded maximum the runner computes from `T`, `k` and the expert count;
    only rows some slot maps to hold data. `topk_weights` carries the output
    dtype (bf16), as the finalize reads it.

    Silent, refused by `is_valid`: the finalized entry's refusals (a
    non-contiguous operand, an uncertified (num_experts, top_k) pair, hidden
    not a multiple of 256).

    Hard limits
      T            certified at 1, 4, 64 and 128 (decode); the decode forward folds at
                   every site that takes the combine
      num_experts  512, top_k 10
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
    ) -> tuple:
        rows, topk_weights, topk_rows = torch.ops.trtllm.fp4_block_scale_moe_runner(
            routing_logits,
            None,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            None,
            None,
            None,
            None,
            gemm2_weights,
            gemm2_weights_scale,
            None,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            0,
            num_experts,
            None,
            ROUTING_RENORMALIZE,
            False,
            ACT_SWIGLU,
        )
        return rows, topk_weights, topk_rows

    def reference(self, *args, **kwargs) -> torch.Tensor:
        """The finalized entry's fp32 reference `[T, H]`: what
        `finalize(*self(...))` must answer (same arguments as that entry's
        `reference`)."""
        return _finalized.reference(*args, **kwargs)

    def is_valid(self, *args, **kwargs) -> None:
        _finalized.is_valid(*args, **kwargs)


fp4_block_scale_moe_renorm_permuted = Fp4BlockScaleMoeRenormPermuted()
