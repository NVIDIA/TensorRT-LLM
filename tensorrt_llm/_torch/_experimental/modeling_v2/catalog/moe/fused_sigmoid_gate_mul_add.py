# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-expert combine: final_hidden_states + sigmoid(gate) * shared_expert_output."""

from __future__ import annotations

from typing import Optional

import torch

# The module callable is the surface: outside torch.compile its body is one Triton
# launch after argument checks. `torch.ops.trtllm.fused_sigmoid_gate_mul_add_` (same
# module) is only its torch.compile boundary, which the callable takes itself when
# compiling, with a different signature (mutated `output`, separate addend).
from tensorrt_llm._torch.moe.fused_shared_expert import (
    fused_sigmoid_gate_mul_add as _fused_sigmoid_gate_mul_add,
)

from .._op import Arch, Cell, OpWrapper


def _ckpt(tokens: int, **extra) -> dict:
    """The checkpoint's MoE combine: hidden 2560 bf16, one gate logit per token, in place."""
    spec = dict(
        tokens=tokens,
        hidden=2560,
        dtype=torch.bfloat16,
        gate_dtype=torch.bfloat16,
        gate_layout="column",
        dest="inplace",
    )
    spec.update(extra)
    return spec


class FusedSigmoidGateMulAdd(OpWrapper):
    """`out[t, i] = final[t, i] + sigmoid(gate[t]) * shared[t, i]`, fp32 inside, rounded once.

    Wraps `tensorrt_llm._torch.moe.fused_shared_expert.fused_sigmoid_gate_mul_add`
    (one Triton kernel, `_sigmoid_gate_mul_add_kernel`).

    `gate_logits` is one pre-sigmoid scalar per token read as a flat vector of
    `num_tokens`: `[T, 1]`, `[T]`, `[1, T]` and a strided column are the same
    call, bit for bit, and its dtype is independent of the activations'.

    Destination: with `output=None` the result overwrites `final_hidden_states`
    and that tensor is returned; otherwise `output` is written and returned and
    both inputs are untouched. Never a fresh tensor.

    Fusion boundary: sigmoid, broadcast, multiply and add. The routed MoE, the
    shared expert and the `[hidden, 1]` gate projection are the caller's. The
    single rounding is the point: a sigmoid-mul then add rounds the product to
    bf16 first and moves the output by up to 0.0312 at [17, 2560] (51.6x the
    bf16 tolerance).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="_moe_block / MTPLayer._moe decode: one token, in place; 8 warps",
            spec=_ckpt(1),
        ),
        Cell(why="2048 tokens: prefill-shaped", spec=_ckpt(2048)),
        Cell(
            why="gate as a strided column of a wider tensor, as the trunk's fused shared-expert "
            "GEMM leaves it",
            spec=_ckpt(33, gate_layout="strided"),
        ),
        # The unified target's call sites, the decode PLE-layer combine and the MTP layer's
        # MoE, on the gate the shared expert's fused GEMM leaves: column 1280 of the
        # replicated [T, 1288] `gug` output.
        Cell(
            why="unified decode PLE-layer / MTP MoE combine: one token, strided gate",
            spec=_ckpt(1, gate_layout="strided"),
        ),
        Cell(
            why="unified tp4/tep4 decode batch at the largest captured size: 128 rows",
            spec=_ckpt(128, gate_layout="strided"),
        ),
        Cell(
            why="unified tep4 MTP-7 at batch 128: 128 x 8 = 1024 rows, strided gate",
            spec=_ckpt(1024, gate_layout="strided"),
        ),
        Cell(
            why="the envelope's largest row: max_num_tokens 8192, strided gate",
            spec=_ckpt(8192, gate_layout="strided"),
        ),
    )

    note = """
    Metadata consumed: none; stateless, no autotuner, no architecture branch.
    Source: a Triton kernel (`_sigmoid_gate_mul_add_kernel`) JIT-compiled on first
    call for the running device; the cells ran that JIT path on GB300 (sm_103).

    Call sites in the unified target: the decode PLE-layer combine
    (`DecodeTarget.forward`) and the MTP layer's MoE (`DecodeTarget.mtp_forward`),
    in place on the routed output. The shared expert is replicated at every member,
    so the gate is column 1280 of this rank's `[T, 1288]` `gug` output (gate|up 2 x
    640 plus 8 gate-logit rows) and hidden is 2560 everywhere. Rows are this rank's
    tokens: decode 1..128 at tp4 / tep4 (1..32 a dep4 rank's own), MTP-7 at batch
    128 up to 128 x 8 = 1024. Every row count takes one path: hidden 2560 gives
    one 4096-wide block and 8 warps per row, grid `(T, 1)`.

    Preconditions, all enforced loudly by the callable (AssertionError before
    any launch; the test holds the wrong-length gate): `final_hidden_states`
    2-D; `shared_expert_output` and `output` match it in shape and dtype;
    `gate_logits.numel() == num_tokens`; the destination has `stride(-1) == 1`
    (its row stride is free). A column-strided `final_hidden_states` or
    `shared_expert_output` is densified inside the call (an extra copy kernel),
    as is a non-contiguous gate. `num_tokens == 0` returns the destination
    without launching (certified).

    Numerics: the default `assert_close` band holds for bf16, the dtype the target
    calls it with; fp16 and fp32 run the same kernel and are not certified.
    bf16 at hidden 2560 is bitwise equal at 1-2 tokens and >= 99.993% bitwise
    at 17-8192 tokens, one ulp otherwise -- `tl.sigmoid`'s fast exponential.
    Wrong variants land 29,000-152,000x outside the bf16 band: no sigmoid,
    multiply instead of add, gate on the routed half, gate dropped, one gate
    for the whole batch.
    Measured at the unified cells (strided `gug` gate, randn operands) against
    `reference`: bitwise at 1 row; 8 of 327,680 elements differ at 128 rows, 64 of
    2,621,440 at 1024, 564 of 20,971,520 at 8192. Wherever |out| >= 2^-6 each
    difference is exactly one bf16 ulp of the output; the rest sit where the sum
    cancels toward zero, the largest 1.6e-2 absolute at 8192 rows (one ulp of a
    [2, 4) value). Against a float64 reference the relative L2 is 1.66e-3 at
    every cell, equal to `reference`'s own (bf16 output rounding); cosine
    0.9999986. A strided `gug` column and the same logits as a dense `[T, 1]` are
    bitwise one call (held at 1024 rows). Graph-capturable: one capture at the
    128-row site, replayed on refilled operands, matches `reference` (held).

    Hard limits
      num_tokens  row offset widened to int64 in the kernel; largest run 8192
                  (max_num_tokens). Column offsets are int32 within one row; the
                  first int32 element-index overflow, 2^31 / 2560 = 838,860 rows,
                  is past the envelope and would not occur (int64 row offset).

    Not certified: hidden sizes other than 2560 (the column block and warp count
    step with it), gate layouts other than `[T, 1]` and the strided column, an
    `output` buffer (including one aliasing `shared_expert_output`), row-padded
    operands, host operands, torch.compile, operands on two devices, float8
    activations, gate dtypes other than bf16, rows past 8192.
    """

    def __call__(
        self,
        final_hidden_states: torch.Tensor,
        gate_logits: torch.Tensor,
        shared_expert_output: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _fused_sigmoid_gate_mul_add(
            final_hidden_states,
            gate_logits,
            shared_expert_output,
            output,
        )

    def reference(
        self,
        final_hidden_states: torch.Tensor,
        gate_logits: torch.Tensor,
        shared_expert_output: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A fresh tensor in `final_hidden_states.dtype`; `output` is ignored.

        Takes the operands as they were before the call: in place, the op
        destroys `final_hidden_states`.
        """
        s = torch.sigmoid(gate_logits.reshape(-1, 1).float())
        out = final_hidden_states.float() + s * shared_expert_output.float()
        return out.to(final_hidden_states.dtype)


fused_sigmoid_gate_mul_add = FusedSigmoidGateMulAdd()
