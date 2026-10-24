# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-expert combine: final_hidden_states + sigmoid(gate) * shared_expert_output."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# No torch.ops boundary exists around this kernel: the module callable is the
# stable surface, and its body is one Triton launch after argument checks.
from tensorrt_llm._torch.moe.fused_shared_expert import (
    fused_sigmoid_gate_mul_add as _fused_sigmoid_gate_mul_add,
)


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
    )

    note = """
    Metadata consumed: none; stateless, no autotuner, no architecture branch.
    The Triton kernel JIT-compiles on first call.

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

    Hard limits
      num_tokens  row offset widened to int64 in the kernel; largest run 8192

    Not certified: hidden sizes other than 2560 (the column block and warp count
    step with it), gate layouts other than `[T, 1]` and the strided column, an
    `output` buffer (including one aliasing `shared_expert_output`), row-padded
    operands, host operands, CUDA-graph capture, torch.compile, operands on two
    devices, float8 activations.
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
