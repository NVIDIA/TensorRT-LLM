# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place sigmoid output gating: ``attention_output *= sigmoid(gate)``."""

from __future__ import annotations

import torch

# Registered by the module that defines it: after `import
# tensorrt_llm._torch.custom_ops` alone, torch.ops.trtllm has no
# `fused_sigmoid_mul_inplace` (measured).
import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The gated attention: 24 q heads of 256, the gate the 3-D view of the interleaved
#: `[T, 14336]` qkv projection (strides (14336, 512, 1)).
_QKV_VIEW = dict(heads=24, head_dim=256, dtype=torch.bfloat16, gate="qkv_view")


class FusedSigmoidMulInplace(OpWrapper):
    """`attention_output[t, i] *= sigmoid(gate[t, i // D, i % D])` in place, `D = gate.shape[-1]`.

    Returns `attention_output`, the tensor it wrote. Both operands are widened to
    fp32 and the product is rounded once to `attention_output.dtype`. `gate` is
    read through its own token and head strides, so the gate halves of the
    interleaved qkv projection are passed as a strided view with no repack.

    Fusion boundary: sigmoid and multiply only. The caller owns producing both
    operands and everything downstream (the o projection). `gate` is read-only.
    """

    ARCHS = frozenset({Arch.SM_120})

    # At decode rows up to 32 the gate is the 3-D view the rope-gate prologue hands
    # back -- the g_i halves of the interleaved [T, 14336] qkv projection.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode 1 token: gate the [1, 24, 256] view of the qkv projection",
            spec=dict(tokens=1, **_QKV_VIEW),
        ),
        Cell(
            why="decode 32 tokens: token stride 14336, head stride 512",
            spec=dict(tokens=32, **_QKV_VIEW),
        ),
    )

    note = """
    Metadata consumed: none. One Triton kernel, JIT-compiled on the first call,
    no autotuner: grid `(T, cdiv(hidden, block))`, `block = min(next_pow2(hidden),
    1024 if T < 1024 else 2048)`, 4 warps.

    Preconditions, every one enforced loudly by the op's own bare `assert`
    (an AssertionError with no message): `attention_output` 2-D with
    `stride(-1) == 1` (row stride free); `gate` 2-D of `attention_output`'s
    shape, or 3-D with `shape[0] == T` and `shape[1] * shape[2] == hidden`,
    `stride(-1) == 1`. A CPU tensor raises ValueError from Triton (measured).
    Hence no `is_valid`. Cross-device operands were not probed.

    The pre-call contents of `attention_output` are destroyed; the gate is
    left bitwise unchanged.

    Numerics: bf16 bitwise equal to the fp32 reference except ~1e-6 of
    elements one ulp off (`tl.sigmoid`'s fast exp); default tolerance holds,
    nothing loosened.

    Certified: bf16 operands at hidden 6144, a compact `attention_output`, and
    the 3-D gate view of the interleaved `[T, 14336]` qkv projection at 1 and 32
    tokens.

    CUDA-graph replay is certified for the qkv view at 32 tokens (the replay reads
    the operands' contents at replay time).

    Not certified: 0 tokens, more than 32 tokens, a 2-D gate, a strided
    `attention_output`, a gate dtype other than `attention_output.dtype`,
    fp16/fp32 operands, other hidden sizes, torch.compile.
    """

    def __call__(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.fused_sigmoid_mul_inplace(attention_output, gate)
        return attention_output

    def reference(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """What `attention_output` holds after the call; neither input is mutated.

        `gate.reshape(T, hidden)` is the per-head `(t, i // D, i % D)` read, for
        the 2-D and 3-D forms alike.
        """
        num_tokens, hidden = attention_output.shape
        g = gate.reshape(num_tokens, hidden).float()
        return (attention_output.float() * torch.sigmoid(g)).to(attention_output.dtype)


fused_sigmoid_mul_inplace = FusedSigmoidMulInplace()
