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

#: The checkpoint's gated attention: 24 q heads of 256, q projection [T, 24, 512].
_CKPT = dict(heads=24, head_dim=256, dtype=torch.bfloat16, gate="interleaved")


class FusedSigmoidMulInplace(OpWrapper):
    """`attention_output[t, i] *= sigmoid(gate[t, i // D, i % D])` in place, `D = gate.shape[-1]`.

    Returns `attention_output`, the tensor it wrote. Both operands are widened to
    fp32 and the product is rounded once to `attention_output.dtype`. `gate` is
    read through its own token and head strides, so the gate half of an
    interleaved `[T, heads, 2 * D]` q projection is passed as a strided view with
    no repack (bitwise equal to its compacted copy, measured). A 2-D `gate` of
    `attention_output`'s shape is one head spanning the row.

    Fusion boundary: sigmoid and multiply only. The caller owns producing both
    operands and everything downstream (the o projection). `gate` is read-only.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: 1 token, 24x256 gate as the strided half of q_proj",
            spec=dict(tokens=1, **_CKPT),
        ),
        Cell(
            why="17 tokens: several rows, 1024-element blocks over 6144",
            spec=dict(tokens=17, **_CKPT),
        ),
        Cell(
            why="prefill 2048 tokens: >= 1024 rows selects 2048-element blocks",
            spec=dict(tokens=2048, **_CKPT),
        ),
        Cell(
            why="attention_output a [33, 6144] window of a [35, 8192] buffer",
            spec=dict(tokens=33, out="window", **_CKPT),
        ),
        Cell(
            why="attention_output every other row: row stride 2 * hidden",
            spec=dict(tokens=32, out="every_other_row", **_CKPT),
        ),
        Cell(
            why="2-D gate: one head spanning the whole row",
            spec=dict(tokens=9, heads=1, head_dim=6144, dtype=torch.bfloat16, gate="flat"),
        ),
        Cell(
            why="gate fp32 under a bf16 output: widened, not rejected",
            spec=dict(
                tokens=9,
                heads=1,
                head_dim=512,
                dtype=torch.bfloat16,
                gate="flat",
                gate_dtype=torch.float32,
            ),
        ),
        Cell(
            why="fp16 operands",
            spec=dict(tokens=33, heads=4, head_dim=128, dtype=torch.float16, gate="packed"),
        ),
        Cell(
            why="fp32 operands",
            spec=dict(tokens=33, heads=4, head_dim=128, dtype=torch.float32, gate="packed"),
        ),
        Cell(
            why="hidden 16: a single block narrower than any vector",
            spec=dict(tokens=1, heads=4, head_dim=4, dtype=torch.bfloat16, gate="packed"),
        ),
        Cell(
            why="hidden 100: not a multiple of any vector width",
            spec=dict(tokens=3, heads=4, head_dim=25, dtype=torch.bfloat16, gate="packed"),
        ),
        Cell(
            why="hidden 2049 at 1030 rows: 2048-element blocks plus a masked tail",
            spec=dict(tokens=1030, heads=3, head_dim=683, dtype=torch.bfloat16, gate="packed"),
        ),
        Cell(
            why="hidden 6145: odd, multi-block, masked tail",
            spec=dict(tokens=4, heads=1, head_dim=6145, dtype=torch.bfloat16, gate="packed"),
        ),
        Cell(
            why="0 tokens: returns without launching",
            spec=dict(tokens=0, **_CKPT),
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
    `stride(-1) == 1`. A CPU tensor raises ValueError from Triton. Hence no
    `is_valid`. Cross-device operands were not probed.

    `gate.dtype` need not equal `attention_output.dtype`: an fp32 gate under a
    bf16 output returns the fp32-gate result. The pre-call contents of
    `attention_output` are destroyed; writes stay inside its `[T, hidden]`
    window (padded columns and trailing rows bitwise unchanged).

    Numerics: bf16/fp16 bitwise equal to the fp32 reference except ~1e-6 of
    elements one ulp off (`tl.sigmoid`'s fast exp); default tolerance holds,
    nothing loosened. Feeding the q half instead of the gate half moves the
    result by ~2.15 max abs at [17, 6144], so the per-head indexing is under
    test.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      (largest run 2048 rows)

    Not certified: CUDA-graph capture, torch.compile.
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
