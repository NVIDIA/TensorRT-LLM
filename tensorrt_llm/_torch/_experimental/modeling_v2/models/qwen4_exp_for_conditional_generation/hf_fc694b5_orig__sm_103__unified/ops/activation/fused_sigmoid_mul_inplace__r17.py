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

#: The unified target at tp4 / tep4: a rank's fused input projection row is
#: [q 6 x 256 | k 256 | v 256 | gate 6 x 256 | index scratch 768] = 4352 columns,
#: so the gate is the 1536-wide window at column qkv_width = 1536 + 2 * 256 = 2048.
_TP4_ROW = dict(
    heads=6,
    head_dim=256,
    dtype=torch.bfloat16,
    gate="fused_row",
    row_stride=4352,
    gate_col=2048,
)

#: The unified target at dep4: tp1's widths, 14080 columns, the gate at 7168.
_DEP4_ROW = dict(
    heads=24,
    head_dim=256,
    dtype=torch.bfloat16,
    gate="fused_row",
    row_stride=14080,
    gate_col=7168,
)


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
            why="decode 128 tokens: 2-D gate off the fused projection row (stride 14080)",
            spec=dict(tokens=128, heads=24, head_dim=256, dtype=torch.bfloat16, gate="fused_row"),
        ),
        Cell(
            why="prefill 8192 tokens: 2-D gate off the fused projection row, 2048-wide blocks",
            spec=dict(tokens=8192, heads=24, head_dim=256, dtype=torch.bfloat16, gate="fused_row"),
        ),
        Cell(
            why="MTP decode: 1 token, 24x256 gate as the strided half of q_proj",
            spec=dict(tokens=1, **_CKPT),
        ),
        # The unified target. Hidden 1536 is not a multiple of the kernel's block,
        # so every tp4 / tep4 row ends in 512 masked lanes: two 1024-wide blocks
        # below 1024 rows, one 2048-wide block from 1024 rows.
        Cell(
            why="unified tp4/tep4 decode, one token: 6 heads off the 4352-wide row",
            spec=dict(tokens=1, **_TP4_ROW),
        ),
        Cell(
            why="unified tp4/tep4 largest captured decode batch: 128 rows, 6 heads",
            spec=dict(tokens=128, **_TP4_ROW),
        ),
        Cell(
            why="unified tep4 MTP-7 verify 128 x 8 = 1024 rows: the first 2048-wide block count",
            spec=dict(tokens=1024, **_TP4_ROW),
        ),
        Cell(
            why="unified tp4/tep4 prefill chunk at max_num_tokens: 8192 rows, 6 heads",
            spec=dict(tokens=8192, **_TP4_ROW),
        ),
        Cell(
            why="unified dep4 rank's largest captured decode batch: 32 rows, 24 heads",
            spec=dict(tokens=32, **_DEP4_ROW),
        ),
    )

    note = """
    Metadata consumed: none. Source: `torch.ops.trtllm.fused_sigmoid_mul_inplace`,
    a Python `torch.library.custom_op` registered by
    `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py` around the
    Triton kernel `_fused_sigmoid_mul_kernel` there (adapted from SGLang, per that
    file's header). JIT-compiled for the running device at the first call of each
    (hidden, head_dim, block) specialization, no autotuner: grid
    `(T, cdiv(hidden, block))`, `block = min(next_pow2(hidden), 1024 if T < 1024
    else 2048)`, 4 warps. The cells ran that JIT path on GB300 (sm_103).

    Call sites (the unified target): the attention output gate after the
    attention core and before o_proj, in `DecodeTarget._attention_block`
    (decode, 1..128 rows at tp4 / tep4, 1..32 at a dep4 rank), its MTP verify
    (up to 128 x 8 = 1024 rows, tep4, draft 7) and
    `PrefillTarget._attention_block` (one chunk of up to 8192 rows,
    `max_num_tokens`). `gate` is the 2-D window of the fused input projection
    row at column qkv_width: at tp4 / tep4 the row is 4352 wide
    (q 1536 | k 256 | v 256 | gate 1536 | index scratch 768), the gate at
    column 2048 and hidden 1536; at dep4 tp1's 14080-wide row, gate at 7168,
    hidden 6144. Hidden 1536 is no multiple of the block, so every row ends in
    512 masked lanes (two 1024-wide blocks below 1024 rows, one 2048-wide block
    from 1024 rows); the test places `attention_output` at the head of a
    larger buffer and holds the elements past its last row unchanged.

    Preconditions, every one enforced loudly by the op's own bare `assert`
    (an AssertionError with no message): `attention_output` 2-D with
    `stride(-1) == 1` (row stride free); `gate` 2-D of `attention_output`'s
    shape, or 3-D with `shape[0] == T` and `shape[1] * shape[2] == hidden`,
    `stride(-1) == 1`. A CPU tensor raises ValueError from Triton (measured).
    Hence no `is_valid`. Cross-device operands were not probed.

    The pre-call contents of `attention_output` are destroyed; the gate is
    left bitwise unchanged.

    Numerics, measured on GB300 at every cell: bf16 bitwise equal to the fp32
    reference except elements one ulp off (`tl.sigmoid`'s fast exp): none at
    1, 128 and 1024 rows x 6 heads or at 1, 32 and 128 rows x 24 heads; 8 of
    12582912 at 8192 rows x 6 heads, 21 of 50331648 at 8192 rows x 24 heads. Against a float64 product
    every cell sits at bf16 output rounding: cosine 0.9999986, relative L2
    1.64e-3 to 1.67e-3, max abs 1.54e-2. Torch's default bf16 tolerance holds,
    nothing loosened. Reading the row's q window instead of the gate window
    moves the result by 1.62 (1 row) to 3.76 (8192 rows) max abs, so the
    column offset and the per-head indexing are under test.

    Certified: bf16 operands, a compact `attention_output`; the 2-D window of
    the fused projection row at hidden 1536 (row stride 4352, gate column
    2048) at 1, 128, 1024 and 8192 rows, and at hidden 6144 (row stride 14080,
    gate column 7168) at 32, 128 and 8192 rows; the strided q_proj half (3-D
    gate, 24 heads) at 1 token; CUDA-graph capture and replay of the 1536-wide
    form at 8 rows.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      in-row offset   int32, < hidden (6144)
      (largest run 8192 x 14080 = 1.2e8 elements)

    Not certified: 0 tokens; rows past 8192; the strided q_proj half above 1
    token or at 6 heads (`[T, 6, 512]`, the MTP layer's bf16 q projection if
    it is head-sharded at tp4 / tep4); a strided `attention_output`; a gate
    dtype other than `attention_output.dtype`; fp16/fp32 operands; hidden
    sizes other than 1536 and 6144; torch.compile.
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
