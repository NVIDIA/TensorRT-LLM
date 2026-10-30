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

#: The unified target's MTP layer head-sharded at tep4: q_proj 12288 -> 3072 columns a
#: rank, read as [T, 6, 512], the gate the second 256 of each head.
_MTP_TEP4 = dict(heads=6, head_dim=256, dtype=torch.bfloat16, gate="interleaved")


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
        # The unified target's MTP layer at tep4. Hidden 1536 is not a multiple of the
        # kernel's block, so every row ends in 512 masked lanes: two 1024-wide blocks
        # below 1024 rows, one 2048-wide block from 1024 rows.
        Cell(
            why="unified MTP drafter, 1 row: 6-head gate half of the [T, 6, 512] q_proj",
            spec=dict(tokens=1, **_MTP_TEP4),
        ),
        Cell(
            why="unified MTP drafter at 128 requests: 128 rows, 6 heads, 1024-wide blocks",
            spec=dict(tokens=128, **_MTP_TEP4),
        ),
        Cell(
            why="unified MTP-3 verify 128 x 4 = 512 rows, 6 heads",
            spec=dict(tokens=512, **_MTP_TEP4),
        ),
        Cell(
            why="unified MTP-7 verify 128 x 8 = 1024 rows: the first 2048-wide block count",
            spec=dict(tokens=1024, **_MTP_TEP4),
        ),
        Cell(
            why="unified MTP prefill chunk at max_num_tokens: 8192 rows, 6 heads",
            spec=dict(tokens=8192, **_MTP_TEP4),
        ),
    )

    note = """
    Metadata consumed: none. Source: `torch.ops.trtllm.fused_sigmoid_mul_inplace`,
    a Python `torch.library.custom_op` registered by
    `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py` around its
    Triton kernel `_fused_sigmoid_mul_kernel` (adapted from SGLang, per that
    file's header). One kernel, JIT-compiled for the running device on the first
    call of each specialization, no autotuner: grid `(T, cdiv(hidden, block))`,
    `block = min(next_pow2(hidden), 1024 if T < 1024 else 2048)`, 4 warps. The
    cells ran that JIT path on GB300 (sm_103).

    Call site (the unified target, site `mtp`): the MTP layer's attention output
    gate, between its attention core and o_proj, with the layer head-sharded at
    tep4 (and tp4): q_proj 12288 -> 3072 columns a rank, read as `[T, 6, 512]`,
    `gate` the second 256 of each head (token stride 3072, head stride 512),
    hidden 1536. Rows: the drafter 1..128, the depth-3 verify 128 x 4 = 512, the
    depth-7 verify 128 x 8 = 1024, a prefill chunk of up to 8192
    (`max_num_tokens`). Hidden 1536 is no multiple of the block, so every row
    ends in 512 masked lanes (two 1024-wide blocks below 1024 rows, one
    2048-wide block from 1024 rows); the test places `attention_output` at the
    head of a larger buffer and holds the elements past its last row unchanged.

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
    nothing loosened. Feeding the q half instead of the gate half moves the
    result by ~2.15 max abs at [17, 6144] (measured), so the per-head indexing
    is under test.

    Measured on GB300 at the MTP cells (6 heads, 3-D gate): bitwise equal to
    the fp32 reference at 1, 128 and 512 rows; one ulp off at 3 of 1572864
    elements (1024 rows) and 7 of 12582912 (8192 rows). Against a float64
    product every cell sits at bf16 output rounding, the op and the reference
    alike: cosine 0.9999986, relative L2 1.65e-3 to 1.69e-3, max abs 3.9e-3 (1
    row) to 1.49e-2 (8192 rows). Reading the q half of each head, or the flat
    row with the head stride ignored, moves the result by 1.8 (1 row) to 3.4
    (8192 rows) max abs; the test holds the default band refusing both.

    Certified: bf16 operands, a compact `attention_output`; at the
    checkpoint's hidden 6144 the 2-D window of the fused projection row at 128
    tokens (decode, 1024-wide blocks) and 8192 tokens (prefill, 2048-wide
    blocks), and the strided q_proj half (24 heads) at 1 token; at hidden 1536
    the strided half of a compact `[T, 6, 512]` q_proj at 1, 128, 512, 1024 and
    8192 rows, and its CUDA-graph capture and replay at 1, 128 and 1024 rows.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      in-row offset   head * head stride + dim, int32; < 2 * hidden at the 3-D form
      (largest run 8192 rows)

    Not certified: 0 tokens; rows past 8192; the 3-D gate at a token stride
    other than its own q_proj's (heads x 2 x head_dim, e.g. a gate inside a
    fused q|k|v row); the 24-head strided q_proj half above 1 token (a dep4
    rank's MTP layer: drafter 1..32 rows and its verify); a strided
    `attention_output`; a gate dtype other than `attention_output.dtype`;
    fp16/fp32 operands; hidden sizes other than 1536 and 6144; CUDA-graph
    capture of the 6144-wide forms; torch.compile.
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
