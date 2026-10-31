# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place sigmoid output gating: ``attention_output *= sigmoid(gate)``.

The gated attention's output gate, between the attention core and o_proj, in
one Triton kernel (`torch.ops.trtllm.fused_sigmoid_mul_inplace`). Shared by the
qwen4_exp tp1 target (`qwen3_8_flash_next_nvfp4__sm_103__tp1`) and the unified
tp4 / tep4 / dep4 target (`hf_fc694b5_orig__sm_103__unified`) -- its trunk,
gate off the fused input-projection row, and its MTP layer, gate as the strided
half of the q projection; promoted from their copies.
"""

from __future__ import annotations

import torch

# Registered by the module that defines it: after `import
# tensorrt_llm._torch.custom_ops` alone, torch.ops.trtllm has no
# `fused_sigmoid_mul_inplace` (measured).
import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate  # noqa: F401

from .._op import Arch, Cell, OpWrapper

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
        # The tp1 target: hidden 6144, the trunk's gate off the 14080-wide fused
        # row (the test's default row stride and gate column), the MTP layer's
        # off its [T, 24, 512] q_proj.
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
        # The unified target's trunk. Hidden 1536 is not a multiple of the kernel's
        # block, so every tp4 / tep4 row ends in 512 masked lanes: two 1024-wide
        # blocks below 1024 rows, one 2048-wide block from 1024 rows.
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
        # The unified target's MTP layer at tep4, on the same 1536-wide masked rows.
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
    `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py` around the
    Triton kernel `_fused_sigmoid_mul_kernel` there (adapted from SGLang, per that
    file's header). One kernel, JIT-compiled for the running device at the first
    call of each (hidden, head_dim, block) specialization, no autotuner: grid
    `(T, cdiv(hidden, block))`, `block = min(next_pow2(hidden), 1024 if T < 1024
    else 2048)`, 4 warps. The cells ran that JIT path on GB300 (sm_103).

    Call sites, each the attention output gate between the attention core and
    o_proj:
      - The tp1 target (`qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1`): the
        trunk's gate as the 2-D window of the fused input projection row
        (14080 wide, the gate at column 7168, hidden 6144) at decode and
        prefill row counts; its MTP layer's gate as the strided second half of
        each head of a `[T, 24, 512]` q_proj.
      - The unified target's trunk
        (`qwen4_exp_for_conditional_generation/hf_fc694b5_orig__sm_103__unified`):
        `DecodeTarget._attention_block` (decode, 1..128 rows at tp4 / tep4,
        1..32 at a dep4 rank), its MTP verify (up to 128 x 8 = 1024 rows, tep4,
        draft 7) and `PrefillTarget._attention_block` (one chunk of up to 8192
        rows, `max_num_tokens`). `gate` is the 2-D window of the fused input
        projection row at column qkv_width: at tp4 / tep4 the row is 4352 wide
        (q 1536 | k 256 | v 256 | gate 1536 | index scratch 768), the gate at
        column 2048 and hidden 1536; at dep4 tp1's 14080-wide row, gate at
        7168, hidden 6144.
      - The unified target's MTP layer, head-sharded at tep4 (and tp4): q_proj
        12288 -> 3072 columns a rank, read as `[T, 6, 512]`, `gate` the second
        256 of each head (token stride 3072, head stride 512), hidden 1536.
        Rows: the drafter 1..128, the depth-3 verify 128 x 4 = 512, the depth-7
        verify 128 x 8 = 1024, a prefill chunk of up to 8192
        (`max_num_tokens`).
    Hidden 1536 is no multiple of the block, so every row of the unified
    target's 6-head forms ends in 512 masked lanes (two 1024-wide blocks below
    1024 rows, one 2048-wide block from 1024 rows); the test places
    `attention_output` at the head of a larger buffer and holds the elements
    past its last row unchanged.

    Preconditions, every one enforced loudly by the op's own bare `assert`
    (an AssertionError with no message): `attention_output` 2-D with
    `stride(-1) == 1` (row stride free); `gate` 2-D of `attention_output`'s
    shape, or 3-D with `shape[0] == T` and `shape[1] * shape[2] == hidden`,
    `stride(-1) == 1`. A CPU tensor raises ValueError from Triton (measured).
    Hence no `is_valid`. Cross-device operands were not probed.

    The pre-call contents of `attention_output` are destroyed; the gate is
    left bitwise unchanged.

    Numerics, measured on GB300 at every cell: bf16 bitwise equal to the fp32
    reference except elements one ulp off (`tl.sigmoid`'s fast exp). None at
    1, 128 and 1024 rows of the 6-head fused row, at 1, 128 and 512 rows of
    the 6-head 3-D gate, or at 1, 32 and 128 rows x 24 heads; 3 of 1572864 at
    1024 rows of the 6-head 3-D gate; at 8192 rows 8 of 12582912 (6-head fused
    row), 7 of 12582912 (6-head 3-D gate) and 21 of 50331648 (24 heads).
    Against a float64 product every cell sits at bf16 output rounding, the op
    and the reference alike: cosine 0.9999986, relative L2 1.64e-3 to 1.69e-3,
    max abs 3.9e-3 (1 row) to 1.54e-2. Torch's default bf16 tolerance holds,
    nothing loosened. A wrong read lands far outside it: the q half instead of
    the gate half moves the result by ~2.15 max abs at [17, 6144]; the fused
    row's q window instead of its gate window by 1.62 (1 row) to 3.76 (8192
    rows); at the 6-head 3-D gate, the q half of each head or the flat row with
    the head stride ignored by 1.8 (1 row) to 3.4 (8192 rows), and the test
    holds the default band refusing both. So the column offset and the
    per-head indexing are under test.

    Certified: bf16 operands, a compact `attention_output`; the 2-D window of
    the fused projection row at hidden 6144 (row stride 14080, gate column
    7168) at 32, 128 and 8192 rows, and at hidden 1536 (row stride 4352, gate
    column 2048) at 1, 128, 1024 and 8192 rows; the strided q_proj half (3-D
    gate) at 24 heads (`[T, 24, 512]`, hidden 6144) at 1 token, and at 6 heads
    (a compact `[T, 6, 512]`, hidden 1536) at 1, 128, 512, 1024 and 8192 rows;
    CUDA-graph capture and replay of the 1536-wide fused-row form at 8 rows and
    of the 6-head 3-D gate at 1, 128 and 1024 rows.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      in-row offset   int32; < hidden (6144) at the 2-D form,
                      head * head stride + dim < 2 * hidden at the 3-D form
      (largest run 8192 x 14080 = 1.2e8 elements)

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
