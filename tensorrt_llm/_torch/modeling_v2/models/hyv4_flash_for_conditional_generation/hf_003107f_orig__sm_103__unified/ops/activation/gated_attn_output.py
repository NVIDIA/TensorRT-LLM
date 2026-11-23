# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated attention epilogue: ``attention_output *= sigmoid(gate)`` in place.

Hy4-Flash's attention layers gate the attention core's output before ``o_proj``
(`attn_output = attn_output * torch.sigmoid(gate_states)` in the HF reference,
`src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`). One Triton
kernel, `torch.ops.trtllm.fused_sigmoid_mul_inplace`.

The pattern is the qwen4_exp family's entry for the same op
(`models/qwen4_exp/ops/activation/fused_sigmoid_mul_inplace.py`): same kernel,
same two operand layouts, same fusion boundary. The cells here are this
checkpoint's widths and row counts.
"""

from __future__ import annotations

import torch

# Registered by the module that defines it: after `import
# tensorrt_llm._torch.custom_ops` alone, torch.ops.trtllm has no
# `fused_sigmoid_mul_inplace` (measured).
import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate  # noqa: F401
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The sparse layers' attention width: 32 heads x 256 value channels.
_SPARSE = 32 * 256

#: One rank's share of the sparse layers at `attn_tp` 2, the disaggregated
#: deployment's context engine: 16 heads x 256 value channels.
_SPARSE_TP2 = 16 * 256

#: The SWA layers' attention width: 64 heads x 256 value channels.
_SWA = 64 * 256


class GatedAttnOutput(OpWrapper):
    """`attention_output[t, i] *= sigmoid(gate[t, i])` in place, both `[T, H * D]`.

    Returns `attention_output`, the tensor it wrote. Both operands are widened
    to fp32 and the product is rounded once to `attention_output.dtype`. The
    gate is read-only. A 3-D `gate` `[T, heads, D]` with `heads * D == H` is
    accepted by the op and read as `gate.reshape(T, H)`.

    Fusion boundary: sigmoid and multiply only. The caller owns producing both
    operands (`linear_gate` in this model) and everything downstream (the o
    projection).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # The sparse layers: 32 heads x 256 value channels, the `linear_gate`
        # output at q_b's 8192 columns. Rows are the engine's call sites.
        Cell(
            why="sparse decode, one token: the attention epilogue before o_proj at 8192 wide",
            spec=dict(rows=1, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse verify and drafter: 6 rows (five drafts plus the anchor)",
            spec=dict(rows=6, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse decode at the widest captured batch: 64 rows",
            spec=dict(rows=64, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse verify at 64 requests: 6 x 64 = 384 rows",
            spec=dict(rows=384, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse chunked prefill: one 512-token chunk",
            spec=dict(rows=512, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse prefill at the envelope's largest row count: 5120 rows",
            spec=dict(rows=5120, width=_SPARSE, dtype=torch.bfloat16),
        ),
        # The same sparse layers on the disaggregated deployment's context
        # engine, `attn_tp` 2: one rank's 16 heads x 256 value channels, the
        # `linear_gate` output at q_b's 4096 columns. The SWA layers there are
        # 32 heads x 256 = 8192 wide, the sparse layers' `attn_tp` 1 shape, so
        # the cells above cover them -- the op sees only (T, hidden).
        Cell(
            why="sparse decode at attn_tp 2, one token: the attention epilogue before o_proj at 4096 wide",
            spec=dict(rows=1, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="the widest captured decode batch at attn_tp 2: 64 rows at 4096 wide",
            spec=dict(rows=64, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="a prefill chunk at attn_tp 2: 512 rows at 4096 wide",
            spec=dict(rows=512, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="the envelope's largest row count at attn_tp 2: 5120 rows at 4096 wide",
            spec=dict(rows=5120, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        # The SWA layers: 64 heads x 256 value channels, 16384 wide.
        Cell(
            why="SWA decode, one token: the same gate at 16384 wide",
            spec=dict(rows=1, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA verify: 6 rows at 16384 wide",
            spec=dict(rows=6, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA decode at the widest captured batch: 64 rows at 16384 wide",
            spec=dict(rows=64, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA verify at 64 requests: 384 rows at 16384 wide",
            spec=dict(rows=384, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA chunked prefill: one 512-token chunk at 16384 wide",
            spec=dict(rows=512, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA prefill at the envelope's largest row count: 5120 rows at 16384 wide",
            spec=dict(rows=5120, width=_SWA, dtype=torch.bfloat16),
        ),
    )

    note = """
    Metadata consumed: none. Source: `torch.ops.trtllm.fused_sigmoid_mul_inplace`,
    a Python `torch.library.custom_op` registered by
    `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py` around the
    Triton kernel `_fused_sigmoid_mul_kernel` there. One kernel, JIT-compiled for
    the running device at the first call of each (hidden, head_dim, block)
    specialization, no autotuner: grid `(T, cdiv(hidden, block))`,
    `block = min(next_pow2(hidden), 1024 if T < 1024 else 2048)`, 4 warps, PDL
    where the device supports it and PDL is enabled. The cells ran on sm_103.

    Call sites, each the attention output gate between the attention core and
    o_proj (`MlaLayer.epilogue`, both phases; `Hyv4FlashDraft.dflash_forward`
    at the 6-row draft width):
      - the sparse layers at 32 heads x 256 value channels, 8192 wide, at 1
        decode row, 6 verify/draft rows, 64 rows at the widest captured decode
        batch, 384 rows at 64 requests x 6, one 512-token chunk and the 5120-row
        prefill envelope;
      - the SWA layers at 64 heads x 256 value channels, 16384 wide, at the
        same row counts;
      - the disaggregated deployment's context engine, `attn_tp` 2: the sparse
        layers at 16 heads x 256 value channels, 4096 wide, at 1 and 64 decode
        rows, one 512-token chunk and the 5120-row prefill envelope; its SWA
        layers at 8192 wide are the sparse `attn_tp` 1 cells above.

    Preconditions, every one enforced loudly by the op's own bare `assert` (an
    AssertionError with no message): `attention_output` 2-D with
    `stride(-1) == 1` (row stride free); `gate` 2-D of `attention_output`'s
    shape, or 3-D with `shape[0] == T` and `shape[1] * shape[2] == hidden`,
    `stride(-1) == 1`. A CPU tensor raises ValueError from Triton (measured).
    Hence no `is_valid`. Cross-device operands were not probed.

    The pre-call contents of `attention_output` are destroyed; the gate is left
    bitwise unchanged.

    Numerics, measured on GB300 (sm_103) at every cell: bitwise equal to the
    fp32 reference at 1, 6 and 64 rows at all three widths; at 384, 512 and
    5120 rows 99.999964 to 99.999988 % of elements are bitwise equal and the
    rest sit within 0.60 ulp of their row's scale (`tl.sigmoid`'s fast exp
    flips a rounding tie), relative L2 4.4e-7 to 4.1e-6. Against a float64
    product at 64 rows x 16384 wide the relative L2 is 1.655e-3, bf16 output
    rounding, the op and the reference alike. The band is 2 ulp of the row
    scale element-wise (one for a tie flip, doubled for the comparison's own
    division) and 1 ulp relative RMS. A wrong read lands far outside it (2.29
    to 3.73 max abs) at the sparse width and at the context engine's 4096-wide
    share of it: the gate ignored entirely, or each row reading its neighbour's
    gate; the test holds the band refusing both. So the gate's row indexing and
    its application are under test.

    Certified: bf16 2-D operands, a compact `attention_output`, at 8192 and
    16384 wide and rows 1, 6, 64, 384, 512 and 5120, and at 4096 wide (the
    context engine's `attn_tp` 2 sparse width) at rows 1, 64, 512 and 5120;
    CUDA-graph capture and replay of the 8192-wide form at 1, 6 and 64 rows, of
    the 16384-wide form at 8 rows and of the 4096-wide form at 1 and 64 rows.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      in-row offset   int32; < hidden (16384)
      (largest run 5120 x 16384 = 8.4e7 elements)

    Not certified: 0 tokens; rows past 5120; the 3-D gate form; a strided
    `attention_output` (rejected); a gate dtype other than
    `attention_output.dtype`; fp16/fp32 operands; widths other than 4096, 8192
    and 16384; the `attn_tp` 2 verify and draft shapes (6 and 384 rows at 4096
    wide); CUDA-graph capture of the 16384-wide form at 1, 6 or 64 rows;
    torch.compile; every arch but sm_103.
    """

    def __call__(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.fused_sigmoid_mul_inplace(attention_output, gate)
        return attention_output

    def reference(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """What `attention_output` holds after the call; neither input is mutated.

        `gate.reshape(T, hidden)` is the per-head `(t, i // D, i % D)` read the
        op makes for a 3-D gate, and the identity for the 2-D form.
        """
        num_tokens, hidden = attention_output.shape
        g = gate.reshape(num_tokens, hidden).float()
        return (attention_output.float() * torch.sigmoid(g)).to(attention_output.dtype)


gated_attn_output = GatedAttnOutput()
