# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place sigmoid output gating: ``attention_output *= sigmoid(gate)``."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Registered by the module that defines it: after `import
# tensorrt_llm._torch.custom_ops` alone, torch.ops.trtllm has no
# `fused_sigmoid_mul_inplace` (measured).
import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's gated attention: 24 q heads of 256, q projection [T, 24, 512].
_CKPT = dict(heads=24, head_dim=256, dtype=torch.bfloat16, gate="interleaved")


@triton.jit
def _sigmoid_mul_amax_kernel(
    out_ptr,
    gate_ptr,
    parts_ptr,
    out_stride_token,
    gate_stride_token,
    gate_stride_head,
    num_parts,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    """The registered `_fused_sigmoid_mul_kernel`'s arithmetic in place, plus the maximum of
    the stored `|out|` folded into `parts[program % num_parts]`."""
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < hidden_size
    head = offsets // head_dim
    dim = offsets - head * head_dim
    attention = tl.load(out_ptr + token * out_stride_token + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + token * gate_stride_token + head * gate_stride_head + dim, mask=mask, other=0.0
    ).to(tl.float32)
    y = (attention * tl.sigmoid(gate)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + token * out_stride_token + offsets, y, mask=mask)
    peak = tl.max(tl.where(mask, tl.abs(y.to(tl.float32)), 0.0), axis=0)
    program = token * tl.num_programs(1) + block
    tl.atomic_max(parts_ptr + program % num_parts, peak)


def _sigmoid_mul_amax(attention_output, gate, amax_parts) -> None:
    # The registered op's layout checks, as loud.
    assert attention_output.dim() == 2 and attention_output.stride(-1) == 1
    num_tokens, hidden_size = attention_output.shape
    assert gate.stride(-1) == 1
    if gate.dim() == 3:
        assert gate.shape[0] == num_tokens and gate.shape[1] * gate.shape[2] == hidden_size
        head_dim, stride_token, stride_head = gate.shape[2], gate.stride(0), gate.stride(1)
    else:
        assert gate.dim() == 2 and gate.shape == attention_output.shape
        head_dim, stride_token, stride_head = hidden_size, gate.stride(0), hidden_size
    assert amax_parts.dtype == torch.float32 and amax_parts.dim() == 1
    if num_tokens == 0:
        return
    block_size = min(triton.next_power_of_2(hidden_size), 1024 if num_tokens < 1024 else 2048)
    _sigmoid_mul_amax_kernel[(num_tokens, triton.cdiv(hidden_size, block_size))](
        attention_output,
        gate,
        amax_parts,
        attention_output.stride(0),
        stride_token,
        stride_head,
        amax_parts.shape[0],
        hidden_size=hidden_size,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )


class FusedSigmoidMulInplace(OpWrapper):
    """`attention_output[t, i] *= sigmoid(gate[t, i // D, i % D])` in place, `D = gate.shape[-1]`.

    Returns `attention_output`, the tensor it wrote. Both operands are widened to
    fp32 and the product is rounded once to `attention_output.dtype`. `gate` is
    read through its own token and head strides, so the gate half of an
    interleaved `[T, heads, 2 * D]` q projection is passed as a strided view with
    no repack (bitwise equal to its compacted copy, measured). A 2-D `gate` of
    `attention_output`'s shape is one head spanning the row.

    With `amax_parts` (a zeroed fp32 buffer, `fp8_per_tensor_mm.amax_partials`) the same
    arithmetic runs in this target's copy of the kernel, which also folds the maximum of the
    stored `|attention_output|` into the buffer (atomic maxima, one slot per program modulo
    its length) for the per-tensor FP8 o projection.

    Fusion boundary: sigmoid and multiply (and the output maximum) only. The caller owns
    producing both operands and everything downstream (the o projection). `gate` is
    read-only.
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
        Cell(
            why="prefill 16384 tokens folding the output maximum for the FP8 o projection",
            spec=dict(
                tokens=16384,
                heads=24,
                head_dim=256,
                dtype=torch.bfloat16,
                gate="fused_row",
                amax=True,
            ),
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
    nothing loosened. Feeding the q half instead of the gate half moves the
    result by ~2.15 max abs at [17, 6144] (measured), so the per-head indexing
    is under test.

    Certified: bf16 operands at the checkpoint's hidden 6144, a compact
    `attention_output`; the 2-D window of the fused projection row at 128
    tokens (decode, 1024-wide blocks) and 8192 tokens (prefill, 2048-wide
    blocks), and the strided q_proj half (MTP) at 1 token.

    Hard limits
      T * row stride  token offset computed in int64  boundary 2^63  not probed
      (largest run 8192 rows)

    Not certified: 0 tokens, the strided q_proj half above 1 token, a strided
    `attention_output`, a gate dtype other than `attention_output.dtype`,
    fp16/fp32 operands, other hidden sizes, CUDA-graph capture, torch.compile.
    """

    def __call__(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        amax_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if amax_parts is None:
            torch.ops.trtllm.fused_sigmoid_mul_inplace(attention_output, gate)
        else:
            _sigmoid_mul_amax(attention_output, gate, amax_parts)
        return attention_output

    def reference(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        amax_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """What `attention_output` holds after the call; neither input is mutated.

        `gate.reshape(T, hidden)` is the per-head `(t, i // D, i % D)` read, for
        the 2-D and 3-D forms alike.
        """
        num_tokens, hidden = attention_output.shape
        g = gate.reshape(num_tokens, hidden).float()
        return (attention_output.float() * torch.sigmoid(g)).to(attention_output.dtype)


fused_sigmoid_mul_inplace = FusedSigmoidMulInplace()
