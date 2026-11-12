# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated full-attention prologue: per-head gemma q/k RMSNorm and interleaved MRoPE on
the interleaved ``[q0, g0, q1, g1, ..., K, V]`` qkv projection, packed into ``[Q | K | V]``
in one Triton kernel, with the output gate handed back as a view of the projection.

The kernel is the in-tree Qwen3.5 fused qk-norm/RoPE/gate kernel's arithmetic (in
``_qkv_gemma_rmsnorm_rope_gate``), reading the MRoPE section rows through their stride and
storing no gate.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._qkv_gemma_rmsnorm_rope_gate import qkv_gemma_rmsnorm_mrope

#: The checkpoint's gated attention: 24 q heads and 4 kv heads of 256, partial rotary
#: 0.25 (64 rotary dims), interleaved MRoPE sections (11, 11, 10), theta 1e7, eps 1e-6.
_CKPT = dict(
    num_q_heads=24,
    num_kv_heads=4,
    head_dim=256,
    rotary_dim=64,
    mrope_section=(11, 11, 10),
    rope_theta=1.0e7,
    eps=1e-6,
)


class QkvGemmaRmsnormRopeGate(OpWrapper):
    """`(qkv_out, gate)` from the interleaved projection `qkv` `[T, 2 * q_size + 2 * kv_size]`.

    Per q head `i`, the projection holds `q_i` then `g_i` (`head_dim` each); then K, then V.

    - `qkv_out` `[T, q_size + 2 * kv_size]`, new: Q and K each gemma-normed per head
      (`x * rsqrt(mean(x^2) + eps) * (1 + w)`, rounded to bf16), the first `rotary_dim`
      dims then rotated NeoX-style by interleaved MRoPE (rotary-half index `j` reads
      position section 1 if `j % 3 == 1 and j < 3 * s1`, 2 if `j % 3 == 2 and j < 3 * s2`,
      else 0), the rest passed through; V copied.
    - `gate` `[T, num_q_heads, head_dim]`: a VIEW of `qkv` (`g_i` of every head), strides
      `(qkv.stride(0), 2 * head_dim, 1)`. Nothing is copied: the caller must leave `qkv`
      unmodified until the gate has been consumed.

    `positions` is `[3, ..., T]` (the runner's `[3, 1, max_tokens]` buffer sliced to `T`
    tokens): it is read through its section stride, so the slice needs no copy.

    Fusion boundary: q/k norm, RoPE, the Q/K/V pack. The projection, the KV-cache append
    (inside attention) and the gate's consumer (`sigmoid(gate) * o`) are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: tokens; positions ("slice": a [3, 1, 8192] int32 buffer sliced to [3, 1, T] |
    # "contiguous_int64": a contiguous [3, T] int64 tensor).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode 1 token: positions a [3, 1, 1] slice of the runner's buffer",
            spec=dict(tokens=1, positions="slice", **_CKPT),
        ),
        Cell(
            why="decode 32 tokens (c32): section stride 8192, not T",
            spec=dict(tokens=32, positions="slice", **_CKPT),
        ),
        Cell(
            why="decode 128 tokens (concurrency 128)",
            spec=dict(tokens=128, positions="slice", **_CKPT),
        ),
        Cell(
            why="mixed step 2111 tokens (2048 context + 63 decode), positions sliced",
            spec=dict(tokens=2111, positions="slice", **_CKPT),
        ),
        Cell(
            why="prefill 8192 tokens (max_num_tokens), contiguous int64 positions",
            spec=dict(tokens=8192, positions="contiguous_int64", **_CKPT),
        ),
    )

    note = """
    Metadata consumed: the step's MRoPE position ids (`positions`). `cos_sin` is the
    model's fp32 NeoX table `[max_positions, 2, rotary_dim / 2]` (cos then sin of
    `position * theta^(-2j / rotary_dim)`), read at `position` per rotary-half index.

    Kernel: one Triton program per (token, head) over the 24 q + 4 k + 4 v heads; q/k
    programs load the head (256 bf16), reduce its mean square in fp32, round the normed
    head to bf16, rotate the first 64 dims in fp32 and store bf16; v programs copy.
    4 warps, no autotuner, no PDL; JIT-compiled on first use.

    Preconditions, enforced loudly (AssertionError before launch): 2-D bf16 `qkv`
    with unit column stride and the exact interleaved width; `q_weight` / `k_weight`
    `[head_dim]`; contiguous fp32 `cos_sin` of `[P, 2, rotary_dim / 2]`; int32 or
    int64 `positions` of `3 * T` elements; `sum(mrope_section) == rotary_dim / 2`
    with nonzero sections 1 and 2. A position outside `[0, P)` yields NaN rows (and a
    device assert under `TRITON_DEBUG`). Hence no `is_valid`.

    A `positions` view whose token stride is not 1 (more than one token) is copied
    to a contiguous `[3, T]` first; the runner's slice never is.

    Numerics: bitwise the in-tree fused q/k-norm RoPE gate kernel on the same inputs
    (qkv_out and gate). Against the fp32 reference: the gate and V bitwise, Q and K
    within one bf16 rounding of the normed head (rsqrt.approx and the tree sum) plus
    one of the output.

    Certified: the checkpoint's shape at 1, 32, 128, 2111 and 8192 tokens, int32
    sliced and int64 contiguous positions. Not certified: fp16, plain (non-MRoPE)
    RoPE, a row-strided `qkv`, torch.compile.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        mrope_section: Tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(qkv_out, gate)`; `gate` is a view of `qkv`, nothing is written to it."""
        qkv_out = qkv_gemma_rmsnorm_mrope(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            mrope_section,
        )
        q_size = num_q_heads * head_dim
        gate = qkv[:, : 2 * q_size].unflatten(1, (num_q_heads, 2 * head_dim))[:, :, head_dim:]
        return qkv_out, gate

    def reference(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        mrope_section: Tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(qkv_out, gate)`: the normed head rounded to bf16 (the op's definition), the
        rotation in fp32 rounded once; `gate` the same view of `qkv`. Nothing is mutated."""
        num_tokens = qkv.shape[0]
        q_size = num_q_heads * head_dim
        kv_size = num_kv_heads * head_dim
        q_gate, k, v = qkv.split((2 * q_size, kv_size, kv_size), dim=-1)
        q_gate = q_gate.unflatten(1, (num_q_heads, 2 * head_dim))
        q, gate = q_gate[..., :head_dim], q_gate[..., head_dim:]
        k = k.unflatten(1, (num_kv_heads, head_dim))

        def gemma_norm(x, weight):
            xf = x.float()
            inverse_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
            return (xf * inverse_rms * (weight.float() + 1.0)).to(qkv.dtype)

        half = rotary_dim // 2
        pos = positions.reshape(3, num_tokens).long()
        j = torch.arange(half, device=qkv.device)
        section = torch.zeros_like(j)
        section[(j % 3 == 1) & (j < mrope_section[1] * 3)] = 1
        section[(j % 3 == 2) & (j < mrope_section[2] * 3)] = 2
        selected = pos[section.unsqueeze(0), torch.arange(num_tokens, device=qkv.device)[:, None]]
        cos = cos_sin[selected, 0, j].unsqueeze(1)
        sin = cos_sin[selected, 1, j].unsqueeze(1)

        def rope(x):
            first, second = x[..., :half].float(), x[..., half:rotary_dim].float()
            rotated = (first * cos - second * sin, second * cos + first * sin)
            return torch.cat((*rotated, x[..., rotary_dim:].float()), dim=-1).to(qkv.dtype)

        q = rope(gemma_norm(q, q_weight)).reshape(num_tokens, q_size)
        k = rope(gemma_norm(k, k_weight)).reshape(num_tokens, kv_size)
        return torch.cat((q, k, v), dim=-1), gate


qkv_gemma_rmsnorm_rope_gate = QkvGemmaRmsnormRopeGate()
