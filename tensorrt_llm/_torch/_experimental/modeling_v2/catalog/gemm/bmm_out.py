# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, assert_within_ulp, true_fp32_matmul

# DeepSeek-V4.1's grouped low-rank output projection (`wo_a`), in the target's
# layout: a = o.view(T, G, 4096).transpose(0, 1), b = w.view(G, 1024,
# 4096).transpose(1, 2), out = oa.view(T, G, 1024).transpose(0, 1). G is the
# rank's share of the 8 output groups. Rows: the ends of the every-T sweep
# (1..32768) the shared test runs at each G, and at G = 1 the edges of the
# split-K bands (T 1..32 and 65..128).
_DSV41_WO_A_ROWS = {
    1: (1, 32, 33, 64, 65, 128, 129, 32768),
    2: (1, 32768),
    4: (1, 32768),
    8: (1, 32768),
}
_DSV41_CELLS = tuple(
    Cell(
        why=f"deepseek-v4.1 wo_a G={g} T={t}: [G, T, 4096] @ [G, 4096, 1024], all three "
        "strided views; every T 1..32768 swept"
        + ("; split-K" if g == 1 and (t <= 32 or 65 <= t <= 128) else ""),
        spec=dict(
            batch=g,
            m=t,
            k=4096,
            n=1024,
            dtype=torch.bfloat16,
            transposed_out=True,
            transposed_a=True,
            transposed_b=True,
        ),
    )
    for g, rows in _DSV41_WO_A_ROWS.items()
    for t in rows
)


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, in place, returns None.

    The point of the op over `torch.bmm` is the caller-owned `out`: both
    shipped call sites pass a transposed view of a buffer the rest of the
    forward already holds, so the product lands where it is needed with no
    copy. That is also the hazard -- see `is_valid`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1's MLA is the only caller, and it uses the op twice per layer
    # in opposite orientations: absorbing the nope half of q into the latent
    # space on the way in, and projecting the latent attention output back out
    # to v. Batch is the head count (128, replicated under attention DP).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, q_nope -> latent: [H, 1, nope] @ [H, nope, kv_lora]",
            spec=dict(batch=128, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="decode, latent -> v: [H, 1, kv_lora] @ [H, kv_lora, v_dim]",
            spec=dict(batch=128, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="prefill token count, the same absorbing orientation",
            spec=dict(batch=128, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="contiguous out, to separate the op's own correctness from the "
            "transposed-view path the targets happen to use",
            spec=dict(batch=128, m=64, k=128, n=512, dtype=torch.bfloat16, transposed_out=False),
        ),
    ) + _DSV41_CELLS

    note = """
    `out` is written, not accumulated into: its prior contents are ignored.

    DeepSeek-V4.1: the grouped low-rank output projection, bf16, K = 4096,
    N = 1024, G in {1, 2, 4, 8} (8 / tp under attention TP; 8 under attention
    DP and in the DSpark drafter's attention), every operand a strided view of
    the target's [T, G * 4096] activation, [G * 1024, 4096] weight and
    [T, G * 1024] output buffer -- no copy or transpose kernel runs, and the
    result lands in that buffer. No metadata. T is the rank's token count, at
    most 16384, or up to 32768 in the decoder bounded-replay pass; the shared
    test runs every T in 1..32768 at every G, ascending and shuffled. At each T
    nothing is written past the call's rows, and the rows are bitwise the
    first T rows of the 32768-row call -- except at G = 1, T 1..32 and
    65..128, where cuBLAS runs a split-K kernel plus a reduction and the
    result, inside the gate and repeatable, differs from a larger call's in
    up to 1.8e-3 of its elements. Every other call is one kernel. CUDA-graph
    replay is bitwise equal to eager at every T of G, 5G and 6G over the
    batch set 1..32, 40..128 step 8, 192, 256 (114 counts to 1536), captured
    into one shared pool after an eager call at each shape. Not certified for
    DeepSeek-V4.1: T above 32768, other G at this K and N, capture at other T
    or before an eager call at that shape, fp16 or fp32 in this layout, and
    torch matmul flags other than the defaults, which the kernel follows.
    """

    def raw_call(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `raw_call`: the op takes a destination,
        the reference returns a value, and hiding the argument would hide that
        the op mutates one of its inputs.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, ~1e-3 relative, which would make the reference the inaccurate
        side of the comparison.
        """
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # A wrong-shaped `out` is silently resized and re-allocated by the op,
        # detaching it from whatever buffer the caller aliased it with -- the
        # product then lands somewhere nobody reads, and the forward continues
        # with stale data rather than failing.
        assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
        )
        # Mixed input dtypes are type-promoted instead of rejected, changing the
        # required out dtype; the catalog exposes only the single-dtype form.
        assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """One GEMM's worth of accumulation-order difference.

        Kernel and reference consume bit-identical operands and differ only in
        the order cuBLAS and torch sum over K. Over random operands a few
        outputs land near zero, where that reordering shows up as a relative
        difference of 10 on an absolute difference of 1.5e-4 -- which is what
        torch's default band rejects, and what no bf16 result could represent
        anyway: one ulp at these rows' scale is about 6e-1.

        So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
        Measured worst case across this entry's cells is 1.5e-4 absolute, or
        roughly 3e-4 ulp -- four orders of margin, and a band that a genuinely
        wrong product would miss by orders rather than squeak past.

        DeepSeek-V4.1's cells (K = 4096, outputs up to about 360 for
        unit-variance operands) take the same gate. Measured on sm_103 over
        every T 1..8192 at G 2 and 8, the largest |op - ref| was one bf16
        rounding step, 2.0 at outputs of 256 to 512 -- at most 2 of these ulp
        against the row scale -- and at most 2.5e-3 of a call's elements
        differed from the reference at all, far inside the 2-ulp relative RMS.
        Swapping the weight's two groups lands about 390 away. The shared
        test's every-T sweep also holds the absolute band measured there
        (rtol 1.6e-2, atol 2e-3).
        """
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


bmm_out = BmmOut()
