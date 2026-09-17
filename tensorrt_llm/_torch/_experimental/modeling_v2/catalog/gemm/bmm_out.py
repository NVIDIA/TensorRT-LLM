# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, assert_within_ulp, true_fp32_matmul


class _BmmOut(OpWrapper):
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
    )

    note = """
    `out` is written, not accumulated into: its prior contents are ignored.
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        torch.ops.trtllm.bmm_out(a, b, out)

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `__call__`: the op takes a destination,
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
        """
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


bmm_out = _BmmOut()
