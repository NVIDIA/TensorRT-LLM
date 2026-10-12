# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU activation via the flashinfer silu_and_mul kernel."""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class FlashinferSiluAndMul(OpWrapper):
    """`silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, new tensor out.

    The gate and up projections arrive concatenated along the last dim, which
    is why the op takes one tensor and halves it rather than taking two.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 is the only shipped caller. It reaches this op from two
    # widths -- the routed/shared expert intermediate and the dense layers'
    # -- at decode and prefill token counts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, 2 x moe_intermediate (2048)",
            spec=dict(shape=(1, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="the dense layers' wider intermediate, 2 x 18432",
            spec=dict(shape=(16, 36864), dtype=torch.bfloat16),
        ),
        # DeepSeek-V4.1's vision tower MLP: [gate ; up] at d = 2816, P patches
        # per tower call, every P in 1..17037 certified by the prefix sweep in
        # the test; these are the sweep's two ends. `data="vit"` is that test's
        # input: N(0, 3) with planted zero, large-gate, large-up, tiny-gate and
        # large-negative-gate rows.
        Cell(
            why="deepseek-v4.1 vision [P, 5632] P=1: the tower MLP, "
            "one row, bottom of the prefix sweep 1..17037",
            spec=dict(shape=(1, 5632), dtype=torch.bfloat16, data="vit"),
        ),
        Cell(
            why="deepseek-v4.1 vision [P, 5632] P=17037: the tower MLP, "
            "its largest call, top of the prefix sweep 1..17037",
            spec=dict(shape=(17037, 5632), dtype=torch.bfloat16, data="vit"),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    fp16 and fp32 are accepted by the op. No shipped target uses them here, so
    they are not cells.

    DeepSeek-V4.1: the vision tower's MLP calls it on cublas_mm's contiguous
    bf16 `[P, 5632]` output (`[gate ; up]` in the checkpoint's w1 order,
    d = 2816), unclamped, eagerly, with P any patch count 1..17037. Metadata
    none. Every P is certified: the 17037-row call meets the reference, every
    prefix of it is bitwise the op on that many rows, nothing is written past
    the last row, and one kernel (block 352, grid P) runs at every count. The
    kernel is built with fast math: gates in (-88.72, -87.33] and results
    below 2^-126 in magnitude come back exactly 0 where the fp32 reference is a
    tiny nonzero (below 1e-36 x max(1, |up|)). The default gate passes them;
    the test's per-element 1-ulp gate admits exactly that set, measured over
    every finite bf16 gate. Not certified for V4.1: CUDA-graph capture at this
    width, fp16, more than 17037 rows, a non-contiguous or row-strided input,
    NaN or inf input.
    """

    def raw_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_silu_and_mul(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """fp32-accumulated; the kernel computes in fp32 and rounds once."""
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor) -> None:
        # The op's own check is on the *row*, `x.shape[-1] * itemsize % 16 == 0`,
        # which a final dimension of 24 passes. The kernel vectorizes over the
        # two halves, so what has to be 16-byte aligned is the *half*: at 24 the
        # second half starts mid-vector and the launch dies with `CUDA
        # misaligned address`, poisoning the context rather than raising.
        width = x.shape[-1]
        assert width % 2 == 0, f"x.shape[-1] must be even to split in half; got {width}"
        half_bytes = (width // 2) * x.element_size()
        assert half_bytes % 16 == 0, (
            f"x.shape[-1] // 2 must be a whole number of 16-byte vectors; "
            f"{width} halves to {half_bytes} bytes, which is not a multiple of 16 "
            f"-- the kernel would fault on the misaligned second half"
        )
        assert half_bytes >= 16, (
            f"x.shape[-1] // 2 must hold at least one 16-byte vector; {width} "
            f"gives {half_bytes} bytes, for which the computed block size is 0"
        )


flashinfer_silu_and_mul = FlashinferSiluAndMul()
