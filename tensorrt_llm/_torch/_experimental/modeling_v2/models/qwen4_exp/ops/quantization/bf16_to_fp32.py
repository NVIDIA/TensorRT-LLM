# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Widening bf16 -> fp32 of a contiguous tensor in one vectorized kernel: the logits
processor's cast of the step's `[rows, vocab]` logits.

Tier 5 (Triton written for this target). torch's `.float()` runs this as an unrolled,
unvectorized `direct_copy` at under half the bandwidth; the values are the same bit for
bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's vocabulary: the width of the logits rows.
VOCAB = 248320
#: Elements a program (8 warps: 16 a thread, two 16-byte loads).
_BLOCK = 4096
_WARPS = 8


@triton.jit
def _bf16_to_fp32_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask).to(tl.float32), mask=mask)


class Bf16ToFp32(OpWrapper):
    """`x.float()` for a contiguous bf16 `x`: a fresh contiguous fp32 tensor of `x`'s
    shape. Widening is exact, so every element is `x`'s value; NaN payloads widen as
    torch widens them (the 16 bits shifted up).

    Fusion boundary: the cast only; the head's GEMM before it and the sampler after it
    are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode logits: 128 rows of the 248320 vocabulary", spec=dict(shape=(128, VOCAB))),
        Cell(why="one row; a ragged last program", spec=dict(shape=(1, VOCAB))),
        Cell(
            why="every bf16 bit pattern: NaNs, infinities, subnormals, signed zeros",
            spec=dict(shape=(1, 65536), source="all_codes"),
        ),
    )

    note = """
    Metadata consumed: none; stateless. One Triton kernel, `cdiv(numel, 4096)` programs
    of 8 warps over the flat tensor. Call site: the target's logits processor
    (`_WideningLogitsProcessor`), on the head's bf16 logits; the shell's `.float()`
    after it is then a no-op.

    GB300, graph replay: 128 x 248320 31.3 us against 82.1 for `.float()`, 124 rows
    30.7 against 79.9, 33 rows 8.3 against 16.5, one row 4.1 against 4.2.

    Preconditions: bf16, contiguous (the kernel reads the flat storage).
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        n = x.numel()
        if n:
            _bf16_to_fp32_kernel[(triton.cdiv(n, _BLOCK),)](x, y, n, BLOCK=_BLOCK, num_warps=_WARPS)
        return y

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def is_valid(self, x: torch.Tensor) -> None:
        assert x.dtype == torch.bfloat16, "bf16 input"
        assert x.is_contiguous(), "contiguous input: the kernel reads the flat storage"


bf16_to_fp32 = Bf16ToFp32()
