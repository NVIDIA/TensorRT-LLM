# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel-major extraction of a prefill row/column slice, the operand a causal
conv1d over a token-major projection needs.

Wraps the in-tree Triton callable
``tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops.extract_transpose_prefill_slice``
(one kernel launch, ``_extract_transpose_prefill_kernel``). A module-plane
callable rather than a ``torch.ops.trtllm.*`` op: there is no op boundary
around this kernel.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice as _extract_transpose_prefill_slice,
)

#: The GDN mixer's packed `[q|k|v]` width for this checkpoint; the target hands
#: the whole row to this op.
GDN_MIXED_QKV_WIDTH = 10240


class ExtractTransposePrefillSlice(OpWrapper):
    """`out[c, t] = src[t, start_col + c]` for `t < num_prefill_tokens`,
    `c < width`: `src[:n, a:a+w].transpose(0, 1).contiguous()` in one pass.

    Fusion boundary: the row crop, the column crop and the transpose. No
    convolution, no conv state, no channel split -- the caller hands the result
    to `causal_conv1d_fwd`. Returns a fresh contiguous `[width,
    num_prefill_tokens]` tensor in `src`'s dtype; `src` is read-only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `num_rows`/`num_cols` build `src`; `row_stop`/`col_slice` turn it into the
    # view the call site passes (a row slice of a split token buffer, or a
    # column slice of a wider projection).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"GDN prefill, whole 10240-wide row, {n} tokens",
                spec=dict(
                    num_rows=n,
                    num_cols=GDN_MIXED_QKV_WIDTH,
                    n=n,
                    start_col=0,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            )
            for n in (1, 7, 33, 128, 1024, 4096)
        )
        + (
            Cell(
                why="mixed batch: prefill half of a split [512, 10240] buffer, a row-slice view",
                spec=dict(
                    num_rows=512,
                    num_cols=GDN_MIXED_QKV_WIDTH,
                    row_stop=400,
                    n=400,
                    start_col=0,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            ),
            Cell(
                why="fewer prefill rows than the buffer holds: the row crop is honoured",
                spec=dict(
                    num_rows=512,
                    num_cols=GDN_MIXED_QKV_WIDTH,
                    n=400,
                    start_col=0,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            ),
            Cell(
                why="interior column range of a contiguous 16384-wide buffer",
                spec=dict(
                    num_rows=200,
                    num_cols=16384,
                    n=150,
                    start_col=1024,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            ),
            Cell(
                why="column-slice view: row stride wider than the extracted width",
                spec=dict(
                    num_rows=200,
                    num_cols=16384,
                    col_slice=(1024, 1024 + GDN_MIXED_QKV_WIDTH),
                    n=150,
                    start_col=0,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"dtype {str(dtype).removeprefix('torch.')}: elements are moved, never interpreted",
                spec=dict(num_rows=64, num_cols=256, n=48, start_col=32, width=128, dtype=dtype),
            )
            for dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
                torch.float8_e4m3fn,
                torch.int32,
                torch.int8,
                torch.bool,
            )
        )
        + (
            Cell(
                why="zero prefill tokens: an empty [width, 0] result",
                spec=dict(
                    num_rows=64, num_cols=256, n=0, start_col=0, width=256, dtype=torch.bfloat16
                ),
            ),
            Cell(
                why="zero width: an empty [0, n] result",
                spec=dict(
                    num_rows=64, num_cols=256, n=1, start_col=0, width=0, dtype=torch.bfloat16
                ),
            ),
        )
    )

    note = """
    Metadata consumed: none. Stateless.

    Preconditions: `src` 2-D on CUDA with `src.stride(1) == 1`; the row stride
    is free. A non-unit column stride (`src.t()`, `src[:, ::2]`) is refused
    loudly by the callable's own bare `assert`, before any launch.

    The kernel masks its loads only against the requested `num_prefill_tokens`
    and `width`, never against `src`'s extent: 128 rows of a 64-row `src`, or
    columns `[200, 328)` of a 256-column one, were accepted and returned a
    full tensor read from out-of-bounds memory. `is_valid` bounds both.

    Hard limits
      width * num_prefill_tokens  int32 destination offset  boundary 2^31
        probed at width 16384: n 131000 (2.146e9) bitwise exact; n 140000
        (2.294e9) raises `CUDA error: an illegal memory access` (loud, so not
        a guard). The source offset is widened to int64 in-kernel: a source
        offset of 2.62e9 was bitwise exact. At width 10240 the destination
        boundary is n > 209715, within reach of a large `max_num_tokens`.
    """

    def __call__(
        self, src: torch.Tensor, num_prefill_tokens: int, start_col: int, width: int
    ) -> torch.Tensor:
        return _extract_transpose_prefill_slice(src, num_prefill_tokens, start_col, width)

    def reference(
        self, src: torch.Tensor, num_prefill_tokens: int, start_col: int, width: int
    ) -> torch.Tensor:
        return src[:num_prefill_tokens, start_col : start_col + width].transpose(0, 1).contiguous()

    def is_valid(
        self, src: torch.Tensor, num_prefill_tokens: int, start_col: int, width: int
    ) -> None:
        # A 3-D src with a unit dim-1 stride passes the callable's own assert
        # and is read as 2-D through stride(0).
        assert src.dim() == 2, f"src must be 2D, got {src.dim()}D"
        assert 0 <= num_prefill_tokens <= src.shape[0], (
            f"num_prefill_tokens {num_prefill_tokens} outside src rows [0, {src.shape[0]}]; "
            "the kernel reads past src rather than failing"
        )
        assert 0 <= start_col and 0 <= width and start_col + width <= src.shape[1], (
            f"column range [{start_col}, {start_col + width}) outside src columns "
            f"[0, {src.shape[1]}); the kernel reads past src rather than failing"
        )


extract_transpose_prefill_slice = ExtractTransposePrefillSlice()
