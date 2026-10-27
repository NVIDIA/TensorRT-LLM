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
#: this op the `[q|k|v]` column view of its one input-projection output.
GDN_MIXED_QKV_WIDTH = 10240
#: That output's width: `[q|k|v | z | a | b]` = 10240 + 6144 + 48 + 48.
GDN_IN_PROJ_WIDTH = 16480


class ExtractTransposePrefillSlice(OpWrapper):
    """`out[c, t] = src[t, start_col + c]` for `t < num_prefill_tokens`,
    `c < width`: `src[:n, a:a+w].transpose(0, 1).contiguous()` in one pass.

    Fusion boundary: the row crop, the column crop and the transpose. No
    convolution, no conv state, no channel split -- the caller hands the result
    to `causal_conv1d_fwd`. Returns a fresh contiguous `[width,
    num_prefill_tokens]` tensor in `src`'s dtype; `src` is read-only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `num_rows`/`num_cols` build the [T, 16480] projection; `row_stop` and
    # `col_slice` turn it into the view the call site passes (its q|k|v
    # columns, and in a mixed step the prefill rows of those).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="mixed batch: prefill rows of the q|k|v view of a [512, 16480] in_proj",
            spec=dict(
                num_rows=512,
                num_cols=GDN_IN_PROJ_WIDTH,
                row_stop=400,
                col_slice=(0, GDN_MIXED_QKV_WIDTH),
                n=400,
                start_col=0,
                width=GDN_MIXED_QKV_WIDTH,
                dtype=torch.bfloat16,
            ),
        ),
        *(
            Cell(
                why=f"GDN prefill, the q|k|v column view of the in_proj, {n} tokens",
                spec=dict(
                    num_rows=n,
                    num_cols=GDN_IN_PROJ_WIDTH,
                    col_slice=(0, GDN_MIXED_QKV_WIDTH),
                    n=n,
                    start_col=0,
                    width=GDN_MIXED_QKV_WIDTH,
                    dtype=torch.bfloat16,
                ),
            )
            for n in (33, 8192)
        ),
    )

    note = """
    Metadata consumed: none. Stateless.

    Certified bitwise in bf16 at the call site's operand, the q|k|v column
    view of the `[T, 16480]` input projection: 33 and 8192 tokens, and a mixed
    step's 400 prefill rows of a 512-row projection.

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

    Not certified: dtypes other than bf16, `start_col` other than 0, a row crop
    shorter than `src`, empty extents.
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
