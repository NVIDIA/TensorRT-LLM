# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiLU-and-mul over the leading `2 * inter` columns of each row, read through the row stride:
the shared expert's activation taken out of its fused gate|up|gate-logit GEMM output.

Tier 5: a Triton kernel, written for the tp1 target (`qwen3_8_flash_next_nvfp4__sm_103__tp1`,
`ops/activation/silu_and_mul_rows`) and kept unchanged here. `flashinfer_silu_and_mul` addresses row `t` at
`t * 2 * inter`, so its input must be exactly the gate|up columns, contiguous; here they are
the head of a wider row (the same GEMM also forms the per-token gate logit) and are read in
place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's shared expert: intermediate 640, the fused GEMM output row 2 * 640 + 8.
_INTER = 640
_WIDTH = 2 * _INTER + 8
_BLOCK = 1024


@triton.jit
def _silu_and_mul_rows_kernel(x_ptr, out_ptr, stride_x, INTER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < INTER
    g = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * stride_x + INTER + cols, mask=mask, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(out_ptr + row * INTER + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class SiluAndMulRows(OpWrapper):
    """`out[t, i] = silu(x[t, i]) * x[t, inter + i]` for `i < inter`: the gate half first, as
    `flashinfer_silu_and_mul` reads it, in fp32 and rounded once to `x`'s dtype. Columns of
    `x` past `2 * inter` are not read. Returns a fresh contiguous `[T, inter]` tensor.

    Fusion boundary: the activation only; the GEMM that forms `x` and the one that reads
    `out` are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="prefill chunk 16384 rows of the fused 1288-wide output", spec=dict(rows=16384)),
        Cell(why="short prefill step, 128 rows of the fused output", spec=dict(rows=128)),
        Cell(why="zero rows: no launch", spec=dict(rows=0)),
        Cell(why="decode, one request: the shared expert's single row", spec=dict(rows=1)),
        Cell(
            why="decode, 32 rows: a dep4 rank's local share of a 128-request batch",
            spec=dict(rows=32),
        ),
        Cell(why="MTP-7 verify: 128 requests x 8 rows", spec=dict(rows=1024)),
        Cell(why="prefill chunk at max_num_tokens 8192", spec=dict(rows=8192)),
    )

    note = """
    Metadata consumed: none; every row independent. Call sites: the decode and prefill
    targets' MoE block, on the replicated shared expert's fused
    `[gate | up | gate logit | pad]` GEMM output, so inter 640 and the 1288-wide row at
    every member (tp4, tep4, dep4). Decode rows: 1..128 per rank at tp4/tep4, 1..32 local
    at dep4, 1024 at an MTP-7 verify of 128 requests; prefill chunks of up to
    max_num_tokens.

    Preconditions: 2-D `x` with unit column stride, `x.shape[1] >= 2 * inter`; bf16 or fp16.

    Source: Triton, JIT-compiled at first call for the running device; the cells ran that
    path on sm_103 (GB300). Launch: `(rows, cdiv(inter, 1024))` programs of 4 warps, the
    same kernel at every row count (no path switch). Row offsets are int64, so no index
    product overflows inside the envelope (2^31 elements of the 1288-wide input is 1.67M
    rows).

    Time per call on GB300, 100 calls captured in one CUDA graph: 1.17 / 1.40 / 1.43 /
    2.17 / 7.50 / 13.50 us at 1 / 32 / 128 / 1024 / 8192 / 16384 rows.
    `flashinfer_silu_and_mul` on already-contiguous gate|up columns: 1.11 / 1.17 / 1.21 /
    2.19 / 9.45 / 17.64 us; with the copy the fused layout needs: 1.11 (one row is
    contiguous) / 3.83 / 4.02 / 5.61 / 23.87 / 60.00 us.

    Numerics: `g / (1 + exp(-g)) * u` in fp32 (Triton's `exp` and division), rounded once;
    bitwise `flashinfer_silu_and_mul` on the same values at every non-empty cell (random
    bf16 values scaled by 3, with 0, -0, 88 and -100 in row 0).

    Certified: bf16 at inter 640 on the 1288-wide fused output, 0, 1, 32, 128, 1024, 8192
    and 16384 rows; CUDA-graph replay at 1, 32 and 128 rows. Not certified: fp16, other
    widths; row counts between the cells run the same kernel but were not driven.
    """

    def __call__(self, x: torch.Tensor, inter: int) -> torch.Tensor:
        rows = x.shape[0]
        out = torch.empty((rows, inter), dtype=x.dtype, device=x.device)
        if rows > 0:
            _silu_and_mul_rows_kernel[(rows, triton.cdiv(inter, _BLOCK))](
                x, out, x.stride(0), INTER=inter, BLOCK=_BLOCK, num_warps=4
            )
        return out

    def reference(self, x: torch.Tensor, inter: int) -> torch.Tensor:
        """fp32; the gate half first."""
        gate, up = x[:, : 2 * inter].float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor, inter: int) -> None:
        assert x.dim() == 2 and x.stride(1) == 1, "x [T, width] with unit column stride"
        assert x.shape[1] >= 2 * inter, "x carries the gate and up halves"
        assert x.dtype in (torch.bfloat16, torch.float16), "bf16 / fp16"


silu_and_mul_rows = SiluAndMulRows()
