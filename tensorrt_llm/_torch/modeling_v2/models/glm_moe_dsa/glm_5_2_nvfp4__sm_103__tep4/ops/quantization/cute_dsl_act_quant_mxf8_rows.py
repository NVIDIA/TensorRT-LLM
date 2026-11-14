# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rows of an MXF8 GEMM operand from bf16 at a row offset: the activation quant's rule
(1x32 blocks, power-of-two scales, e4m3 values) into rows `row0 ..` of an operand whose
earlier rows another kernel wrote, one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _kernel

CompiledActQuantRows = _kernel.CompiledKernel
sf_bytes = _kernel.sf_bytes
padded_rows = _kernel.padded_rows
#: The e8m0 code of 2^0, which the padding scale rows get.
_UNIT_CODE = 127


def compile(device_index: int) -> CompiledActQuantRows:  # noqa: A001 -- the kernel's own name
    """Compile the kernel for `device_index`; return the handle the op takes as `compiled`.
    Call once at load; one compile serves every row count, offset and width. Launches once on
    a one-row case, so no step meets a first module load."""
    fn = _kernel.compile_quant_rows(int(device_index))
    dev = torch.device("cuda", int(device_index))
    with torch.cuda.device(dev):
        q = torch.empty(2, 128, dtype=torch.uint8, device=dev)
        sf = torch.empty(sf_bytes(2, 128), dtype=torch.uint8, device=dev)
        _kernel.quantize_rows(fn, torch.zeros(1, 128, dtype=torch.bfloat16, device=dev), q, sf, 1)
        torch.cuda.synchronize(dev)
    return fn


def _sf_offsets(rows: torch.Tensor, k: int) -> torch.Tensor:
    """Byte offsets of rows `rows`' scales in the blocked layout of a `k`-wide operand."""
    kf = k // 32
    m = rows[:, None]
    f = torch.arange(kf, device=rows.device)[None, :]
    return (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + f % 4 + (f // 4) * 512 + (m // 128) * 512 * (k // 128)
    )


class CuteDslActQuantMxf8Rows(OpWrapper):
    """Write bf16 `x [R, K]` (K a multiple of 128, unit inner stride, row pitch a multiple of
    8, 16-byte aligned) as rows `row0 .. row0 + R - 1` of the MXF8 operand `(q [T, K] uint8,
    sf)`: per 32-element block a power-of-two scale (the block's amax / 448, rounded up) and
    e4m3 values, the activation quant's bytes. The scale rows from `row0 + R` to the next
    multiple of 128 get 2^0. Rows of `q` and scales below `row0` are not touched. Returns
    `(q, sf)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="16 generation rows after 7239 context rows of o_proj's operand",
            spec=dict(row0=7239, rows=16, k=4096),
        ),
        Cell(
            why="58 generation rows starting at a 128-row block boundary",
            spec=dict(row0=7168, rows=58, k=4096),
        ),
        Cell(
            why="no rows: only the padding scale rows past the context rows",
            spec=dict(row0=1000, rows=0, k=4096),
        ),
    )

    note = """
    The target takes this op on a mixed step of the fp8 context attention: the MHA's epilogue
    writes the context rows of o_proj's MXF8 operand, and the generation rows' attention output
    (bf16, from the decode attention) is quantized into the rows after them, so o_proj runs one
    GEMM on the whole operand without its own quant launch.
    """

    def __call__(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        row0: int,
        compiled: CompiledActQuantRows,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _kernel.quantize_rows(compiled, x, q, sf, int(row0))
        return q, sf

    def reference(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        row0: int,
        compiled: CompiledActQuantRows,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(q, sf)` as the op leaves them, as new tensors: the activation quant's torch twin on
        `x`, placed at `row0`, and the padding scale rows at 2^0. `compiled` is ignored."""
        q, sf = q.clone(), sf.clone()
        rows, k = int(x.shape[0]), int(q.shape[1])
        dev = q.device
        if rows:
            xq, xsf = _kernel.reference(x.contiguous())
            q[row0 : row0 + rows] = xq
            src = _sf_offsets(torch.arange(rows, device=dev), k)
            dst = _sf_offsets(torch.arange(row0, row0 + rows, device=dev), k)
            sf[dst.flatten()] = xsf[src.flatten()]
        pad = torch.arange(row0 + rows, padded_rows(row0 + rows), device=dev)
        if pad.numel():
            sf[_sf_offsets(pad, k).flatten()] = _UNIT_CODE
        return q, sf

    def is_valid(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        sf: torch.Tensor,
        row0: int,
        compiled: CompiledActQuantRows,
    ) -> None:
        rows, k = int(x.shape[0]), int(x.shape[1])
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and k % 128 == 0, (
            f"x must be bf16 [R, K] with K a multiple of 128, got {x.dtype} {tuple(x.shape)}"
        )
        assert q.dtype == torch.uint8 and q.is_contiguous() and int(q.shape[1]) == k, (
            f"q must be a contiguous uint8 [T, {k}]"
        )
        assert 0 <= row0 and row0 + rows <= int(q.shape[0]), (
            f"rows {row0}..{row0 + rows} must fit q's {int(q.shape[0])}"
        )
        assert sf.dtype == torch.uint8 and sf.numel() >= sf_bytes(row0 + rows, k), (
            f"sf must hold at least {sf_bytes(row0 + rows, k)} bytes"
        )


cute_dsl_act_quant_mxf8_rows = CuteDslActQuantMxf8Rows()
