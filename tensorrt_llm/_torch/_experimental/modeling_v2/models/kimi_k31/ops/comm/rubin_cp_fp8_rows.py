# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 rows of the cep KDA Ulysses all-to-alls (``kda_split: "ulysses_fp8"``): BF16 rows packed as E4M3
codes with one FP32 scale per row and 128 columns into the byte rows of a send buffer, and back.

Halves the bytes of the two all-to-alls (q / k / v before the convolution, the recurrence output
before the gated norm), as the generic Kimi K3.1 context parallelism does
(``TLLM_KIMI_K31_CP_FP8_A2A``). A byte row: groups of 128 codes at ``code_off + j * 128``, their
scales at ``scale_off + j * 4``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.comm.rubin_cp_fp8_rows import fp8_pack_rows, fp8_unpack_rows


def _scales(groups: torch.Tensor) -> torch.Tensor:
    return groups.abs().amax(-1).clamp_min(1e-12) / 448.0


class RubinCpFp8Pack(OpWrapper):
    """Pack BF16 ``x`` ``[R, G * 128]`` into uint8 ``buf``: group ``g`` of row ``r`` goes to byte row
    ``(g // gd) * dest_stride + r * row_stride`` (codes at ``code_off + (g % gd) * 128``, the FP32 scale
    ``amax / 448`` at ``scale_off + (g % gd) * 4``)."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = (
        Cell(
            why="cep8 Ulysses q (or k, v) section: 96 heads to 8 ranks of 12, 2048 rows",
            spec=dict(rows=2048, groups=96, gd=12, dests=8),
        ),
        Cell(
            why="cep8 Ulysses recurrence output: 12 heads of 8 x 2048 rows, one destination per row",
            spec=dict(rows=16384, groups=12, gd=12, dests=1),
        ),
    )
    note = """
    Triton, one program per 32 rows and group; E4M3 with saturation at 448.
    Bit-exact against the reference (the same scale and rounding).
    """

    def __call__(
        self,
        x: torch.Tensor,
        buf: torch.Tensor,
        dest_stride: int,
        row_stride: int,
        gd: int,
        code_off: int,
        scale_off: int,
    ) -> None:
        fp8_pack_rows(x, buf, dest_stride, row_stride, gd, code_off, scale_off)

    def reference(
        self,
        x: torch.Tensor,
        buf: torch.Tensor,
        dest_stride: int,
        row_stride: int,
        gd: int,
        code_off: int,
        scale_off: int,
    ) -> None:
        """Group by group in fp32."""
        rows, groups = x.shape[0], x.shape[1] // 128
        xg = x.float().view(rows, groups, 128)
        scale = _scales(xg)
        codes = (xg / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        flat = buf.view(-1)
        for g in range(groups):
            j = g % gd
            for r in range(rows):
                base = (g // gd) * dest_stride + r * row_stride
                flat[base + code_off + j * 128 : base + code_off + (j + 1) * 128] = codes[
                    r, g
                ].view(torch.uint8)
                at = base + scale_off + j * 4
                flat[at : at + 4] = scale[r, g].reshape(1).view(torch.uint8)


rubin_cp_fp8_pack = RubinCpFp8Pack()


class RubinCpFp8Unpack(OpWrapper):
    """Unpack rows ``[0, rows)`` of the ``num_src`` byte blocks of ``buf`` (block stride
    ``src_stride``) into BF16 ``out``: slot ``j`` of block ``s`` lands in columns
    ``out_col_off + (s * gd + j) * 128``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = (
        Cell(
            why="cep8 Ulysses head side: one q / k / v section of 12 heads, 16384 rows",
            spec=dict(rows=16384, gd=12, num_src=1),
        ),
        Cell(
            why="cep8 Ulysses row side: 8 ranks' 12 heads of 2048 rows",
            spec=dict(rows=2048, gd=12, num_src=8),
        ),
    )
    note = """
    Triton, one program per 64 rows, slot and source block. Bit-exact against
    the reference.
    """

    def __call__(
        self,
        buf: torch.Tensor,
        out: torch.Tensor,
        rows: int,
        num_src: int,
        src_stride: int,
        row_stride: int,
        gd: int,
        code_off: int,
        scale_off: int,
        out_col_off: int,
    ) -> None:
        fp8_unpack_rows(
            buf, out, rows, num_src, src_stride, row_stride, gd, code_off, scale_off, out_col_off
        )

    def reference(
        self,
        buf: torch.Tensor,
        out: torch.Tensor,
        rows: int,
        num_src: int,
        src_stride: int,
        row_stride: int,
        gd: int,
        code_off: int,
        scale_off: int,
        out_col_off: int,
    ) -> None:
        """Slot by slot in fp32."""
        flat = buf.view(-1)
        for s in range(num_src):
            for r in range(rows):
                base = s * src_stride + r * row_stride
                for j in range(gd):
                    codes = flat[base + code_off + j * 128 : base + code_off + (j + 1) * 128]
                    at = base + scale_off + j * 4
                    scale = flat[at : at + 4].view(torch.float32)
                    col = out_col_off + (s * gd + j) * 128
                    out[r, col : col + 128] = (codes.view(torch.float8_e4m3fn).float() * scale).to(
                        out.dtype
                    )


rubin_cp_fp8_unpack = RubinCpFp8Unpack()
