# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused MXFP8 activation quantization + low-M MXFP8 GEMM of the decode projections."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

_E4M3_MAX = 448.0


def _ue8m0_unswizzle(wsf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[N, K / 32] scale bytes out of the 128x4-swizzled buffer of ``mxfp8_quantize(w, True)``."""
    cols = k // 32
    padded_cols = (cols + 3) // 4 * 4
    r = torch.arange(n, device=wsf.device).view(-1, 1)
    c = torch.arange(cols, device=wsf.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    return wsf.reshape(-1)[index]


def _mxfp8_dequant_rows(x: torch.Tensor) -> torch.Tensor:
    """The activation as the kernel quantizes it (``trtllm::mxfp8_quantize``), dequantized back to fp32.

    Per 32 columns: scale = the smallest power of two at or above ``amax / 448`` (UE8M0, round up), data = x / scale
    rounded to nearest E4M3; an all-zero block stays zero.
    """
    m, k = x.shape
    blocks = x.float().view(m, k // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    bits = (amax / _E4M3_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    scale = torch.exp2(byte.float() - 127.0)
    data = torch.where(amax != 0, blocks / scale, torch.zeros_like(blocks)).to(torch.float8_e4m3fn)
    return (data.float() * scale).view(m, k)


# The target's configuration per (N, K) projection marked for this kernel, and the most rows it routes here
# (dispatch.LOW_M_GEMM_MAX_TOKENS): (warps, cluster_k, row_tiles, max rows).
CONFIGS = {
    (2560, 6144): (4, 4, 2, 8),
    (6144, 2048): (8, 2, 4, 8),
    (6144, 6144): (4, 4, 1, 1),
    (6144, 3072): (4, 2, 2, 8),
}


def _config_cells(named: tuple[Cell, ...]) -> tuple[Cell, ...]:
    """A cell for every (projection, rows) of ``CONFIGS`` that ``named`` does not list: certification is per
    projection, since each runs its own configuration."""
    listed = {(cell.spec["n"], cell.spec["k"], cell.spec["m"]) for cell in named}
    return tuple(
        Cell(
            why=f"{n} x {k}, {m} rows",
            spec=dict(m=m, n=n, k=k, warps=warps, cluster_k=cluster_k, row_tiles=row_tiles),
        )
        for (n, k), (warps, cluster_k, row_tiles, max_m) in CONFIGS.items()
        for m in range(1, max_m + 1)
        if (n, k, m) not in listed
    )


class Mxfp8LowMGemm(OpWrapper):
    """``out[M, N] = mxfp8(x)[M, K] @ dequant(w, wsf)[N, K]^T`` in BF16, quantizing ``x`` in the kernel.

    ``w`` is E4M3 [N, K] with UE8M0 block-32 scales ``wsf`` in the 128x4 swizzled layout; ``x`` BF16 [M, K] with unit
    column stride. ``(warps, cluster_k, row_tiles)`` is the kernel configuration the target's table picks per
    weight shape; each CTA owns ``16 * row_tiles`` weight rows and ``K / cluster_k`` columns, split-K partials meet
    in the cluster root and are summed in rank order, one BF16 rounding at the end.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The tp4 projections marked for this kernel (the decode GEMV runs them first and hands over when it declines),
    # each at the configuration the target's table assigns and up to the row count it routes here.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="QKV with the index q / k projections (2560 x 6144), one token",
            spec=dict(m=1, n=2560, k=6144, warps=4, cluster_k=4, row_tiles=2),
        ),
        Cell(
            why="QKV at the configured row limit, 8 tokens",
            spec=dict(m=8, n=2560, k=6144, warps=4, cluster_k=4, row_tiles=2),
        ),
        Cell(
            why="o_proj (6144 x 2048), 8 warps x 4 row tiles, odd row count",
            spec=dict(m=5, n=6144, k=2048, warps=8, cluster_k=2, row_tiles=4),
        ),
        Cell(
            why="dense MLP gate_up (6144 x 6144), configured for one token only",
            spec=dict(m=1, n=6144, k=6144, warps=4, cluster_k=4, row_tiles=1),
        ),
        Cell(
            why="dense MLP down (6144 x 3072), 8 tokens",
            spec=dict(m=8, n=6144, k=3072, warps=4, cluster_k=2, row_tiles=2),
        ),
    )
    # Every other (projection, rows) of its configurations.
    CELLS = CELLS + _config_cells(CELLS)

    note = """
    Only the four configurations the target's table assigns are built, each for 1..8 rows (one 8-row MMA tile).
    `pdl` changes launch overlap only and is driven off. Activation blocks whose amax is below `448 * 2^-127` take a
    flush-to-zero path the reference does not model; no decode activation is that small.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        wsf: torch.Tensor,
        warps: int,
        cluster_k: int,
        row_tiles: int,
        pdl: bool,
    ) -> torch.Tensor:
        return torch.ops.trtllm.minimax_m3_mxfp8_low_m_gemm(
            x, w, wsf, warps, cluster_k, row_tiles, pdl
        )

    def reference(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        wsf: torch.Tensor,
        warps: int,
        cluster_k: int,
        row_tiles: int,
        pdl: bool,
    ) -> torch.Tensor:
        """Dequantized fp32 product of the MXFP8 activation and weight, rounded to BF16 once.

        ``warps``, ``cluster_k``, ``row_tiles`` and ``pdl`` only pick the kernel's split; they do not enter the
        arithmetic.
        """
        n, k = w.shape
        w_scale = torch.exp2(_ue8m0_unswizzle(wsf, n, k).float() - 127.0)
        w_deq = (w.float().view(n, k // 32, 32) * w_scale.unsqueeze(-1)).view(n, k)
        with true_fp32_matmul():
            out = _mxfp8_dequant_rows(x) @ w_deq.t()
        return out.to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        wsf: torch.Tensor,
        warps: int,
        cluster_k: int,
        row_tiles: int,
        pdl: bool,
    ) -> None:
        n, k = w.shape
        # The kernel bulk-copies 256-byte scale chunks by computed offset; the op checks the dtype only, so a short
        # or linear-layout scale buffer is read past its end or in the wrong order.
        need = (n + 127) // 128 * 128 * ((k // 32 + 3) // 4 * 4)
        assert wsf.is_contiguous() and wsf.numel() >= need, (
            f"wsf must be the contiguous 128x4-swizzled UE8M0 buffer of {need} bytes; got {wsf.numel()}"
        )


mxfp8_low_m_gemm = Mxfp8LowMGemm()
