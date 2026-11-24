# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's conv kernel projection and its side-0 convolution in one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.rubin_ull_dflash2_conv_proj_fold import GROUP, TAPS, TOKEN_TILE
from .dflash2_grouped_conv import _torch_dflash2_grouped_conv

#: Kimi K3.1 DFlash2 drafter: hidden 7168, kernel projection [2 * taps * groups, hidden] = [1792, 7168].
_HIDDEN = 7168


def _cell(why: str, rows: int, block: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, block=block, hidden=_HIDDEN))


class K31DFlash2ConvProjFold(OpWrapper):
    """``coef = x @ weight^T`` and ``conv[t] = sum_i (base[i] + coef[t, 0, i]) * x[t - i]`` in one launch.

    ``engine`` is a ``PreparedConvProj`` (``kernels/spec/rubin_ull_dflash2_conv_proj_fold.py``) holding the
    kernel projection ``[2 * taps * groups, hidden]``. ``coef`` is the projection's ``[rows, 2 * taps * groups]``
    output in its plain layout (the caller views side 1 for the sublayer's output convolution); ``conv`` is the
    side-0 convolution of ``x`` within each ``block_size`` block. Bitwise the vendored warp split-K GEMM of the
    same tactic followed by ``dflash2_grouped_conv``. Returns ``(coef, conv)``, new tensors.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("helix16 b1: one request's 8-row block (the served drafter step)", 8, 8),
        _cell("two 4-row blocks in one token tile: tap 1 stops at each block start", 8, 4),
        _cell("5 rows of one 5-row block: a partial token tile", 5, 5),
    )

    note = """
    The Helix16 drafter runs it at one request's 8-row block, for the
    attention and the MLP conv (10 launches a step, each replacing a GEMM and a convolution launch). The engine
    compiles on construction, which must happen outside CUDA-graph capture. Only 2 taps, group 16 and up to
    ``TOKEN_TILE`` rows; anything else keeps the GEMM and ``dflash2_grouped_conv``.
    """

    def __call__(self, engine, x: torch.Tensor, base_kernel: torch.Tensor, block_size: int):
        rows = x.shape[0]
        coef = torch.empty((rows, engine.weight.shape[0]), dtype=x.dtype, device=x.device)
        conv = torch.empty_like(x)
        return engine.run(x, base_kernel, coef, conv, block_size)

    def reference(self, engine, x: torch.Tensor, base_kernel: torch.Tensor, block_size: int):
        """The FP64 projection rounded once to BF16, then the torch convolution of its side 0."""
        rows, hidden = x.shape
        groups = hidden // GROUP
        coef = (x.double() @ engine.weight.double().t()).to(x.dtype)
        delta = coef.view(rows, 2, TAPS, groups)[:, 0]
        return coef, _torch_dflash2_grouped_conv(x, delta, base_kernel, block_size, GROUP)

    def is_valid(self, engine, x: torch.Tensor, base_kernel: torch.Tensor, block_size: int) -> None:
        rows, hidden = x.shape
        assert x.dtype == torch.bfloat16 and base_kernel.dtype == torch.bfloat16, (
            "BF16 x and base kernel"
        )
        assert 1 <= rows <= TOKEN_TILE, f"1..{TOKEN_TILE} rows (one token tile), got {rows}"
        assert x.is_contiguous() and x.data_ptr() % 32 == 0, "x must be packed and 32-byte aligned"
        assert tuple(engine.weight.shape) == (2 * TAPS * (hidden // GROUP), hidden), (
            f"the kernel projection of hidden {hidden} is [{2 * TAPS * (hidden // GROUP)}, {hidden}]"
        )
        assert tuple(base_kernel.shape) == (TAPS, hidden) and base_kernel.is_contiguous(), (
            f"base_kernel must be a contiguous [{TAPS}, {hidden}]"
        )
        assert base_kernel.data_ptr() % 32 == 0, "base_kernel must be 32-byte aligned"
        assert block_size >= 1, f"block_size must be positive, got {block_size}"


dflash2_conv_proj_fold = K31DFlash2ConvProjFold()

__all__ = ["K31DFlash2ConvProjFold", "dflash2_conv_proj_fold"]
