# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``F.linear`` without bias, optionally on TF32 tensor cores: the GEMM an unquantized engine
``Linear`` runs on its default path.

The K3.1 Rubin targets call it where an unquantized engine ``Linear`` would run ``F.linear``, so
those GEMMs are bitwise the engine's: the MLA ``q_b_proj``, ``g_proj`` and ``o_proj`` (default
``Linear`` path), the DSA indexer's fused ``[wk | weights_proj]`` projection (FP32 operands with
TF32 enabled), and the FP32 MoE router of the DEP16 generation server. ``tf32=True`` enables TF32
for this call only, as the engine indexer's ``_tf32_matmul_enabled`` does; ``tf32=False`` leaves
torch's matmul settings as they are.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class RubinLinear(OpWrapper):
    """``x @ weight.T``: ``x`` ``[M, K]``, ``weight`` ``[N, K]`` (the ``nn.Linear`` layout), one
    dtype; returns a new ``[M, N]`` tensor of that dtype. With ``tf32`` the call runs with
    ``torch.backends.cuda.matmul.allow_tf32 = True`` and restores the previous value after it."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MLA q_b_proj, TEP8 decode: 8 rows x 1536 -> 12 heads x 192",
            spec=dict(m=8, k=1536, n=2304, dtype=torch.bfloat16, tf32=False),
        ),
        Cell(
            why="MLA q_b_proj, DEP16 / helix8 decode: 96 rows x 1536 -> 96 heads x 192",
            spec=dict(m=96, k=1536, n=18432, dtype=torch.bfloat16, tf32=False),
        ),
        Cell(
            why="MLA o_proj, TEP8 decode: 8 rows x 12 heads x 128 -> 7168",
            spec=dict(m=8, k=1536, n=7168, dtype=torch.bfloat16, tf32=False),
        ),
        Cell(
            why="MLA g_proj, DEP16 prefill chunk: 2048 rows x 7168 -> 96 heads x 128",
            spec=dict(m=2048, k=7168, n=12288, dtype=torch.bfloat16, tf32=False),
        ),
        Cell(
            why="DSA indexer [wk | weights_proj] in FP32 with TF32: 96 rows x 7168 -> 128 + 32",
            spec=dict(m=96, k=7168, n=160, dtype=torch.float32, tf32=True),
        ),
        Cell(
            why="DSA indexer [wk | weights_proj], a 16k prefill chunk",
            spec=dict(m=16384, k=7168, n=160, dtype=torch.float32, tf32=True),
        ),
        Cell(
            why="FP32 MoE router of the DEP16 generation server: 96 rows x 7168 -> 896 experts",
            spec=dict(m=96, k=7168, n=896, dtype=torch.float32, tf32=False),
        ),
    )
    note = """
    The op is torch's cuBLASLt GEMM, so bits depend on cuBLAS's algorithm choice for the shape:
    the test compares with the reference within the accumulation band of the dtype (and TF32's
    10-bit mantissa where ``tf32`` is set), and with ``F.linear`` itself bit for bit.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, tf32: bool) -> torch.Tensor:
        if not tf32:
            return F.linear(x, weight)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return F.linear(x, weight)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def reference(self, x: torch.Tensor, weight: torch.Tensor, tf32: bool) -> torch.Tensor:
        """fp64 product rounded once to ``x``'s dtype; ``tf32`` only widens the test's band."""
        return (x.double() @ weight.double().t()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, tf32: bool) -> None:
        # Mixed operand dtypes are promoted by F.linear without an error: a BF16 activation
        # against an FP32 weight is an FP32 GEMM of the rounded activation, not the BF16 one.
        assert x.dtype == weight.dtype, f"x {x.dtype} and weight {weight.dtype} must match"


rubin_linear = RubinLinear()
