# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram table's per-row e4m3 round trip, HF's ``_fp8_round_trip`` (in bf16)::

    scale = clamp(row_amax, min=1e-12) / 448
    q = clamp(x / scale, -448, 448).to(torch.float8_e4m3fn).to(bf16) * scale

HF's straight-through ``x + (q - x)`` is exactly ``q`` in bf16, so ``q`` is returned.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0
#: HF's amax floor: a zero row keeps a zero scale and rounds to 0, not NaN.
_SCALE_MIN = 1e-12


class EngramFp8Roundtrip(OpWrapper):
    """``rows`` bf16 ``[tokens * 16 heads, 96]`` -> bf16, each row e4m3-round-tripped."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token: 16 gathered heads", spec=dict(rows=16)),
        Cell(why="a six-token DFly2 verify step: 96 rows", spec=dict(rows=96)),
        Cell(why="CUDA-graph batch of 64 decode tokens: 1024 rows", spec=dict(rows=1024)),
        Cell(why="a 512-token chunked-prefill chunk: 8192 rows", spec=dict(rows=8192)),
        Cell(why="the 8192-token max_num_tokens envelope: 131072 rows", spec=dict(rows=131072)),
    )
    note = """
    Not `torch.ops.trtllm.quantize_e4m3_activation`: it floors the scale at 1 / (448 * 512) and so
    computes a different function on dim rows (amax below 2e-3).
    """

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_SCALE_MIN) / _E4M3_MAX
        quantized = (rows / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        return quantized.to(rows.dtype) * scale

    def reference(self, rows: torch.Tensor) -> torch.Tensor:
        """HF's function with the quotient and dequant multiply in fp32; the scale stays bf16 as HF defines it."""
        scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_SCALE_MIN) / _E4M3_MAX
        x = rows.float() / scale.float()
        quantized = x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        return (quantized.float() * scale.float()).to(rows.dtype)


engram_fp8_roundtrip = EngramFp8Roundtrip()
