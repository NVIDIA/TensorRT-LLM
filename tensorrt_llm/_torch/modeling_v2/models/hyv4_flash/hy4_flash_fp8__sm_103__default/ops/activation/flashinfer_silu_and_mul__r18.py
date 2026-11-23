# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU via flashinfer silu_and_mul; local variant of the catalog entry with this target's cells."""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FlashinferSiluAndMul(OpWrapper):
    """`silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, new tensor out."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: layer-0 dense MLP gate+up, 2 x 13824",
            spec=dict(shape=(1, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="64-row end: layer-0 dense MLP gate+up, 2 x 13824",
            spec=dict(shape=(64, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill chunk: layer-0 dense MLP gate+up over 5120 rows",
            spec=dict(shape=(5120, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="decode: shared expert gate+up, 2 x 1536",
            spec=dict(shape=(1, 3072), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill chunk: shared expert gate+up over 5120 rows",
            spec=dict(shape=(5120, 3072), dtype=torch.bfloat16),
        ),
        Cell(
            why="drafter MLP at the five-draft width, 2 x 9216",
            spec=dict(shape=(6, 18432), dtype=torch.bfloat16),
        ),
    )

    note = "Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`)."

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_silu_and_mul(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor) -> None:
        # The kernel vectorizes over each half; a misaligned second half faults the CUDA context.
        width = x.shape[-1]
        assert width % 2 == 0, f"x.shape[-1] must be even to split in half; got {width}"
        half_bytes = (width // 2) * x.element_size()
        assert half_bytes % 16 == 0, (
            f"x.shape[-1] // 2 must be a whole number of 16-byte vectors; got {half_bytes} B"
        )
        assert half_bytes >= 16, (
            f"x.shape[-1] // 2 must hold at least one 16-byte vector; got {half_bytes} B"
        )


flashinfer_silu_and_mul = FlashinferSiluAndMul()
