# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul into a caller-owned output; local variant of the catalog bmm_out entry with this target's cells."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, written (not accumulated) into `out`, which is returned."""

    ARCHS = frozenset({Arch.SM_103})

    # The first four cells are the shared entry's; the rest are this checkpoint's latent -> v projection.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, q_nope -> latent: [H, 1, nope] @ [H, nope, kv_lora]",
            spec=dict(batch=128, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="decode, latent -> v: [H, 1, kv_lora] @ [H, kv_lora, v_dim]",
            spec=dict(batch=128, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="prefill token count, the same absorbing orientation",
            spec=dict(batch=128, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="contiguous out, to separate the op's own correctness from the "
            "transposed-view path the targets happen to use",
            spec=dict(batch=128, m=64, k=128, n=512, dtype=torch.bfloat16, transposed_out=False),
        ),
        Cell(
            why="hy4 decode, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 6-row verify step, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=6, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 512-row block, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=512, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 5120-token prefill chunk, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=5120, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode, latent -> v: 64 SWA heads",
            spec=dict(batch=64, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode at 8 heads per rank: sparse 32/4 at tp4 or SWA 64/8 at tp8",
            spec=dict(batch=8, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 512-row block at 8 heads per rank: sparse at tp4 or SWA at tp8",
            spec=dict(batch=8, m=512, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode at 16 heads per rank: SWA 64/4 at tp4",
            spec=dict(batch=16, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 5120-token prefill chunk through the SWA layers: 64 heads, "
            "the envelope's largest workload",
            spec=dict(batch=64, m=5120, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
    )

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated with TF32 off."""
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
        # A wrong-shaped `out` is silently re-allocated, detaching it from the caller's buffer.
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
        )
        # Mixed dtypes are promoted rather than rejected.
        assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"


bmm_out = BmmOut()
