# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's latent_up behind the latent all-reduce row kernel, with its weight on chip before its
grid wait (``kernels/gemm/rubin_latent_up_resident.py``).

The row kernel releases its programmatic dependents right after its own grid wait, while it waits
for the other ranks. This GEMM is that dependent: at launch its CTAs load as much of their share
of the BF16 weight as their shared memory holds; after the wait they read the normed latent and
stream the rest of their weight from HBM.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: helix8's latent_up: ``[hidden, latent]`` BF16 (the weight is replicated on every rank).
HIDDEN = 7168
LATENT = 3584
#: The kernel's cluster split along K, its row bound and its default shared-memory budget.
SPLIT_K = 3
MAX_M = 128
DEFAULT_SMEM_BUDGET = 320 * 1024


@functools.cache
def _kernel():
    """The kernel module; imported on first use (it needs the GPU stack)."""
    from ...kernels.gemm import rubin_latent_up_resident as kernel

    return kernel


def k_chunks(k: int) -> list[tuple[int, int]]:
    """``(first, count)`` 128-wide K tiles of each cluster rank, front-heavy (the kernel's)."""
    tiles = k // 128
    base, extra = divmod(tiles, SPLIT_K)
    chunks, first = [], 0
    for rank in range(SPLIT_K):
        count = base + (1 if rank < extra else 0)
        chunks.append((first, count))
        first += count
    return chunks


def _cell(rows: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, n=HIDDEN, k=LATENT))


class RubinLatentUpResident(OpWrapper):
    """``x @ weight.T`` for BF16 ``x [M, K]`` (``1 <= M <= 128``) and the stored BF16 ``weight
    [N, K]`` (``N % 128 == 0``), FP32 accumulation rounded once to BF16; returns a new ``[M, N]``.

    Three cluster ranks each accumulate a front-heavy K chunk in order, and rank 0 adds the FP32
    partials in rank order. The launch is a programmatic dependent of the previous kernel on the
    stream; the weight is read before the grid wait, the activation after it.
    """

    ARCHS = frozenset({Arch.SM_107})

    # helix8's step rows (6 per request, k5 verify): b1, b2, b4, b9, b11, b16, b21 and the bound.
    CELLS: tuple[Cell, ...] = tuple(
        _cell(rows, why)
        for rows, why in (
            (6, "b1 verify rows (one 16-token tile, mostly padding)"),
            (12, "b2 verify rows"),
            (24, "b4 verify rows (32-token tile)"),
            (54, "b9 verify rows (64-token tile)"),
            (66, "b11 verify rows (80-token tile, two-stage activation ring)"),
            (96, "b16 verify rows (96-token tile)"),
            (126, "b21 verify rows"),
            (128, "the row bound (128-token tile)"),
        )
    )

    note = """
    The kernel compiles once per (token tile, stages, device) on its first call: make that call
    before CUDA-graph capture. It is not bitwise a single-CTA GEMM or ``cublas_mm``: it adds
    three FP32 partials in another order than one K loop (one BF16 rounding either way); its
    words are FlashInfer's cluster split-K over the same chunks. ``smem_budget`` (bytes per CTA,
    default 320 KiB) sizes the resident weight stages; the kernel's launch uses no more.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        smem_budget: int = DEFAULT_SMEM_BUDGET,
    ) -> torch.Tensor:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
        return _kernel().run_latent_up_resident(x, weight, out, smem_budget=smem_budget)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        smem_budget: int = DEFAULT_SMEM_BUDGET,
    ) -> torch.Tensor:
        """Each K chunk's FP32 product, summed in rank order, rounded once. TF32 off; the
        per-element order inside a chunk is torch's, so the test bounds the difference."""
        with true_fp32_matmul():
            acc = None
            for first, count in k_chunks(x.shape[1]):
                lo, hi = first * 128, (first + count) * 128
                part = x[:, lo:hi].float() @ weight[:, lo:hi].float().t()
                acc = part if acc is None else acc + part
        return acc.to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        smem_budget: int = DEFAULT_SMEM_BUDGET,
    ) -> None:
        """What the launcher refuses itself (ValueError), checked here before the call: the kernel
        reads ``x`` and ``weight`` as dense row-major tensors through TMA."""
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("rubin_latent_up_resident: x and weight must be BF16")
        if not (x.is_contiguous() and weight.is_contiguous()):
            raise ValueError("rubin_latent_up_resident: x and weight must be contiguous")
        if not 1 <= x.shape[0] <= MAX_M:
            raise ValueError(f"rubin_latent_up_resident: {x.shape[0]} rows outside 1..{MAX_M}")
        if x.shape[1] != weight.shape[1] or x.shape[1] % 128 or x.shape[1] // 128 < SPLIT_K:
            raise ValueError(f"rubin_latent_up_resident: K {x.shape[1]} vs weight {weight.shape}")
        if weight.shape[0] % 128:
            raise ValueError(f"rubin_latent_up_resident: N {weight.shape[0]} not a multiple of 128")


rubin_latent_up_resident = RubinLatentUpResident()

__all__ = ["RubinLatentUpResident", "rubin_latent_up_resident", "k_chunks"]
