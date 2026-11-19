# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MLA q_a RMSNorm that releases the q_b GEMM at launch (bitwise the FlashInfer RMSNorm)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm.q_norm_release import can_use
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm.q_norm_release import (
    k31_rmsnorm_release as _k31_rmsnorm_release,
)


def _norm(rows: int, row_stride: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, cols=1536, row_stride=row_stride))


class K31RmsnormRelease(OpWrapper):
    """``x / rms(x) * weight`` over the last dim of BF16 ``x [M, H]``: FlashInfer's CuTe DSL RMSNorm, bit for bit,
    with the programmatic-dependent-launch trigger at launch instead of after the store.

    The K3.1 MLA ``q_a`` norm in the decode / verify step: its dependent, the planned ``q_b`` GEMM, then streams its
    weights while the norm runs. ``x`` may be a column block of the packed DSA projection's output (unit column
    stride, row stride a multiple of 8 elements). Returns a contiguous tensor.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _norm(8, 2656, "TEP32 q_a block of the packed hidden projection, a width-8 verify"),
        _norm(16, 3040, "TEP16 q_a block of the packed hidden projection, two width-8 requests"),
        _norm(72, 14560, "DEP16 q_a block of the packed hidden projection, ~14 width-5 requests"),
        _norm(5, 1536, "a contiguous q_a (the unpacked path)"),
    )

    note = """
    Bitwise the engine's ``flashinfer_rmsnorm`` (same kernel code, same compile options). The release is safe only
    when the next kernel on the stream reads nothing but its weights before its own ``griddepcontrol.wait``: the
    low-M GEMMs of ``ops/gemm/low_m_linear`` do. Inputs outside ``can_use`` (not 2D BF16, column stride not 1,
    misaligned rows) are refused by the guard; call ``flashinfer_rmsnorm`` there.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _k31_rmsnorm_release(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """The engine's FlashInfer RMSNorm (PDL trigger after the store)."""
        return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        # The kernel walks rows with x's row stride in 16-byte vectors: a stride it cannot vectorize is read wrong.
        assert can_use(x, weight), "2D BF16 x with unit column stride, 16-byte aligned rows"


k31_rmsnorm_release = K31RmsnormRelease()
