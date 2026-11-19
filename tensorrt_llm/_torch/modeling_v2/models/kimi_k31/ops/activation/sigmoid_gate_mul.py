# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MLA output gate ``attn_output * sigmoid(gate)`` in one kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.activation.sigmoid_gate_mul import (
    k31_sigmoid_gate_mul as _k31_sigmoid_gate_mul,
)


def _gate(rows: int, cols: int, why: str, gate_row_stride: int | None = None) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, cols=cols, gate_row_stride=gate_row_stride))


class K31SigmoidGateMul(OpWrapper):
    """``attn_output * gate.sigmoid()`` for BF16 ``[M, N]``, bit-identical to the two PyTorch
    kernels (FP32 ``1 / (1 + expf(-g))`` rounded to BF16, then the BF16 product).

    The MLA output gate of a Kimi K3.1 layer: ``attn_output`` is the attention output on its way
    into ``o_proj`` (``heads_per_rank * v_head_dim`` columns), ``gate`` is ``g_proj(hidden)`` or
    the gate columns of a packed projection. Returns a new tensor. One CuTe DSL kernel.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _gate(1, 96 * 128, "attention DP / TP1 decode row: 96 heads x 128"),
        _gate(8, 12 * 128, "TP8 DFlash2 verify step (1 + 7 rows), 12 heads x 128"),
        _gate(37, 6 * 128, "TP16 odd row count, 6 heads x 128 (one partial block)"),
        _gate(2954, 24 * 128, "TP4 prefill chunk, 24 heads x 128"),
        _gate(
            5,
            12 * 128,
            "gate as a column slice of a wider row-major GEMM output (the rows kernel)",
            gate_row_stride=12 * 128 + 2112,
        ),
        _gate(8, 3 * 128, "TEP32 verify step at B1 (8 rows), 3 heads x 128"),
        _gate(
            8,
            3 * 128,
            "TEP32 packed gate: 3 heads x 128 columns of the 2656-wide hidden projection",
            gate_row_stride=2656,
        ),
        _gate(72, 6 * 128, "TEP16 verify step at B12 (72 rows), 6 heads x 128"),
        _gate(99, 96 * 128, "attention-DP verify step, ~14 requests x 7 rows, 96 heads x 128"),
        _gate(256, 96 * 128, "attention-DP 256 rows: more than one wave of CTAs"),
        _gate(3, 100, "row length off the 8-element vector (one element per step)"),
        _gate(
            96, 96 * 128, "sm_107 Helix CP 8 b16 k5 / DEP16 b12 k7 verify step: 96 rows x 96 heads"
        ),
        _gate(16384, 12 * 128, "sm_107 TEP8 16k context chunk, 12 heads x 128"),
        _gate(16384, 96 * 128, "sm_107 DEP16 16k context chunk, 96 heads x 128"),
    )

    note = """
    Runs for every Kimi K3.1 MLA output gate, at every row count. attn_output must be
    contiguous; gate contiguous or a column slice of a row-major tensor (the op raises
    otherwise, as it does for other dtypes or shapes). g in about [-103, -87.3] gives a denormal
    sigmoid, kept as PyTorch keeps it. The kernel moves 16-byte vectors when the three base
    pointers are 16-byte aligned and the row length (and gate row stride) are multiples of 8,
    single elements otherwise; it is bitwise the same either way. Programmatic dependent launch
    follows TRTLLM_ENABLE_PDL (default on): the kernel waits for its producer before its first
    load and, at up to one wave of CTAs, releases its dependent once its loads are issued. The
    first call per variant compiles, so it must run outside CUDA-graph capture.
    """

    def __call__(self, attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return _k31_sigmoid_gate_mul(attn_output, gate)

    def reference(self, attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """The two PyTorch BF16 kernels the op replaces."""
        return attn_output * gate.sigmoid()

    def is_valid(self, attn_output: torch.Tensor, gate: torch.Tensor) -> None:
        """Nothing to add: the op raises on every layout it would read wrongly."""
        return None


k31_sigmoid_gate_mul = K31SigmoidGateMul()
