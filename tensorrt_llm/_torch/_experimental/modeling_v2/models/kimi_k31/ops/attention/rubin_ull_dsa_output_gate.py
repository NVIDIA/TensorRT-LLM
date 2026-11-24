# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA output gate ``out = x * sigmoid(gate)`` in one Triton launch
(``kernels/attention/rubin_ull_mla_output_gate.py`` ``_sigmoid_gate_mul_kernel``, the K3 MLA output-gate
kernel), into a caller's buffer."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _cell(why: str, rows: int, scale: float = 2.0) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, scale=scale))


class DsaOutputGate(OpWrapper):
    """``out = bf16(x * bf16(sigmoid(gate)))`` elementwise over ``n = x.numel()`` BF16 values, bitwise
    PyTorch's ``x * gate.sigmoid()`` (two BF16 kernels): the sigmoid in FP32 (``expf``, a correctly
    rounded division) rounded to BF16, then the FP32 product rounded to BF16, denormals kept.
    ``x``, ``gate`` and ``out`` are contiguous BF16 tensors of one shape on one GPU. Returns
    ``out``.

    ``kernel``: the Triton kernel; ``block``: its BLOCK (the grid is ``ceil(n / block)``).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b2 rows on the fmha path without the gated Helix tail: 16 x 6 heads x 128", 16),
        _cell("32 rows", 32),
        _cell("one row (a partial block)", 1),
        _cell("gates where the sigmoid is denormal or saturates (|g| up to 120)", 8, scale=60.0),
    )

    note = """
    The K3.1 ULL generation target's DSA gate where neither the DSA core's value stage nor the
    fused Helix tail applied it (``dsa.sigmoid_gate_into`` via ``FlatDsa.gate_rows``, from the
    fused o-projection's output stage): the fmha fallback core only, not the served core path.
    Not PDL.
    """

    def __call__(self, kernel, block: int, x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor):
        import triton

        n = x.numel()
        kernel[(triton.cdiv(n, block),)](x, gate, out, n, BLOCK=block, num_warps=4)
        return out

    def reference(self, kernel, block: int, x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor):
        """PyTorch's two BF16 kernels (the kernel's own contract is bitwise these)."""
        return x * gate.sigmoid()

    def is_valid(self, kernel, block: int, x: torch.Tensor, gate: torch.Tensor, out: torch.Tensor):
        """The kernel reads ``n`` contiguous values from each pointer."""
        if (
            any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x, gate, out))
            or not x.shape == gate.shape == out.shape
            or not x.is_cuda
        ):
            raise ValueError("dsa_output_gate: contiguous BF16 rows of one shape")


dsa_output_gate = DsaOutputGate()
