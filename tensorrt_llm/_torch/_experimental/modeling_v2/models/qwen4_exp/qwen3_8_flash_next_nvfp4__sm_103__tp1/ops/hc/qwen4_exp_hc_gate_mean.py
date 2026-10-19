# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer epilogue: the sigmoid gate applied to the normed
stream and the mean over streams, from the up projection's logits, in one
Triton kernel.

Tier 5 (Triton written for this target). Used where the up projection runs on
cuBLAS (decode below `_HC_FUSED_UP_MIN_ROWS` rows): `qwen4_exp_hc_lowrank_act` ->
`cublas_mm(low, up^T)` -> this. Elsewhere `qwen4_exp_hc_up_gate_mean` computes the
up projection and this epilogue in one kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _hc_gate_mean_kernel(
    gate_ptr,
    normed_ptr,
    mixed_ptr,
    stride_gate,
    stride_normed,
    stride_mixed,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = h < HIDDEN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(HC):
        g = tl.load(gate_ptr + row * stride_gate + s * HIDDEN + h, mask=mask, other=0.0)
        x = tl.load(normed_ptr + row * stride_normed + s * HIDDEN + h, mask=mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    tl.store(
        mixed_ptr + row * stride_mixed + h,
        (acc * (1.0 / HC)).to(mixed_ptr.dtype.element_ty),
        mask=mask,
    )


class Qwen4ExpHcGateMean(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `gate_logits` and `normed` both
    `[T, C*H]` stream-major::

        mixed[t, i] = mean over s of sigmoid(gate_logits[t, s*H + i]) * normed[t, s*H + i]

    fp32 inside, one rounding. Returns a fresh `[T, H]` tensor in `normed`'s
    dtype. Fusion boundary: the sigmoid, the product and the stream mean; the
    up projection producing `gate_logits` is the caller's (`cublas_mm`).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, hc_count (4), hidden (2560), dtype (bf16).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560 bf16, {why}", spec=dict(rows=t))
        for t, why in (
            (1, "1 row"),
            (128, "128 rows"),
            (2048, "prefill 2048"),
            (8192, "prefill 8192"),
        )
    ) + (
        Cell(why="hc_count 2, 64 rows", spec=dict(rows=64, hc_count=2)),
        Cell(why="hc_count 8, 64 rows", spec=dict(rows=64, hc_count=8)),
        Cell(why="hidden 128 (not the checkpoint), 48 rows", spec=dict(rows=48, hidden=128)),
        Cell(why="float16 operands, 64 rows", spec=dict(rows=64, dtype=torch.float16)),
        Cell(why="zero rows", spec=dict(rows=0)),
    )

    note = """
    Metadata consumed: none. Call sites: the trunk's hyper-connection sites in
    `DecodeTarget` below `_HC_FUSED_UP_MIN_ROWS` rows, after `cublas_mm(low, up^T)`. Preconditions: both
    operands 2-D `[T, C*H]` with unit column stride. Graph-capturable.
    Measured (GB300, graph): the three-launch cuBLAS route this ends is 2.7x
    faster than `qwen4_exp_hc_gate_mix` at 512..8192 rows (106 vs 287 us at
    8192) and 1.7x at 128.
    """

    def __call__(
        self, gate_logits: torch.Tensor, normed: torch.Tensor, hc_count: int
    ) -> torch.Tensor:
        rows, width = normed.shape
        hidden = width // hc_count
        mixed = torch.empty((rows, hidden), dtype=normed.dtype, device=normed.device)
        if rows > 0:
            block = min(1024, triton.next_power_of_2(hidden))
            _hc_gate_mean_kernel[(rows, triton.cdiv(hidden, block))](
                gate_logits,
                normed,
                mixed,
                gate_logits.stride(0),
                normed.stride(0),
                mixed.stride(0),
                HIDDEN=hidden,
                HC=hc_count,
                BLOCK=block,
                num_warps=4,
            )
        return mixed

    def reference(
        self, gate_logits: torch.Tensor, normed: torch.Tensor, hc_count: int
    ) -> torch.Tensor:
        """float64, one rounding."""
        hidden = normed.shape[1] // hc_count
        gate = torch.sigmoid(gate_logits.double()).unflatten(-1, (hc_count, hidden))
        x = normed.double().unflatten(-1, (hc_count, hidden))
        return (gate * x).mean(dim=-2).to(normed.dtype)

    def is_valid(self, gate_logits: torch.Tensor, normed: torch.Tensor, hc_count: int) -> None:
        assert gate_logits.shape == normed.shape, "gate_logits and normed must match"
        assert gate_logits.stride(1) == 1 and normed.stride(1) == 1, "unit column stride"
        assert normed.shape[1] % hc_count == 0


qwen4_exp_hc_gate_mean = Qwen4ExpHcGateMean()
