# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply (Kimi K3's MLP activation) via the fused Triton situ_and_mul kernel."""

from typing import Optional

import torch

import tensorrt_llm._torch.modules.situ  # noqa: F401 — registers torch.ops.trtllm.situ_and_mul
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class SituAndMul(OpWrapper):
    """Return `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)` for `[g | u] = x`.

    `g, u = x[:, :d], x[:, d:]` with `d = x.shape[-1] // 2`; `linear_beta=None` leaves `u` as is.
    A new `[M, d]` tensor in `x.dtype`.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 width {width}, M={rows}: shared/dense TP16/TP4 decode-to-prefill",
                spec=dict(shape=(rows, width), dtype=torch.bfloat16, beta=4.0, linear_beta=25.0),
            )
            for width in (768, 3072, 4224, 16896)
            for rows in (1, 8, 64, 2048)
        ),
        *(
            Cell(
                why=f"width 6144 with caps ({beta}, {linear_beta}): asymmetric or uncapped up",
                spec=dict(
                    shape=(33, 6144), dtype=torch.bfloat16, beta=beta, linear_beta=linear_beta
                ),
            )
            for beta, linear_beta in ((2.5, 7.0), (1.0, None), (4.0, None))
        ),
        *(
            Cell(
                why=f"{dtype} output rounding at width 1536",
                spec=dict(shape=(17, 1536), dtype=dtype, beta=4.0, linear_beta=25.0, scale=3.0),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        ),
        Cell(
            why="row-strided column slice: only the last dimension must be dense",
            spec=dict(
                shape=(64, 768),
                dtype=torch.bfloat16,
                beta=4.0,
                linear_beta=25.0,
                row_width=2048,
                scale=1.0,
            ),
        ),
        Cell(
            why="saturated positive, negative and zero gates at magnitude 1e4",
            spec=dict(
                shape=(4, 512), dtype=torch.float32, beta=4.0, linear_beta=25.0, saturated=True
            ),
        ),
    )
    note = """
    Activation and gating only, without quantization; x is read-only. The caller
    owns the projections. Dense layers and shared experts call this entry;
    routed experts use act_type=3 inside the MXFP8 x MXFP4 MoE runner.
    Positive beta and positive linear_beta (or None) describe soft caps. Neither
    is checked: zero produces NaN at a zero input, otherwise a zero factor;
    negative caps compute the same values as their magnitude.

    No metadata or state is consumed. Triton specializes on dtype, whether the
    up cap is None, and divisibility of widths, strides and pointers. Warm each
    specialization outside CUDA-graph capture; replay with rewritten inputs is
    bit-identical to eager. libdevice tanh differs from torch in the last fp32
    bits. SituAndMul's use_fused_activation=True CUDA path calls this op; its
    other path evaluates the same formula eagerly.
    """

    def __call__(
        self, x: torch.Tensor, beta: float, linear_beta: Optional[float] = None
    ) -> torch.Tensor:
        return torch.ops.trtllm.situ_and_mul(x, beta, linear_beta)

    def reference(
        self, x: torch.Tensor, beta: float, linear_beta: Optional[float] = None
    ) -> torch.Tensor:
        """fp32 reference from native torch ops: SiTU on the gate half times the (soft-capped) up half."""
        d = x.shape[-1] // 2
        gate = x[:, :d].float()
        up = x[:, d:].float()
        situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        return (situ * up).to(x.dtype)


situ_and_mul = SituAndMul()
