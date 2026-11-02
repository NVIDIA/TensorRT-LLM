# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU with the SiLU and the product in fp32, rounded once to the projections' dtype."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Block 3's experts: the gate and up projections are [T, 2560] bf16.
_EXPERT = dict(intermediate=2560, dtype=torch.bfloat16)


class SwigluFp32(OpWrapper):
    """`silu(gate.float()) * up.float()`, rounded once to `gate.dtype`, every step returned.

    `gate` and `up` are an expert's two projections, `[T, I]` bf16 each. The steps, as the
    source model computes them::

        silu    = silu(gate.float())         [T, I] fp32
        product = silu * up.float()          [T, I] fp32
        act     = product.to(gate.dtype)     [T, I] bf16, the down projection's input

    Returns all three in a dict under those names, so a caller that needs only `act` and a
    test that checks each step observe the same call.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block 3 expert, 1 token", spec=dict(_EXPERT, tokens=1)),
        Cell(why="block 3 expert, 17 tokens", spec=dict(_EXPERT, tokens=17)),
        Cell(why="block 3 expert, 129 tokens: the longest context", spec=dict(_EXPERT, tokens=129)),
    )

    note = """
    Not the shared `flashinfer_silu_and_mul` entry. That kernel also computes in fp32 and
    rounds once, but it returns only the rounded activation: the fp32 SiLU and product it
    computed cannot be observed, and here they are outputs of the op.

    Rounding the SiLU or the product to bf16 before the final cast is a different answer:
    it rounds twice, and on inputs away from any bf16 rounding boundary the activation
    moves by a bf16 unit wherever the two roundings disagree.
    """

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> dict[str, torch.Tensor]:
        silu = F.silu(gate.float())
        product = silu * up.float()
        return dict(silu=silu, product=product, act=product.to(gate.dtype))

    def reference(self, gate: torch.Tensor, up: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each step in fp64, rounded to fp32 where the source holds fp32, then to `gate.dtype`.

        The product of an fp32 SiLU and a bf16 `up` is exact in fp64, so `product` is the
        correctly rounded fp32 product of the reference's own SiLU.
        """
        silu = F.silu(gate.double()).float()
        product = (silu.double() * up.double()).float()
        return dict(silu=silu, product=product, act=product.to(gate.dtype))

    def is_valid(self, gate: torch.Tensor, up: torch.Tensor) -> None:
        # Mismatched shapes broadcast, and a gate that is not bf16 removes (fp32) or moves
        # (fp16) the one rounding the down projection's input carries. Neither raises.
        assert gate.shape == up.shape, (
            f"gate {tuple(gate.shape)} and up {tuple(up.shape)} must have one shape"
        )
        assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16, (
            f"gate and up must be torch.bfloat16, got {gate.dtype} and {up.dtype}"
        )


swiglu_fp32 = SwigluFp32()
