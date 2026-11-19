# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's MXFP8 routed experts as a grouped MoE.

One rank's experts `[offset, offset + E)` over every token of the step; the output is this
rank's partial sum. Per token and chosen expert `e` with weight `w`:

    x^   = mxfp8(x)                    E4M3 values, one power-of-two scale per 32 along K
    h    = situ(x^ @ gate_e^T, x^ @ up_e^T)
    out += w * mxfp8(h) @ down_e^T

`mxfp8` rounds each block's scale up to a power of two, `2 ** ceil(log2(amax / 448))`, and the
values to the nearest E4M3. Weights are the checkpoint's: `w13` `[E, 2I, H]` E4M3 with gate `j`
at row `2j` and up `j` at row `2j + 1`, `w2` `[E, H, I]`, and UE8M0 scale exponents `[E, rows,
K / 32]` (scale `2 ** (s - 127)`).

The routes are grouped by expert on the device and both GEMMs run as one Triton launch each
(`kernels/mxfp8_moe.py`): the kernels expand the E4M3 weights to bf16 in registers, which is
exact, and accumulate in fp32, so only the summation order differs from an MXFP8 MMA. Experts'
outputs meet in fp32 atomic adds. Nothing synchronizes with the host.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

from ...kernels.mxfp8_moe import K_TILE, grouped_mxfp8_moe

#: Values per MXFP8 scale.
BLOCK = 32
_E4M3_MAX = 448.0
_MIN_SCALE = 2.0**-127

_EP4 = dict(experts=384, local=96, hidden=4608, inter=2304, top_k=8)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, K]` -> (E4M3 values `[N, K]`, uint8 scale exponents `[N, K / 32]`)."""
    n, k = x.shape
    groups = x.float().view(n, k // BLOCK, BLOCK)
    raw = groups.abs().amax(dim=-1) / _E4M3_MAX
    # Round the scale up to a power of two: add the largest mantissa and keep the exponent.
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= _MIN_SCALE, torch.zeros_like(exponent), exponent.clamp(max=254))
    scale = torch.clamp((exponent << 23).view(torch.float32), min=_MIN_SCALE)
    inverse = torch.where(exponent == 0, torch.full_like(scale, 2.0**127), 1.0 / scale)
    values = (groups * inverse.unsqueeze(-1)).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return values.view(n, k), exponent.to(torch.uint8)


def mxfp8_dequantize(
    values: torch.Tensor, exponents: torch.Tensor, dtype=torch.float32
) -> torch.Tensor:
    """E4M3 values `[..., K]` and uint8 exponents `[..., K / 32]` -> `[..., K]` in `dtype`."""
    scale = torch.exp2(exponents.view(torch.uint8).float() - 127.0).clamp(min=_MIN_SCALE)
    out = values.float().unflatten(-1, (-1, BLOCK)) * scale.unsqueeze(-1)
    return out.flatten(-2).to(dtype)


def _situ(gate, up, gate_softcap, up_softcap):
    return (
        gate_softcap
        * torch.tanh(gate / gate_softcap)
        * torch.sigmoid(gate)
        * up_softcap
        * torch.tanh(up / up_softcap)
    )


class Inkling2Mxfp8Experts(OpWrapper):
    """This rank's MXFP8 experts: `x` `[T, H]` bf16, `topk_ids` `[T, k]` int32 (global ids),
    `topk_weights` `[T, k]` fp32 -> `[T, H]` bf16, the sum over the chosen experts this rank holds."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="EP4 rank 2, one decode token", spec=dict(_EP4, tokens=1, rank=2)),
        Cell(why="EP4 rank 0, 64 tokens", spec=dict(_EP4, tokens=64, rank=0)),
        Cell(
            why="EP4 rank 3, 300 tokens: several row blocks per expert",
            spec=dict(_EP4, tokens=300, rank=3),
        ),
    )

    note = """
    Both GEMM inputs are MXFP8-quantized, the SiTU output from fp32. The gate and up rows
    interleave per row in the checkpoint, unlike the NVFP4 layers' kernel layout; reading
    them as two halves still gives a finite, wrong result. The fp32 atomics make the last
    bits of repeated calls differ.
    """

    def __call__(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_s: torch.Tensor,
        w2: torch.Tensor,
        w2_s: torch.Tensor,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> torch.Tensor:
        x_hat = mxfp8_dequantize(*mxfp8_quantize(x), dtype=torch.bfloat16)
        out = grouped_mxfp8_moe(
            x_hat,
            topk_ids,
            topk_weights,
            w13,
            w13_s,
            w2,
            w2_s,
            local_expert_offset,
            gate_softcap,
            up_softcap,
        )
        return out.to(x.dtype)

    def reference(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_s: torch.Tensor,
        w2: torch.Tensor,
        w2_s: torch.Tensor,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> torch.Tensor:
        """True fp32 GEMMs on the fp32-dequantized weights and activations, gate and up taken from
        the even and odd rows."""
        local = w13.shape[0]
        x_hat = mxfp8_dequantize(*mxfp8_quantize(x))
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in range(local):
            tokens, slots = torch.where(topk_ids == local_expert_offset + e)
            if tokens.numel() == 0:
                continue
            fc1 = mxfp8_dequantize(w13[e], w13_s[e])
            down = mxfp8_dequantize(w2[e], w2_s[e])
            with true_fp32_matmul():
                h = _situ(
                    x_hat[tokens] @ fc1[0::2].t(),
                    x_hat[tokens] @ fc1[1::2].t(),
                    gate_softcap,
                    up_softcap,
                )
                y = mxfp8_dequantize(*mxfp8_quantize(h)) @ down.t()
            out.index_add_(0, tokens, y * topk_weights[tokens, slots].float().unsqueeze(1))
        return out.to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_s: torch.Tensor,
        w2: torch.Tensor,
        w2_s: torch.Tensor,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> None:
        local, rows, hidden = w13.shape
        assert w13.dtype == w2.dtype == torch.float8_e4m3fn, "MXFP8 weights are E4M3"
        assert w13_s.shape == (local, rows, hidden // BLOCK) and w13_s.dtype == torch.uint8
        assert w2.shape == (local, hidden, rows // 2), "w2 must be [E, H, I]"
        assert w2_s.shape == (local, hidden, rows // 2 // BLOCK) and w2_s.dtype == torch.uint8
        assert x.dtype == torch.bfloat16 and x.shape[1] == hidden
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        assert hidden % K_TILE == 0 and (rows // 2) % K_TILE == 0, (
            f"hidden and intermediate sizes must be multiples of the {K_TILE}-wide K tile"
        )


inkling2_mxfp8_experts = Inkling2Mxfp8Experts()
