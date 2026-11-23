# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 block-FP8 quantization of bf16 rows: e4m3 codes plus K-block-major fp32 scales."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

BLOCK = 128
E4M3_MAX = 448.0

#: The kernel's amax floor, as the bf16 constant it compares against.
AMAX_FLOOR = float(torch.tensor(1e-10, dtype=torch.bfloat16))

_Q_B = 1024
_GATE = 3072
_O_PROJ = 8192


def block_scales(input: torch.Tensor, use_ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """`(quant_scale, dequant_scale)`, each `[M, ceil(K/128)]` fp32, from a bf16 amax floored at `AMAX_FLOOR`.

    Every quotient is tensor / tensor: torch evaluates `scalar / tensor` as reciprocal-times-scalar,
    which can sit one ulp off the kernel's division.
    """
    rows, cols = input.shape
    nb = (cols + BLOCK - 1) // BLOCK
    mag = torch.nn.functional.pad(input.abs(), (0, nb * BLOCK - cols))
    amax = mag.reshape(rows, nb, BLOCK).amax(dim=2).float().clamp(min=AMAX_FLOOR)
    one = torch.ones_like(amax)
    qs = torch.full_like(amax, E4M3_MAX) / amax
    if not use_ue8m0:
        return qs, one / qs
    raw = one / qs
    mant, exp = torch.frexp(raw)
    ds = torch.ldexp(one, torch.where(mant == 0.5, exp - 1, exp))
    return one / ds, ds


class Fp8Quantize1x128(OpWrapper):
    """Quantize bf16 `[M, K]` to `(data [M, K] e4m3, scales [ceil(K/128), M] fp32)`.

    Scales are transposed (K-block-major), the layout `cute_dsl_fp8_gemm_blackwell` reads; pass them on as returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode row before q_b at 1024 wide: the vocabulary's decode call",
            spec=dict(rows=1, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="64 rows before q_b: the widest decode batch the engine captures",
            spec=dict(rows=64, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="6 rows before the drafter's q_b: five drafts plus the anchor",
            spec=dict(rows=6, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="384 rows before the drafter's q_b: width 6 at 64 requests",
            spec=dict(rows=384, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="512 rows before q_b: one chunked-prefill chunk",
            spec=dict(rows=512, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before q_b: the prefill envelope's largest row count",
            spec=dict(rows=5120, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="one decode row before linear_gate at 3072 wide",
            spec=dict(rows=1, cols=_GATE, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before linear_gate at the envelope's largest row count",
            spec=dict(rows=5120, cols=_GATE, use_ue8m0=False),
        ),
        Cell(
            why="one decode row before o_proj at 8192 wide",
            spec=dict(rows=1, cols=_O_PROJ, use_ue8m0=False),
        ),
        Cell(
            why="64 rows == 8192/128 before o_proj: the scale buffer is square",
            spec=dict(rows=64, cols=_O_PROJ, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before o_proj: the widest row count at the widest width",
            spec=dict(rows=5120, cols=_O_PROJ, use_ue8m0=False),
        ),
    )

    note = """
    Off the sm_100 family the op returns a 1-D padded scale buffer instead. `data` is a row view of an
    allocation padded to a multiple of 4 rows; a raw-pointer reader must stop at `M`.
    """

    def __call__(
        self,
        input: torch.Tensor,
        use_ue8m0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp8_quantize_1x128(input, use_ue8m0)

    def reference(
        self,
        input: torch.Tensor,
        use_ue8m0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(data, scales[ceil(K/128), M])` from exactly rounded fp32 divisions, as a fresh contiguous pair."""
        qs, ds = block_scales(input, use_ue8m0)
        cols = qs.repeat_interleave(BLOCK, dim=1)[:, : input.shape[1]]
        data = (input.float() * cols).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return data, ds.t().contiguous()


fp8_quantize_1x128 = Fp8Quantize1x128()
