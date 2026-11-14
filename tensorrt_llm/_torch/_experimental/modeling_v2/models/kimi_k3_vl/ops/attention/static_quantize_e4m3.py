# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 -> FP8 E4M3 per-tensor static quantization of the K3 FP8 context FMHA inputs (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_mla_ctx import fp8_static_quant as _quant

_E4M3_MAX = 448.0


class StaticQuantizeE4m3(OpWrapper):
    """``x * (1 / scale)`` in E4M3 (round to nearest even, saturating), a new FP8 tensor.

    ``rows=False``: ``x`` contiguous BF16 of even size, the output has ``x``'s shape.
    ``rows=True``: ``x`` a row-strided BF16 ``[rows, width]`` view (unit column stride, even row
    stride and storage offset, power-of-two width >= 2), the output is contiguous ``[rows, width]``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Spec: `rows` (the op's flag), `tokens`, and the input: `q` (the context Q rows, 96 heads x
    # 192, contiguous), `latent_k_pe` (the K-pe columns of latent_cache [T, 576], row stride 576)
    # or `k_pe` (a gathered BF16 K-pe partition [T, 64], contiguous). Unit scale, as K3 runs.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp1 context: Q of a full 8k chunk, 96 heads",
            spec=dict(rows=False, tokens=8192, source="q"),
        ),
        Cell(
            why="tp1 context: Q of a short last chunk",
            spec=dict(rows=False, tokens=17, source="q"),
        ),
        Cell(
            why="tp1 context: current chunk's K-pe read in place from latent_cache",
            spec=dict(rows=True, tokens=8192, source="latent_k_pe"),
        ),
        Cell(
            why="tp1 context: K-pe of a few current tokens, one program",
            spec=dict(rows=True, tokens=17, source="latent_k_pe"),
        ),
        Cell(
            why="tp1 context: a BF16-gathered cached-prefix K-pe partition",
            spec=dict(rows=True, tokens=3000, source="k_pe"),
        ),
    )

    note = """
    The bytes of `torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)` (the C++
    quantizer the attention op's context copy uses): `cvt.rn.satfinite.e4m3` of
    `float(x) * (1 / scale)` with an IEEE FP32 reciprocal. `scale` is a one-element FP32 tensor on
    `x`'s device, read on the device (no sync). The contiguous form converts 8192 BF16 pairs per
    program; the row form 64 rows per program. Triton; needs SM89+ for the E4M3 conversion.
    """

    def __call__(self, x: torch.Tensor, scale: torch.Tensor, rows: bool = False) -> torch.Tensor:
        if rows:
            return _quant.static_quantize_e4m3_rows(x, scale)
        return _quant.static_quantize_e4m3(x, scale)

    def reference(self, x: torch.Tensor, scale: torch.Tensor, rows: bool = False) -> torch.Tensor:
        """``x * (1 / scale)`` with the reciprocal rounded once to FP32, then saturated and
        rounded to E4M3 (``rows`` only decides the input layout the op reads)."""
        inv_scale = (1.0 / scale.double()).float()
        y = (x.float() * inv_scale).clamp(-_E4M3_MAX, _E4M3_MAX)
        return y.to(torch.float8_e4m3fn).contiguous()


static_quantize_e4m3 = StaticQuantizeE4m3()
