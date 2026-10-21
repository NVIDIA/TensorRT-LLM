# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer second half at an MoE site: `qwen4_exp_hc_up_gate_mean`
(the up projection as a tcgen05 GEMM, sigmoid gate and stream mean in its
epilogue) whose epilogue also writes the routed experts' NVFP4 input.

Tier 6 (CuTe DSL written for this target): the `fp4=True` build of
`qwen4_exp_hc_up_gate_mean`'s kernel. The routed experts read the mixer's output
only as NVFP4 (packed e2m1 + one e4m3 scale per 16 elements, linear scale
layout); the bf16 mix is still written for the router and the shared expert.
Each epilogue thread already holds one row's 32 consecutive mix values, i.e.
two whole scale blocks, so the quantization needs no data exchange and removes
the `fp4_quantize` launch and its re-read of the bf16 activation.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from .qwen4_exp_hc_up_gate_mean import _BLOCK_H, hc_up_gate_mean_fp4, qwen4_exp_hc_up_gate_mean

#: NVFP4 block: one e4m3 scale per this many contiguous elements.
SF_VEC = 16


class Qwen4ExpHcUpGateMeanFp4(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `low` the `[T, R]` SiLU bottleneck,
    `up` the `[R, C*H]` transposed view of the `[C*H, R]` up weight::

        mixed[t, i] = mean over s of sigmoid((low @ up)[t, s*H + i]) * normed[t, s*H + i]
        xq, xsf     = NVFP4(mixed)        # e2m1 pairs, e4m3 scale per 16, linear layout

    `mixed` is bitwise `qwen4_exp_hc_up_gate_mean`'s (same kernel, same
    epilogue arithmetic); the quantization reads that bf16-rounded value, so
    `xq` / `xsf` are bitwise `torch.ops.trtllm.fp4_quantize(mixed, gs, 16,
    False, False)`. Returns fresh `(mixed [T, H], xq [T, H/2] uint8, xsf
    [T*H/16] uint8)`. `global_scale` is the 1-element fp32 NVFP4 global scale
    (`1 / input_scale`). Fusion boundary: the up projection, the sigmoid
    gate, the stream mean and the NVFP4 quantization of the mean.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, hc_count (4), hidden (2560), rank (320); bf16 only.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560, rank 320, {why}", spec=dict(rows=t))
        for t, why in (
            (128, "decode 128 rows (max_con)"),
            (16384, "prefill chunk 16384 (max_num_tokens)"),
        )
    ) + (Cell(why="ragged tail, 16320 + 70 rows (mixed step)", spec=dict(rows=16390)),)

    note = """
    Tier 6 (CuTe DSL written for this target): `qwen4_exp_hc_up_gate_mean`'s
    kernel compiled with its NVFP4 epilogue; tiers 1-5 have no GEMM with this
    multi-input epilogue, and the fused up projection itself is tier 6.
    Metadata consumed: none. Call sites: the 48 MoE-site mixers (prefill, and
    decode past `_HC_DECODE_GEMV_MAX_ROWS` rows); `xq`/`xsf` feed
    `fp4_block_scale_moe_renorm_routed` (linear scales), `mixed` the router and
    the shared expert. Preconditions: those of `qwen4_exp_hc_up_gate_mean`
    (bf16, `R % 64 == 0`, `H % 32 == 0`, `C * 32 <= 256`, 16-byte aligned
    operands) and `global_scale` one fp32 element on the device. Compiles once
    per `(C, H, R)` on first call; graph-capturable (current stream, no host
    sync).

    Numerics: the quantization is the trtllm NVFP4 recipe instruction for
    instruction (`rcp.approx.ftz.f32` reciprocals, `cvt.rn.satfinite` to e4m3
    and to e2m1, every multiply `mul.rn.f32`); the GPU test checks it against
    `fp4_quantize`'s native-torch reference on the returned `mixed` (scales
    bitwise, e2m1 codes bitwise outside the reciprocal tie window). Bitwise
    `torch.ops.trtllm.fp4_quantize`, ties included, and across tiny and
    saturating global scales: measured, not certified.

    Certified: bf16 at the checkpoint's 4 x 2560, rank 320, a calibrated global
    scale: 128 rows, 16384 rows, 16390 rows (a partial last tile); graph replay.

    Not certified: zero rows, fewer rows than a tile, hc_count other than 4,
    hidden other than 2560, rank other than 320 (the kernel compiles for them;
    no cell drives them).
    """

    def __call__(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        normed: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_up_gate_mean_fp4(low, up, normed, hc_count, global_scale, _BLOCK_H)

    def reference(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        normed: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`mixed` in float64 with one rounding, then the NVFP4 of that rounded
        value in native torch."""
        mixed = qwen4_exp_hc_up_gate_mean.reference(low, up, normed, hc_count)
        return (mixed,) + fp4_quantize.reference(mixed, global_scale, SF_VEC, False, False)

    def is_valid(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        normed: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> None:
        qwen4_exp_hc_up_gate_mean.is_valid(low, up, normed, hc_count)
        assert (normed.shape[1] // hc_count) % SF_VEC == 0, "hidden must be whole 16-blocks"
        # The kernel loads element 0 only; a per-row scale would be applied as row 0's.
        assert global_scale.numel() == 1 and global_scale.dtype == torch.float32
        assert global_scale.is_cuda


qwen4_exp_hc_up_gate_mean_fp4 = Qwen4ExpHcUpGateMeanFp4()
