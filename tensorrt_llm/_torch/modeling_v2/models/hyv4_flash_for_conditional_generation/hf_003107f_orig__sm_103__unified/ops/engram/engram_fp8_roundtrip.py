# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram table's per-row e4m3 round trip, as the checkpoint's HF module computes it.

``HYV4FlashEngramLayer.forward`` in
``transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`` runs ``_fp8_round_trip`` over
the gathered 96-wide embedding rows when ``engram_fp8_embedding`` is set -- true for this
checkpoint -- before the per-head RMSNorm. Per row, in the input dtype (bf16)::

    scale = clamp(row_amax, min=1e-12) / 448
    q = clamp(x / scale, -448, 448).to(torch.float8_e4m3fn).to(bf16) * scale
    out = x + (q - x).detach()  # straight-through estimator

Inference has no autograd, and in bf16 the straight-through form is the identity: ``q`` is
a bf16 value within a few percent of ``x``, so ``q - x`` is exact and ``x + (q - x)``
rounds back to ``q``. ``__call__`` returns ``q``, the same tensor HF's caller sees.

The entry is the composition of native torch calls below: nothing in the tree fuses this
round trip into one launch, and the two-launch engine path computes a different function
(see ``note``).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: ``torch.finfo(torch.float8_e4m3fn).max``, the scale denominator in ``_fp8_round_trip``.
_E4M3_MAX = 448.0
#: The floor HF puts on a row's amax before dividing: a zero row keeps a zero scale, so
#: ``0 / scale`` is 0 rather than NaN.
_SCALE_MIN = 1e-12


class EngramFp8Roundtrip(OpWrapper):
    """``rows`` bf16 ``[tokens * 16 heads, 96]`` -> bf16, each row e4m3-round-tripped."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token: 16 gathered heads", spec=dict(rows=16)),
        Cell(why="a six-token DFly2 verify step: 96 rows", spec=dict(rows=96)),
        Cell(why="CUDA-graph batch of 64 decode tokens: 1024 rows", spec=dict(rows=1024)),
        Cell(why="a 512-token chunked-prefill chunk: 8192 rows", spec=dict(rows=8192)),
        Cell(why="the 8192-token max_num_tokens envelope: 131072 rows", spec=dict(rows=131072)),
    )
    note = (
        "A composition of native torch calls, and the entry is that composition: no in-tree "
        "kernel fuses the round trip. torch.ops.trtllm.quantize_e4m3_activation and "
        "dequantize_e4m3_activation are two launches and floor the per-row scale at "
        "1 / (448 * 512) = 4.36e-6, which binds for a row whose amax is below 1 / 512 = 2.0e-3 "
        "and rounds a row below 4.3e-9 to zero; the checkpoint's floor is 1e-12, so that pair "
        "computes a different function on dim rows. "
        "catalog/quantization/mxfp8_quantize scales per 32-element block with an e8m0 "
        "power-of-two and catalog/quantization/fp4_quantize is fp4; neither is e4m3 per row. "
        "SGLang folds the round trip into its engram gather kernel, which needs the table "
        "resident on the device. The scale is rounded to bf16 and the dequantized value is "
        "multiplied in bf16, as HF does; the reference evaluates the quotient and the dequant "
        "multiply in fp32. Measured on sm_103 over rows 16 to 131072: 19.6 bf16 ulps of the "
        "row maximum at the worst element (1.25 e4m3 steps is 40), 4.05 ulps relative RMS, "
        "and at least 97.0% of elements bit-identical to the reference."
    )

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_SCALE_MIN) / _E4M3_MAX
        quantized = (rows / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        return quantized.to(rows.dtype) * scale

    def reference(self, rows: torch.Tensor) -> torch.Tensor:
        """HF's function with the quotient and the dequant multiply kept in fp32.

        The scale stays a tensor of the input dtype: ``_fp8_round_trip`` defines it
        that way, and a row's ``x / scale`` lands on the other side of an e4m3
        midpoint when the scale moves by one bf16 ulp, so rounding it differently
        computes a different function. What the reference does make more accurate is
        everything the function leaves to the elementwise arithmetic: the division,
        the clamp and the dequant multiply run in fp32 and round once, at the end.
        """
        scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_SCALE_MIN) / _E4M3_MAX
        x = rows.float() / scale.float()
        quantized = x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        return (quantized.float() * scale.float()).to(rows.dtype)


engram_fp8_roundtrip = EngramFp8Roundtrip()
