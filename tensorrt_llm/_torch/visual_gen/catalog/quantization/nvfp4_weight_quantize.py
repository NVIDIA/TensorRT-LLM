# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

_SCALE_RANGE = torch.finfo(torch.float8_e4m3fn).max * 6.0


class Nvfp4WeightQuantize(OpWrapper):
    """Return packed E2M1 weights, a linear E4M3 block-scale matrix and a global divisor."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("BF16 signed data and zero blocks", dict(shape=(7, 96), dtype=torch.bfloat16)),
        Cell("FP32 amax before the BF16 cast", dict(shape=(9, 64), dtype=torch.float32)),
        Cell("integer weights cast to BF16", dict(shape=(5, 32), dtype=torch.int16)),
        Cell("zero global amax", dict(shape=(2, 32), dtype=torch.float16, zero=True)),
    )

    def __call__(
        self, weight: torch.Tensor, block_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols = weight.shape
        amax = weight.float().abs().max()
        packed, scales = fp4_quantize(
            weight.to(torch.bfloat16), _SCALE_RANGE / amax, block_size, False, False
        )
        return (
            packed,
            scales.view(torch.float8_e4m3fn).reshape(rows, cols // block_size),
            amax / _SCALE_RANGE,
        )

    def reference(
        self, weight: torch.Tensor, block_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols = weight.shape
        amax = weight.float().abs().max()
        global_divisor = amax / _SCALE_RANGE
        packed, scales = fp4_quantize.reference(
            weight.to(torch.bfloat16), _SCALE_RANGE / amax, block_size, False, False
        )
        return (
            packed,
            scales.view(torch.float8_e4m3fn).reshape(rows, cols // block_size),
            global_divisor,
        )


quantize_nvfp4 = Nvfp4WeightQuantize()
