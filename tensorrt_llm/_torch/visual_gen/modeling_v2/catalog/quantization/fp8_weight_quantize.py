# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .fp8_quantize import _reference_scale, native_fp8_per_tensor_quantize

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max


class Fp8PerTensorWeightQuantize(OpWrapper):
    """Quantize a contiguous tensor of rank >= 2 with one FP32 dequantization scale."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "aligned BF16 midpoint with a tail", dict(shape=(3, 3), dtype=torch.bfloat16, offset=0)
        ),
        Cell(
            "unaligned contiguous BF16 midpoint", dict(shape=(3, 3), dtype=torch.bfloat16, offset=1)
        ),
        Cell("rank-three FP32 weights", dict(shape=(2, 3, 7), dtype=torch.float32)),
        Cell("FP16 zero range", dict(shape=(3, 16), dtype=torch.float16, zero=True)),
    )

    def __call__(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed, scale = native_fp8_per_tensor_quantize(weight)
        return packed, scale.float()

    def reference(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes, scale = native_fp8_per_tensor_quantize.reference(weight)
        return codes, scale.float()


class Fp8BlockwiseWeightQuantize(OpWrapper):
    """Quantize a weight matrix in square blocks, including partial edge blocks."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("BF16 partial edge blocks", dict(shape=(145, 259), block=128, dtype=torch.bfloat16)),
        Cell(
            "strided FP32 weights",
            dict(shape=(33, 63), block=16, dtype=torch.float32, strided=True),
        ),
        Cell("FP16 scale floor", dict(shape=(12, 24), block=8, dtype=torch.float16)),
    )

    def __call__(
        self, weight: torch.Tensor, block_size: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = weight.shape
        block_rows = (rows + block_size - 1) // block_size
        block_cols = (cols + block_size - 1) // block_size
        pad_rows, pad_cols = block_rows * block_size - rows, block_cols * block_size - cols
        if pad_rows or pad_cols:
            weight = F.pad(weight, (0, pad_cols, 0, pad_rows))

        blocks = (
            weight.reshape(block_rows, block_size, block_cols, block_size)
            .permute(0, 2, 1, 3)
            .reshape(block_rows * block_cols, block_size * block_size)
        )
        packed, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(blocks.contiguous())
        packed = (
            packed.reshape(block_rows, block_cols, block_size, block_size)
            .permute(0, 2, 1, 3)
            .reshape(block_rows * block_size, block_cols * block_size)
        )
        return packed[:rows, :cols].contiguous(), scales.reshape(block_rows, block_cols).float()

    def reference(
        self, weight: torch.Tensor, block_size: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = weight.shape
        maxima = F.max_pool2d(weight.float().abs().unsqueeze(0), block_size, ceil_mode=True)[0]
        scales = _reference_scale(maxima, weight.dtype).float()
        row_blocks = torch.arange(rows, device=weight.device) // block_size
        col_blocks = torch.arange(cols, device=weight.device) // block_size
        packed = weight.float() / scales[row_blocks[:, None], col_blocks]
        return packed.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), scales


quantize_fp8_per_tensor = Fp8PerTensorWeightQuantize()
quantize_fp8_blockwise = Fp8BlockwiseWeightQuantize()
