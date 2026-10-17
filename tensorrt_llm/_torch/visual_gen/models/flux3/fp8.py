# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8R activation quantization with FP32 per-row scales."""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, Fp8RowwiseQuantizedTensor


@triton.jit
def _one_pass_fp8(x, output, scales, stride, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, block)
    values = tl.load(x + row * stride + col, col < width, other=0).to(tl.float32)
    # Preserve TRT-LLM's checkpoint scale convention and round-to-nearest division.
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), 0), 448.0), 1.0e-12)
    quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale), -448.0), 448.0)
    tl.store(output + row * width + col, quantized, col < width)
    tl.store(scales + row, scale)


@torch.library.custom_op("trtllm::flux3_one_pass_fp8_quant", mutates_args=())
def one_pass_fp8_quant(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows.shape[0], device=rows.device, dtype=torch.float32)
    if rows.shape[0]:
        block = triton.next_power_of_2(rows.shape[1])
        _one_pass_fp8[(rows.shape[0],)](
            rows,
            output,
            scales,
            rows.stride(0),
            rows.shape[1],
            block,
            num_warps=min(max(block // 512, 4), 16),
        )
    return output, scales


@one_pass_fp8_quant.register_fake
def _one_pass_fp8_quant_fake(rows):
    return (
        torch.empty_like(rows, dtype=torch.float8_e4m3fn),
        rows.new_empty((rows.shape[0],), dtype=torch.float32),
    )


def quantize_fp8r(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] >= 16384 and rows.shape[1] <= 4096:
        return one_pass_fp8_quant(rows)
    data, scales = torch.ops.trtllm.warp_fp8_per_token_quant(rows, 1e-12)
    return data, scales.reshape(-1)


class Flux3FP8RowwiseLinearMethod(FP8RowwiseLinearMethod):
    def apply(self, module, input, bias):
        if not isinstance(input, Fp8RowwiseQuantizedTensor) and input.dtype != torch.float8_e4m3fn:
            data, scales = quantize_fp8r(input)
            input = Fp8RowwiseQuantizedTensor(input, data, scales)
        return super().apply(module, input, bias)
