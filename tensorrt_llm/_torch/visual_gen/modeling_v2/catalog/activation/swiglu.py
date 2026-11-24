# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _swiglu_kernel(output, x, scale, stride, width, QUANTIZED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(x + row * stride + col, mask=col < width, other=0).to(tl.float32)
    up = tl.load(x + row * stride + width + col, mask=col < width, other=0).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    if QUANTIZED:
        result = tl.clamp(result / tl.load(scale), -448.0, 448.0)
    tl.store(output + row * width + col, result, mask=col < width)


@torch.library.custom_op("trtllm::visual_gen_v2_swiglu", mutates_args=(), device_types="cuda")
def _swiglu(x: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    rows, cols = x.shape
    if cols % 2 or x.stride(1) != 1:
        raise ValueError("SwiGLU requires packed gate/up rows with a contiguous final dimension.")
    width = cols // 2
    out = x.new_empty((rows, width), dtype=torch.float8_e4m3fn if scale is not None else x.dtype)
    _swiglu_kernel[(rows, triton.cdiv(width, 1024))](
        out, x, scale, x.stride(0), width, scale is not None, 1024
    )
    return out


@_swiglu.register_fake
def _(x, scale):
    return x.new_empty(
        (x.shape[0], x.shape[1] // 2), dtype=torch.float8_e4m3fn if scale is not None else x.dtype
    )


class SwiGLU(OpWrapper):
    """SwiGLU over packed gate/up rows with an optional scalar FP8 dequantization scale."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell("Odd packed sequence length", dict(rows=17, width=128, quantized=False)),
        Cell(
            "FP8 activations across a partial launch tile", dict(rows=3, width=1025, quantized=True)
        ),
    )

    def __call__(self, x: torch.Tensor, quant_scale: torch.Tensor | None = None) -> torch.Tensor:
        return _swiglu(x, quant_scale)

    def reference(self, x, quant_scale=None):
        gate, up = x.float().chunk(2, dim=-1)
        out = F.silu(gate) * up
        if quant_scale is None:
            return out.to(x.dtype)
        return (out / quant_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    def is_valid(self, x, quant_scale=None):
        assert x.ndim == 2 and x.shape[1] % 2 == 0 and x.stride(1) == 1
        assert quant_scale is None or (quant_scale.numel() == 1 and quant_scale.device == x.device)


swiglu = SwiGLU()
