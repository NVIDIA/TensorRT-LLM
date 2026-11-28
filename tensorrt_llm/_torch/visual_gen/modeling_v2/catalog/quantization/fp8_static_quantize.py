# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8_static_quant import fp8_static_quant


class Fp8StaticQuantize(OpWrapper):
    """Contiguous BF16 rounded to E4M3 with a supplied FP32 scale."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell("Partial final tile", dict(shape=(17, 63))),)

    def __call__(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return fp8_static_quant(x, scale)

    def reference(self, x, scale):
        return (x.float() * scale.float().reciprocal()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    def is_valid(self, x, scale):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.is_cuda
        assert scale.dtype == torch.float32 and scale.numel() == 1 and scale.device == x.device


fp8_static_quantize = Fp8StaticQuantize()
