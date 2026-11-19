# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class SwiGLU(OpWrapper):
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell(why="Odd packed sequence length", spec=dict(rows=17, width=128)),)

    def __call__(self, x: torch.Tensor, quant_scale: torch.Tensor | None = None) -> torch.Tensor:
        if quant_scale is None:
            return torch.ops.trtllm.silu_and_mul(x)
        return torch.ops.trtllm.silu_and_mul(x, scale=quant_scale, dtype=torch.float8_e4m3fn)

    def reference(self, x: torch.Tensor, quant_scale: torch.Tensor | None = None) -> torch.Tensor:
        gate, up = x.float().chunk(2, dim=-1)
        out = F.silu(gate) * up
        if quant_scale is None:
            return out.to(x.dtype)
        return (out / quant_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    def is_valid(self, x: torch.Tensor, quant_scale: torch.Tensor | None = None) -> None:
        assert x.shape[-1] % 2 == 0


swiglu = SwiGLU()
