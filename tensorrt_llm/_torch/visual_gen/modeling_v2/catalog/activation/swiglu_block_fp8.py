# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class SwiGLUBlockFp8(OpWrapper):
    """BF16 SwiGLU rounded before 1x128 E4M3 quantization, with column-major packed UE8M0 scales."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell("Partial scale word and padded column stride", dict(rows=5, width=656)),
        Cell("Contiguous BF16 slice with a two-byte offset", dict(rows=3, width=144, offset=1)),
    )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from ._native import swiglu_block_fp8_kernel

        return swiglu_block_fp8_kernel(x)

    def reference(self, x):
        rows, cols = x.shape
        gate, up = x.float().chunk(2, dim=-1)
        width = cols // 2
        activated = (F.silu(gate) * up).bfloat16().float()
        groups = (width + 127) // 128
        values = F.pad(activated, (0, -width % 128)).reshape(rows, groups, 128)
        amax = values.abs().amax(dim=-1).clamp_min(1e-10)
        exponents = torch.ceil(torch.log2(amax / 448))
        codes = (values * (-exponents).exp2().unsqueeze(-1)).to(torch.float8_e4m3fn)
        scale_bytes = F.pad((exponents + 127).byte(), (0, -exponents.shape[1] % 4, 0, -rows % 4))
        words = (
            scale_bytes.flatten().view(torch.int32).reshape(scale_bytes.shape[0], (groups + 3) // 4)
        )
        scales = words.new_empty_strided(words.shape, (1, words.shape[0]))
        scales.copy_(words)
        return codes.reshape(rows, groups * 128)[:, :width], scales[:rows]

    def is_valid(self, x):
        assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert x.shape[1] % 32 == 0


swiglu_block_fp8 = SwiGLUBlockFp8()
