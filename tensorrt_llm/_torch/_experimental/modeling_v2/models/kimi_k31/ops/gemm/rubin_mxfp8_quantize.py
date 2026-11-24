# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 activation quantization for ``rubin_mxfp8_gemm`` above the low-M GEMM's rows (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.mxfp8_quantize_r128c4 import (  # noqa: E501
    mxfp8_quantize_r128c4 as _mxfp8_quantize_r128c4,
)

#: Rows of the K3.1 prefill calls: just above the low-M cut-over, a 1k chunk, a partial last M
#: tile, and the 8192-token chunk.
_ROWS = (256, 1024, 2954, 8192)
#: Input widths: the hidden size (KDA stack, shared gate_up), the KDA o_proj at TP8 (12 heads x 128)
#: and at DEP16 (96 heads x 128), the shared down at TP8 and whole.
_WIDTHS = {
    7168: "hidden (KDA stack, shared gate_up)",
    1536: "KDA o_proj at TP8",
    12288: "KDA o_proj at DEP16",
    768: "shared down at TP8",
    6144: "shared down whole",
}


class RubinMxfp8Quantize(OpWrapper):
    """BF16 ``[M, K]`` -> ``(E4M3 [M, K], UE8M0 scales [pad128(M) * pad4(K / 32)])``, R128c4.

    Bitwise ``mxfp8_quantize(input, True, 32)`` (the catalog entry, one C++ kernel): one
    power-of-two scale per 32 values, the smallest at or above ``amax * rcp.approx(448)``; the
    scales in the 128x4-swizzled order ``rubin_mxfp8_gemm`` reads, padding rows and columns 0.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"K3.1 Rubin {what} (K={k}), M={m}", spec=dict(m=m, k=k))
        for k, what in _WIDTHS.items()
        for m in _ROWS
    )

    note = """
    One CuTe DSL kernel over warp tiles of rows x (128 | 256) values, where the C++ kernel walks one
    row per CTA and keeps too few loads in flight to stream HBM. Launched with PDL when
    TRTLLM_ENABLE_PDL is set (waits on the previous grid before its loads). An all-zero block
    gets scale byte 0 and zero values, as the C++ kernel's; `reference` does not model blocks whose
    amax is below 448 * 2^-127 (see the catalog entry).
    """

    def __call__(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _mxfp8_quantize_r128c4(input)

    def reference(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The catalog ``mxfp8_quantize.reference`` with swizzled scales and alignment 32."""
        return mxfp8_quantize.reference(input, True, 32)

    def is_valid(self, input: torch.Tensor) -> None:
        assert input.dtype == torch.bfloat16 and input.dim() == 2, "a BF16 [M, K] input"
        assert input.is_contiguous(), "a contiguous input; the kernel reads it flat"
        assert input.shape[1] % 32 == 0, "K must be a whole number of 32-value scale blocks"
        assert input.is_cuda, "a CUDA input"


rubin_mxfp8_quantize = RubinMxfp8Quantize()
