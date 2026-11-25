# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FFN input in both of its quantized forms from one pass, as stock's mHC boundary hands it to the MoE
(`hyper_connection.py`: `nvfp4_fp8_quantize` for `>= FP8_1X128_MIN_ROWS` rows): NVFP4 for the routed experts and FP8
1x128 (UE8M0, DeepGEMM's packed layout) for the shared expert's first GEMM."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

MIN_ROWS = 1024  # stock's FP8_1X128_MIN_ROWS: below it the boundary quantizes NVFP4 alone


class Nvfp4Fp8Quantize(OpWrapper):
    """`x` bf16 `[n, H]` contiguous (`H % 512 == 0`), `global_scale` fp32 `[1]` -> `(fp4 uint8 [n, H/2], sf uint8
    [n, H/16] (row-major e4m3), fp8 e4m3 [n, H], fp8_sf int32 [n, H/512])`: `fp4_quantize(x, global_scale, 16)` and
    `fp8_quantize_1x128_packed_ue8m0(x, False)` (`fp8_sf` is the MN-major view DeepGEMM reads)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a 1024-row step (the smallest that takes it)", spec=dict(n=1024, hidden=5120)),
        Cell(why="a 32300-row context step", spec=dict(n=32300, hidden=5120)),
    )
    note = "Both outputs are bitwise those of the two separate quantizers."

    @staticmethod
    def ready(hidden: int) -> bool:
        """Whether a launch may run now: compiled, or outside graph capture (where the first call compiles it)."""
        from tensorrt_llm._torch.kernels.blackwell.mhc.mhc_boundary import (
            nvfp4_fp8_quantize_compiled,
        )

        return not torch.cuda.is_current_stream_capturing() or nvfp4_fp8_quantize_compiled(hidden)

    def __call__(self, x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
        from tensorrt_llm._torch.kernels.blackwell.mhc.mhc_boundary import nvfp4_fp8_quantize

        return nvfp4_fp8_quantize(x, global_scale)

    def reference(self, x: torch.Tensor, global_scale: torch.Tensor) -> tuple:
        fp4, sf = fp4_quantize.reference(x, global_scale, 16, False, False)
        fp8, fp8_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
        return fp4, sf.view(x.shape[0], x.shape[1] // 16), fp8, fp8_sf  # [n, H/16], also for n == 0

    def is_valid(self, x: torch.Tensor, global_scale: torch.Tensor) -> None:
        assert (
            x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
            and x.shape[1] % 512 == 0
        )
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1


nvfp4_fp8_quantize = Nvfp4Fp8Quantize()
