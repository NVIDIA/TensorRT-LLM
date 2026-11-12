# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 dense MXFP8 GEMM writing FP8 E4M3: the MLA context kv_b expansion into the FP8 FMHA scratch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _fp8_reference as ref
from . import _mxfp8_gemm_fp8_out

#: The largest finite E4M3 magnitude: the unit-scale quantization saturates to it.
E4M3_MAX = 448.0


class K3Mxfp8GemmFp8Out(OpWrapper):
    """E4M3 ``input @ weight.T`` of E4M3 ``input [M, K]`` / ``weight [N, K]`` with their K32 R128c4
    UE8M0 scales (``fp8_quantize_1x128_packed_ue8m0`` / the projection's prepared scale), rounded
    to BF16 and then converted to E4M3 with a unit scale."""

    ARCHS = frozenset({Arch.SM_107})
    #: The kv_b expansion of the tp1 context (K = kv_lora_rank 512, N = 96 heads x (128 + 128)) on
    #: chunks below and above the untuned 256 x 256 tile's 1024 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="kv_b expansion of a short cached prefix (fallback / tuned tile)",
            spec={"n": 24576, "k": 512, "m": 300},
        ),
        Cell(
            why="kv_b expansion of a long cached prefix (the untuned 2-CTA 256x256 tile)",
            spec={"n": 24576, "k": 512, "m": 8192},
        ),
    )

    note = """
    Main's CuTe DSL dense MXFP8 Rubin runner on the family's FP8-output kernels
    (``kernels/k3r_gemm/mxfp8_fp8_out_gemm``): the epilogue rounds through BF16 and converts to E4M3
    with a unit scale, so the output equals the BF16 GEMM followed by
    ``static_quantize_e4m3_per_tensor`` with scale 1, and does not depend on the tactic. The AutoTuner
    picks the tactic per bucket of M under main's MXFP8 Rubin key; untuned M >= 1024 runs the 2-CTA
    256x256 tile with 128x128 epilogue subtiles. N must be a multiple of 16. The output is a new
    contiguous E4M3 ``[M, N]`` tensor; the current stream is used. Raises on any GPU but SM107.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return _mxfp8_gemm_fp8_out.k3_mxfp8_gemm_fp8_out_rubin(
            input, weight, input_scale, weight_scale
        )

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """FP64 product of the decoded operands (both scales K32 R128c4), rounded to BF16, then
        saturated to the E4M3 range and rounded to E4M3."""
        m, k = input.shape
        n = weight.shape[0]
        a = ref.dequant(input, ref.r128c4_exponents(input_scale, m, k), 32)
        b = ref.dequant(weight, ref.r128c4_exponents(weight_scale, n, k), 32)
        out = ref.matmul_bf16(a, b).float().clamp(-E4M3_MAX, E4M3_MAX)
        return out.to(torch.float8_e4m3fn)


k3_mxfp8_gemm_fp8_out = K3Mxfp8GemmFp8Out()
