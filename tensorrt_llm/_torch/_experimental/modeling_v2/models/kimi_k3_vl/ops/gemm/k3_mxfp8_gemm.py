# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 dense MXFP8 GEMM of an FP8 block-scale projection, autotuned over K3's tactics."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _fp8_reference as ref
from . import _mxfp8_gemm


class K3Mxfp8Gemm(OpWrapper):
    """BF16 ``input @ weight.T`` of E4M3 ``input [M, K]`` / ``weight [N, K]`` with their K32 R128c4
    UE8M0 scales (``fp8_quantize_1x128_cutedsl_ue8m0`` / the projection's prepared scale)."""

    ARCHS = frozenset({Arch.SM_107})
    #: The FP8 block-scale projections that reach the MXFP8 GEMM on SM107 (tp1 context, and the
    #: shapes the low-M kernels leave to it), with rows from decode to a context chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MLA q_a|kv_a|k_pe (a partial last N tile) on a tp1 context chunk",
            spec={"n": 2112, "k": 7168, "m": 1000},
        ),
        Cell(
            why="o_proj-like K = 1024 weight on 48 rows (a split-K bucket in main's tactic set)",
            spec={"n": 7168, "k": 1024, "m": 48},
        ),
        Cell(
            why="a decode row of the K = 7168, N = 4096 projection",
            spec={"n": 4096, "k": 7168, "m": 1},
        ),
        Cell(
            why="K = 512 projection on a 4096-row context chunk",
            spec={"n": 2304, "k": 512, "m": 4096},
        ),
    )

    note = """
    Main's CuTe DSL persistent block-scaled GEMM (``CuteDSLKimiK3MXFP8RubinLinear``) with K3's
    tactic subset (``ops.gemm._mxfp8_gemm.keeps_tactic``); the AutoTuner picks the tactic per bucket of
    M under the cache key ``ops.gemm._mxfp8_gemm.OP_NAME``. No split-K tactic is tuned, so every
    bucket is run-to-run deterministic. The output is a new contiguous BF16 ``[M, N]`` tensor; the
    current stream is used. Raises on any GPU but SM107.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return _mxfp8_gemm.k3_mxfp8_gemm_rubin(input, weight, input_scale, weight_scale)

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """FP64 product of the decoded operands (both scales K32 R128c4), rounded to BF16."""
        m, k = input.shape
        n = weight.shape[0]
        a = ref.dequant(input, ref.r128c4_exponents(input_scale, m, k), 32)
        b = ref.dequant(weight, ref.r128c4_exponents(weight_scale, n, k), 32)
        return ref.matmul_bf16(a, b)


k3_mxfp8_gemm = K3Mxfp8Gemm()
