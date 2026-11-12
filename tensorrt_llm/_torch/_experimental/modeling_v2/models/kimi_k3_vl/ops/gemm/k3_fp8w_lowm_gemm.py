# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M FP8-weight GEMM of BF16 rows (activation quantized to MXFP8 in the kernel), with the
output-gated and sigmoid variants."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_gemm import k3_fp8w_lowm_gemm as _lowm
from . import _fp8_reference as ref


class K3Fp8wLowmGemm(OpWrapper):
    """BF16 ``[M, N] = q(x) @ dequant(w, w_scale)^T`` (``q``: MXFP8, one UE8M0 exponent per 1 x 32),
    ``w`` the UE8M0-resmoothed E4M3 ``[N, K]`` weight and ``w_scale`` its DeepGEMM-packed scale.

    ``gate``: ``x`` is first multiplied by ``gate`` (PyTorch's BF16 multiply). ``sigmoid``:
    ``torch.sigmoid`` of the BF16 output. ``out``: a contiguous BF16 ``[M, N]`` tensor the result
    goes into (returned); else a new one.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: K3 decode projections at the row counts their CUDA graphs run: MTP with seven draft tokens
    #: puts 8 rows per request, the helix deployments batch 1..8 requests (8..64 rows), dep16 up
    #: to 32 per rank. ``variant``: ``new`` (no ``out``), ``out``, ``gated`` or ``sigmoid``.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 KDA fused q/k/v/g head shard, one request: the b1 decode step",
            spec={"n": 3072, "k": 7168, "m": 8, "variant": "out"},
        ),
        Cell(
            why="MoE latent_down (every deployment) on a single row: the smallest MMA bucket",
            spec={"n": 3584, "k": 7168, "m": 1, "variant": "new"},
        ),
        Cell(
            why="MoE latent_up, four requests: the 32-row bucket both tables run on the low-M kernel",
            spec={"n": 7168, "k": 3584, "m": 32, "variant": "out"},
        ),
        Cell(
            why="MLA q_a|kv_a|k_pe (2112 rows, a partial last N tile), eight requests: low-M on "
            "GB300, mid-M above Rubin's 32-row cut-over",
            spec={"n": 2112, "k": 7168, "m": 64, "variant": "out"},
        ),
        Cell(
            why="helix16 KDA / MLA o_proj with the output gate multiplied in (gated kernel)",
            spec={"n": 7168, "k": 768, "m": 8, "variant": "gated"},
        ),
        Cell(
            why="helix16 MLA output gate projection with the sigmoid epilogue, two requests",
            spec={"n": 768, "k": 7168, "m": 16, "variant": "sigmoid"},
        ),
        Cell(
            why="dep16 shared-expert gate_up, 16 requests: quantization kernel + mid-M GEMM",
            spec={"n": 12288, "k": 7168, "m": 128, "variant": "out"},
        ),
        Cell(
            why="dep16 KDA / MLA o_proj (K = 12288), 12 requests: the 96-row mid-M bucket",
            spec={"n": 7168, "k": 12288, "m": 96, "variant": "new"},
        ),
    )

    note = """
    ``M <= low_m_max_for(N, K)`` runs the low-M kernel (activation quantized inside the GEMM);
    larger ``M`` (up to 256, plain variant only) a quantization kernel followed by the mid-M GEMM,
    chained with PDL. The gated and sigmoid variants are low-M only. Per-shape tactics and
    cut-overs come from per-architecture tables (GB300 for SM100 / SM103, Rubin for SM107); split-K
    partials are reduced in a fixed order, so repeated calls give the same bits. A PDL dependent
    launch: the weight streams before ``griddepcontrol.wait``, the activation (and gate) after.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        sigmoid: bool = False,
    ) -> torch.Tensor:
        if gate is not None:
            return _lowm.k3_fp8w_lowm_gemm_gated(x, gate, w, w_scale, out)
        if sigmoid:
            return _lowm.k3_fp8w_lowm_gemm_sigmoid(x, w, w_scale, out)
        if out is not None:
            return _lowm.k3_fp8w_lowm_gemm_out(x, w, w_scale, out)
        return _lowm.k3_fp8w_lowm_gemm(x, w, w_scale)

    def reference(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        sigmoid: bool = False,
    ) -> torch.Tensor:
        """The FP64 product of the MXFP8-quantized rows and the dequantized weight, rounded to
        BF16 (then ``torch.sigmoid``); ``out`` is not written."""
        if gate is not None:
            x = x * gate
        y = ref.lowm_gemm(x, w, w_scale)
        return torch.sigmoid(y) if sigmoid else y

    def is_valid(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        sigmoid: bool = False,
    ) -> None:
        # With a gate the gated kernel runs and `sigmoid` would be dropped without an error.
        assert gate is None or not sigmoid, "the gated GEMM has no sigmoid epilogue"


k3_fp8w_lowm_gemm = K3Fp8wLowmGemm()
