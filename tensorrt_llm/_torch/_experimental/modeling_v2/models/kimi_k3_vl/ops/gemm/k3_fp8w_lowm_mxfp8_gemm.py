# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M FP8-weight GEMM on an MXFP8 activation a producer already quantized (row-major 1 x 32
exponents)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_gemm import k3_fp8w_lowm_mxfp8 as _mx
from . import _fp8_reference as ref


class K3Fp8wLowmMxfp8Gemm(OpWrapper):
    """``out [M, N]`` = BF16 ``dequant(xq, xs) @ dequant(w, w_scale)^T``; returns ``out``.

    ``xq``: E4M3 ``[M, K]``; ``xs``: UE8M0 ``[M, K / 32]`` row-major; ``w_scale``: the weight's
    DeepGEMM-packed scale. ``stage_weights``: the tactic that keeps the CTA's weight slice on chip
    before ``griddepcontrol.wait`` (``staged_mx_tactic``, the same bits).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Helix decode projections whose producer emits the MXFP8 operand, at 8 rows per request.
    #: ``stage``: ``stage_weights``.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 KDA q/k/v/g shard with its weight slice staged before the wait, b1",
            spec={"n": 3072, "k": 7168, "m": 8, "stage": True},
        ),
        Cell(
            why="helix8 KDA q/k/v/g shard, eight requests: above Rubin's 32-row cut-over",
            spec={"n": 6144, "k": 7168, "m": 64, "stage": False},
        ),
        Cell(
            why="MoE latent_down, two requests",
            spec={"n": 3584, "k": 7168, "m": 16, "stage": False},
        ),
        Cell(
            why="MLA q_a|kv_a|k_pe (a partial last N tile), four requests",
            spec={"n": 2112, "k": 7168, "m": 32, "stage": False},
        ),
        Cell(
            why="helix16 KDA / MLA o_proj on one row (decode without MTP)",
            spec={"n": 7168, "k": 768, "m": 1, "stage": False},
        ),
    )

    note = """
    Same tiling, split-K, K-tile order and scale folding as the low-M kernel of
    :class:`K3Fp8wLowmGemm`, so on the operand that kernel's quantization produces the output has
    its bits. Inputs are not re-validated (``k3_fp8w_lowm_mxfp8.can_run_mxfp8`` is the caller's
    check). A PDL dependent: the weight warp releases the dependent grid at entry and streams before
    ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        stage_weights: bool = False,
    ) -> torch.Tensor:
        return _mx.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, w, w_scale, out, stage_weights=stage_weights)

    def reference(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        stage_weights: bool = False,
    ) -> torch.Tensor:
        """The expected ``out``: FP64 product of the decoded operands, rounded to BF16."""
        return ref.matmul_bf16(ref.dequant(xq, xs, 32), ref.fp8_block_weight(w, w_scale))

    def is_valid(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        stage_weights: bool = False,
    ) -> None:
        # The launcher passes raw pointers; can_run_mxfp8 is the contract it does not re-check.
        assert _mx.can_run_mxfp8(xq, xs, w, w_scale, allow_partial_n_tile=True, above_cutover=True)
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
        assert out.shape == (xq.shape[0], w.shape[0])


k3_fp8w_lowm_mxfp8_gemm = K3Fp8wLowmMxfp8Gemm()
