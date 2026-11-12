# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-M FP8-weight GEMM on a prequantized MXFP8 operand whose epilogue pushes the output into the
next MNNVL all-reduce on a Lamport workspace."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm import k3_lamport_ar
from ...kernels.k3r_gemm import k3_fp8w_lowm_gemm as _lowm
from . import _fp8_reference as ref


class K3Fp8wMidGemmPush(OpWrapper):
    """Push this rank's BF16 ``dequant(xq, xs) @ dequant(w, w_scale)^T`` (``[M, N]``, the mid-M
    kernel of ``tactic``) into the Lamport slots of the next all-reduce on ``ws``; returns None.
    Operands as :class:`K3Fp8wMidGemm`'s (``xs`` in the mid-M layout)."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Rows above the low-M cut-over of both tables, on ``quantize_mid_operand``'s operand for
    #: ``mid_ar_push_tactic``'s tactic.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MoE latent_up all-reduce on 12 requests, two-shot transport",
            spec={"n": 7168, "k": 3584, "m": 96, "one_shot": False},
        ),
        Cell(
            why="o_proj (K = 768) all-reduce on 16 requests, two-shot transport",
            spec={"n": 7168, "k": 768, "m": 128, "one_shot": False},
        ),
        Cell(
            why="MoE latent_up all-reduce on 9 requests, one-shot transport",
            spec={"n": 7168, "k": 3584, "m": 72, "one_shot": True},
        ),
    )

    note = """
    :class:`K3Fp8wMidGemm` with the all-reduce push epilogue: with the finish
    (``k3_lamport_ar.run_lamport_finish``, same ``one_shot``) the call is bitwise that GEMM followed
    by the MNNVL all-reduce of its output. A PDL dependent: the weight streams before
    ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        tactic: "_lowm.K3MidMTactic",
    ) -> None:
        _lowm.run_mid_prequantized_ar_push(xq, xs, w, w_scale, ws, one_shot, tactic)

    def reference(
        self,
        xq: Sequence[torch.Tensor],
        xs: Sequence[torch.Tensor],
        w: Sequence[torch.Tensor],
        w_scale: Sequence[torch.Tensor],
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        tactic: "_lowm.K3MidMTactic",
    ) -> torch.Tensor:
        """A collective: ``xq``, ``xs``, ``w``, ``w_scale`` are the group's inputs in rank order.
        Returns the BF16 ``[M, N]`` the finish yields: the all-reduce of the ranks' BF16 GEMM
        outputs. ``ws`` and ``one_shot`` do not change the result."""
        sf_blocks = -(-tactic.mma_n // 128)
        parts = []
        for r in range(len(xq)):
            m, k = xq[r].shape
            a = ref.dequant(xq[r], ref.mid_exponents(xs[r], m, k, sf_blocks), 32)
            parts.append(ref.matmul_bf16(a, ref.fp8_block_weight(w[r], w_scale[r])))
        return ref.all_reduce_bf16(parts)

    def is_valid(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        tactic: "_lowm.K3MidMTactic",
    ) -> None:
        # The launcher passes raw pointers: nothing checks dtypes, layouts or strides.
        k = xq.shape[1]
        assert xq.dtype == torch.float8_e4m3fn and xq.is_contiguous()
        assert w.dtype == torch.float8_e4m3fn and w.is_contiguous() and w.shape[1] == k
        assert w_scale.dtype == torch.int32, "w_scale must be the DeepGEMM-packed UE8M0 scale"
        sf_blocks = -(-tactic.mma_n // 128)
        assert xs.dtype == torch.uint8 and xs.numel() == (k // 128) * sf_blocks * 512, (
            "xs must be the mid-M layout of the tactic"
        )


k3_fp8w_mid_gemm_push = K3Fp8wMidGemmPush()
