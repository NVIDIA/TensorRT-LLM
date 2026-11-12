# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-M FP8-weight GEMM on an MXFP8 operand a producer already quantized (no quantization
kernel)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_gemm import k3_fp8w_lowm_gemm as _lowm
from . import _fp8_reference as ref


class K3Fp8wMidGemm(OpWrapper):
    """``out [M, N]`` = BF16 ``dequant(xq, xs) @ dequant(w, w_scale)^T`` with the mid-M kernel of
    ``tactic`` (``k3_fp8w_lowm_gemm.mid_prequant_plan``); returns ``out``.

    ``xq``: E4M3 ``[M, K]``; ``xs``: UE8M0 1 x 32 exponents in the mid-M layout
    (``[K / 128][sf_blocks][512 B]``, ``sf_blocks = tactic.mma_n`` rows / 128 rounded up);
    ``w_scale``: the weight's DeepGEMM-packed scale.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Rows above the low-M cut-over of both tables (GB300: 64 for these weights), on the operand
    #: ``quantize_mid_operand`` produces for ``mid_prequant_plan``'s tactic.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MoE latent_up on the latent all-reduce's MXFP8 rows, 12 requests",
            spec={"n": 7168, "k": 3584, "m": 96},
        ),
        Cell(
            why="MoE latent_up at dep16's largest graph (32 requests x 8 rows): mma_n 256",
            spec={"n": 7168, "k": 3584, "m": 256},
        ),
        Cell(
            why="dep16 KDA / MLA o_proj (K = 12288) on the producer's MXFP8 rows, 20 requests",
            spec={"n": 7168, "k": 12288, "m": 160},
        ),
        Cell(
            why="dep16 MLA q_b (N = 18432, K = 1536), 28 requests: the 224-row bucket",
            spec={"n": 18432, "k": 1536, "m": 224},
        ),
    )

    note = """
    The GEMM :class:`K3Fp8wLowmGemm` runs above the low-M cut-over, without its quantization
    kernel: on the operand ``K3MxFp8QuantKernel`` (or a producer reproducing it) writes, the output
    has the BF16 path's bits. A PDL dependent: the weight streams before ``griddepcontrol.wait``.
    Inputs are not validated.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        tactic: "_lowm.K3MidMTactic",
    ) -> torch.Tensor:
        return _lowm.run_mid_prequantized(xq, xs, w, w_scale, out, tactic)

    def reference(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        tactic: "_lowm.K3MidMTactic",
    ) -> torch.Tensor:
        """The expected ``out``: FP64 product of the decoded operands, rounded to BF16."""
        m, k = xq.shape
        sf_blocks = -(-tactic.mma_n // 128)
        a = ref.dequant(xq, ref.mid_exponents(xs, m, k, sf_blocks), 32)
        return ref.matmul_bf16(a, ref.fp8_block_weight(w, w_scale))

    def is_valid(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        tactic: "_lowm.K3MidMTactic",
    ) -> None:
        # The launcher passes raw pointers: nothing checks shapes, dtypes or strides.
        m, k = xq.shape
        n = w.shape[0]
        assert xq.dtype == torch.float8_e4m3fn and xq.is_contiguous()
        assert w.dtype == torch.float8_e4m3fn and w.is_contiguous() and w.shape[1] == k
        assert w_scale.dtype == torch.int32, "w_scale must be the DeepGEMM-packed UE8M0 scale"
        sf_blocks = -(-tactic.mma_n // 128)
        assert xs.dtype == torch.uint8 and xs.numel() == (k // 128) * sf_blocks * 512, (
            "xs must be the mid-M layout of the tactic"
        )
        assert out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape == (m, n)
        assert m <= tactic.mma_n, f"M={m} exceeds the tactic's {tactic.mma_n} rows"


k3_fp8w_mid_gemm = K3Fp8wMidGemm()
