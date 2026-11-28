# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply handing the next MXFP8 GEMM its operand: the activation quantized in the same
kernel (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

from ....ops.activation.situ_and_mul import situ_and_mul
from ....ops.gemm.rubin_mxfp8_gemm import mxfp8_scales_to_r128c4, quantize_mxfp8_reference
from ...kernels.activation.rubin_situ_and_mul_mxfp8 import situ_and_mul_mxfp8 as _mx

#: Kimi K3.1's activation_situ_beta and activation_situ_linear_beta.
_BETA, _LINEAR_BETA = 4.0, 25.0


class RubinSituAndMulMxfp8(OpWrapper):
    """Return ``(codes, scales)``: the MXFP8 form of ``situ_and_mul(x, beta, linear_beta)``.

    ``codes`` is the E4M3 ``[M, d]`` operand and ``scales`` its flat R128c4 UE8M0 bytes, both bit
    for bit what ``mxfp8_quantize(situ_and_mul(x, ...), True, 32)`` makes of the BF16 rows the
    SiTU kernel stores, padding rows' scale bytes zero included. ``rubin_mxfp8_gemm`` takes
    them as its input operand.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"dep16 shared experts' down input (d=6144), a context chunk of {rows} rows",
                spec=dict(shape=(rows, 2 * 6144), beta=_BETA, linear_beta=_LINEAR_BETA),
            )
            for rows in (16384, 2954)
        ),
        Cell(
            why="one row",
            spec=dict(shape=(1, 2 * 6144), beta=_BETA, linear_beta=_LINEAR_BETA),
        ),
        Cell(
            why="shared experts over TP8 (d=768, one program per row narrower than the block)",
            spec=dict(shape=(37, 2 * 768), beta=_BETA, linear_beta=_LINEAR_BETA),
        ),
        Cell(
            why="saturated positive, negative and zero gates at magnitude 1e4 (bf16)",
            spec=dict(shape=(4, 2 * 768), beta=_BETA, linear_beta=_LINEAR_BETA, saturated=True),
        ),
    )
    note = """
    The dep16 context server's shared experts call it between the MXFP8 gate_up GEMM and
    rubin_mxfp8_gemm with the down weight: the BF16 activation is never stored and the standalone
    mxfp8_quantize of down's input does not run. Each value is the SiTU kernel's FP32
    expression (libdevice tanh, beta and linear_beta at run time) rounded to BF16, and the
    epilogue quantizes those values with the catalog quantizer's instructions. d a multiple of
    128. The first call of a process logs one INFO line naming this hand-off.
    """

    def __call__(
        self, x: torch.Tensor, beta: float, linear_beta: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info_once(
            "Kimi K3.1 Rubin MXFP8 hand-off: rubin_situ_and_mul_mxfp8 quantizes the shared "
            "experts' SiTU rows for down (no standalone quantization)",
            key="rubin_situ_and_mul_mxfp8 hand-off",
        )
        return _mx(x, beta, linear_beta)

    def reference(
        self, x: torch.Tensor, beta: float, linear_beta: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``situ_and_mul``'s reference in BF16, then the quantizer's reference (R128c4 scales)."""
        codes, scales = quantize_mxfp8_reference(situ_and_mul.reference(x, beta, linear_beta))
        return codes, mxfp8_scales_to_r128c4(scales)

    def is_valid(self, x: torch.Tensor, beta: float, linear_beta: float) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
        assert x.shape[1] % 256 == 0, "d = x.shape[1] / 2 a multiple of 128 (whole scale groups)"


rubin_situ_and_mul_mxfp8 = RubinSituAndMulMxfp8()
