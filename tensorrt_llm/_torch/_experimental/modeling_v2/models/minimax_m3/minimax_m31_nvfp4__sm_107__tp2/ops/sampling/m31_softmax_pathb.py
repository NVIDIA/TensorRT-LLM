# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 fixed-vocabulary softmax for FlashInfer Path B, rows 129 through 448.

One CTA per row preserves the ordered denominator recurrence and final CUB sum.
Warp integer maxima replace the per-chunk CUB max. The library is loaded outside
capture; the operation needs only its output, with no persistent workspace.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

MIN_ROWS = 129
MAX_ROWS = 448
VOCAB = 200064
ENABLED = not any(m31_disabled(item) for item in ("SOFTMAX_MC", "SOFTMAX_MC_WIDE", "SOFTMAX_PATHB"))
_STATE: dict = {}


def softmax_pathb_op():
    """Resolve the separately built native library, never loading during capture."""
    if "op" not in _STATE:
        if torch.cuda.is_current_stream_capturing():
            return None
        op = getattr(torch.ops.trtllm, "minimax_m31_softmax_pathb", None)
        if op is not None:
            m31_fake_ops.register()
            logger.info(
                "[M3.1] SOFTMAX_PATHB available: rows 129..448, vocab 200064, one CTA per row"
            )
        _STATE["op"] = op
    return _STATE["op"]


class M31SoftmaxPathB(OpWrapper):
    """Contiguous aligned fp32 [129..448, 200064] logits on SM107.

    Temperature is None, a scalar number, or contiguous fp32 [rows] on the
    same device. Unsupported inputs decline to the caller's stock operation.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(why=f"Path B row coverage: {rows}", spec=dict(rows=rows, d=VOCAB))
        for rows in range(MIN_ROWS, MAX_ROWS + 1)
    )
    note = "Preserves FlashInfer's ordered denominator and output arithmetic; no PDL or workspace."

    def __call__(
        self, logits: torch.Tensor, temperature: torch.Tensor | float | None
    ) -> torch.Tensor | None:
        if (
            not isinstance(logits, torch.Tensor)
            or not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.dim() != 2
            or not logits.is_contiguous()
            or not MIN_ROWS <= logits.shape[0] <= MAX_ROWS
            or logits.shape[1] != VOCAB
            or logits.data_ptr() % 16 != 0
        ):
            return None
        if temperature is None:
            temp_arr, temp_val = None, 1.0
        elif isinstance(temperature, torch.Tensor):
            if (
                temperature.dtype != torch.float32
                or temperature.dim() != 1
                or temperature.shape[0] != logits.shape[0]
                or not temperature.is_contiguous()
                or temperature.device != logits.device
            ):
                return None
            temp_arr, temp_val = temperature, 0.0
        elif isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            temp_arr, temp_val = None, float(temperature)
        else:
            return None
        op = softmax_pathb_op()
        if op is None:
            return None
        out = torch.empty_like(logits)
        op(logits, temp_arr, temp_val, out)
        return out

    def reference(
        self, logits: torch.Tensor, temperature: torch.Tensor | float | None
    ) -> torch.Tensor:
        """The installed FlashInfer softmax with the same temperature."""
        import flashinfer.sampling

        return flashinfer.sampling.softmax(logits, temperature)


m31_softmax_pathb = M31SoftmaxPathB()
