# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`q_b_e4m3`: the fp8 context attention's query, `q_a x (q_b / s_q)` in e4m3, on the MXF8
GEMM where it built, else on the prefill GEMM's e4m3 entry (`_build_context_mha_fp8` builds
one of the two, the layer's `qb8` weight in that GEMM's form)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from . import cute_dsl_projection_mxf8 as _mxf8_op
from .cute_dsl_projection import cute_dsl_projection
from .cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8


@dataclass(frozen=True)
class QbE4m3Handle:
    """`runner`: the MXF8 GEMM or the prefill GEMM's e4m3 entry (None: the fp8 context
    attention did not build, and nothing calls this op); `n` the query width,
    `heads * qk_dim`."""

    runner: Any
    n: int


class QbE4m3(OpWrapper):
    """The e4m3 query `[rows, n]` of `q_a_ctx` against the layer's `qb8`. `pre` is q_a's MXF8
    operand written by the q_a norm (`qkv_a_rmsnorm`'s `q8`): the MXF8 GEMM then skips its
    activation quant; only the MXF8 GEMM takes it."""

    def __call__(self, q_a_ctx: torch.Tensor, qb8, pre, handle: QbE4m3Handle) -> torch.Tensor:
        rows = int(q_a_ctx.shape[0])
        q8 = torch.empty([rows, handle.n], dtype=torch.float8_e4m3fn, device=q_a_ctx.device)
        if isinstance(handle.runner, _mxf8_op.Mxf8ProjectionRunner):
            return cute_dsl_projection_mxf8(q_a_ctx, qb8, q8, handle.runner, pre=pre)
        assert pre is None, "the q_a norm's MXF8 operand needs the MXF8 q_b"
        return cute_dsl_projection(q_a_ctx, qb8, q8, handle.runner)

    def reference(self, q_a_ctx: torch.Tensor, qb8, pre, handle: QbE4m3Handle) -> torch.Tensor:
        """Not modelled in plain torch: `qb8` is a quantized weight in the serving GEMM's own
        form; the two GEMMs' entries (`cute_dsl_projection_mxf8`, `cute_dsl_projection`)
        carry the references."""
        raise NotImplementedError("see cute_dsl_projection_mxf8 / cute_dsl_projection")


q_b_e4m3 = QbE4m3()
