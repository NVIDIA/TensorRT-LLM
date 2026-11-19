# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 router logits on SM107 for up to 16 BF16 rows, into a new FP32 tensor."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_moe_common import router_logits_reference

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_router_exact import k3_router_logits as _launch


class RubinRouterLogits(OpWrapper):
    """FP32 ``[M, 896]`` logits of ``M <= 16`` BF16 rows ``[M, 7168]`` against the BF16 router
    ``[896, 7168]``, allocated per call.

    The bits are ``dsv3_router_gemm_op(hidden_states, weight.t())``'s tensor-core kernel: same
    fragments, MMA chains and slice order, cut over 2-CTA clusters. The router weight is read
    into shared memory before ``griddepcontrol.wait``, so under PDL that copy overlaps the
    previous kernel.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=rows))
        for rows, why in (
            (1, "one row: the smallest launch"),
            (2, "two rows"),
            (6, "TEP8 generation b1 k5: 6 rows"),
            (8, "TEP8 generation b1 k7: 8 rows, the KDA layers' router"),
            (13, "a row count that is not a power of two"),
            (16, "TEP8 generation b2 k7: 16 rows, the whole m16 tile"),
        )
    )

    note = """
    Kimi K3.1 calls it only on its KDA layers' routers and only up to 16 rows; above that the
    target takes ``dsv3_router_gemm_op`` (cuBLAS there). Hidden states must be BF16, dense,
    16-byte aligned ``[M, 7168]`` with ``1 <= M <= 16``, the weight BF16 dense aligned
    ``[896, 7168]`` on the same SM100/103/107 device; the op raises (ValueError) otherwise. No
    workspace or state; launches capture in a CUDA graph after the first call compiled.
    """

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _launch(hidden_states, weight)

    def reference(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``router_logits_reference``."""
        return router_logits_reference(hidden_states, weight)

    def is_valid(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> None:
        """Nothing to add: the op checks every input it reads and raises on the rest."""
        return None


rubin_router_logits = RubinRouterLogits()
