# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 router logits of up to 16 tokens with the router weight read before
``griddepcontrol.wait``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.k3r_moe import k3_router_exact as _router


class K3RouterLogits(OpWrapper):
    """FP32 ``[M, 896]`` logits ``hidden_states @ weight.T`` of BF16 ``hidden_states [M, 7168]``
    (``1 <= M <= 16``) and the BF16 ``[896, 7168]`` router weight."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Decode rows of the helix deployments (router_exact_kda): one row, one and two requests.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row: decode without MTP", spec={"m": 1}),
        Cell(why="one request with seven draft tokens", spec={"m": 8}),
        Cell(why="two requests: the op's 16-row limit", spec={"m": 16}),
    )

    note = """
    Bit-identical to ``dsv3_router_gemm_op(hidden_states, weight.t())``: the same 14 K slices of
    512, ``mma.sync`` fragments and slice-order sum, cut over clusters of two CTAs that hand the
    running sum over distributed shared memory. Each CTA copies its 112 KB weight slice into shared
    memory by TMA before ``griddepcontrol.wait`` and releases its dependents right after it. The
    output is a new FP32 tensor; unsupported inputs raise ValueError (``k3_router_exact.can_run``).
    """

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _router.k3_router_logits(hidden_states, weight)

    def reference(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """FP32 product of the BF16 operands (IEEE FP32 accumulation)."""
        with true_fp32_matmul():
            return hidden_states.float() @ weight.float().T


k3_router_logits = K3RouterLogits()
