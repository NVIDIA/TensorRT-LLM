# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing on SM107: ``noaux_tc`` top-16 written into the caller's rows (no sort)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....ops.moe._rubin_moe_common import NUM_EXPERTS, TOP_K, noaux_topk

if IS_CUTLASS_DSL_AVAILABLE:
    from ....kernels.moe.rubin_route_tail import run_k3_route_tail as _cluster
    from ....kernels.moe.rubin_route_tail import run_k3_route_wide as _wide

#: Rows one cluster routes; the multi-CTA grid takes up to ``MAX_M``.
CLUSTER_MAX_M = 256
MAX_M = 2048


def _cell(why: str, rows: int, out_rows: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, out_rows=out_rows, **spec))


class RubinRouteTopkOut(OpWrapper):
    """``noaux_tc_op``'s top-16 of FP32 router logits, written into rows ``[0, M)`` of the caller's
    ``topk_ids`` (int32) and ``topk_weights`` (FP32), each ``[>= M, 16]`` and dense.

    The experts in ``noaux_tc_op``'s order and its weights, bit for bit (one expert group): the
    family's routing tail without its sort, one cluster up to 256 rows, the multi-CTA grid up to
    2048. Nothing else is written; one launch on the current stream. Returns nothing.

    Up to 256 rows the kernel picks the candidates on ``tanh.approx`` scores first; that does not
    change the result.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("DEP16 generation b12 k7: 96 rows into bucket 128's rows", 96, 128),
        _cell("one row", 1, 32),
        _cell("the largest one-cluster launch: 256 rows", 256, 256),
        _cell("the first multi-CTA row count: 257 rows", 257, 384),
        _cell("generation with context rows: 1024 rows on the multi-CTA grid", 1024, 1024),
        _cell("a scale other than 1: the FP64 renormalization", 32, 64, scale=2.5),
        _cell("tied scores: ties go to the lower expert", 16, 32, mode="ties"),
    )

    note = """
    Logits are FP32 ``[M, 896]`` and the bias FP32 ``[896]``, dense and 16-byte aligned on one
    device, ``1 <= M <= 2048``; the launcher checks none of it (``is_valid`` does). The kernels
    keep no state between launches (the multi-CTA grid without its sort uses no workspace), so a
    captured CUDA graph replays them as is. Under PDL the logits are read after
    ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        args = (logits, e_score_correction_bias, routed_scaling_factor, topk_ids, topk_weights)
        if logits.shape[0] <= CLUSTER_MAX_M:
            _cluster(*args, None, 0, NUM_EXPERTS)
        else:
            _wide(*args)

    def reference(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ids int32 [M, 16], weights FP32 [M, 16])``: ``noaux_topk``, what rows ``[0, M)`` of
        the outputs receive. ``topk_ids`` / ``topk_weights`` are not read."""
        return noaux_topk(logits, e_score_correction_bias, routed_scaling_factor)

    def is_valid(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """The launchers take raw pointers: a mistyped, strided, misaligned or short operand, or a
        row count they do not serve, is read or written silently wrong."""
        m = logits.shape[0] if logits.dim() == 2 else -1
        assert 1 <= m <= MAX_M and logits.shape[1] == NUM_EXPERTS, (
            f"logits must be [1 <= M <= {MAX_M}, {NUM_EXPERTS}], got {tuple(logits.shape)}"
        )
        assert logits.dtype == e_score_correction_bias.dtype == torch.float32, "FP32 operands"
        assert e_score_correction_bias.numel() == NUM_EXPERTS, "one bias per expert"
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        for name, t in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
            assert t.dim() == 2 and t.shape[0] >= m and t.shape[1] == TOP_K, (
                f"{name} must hold [{m}, {TOP_K}], got {tuple(t.shape)}"
            )
        for name, t in (("logits", logits), ("e_score_correction_bias", e_score_correction_bias),
                        ("topk_ids", topk_ids), ("topk_weights", topk_weights)):  # fmt: skip
            assert t.is_contiguous() and t.data_ptr() % 16 == 0, (
                f"{name} must be dense and 16-byte aligned"
            )
            assert t.device == logits.device, f"{name} is on another device"


rubin_route_topk_out = RubinRouteTopkOut()
