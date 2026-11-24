# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing on SM107: ``noaux_tc`` top-16 written into the caller's rows (no sort)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_moe_common import NUM_EXPERTS, TOP_K, noaux_topk

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_route_tail import MAX_TOPK_CTAS
    from ...kernels.moe.rubin_route_tail import run_k3_route_tail as _cluster
    from ...kernels.moe.rubin_route_tail import run_k3_route_wide as _wide
else:
    MAX_TOPK_CTAS = 96

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

    ``logits`` is ``[M, 896]``, or ``S`` split-K partials ``[S, R, 896]`` of which rows ``[0, M)``
    (``M = rows``, at most 256) are routed: each logit is then ``((p[0] + p[1]) + ...) + p[S - 1]``,
    the sum in split order (``splitKreduce``'s for cuBLAS's split-K router GEMM), and the routing
    is ``noaux_tc_op``'s on those sums.

    ``route_approx`` (up to 256 rows) picks the candidates on ``tanh.approx`` scores first; it
    does not change the result. ``tail`` (up to 256 rows) sets the launch, ``(CTAs, warps per CTA,
    warps per token)`` (``rubin_route_tail.run_k3_route_tail``; above 8 CTAs a grid without a
    cluster); every launch runs the same per-token code, so it does not change the result either.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "DEP16 generation b12 k7: 96 rows into bucket 128's rows, approx candidates",
            96,
            128,
            route_approx=True,
        ),
        _cell("the same rows with exact candidates", 96, 128),
        _cell("one row", 1, 32, route_approx=True),
        _cell("the largest one-cluster launch: 256 rows", 256, 256, route_approx=True),
        _cell("the first multi-CTA row count: 257 rows", 257, 384),
        _cell("generation with context rows: 1024 rows on the multi-CTA grid", 1024, 1024),
        _cell("a scale other than 1: the FP64 renormalization", 32, 64, scale=2.5),
        _cell("tied scores: ties go to the lower expert", 16, 32, mode="ties", route_approx=True),
        _cell(
            "DEP16 generation b12 k7 under router_tf32_splitk: 8 partials of 96 rows, 96 CTAs of "
            "4-warp teams",
            96,
            128,
            route_approx=True,
            splits=8,
            tail=(96, 4, 4),
        ),
        _cell(
            "8 partials of 96 rows on the default cluster launch",
            96,
            128,
            route_approx=True,
            splits=8,
        ),
        _cell(
            "8 partials, rows past the routed ones (R 104 > M 96), 32 CTAs of 4-warp teams",
            96,
            128,
            route_approx=True,
            splits=8,
            tail=(32, 12, 4),
            partial_rows=104,
        ),
        _cell(
            "logits of 96 rows on 48 CTAs of 4-warp teams",
            96,
            128,
            route_approx=True,
            tail=(48, 8, 4),
        ),
        _cell(
            "2 partials of 7 rows on one 7-warp team per CTA",
            7,
            32,
            route_approx=True,
            splits=2,
            tail=(7, 7, 7),
        ),
    )

    note = """
    Logits are FP32 ``[M, 896]`` (``1 <= M <= 2048``) or ``[S, R, 896]`` (``1 <= S <= 64``,
    ``1 <= M <= min(R, 256)``) and the bias FP32 ``[896]``, dense and 16-byte aligned on one
    device; the launcher checks none of it (``is_valid`` does). The kernels keep no state between
    launches (the multi-CTA grid without its sort uses no workspace), so a captured CUDA graph
    replays them as is. Under PDL the logits are read after ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        route_approx: bool = False,
        rows: int | None = None,
        tail: tuple[int, int, int] | None = None,
    ) -> None:
        m = rows if logits.dim() == 3 else logits.shape[0]
        args = (logits, m, e_score_correction_bias, routed_scaling_factor, topk_ids, topk_weights)
        if m <= CLUSTER_MAX_M:
            launch = {}
            if tail is not None:
                launch = dict(num_ctas=tail[0], num_warps=tail[1], warps_per_token=tail[2])
            _cluster(*args, None, 0, NUM_EXPERTS, route_approx=route_approx, **launch)
        else:
            _wide(*args, None, 0, NUM_EXPERTS)

    def reference(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        route_approx: bool = False,
        rows: int | None = None,
        tail: tuple[int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ids int32 [M, 16], weights FP32 [M, 16])``: ``noaux_topk`` of the logits (of split-K
        partials, their sum in split order over rows ``[0, M)``), what rows ``[0, M)`` of the
        outputs receive. ``topk_ids`` / ``topk_weights`` are not read; ``route_approx`` and
        ``tail`` change how the kernel gets there, not the result."""
        if logits.dim() == 3:
            summed = logits[0, :rows].clone()
            for part in logits[1:]:
                summed = summed + part[:rows]
            logits = summed
        return noaux_topk(logits, e_score_correction_bias, routed_scaling_factor)

    def is_valid(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        route_approx: bool = False,
        rows: int | None = None,
        tail: tuple[int, int, int] | None = None,
    ) -> None:
        """The launchers take raw pointers: a mistyped, strided, misaligned or short operand, or a
        row count they do not serve, is read or written silently wrong."""
        if logits.dim() == 3:
            assert rows is not None and 1 <= rows <= min(logits.shape[1], CLUSTER_MAX_M), (
                f"split-K partials route rows [0, M), 1 <= M <= min(R, {CLUSTER_MAX_M}); got "
                f"rows={rows} of {tuple(logits.shape)}"
            )
            assert 1 <= logits.shape[0] <= 64 and logits.shape[2] == NUM_EXPERTS, (
                f"partials must be [1 <= S <= 64, R, {NUM_EXPERTS}], got {tuple(logits.shape)}"
            )
            m = rows
        else:
            assert rows is None, "rows is the routed count of split-K partials only"
            m = logits.shape[0] if logits.dim() == 2 else -1
            assert 1 <= m <= MAX_M and logits.shape[-1] == NUM_EXPERTS, (
                f"logits must be [1 <= M <= {MAX_M}, {NUM_EXPERTS}], got {tuple(logits.shape)}"
            )
        if tail is not None:
            ctas, warps, team = tail
            assert m <= CLUSTER_MAX_M and team in (1, 2, 4, 7) and warps % team == 0, tail
            assert warps <= 32 and (team == 1 or warps // team <= 15), (
                f"{tail}: at most 32 warps, at most 15 multi-warp teams"
            )
            assert 1 <= ctas <= MAX_TOPK_CTAS and -(-m // ctas) <= 32, (
                f"{tail}: 1..{MAX_TOPK_CTAS} CTAs of at most 32 rows each for {m} rows"
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
