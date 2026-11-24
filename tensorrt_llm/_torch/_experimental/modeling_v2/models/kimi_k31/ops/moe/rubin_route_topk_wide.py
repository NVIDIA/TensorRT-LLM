# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing on SM107: ``noaux_tc`` top-16 of a context chunk's rows in one launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_moe_common import NUM_EXPERTS, TOP_K, noaux_topk

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_route_tail import run_k3_route_wide as _wide

#: Rows of one launch: a context MoE chunk of up to 64k rows per rank.
MAX_M = 65536
#: The multi-CTA grid's shape, ``(tokens_per_cta, num_warps, warps_per_token)``: CTAs of 8 rows,
#: one warp per row, the fastest of the shapes tried on a 16k-row chunk. Every shape gives the
#: same bits.
LAUNCH = (8, 8, 1)


def _cell(why: str, rows: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, **spec))


class RubinRouteTopkWide(OpWrapper):
    """``noaux_tc_op``'s top-16 of FP32 router logits ``[M, 896]``, ``1 <= M <= 65536``.

    The experts in ``noaux_tc_op``'s order and its FP32 weights (the unbiased ``sigmoid`` scores of
    the picks renormalized to sum to 1 and scaled by ``routed_scaling_factor``), bit for bit, with
    one expert group: the family's routing tail run top-k only on its multi-CTA grid (independent
    CTAs of 8 rows, no sort, no workspace), one launch on the current stream.

    Returns ``(topk_ids int32 [M, 16], topk_weights fp32 [M, 16])``, newly allocated (the order
    of ``noaux_tc_op``'s ``(weights, ids)`` swapped, as the MoE block takes them).
    """

    ARCHS = frozenset({Arch.SM_107})

    # A context server's MoE routes every row of its rank's chunk at once: up to 16384 rows per
    # rank on DEP16 / DEP12 / DEP8 (max_num_tokens), fewer on a short chunk or a decode step.
    CELLS: tuple[Cell, ...] = (
        _cell("one row: one 8-row CTA, seven idle rows", 1),
        _cell("a decode step's rows on a context server: 96 rows", 96),
        _cell("the largest row count rubin_route_topk_out serves: 2048 rows", 2048),
        _cell("first row count past it: a partial last CTA", 2049),
        _cell("half a 16k chunk: 8192 rows", 8192),
        _cell("a DEP16 / DEP12 / DEP8 context rank's 16k chunk in one launch", 16384),
        _cell("the largest launch: 65536 rows", MAX_M),
        _cell("a scale other than 1: the FP64 renormalization", 3000, scale=2.5),
        _cell("tied scores: ties go to the lower expert", 4096, mode="ties"),
    )

    note = """
    Logits are FP32 ``[M, 896]`` and the bias FP32 ``[896]``, dense and 16-byte aligned on one
    device, ``1 <= M <= 65536``; the launcher checks the row count only (``is_valid`` checks the
    rest). The kernel serves exactly 896 experts and top-16 (compile-time constants) and keeps no
    state between launches. Under PDL the logits are read after ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = logits.shape[0]
        ids = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
        weights = torch.empty((m, TOP_K), dtype=torch.float32, device=logits.device)
        tokens_per_cta, num_warps, warps_per_token = LAUNCH
        _wide(
            logits,
            m,
            e_score_correction_bias,
            routed_scaling_factor,
            ids,
            weights,
            None,
            0,
            NUM_EXPERTS,
            tokens_per_cta=tokens_per_cta,
            num_warps=num_warps,
            warps_per_token=warps_per_token,
        )
        return ids, weights

    def reference(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``noaux_topk``: ``noaux_tc_op``'s arithmetic (tanh-form sigmoid, ties to the lower
        expert, FP64 renormalization rounded once to FP32) in native torch."""
        return noaux_topk(logits, e_score_correction_bias, routed_scaling_factor)

    def is_valid(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> None:
        """The launcher takes raw pointers: a mistyped, strided, misaligned or short operand is
        read silently wrong, and a row count past the bound is refused by the launcher."""
        m = logits.shape[0] if logits.dim() == 2 else -1
        assert 1 <= m <= MAX_M and logits.shape[1] == NUM_EXPERTS, (
            f"logits must be [1 <= M <= {MAX_M}, {NUM_EXPERTS}], got {tuple(logits.shape)}"
        )
        assert logits.dtype == e_score_correction_bias.dtype == torch.float32, "FP32 operands"
        assert e_score_correction_bias.numel() == NUM_EXPERTS, "one bias per expert"
        for name, t in (("logits", logits), ("e_score_correction_bias", e_score_correction_bias)):
            assert t.is_contiguous() and t.data_ptr() % 16 == 0, (
                f"{name} must be dense and 16-byte aligned"
            )
            assert t.device == logits.device, f"{name} is on another device"


rubin_route_topk_wide = RubinRouteTopkWide()
