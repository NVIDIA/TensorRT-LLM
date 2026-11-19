# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing tail on SM107: ``noaux_tc`` top-16 and one rank's ``moe_sort`` tables in one
launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_moe_common import TILE, noaux_topk, token_ordered_sort

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_route_tail import k3_route_tail as _launch

#: Rows of the one-cluster kernel.
CLUSTER_MAX_M = 256


def _cell(why: str, rows: int, offset: int, local: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, offset=offset, local=local, **spec))


class RubinRouteTail(OpWrapper):
    """``noaux_tc_op`` top-16 of FP32 router logits plus this rank's ``moe_sort`` tables.

    Returns ``(topk_ids int32 [M, 16], topk_weights fp32 [M, 16], tile_idx_to_expert_idx,
    tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
    total_num_padded_tokens, num_non_exiting_tiles)``, newly allocated. Top-k is ``noaux_tc_op``'s
    on the same logits bit for bit; the tables are ``moe_sort``'s at tile 128 with the rows of an
    expert in token order. The CuteDSL backends take them as presorted routing and skip their own
    sort. One 8-CTA cluster runs it, up to 256 rows.

    The kernel picks candidates on ``tanh.approx`` scores first; ``route_tail_bands`` names the
    launch shapes by row count (``"helix8"``: helix8 generation's; ``""``: the default ones).
    Neither changes the result.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("TEP8 generation b1 k7: 8 rows, EP8 rank 0's 112 experts", 8, 0, 112),
        _cell("TEP8 generation: the last EP8 rank's window", 8, 784, 112),
        _cell("TEP8 generation b1 k7: EP8 rank 3's window", 8, 336, 112),
        _cell(
            "helix8 generation b8 k7: 64 rows, a TP2xEP4 rank's 224 experts, helix8 bands",
            64,
            224,
            224,
            route_tail_bands="helix8",
        ),
        _cell(
            "helix8 generation b16 k5: 96 rows, helix8 bands",
            96,
            672,
            224,
            route_tail_bands="helix8",
        ),
        _cell(
            "helix8 generation b24 k5: 144 rows, helix8 bands",
            144,
            0,
            224,
            route_tail_bands="helix8",
        ),
        _cell("largest one-cluster launch: 256 rows", 256, 448, 112),
        _cell("a scale other than 1: the FP64 renormalization", 32, 0, 112, scale=2.5),
        _cell("tied scores: ties go to the lower expert", 16, 0, 112, mode="ties"),
    )

    note = """
    Logits are FP32 ``[M, 896]``, the bias FP32 ``[896]``, both dense and 16-byte aligned on one
    device; the op raises (ValueError) on anything else, on ``tile_tokens_dim != 128``, on a row
    count outside 1..256 and on a local window outside the 896 experts. The kernel keeps no
    state, so a captured graph replays it as is.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int = TILE,
        *,
        route_tail_bands: str = "",
    ) -> tuple[torch.Tensor, ...]:
        return _launch(
            logits,
            e_score_correction_bias,
            routed_scaling_factor,
            local_expert_offset,
            local_num_experts,
            tile_tokens_dim,
            route_tail_bands=route_tail_bands,
        )

    def reference(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int = TILE,
        *,
        route_tail_bands: str = "",
    ) -> tuple[torch.Tensor, ...]:
        """``noaux_topk`` then ``token_ordered_sort``. ``route_tail_bands`` is ignored: it changes
        how the kernel gets there, not the result."""
        ids, weights = noaux_topk(logits, e_score_correction_bias, routed_scaling_factor)
        tables = token_ordered_sort(ids, local_expert_offset, local_num_experts, tile_tokens_dim)
        return (ids, weights, *tables)

    def is_valid(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int = TILE,
        *,
        route_tail_bands: str = "",
    ) -> None:
        """Nothing to add: the op raises (ValueError) on every input it would answer wrongly -- a
        dtype other than FP32, a width other than 896, a strided or misaligned tensor, a row count
        outside 1..256, a window outside the experts."""
        return None


rubin_route_tail = RubinRouteTail()
