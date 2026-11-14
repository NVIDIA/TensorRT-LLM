# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routing of FP32 router logits: ``noaux_tc`` top-16 and this rank's ``moe_sort`` tables
in one launch."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op

from ...kernels.k3r_moe import k3_route_tail as _route_tail


def sort_tables_reference(
    ids: torch.Tensor, local_expert_offset: int, local_num_experts: int, tile: int
) -> tuple[torch.Tensor, ...]:
    """``moe_sort``'s six tables of the top-k ``ids [M, top_k]`` for the local experts
    ``[local_expert_offset, local_expert_offset + local_num_experts)``, int32: a counting sort by
    local expert, rows of an expert in increasing expanded index (``token * top_k + slot``), each
    expert's rows padded to whole ``tile``-row tiles in local-expert order. Entries the op leaves
    unwritten are ``-1``."""
    dev = ids.device
    num_tokens, width = ids.shape
    num_expanded = num_tokens * width
    local = ids.reshape(-1).long() - local_expert_offset
    is_local = (local >= 0) & (local < local_num_experts)
    counts = torch.bincount(local[is_local], minlength=local_num_experts)
    tiles_per_expert = (counts + tile - 1) // tile
    num_tiles = int(tiles_per_expert.sum())
    filled = min(local_num_experts, num_expanded)
    max_tiles = filled + (num_expanded - filled) // tile
    run_start = (torch.cumsum(tiles_per_expert, 0) - tiles_per_expert) * tile
    tile_expert = torch.repeat_interleave(
        torch.arange(local_num_experts, device=dev), tiles_per_expert
    )
    tile_idx = torch.arange(num_tiles, device=dev)
    mn_limit = torch.minimum((tile_idx + 1) * tile, run_start[tile_expert] + counts[tile_expert])
    expanded = torch.arange(num_expanded, device=dev)[is_local]
    expert = local[is_local]
    order = torch.sort(expert, stable=True).indices
    expanded, expert = expanded[order], expert[order]
    rank = torch.arange(expert.numel(), device=dev) - (torch.cumsum(counts, 0) - counts)[expert]
    permuted = run_start[expert] + rank

    def unwritten(n: int) -> torch.Tensor:
        return torch.full((n,), -1, dtype=torch.long, device=dev)

    tile_idx_to_expert_idx = unwritten(max_tiles)
    tile_idx_to_expert_idx[:num_tiles] = tile_expert
    tile_idx_to_mn_limit = unwritten(max_tiles)
    tile_idx_to_mn_limit[:num_tiles] = mn_limit
    permuted_idx_to_expanded_idx = unwritten(max_tiles * tile)
    permuted_idx_to_expanded_idx[permuted] = expanded
    expanded_idx_to_permuted_idx = unwritten(num_expanded)
    expanded_idx_to_permuted_idx[expanded] = permuted
    outs = (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx.view(num_tokens, width),
        permuted_idx_to_expanded_idx,
        torch.tensor([num_tiles * tile], device=dev),
        torch.tensor([num_tiles], device=dev),
    )
    return tuple(t.to(torch.int32) for t in outs)


class K3RouteTail(OpWrapper):
    """``noaux_tc`` top-16 of FP32 router logits ``[M, 896]`` (or ``S`` split-K partials
    ``[S, R, 896]``, summed in split order) and the ``moe_sort`` tables (tile 128) of the experts
    ``[local_expert_offset, local_expert_offset + local_num_experts)``.

    Returns ``(topk_ids int32 [M, 16], topk_weights fp32 [M, 16], tile_idx_to_expert_idx,
    tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
    total_num_padded_tokens, num_non_exiting_tiles)``, new tensors with ``moe_sort``'s shapes;
    only the valid table entries are written. ``num_tokens``: ``M`` (default every row).
    ``workspace``: above 256 tokens, the device's persistent multi-CTA sort workspace
    (``k3_route_tail.wide_workspace``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: The local expert windows of the deployments (896 experts): tp1 (all), dep16 EP16 (56),
    #: helix8 EP8 (112) and helix16 MoE TP4 x EP4 (224), at decode rows and above the cluster
    #: kernel's 256 tokens; ``splits``: split-K partials ``[S, M, 896]`` summed by the op.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp1: every expert local, one row",
            spec={"m": 1, "offset": 0, "local": 896, "splits": 0},
        ),
        Cell(
            why="dep16 EP16: a 56-expert window in the middle, 32 rows",
            spec={"m": 32, "offset": 392, "local": 56, "splits": 0},
        ),
        Cell(
            why="helix8 EP8: the last 112-expert window, one request with seven draft tokens",
            spec={"m": 8, "offset": 784, "local": 112, "splits": 0},
        ),
        Cell(
            why="helix16 MoE TP4 x EP4: a 224-expert window on two split-K logit partials",
            spec={"m": 16, "offset": 224, "local": 224, "splits": 2},
        ),
        Cell(
            why="dep16 at 32 requests x 8 rows: the cluster kernel's 256-token limit",
            spec={"m": 256, "offset": 840, "local": 56, "splits": 0},
        ),
        Cell(
            why="dep16 EP16 context, 896 tokens: the multi-CTA sort with its workspace (it serves "
            "at most 224 local experts, so not tp1 above 256 tokens)",
            spec={"m": 896, "offset": 392, "local": 56, "splits": 0},
        ),
    )

    note = """
    Up to 256 tokens one cluster of at most eight CTAs; 257..2048 tokens independent CTAs whose
    last arrival (a counter in ``workspace``, reset by the same launch) builds the tables. The
    top-k is bit-identical to ``noaux_tc_op`` on the same logits (selection, order, weights); an
    expert's rows are in ascending token order. ``workspace`` must be created outside CUDA graph
    capture; the kernels keep no other state between launches. Unsupported inputs raise ValueError.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int,
        num_tokens: int = -1,
        workspace: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        return _route_tail.k3_route_tail(
            logits,
            e_score_correction_bias,
            routed_scaling_factor,
            local_expert_offset,
            local_num_experts,
            tile_tokens_dim,
            num_tokens,
            workspace,
        )

    def reference(
        self,
        logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int,
        num_tokens: int = -1,
        workspace: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """The catalog ``noaux_tc_op`` reference (one group, top-16) on the logits, then
        :func:`sort_tables_reference` of its top-k; ``workspace`` is not read."""
        if logits.dim() == 3:
            rows = logits.shape[1] if num_tokens < 0 else num_tokens
            total = logits[0, :rows].float()
            for split in range(1, logits.shape[0]):
                total = total + logits[split, :rows].float()
            logits = total
        elif num_tokens >= 0:
            logits = logits[:num_tokens]
        weights, ids = noaux_tc_op.reference(
            logits, e_score_correction_bias, 1, 1, _route_tail.TOP_K, routed_scaling_factor
        )
        tables = sort_tables_reference(ids, local_expert_offset, local_num_experts, tile_tokens_dim)
        return (ids, weights.float(), *tables)


k3_route_tail = K3RouteTail()
