# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE sort for the K3.1 CuteDSL experts on SM107: the routed (token, expert) pairs of one rank's
expert window bucketed into per-expert row tiles, in one engine call (``trtllm::moe_sort``)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The K3.1 router: routed experts and experts per token.
_EXPERTS, _TOP_K = 896, 16


def _cell(rows: int, local: int, offset: int, tile: int) -> dict:
    return dict(
        tokens=rows,
        num_experts=_EXPERTS,
        top_k=_TOP_K,
        local_expert_offset=offset,
        local_num_experts=local,
        tile_tokens_dim=tile,
    )


class RubinMoeSort(OpWrapper):
    """``torch.ops.trtllm.moe_sort``: the routed pairs of experts ``[local_expert_offset,
    local_expert_offset + local_num_experts)`` into tiles of ``tile_tokens_dim`` rows.

    Numbering: an *expanded* index is ``t * top_k + k``; a *permuted* index is a row of the
    grouped GEMMs. Each local expert owns a run of permuted rows starting on a tile boundary,
    experts in increasing order, the run padded to whole tiles. Returns six int32 tensors:
    ``tile_idx_to_expert_idx`` and ``tile_idx_to_mn_limit`` ``[max_tiles]``,
    ``expanded_idx_to_permuted_idx`` ``[T, top_k]`` (-1 for a pair outside the window),
    ``permuted_idx_to_expanded_idx`` ``[max_tiles * tile]``, ``total_num_padded_tokens`` ``[1]``
    and ``num_non_exiting_tiles`` ``[1]``. ``max_tiles`` follows from ``T`` and the scalars, so
    the shapes are fixed by ``T`` and the call captures into a CUDA graph. ``token_final_scales``
    is read for its shape only.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 context: a 2048-row chunk, EP8 rank 0's 112 experts, tile 128",
            spec=_cell(2048, 112, 0, 128),
        ),
        Cell(
            why="TEP8 context at tile 256 (the tuner's other tile)", spec=_cell(2048, 112, 0, 256)
        ),
        Cell(why="TEP8 context at tile 512 (SM107's largest tile)", spec=_cell(2048, 112, 0, 512)),
        Cell(
            why="DEP16 context after the all-to-all: 16 ranks x 64 rows, rank 3's 56 experts",
            spec=_cell(1024, 56, 168, 128),
        ),
        Cell(
            why="TEP8 generation b64 k7: 512 rows for FC12, EP8 rank 5's window, tile 128",
            spec=_cell(512, 112, 560, 128),
        ),
        Cell(why="one row: the block kernel", spec=_cell(1, 112, 0, 128)),
    )

    note = """
    Order within an expert's run is unspecified (atomics, except the block kernel for T <= 4);
    the tile arrays and counts are deterministic. Entries past ``num_non_exiting_tiles`` and
    permuted rows from a tile's ``mn_limit`` to its end are not written: every consumer bounds its
    reads by those two. The kernel reads ids through int16 on the cluster path; K3.1 ids are below
    896.
    """

    def __call__(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int,
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            tile_tokens_dim=tile_tokens_dim,
        )

    def reference(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int,
    ) -> list[torch.Tensor]:
        """Counting sort by local expert, stable in the expanded index; what the op leaves
        unwritten is -1 here. ``token_final_scales``, ``num_experts`` and ``top_k`` are not read."""
        ids = token_selected_experts
        dev = ids.device
        tile = tile_tokens_dim
        num_tokens, width = ids.shape
        num_expanded = num_tokens * width
        local = ids.reshape(-1).long() - local_expert_offset
        is_local = (local >= 0) & (local < local_num_experts)
        counts = torch.bincount(local[is_local], minlength=local_num_experts)
        tiles_per_expert = (counts + tile - 1) // tile
        num_tiles = int(tiles_per_expert.sum())
        first = min(local_num_experts, num_expanded)
        max_tiles = first + (num_expanded - first) // tile

        run_start = (torch.cumsum(tiles_per_expert, 0) - tiles_per_expert) * tile
        tile_expert = torch.repeat_interleave(
            torch.arange(local_num_experts, device=dev), tiles_per_expert
        )
        tile_idx = torch.arange(num_tiles, device=dev)
        mn_limit = torch.minimum(
            (tile_idx + 1) * tile, run_start[tile_expert] + counts[tile_expert]
        )
        expanded = torch.arange(num_expanded, device=dev)[is_local]
        expert = local[is_local]
        order = torch.sort(expert, stable=True).indices
        expanded, expert = expanded[order], expert[order]
        rank = torch.arange(expert.numel(), device=dev) - (torch.cumsum(counts, 0) - counts)[expert]
        permuted = run_start[expert] + rank

        def unwritten(n: int) -> torch.Tensor:
            return torch.full((n,), -1, dtype=torch.long, device=dev)

        t2e = unwritten(max_tiles)
        t2e[:num_tiles] = tile_expert
        t2lim = unwritten(max_tiles)
        t2lim[:num_tiles] = mn_limit
        p2e = unwritten(max_tiles * tile)
        p2e[permuted] = expanded
        e2p = unwritten(num_expanded)
        e2p[expanded] = permuted
        outs = (
            t2e,
            t2lim,
            e2p.view(num_tokens, width),
            p2e,
            torch.tensor([num_tiles * tile], device=dev),
            torch.tensor([num_tiles], device=dev),
        )
        return [t.to(torch.int32) for t in outs]

    def is_valid(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        local_num_experts: int,
        tile_tokens_dim: int,
    ) -> None:
        # The kernels read the ids from data_ptr() as a dense [T, top_k] int32 buffer.
        assert token_selected_experts.is_contiguous(), (
            "token_selected_experts must be contiguous: a strided view sorts the wrong ids"
        )
        assert token_selected_experts.dtype == torch.int32, "expert ids are read as int32"
        # The host counts pairs in int32 and silently sorts a wrapped remainder past 2^32.
        assert token_selected_experts.numel() < 2**31, "2^31 or more routed pairs"
        assert tile_tokens_dim > 0, "tile_tokens_dim must be positive (the host divides by it)"
        # The cluster kernel reads ids through int16: 65536 + e would sort as expert e.
        assert num_experts < 2**15, f"{num_experts} experts overflow the int16 id read"


rubin_moe_sort = RubinMoeSort()
