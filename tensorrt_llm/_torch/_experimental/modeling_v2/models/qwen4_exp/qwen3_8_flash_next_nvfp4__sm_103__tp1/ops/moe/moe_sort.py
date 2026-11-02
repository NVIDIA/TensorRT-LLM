# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE sort: bucket the routed (token, expert) pairs into per-expert row tiles
for the CuteDSL grouped GEMMs, in one trtllm call."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _ckpt(tokens: int, tile_tokens_dim: int, dist: str = "router") -> dict:
    """The checkpoint's trunk MoE: 512 experts, top-10, all local (tp1), fp32 scales.

    `dist` names how the test draws the expert ids: `router` = each row's top-10 of
    uniform scores over the 512 experts; `tile_edges` = three experts hold exactly
    tile-1, tile and tile+1 rows, the rest `router`. Both put 10 distinct experts on
    every row, as the router does.
    """
    return dict(
        tokens=tokens,
        dist=dist,
        num_experts=512,
        top_k=10,
        local_expert_offset=0,
        local_num_experts=512,
        tile_tokens_dim=tile_tokens_dim,
        scales_dtype=torch.float32,
    )


class MoeSort(OpWrapper):
    """`torch.ops.trtllm.moe_sort`: routed (token, slot) pairs -> per-expert row tiles.

    Input: `token_selected_experts` `[T, top_k]` int32 global expert ids (the
    router's output), `token_final_scales` `[T, top_k]` (its weights). Call it
    with the op's own keyword names, as the CuteDSL MoE does.

    Numbering: an *expanded* index is `t * top_k + k` (row-major over
    `token_selected_experts`); a *permuted* index is a row of the grouped GEMM's
    M dimension. Each local expert `e = id - local_expert_offset` owns a
    contiguous run of `count_e` permuted rows starting on a tile boundary,
    experts in increasing order, the run padded up to whole tiles of
    `tile_tokens_dim` rows. Returns a list of six int32 tensors, in this order:

    - `tile_idx_to_expert_idx` `[max_tiles]`: the local expert of tile `i`;
    - `tile_idx_to_mn_limit` `[max_tiles]`: one past the last real row of tile
      `i`, as an absolute permuted index: `min((i + 1) * tile, start_e + count_e)`;
    - `expanded_idx_to_permuted_idx` `[T, top_k]`: the permuted row of each
      pair, `-1` for a pair whose expert is not local;
    - `permuted_idx_to_expanded_idx` `[max_tiles * tile]`: the inverse on real rows;
    - `total_num_padded_tokens` `[1]` = `num_non_exiting_tiles * tile`;
    - `num_non_exiting_tiles` `[1]`: tiles actually used.

    `max_tiles = m + (T * top_k - m) // tile` with `m = min(local_num_experts,
    T * top_k)` is computed on the host from the shapes and scalars alone, so
    every output shape is fixed by `T`, which is what lets a CUDA graph capture
    it. Only the first `num_non_exiting_tiles` entries of the tile arrays and
    the rows below each tile's `mn_limit` are written; everything else is left
    as allocated (see `note`).

    Fusion boundary: sorting only. Routing (the top-k) is the caller's, and so
    are the permutation of the activations (the FC1 gather does it) and the
    routing-weighted unpermute; this op does not read `token_final_scales` into
    any output.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per kernel path the target reaches at its tile, 128: block (T <= 4),
    # cluster (T 5..8192) and cooperative (T >= 8193).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tile 128: decode, 1 request without MTP; block kernel",
            spec=_ckpt(1, 128),
        ),
        Cell(
            why="tile 128: expert counts tile-1, tile, tile+1; the mn_limit clamp; cluster kernel",
            spec=_ckpt(300, 128, "tile_edges"),
        ),
        Cell(
            why="tile 128: 8193 rows; first cooperative-kernel T (prefill)",
            spec=_ckpt(8193, 128),
        ),
    )

    note = """
    Metadata consumed: none. Stateless; allocates its six outputs plus scratch on
    every call and runs on the current stream, with no host synchronization, so
    it captures into a CUDA graph.

    Call it as the engine's CuteDSL MoE does (`run_moe_nvfp4_impl`,
    `_torch/moe/fused_moe/fused_moe_cute_dsl.py`): `num_experts` = the slot
    count, `local_expert_offset` / `local_num_experts` = this rank's slots, and
    `tile_tokens_dim` = the grouped GEMMs' tile, 128 or 256 on sm_103. The
    downstream FC1 gather GEMM, output memset and FC2 take these six tensors as
    they are.

    Kernel path, by `T`:
      T 1..4         routingCustom::routingIndicesBlockKernel
      T 5..8192      routingCustom::routingIndicesClusterKernel
      T 8193..471859 routingInitExpertCounts + routingIndicesCoopKernel
      T >= 471860    routingInitExpertCounts + routingIndicesHistogramKernel
                     + routingIndicesOffsetsKernel
    The cooperative ceiling is (SMs - 8) * 512 * 64 / top_k rows with 512
    experts: 471859 on a 152-SM GB300, a different T on another SM count.

    Order within an expert's run is unspecified. The block kernel (T <= 4) lays
    each expert's rows out in token order and is bitwise repeatable; every other
    path assigns them by atomics, so `expanded_idx_to_permuted_idx` differs
    between identical calls. A test therefore compares each expert's rows as a
    set and the two mappings as exact inverses; the tile arrays and counts are
    deterministic.

    Not written: the tile arrays past `num_non_exiting_tiles`, and the rows of
    `permuted_idx_to_expanded_idx` from each tile's `mn_limit` to the tile's
    end. Eager, those hold whatever the allocator handed back; under a CUDA
    graph they hold the previous replay's indices, which look valid. Every
    consumer must bound its reads by `num_non_exiting_tiles` and
    `tile_idx_to_mn_limit`.

    `token_final_scales`: shape-checked and never written; its values never
    reach an output. Its pointer must be non-null, which is why T = 0 raises.

    Data hazards `is_valid` cannot see (values): a row naming the same expert
    twice keeps both pairs on the cluster kernel but loses one on the block
    kernel (that pair gets -1). An id outside [local_expert_offset,
    local_expert_offset + local_num_experts) gets -1, except that the cluster
    kernel reads ids through int16, so an id of 65536 + e sorts as expert e. The
    router produces neither.

    Loud: int64 ids, 1-D ids, a `[T, top_k]` size mismatch on either input,
    CPU tensors, top_k above 32, num_experts above 2048, T = 0.

    Silent, refused by `is_valid`: a non-contiguous `token_selected_experts`
    (read from `data_ptr()` as dense `[T, top_k]`); 2^31 or more elements (the
    host counts pairs in int32: it raises a negative-dimension error from 2^31
    and silently sorts only the wrapped remainder from 2^32); `tile_tokens_dim`
    <= 0 (the host divides by it).

    Certified only at the checkpoint's values: num_experts 512, top_k 10, all
    experts local (offset 0), tile 128 (the target's), fp32 scales, and T 1,
    300 and 8193 (block, cluster and the first cooperative T); CUDA-graph replay
    on new ids at T = 128; the contiguity guard. The order, unwritten-region,
    data-hazard and loud-rejection statements above were measured at bring-up
    and are not re-driven by the test. Expert parallelism, tile 256 and other
    tiles, other expert counts and the histogram path are uncertified.

    Hard limits (sm_103)
      T * top_k         boundary 2^31 pairs (int32 product)
      max_tiles * tile  boundary 2^31 padded rows (raises past it)
      expert id         int16 in the cluster kernel; ids < 2048 never reach it
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
        """Counting sort by local expert, stable in the expanded index.

        Within an expert, rows are in increasing expanded index; the test's compare
        accepts any order there. What the op leaves unwritten is `-1` here.
        `token_final_scales`, `num_experts` and `top_k` are not read: the scales
        never reach an output, and the ids' shape carries the rest.
        """
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
        # The host computes T * top_k in int32: from 2^31 the op raises a
        # negative-dimension error, from 2^32 it sorts only the wrapped remainder.
        numel = token_selected_experts.numel()
        assert numel < 2**31, (
            f"token_selected_experts has {numel} >= 2^31 elements; moe_sort counts "
            "pairs in int32 and silently sorts only the wrapped remainder past 2^32"
        )
        # The kernels read the ids from data_ptr() as a dense [T, top_k] buffer.
        assert token_selected_experts.is_contiguous(), (
            "token_selected_experts must be contiguous; a strided view is read as a "
            "dense [T, top_k] buffer and silently sorts the wrong ids"
        )
        assert tile_tokens_dim > 0, (
            f"tile_tokens_dim must be positive, got {tile_tokens_dim}; 0 returns a "
            "negative num_non_exiting_tiles without raising"
        )


moe_sort = MoeSort()
