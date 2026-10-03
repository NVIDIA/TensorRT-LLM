# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 MoE front's geometry choice and the weight rows each plan reads, without a GPU.

trtllm::k3_moe_front runs the head in 64-row half-tiles, one round with every head k-tile of a CTA on chip before
the grid wait (``half_geometry``), only when those half-tiles fit beside the shared tiles in the GEMV clusters the
device leaves next to the two role clusters; otherwise in 128-row tiles (``geometry``). With 384 shared columns
(Kimi K3 TP16's per-rank width) and a GB200's 15 clusters of 8 CTAs, that is W = 16 alone: 280 head rows per rank
are 5 half-tiles, + 6 shared tiles = 11 GEMV clusters of the 13 left. At W = 4 (1120 rows, 18 half-tiles) and W = 8
(560 rows, 9 half-tiles) they do not fit. A 4-rank run therefore never reaches the half-tile path; this checks the
selection, and that ``front_weight``'s rows are the rows each plan's weight descriptor covers.
"""

import pytest
import torch

kernel = pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.k3_moe_front")
front_op = pytest.importorskip("tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.front_op")

K = 7168  # the MoE input's width
SHARED_COLS = 384
RING = 4
# Clusters of 8 CTAs, one per SM, that a GB200 holds at once (the kernel module's statement).
GB200_CLUSTERS = 15


def _plan(world: int, max_clusters: int = GB200_CLUSTERS):
    """(head tiles of the plan, shared tiles, GEMV clusters, half-tile head) as trtllm::k3_moe_front picks them."""
    half = kernel.half_geometry(world, SHARED_COLS, max_clusters, K, RING)
    if half is not None:
        return (*half, True)
    return (*kernel.geometry(world, SHARED_COLS, max_clusters), False)


def test_half_tile_head_only_at_w16():
    """W 16 runs the head as 5 half-tiles in one round of 11 GEMV clusters; W 4 and W 8 run 128-row tiles."""
    assert kernel.half_geometry(16, SHARED_COLS, GB200_CLUSTERS, K, RING) == (5, 6, 11)
    assert _plan(16) == (5, 6, 11, True)
    assert kernel.half_geometry(8, SHARED_COLS, GB200_CLUSTERS, K, RING) is None
    assert _plan(8) == (5, 6, 11, False)
    assert kernel.half_geometry(4, SHARED_COLS, GB200_CLUSTERS, K, RING) is None
    assert _plan(4) == (9, 6, 13, False)
    for world in (4, 8, 16):
        assert kernel.supports(world, SHARED_COLS, GB200_CLUSTERS, K, RING), world


def test_half_tile_boundary():
    """At W 16 the half-tile plan needs 11 GEMV clusters beside the 2 role clusters: 13 clusters fit it, 12 do not."""
    assert _plan(16, max_clusters=13) == (5, 6, 11, True)
    assert _plan(16, max_clusters=12)[-1] is False


@pytest.mark.parametrize("world", [4, 8, 16])
def test_front_weight_rows_match_the_plan(world):
    """``front_weight``'s rows are the rows the plan's weight descriptor covers, and the shared rows start where the
    plan's first shared tile reads; the half-tiles stay inside the zero-padded head rows."""
    head_rows = kernel.head_rows(world)
    head = torch.zeros(head_rows, K, dtype=torch.bfloat16)
    gate_up = torch.zeros(2 * SHARED_COLS, K, dtype=torch.bfloat16)
    rows = front_op.front_weight(head, gate_up).shape[0]
    padded_head = kernel.head_tiles(world) * kernel.CTA_M
    assert rows == padded_head + 2 * SHARED_COLS
    n_head, n_shared, _, half = _plan(world)
    assert kernel._weight_rows(world, n_head, n_head + n_shared, half) == rows
    my_tiles = K // kernel.CTA_K // kernel.SPLIT
    shift, _ = kernel._half_plan(half, world, n_head, RING, my_tiles)
    assert n_head * kernel.CTA_M + shift == padded_head  # the first shared tile's first weight row
    if half:
        assert head_rows <= n_head * kernel.HALF_M <= padded_head


def test_front_weight_packs_the_shared_rows():
    """The head rows first, zero-padded to whole 128-row tiles; then every 32 rows hold 16 gate rows and the 16 up rows
    of the same columns."""
    world, k = 16, 64
    head_rows = kernel.head_rows(world)
    head = torch.arange(head_rows, dtype=torch.float32)[:, None].expand(head_rows, k).contiguous()
    gate_up = (10_000 + torch.arange(2 * SHARED_COLS, dtype=torch.float32))[:, None].expand(-1, k)
    w = front_op.front_weight(head, gate_up.contiguous())
    padded_head = kernel.head_tiles(world) * kernel.CTA_M
    assert torch.equal(w[:head_rows], head)
    assert bool((w[head_rows:padded_head] == 0).all())
    shared = w[padded_head:, 0]
    for block in range(SHARED_COLS // 16):
        gate = 10_000 + torch.arange(16 * block, 16 * block + 16, dtype=torch.float32)
        up = gate + SHARED_COLS
        assert torch.equal(shared[32 * block : 32 * block + 16], gate)
        assert torch.equal(shared[32 * block + 16 : 32 * block + 32], up)
