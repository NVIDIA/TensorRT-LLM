# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout arithmetic of the in-cluster K split of the Rubin block-scaled dense GEMM.

Pure Python (no CuTe DSL import), shared by the kernel (``cluster_k``), its epilogue
(``streamk_epilogue``) and the tactic generation of the runner.

``cluster_k`` CTAs of a cluster compute one output tile from disjoint K slices; in the epilogue the
tile is cut into subtiles of ``epi_n`` columns, subtile ``s`` is owned by K rank ``s % cluster_k``,
and every other rank pushes its FP32 partial of the subtile into a mailbox slot of the owner. The
mailbox is sized for the rank that owns the most subtiles, so it takes shared memory from the A/B
pipeline stages (about 20-30 KB per stage); the subtile width is chosen to keep it small.
"""

EPILOGUE_THREADS = 128  # four epilogue warps, one tensor-memory lane (output row) per thread
MAX_EPILOGUE_SUBTILE = 32  # columns of an epilogue subtile (also bounded by the base epilogue tile)
MIN_EPILOGUE_SUBTILE = 16  # 4 FP32 per push x 4 pushes: the narrowest subtile the epilogue copies


def mailbox_barriers(cluster_k: int, epi_n: int, tile_n: int) -> int:
    """mbarriers of the mailbox: one per subtile the busiest K rank owns."""
    return -(-(tile_n // epi_n) // cluster_k)


def mailbox_floats(cluster_k: int, epi_n: int, tile_n: int) -> int:
    """FP32 elements of the mailbox: one slot per (owned subtile, source rank)."""
    return mailbox_barriers(cluster_k, epi_n, tile_n) * (cluster_k - 1) * epi_n * EPILOGUE_THREADS


def select_epilogue_subtile(cluster_k: int, tile_n: int, default_epi_n: int) -> int:
    """Width (columns) of the epilogue subtile of a ``tile_n``-wide CTA tile.

    Candidates are 32 and 16 columns (never wider than ``default_epi_n``, the base epilogue's
    choice) that divide the tile. Every K rank should own at least one subtile, so candidates with
    fewer than ``cluster_k`` subtiles are used only when no candidate has enough. Among them the one
    with the smallest mailbox wins, ties going to the wider subtile (fewer stores and barriers).
    """
    widths = [
        w
        for w in (MAX_EPILOGUE_SUBTILE, MIN_EPILOGUE_SUBTILE)
        if w <= default_epi_n and tile_n % w == 0
    ]
    if not widths:
        raise ValueError(f"no epilogue subtile of {tile_n} columns (base subtile {default_epi_n})")
    enough = [w for w in widths if tile_n // w >= cluster_k]
    pool = enough or [min(widths)]
    return min(pool, key=lambda w: (mailbox_floats(cluster_k, w, tile_n), -w))


def mailbox_bytes(cluster_k: int, tile_n: int, default_epi_n: int = MAX_EPILOGUE_SUBTILE) -> int:
    """Shared memory (bytes) of the mailbox for the subtile ``select_epilogue_subtile`` picks."""
    epi_n = select_epilogue_subtile(cluster_k, tile_n, default_epi_n)
    return 4 * mailbox_floats(cluster_k, epi_n, tile_n)
