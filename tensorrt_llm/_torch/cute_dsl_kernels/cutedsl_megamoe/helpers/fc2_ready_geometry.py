# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host geometry for FC2 completion counters in the swap-AB kernel."""


def derive_fc2_ready_geometry(
    *,
    hidden_size: int,
    mma_tiler_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    fallback_cluster_shape_mn: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return (tokens per ready slot, CTA completions per ready slot).

    Work IDs and token slots use the preferred *logical* cluster. Fixed groups
    of fallback clusters partition that cluster's CTA coordinates; they do not
    create extra token slots or smaller completion targets. Every logical CTA
    publishes once, including padded hidden tiles in the last cluster tile.
    """
    mma_cta_count = 2 if use_2cta_instrs else 1
    if cluster_shape_mn[1] != 1:
        raise ValueError("FC2 token-ready geometry requires hidden-only cluster splitting (N=1).")
    if mma_tiler_mnk[0] % mma_cta_count or cluster_shape_mn[0] % mma_cta_count:
        raise ValueError("FC2 token-ready geometry must preserve complete MMA CTA groups.")
    if fallback_cluster_shape_mn is not None:
        fallback_m, fallback_n = fallback_cluster_shape_mn
        if (
            fallback_n != 1
            or fallback_m <= 0
            or cluster_shape_mn[0] % fallback_m
            or fallback_m % mma_cta_count
        ):
            raise ValueError(
                "FC2 fallback clusters must partition the preferred hidden axis into complete MMA groups."
            )
    cta_hidden = mma_tiler_mnk[0] // mma_cta_count
    hidden_per_logical_cluster = cta_hidden * cluster_shape_mn[0]
    hidden_cluster_tiles = (
        hidden_size + hidden_per_logical_cluster - 1
    ) // hidden_per_logical_cluster
    return mma_tiler_mnk[1], hidden_cluster_tiles * cluster_shape_mn[0]
