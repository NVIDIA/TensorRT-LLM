# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The plan's lever decisions, pushed into the kernel modules that read them at launch.

Several kernel modules of this target hold module-level settings their kernels read when
they build or launch (a split-K stage count, a transport layout, whether a launch releases
its dependents at entry). Each module declares them with the default of the path it refines
and one keyword-only ``configure``; ``apply`` calls every ``configure`` once per process,
from the model constructor, right after ``dispatch.compose`` and before any installer runs,
with what the plan resolved (``dispatch.Plan.enabled``, ``dispatch.Plan.param``). Under
``NO_PATHS`` it changes nothing, so a deployment without a composition keeps every default.

A module whose settings all keep their defaults is not imported for it: some kernel modules
load a compiler, or change what a worker does, by being imported.

One setting follows the composition rather than a lever: a sharded drafter fc creates its BLAS
handles at install. (The split-K kernels' conflict-free mailbox, reduce-scatter epilogue for
split-K 8 and deeper stages in oversized shared memory are ``ops/gemm/splitk_gemm``'s own
defaults, so no lever configures them.)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from . import dispatch as _dispatch

#: ``splitk_frontier_tables``: the TP4 sites' 9..32-row kernels, ``site -> [[last row count,
#: kernel], ...]`` (``dense_gemms.dispatcher``'s table format).
TEP4_WIDE_GEMMS: Mapping[str, Sequence[Tuple[int, str]]] = {
    "qkv": ((16, "splitk:64,16,2,16"), (32, "splitk:64,32,2,13")),
    "o_proj": ((16, "splitk:64,16,1,13"), (32, "splitk:64,32,1,13")),
    "pool": ((16, "splitk:64,16,8,10"), (32, "splitk:64,32,8,9")),
}
#: ``splitk_frontier_tables``: the TP4 sites' 1..8-row kernels.
TEP4_NARROW_GEMMS: Mapping[str, Sequence[Tuple[int, str]]] = {
    "qkv": ((1, "direct:256,2"), (2, "direct:64,4"), (3, "direct:32,2"), (8, "splitk:64,8,2,18")),
    "o_proj": ((2, "direct:128,4"), (8, "splitk:64,8,1,14")),
    "pool": ((1, "direct:256,2"), (6, "direct:128,2"), (8, "splitk:64,8,4,14")),
}
#: ``tep8_wide64_splitk``: the TP8 33..64-row kernels by site.
TEP8_WIDE64_KERNELS: Mapping[str, str] = {"qkv": "splitk:64,64,4,8"}
#: ``tep8_drafter_wide_gemms``: the default SITES of its 65..MAX_ROWS-row one launch (the
#: parameter's default; ``fixed_low_m_gemm.TEP8_DRAFTER_WIDE_SITES``).
TEP8_DRAFTER_WIDE_SITES: Tuple[str, ...] = ("conv_coef", "qkv")
#: ``tep8_drafter_wide_gemms``: the default PREFETCH_WEIGHTS (``drafter_l2_prefetch.WEIGHT_NAMES``).
DRAFTER_PREFETCH_WEIGHTS: Tuple[str, ...] = ("o_proj", "conv_mlp", "conv_attn", "qkv")
#: ``tep8_drafter_wide_gemms``: the default WIDE_TILE of that launch.
TEP8_DRAFTER_WIDE_TILE = "64x128"
#: ``tep8_fc_shard_wide``: the programs of the TP8 drafter fc's gather.
TEP8_FC_GATHER_PROGRAMS = 128
#: ``split_kv_deep_rings``: the TMA ring depth of the split-KV decode FMHA kernels.
SPLIT_KV_RING_STAGES = 20
#: ``wide_norm_gate_pdl``: the most rows a PDL post-FFN norm-gate takes.
NORM_GATE_PDL_MAX_ROWS = 128
#: ``tep8_early_multicast_hop``: the row band whose CTAs pad their shared memory, the rows
#: from which a CTA joins two rows, and the pad.
HOP_PAD_ROWS = (9, 40)
HOP_TWO_ROWS_MIN = 41
HOP_SMEM_PAD = 177408
#: ``wide_router_per_row``: the per-row router shard of 16..32-row steps tiles its rows as
#: ``rows:rows per CTA:experts per CTA`` (``router_shard``'s format).
SM107_ROUTER_ROWS_TILING = "16:1:2,24:2:2,32:2:2"


def _tables(table: Mapping[str, Sequence[Tuple[int, str]]]) -> dict:
    return {site: [[rows, kernel] for rows, kernel in cells] for site, cells in table.items()}


def _configure(module: str, settings: dict, defaults: dict) -> None:
    """Import ``module`` (relative to this target) and call its ``configure`` with ``settings``,
    unless every setting is its default: then the module is not imported for it, since some
    kernel modules load their compilers or change a worker's behaviour by being imported."""
    if settings == defaults:
        return
    importlib.import_module(f"{__package__}.{module}").configure(**settings)


def _context_sp_parts(plan: "_dispatch.Plan") -> Tuple[str, ...]:
    """The parts the sequence-parallel context step adds to its base parts:
    ``context_sp_extra_parts``'s PARTS, and the router input under ``context_sp_router_input``
    (the aggregated TEP4 selection)."""
    parts = (
        tuple(plan.param("context_sp_extra_parts", "PARTS"))
        if plan.enabled("context_sp_extra_parts")
        else ()
    )
    if plan.enabled("context_sp_router_input") and "router_input" not in parts:
        parts += ("router_input",)
    return parts


#: ``kernels.context_sequence_parallel.overlap``'s ``configure`` defaults: every overlap off.
CONTEXT_SP_OVERLAP_DEFAULTS = dict(
    oproj_chunks=0,
    oproj_min_rows=8192,
    oproj_high_priority=True,
    tail_pipeline=False,
    moe_rs_chunks=0,
    moe_rs_min_rows=8192,
    moe_rs_high_priority=True,
)


def _context_sp_overlap(plan: "_dispatch.Plan") -> dict:
    """The sequence-parallel step's reduce-scatter overlaps: ``context_sp_oproj_overlap``,
    ``context_sp_tail_pipeline`` and ``context_sp_moe_rs_overlap`` with their parameters."""
    settings = dict(CONTEXT_SP_OVERLAP_DEFAULTS)
    levers = (("context_sp_oproj_overlap", "oproj"), ("context_sp_moe_rs_overlap", "moe_rs"))
    for lever, prefix in levers:
        if plan.enabled(lever):
            settings[f"{prefix}_chunks"] = int(plan.param(lever, "CHUNKS"))
            settings[f"{prefix}_min_rows"] = int(plan.param(lever, "MIN_ROWS"))
            settings[f"{prefix}_high_priority"] = bool(plan.param(lever, "HIGH_PRIORITY"))
    settings["tail_pipeline"] = plan.enabled("context_sp_tail_pipeline")
    return settings


def apply(plan: "_dispatch.Plan") -> None:
    """Configure the kernel modules from ``plan``, each with one ``configure`` call.

    A no-op for ``NO_PATHS``: every setting keeps its default. A module whose settings all keep
    their defaults is not imported.
    """
    if not plan:
        return
    on = plan.enabled
    tables = on("splitk_frontier_tables")
    after_join = on("weight_prefetch_after_join")
    wide64 = on("tep8_wide64_splitk")
    drafter_wide = on("tep8_drafter_wide_gemms")
    fc_wide = on("tep8_fc_shard_wide")
    one_cluster = on("swa_one_cluster_block")
    deep = on("split_kv_deep_rings")
    deep_multi = on("split_kv_deep_ring")
    prerouted = on("prerouted_experts")
    push = on("tep8_router_push")
    norm_prefetch = (
        bool(plan.param("tep8_pool_programmatic_fork", "NORM_PREFETCH"))
        if on("tep8_pool_programmatic_fork")
        else False
    )
    early = on("tep8_early_multicast_hop")
    join = on("tep8_publish_join")
    wide_gathers = on("tep8_wide_projection_gathers")
    direct = on("tep8_direct_hop")
    publish_splitk = on("tep8_publish_splitk")
    tile_placement = on("tep8_tile_placement")
    finalize64 = on("tep8_finalize_transport_64")
    # The 2..4-request bands (decode steps of 9..32 rows) of the 5..8-request paths: each the
    # first row count of its lever (33, the band served before, unless its MIN_ROWS knob moves).
    wide64_min_rows = plan.param("tep8_wide64_splitk", "MIN_ROWS") if wide64 else 33
    publish_min_rows = plan.param("tep8_publish_splitk", "MIN_ROWS") if publish_splitk else 33
    settings = (
        # Dense GEMMs.
        (
            "ops.gemm.splitk_gemm",
            dict(wide64=wide64),
            dict(wide64=False),
        ),
        (
            "kernels.dense_gemms.dispatcher",
            dict(
                pool_splitk8=on("splitk_pool_split8"),
                tep4_wide_gemms=_tables(TEP4_WIDE_GEMMS) if tables else None,
                tep4_narrow_gemms=_tables(TEP4_NARROW_GEMMS) if tables else None,
                tep8_wide64=wide64,
                tep8_wide64_kernels=dict(TEP8_WIDE64_KERNELS) if wide64 else None,
                tep8_wide64_min_rows=wide64_min_rows,
                # 65..MAX_ROWS-row QKV in one launch of the wide64 tactic (64: cuBLAS there).
                tep8_wide64_max_rows=plan.param("tep8_wide64_splitk", "MAX_ROWS") if wide64 else 64,
                tep8_wide64_wide_tile=(
                    plan.param("tep8_wide64_splitk", "WIDE_TILE") if wide64 else "64x64"
                ),
                tep8_wide64_wide_split_k=(
                    plan.param("tep8_wide64_splitk", "WIDE_SPLIT_K") if wide64 else 0
                ),
                # PREP_FOLD: the multi-request decode prep in the 2-CTA QKV launch's epilogue.
                tep8_wide64_prep_fold=(
                    bool(plan.param("tep8_wide64_splitk", "PREP_FOLD")) if wide64 else False
                ),
                # The QKV call of an armed direct-hop step forks o_proj's L2 prefetch (0: none).
                tep8_direct_prefetch_min_rows=(
                    plan.param("tep8_direct_hop", "PREFETCH_MIN_ROWS") if direct else 0
                ),
                prefetch_join="after" if after_join else "before",
                drop_weight_prefetch=("qkv",) if after_join else (),
            ),
            dict(
                pool_splitk8=False,
                tep4_wide_gemms=None,
                tep4_narrow_gemms=None,
                tep8_wide64=False,
                tep8_wide64_kernels=None,
                tep8_wide64_min_rows=33,
                tep8_wide64_max_rows=64,
                tep8_wide64_wide_tile="64x64",
                tep8_wide64_wide_split_k=0,
                tep8_wide64_prep_fold=False,
                tep8_direct_prefetch_min_rows=0,
                prefetch_join="before",
                drop_weight_prefetch=(),
            ),
        ),
        (
            "ops.gemm.fixed_low_m_gemm",
            dict(
                tep8_wide64=wide64,
                tep8_drafter_wide=drafter_wide,
                tep8_fc_wide=fc_wide,
                # 65..MAX_ROWS-row drafter GEMMs in one launch of the 64-row tactic (64: cuBLAS).
                tep8_drafter_max_rows=(
                    plan.param("tep8_drafter_wide_gemms", "MAX_ROWS") if drafter_wide else 64
                ),
                tep8_drafter_wide_tile=(
                    plan.param("tep8_drafter_wide_gemms", "WIDE_TILE")
                    if drafter_wide
                    else TEP8_DRAFTER_WIDE_TILE
                ),
                tep8_drafter_wide_sites=(
                    tuple(plan.param("tep8_drafter_wide_gemms", "SITES"))
                    if drafter_wide
                    else TEP8_DRAFTER_WIDE_SITES
                ),
                # tep8_drafter_wide_gemms__DEEP_STAGES: deeper pipelines on the 16..64-row entries.
                tep8_drafter_deep_stages=bool(
                    drafter_wide and plan.param("tep8_drafter_wide_gemms", "DEEP_STAGES")
                ),
            ),
            dict(
                tep8_wide64=False,
                tep8_drafter_wide=False,
                tep8_fc_wide=False,
                tep8_drafter_max_rows=64,
                tep8_drafter_wide_tile=TEP8_DRAFTER_WIDE_TILE,
                tep8_drafter_wide_sites=TEP8_DRAFTER_WIDE_SITES,
                tep8_drafter_deep_stages=False,
            ),
        ),
        (
            "ops.spec.dflash_window_attention",
            # tep8_drafter_wide_gemms__WINDOW_TILES: multi-tile window splits on wide calls.
            dict(
                multi_tiles=(
                    plan.param("tep8_drafter_wide_gemms", "WINDOW_TILES") if drafter_wide else 1
                ),
                multi_min_rows=(
                    plan.param("tep8_drafter_wide_gemms", "WINDOW_MIN_ROWS") if drafter_wide else 65
                ),
            ),
            dict(multi_tiles=1, multi_min_rows=65),
        ),
        (
            "kernels.drafter_block_decode.block_decode",
            # tep8_drafter_wide_gemms__FINISH_NORM: the non-fused hops' tail in one launch.
            dict(
                finish_norm=bool(
                    drafter_wide and plan.param("tep8_drafter_wide_gemms", "FINISH_NORM")
                ),
                qk_fold=bool(drafter_wide and plan.param("tep8_drafter_wide_gemms", "QK_FOLD")),
            ),
            dict(finish_norm=False, qk_fold=False),
        ),
        (
            "kernels.context_kv_fork.fork",
            # drafter_kv_fork_before_head__PAGED_CHAIN: the side chain over the manager's pages in two launches.
            dict(
                paged_side_chain=(
                    int(plan.param("drafter_kv_fork_before_head", "PAGED_CHAIN"))
                    if on("drafter_kv_fork_before_head")
                    else 0
                )
            ),
            dict(paged_side_chain=0),
        ),
        (
            "kernels.drafter_silu_fold.fold",
            # tep8_drafter_wide_gemms__SILU_FOLD: the drafter MLP's activation in the gate_up GEMM's epilogue.
            dict(
                silu_fold=bool(drafter_wide and plan.param("tep8_drafter_wide_gemms", "SILU_FOLD"))
            ),
            dict(silu_fold=False),
        ),
        (
            "kernels.drafter_pdl_chain.chain",
            # tep8_drafter_wide_gemms__PDL_CONV_MAX_ROWS: the conv entry trigger past 32 rows.
            # tep8_drafter_wide_gemms__SELECTOR_FORK_MAX_ROWS: the selector fork on multi-request heads.
            dict(
                conv_max_rows=(
                    plan.param("tep8_drafter_wide_gemms", "PDL_CONV_MAX_ROWS")
                    if drafter_wide
                    else 32
                ),
                selector_max_rows=(
                    plan.param("tep8_drafter_wide_gemms", "SELECTOR_FORK_MAX_ROWS")
                    if drafter_wide
                    else 7
                ),
            ),
            dict(conv_max_rows=32, selector_max_rows=7),
        ),
        (
            "kernels.drafter_l2_prefetch.prefetch",
            # tep8_drafter_wide_gemms__PREFETCH: drafter GEMM weights asked of the L2 ahead of use.
            dict(
                level=plan.param("tep8_drafter_wide_gemms", "PREFETCH") if drafter_wide else 0,
                min_rows=(
                    plan.param("tep8_drafter_wide_gemms", "PREFETCH_MIN_ROWS")
                    if drafter_wide
                    else 1
                ),
                weights=(
                    tuple(plan.param("tep8_drafter_wide_gemms", "PREFETCH_WEIGHTS"))
                    if drafter_wide
                    else DRAFTER_PREFETCH_WEIGHTS
                ),
            ),
            dict(level=0, min_rows=1, weights=DRAFTER_PREFETCH_WEIGHTS),
        ),
        (
            "kernels.next_qkv_prefetch.prefetch",
            dict(
                skip_rows=plan.param("next_qkv_prefetch_skip", "MAX_ROWS")
                if on("next_qkv_prefetch_skip")
                else 0,
                # The rows the publication takes below 33 fork no next-QKV prefetch, as 33..64-row
                # steps fork none (tep8_publish_splitk's MIN_ROWS; 0: none).
                published_min_rows=publish_min_rows if publish_min_rows < 33 else 0,
            ),
            dict(skip_rows=0, published_min_rows=0),
        ),
        (
            "spec_worker",
            dict(
                fc_shard_wide=fc_wide,
                fc_shard_wide_rows=plan.param("tep8_fc_shard_wide", "MAX_ROWS") if fc_wide else 64,
                fc_gather_programs=TEP8_FC_GATHER_PROGRAMS if fc_wide else 0,
                # The sharded fc's BLAS handles are created at install (a composition constant).
                fc_shard_prewarm=on("drafter_fc_shard"),
            ),
            dict(
                fc_shard_wide=False,
                fc_shard_wide_rows=64,
                fc_gather_programs=0,
                fc_shard_prewarm=False,
            ),
        ),
        (
            "kernels.boundary_early_release.triggers",
            dict(norm_gate_pdl_max_rows=NORM_GATE_PDL_MAX_ROWS if on("wide_norm_gate_pdl") else 0),
            dict(norm_gate_pdl_max_rows=0),
        ),
        # Attention.
        (
            "ops.attention.swa_block",
            dict(
                cl8_tep4="all" if one_cluster else "0",
                cl8_tep8=one_cluster,
                cl8_tep8_min_requests=(
                    plan.param("swa_one_cluster_block", "TEP8_MIN_REQUESTS") if one_cluster else 5
                ),
                cl8_tep8_max_requests=(
                    plan.param("swa_one_cluster_block", "TEP8_MAX_REQUESTS") if one_cluster else 8
                ),
                cl8_tep8_wide_min_requests=(
                    plan.param("swa_one_cluster_block", "TEP8_WIDE_MIN_REQUESTS")
                    if one_cluster
                    else 0
                ),
                prep_split=on("swa_prep_split"),
                one2=on("swa_eight_requests"),
                # swa_one_cluster_block's EXACT (int9: fixed at 1), passed with the lever on: every CL8
                # build is the exact one. With the lever off nothing runs CL8 and nothing is passed.
                **(
                    {"cl8_exact": True}
                    if on("swa_one_cluster_block") and plan.param("swa_one_cluster_block", "EXACT")
                    else {}
                ),
            ),
            dict(
                cl8_tep4="0",
                cl8_tep8=False,
                cl8_tep8_min_requests=5,
                cl8_tep8_max_requests=8,
                cl8_tep8_wide_min_requests=0,
                prep_split=False,
                one2=False,
            ),
        ),
        (
            "kernels.full_attention_prep_fold.attention",
            dict(fold_multi=on("full_attention_fold_multi")),
            dict(fold_multi=False),
        ),
        (
            "ops.attention.full_attention_split_kv_kernel",
            dict(stages=SPLIT_KV_RING_STAGES if deep else 16),
            dict(stages=16),
        ),
        (
            "ops.attention.multi_request_split_kv",
            # ``split_kv_deep_ring`` deepens this ring only (the aggregated TEP4 selection).
            dict(sm107_stages=SPLIT_KV_RING_STAGES if deep or deep_multi else None),
            dict(sm107_stages=None),
        ),
        # Experts and routing.
        (
            "kernels.replicated_experts.ring_buddy",
            dict(
                tep4_prerouted=prerouted,
                tep8_prerouted=prerouted,
                tep8_prerouted_max_rows=(
                    plan.param("prerouted_experts", "TEP8_MAX_ROWS") if prerouted else 64
                ),
                bf16_half_window=on("bf16_half_window"),
                tep8_max_rows=plan.param("tep8_ring_buddy_wide", "MAX_ROWS")
                if on("tep8_ring_buddy_wide")
                else None,
            ),
            dict(
                tep4_prerouted=False,
                tep8_prerouted=False,
                tep8_prerouted_max_rows=64,
                bf16_half_window=False,
                tep8_max_rows=None,
            ),
        ),
        (
            "ops.comm.pool_direct_route",
            dict(prerouted_meta=prerouted),
            dict(prerouted_meta=False),
        ),
        (
            "ops.moe.ring_buddy_assign",
            dict(
                rb_pdl=on("tep8_ring_buddy_pdl"),
                tile_placement=tile_placement,
                tile_min_rows=(
                    plan.param("tep8_tile_placement", "MIN_ROWS") if tile_placement else 33
                ),
                tile_max_rows=(
                    plan.param("tep8_tile_placement", "MAX_ROWS") if tile_placement else 64
                ),
                tile_cost=(
                    (
                        plan.param("tep8_tile_placement", "COST_A"),
                        plan.param("tep8_tile_placement", "COST_B"),
                    )
                    if tile_placement
                    else (20, 11)
                ),
                bf16_tile_cost=(
                    (
                        plan.param("tep8_tile_placement", "BF16_COST_A"),
                        plan.param("tep8_tile_placement", "BF16_COST_B"),
                    )
                    if tile_placement
                    else (0, 0)
                ),
                bf16_tile=plan.param("tep8_tile_placement", "BF16_TILE") if tile_placement else 8,
                wide_warps=(
                    plan.param("tep8_ring_buddy_wide", "WIDE_WARPS")
                    if on("tep8_ring_buddy_wide")
                    else 8
                ),
                cuda_tables=(bool(plan.param("tep8_router_push", "RB_CUDA")) if push else False),
            ),
            dict(
                rb_pdl=False,
                tile_placement=False,
                tile_min_rows=33,
                tile_max_rows=64,
                tile_cost=(20, 11),
                bf16_tile_cost=(0, 0),
                bf16_tile=8,
                wide_warps=8,
                cuda_tables=False,
            ),
        ),
        (
            "ops.moe.router_push",
            dict(
                tep8_push=push,
                tep8_min_rows=plan.param("tep8_router_push", "MIN_ROWS") if push else 9,
                tep8_route=push,
                tep8_cl16_max_rows=plan.param("tep8_router_push", "CL16_MAX_ROWS") if push else 0,
                tep8_route_row_max=plan.param("tep8_router_push", "ROUTE_ROW_MAX") if push else 64,
                tep8_route_rows_per_cta=(
                    plan.param("tep8_router_push", "ROUTE_ROWS_PER_CTA") if push else 1
                ),
                tep8_early_trigger=(
                    bool(plan.param("tep8_router_push", "RB_CUDA")) if push else False
                ),
            ),
            dict(
                tep8_push=False,
                tep8_min_rows=9,
                tep8_route=False,
                tep8_cl16_max_rows=0,
                tep8_route_row_max=64,
                tep8_route_rows_per_cta=1,
                tep8_early_trigger=False,
            ),
        ),
        (
            "ops.moe.bf16x3_router_gemm",
            dict(prewait=on("tep8_router_prewait")),
            dict(prewait=False),
        ),
        (
            "ops.moe.router_shard",
            # The per-row router shard's tiling, wherever that shard runs.
            dict(rows_tiling=SM107_ROUTER_ROWS_TILING if on("wide_router_per_row") else None),
            dict(rows_tiling=None),
        ),
        # Communication and step glue.
        (
            "ops.moe.finalize_allreduce_transport",
            dict(
                tep8_max_tokens=(
                    plan.param("tep8_finalize_transport_64", "TOKENS") if finalize64 else 16
                ),
                trigger="entry" if on("finalize_entry_trigger") else "late",
                index_prefetch=(
                    bool(plan.param("tep8_finalize_transport_64", "INDEX_PREFETCH"))
                    if finalize64
                    else False
                ),
            ),
            dict(tep8_max_tokens=16, trigger="late", index_prefetch=False),
        ),
        (
            "ops.allreduce.post_attn_hop",
            dict(entry_trigger=on("hop_entry_trigger")),
            dict(entry_trigger=False),
        ),
        (
            "kernels.drafter_hops.hops",
            # 65..TEP8_MAX_ROWS-row TP8 drafter forwards on the wide fused hop (64: the served rows).
            dict(
                tep8_wide_max_rows=(
                    plan.param("wide_drafter_hops", "TEP8_MAX_ROWS")
                    if on("wide_drafter_hops")
                    else 64
                ),
                tep8_two_shot_min_rows=(
                    plan.param("wide_drafter_hops", "TEP8_TWO_SHOT_MIN_ROWS")
                    if on("wide_drafter_hops")
                    else 0
                ),
                tep8_two_shot_max_rows=(
                    plan.param("wide_drafter_hops", "TEP8_TWO_SHOT_MAX_ROWS")
                    if on("wide_drafter_hops")
                    else 128
                ),
            ),
            dict(tep8_wide_max_rows=64, tep8_two_shot_min_rows=0, tep8_two_shot_max_rows=128),
        ),
        (
            "ops.allreduce.post_attn_hop_wide",
            dict(
                direct=direct,
                direct_max_rows=plan.param("tep8_direct_hop", "MAX_ROWS") if direct else 64,
                direct_producer=plan.param("tep8_direct_hop", "PRODUCER") if direct else "grid",
                direct_producer_stages=(
                    plan.param("tep8_direct_hop", "PRODUCER_STAGES") if direct else 0
                ),
                direct_min_rows=plan.param("tep8_direct_hop", "MIN_ROWS") if direct else 33,
                direct_poll_first=(
                    bool(plan.param("tep8_direct_hop", "POLL_FIRST")) if direct else False
                ),
                early_launch=early,
                multicast=early,
                pad_rows=HOP_PAD_ROWS if early else (0, 1024),
                r2="w" if early else "0",
                r2_min_rows=HOP_TWO_ROWS_MIN if early else 33,
                smem_pad=HOP_SMEM_PAD if early else 0,
            ),
            dict(
                direct=False,
                direct_max_rows=64,
                direct_producer="grid",
                direct_producer_stages=0,
                direct_min_rows=33,
                direct_poll_first=False,
                early_launch=False,
                multicast=False,
                pad_rows=(0, 1024),
                r2="0",
                r2_min_rows=33,
                smem_pad=0,
            ),
        ),
        (
            "ops.comm.projection_publish",
            dict(
                tep8_publish_splitk=publish_splitk,
                tep8_publish_splitk_max_rows=(
                    plan.param("tep8_publish_splitk", "MAX_ROWS") if publish_splitk else 64
                ),
                tep8_publish_splitk_min_rows=publish_min_rows,
                tep8_publish_splitk_wide_pool_tile=(
                    plan.param("tep8_publish_splitk", "WIDE_POOL_TILE")
                    if publish_splitk
                    else "64x16"
                ),
                tep8_publish_splitk_wide_unpool_tile=(
                    plan.param("tep8_publish_splitk", "WIDE_UNPOOL_TILE")
                    if publish_splitk
                    else "64x32"
                ),
                tep8_publish_splitk_wide_tile_min_rows=(
                    plan.param("tep8_publish_splitk", "WIDE_TILE_MIN_ROWS")
                    if publish_splitk
                    else 65
                ),
                tep8_pool_programmatic=on("tep8_pool_programmatic_fork"),
                tep8_join_pdl=join,
                join_cluster=join,
            ),
            dict(
                tep8_publish_splitk=False,
                tep8_publish_splitk_max_rows=64,
                tep8_publish_splitk_min_rows=33,
                tep8_publish_splitk_wide_pool_tile="64x16",
                tep8_publish_splitk_wide_unpool_tile="64x32",
                tep8_publish_splitk_wide_tile_min_rows=65,
                tep8_pool_programmatic=False,
                tep8_join_pdl=False,
                join_cluster=False,
            ),
        ),
        (
            "kernels.projection_gathers.gathers",
            dict(tep8_wide_pool_gather=wide_gathers, tep8_wide_unpool_gather=wide_gathers),
            dict(tep8_wide_pool_gather=False, tep8_wide_unpool_gather=False),
        ),
        (
            "ops.comm.pool_publish_fp4quant",
            dict(norm_prefetch=norm_prefetch),
            dict(norm_prefetch=False),
        ),
        (
            "ops.comm.pool_gather_fp4quant",
            dict(norm_prefetch=norm_prefetch),
            dict(norm_prefetch=False),
        ),
        (
            "ops.runtime.step_commit",
            dict(v2_mode="1" if on("step_commit_v2") else ""),
            dict(v2_mode=""),
        ),
        (
            "ops.spec.head_sampler",
            dict(
                vp_min_rows=plan.param("vocab_parallel_head", "MIN_ROWS")
                if on("vocab_parallel_head")
                else 16,
                topp_rounds=plan.param("head_sampler_coop_fold", "TOPP_ROUNDS")
                if on("head_sampler_coop_fold")
                else 0,
                dist_min_rows=plan.param("head_sampler_coop_fold", "DIST_MIN_ROWS")
                if on("head_sampler_coop_fold")
                else 0,
                dist_fabric=bool(plan.param("head_sampler_coop_fold", "DIST_FABRIC"))
                if on("head_sampler_coop_fold")
                else False,
            ),
            dict(vp_min_rows=16, topp_rounds=0, dist_min_rows=0, dist_fabric=False),
        ),
        (
            "ops.spec.dflash_verify_fold",
            dict(
                grid=bool(plan.param("head_sampler_coop_fold", "VERIFY_GRID"))
                if on("head_sampler_coop_fold")
                else False
            ),
            dict(grid=False),
        ),
        # Context steps.
        (
            "kernels.context_sequence_parallel.forward",
            dict(sm107_extra_parts=_context_sp_parts(plan)),
            dict(sm107_extra_parts=()),
        ),
        (
            "kernels.context_sequence_parallel.overlap",
            _context_sp_overlap(plan),
            dict(CONTEXT_SP_OVERLAP_DEFAULTS),
        ),
    )
    for module, values, defaults in settings:
        _configure(module, values, defaults)
