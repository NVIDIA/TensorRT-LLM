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

Two settings follow the architecture rather than a lever: on sm107 the split-K kernels always
build their conflict-free mailbox, their reduce-scatter epilogue for split-K 8 and their deeper
stages in oversized shared memory (each writes the same bits as the build it replaces), and a
sharded drafter fc creates its BLAS handles at install.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple

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
#: ``tep8_fc_shard_wide``: the programs of the TP8 drafter fc's gather.
TEP8_FC_GATHER_PROGRAMS = 128
#: ``split_kv_deep_rings``: the TMA ring depth of the split-KV decode FMHA kernels.
SPLIT_KV_RING_STAGES = 20
#: ``wide_norm_gate_pdl``: the most rows a PDL post-FFN norm-gate takes.
NORM_GATE_PDL_MAX_ROWS = 128
#: ``tep8_router_push``: the fewest rows the router push takes.
ROUTER_PUSH_MIN_ROWS = 33
#: ``tep8_tile_placement``: the fewest rows placed by tile cost.
TILE_PLACEMENT_MIN_ROWS = 41
#: ``tep8_ring_buddy_wide``: the most rows the TP8 ring-buddy placement takes.
RING_BUDDY_MAX_ROWS = 64
#: ``tep8_early_multicast_hop``: the row band whose CTAs pad their shared memory, the rows
#: from which a CTA joins two rows, and the pad.
HOP_PAD_ROWS = (9, 40)
HOP_TWO_ROWS_MIN = 41
HOP_SMEM_PAD = 177408
#: On sm107 the per-row router shard of 16..32-row steps tiles its rows as
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
    """sm103's parts the sm107 sequence-parallel context step adds: ``context_sp_extra_parts``'s
    PARTS, and the router input under ``context_sp_router_input`` (the aggregated TEP4
    selection)."""
    parts = (
        tuple(plan.param("context_sp_extra_parts", "PARTS"))
        if plan.enabled("context_sp_extra_parts")
        else ()
    )
    if plan.enabled("context_sp_router_input") and "router_input" not in parts:
        parts += ("router_input",)
    return parts


def apply(plan: "_dispatch.Plan", sm: Optional[int]) -> None:
    """Configure the kernel modules from ``plan``, each with one ``configure`` call.

    A no-op for ``NO_PATHS`` and off sm107: none of these levers is served there, so every
    setting keeps its default. A module whose settings all keep their defaults is not imported.
    """
    if not plan or sm != 107:
        return
    on = plan.enabled
    tables = on("splitk_frontier_tables")
    after_join = on("weight_prefetch_after_join")
    wide64 = on("tep8_wide64_splitk")
    fc_wide = on("tep8_fc_shard_wide")
    one_cluster = on("swa_one_cluster_block")
    deep = on("split_kv_deep_rings")
    deep_multi = on("split_kv_deep_ring")
    prerouted = on("prerouted_experts")
    push = on("tep8_router_push")
    early = on("tep8_early_multicast_hop")
    join = on("tep8_publish_join")
    wide_gathers = on("tep8_wide_projection_gathers")
    settings = (
        # Dense GEMMs: the split-K constants of sm107, then the levers.
        (
            "ops.gemm.splitk_gemm",
            dict(mailbox_v4=True, oversized_smem=True, rs_split_k=(8,), wide64=wide64),
            dict(mailbox_v4=False, oversized_smem=False, rs_split_k=(), wide64=False),
        ),
        (
            "kernels.dense_gemms.dispatcher",
            dict(
                pool_splitk8=on("splitk_pool_split8"),
                tep4_wide_gemms=_tables(TEP4_WIDE_GEMMS) if tables else None,
                tep4_narrow_gemms=_tables(TEP4_NARROW_GEMMS) if tables else None,
                tep8_wide64=wide64,
                tep8_wide64_kernels=dict(TEP8_WIDE64_KERNELS) if wide64 else None,
                prefetch_join="after" if after_join else "before",
                drop_weight_prefetch=("qkv",) if after_join else (),
            ),
            dict(
                pool_splitk8=False,
                tep4_wide_gemms=None,
                tep4_narrow_gemms=None,
                tep8_wide64=False,
                tep8_wide64_kernels=None,
                prefetch_join="before",
                drop_weight_prefetch=(),
            ),
        ),
        (
            "ops.gemm.fixed_low_m_gemm",
            dict(
                tep8_wide64=wide64,
                tep8_drafter_wide=on("tep8_drafter_wide_gemms"),
                tep8_fc_wide=fc_wide,
            ),
            dict(tep8_wide64=False, tep8_drafter_wide=False, tep8_fc_wide=False),
        ),
        (
            "kernels.next_qkv_prefetch.prefetch",
            dict(skip_rows=8 if on("next_qkv_prefetch_skip") else 0),
            dict(skip_rows=0),
        ),
        (
            "spec_worker",
            dict(
                fc_shard_wide=fc_wide,
                fc_gather_programs=TEP8_FC_GATHER_PROGRAMS if fc_wide else 0,
                # The sharded fc's BLAS handles are created at install (an sm107 constant).
                fc_shard_prewarm=on("drafter_fc_shard"),
            ),
            dict(fc_shard_wide=False, fc_gather_programs=0, fc_shard_prewarm=False),
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
                prep_split=on("swa_prep_split"),
                one2=on("swa_eight_requests"),
            ),
            dict(cl8_tep4="0", cl8_tep8=False, prep_split=False, one2=False),
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
                bf16_half_window=on("bf16_half_window"),
                tep8_max_rows=RING_BUDDY_MAX_ROWS if on("tep8_ring_buddy_wide") else None,
            ),
            dict(
                tep4_prerouted=False,
                tep8_prerouted=False,
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
                tile_placement=on("tep8_tile_placement"),
                tile_min_rows=TILE_PLACEMENT_MIN_ROWS if on("tep8_tile_placement") else 33,
            ),
            dict(rb_pdl=False, tile_placement=False, tile_min_rows=33),
        ),
        (
            "ops.moe.router_push",
            dict(
                tep8_push=push, tep8_min_rows=ROUTER_PUSH_MIN_ROWS if push else 9, tep8_route=push
            ),
            dict(tep8_push=False, tep8_min_rows=9, tep8_route=False),
        ),
        (
            "ops.moe.bf16x3_router_gemm",
            dict(prewait=on("tep8_router_prewait")),
            dict(prewait=False),
        ),
        (
            "ops.moe.router_shard",
            # The per-row router shard's sm107 tiling, wherever that shard runs.
            dict(rows_tiling=SM107_ROUTER_ROWS_TILING if on("wide_router_per_row") else None),
            dict(rows_tiling=None),
        ),
        # Communication and step glue.
        (
            "ops.moe.finalize_allreduce_transport",
            dict(
                tep8_max_tokens=64 if on("tep8_finalize_transport_64") else 16,
                trigger="entry" if on("finalize_entry_trigger") else "late",
            ),
            dict(tep8_max_tokens=16, trigger="late"),
        ),
        (
            "ops.allreduce.post_attn_hop",
            dict(entry_trigger=on("hop_entry_trigger")),
            dict(entry_trigger=False),
        ),
        (
            "ops.allreduce.post_attn_hop_wide",
            dict(
                direct=on("tep8_direct_hop"),
                early_launch=early,
                multicast=early,
                pad_rows=HOP_PAD_ROWS if early else (0, 1024),
                r2="w" if early else "0",
                r2_min_rows=HOP_TWO_ROWS_MIN if early else 33,
                smem_pad=HOP_SMEM_PAD if early else 0,
            ),
            dict(
                direct=False,
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
                tep8_publish_splitk=on("tep8_publish_splitk"),
                tep8_pool_programmatic=on("tep8_pool_programmatic_fork"),
                tep8_join_pdl=join,
                join_cluster=join,
            ),
            dict(
                tep8_publish_splitk=False,
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
            "ops.runtime.step_commit",
            dict(v2_mode="1" if on("step_commit_v2") else ""),
            dict(v2_mode=""),
        ),
        (
            "ops.spec.head_sampler",
            dict(
                vp_min_rows=plan.param("vocab_parallel_head", "MIN_ROWS")
                if on("vocab_parallel_head")
                else 16
            ),
            dict(vp_min_rows=16),
        ),
        # Context steps.
        (
            "kernels.context_sequence_parallel.forward",
            dict(sm107_extra_parts=_context_sp_parts(plan)),
            dict(sm107_extra_parts=()),
        ),
    )
    for module, values, defaults in settings:
        _configure(module, values, defaults)
