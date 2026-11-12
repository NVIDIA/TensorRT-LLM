# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The plan's lever decisions, pushed into the kernel modules that read them at launch.

Several kernel modules of this target hold module-level settings their kernels read when they
build or launch (a row bound, a transport layout, whether a launch releases its dependents at
entry). Each module declares them with the default of the path it refines and one keyword-only
``configure``; ``apply`` calls every ``configure`` once per process, from the model constructor,
right after ``dispatch.compose`` and before any installer runs, with what the plan resolved
(``dispatch.Plan.enabled``, ``dispatch.Plan.param``). Under ``NO_PATHS`` it changes nothing, so a
deployment without a composition keeps every default.

A module whose settings all keep their defaults is not imported for it: some kernel modules load
a compiler, or change what a worker does, by being imported.

The levers that do not reach their site this way are read at their install instead, where the
plan is already in scope: ``tep8_router_shard`` and ``tep8_wide_direct_transports``
(``layers.adopt_model``), ``tep8_persistent_batch_experts``, ``tep8_experts_early_start`` and
``tep8_wide_direct_pool`` (``weights.install_loaded_features``), and ``drafter_fused_prep``
(``drafter``'s ``context_kv_fork`` install).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import dispatch as _dispatch


def _configure(module: str, settings: dict, defaults: dict) -> None:
    """Import ``module`` (relative to this target) and call its ``configure`` with ``settings``,
    unless every setting is its default: then the module is not imported for it, since some
    kernel modules load their compilers or change a worker's behaviour by being imported."""
    if settings == defaults:
        return
    importlib.import_module(f"{__package__}.{module}").configure(**settings)


def _router_staged_build(plan: "_dispatch.Plan") -> Optional[str]:
    """The staged router GEMM build ``router_staged_gemm__BUILD`` names, or None when the lever
    is off.

    Raises:
        ValueError: when ``BUILD`` names more than one build (the knob takes a subset of
            ``levers.ROUTER_STAGED_BUILDS``; exactly one build runs).
    """
    if not plan.enabled("router_staged_gemm"):
        return None
    build = tuple(plan.param("router_staged_gemm", "BUILD"))
    if len(build) != 1:
        from . import levers as _levers

        raise ValueError(
            f"{_levers.knob_name('router_staged_gemm', 'BUILD')}={','.join(build)}: name exactly "
            f"one of {list(_levers.ROUTER_STAGED_BUILDS)}"
        )
    return build[0]


def apply(plan: "_dispatch.Plan") -> None:
    """Configure the kernel modules from ``plan``, each with one ``configure`` call.

    A no-op for ``NO_PATHS``: every setting keeps its default. A module whose settings all keep
    their defaults is not imported.

    Raises:
        ValueError: for a lever parameter a module refuses, named with its knob.
    """
    if not plan:
        return
    on = plan.enabled
    # ``finalize_two_shot`` and ``wide_drafter_hops`` run at TP4 as well, but their parameters
    # here are TP8 row bounds the modules ignore at any other size. Pushing them anyway would
    # import those modules at construction for nothing, and both load the CuTe DSL by being
    # imported, so at TP4 they keep the module default and the module is left alone.
    tp = plan.levers.deployment.tp if plan.levers is not None else None
    tep8 = tp is None or tp == 8
    fc_wide = on("tep8_fc_shard_wide")
    two_shot = on("finalize_two_shot") and tep8
    wide_hops = on("wide_drafter_hops") and tep8
    transport_wide = on("tep8_finalize_transport_wide")
    settings = (
        # Dense GEMMs.
        (
            "kernels.dense_gemms.dispatcher",
            dict(
                batch_splitk=on("tep8_batch_splitk"),
                attn_block_splitk=on("tep8_attn_block_splitk"),
            ),
            dict(batch_splitk=False, attn_block_splitk=False),
        ),
        # Attention.
        (
            "ops.attention.swa_block",
            dict(single_early_loads=on("swa_block_single_early_loads")),
            dict(single_early_loads=False),
        ),
        (
            "ops.attention.multi_request_split_kv",
            dict(combine_gate=on("multi_request_combine_gate")),
            dict(combine_gate=False),
        ),
        (
            "kernels.replicated_experts.ring_buddy",
            dict(
                wide_balance_rows=(
                    plan.param("tep8_wide_balance", "MAX_ROWS") if on("tep8_wide_balance") else None
                ),
                tep8_max_rows=(
                    plan.param("tep8_ring_buddy_wide", "MAX_ROWS")
                    if on("tep8_ring_buddy_wide")
                    else None
                ),
            ),
            dict(wide_balance_rows=None, tep8_max_rows=None),
        ),
        (
            "ops.moe.persistent_experts",
            dict(interleaved_fetch=on("tep8_experts_interleaved_fetch")),
            dict(interleaved_fetch=False),
        ),
        # Routing.
        (
            "kernels.wide_prologue.prologue",
            dict(early=on("tep8_router_shard_early")),
            dict(early=False),
        ),
        (
            "ops.moe.router_gemm",
            dict(staged=_router_staged_build(plan)),
            dict(staged=None),
        ),
        # Communication and step glue.
        (
            "ops.moe.finalize_allreduce_transport",
            dict(
                tep8_max_tokens=(
                    plan.param("tep8_finalize_transport_wide", "MAX_TOKENS")
                    if transport_wide
                    else 16
                ),
                trigger="entry" if on("finalize_entry_trigger") else "late",
            ),
            dict(tep8_max_tokens=16, trigger="late"),
        ),
        (
            "kernels.finalize_two_shot.install",
            dict(
                tep8_max_rows=(
                    plan.param("finalize_two_shot", "TP8_MAX_ROWS") if two_shot else None
                )
            ),
            dict(tep8_max_rows=None),
        ),
        # Drafter.
        (
            "ops.gemm.fixed_low_m_gemm",
            dict(multi_request_cells=on("drafter_multi_request_cells")),
            dict(multi_request_cells=False),
        ),
        (
            "ops.allreduce.drafter_hop",
            dict(
                tep8_wide_rows=(
                    plan.param("wide_drafter_hops", "TP8_MAX_ROWS") if wide_hops else 64
                )
            ),
            dict(tep8_wide_rows=64),
        ),
        (
            "spec_worker",
            dict(
                fc_shard_tp8_max_rows=plan.param("tep8_fc_shard_wide", "MAX_ROWS") if fc_wide else 8
            ),
            dict(fc_shard_tp8_max_rows=8),
        ),
    )
    for module, values, defaults in settings:
        _configure(module, values, defaults)
