# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``wide_prologue``: the TP4 sharded MoE prologue on decode steps of two to four requests.

At TP4 a decode step of one request (8 rows at K=7) runs the sharded MoE: the
pool and unpool projections as per-rank column slices whose gathers are fused into
their consumers (``projection_gathers``), and the MoE prologue as one chain
(``moe_prologue``: router shard, pool slice GEMM, ``pool_gather_route``). Without this
fast path a step of two to four requests (16..32 rows) runs the built-in replicated path:
every rank reads the whole 8 MiB router weight with the tensor-core router kernel and then
routes, and multiplies the whole 64 MiB pool and unpool weights, because the
built-in sharded projections, their fused gathers and the projection inboxes all
stop at 8 rows (``ops/comm/projection_inbox.MAX_ROWS``).

This fast path gives those steps the same structure on wide inboxes
(``ProjectionInbox(max_rows=WIDE_MAX_ROWS)``, separate allocations whose slot layout
holds 32 rows; the 8-row inboxes and kernels are unchanged):

* the prologue MoE (``moe_prologue.MuseSparkOpenTepPrologueMoE._wide_prepare``) runs
  ``ops/moe/router_shard.router_shard_wide`` (this rank's 64 logit columns, bit for
  bit the built-in tensor-core router's) on the router fork's side stream, the pool
  slice GEMM, and ``pool_gather_route`` on the wide route inbox (routing only: above
  8 rows the experts are the built-in ones and read the routing memo);
* the projection-gather MoE hands its pool and unpool slices on ungathered up to
  ``WIDE_MAX_ROWS`` rows (``_defer_max_rows``), and the post-FFN join takes 9..32-row
  calls through ``ops/comm/unpool_gather_boundary`` on the wide unpool inbox
  (``_gather_inbox_wide``).

Values: the router logits, the routing and the boundary are the built-in path's bit
for bit given the same projections; the pool and unpool slices are each rank's
columns of the built-in projection by a GEMM of the slice's shape, so they may
differ from the replicated GEMM's in FP32 summation order (each output one BF16
rounding of its full-K FP32 dot product either way).

Two levers, given to ``install`` by the plan (the same on every rank):

* ``wide_router_per_row``: the router shard of these steps is
  ``router_shard_rows``, which gives each row its one-request logits (those the built-in
  router gives the row when its request decodes alone) in one launch without the split-K
  reduce. Those are other bits than the built-in 16..32-row launch's, so the routing of
  a wide step may differ from the built-in path's.
* ``wide_direct_transports``: ``pool_into_route`` and ``unpool_into_boundary`` also build their
  wide direct transports (the pool and unpool GEMMs and the router shard publish into
  the routed kernel's and the boundary's transports on 16..32-row CUDA-graph steps, and on
  a half-window layer the routed kernel places). Same bits as the tagged wide path given
  the same router logits.

Every rank takes the same path at every call: the choice reads shapes, dtypes,
contiguity and construction-time state only.
"""

from __future__ import annotations

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import pool_gather_route as _route
from ...ops.comm import unpool_gather_boundary as _unpool
from ...ops.comm.projection_inbox import WIDE_MAX_ROWS, ProjectionInbox
from ...ops.moe import router_shard as _shard
from ..moe_prologue.prologue import MuseSparkOpenTepPrologueMoE
from ..projection_gathers.gathers import MuseSparkOpenTepGatedResidual

#: The TP sizes the wide inboxes are laid out for: one tray of four, or (wide pool inbox, pool side only) two trays.
WORLDS_SERVED = (4, 8)
#: The TP4 value, for the log line of a TP4 install.
WORLD = 4


def refusal(model: MuseSparkOpenModel):
    """Why ``install`` cannot run on this rank, or None. Reads construction-time state only."""
    layers = list(model.layers)
    if not layers:
        return "wide_prologue: the model has no layers"
    world = layers[0].mapping.tp_size
    if world not in WORLDS_SERVED:
        return f"wide_prologue is laid out for TP4 or TP8, got TP{world}"
    for layer in layers:
        moe = layer.feed_forward
        if type(moe) is not MuseSparkOpenTepPrologueMoE or moe._prologue_inbox is None:
            return (
                f"wide_prologue extends the one-chain MoE prologue; layer {layer.layer_idx} has "
                f"{type(moe).__name__} without its inbox"
            )
        join = layer.add_ffn
        if not isinstance(join, MuseSparkOpenTepGatedResidual) or join._gather_inbox is None:
            return (
                f"wide_prologue extends the fused unpool gather; layer {layer.layer_idx}'s post-FFN "
                "join has no unpool inbox"
            )
    if not (_route.available() and _unpool.available()):
        return "wide_prologue needs CuTe DSL, flashinfer's RMSNormFP4QuantKernel and the cluster boundary"
    return None


def install(model: MuseSparkOpenModel, router_rows: bool = False, direct: bool = False) -> int:
    """Build the wide route and unpool inboxes and hand them to every layer.

    Called after ``moe_prologue.install`` and ``install_router_fork``. The refusal is
    agreed over the TP group before the inboxes (fabric memory, collective) are built.
    ``router_rows`` (``wide_router_per_row``) and ``direct`` (``wide_direct_transports``)
    are the plan's levers. Returns the number of layers installed.

    Raises:
        ValueError: on every rank alike, naming the refusal.
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    reason = _tp_agreement.agreed_refusal(mapping, refusal(model))
    if reason is not None:
        raise ValueError(f"[musespark_open] {reason}")
    world = mapping.tp_size
    route_inbox = ProjectionInbox(
        mapping, row_words=_route.row_words(world), max_rows=WIDE_MAX_ROWS
    )
    # TP8 (wide pool inbox): the pool side only. The unpool of 9..32-row calls stays replicated with the built-in
    # boundary, so only the pool's slices are handed on ungathered up to WIDE_MAX_ROWS rows.
    unpool_inbox = (
        ProjectionInbox(
            mapping, row_words=_unpool.local_columns(world) // 2, max_rows=WIDE_MAX_ROWS
        )
        if world == WORLD
        else None
    )
    router_rows, direct = bool(router_rows), bool(direct)
    for layer in layers:
        moe = layer.feed_forward
        moe._wide_inbox = route_inbox
        if unpool_inbox is not None:
            moe._defer_max_rows = WIDE_MAX_ROWS
            layer.add_ffn._gather_inbox_wide = unpool_inbox
        else:
            moe._defer_max_rows_pool = WIDE_MAX_ROWS
        moe._wide_router_rows = router_rows
        moe._wide_direct = direct
    router = (
        "router_shard_rows, each row's one-request bits"
        if router_rows
        else "router_shard_wide, the built-in bits"
    )
    logger.info(
        f"[musespark_open] wide_prologue: TP{world} sharded MoE prologue (router shard of "
        f"{_shard.local_experts(world)} experts ({router}), routing in the pool kernel) and "
        f"{'fused' if unpool_inbox is not None else 'built-in replicated'} unpool "
        f"gather on decode steps of {_shard.WIDE_MIN_ROWS}..{_shard.WIDE_MAX_ROWS} rows "
        f"on all {len(layers)} layers; wide direct transports "
        f"{'on' if direct else 'off'} (wide_direct_transports)"
    )
    return len(layers)
