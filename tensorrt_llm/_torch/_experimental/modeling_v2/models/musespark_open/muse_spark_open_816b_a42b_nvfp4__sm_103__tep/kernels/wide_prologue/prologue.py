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

#: The TP size the wide inboxes are laid out for: one tray of four.
WORLD = 4


def refusal(model: MuseSparkOpenModel):
    """Why ``install`` cannot run on this rank, or None. Reads construction-time state only."""
    layers = list(model.layers)
    if not layers:
        return "wide_prologue: the model has no layers"
    world = layers[0].mapping.tp_size
    if world != WORLD:
        return f"wide_prologue is laid out for TP{WORLD}, got TP{world}"
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


def install(model: MuseSparkOpenModel) -> int:
    """Build the wide route and unpool inboxes and hand them to every layer.

    Called after ``moe_prologue.install`` and ``install_router_fork``. The refusal is
    agreed over the TP group before the inboxes (fabric memory, collective) are built.
    Returns the number of layers installed.

    Raises:
        ValueError: on every rank alike, naming the refusal.
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    reason = _tp_agreement.agreed_refusal(mapping, refusal(model))
    if reason is not None:
        raise ValueError(f"[musespark_open] {reason}")
    route_inbox = ProjectionInbox(
        mapping, row_words=_route.row_words(WORLD), max_rows=WIDE_MAX_ROWS
    )
    unpool_inbox = ProjectionInbox(
        mapping, row_words=_unpool.local_columns(WORLD) // 2, max_rows=WIDE_MAX_ROWS
    )
    for layer in layers:
        moe = layer.feed_forward
        moe._wide_inbox = route_inbox
        moe._defer_max_rows = WIDE_MAX_ROWS
        layer.add_ffn._gather_inbox_wide = unpool_inbox
    logger.info(
        f"[musespark_open] wide_prologue: TP{WORLD} sharded MoE prologue (router shard of "
        f"{_shard.local_experts(WORLD)} experts, routing in the pool kernel) and fused unpool "
        f"gather on decode steps of {_shard.WIDE_MIN_ROWS}..{_shard.WIDE_MAX_ROWS} rows "
        f"on all {len(layers)} layers"
    )
    return len(layers)
