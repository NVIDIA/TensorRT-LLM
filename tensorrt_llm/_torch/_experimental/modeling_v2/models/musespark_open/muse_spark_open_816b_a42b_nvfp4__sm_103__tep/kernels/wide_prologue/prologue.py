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
  (``_wide_gather_inbox``).

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

TEP8 (sm103, two trays), default off: ``install_tep8``, which ``layers.adopt`` calls after
``wide_projection_gathers`` under ``tep8_router_shard`` (with ``tep8_router_shard_wide``),
gives the 16..32-row decode steps the same prologue on a TP8 wide route inbox
(``ProjectionInbox(row_words=row_words(8), max_rows=WIDE_MAX_ROWS)``, 2 MiB of fabric memory
per rank): ``_wide_prepare`` runs this rank's 32 router logit columns on the router fork's side
stream (``router_shard_rows``: each row's one-request bits, the ones its request gets in a
one-request step; with ``wide``, ``router_shard_wide``: the built-in 16..32-row tensor-core
router's bits, two launches and slower at EP8), the pool slice GEMM that
``wide_projection_gathers`` already defers at those rows (its split-K table), and
``pool_gather_route`` (routing only) in place of ``pool_gather_fp4quant``. That replaces the
served chain of those steps -- ``router_gemm``'s BF16x3 split-K GEMM and sum, the built-in
``_full_routing_kernel`` -- with one side-stream launch (two with ``router_shard_wide``'s
reduce) and the routing warps of the pool gather kernel; ``ring_buddy_assign`` then follows
the routed kernel instead of running beside the pool chain. The experts are reached as before:
``replicated_experts``' routing hook places the memo's selection with ``ring_buddy_assign``.
The pool site's unpool side stays ``wide_projection_gathers``'; ``router_gemm`` stays installed
for every other row count and for any declined call. Values: the routing and the quantization
are the built-in kernels' bits on the gathered logits and rows; the logits are not the served
BF16x3 router's, so top-8 near-ties can flip (N-class).
"""

from __future__ import annotations

from typing import Optional

import torch

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
    route_inbox = ProjectionInbox(
        mapping, row_words=_route.row_words(WORLD), max_rows=WIDE_MAX_ROWS
    )
    unpool_inbox = ProjectionInbox(
        mapping, row_words=_unpool.local_columns(WORLD) // 2, max_rows=WIDE_MAX_ROWS
    )
    router_rows, direct = bool(router_rows), bool(direct)
    for layer in layers:
        moe = layer.feed_forward
        moe._wide_inbox = route_inbox
        moe._defer_max_rows = WIDE_MAX_ROWS
        moe._wide_router_rows = router_rows
        # NOT ``_wide_direct``: that attribute is the wide direct transport OBJECT, set by
        # kernels/wide_direct_transports/transports.install and read in projection_gathers as
        # ``.is_armed`` / ``.project_pool`` / ``.consume_pool``. layers.py installs the
        # transports first and this prologue second, so assigning the bool here overwrote the
        # object and every 8-row decode step raised
        #   AttributeError("'bool' object has no attribute 'is_armed'")
        # at warmup. Nothing reads this flag; it is kept under its own name.
        moe._wide_direct_enabled = direct
        layer.add_ffn._wide_gather_inbox = unpool_inbox
    router = (
        "router_shard_rows, each row's one-request bits"
        if router_rows
        else "router_shard_wide, the built-in bits"
    )
    logger.info(
        f"[musespark_open] wide_prologue: TP{WORLD} sharded MoE prologue (router shard of "
        f"{_shard.local_experts(WORLD)} experts ({router}), routing in the pool kernel) and fused "
        f"unpool gather on decode steps of {_shard.WIDE_MIN_ROWS}..{_shard.WIDE_MAX_ROWS} rows "
        f"on all {len(layers)} layers; wide direct transports "
        f"{'on' if direct else 'off'} (wide_direct_transports)"
    )
    return len(layers)


#: The modes of the TEP8 install (``install_tep8``): ``rows`` serves the router shard with
#: ``router_shard_rows``, ``wide`` with ``router_shard_wide``; None keeps the served
#: 16..32-row prologue. The levers are ``tep8_router_shard`` and ``tep8_router_shard_wide``.
TEP8_ROUTER_SHARD_MODES = ("rows", "wide")
#: The TP size and device capability ``install_tep8`` serves.
TEP8_WORLD = 8
TEP8_CAPABILITY = (10, 3)
#: With the ``tep8_router_shard`` lever in ``rows`` mode, default off: ``1`` makes the router
#: fork's ``router_shard_rows`` launch programmatic (``router_shard_rows(enable_pdl=True)``: the
#: weight tile streams while the attention hop finishes, the rows and the stores follow the hop;
#: ``TEP8_ROUTER_SHARD_EARLY_TRIGGER`` also releases the routed pool kernel at the shard's entry)
#: and launches it with ``TEP8_ROUTER_SHARD_EARLY_TILING``. Same bits. Armed by the
#: ``tep8_router_shard_early`` lever, read once in ``install_tep8``.
#: ``router_shard_rows``' ``rows:per:experts`` tiling under that lever: two rows
#: and two experts per program (128 / 192 / 256 programs at 16 / 24 / 32 rows, 64 registers, one
#: wave beside the pool GEMM). Chosen on GB300 in the served 62-layer TEP8 chain (8 ranks, two tray
#: pairs), paired against the plain launch: -10..-15 / -48..-62 / -104..-113 us per step at 16 / 24 /
#: 32 rows, the fastest at 32 rows and within the spread of the fastest at 16 and 24; one expert per
#: program saves less at 32 rows (-53 us), four save less (-59 / -47 us), and the same tilings as
#: plain launches lose or tie (+9 / -68 / +39 us for this one).
TEP8_ROUTER_SHARD_EARLY_TILING = "16:2:2,24:2:2,32:2:2"
#: Whether the early router shard releases its dependents at entry (``"entry"``) or at exit
#: (``"wait"``: only its weight tile is early). ``"wait"`` measured slower than the plain launch
#: (+43..+56 us per step): the routed pool kernel then launches only once the shard has exited.
TEP8_ROUTER_SHARD_EARLY_TRIGGER = "entry"


#: ``tep8_router_shard_early``'s decision, set by ``lever_settings.apply`` before any install.
#: ``False`` until then, so a tree nobody configured keeps the plain launch.
_EARLY: bool = False


def configure(early: bool = False) -> None:
    """Set whether the router shard launches early. ``lever_settings`` owns this."""
    global _EARLY
    _EARLY = bool(early)


def tep8_router_shard_early() -> bool:
    """Whether the early router shard is armed, from the lever rather than the environment."""
    return _EARLY


def tep8_router_shard_mode(wide: bool = False) -> str:
    """The install mode the ``tep8_router_shard`` levers select: ``"wide"`` or ``"rows"``."""
    return "wide" if wide else "rows"


def tep8_refusal(model: MuseSparkOpenModel) -> Optional[str]:
    """Why ``install_tep8`` cannot run on this rank, or None. Reads construction-time state only."""
    layers = list(model.layers)
    if not layers:
        return "the model has no layers"
    world = layers[0].mapping.tp_size
    if world != TEP8_WORLD:
        return f"the TEP8 router shard is laid out for TP{TEP8_WORLD}, got TP{world}"
    if not torch.cuda.is_available():
        return "no CUDA device"
    capability = tuple(torch.cuda.get_device_capability())
    if capability != TEP8_CAPABILITY:
        return f"the TEP8 router shard is validated on sm103 only, got {capability}"
    for layer in layers:
        moe = layer.feed_forward
        if not isinstance(moe, MuseSparkOpenTepPrologueMoE) or moe._prologue_inbox is None:
            return (
                f"the TEP8 router shard extends the one-chain MoE prologue; layer "
                f"{layer.layer_idx} has {type(moe).__name__} without its inbox"
            )
        if moe._router_stream is None:
            return (
                f"the TEP8 router shard runs on the router fork; layer {layer.layer_idx} has none"
            )
        if getattr(moe, "_wide", None) is None or moe._wide_pool_inbox is None:
            return (
                "the TEP8 router shard takes the pool slices wide_projection_gathers defers; layer "
                f"{layer.layer_idx} has no wide pool site"
            )
        if moe._wide_inbox is not None:
            return f"layer {layer.layer_idx} already has a wide route inbox"
    if not _route.available():
        return "the TEP8 router shard needs CuTe DSL and flashinfer's RMSNormFP4QuantKernel"
    return None


def install_tep8(model: MuseSparkOpenModel, mode: str) -> int:
    """The TEP8 router shard (``tep8_router_shard``): give every prologue MoE
    the TP8 wide route inbox, so ``_wide_prepare`` serves its 16..32-row decode steps.

    Called after ``moe_prologue.install``, ``install_router_fork`` and
    ``wide_projection_gathers.install``. The refusal is agreed over the TP group before the
    inbox (fabric memory, collective) is built; a refused install logs a warning on every
    rank and changes nothing. Returns the number of layers given the inbox (0 when refused).
    """
    if mode not in ("wide", "rows"):
        raise ValueError(f"[musespark_open] install_tep8: mode {mode!r} is not rows or wide")
    layers = list(model.layers)
    mapping = layers[0].mapping if layers else None
    reason = _tp_agreement.agreed_refusal(mapping, tep8_refusal(model))
    if reason is not None:
        logger.warning(
            "[musespark_open] tep8_router_shard is on, but the TEP8 router shard is not "
            f"installed: {reason}; 16..32-row decode steps keep the served router"
        )
        return 0
    early = mode == "rows" and tep8_router_shard_early()
    if early:
        _shard.configure(rows_tiling=TEP8_ROUTER_SHARD_EARLY_TILING)
    route_inbox = ProjectionInbox(
        mapping, row_words=_route.row_words(TEP8_WORLD), max_rows=WIDE_MAX_ROWS
    )
    for layer in layers:
        moe = layer.feed_forward
        moe._wide_inbox = route_inbox
        moe._wide_router_rows = mode == "rows"
        moe._wide_router_early = TEP8_ROUTER_SHARD_EARLY_TRIGGER if early else None
    router = (
        "router_shard_rows: each row's one-request bits"
        if mode == "rows"
        else "router_shard_wide: the built-in tensor-core router's bits"
    )
    logger.info(
        f"[musespark_open] tep8_router_shard ({mode}): TP{TEP8_WORLD} "
        f"decode steps of {_shard.WIDE_MIN_ROWS}..{_shard.WIDE_MAX_ROWS} rows on {len(layers)} "
        f"layers run this rank's {_shard.local_experts(TEP8_WORLD)} router logit columns "
        f"({router}) on the router fork and the logit exchange + routing in pool_gather_route "
        f"on a {WIDE_MAX_ROWS}-row TP{TEP8_WORLD} inbox, in place of router_gemm and the built-in "
        "routing; the BF16-expert layers and every other row count keep the served path"
    )
    if early:
        logger.info(
            "[musespark_open] tep8 router shard: tep8_router_shard_early: router_shard_rows "
            "on the router fork is a programmatic launch (weight tile before its dependency wait, "
            f"dependents released at {'entry' if TEP8_ROUTER_SHARD_EARLY_TRIGGER == 'entry' else 'exit'}"
            f"), tiling rows:per:experts {TEP8_ROUTER_SHARD_EARLY_TILING}; same bits"
        )
    elif tep8_router_shard_early():
        logger.warning(
            "[musespark_open] tep8_router_shard_early needs the tep8_router_shard lever "
            f"in rows mode (router_shard_rows); with {mode} it changes nothing"
        )
    return len(layers)
