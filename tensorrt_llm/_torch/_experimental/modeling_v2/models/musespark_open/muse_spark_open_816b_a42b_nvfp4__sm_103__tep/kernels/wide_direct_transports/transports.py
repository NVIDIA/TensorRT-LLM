# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``wide_direct_transports``: on sm103 TEP8 decode steps of two to four requests (16, 24 and 32
rows at K = 7), the pool and unpool slice GEMMs publish straight into their consumers' transports.

Off unless ``tep8_wide_direct_transports`` is on (its ``SITES``: both sites, or a
list of ``pool`` / ``unpool``); unset, nothing here is imported and every step runs as before.

At 9..32 rows ``wide_projection_gathers`` computes each rank's pool (512 columns) and unpool (1024
columns) slices with flashinfer's split-K into fresh tensors; their consumers -- the pre-expert
RMSNorm + NVFP4 quantization (``pool_gather_fp4quant``) and the post-FFN boundary
(``unpool_gather_boundary``) -- wait on the GEMM, reload the slice, publish it to the seven peers
as tagged 64-bit words through a 32-row inbox, and then poll the peers' words. Here, on the steps of
``ROWS`` rows, the same split-K cell's output tensor is this rank's slot of a multicast transport,
so its epilogue stores are the publication (the 8-row ``unpool_into_boundary`` /
``pool_into_route`` mechanism, and ``projection_publish``'s at 33..512 rows, on the 9..32-row
band neither covers):

* unpool: a 32-row ``UnpoolDirectWorkspace`` (``unpool_direct_boundary``'s wide layout, which
  b5ca973 / #21377 landed in this tree TP4 only; ``check_layout`` here admits TP8 as well),
  consumed by ``UnpoolDirectBoundary`` -- the gather boundary's
  geometry and arithmetic at 9..32 rows (a cluster of four CTAs, two peers' slices per CTA at TP8),
  polling its own copy -- through ``unpool_into_boundary``'s re-classed join, early-release trigger
  and DFlash capture fold included. Its sites and buffers are ``unpool_into_boundary``'s.
* pool: a 32-row ``projection_publish.PublishTransport`` of the 512-column slices, consumed by
  ``pool_publish_fp4quant`` (flashinfer's RMSNorm + NVFP4 quantization reading its row from the
  transport) on the MoE preparation stream, launched with programmatic dependent launch as the
  gather consumer it replaces. The publishing layers are those whose pre-expert input is the fused
  NVFP4 quantization (``projection_publish``'s choice), fixed at the first eager 16/24/32-row forward
  (the engine's CUDA-graph warm-up), once the weights are final, and agreed over the TP group. The
  pool site yields to ``tep8_router_shard`` (a TP8 ``_wide_inbox`` on the prologue
  MoE): that path's ``_wide_prepare`` owns the 16..32-row pool prepare and feeds the pool slice to
  ``pool_gather_route``, which must not be handed a write-only multicast slot. The unpool site
  composes with it.

Both sites arm only on forwards captured into a CUDA graph (as the 8-row ``unpool_into_boundary`` /
``pool_into_route`` and upstream's 16..32-row lever do); eager forwards keep the 32-row inboxes, and
the first eager one at a ``ROWS`` row count only chooses the pool layers.

Producers: the split-K cells of ``wide_projection_gathers.TP8_WIDE_KERNELS`` (flashinfer's: the
owner CTA of each cluster stores each output element once, from its epilogue, after its dependency
wait). The sites' producer policies are ``unpool_into_boundary``'s and ``pool_into_route``'s (same
families). Slot tails: a 24-row call runs the ``mma_n`` 32 cell into the first 24 rows of a 32-row
slot; its stores past row 24 would be read as arrived by a later 32-row call on that buffer, so the
tail is predicated by the kernel (``exp/mb-tep8-wide-direct-*``: every half past the call's rows
still holds the sentinel after the call).

Numerics: bitwise. The producer is the served cell with another output pointer, and both consumers
are certified bit for bit against the gather consumers they replace (same thread-to-column maps and
arithmetic, statement for statement).

Every rank takes the same path at every call: arming reads the step's row count only, the layer
choice is agreed over the TP group, and an armed call that does not fit raises on every rank alike
(its GEMM already published).
"""

from __future__ import annotations

import contextlib
from typing import FrozenSet, Iterator, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import pool_publish_fp4quant as _pool
from ...ops.comm import projection_publish as _publish
from ...ops.comm import unpool_direct_boundary as _direct
from ...ops.comm.projection_inbox import WIDE_MAX_ROWS
from ...ops.gemm.dense_gemm import parse_kernel
from ..dense_gemms.dispatcher import FLASHINFER_FAMILIES, WIDE_MIN_ROWS, claim_publish

#: The sites the path can publish; ``tep8_wide_direct_transports__SITES`` names a subset.
SITES = ("pool", "unpool")
#: The decode row counts the path arms: two, three and four requests at K = 7 (the 9..32-row
#: steps of the TEP8 generation workers' CUDA graphs); other counts keep the 32-row inboxes.
ROWS = (16, 24, 32)
#: The TP size and device the path serves: sm103 TEP8 over two trays.
TP_SIZE = 8
CAPABILITY = (10, 3)


def requested_sites(sites: Optional[Sequence[str]] = None) -> FrozenSet[str]:
    """The sites ``sites`` names, as a set; every site when it is None.

    Raises:
        ValueError: for a name outside ``SITES``.
    """
    if sites is None:
        return frozenset(SITES)
    names = frozenset(str(part).strip().lower() for part in sites if str(part).strip())
    unknown = sorted(names - set(SITES))
    if unknown:
        raise ValueError(
            f"[musespark_open] wide_direct_transports: {unknown}: expected a subset of "
            f"{', '.join(SITES)}"
        )
    return names


def _cells(plan, site: str) -> List[str]:
    """The kernel the dense GEMM plan's armed 9..32-row table runs at each of ``ROWS`` for ``site``."""
    table = getattr(plan, "wide_kernels", {}).get(site)
    if table is None or len(table) != WIDE_MAX_ROWS - WIDE_MIN_ROWS + 1:
        return []
    return [table[rows - WIDE_MIN_ROWS] for rows in ROWS]


def refusal(
    model, plan, wide_projections, unpool_direct, sites: FrozenSet[str], capability
) -> Optional[str]:
    """Why the path cannot run here, or None. Reads construction-time state only (rank-agnostic)."""
    layers = list(model.layers)
    mapping = layers[0].mapping
    if int(mapping.tp_size) != TP_SIZE:
        return f"TP{mapping.tp_size}: the path serves TP{TP_SIZE} (sm103 TEP8) only"
    if capability is not None and tuple(capability) != CAPABILITY:
        return f"SM {tuple(capability)}: the path serves sm103 only"
    if plan is None or wide_projections is None:
        return "wide_projection_gathers (and its dense GEMM plan) is not installed"
    for site in sorted(sites):
        cells = _cells(plan, site)
        if len(cells) != len(ROWS):
            return f"the dense GEMM plan has no 9..32-row {site} table"
        for rows, cell in zip(ROWS, cells):
            if parse_kernel(cell)[0] not in FLASHINFER_FAMILIES:
                return (
                    f"the {rows}-row {site} cell {cell} is not a flashinfer kernel, which alone stores "
                    "into a transport slot"
                )
    if "unpool" in sites and (
        unpool_direct is None or len(getattr(unpool_direct, "sites", ())) < 2
    ):
        return "unpool_into_boundary (whose joins, sites and buffers the unpool site rides) is not installed"
    return None


def pool_owner_conflict(layers) -> Optional[str]:
    """Why another fast path owns the 16..32-row pool prepare of ``layers``, or None.

    ``tep8_router_shard`` (``wide_prologue.install_tep8``) gives the prologue MoE a
    TP8 ``_wide_inbox``: its ``_wide_prepare`` then runs the pool slice GEMM itself and hands the
    product to ``pool_gather_route`` as its local slice, so that GEMM must not publish into this
    path's write-only multicast slot. Reads construction-time state only.
    """
    for layer in layers:
        moe = layer.feed_forward
        if getattr(moe, "_wide_inbox", None) is not None:
            return (
                f"layer {getattr(layer, 'layer_idx', '?')}'s 16..32-row pool prepare belongs to the TEP8 "
                "router shard (tep8_router_shard, a wide route inbox), which routes the "
                "pool slice itself"
            )
    return None


class WideDirectTransports:
    """What ``DecodeTarget`` holds under ``tep8_wide_direct_transports``.

    ``unpool`` is the 32-row direct unpool workspace and ``unpool_sites`` its ``(dispatcher, join,
    buffer)`` per fused layer (``unpool_into_boundary``'s); ``pool`` the 32-row pool transport and,
    once built, ``pool_layers`` the publishing layers' indices (None: not built yet).
    """

    def __init__(
        self,
        model,
        plan,
        wide_projections,
        sites: FrozenSet[str],
        unpool: Optional[_direct.UnpoolDirectWorkspace],
        unpool_sites: Sequence[Tuple[object, object, int]],
        pool: Optional[_publish.PublishTransport],
    ) -> None:
        self._layers = list(model.layers)
        self.mapping = self._layers[0].mapping
        self.plan = plan
        self.wide_projections = wide_projections
        self.sites = frozenset(sites)
        self.unpool = unpool
        self.unpool_sites = tuple(unpool_sites)
        self.pool = pool
        self.pool_layers: Optional[Tuple[int, ...]] = None
        self._pool_refused: Optional[str] = None
        self.enable_pdl = bool(getattr(plan, "enable_pdl", True))
        self._armed = False
        self._rows = 0
        self._unconsumed = 0
        self._reported = set()
        for layer in self._layers:
            layer.feed_forward._wide_direct = self

    @property
    def is_armed(self) -> bool:
        return self._armed

    def _pool_ready(self) -> bool:
        return self.pool is not None and self.pool_layers is not None and self._pool_refused is None

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows publishes on this path.

        Only a forward of ``ROWS`` rows captured into a CUDA graph. An eager one (the warm-up, or a
        step without a graph) is never armed; the first eager one at a ``ROWS`` row count chooses
        the pool layers (``_build_pool``, collective: ``DecodeTarget`` asks at the same point of the
        same forward on every rank). A capture before that arms the unpool site alone. Every rank
        warms up and captures the same graphs, so every rank answers alike.
        """
        if rows not in ROWS or torch.compiler.is_compiling():
            return False
        if not torch.cuda.is_current_stream_capturing():
            if "pool" in self.sites and self.pool is not None and self.pool_layers is None:
                self._build_pool()
            return False
        return ("unpool" in self.sites and self.unpool is not None) or self._pool_ready()

    def _build_pool(self) -> None:
        """Choose the publishing pool layers (collective; at the first eager 16/24/32-row forward)."""
        from ..pool_into_route.route import PUBLISH_POLICY as POOL_POLICY
        from ..projection_gathers.gathers import MuseSparkOpenTepGatherMoE
        from ..projection_publish.publish import _pool_args

        chosen: List[int] = []
        for index, layer in enumerate(self._layers):
            moe = layer.feed_forward
            if (
                isinstance(moe, MuseSparkOpenTepGatherMoE)
                and _pool_args(moe, WIDE_MAX_ROWS) is not None
            ):
                chosen.append(index)
        local = tuple(chosen)
        everyone = Distributed.get(self.mapping).tp_allgather(local, small_payload=True)
        reason = None
        if any(tuple(other) != local for other in everyone):
            reason = "the TP ranks chose different publishing pool layers"
        elif len(chosen) < 2:
            reason = f"fewer than two publishing pool layers ({len(chosen)})"
        else:
            reason = pool_owner_conflict(self._layers[index] for index in chosen)
        if reason is None:
            for index in chosen:
                dispatcher = getattr(
                    self._layers[index].feed_forward.pooler, "_low_m_gemm_dispatcher", None
                )
                held = getattr(dispatcher, "publish_policy", None)
                if dispatcher is None or not hasattr(dispatcher, "publish"):
                    reason = f"layer {index}'s pool projection is not on the dense GEMM dispatcher"
                    break
                if held is not None and held.owner != POOL_POLICY.owner:
                    reason = f"layer {index}'s pool GEMM already publishes for {held.owner}"
                    break
        reason = _tp_agreement.agreed_refusal(self.mapping, reason)
        if reason is not None:
            self._pool_refused = reason
            self.pool_layers = ()
            logger.warning(
                f"[musespark_open] wide_direct_transports: the pool site is not armed ({reason}); "
                f"{'/'.join(map(str, ROWS))}-row steps keep the 32-row pool inbox"
            )
            return
        for index, buffer in zip(chosen, _publish.buffer_sequence(len(chosen))):
            moe = self._layers[index].feed_forward
            moe._wide_direct_pool_buffer = buffer
            dispatcher = moe.pooler._low_m_gemm_dispatcher
            if dispatcher.publish_policy is None:
                claim_publish(dispatcher, POOL_POLICY)
        self.pool_layers = tuple(chosen)
        self.pool.start_guard_watch("wide pool")
        logger.info(
            f"[musespark_open] wide_direct_transports: TP{TP_SIZE} decode steps of "
            f"{'/'.join(map(str, ROWS))} rows publish the pool slice from {len(chosen)} layers into the "
            f"32-row pool transport (cells {'|'.join(dict.fromkeys(_cells(self.plan, 'pool')))}; "
            f"consumer pool_publish_fp4quant, PDL {'on' if self.enable_pdl else 'off'})"
        )

    @contextlib.contextmanager
    def armed(self, rows: int) -> Iterator[None]:
        """Arm the requested sites for one decode forward of ``rows`` rows; disarm on any exit.

        Raises:
            RuntimeError: if the forward returned with a published pool slot no consumer took.
        """
        pool_on = "pool" in self.sites and self._pool_ready()
        unpool_on = "unpool" in self.sites and self.unpool is not None
        self._rows = int(rows)
        self._unconsumed = 0
        if unpool_on:
            for dispatcher, join, buffer in self.unpool_sites:
                dispatcher.publish = self.unpool.slot(buffer, rows)
                join._unpool_direct_armed = (self.unpool, buffer)
            self._report("unpool", rows)
        self._armed = pool_on
        try:
            yield
            if self._unconsumed:
                raise RuntimeError(
                    f"[musespark_open] wide_direct_transports: a {rows}-row decode forward returned "
                    f"with {self._unconsumed} published pool slot(s) no consumer took"
                )
        finally:
            self._armed = False
            self._rows = 0
            self._unconsumed = 0
            if unpool_on:
                for dispatcher, join, _ in self.unpool_sites:
                    dispatcher.publish = None
                    join._unpool_direct_armed = None

    def _report(self, site: str, rows: int) -> None:
        if (site, rows) in self._reported:
            return
        self._reported.add((site, rows))
        cell = _cells(self.plan, site)[ROWS.index(rows)] if rows in ROWS else "?"
        logger.info(
            f"[musespark_open] wide_direct_transports: a {rows}-row decode step publishes the {site} "
            f"slice from its GEMM ({cell}) into the 32-row direct transport"
        )

    def project_pool(self, moe, module, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """The pool slot this layer's slice GEMM just published into, or None when the layer does
        not publish."""
        buffer = getattr(moe, "_wide_direct_pool_buffer", None)
        if buffer is None or module is not moe.pooler:
            return None
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
        if (
            rows != self._rows
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            raise RuntimeError(
                f"[musespark_open] wide_direct_transports: armed for {self._rows} rows, the pool "
                f"projection got {tuple(hidden_states.shape)} {hidden_states.dtype}"
            )
        from tensorrt_llm._torch.modules.low_m_gemm import apply_low_m_gemm

        slot = self.pool.slot(buffer, rows)
        columns = self.pool.columns
        begin = self.mapping.tp_rank * columns
        weight = module.weight[begin : begin + columns]
        dispatcher = module._low_m_gemm_dispatcher
        dispatcher.publish = slot
        try:
            projected = apply_low_m_gemm(module, hidden_states, weight, None, force_active=True)
        finally:
            dispatcher.publish = None
        if projected is None or projected.data_ptr() != slot.data_ptr():
            raise RuntimeError(
                "[musespark_open] wide_direct_transports: the pool GEMM of an armed layer did not write "
                "its transport slot (it bypassed the dense GEMM dispatcher); its peers poll for it"
            )
        self._unconsumed += 1
        self._report("pool", rows)
        return slot

    def consume_pool(self, moe, pooled: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """The pre-expert NVFP4 input from the published pool slot ``pooled``, or None when ``pooled``
        is not this layer's slot."""
        buffer = getattr(moe, "_wide_direct_pool_buffer", None)
        if buffer is None or not self.pool.is_slot(pooled, buffer, self._rows):
            return None
        from ..projection_publish.publish import _pool_args

        args = _pool_args(moe, WIDE_MAX_ROWS)
        if args is None:
            raise RuntimeError(
                f"[musespark_open] wide_direct_transports: layer {moe.layer_idx}'s pool slice was "
                "published but its pre-expert quantization no longer fits"
            )
        gain, global_scale = args
        fp4, scale = _pool.pool_publish_rmsnorm_fp4quant(
            self.pool,
            buffer,
            self._rows,
            gain,
            global_scale,
            moe.pre_expert_norm.variance_epsilon,
            enable_pdl=self.enable_pdl,
        )
        self._unconsumed -= 1
        return Fp4QuantizedTensor(fp4, scale, is_sf_swizzled=False)


def _unpool_variants(layers, unpool_sites, captured: FrozenSet[int]) -> list:
    """The direct boundary variants the fused layers launch at ``ROWS`` (``unpool_into_boundary``'s
    settings per layer, at the wide row counts)."""
    index_of = {id(layer.add_ffn): index for index, layer in enumerate(layers)}
    keys = set()
    for _, join, _ in unpool_sites:
        index = index_of[id(join)]
        layer = layers[index]
        for rows in ROWS:
            for capture in (False, True) if index in captured else (False,):
                keys.add(
                    _direct.variant(
                        rows,
                        True,
                        layer.post_ffn_norm.variance_epsilon,
                        layers[index + 1].pre_qkv_proj_norm.variance_epsilon,
                        join._preqkv_norm_fp32,
                        join._boundary_trigger,
                        capture,
                    )
                )
    return sorted(keys)


def install(
    model,
    plan,
    wide_projections,
    unpool_direct,
    captured: FrozenSet[int],
    capability=None,
    sites: Optional[Sequence[str]] = None,
) -> Optional[WideDirectTransports]:
    """Build ``sites``' 32-row transports (collective) and hand the arming to the layers.

    ``sites`` is the ``tep8_wide_direct_transports__SITES`` subset the plan resolved; None is
    every site. The refusal is agreed over the TP group before anything collective; a refused
    path logs a warning and returns None (the steps keep the 32-row inboxes). The unpool
    variants are compiled here (agreed), the pool layers chosen at the first eager armed forward.
    """
    sites = requested_sites(sites)
    if not sites:
        return None
    layers = list(model.layers)
    mapping = layers[0].mapping
    reason = _tp_agreement.agreed_refusal(
        mapping, refusal(model, plan, wide_projections, unpool_direct, sites, capability)
    )
    if reason is not None:
        logger.warning(
            f"[musespark_open] tep8_wide_direct_transports is on but not armed: {reason}"
        )
        return None
    unpool = None
    unpool_sites: Tuple = ()
    if "unpool" in sites:
        unpool_sites = tuple(unpool_direct.sites)
        variants = _unpool_variants(layers, unpool_sites, frozenset(captured))
        unpool = _direct.shared_workspace(mapping, max_rows=WIDE_MAX_ROWS)
        failed: Optional[Exception] = None
        try:
            unpool.build_variants(variants)
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            failed = error
        reason = _tp_agreement.agreed_refusal(
            mapping,
            None if failed is None else f"the wide boundary variants did not build: {failed!r}",
        )
        if reason is not None:
            raise ValueError(f"[musespark_open] wide_direct_transports: {reason}") from failed
        unpool.start_guard_watch()
    pool = None
    if "pool" in sites:
        # Order-independent backstop: _build_pool checks the same again at the first eager step.
        conflict = _tp_agreement.agreed_refusal(mapping, pool_owner_conflict(layers))
        if conflict is not None:
            logger.warning(
                f"[musespark_open] wide_direct_transports: the pool site is not armed ({conflict}); "
                f"{'/'.join(map(str, ROWS))}-row steps keep that path's pool prepare"
            )
            sites = sites - {"pool"}
            if not sites:
                return None
    if "pool" in sites:
        from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

        pool = _publish.PublishTransport(
            mapping,
            _publish.local_columns(_publish.POOL_HIDDEN, TP_SIZE),
            WIDE_MAX_ROWS,
            comm=_get_mnnvl_workspace_comm(mapping),
        )
    wide = WideDirectTransports(model, plan, wide_projections, sites, unpool, unpool_sites, pool)
    logger.info(
        f"[musespark_open] tep8_wide_direct_transports ({', '.join(sorted(sites))}): TP{TP_SIZE} "
        f"decode steps of {'/'.join(map(str, ROWS))} rows publish the "
        f"{' and '.join(sorted(sites))} slice(s) from their split-K GEMMs into 32-row multicast "
        "transports"
        + (
            f"; unpool on {len(unpool_sites)} layers (cells "
            f"{'|'.join(dict.fromkeys(_cells(plan, 'unpool')))}, buffers "
            f"{unpool_sites[-1][2] if unpool_sites else '-'} last)"
            if unpool is not None
            else ""
        )
        + ("; pool layers chosen at the first eager 16/24/32-row step" if pool is not None else "")
        + "; captured graphs only (eager forwards and other row counts keep the 32-row inboxes)"
    )
    return wide
