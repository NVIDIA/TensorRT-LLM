# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``pool_into_route``: the pool GEMM and the forked router shard publish into the routed pool kernel.

With ``moe_prologue`` and ``router_fork`` a decode MoE layer of at most 8 rows runs the router
shard on a side stream (a plain launch that writes ``[M, L]`` FP32 logits, L = 32 at TP8 and 64 at TP4) and
the pool slice GEMM on the main stream. After the join, the routed pool kernel
(``ops/comm/pool_gather_route``) waits for both, reloads this rank's
``[M, C]`` slice (C = 512 at TP8, 1024 at TP4) and logits, publishes them to its peers as tagged words, and
polls the peers' words.

Here, on the decode CUDA graph of 8 rows (the DFlash2 K = 7 verify step), the
pool GEMM's output tensor is this rank's pool slot of a multicast transport
(``ops/comm/pool_direct_route``), and the router shard's publish twin
(``ops/moe/router_shard.router_shard_publish``, a plain launch on the
router fork's side stream) stores its logit block into this rank's logit boxes of the
same transport. The routed kernel (``pool_direct_route``) then only polls and
resets, and runs the tagged kernel's routing, placement, early-id publication and RMSNorm +
NVFP4 quantization statement for statement. Same bits.

The join (``main.wait_stream(side)`` in front of the routed kernel's launch with
programmatic dependent launch) is captured as a programmatic edge from the twin,
so the routed kernel can start before the twin's grid has completed. Its routing
warps read the logit boxes only after ``griddepcontrol.wait``, which returns once
every programmatic predecessor (the pool GEMM and the twin) has completed, and
take each logit word only once its sentinel test passes.

What this module does:

* ``install`` (from the modeling shell, after ``router_fork``) refuses on every TP rank
  alike in these cases: without the ``dense_gemms`` plan, with programmatic dependent
  launch off, when a layer is not the prologue MoE with the router fork's side stream, or
  when its pooler is not the ``dense_gemms`` ``Linear``. Otherwise it builds the group's
  workspace (collective) and returns the ``DirectPool`` that ``DecodeTarget``
  arms.
* ``DirectPool.install_sites`` runs from ``modeling.install_loaded_features``,
  after ``replicated_experts`` / ``persistent_experts`` / ``three_window_placement`` and
  ``early_expert_ids``, whose placement and early ids
  the routed kernel's variants are compiled with. It does the following:

  - finds the layers whose 8-row decode call runs the routed path (the
    NVFP4-expert layers), using the predicates the forward evaluates, on shapes
    and construction-time state only;
  - gives each layer a buffer (``buffer_sequence``);
  - compiles the variants (collective, agreed);
  - claims each layer's pool dispatcher with this fast path's producer policy
    (``claim_publish``);
  - starts the transport's guard watch.
* ``DirectPool.armed`` (from ``DecodeTarget``, for one forward of exactly 8 rows
  captured into a CUDA graph) gives each site's MoE its ``DirectPoolSite``.
  ``moe_prologue`` then runs ``DirectPoolSite.prepare`` in place of the tagged
  chain. Everything is disarmed on any exit.

Steps of two to four requests at TP8 (``tep8_wide_direct_pool``, default off;
the TP8 side of upstream b5ca973's TP4 wide direct pool sites, #21377): when the TEP8 router
shard (``tep8_router_shard``) gave every routed layer its TP8 wide route inbox,
``install_sites`` also builds a second transport on the wide layout
(``PoolDirectWorkspace(max_rows=WIDE_MAX_ROWS)``, routing only) with the same layers and buffer
sequence, and a 16..32-row forward captured into a CUDA graph arms those sites. The router
shard's wide path (``_wide_prepare``) then runs ``DirectPoolSite.prepare``: its router shard
(``_wide_router_shard``) writes this rank's ``[rows, 32]`` logits into the record's logit block
on the router fork's side stream, the pool GEMM (``wide_projection_gathers``' split-K cell)
writes the slot, and the routed kernel only polls them and runs the wide tagged kernel's
statements (``pool_gather_route``'s on the 32-row route inbox). Same bits as that path; the
placement stays ``ring_buddy_assign``'s, from the routing memo.

An armed layer whose call does not take the prologue's routed path raises: its peers
would poll for a record nobody publishes. Eager steps and other row counts keep
the tagged chain; the transports never share memory.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import pool_direct_route as _pdr
from ...ops.comm import unpool_direct_boundary as _direct
from ...ops.comm.projection_inbox import WIDE_MAX_ROWS
from ...ops.moe import router_shard as _shard
from ..dense_gemms.dispatcher import (
    FLASHINFER_FAMILIES,
    DenseGemmDispatcher,
    MuseSparkOpenTepDenseLinear,
    PublishPolicy,
    claim_publish,
)
from ..projection_gathers.gathers import _require_aligned

#: The one row count the 8-row transport is armed for: the K = 7 verify step.
ROWS = 8
#: Rows of the narrowest wide call: two requests' K = 7 verify step. Pre-#21754 this was
#: ``router_shard.WIDE_MIN_ROWS``; 736cfdc7 moved the wide router (and that constant) out of
#: the sm103 tree into sm107's, so the wide sites carry the bound themselves. Same value.
WIDE_MIN_ROWS = 16
#: The lever of the TP8 wide sites (16..32-row steps beside the TEP8 router shard):
#: ``tep8_wide_direct_pool``. Off, nothing is built.


def _publish_copy(slot: torch.Tensor, product: torch.Tensor) -> None:
    _direct.publish_copy(slot, product)


#: The pool sites' producer policy. flashinfer's split-K and direct kernels store each
#: element once, from their epilogue, after their dependency wait, so they write the
#: slot themselves. A cuBLAS kernel, or a call that ``fits`` declines, is computed as
#: usual and its product is copied into the slot.
PUBLISH_POLICY = PublishPolicy("pool_direct", FLASHINFER_FAMILIES, _publish_copy)


class DirectPoolSite:
    """One layer's place on the transport: the workspace, its buffer and its pool dispatcher."""

    def __init__(
        self, workspace: _pdr.PoolDirectWorkspace, buffer: int, dispatcher: DenseGemmDispatcher
    ) -> None:
        self.workspace = workspace
        self.buffer = int(buffer)
        self.dispatcher = dispatcher

    def prepare(self, moe, hidden_states: torch.Tensor, warps, gain, global_scale):
        """The prologue MoE's ``prepare_moe_inputs`` result for an armed call, through the transport.

        ``moe`` is the layer's prologue MoE with the router fork's side stream. ``warps``,
        ``gain`` and ``global_scale`` are what its routed path computed for this
        call (its 8-row path, or its wide path on a wide site, where ``warps`` is unused:
        the router shard picks its own).
        """
        from ..moe_prologue.prologue import _placement

        rows = hidden_states.shape[0]
        if self.workspace.wide:
            if not WIDE_MIN_ROWS <= rows <= self.workspace.max_rows:
                raise RuntimeError(
                    f"[musespark_open] pool_into_route: a layer armed for wide calls got {rows} rows; "
                    "its peers poll the transport for its records"
                )
            return self._prepare_wide(moe, hidden_states, warps, gain, global_scale)
        if rows != ROWS:
            raise RuntimeError(
                f"[musespark_open] pool_into_route: a layer armed for {ROWS}-row calls got {rows} rows; its peers "
                "poll the transport for its records"
            )
        workspace, buffer = self.workspace, self.buffer
        routing = moe.experts.routing_method
        side = moe._router_stream
        main = torch.cuda.current_stream()
        with moe.nvtx("router"):
            boxes = workspace.boxes(buffer, rows)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                _shard.router_shard_publish(
                    hidden_states,
                    moe.router.weight,
                    workspace.rank,
                    moe.metap_multiplier,
                    warps,
                    boxes,
                    ranks=workspace.world,
                )
        with moe.nvtx("pool"):
            slot = workspace.slot(buffer, rows)
            self.dispatcher.publish = slot
            try:
                projected = moe._replicated_projection(
                    moe.pooler, hidden_states, moe._pool_gather_workspace
                )
            finally:
                self.dispatcher.publish = None
            if projected.data_ptr() != slot.data_ptr():
                raise RuntimeError(
                    "[musespark_open] pool_into_route: the pool GEMM of an armed layer did not write the transport "
                    "slot "
                    "(it bypassed the dense_gemms pool dispatcher); its peers poll for that slot"
                )
            main.wait_stream(side)
            place, static_from, state = _placement(moe)
            bias = routing.e_score_correction_bias
            _require_aligned(gain=gain, bias=bias)
            early = getattr(moe, "_early_ids_state", None)
            out = _pdr.pool_direct_route(
                workspace,
                buffer,
                rows,
                gain,
                global_scale,
                moe.pre_expert_norm.variance_epsilon,
                bias,
                routing.renorm_eps,
                place,
                static_from,
                enable_pdl=moe._prologue_enable_pdl,
                return_gathered=True,
                early=early,
            )
        moe._prologue_record_routing(out, place, state)
        return out.logits, Fp4QuantizedTensor(out.fp4, out.scale, is_sf_swizzled=False)

    def _projected_into_slot(self, moe, hidden_states: torch.Tensor, slot: torch.Tensor) -> None:
        self.dispatcher.publish = slot
        try:
            projected = moe._replicated_projection(
                moe.pooler, hidden_states, moe._pool_gather_workspace
            )
        finally:
            self.dispatcher.publish = None
        if projected.data_ptr() != slot.data_ptr():
            raise RuntimeError(
                "[musespark_open] pool_into_route: the pool GEMM of an armed layer did not write the transport "
                "slot (it bypassed the dense_gemms pool dispatcher); its peers poll for that slot"
            )

    def _prepare_wide(self, moe, hidden_states: torch.Tensor, warps, gain, global_scale):
        """``prepare`` of a 16..32-row call on the wide layout: ``_wide_prepare``'s chain with the
        producers publishing (upstream b5ca973's ``_prepare_wide``, routing only at TP8)."""
        from ..moe_prologue.prologue import _report_tep8

        rows = hidden_states.shape[0]
        world = self.workspace.world
        place, static_from, state = _wide_placement(moe, world, rows)
        workspace, buffer = self.workspace, self.buffer
        routing = moe.experts.routing_method
        side = moe._router_stream
        main = torch.cuda.current_stream()
        with moe.nvtx("router"):
            block = workspace.block(buffer, rows)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                moe._wide_router_shard(hidden_states, workspace.rank, warps, block, world)
        with moe.nvtx("pool"):
            self._projected_into_slot(moe, hidden_states, workspace.slot(buffer, rows))
            main.wait_stream(side)
            bias = routing.e_score_correction_bias
            _require_aligned(gain=gain, bias=bias)
            out = _pdr.pool_direct_route(
                workspace,
                buffer,
                rows,
                gain,
                global_scale,
                moe.pre_expert_norm.variance_epsilon,
                bias,
                routing.renorm_eps,
                place,
                static_from,
                enable_pdl=moe._prologue_enable_pdl,
                return_gathered=False,
            )
        moe._prologue_record_routing(out, place, state, world)
        _report_tep8(world, rows, moe.layer_idx, None, moe._wide_router_rows, direct=True)
        return out.logits, Fp4QuantizedTensor(out.fp4, out.scale, is_sf_swizzled=False)


class DirectPool:
    """What ``DecodeTarget`` holds under ``pool_into_route``: the workspaces and, once loaded, their sites.

    ``sites`` is ``(moe, site)`` per layer that runs the routed path, in call order;
    ``wide_sites`` the same layers on the TP8 wide workspace (``wide_workspace``, built by
    ``install_sites`` under ``tep8_wide_direct_pool`` beside the TEP8 router shard), else empty.
    """

    def __init__(
        self,
        workspace: _pdr.PoolDirectWorkspace,
        wide_workspace: Optional[_pdr.PoolDirectWorkspace] = None,
    ) -> None:
        self.workspace = workspace
        self.wide_workspace = wide_workspace
        self.sites: Tuple[Tuple[torch.nn.Module, DirectPoolSite], ...] = ()
        self.wide_sites: Tuple[Tuple[torch.nn.Module, DirectPoolSite], ...] = ()

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows runs on a direct transport.

        Inside a CUDA graph capture: exactly 8 rows, or 16..32 rows when the wide sites are
        installed. Every rank captures the same graphs with the same row counts and installs
        the same sites, so either every rank arms or none does.
        """
        if not torch.cuda.is_current_stream_capturing():
            return False
        if rows == ROWS:
            return True
        return bool(self.wide_sites) and WIDE_MIN_ROWS <= rows <= WIDE_MAX_ROWS

    @contextlib.contextmanager
    def armed(self, rows: int, skip: Optional[Callable[[int], bool]] = None):
        """Arm the sites of the ``rows``-row transport for one forward; disarm on any exit.

        ``skip``: a predicate on the layer index; the
        sites of the layers it accepts stay unarmed (their calls never reach the prologue). Every
        rank passes the same predicate, so every rank arms the same sites. Without it every site
        of the transport is armed.
        """
        sites = self.sites if rows <= ROWS else self.wide_sites
        if skip is not None:
            sites = tuple((moe, site) for moe, site in sites if not skip(int(moe.layer_idx)))
        for moe, site in sites:
            moe._pool_direct_armed = site
        try:
            yield
        finally:
            for moe, _ in sites:
                moe._pool_direct_armed = None

    def variants(self, layers: Sequence) -> list:
        """The routed kernel's variants the layers' calls launch (see ``pool_direct_route.variant``).

        A layer with early expert ids gets the publishing variant as well as the plain
        one: the host decides per call whether a call publishes
        (``EarlyExpertIds.may_publish``).
        """
        from ..moe_prologue.prologue import _placement

        keys = set()
        for layer in layers:
            moe = layer.feed_forward
            place, static_from, _ = _placement(moe)
            enable_pdl = bool(moe._prologue_enable_pdl)
            keys.add(_pdr.variant(place, static_from, enable_pdl, True))
            early = getattr(moe, "_early_ids_state", None)
            if early is not None and place != _pdr.PLACE_NONE and enable_pdl:
                keys.add(_pdr.variant(place, static_from, enable_pdl, True, early))
        return sorted(keys)

    def install_sites(self, model, mapping, wide: bool = False) -> int:
        """Find, compile and claim the sites once the weights are final (collective).

        ``wide`` is the ``tep8_wide_direct_pool`` lever: with it the TP8 wide sites are
        built too. Returns the number of sites. Raises ``ValueError`` on every TP rank alike
        when fewer than two layers run the routed path, when another fast path already
        publishes from a pool GEMM, or when a variant fails to compile on some rank.
        """
        layers = list(model.layers)
        routed, reason = [], None
        for layer in layers:
            if _routes(layer.feed_forward):
                routed.append(layer)
        if not getattr(model, "_layer_kwargs", {}).get("fused_pool_gather"):
            # The decode forward passes fused_pool_gather to every layer only when the pool site
            # is fused; without it no layer defers its pool GEMM to the prologue's chain.
            reason = (
                "[musespark_open] pool_into_route needs the fused pool site, which this installation "
                "keeps built-in"
            )
        elif len(routed) < 2:
            why = sorted(
                {f"{_not_routed(layer.feed_forward)}" for layer in layers if layer not in routed}
            )
            reason = (
                f"[musespark_open] pool_into_route found {len(routed)} layer(s) whose 8-row decode call runs the "
                f"routed pool kernel; it needs at least two (the others: {'; '.join(why)})"
            )
        else:
            for layer in routed:
                held = getattr(_pool_dispatcher(layer.feed_forward), "publish_policy", None)
                if held is not None and held.owner != PUBLISH_POLICY.owner:
                    reason = (
                        f"[musespark_open] pool_into_route: layer {layer.layer_idx}'s pool GEMM already publishes for "
                        f"{held.owner}"
                    )
                    break
        reason = _tp_agreement.agreed_refusal(mapping, reason)
        if reason is not None:
            raise ValueError(reason)
        variants = self.variants(routed)
        failed: Optional[Exception] = None
        try:
            self.workspace.build_variants(variants)
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            failed = error
        reason = _tp_agreement.agreed_refusal(
            mapping,
            None
            if failed is None
            else f"[musespark_open] pool_into_route could not build its routed kernel variants: {failed!r}",
        )
        if reason is not None:
            raise ValueError(reason) from failed
        buffers = _direct.buffer_sequence(len(routed))
        sites = []
        for layer, buffer in zip(routed, buffers):
            moe = layer.feed_forward
            dispatcher = _pool_dispatcher(moe)
            claim_publish(dispatcher, PUBLISH_POLICY)
            moe._pool_direct_armed = None
            sites.append((moe, DirectPoolSite(self.workspace, buffer, dispatcher)))
        self.sites = tuple(sites)
        self.workspace.start_guard_watch()
        logger.info(
            f"[musespark_open] pool_into_route: the pool GEMM and the forked router shard publish into the routed pool "
            f"kernel's direct transport on {len(sites)} of {len(layers)} layers (8-row decode "
            "graphs; "
            f"buffers alternate per layer, the last on buffer {buffers[-1]}; "
            f"{len(variants)} kernel variants); eager steps and other row "
            "counts keep the tagged chain"
        )
        if wide:
            self._install_wide_sites(routed, buffers, mapping)
        return len(sites)

    def _install_wide_sites(self, routed: Sequence, buffers: Sequence[int], mapping) -> None:
        """``tep8_wide_direct_pool``: the TP8 wide workspace and its sites on the routed layers (collective).

        Needs the TEP8 router shard's wide route inbox (``_wide_inbox``, TP8) on every routed
        layer: its ``_wide_prepare`` is the path the wide sites publish for. A refusal is
        agreed over the TP group and only warns (16..32-row steps keep that path's tagged
        route inbox); a variant that does not compile raises on every rank.
        """
        reason = None
        if self.workspace.world != 8:
            reason = f"TP{self.workspace.world}: the wide sites are built at TP8 only"
        else:
            for layer in routed:
                inbox = getattr(layer.feed_forward, "_wide_inbox", None)
                if inbox is None or getattr(inbox, "world", None) != 8:
                    reason = (
                        f"layer {layer.layer_idx} has no TP8 wide route inbox (set "
                        "tep8_router_shard, whose 16..32-row path these sites publish for)"
                    )
                    break
        reason = _tp_agreement.agreed_refusal(mapping, reason)
        if reason is not None:
            logger.warning(
                "[musespark_open] tep8_wide_direct_pool is on but the wide sites are not built: "
                f"{reason}"
            )
            return
        # The group the 8-row workspace was built for (``install``: the layers' mapping).
        wide = _pdr.shared_workspace(routed[0].mapping, max_rows=WIDE_MAX_ROWS)
        self.wide_workspace = wide
        wide_variants = self.wide_variants(routed)
        failed: Optional[Exception] = None
        try:
            wide.build_variants(wide_variants)
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            failed = error
        reason = _tp_agreement.agreed_refusal(
            mapping,
            None
            if failed is None
            else f"[musespark_open] pool_into_route could not build its wide routed kernel: {failed!r}",
        )
        if reason is not None:
            raise ValueError(reason) from failed
        wide_sites = []
        for (moe, site), buffer in zip(self.sites, buffers):
            wide_sites.append((moe, DirectPoolSite(wide, buffer, site.dispatcher)))
        self.wide_sites = tuple(wide_sites)
        wide.start_guard_watch()
        logger.info(
            f"[musespark_open] tep8_wide_direct_pool: TP8 decode steps of "
            f"{WIDE_MIN_ROWS}..{WIDE_MAX_ROWS} rows (captured) publish the pool slice and the "
            f"router shard's logits into the wide routed kernel's direct transport on {len(wide_sites)} "
            f"layers (routing only, {len(wide_variants)} kernel variant(s); buffers as the 8-row sites')"
        )

    def wide_variants(self, layers: Sequence) -> list:
        """The wide routed kernel's variants the layers' wide calls launch (``_wide_placement``)."""
        keys = set()
        for layer in layers:
            moe = layer.feed_forward
            for rows in (WIDE_MIN_ROWS, WIDE_MAX_ROWS):
                place, static_from, _ = _wide_placement(moe, self.wide_workspace.world, rows)
                keys.add(_pdr.variant(place, static_from, bool(moe._prologue_enable_pdl), False))
        return sorted(keys)


def _wide_placement(moe, world: int, rows: int):
    """``(kind, static_from, state)`` of a wide call: the half window's placement on a layer
    whose placement serves ``rows`` rows (TP4), else routing only (``ring_buddy_assign`` places)."""
    from ..moe_prologue.prologue import _placement

    place, static_from, state = _placement(moe, world)
    if place == _pdr.PLACE_HALF and world == 4 and state is not None and rows <= state.max_rows:
        return place, static_from, state
    return _pdr.PLACE_NONE, _pdr.LOCAL_EXPERTS, None


def _pool_dispatcher(moe) -> Optional[DenseGemmDispatcher]:
    pooler = getattr(moe, "pooler", None)
    if not isinstance(pooler, MuseSparkOpenTepDenseLinear):
        return None
    dispatcher = pooler._low_m_gemm_dispatcher
    return dispatcher if isinstance(dispatcher, DenseGemmDispatcher) else None


def _not_routed(moe) -> Optional[str]:
    """Why ``moe``'s 8-row decode call is not to run the prologue's routed path, or None.

    The construction-time part of the forward's predicates (``_prologue_router_warps`` and
    ``_pool_quant_args``): the prologue MoE with the router fork's side stream, the deferred
    and fused pool site, the prologue's inbox, the fused pre-expert quantization, NVFP4
    experts without an activation scale and without an expert-parallel communicator.
    The rest of those predicates reads values the loader's post-load walk finalizes
    after this install (such as the router weight's dtype and the experts' input
    scale), so the forward checks them on every armed call and raises if one fails
    (``MuseSparkOpenTepPrologueMoE._pool_direct_refuse_declined``). Reads no pointer and no
    rank-specific value, so every rank gets the same answer.
    """
    from ..moe_prologue.prologue import MuseSparkOpenTepPrologueMoE

    if not isinstance(moe, MuseSparkOpenTepPrologueMoE):
        return f"{type(moe).__name__} is not the prologue MoE"
    if moe._router_stream is None:
        return "no router-fork side stream"
    if getattr(moe, "_pool_inbox", None) is None:
        return "the pool site is not fused"
    if getattr(moe, "_prologue_inbox", None) is None:
        return "no prologue inbox"
    if not getattr(moe, "_pre_expert_quant_fused", True):
        return "the pre-expert quantization is not fused"
    if getattr(moe.experts, "comm", None) is not None:
        return "the experts have a communicator"
    backend = getattr(moe.experts, "backend", moe.experts)
    if not getattr(backend, "has_nvfp4", False):
        return "the experts are not NVFP4"
    if getattr(backend, "fc31_act_scale", None) is not None:
        return "the experts have an activation scale"
    return None


def _routes(moe) -> bool:
    """Whether ``moe``'s 8-row decode call is to run the prologue's routed path (``_not_routed``)."""
    return _not_routed(moe) is None


def install(model, plan, mapping=None) -> DirectPool:
    """Check every layer and build the group's direct pool workspace (collective).

    ``plan`` is the ``dense_gemms`` plan. The outcome of the checks is agreed over
    ``mapping``'s TP group (``tp_agreement.agreed_refusal``) before the workspace's
    collective construction. The sites are found once the weights are final
    (``DirectPool.install_sites``).

    Raises:
        ValueError: on every rank alike: without the ``dense_gemms`` plan, when a layer is not
            the prologue MoE with the router fork's side stream, or when a layer's pooler is not
            the ``dense_gemms`` ``Linear``.
    """
    from ..moe_prologue.prologue import MuseSparkOpenTepPrologueMoE

    layers = list(model.layers)
    reason = None
    if plan is None:
        reason = (
            "[musespark_open] pool_into_route publishes through the dense_gemms pool dispatcher; "
            "select dense_gemms"
        )
    else:
        for layer in layers:
            moe = layer.feed_forward
            if not isinstance(moe, MuseSparkOpenTepPrologueMoE) or moe._router_stream is None:
                reason = (
                    f"[musespark_open] pool_into_route publishes from the forked router shard; layer "
                    f"{layer.layer_idx} has {type(moe).__name__} without the router fork's side stream"
                )
                break
            if _pool_dispatcher(moe) is None:
                reason = (
                    f"[musespark_open] pool_into_route: layer {layer.layer_idx}'s pool projection is "
                    f"{type(moe.pooler).__name__}, not the dense_gemms MuseSparkOpenTepDenseLinear"
                )
                break
    reason = _tp_agreement.agreed_refusal(mapping, reason)
    if reason is not None:
        raise ValueError(reason)
    group = layers[0].mapping
    workspace = _pdr.shared_workspace(group)
    return DirectPool(workspace)
