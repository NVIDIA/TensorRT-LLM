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

An armed layer whose call does not take the prologue's routed path raises: its peers
would poll for a record nobody publishes. Eager steps and other row counts keep
the tagged chain; the two transports never share memory.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import pool_direct_route as _pdr
from ...ops.comm import unpool_direct_boundary as _direct
from ...ops.moe import router_shard as _shard
from ..dense_gemms.dispatcher import (
    FLASHINFER_FAMILIES,
    DenseGemmDispatcher,
    MuseSparkOpenTepDenseLinear,
    PublishPolicy,
    claim_publish,
)
from ..projection_gathers.gathers import _require_aligned

#: The one row count the transport is armed for: the K = 7 verify step.
ROWS = 8


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

    def prepare(self, moe, hidden_states: torch.Tensor, warps: int, gain, global_scale):
        """The prologue MoE's ``prepare_moe_inputs`` result for an armed 8-row call, through the transport.

        ``moe`` is the layer's prologue MoE with the router fork's side stream. ``warps``,
        ``gain`` and ``global_scale`` are what its routed path computed for this
        call.
        """
        from ..moe_prologue.prologue import _placement

        rows = hidden_states.shape[0]
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


class DirectPool:
    """What ``DecodeTarget`` holds under ``pool_into_route``: the workspace and, once loaded, its sites.

    ``sites`` is ``(moe, site)`` per layer that runs the routed path, in call order.
    """

    def __init__(self, workspace: _pdr.PoolDirectWorkspace) -> None:
        self.workspace = workspace
        self.sites: Tuple[Tuple[torch.nn.Module, DirectPoolSite], ...] = ()

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows runs on the direct transport.

        Exactly 8 rows, inside a CUDA graph capture. Every rank captures the same
        graphs with the same row counts, so either every rank arms or none does.
        """
        return rows == ROWS and torch.cuda.is_current_stream_capturing()

    @contextlib.contextmanager
    def armed(self, rows: int):
        """Arm every site for one forward of ``rows`` rows; disarm on any exit."""
        for moe, site in self.sites:
            moe._pool_direct_armed = site
        try:
            yield
        finally:
            for moe, _ in self.sites:
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

    def install_sites(self, model, mapping) -> int:
        """Find, compile and claim the sites once the weights are final (collective).

        Returns the number of sites. Raises ``ValueError`` on every TP rank alike when
        fewer than two layers run the routed path, when another fast path already
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
            f"graphs; buffers alternate per layer, the last on buffer {buffers[-1]}; "
            f"{len(variants)} kernel variants); eager steps and other row counts keep the tagged "
            "chain"
        )
        return len(sites)


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
    return DirectPool(_pdr.shared_workspace(layers[0].mapping))
