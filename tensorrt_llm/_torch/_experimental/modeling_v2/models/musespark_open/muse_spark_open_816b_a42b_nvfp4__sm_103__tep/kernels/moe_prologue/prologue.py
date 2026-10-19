# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``moe_prologue``: the decode MoE prologue as one chain on one stream.

Built-in decode at TP8 (TP4) prepares a MoE layer's input on two streams that join
before the experts: on the main stream the replicated FP32 router reads its whole
8 MiB weight on every rank (256 logits per row), the routing kernel selects the
top-8, and with ``replicated_experts`` its placement kernel places the selection; on
the preparation stream the rank's pool slice GEMM and the pool gather +
pre-expert NVFP4 quantization (the ``projection_gathers`` fused kernel) run. Each rank
only ever runs 32 (64) of the 256 experts, and the logits of the others are
computed by the ranks that own them as well.

With this fast path a decode step of at most 8 rows runs, on the current stream:

1. ``ops/moe/router_shard``: this rank's 32 (64) logit columns (1 (2) MiB of weight),
   bit for bit the built-in router's columns;
2. the pool slice GEMM, as on the built-in path (the ``dense_gemms`` dispatcher when selected);
3. ``ops/comm/pool_gather_route``: the fused pool gather + quantization,
   whose inbox message also carries each rank's logit block, followed in
   the same kernel by the routing of every row and, with ``replicated_experts``, its placement.

No fork and no join: the router's read does not compete with the pool chain
for SMs and HBM, and the routing, the placement and the BF16 cast of the weights
are not launches of their own. The experts and the finalize are reached
unchanged: their routing calls read the routing method's memo and the replicated
experts' placement, which this module fills for the logits tensor the fused op returns.
On a layer that ``three_window_placement`` gave three ring windows the placement is the
three-window one (``PLACE_BALANCE3``) and is recorded with its 96-expert window.

Values. The logits, the selected ids, the routing weights, the placement and
the quantized expert input are the built-in path's bit for bit (the ops' tests
certify each against the built-in chain); so the experts' inputs and therefore
every layer output are unchanged.

``router_fork`` takes the router shard off that chain. The router and the pool GEMM
both read only the attention hop's output, and neither reads the other's. A
pool GEMM CTA needs almost all of an SM's shared memory, so behind the router's
grid (a CTA on every SM) the GEMM cannot become resident before the router
ends, and its weight stream does not overlap anything. With ``router_fork`` a decode
step of at most 8 rows runs:

1. on the current stream, the router's ``[M, 32]`` FP32 output is allocated,
   and the side stream (created at install, one per device, a stream of its
   own rather than one of PyTorch's pooled streams) waits for the current one;
2. on the side stream, ``router_shard_side``: the same kernel statements as a
   plain launch, with no programmatic-dependent-launch attribute and no
   ``griddepcontrol``, so it starts once the hop has completed;
3. on the current stream, the pool GEMM, directly behind the hop on the
   programmatic chain (with the hop's early trigger the hop releases it right after its last
   transport access, and its CTAs become resident and stream weights while the
   hop finishes);
4. the current stream waits for the side stream, then the routed pool kernel
   runs as before. Its routing warps read the logits only after their
   ``griddepcontrol.wait`` (``ops/comm/pool_gather_route``), which returns once
   both the GEMM and the router completed.

The same kernels compute the same values (``router_shard_side`` is
``router_shard``'s kernel without its launch attribute); only the launch order
and streams change. Decode steps of more than 8 rows and prefill keep the
projection-gather path. The modeling shell refuses ``router_fork`` without
``moe_prologue`` or the hop's early trigger;
``install`` refuses it, on every rank alike, with programmatic
dependent launch off.

Every rank takes the same path at every call: the choice reads shapes, dtypes,
contiguity and construction-time state only, never pointer values.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenModel,
    MuseSparkOpenRoutingMethod,
    _version_or_none,
)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...dispatch import LAUNCH_PDL
from ...ops.comm import pool_gather_route as _route
from ...ops.comm.projection_inbox import MAX_ROWS, ProjectionInbox
from ...ops.moe import router_shard as _shard
from ..context_kv_fork.fork import dedicated_stream
from ..projection_gathers.gathers import MuseSparkOpenTepGatherMoE, _require_aligned

#: The built-in code the routed prologue reproduces: the router kernel and its
#: launch choice (``_router_projection_kernel``, ``MuseSparkOpenMoE.router_logits``),
#: the routing kernel (``_full_routing_kernel``) and the memo the experts and the
#: finalize read (``MuseSparkOpenRoutingMethod.apply``); its ``reference`` runs them.


def _placement(moe, world: Optional[int] = None):
    """``(kind, static_from, state)`` of the replicated experts' placement on this layer, if installed and active.

    ``world`` is the TP size; by default the prologue inbox's (TP8 without one).
    """
    if world is None:
        world = getattr(getattr(moe, "_prologue_inbox", None), "world", 8)
    local = _route.local_experts(world)
    state = getattr(moe, "_replicated_experts", None)
    if state is None or not state.active:
        return _route.PLACE_NONE, local, None
    if state.balance and getattr(state, "windows", 2) == 3:
        return _route.PLACE_BALANCE3, local, state
    if state.balance:
        return _route.PLACE_BALANCE, local, state
    if state.static_from < local:
        return _route.PLACE_STATIC, int(state.static_from), state
    return _route.PLACE_OWNER, local, state


class MuseSparkOpenTepPrologueMoE(MuseSparkOpenTepGatherMoE):
    """The projection-gather MoE whose decode input prep is one chain: router shard, pool GEMM, routed pool kernel."""

    _prologue_inbox: Optional[ProjectionInbox]
    _prologue_enable_pdl: bool
    #: ``router_fork``: the side stream the router shard runs on, or None (the one-chain order).
    _router_stream: Optional[torch.cuda.Stream] = None
    #: ``pool_into_route``: while ``pool_direct.DirectPool.armed`` holds, this layer's
    #: ``pool_direct.DirectPoolSite`` (its producers publish into the routed kernel's
    #: transport); None otherwise.
    _pool_direct_armed = None

    def prepare_moe_inputs(self, hidden_states: torch.Tensor):
        warps = self._prologue_router_warps(hidden_states)
        if warps is None:
            self._pool_direct_refuse_declined(hidden_states)
            return super().prepare_moe_inputs(hidden_states)
        rows = hidden_states.shape[0]
        inbox = self._prologue_inbox
        world = inbox.world
        # The pool quantization's arguments, checked as the projection gathers check them on the
        # slice the GEMM will produce (a contiguous [rows, 4096 / TP] BF16 view).
        columns = _route.local_columns(world)
        probe = hidden_states.view(-1)[: rows * columns].view(rows, columns)
        args = self._pool_quant_args(probe)
        if args is None:
            self._pool_direct_refuse_declined(hidden_states)
            return super().prepare_moe_inputs(hidden_states)
        gain, global_scale = args
        if self._pool_direct_armed is not None:
            return self._pool_direct_armed.prepare(self, hidden_states, warps, gain, global_scale)
        routing = self.experts.routing_method
        side = self._router_stream
        with self.nvtx("router"):
            if side is None:
                logits_local = _shard.router_shard(
                    hidden_states,
                    self.router.weight,
                    inbox.rank,
                    self.metap_multiplier,
                    warps,
                    self._prologue_enable_pdl,
                    ranks=world,
                )
            else:
                # router_fork: allocated on the current stream, which reads it after the join;
                # written on the side stream, which starts after the hop completed.
                main = torch.cuda.current_stream()
                logits_local = torch.empty(
                    (rows, _shard.local_experts(world)),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    _shard.router_shard_side(
                        hidden_states,
                        self.router.weight,
                        inbox.rank,
                        self.metap_multiplier,
                        warps,
                        logits_local,
                        ranks=world,
                    )
        with self.nvtx("pool"):
            projected = self._replicated_projection(
                self.pooler, hidden_states, self._pool_gather_workspace
            )
            if side is not None:
                main.wait_stream(side)
            place, static_from, state = _placement(self, world)
            bias = routing.e_score_correction_bias
            _require_aligned(pooled=projected, gain=gain, bias=bias, logits=logits_local)
            # early_expert_ids: the expert ids published to the persistent expert
            # kernel before this kernel ends.
            early = getattr(self, "_early_ids_state", None)
            out = _route.pool_gather_route(
                projected,
                logits_local,
                inbox,
                gain,
                global_scale,
                self.pre_expert_norm.variance_epsilon,
                bias,
                routing.renorm_eps,
                place,
                static_from,
                enable_pdl=self._prologue_enable_pdl,
                return_gathered=True,
                **({} if early is None else {"early": early}),
            )
        self._prologue_record_routing(out, place, state, world)
        return out.logits, Fp4QuantizedTensor(out.fp4, out.scale, is_sf_swizzled=False)

    def _prologue_record_routing(self, out, place: int, state, world: int = 8) -> None:
        """What the experts and the finalize read, keyed on the logits object: the
        routing memo, and the replicated experts' placement."""
        routing = self.experts.routing_method
        version = _version_or_none(out.logits)
        routing._routing_memo = (out.logits, version, out.selected, out.weights)
        if state is not None:
            state.placement = (out.logits, version, out.local_ids, out.weights_bf16, out.stats)
            if place == _route.PLACE_BALANCE3:
                state.placement += (3 * _route.local_experts(world),)
            state.last_stats = out.stats

    def _pool_direct_refuse_declined(self, hidden_states: torch.Tensor) -> None:
        """Raise when ``pool_into_route`` armed this layer and its call does not take the routed path.

        The armed layers are exactly those whose 8-row call takes it
        (``pool_direct.DirectPool.install_sites``). Every peer polls the transport for
        this layer's records, so a fallback would leave them polling.
        """
        if self._pool_direct_armed is not None:
            raise RuntimeError(
                f"[musespark_open] pool_into_route: layer {self.layer_idx} was armed for the direct pool transport, "
                f"but its call ({tuple(hidden_states.shape)} {hidden_states.dtype}) does not "
                "take the routed prologue path; its peers poll for its records"
            )

    def _prologue_router_warps(self, hidden_states: torch.Tensor) -> Optional[int]:
        """The built-in router launch's warp count when this path runs, else None.

        Exactly where the built-in decode would launch the small-batch router
        kernel and route with the fused routing kernel, on a pool-deferring
        call of at most 8 BF16 rows. Reads nothing that differs between ranks.
        """
        if not getattr(self, "_defer_pool", False) or self._prologue_inbox is None:
            return None
        routing = getattr(self.experts, "routing_method", None)
        weight = getattr(self.router, "weight", None)
        if (
            torch.compiler.is_compiling()
            or not isinstance(routing, MuseSparkOpenRoutingMethod)
            or not routing._memo_enabled
            or routing.top_k != _route.TOP_K
            or routing.output_dtype != torch.float32
            or self.num_experts != _route.EXPERTS
            or weight is None
            or weight.dtype != torch.float32
            or tuple(weight.shape) != (_shard.EXPERTS, _shard.HIDDEN)
            or not weight.is_contiguous()
            or getattr(self.experts, "comm", None) is not None
            or hidden_states.dim() != 2
            or not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or hidden_states.shape[1] != _shard.HIDDEN
            or not 1 <= hidden_states.shape[0] <= MAX_ROWS
            or weight.device != hidden_states.device
        ):
            return None
        bias = routing.e_score_correction_bias
        if (
            bias is None
            or not bias.is_cuda
            or bias.dtype != torch.float32
            or tuple(bias.shape) != (_route.EXPERTS,)
            or not bias.is_contiguous()
        ):
            return None
        overlaps = getattr(self, "_prepare_stream", None) is not None
        return _shard.builtin_warps(hidden_states.shape[0], overlaps)


def agreed_build(mapping) -> bool:
    """Whether every TP rank can build the routed kernel (collective over the TP group).

    Agreeing first makes every rank raise together when one cannot.
    """
    everyone = Distributed.get(mapping).tp_allgather(bool(_route.available()), small_payload=True)
    return all(everyone)


def install(model: MuseSparkOpenModel) -> int:
    """Re-class every layer's projection-gather MoE as the prologue MoE; build its inbox.

    The inbox rows carry the pool slice and the logit block: 512 words at TP8,
    1024 at TP4 (``pool_gather_route.row_words``).

    Returns the number of layers installed: 0, on every rank alike, when the
    pool site is not fused (``projection_gathers.fusable_sites`` agrees on that
    over the TP group). Collective over the TP group (``agreed_build`` and the
    inbox, which is fabric memory).

    Raises:
        ValueError: if a layer's MoE is not the projection-gather class, or (on every rank) if
            some rank cannot build the routed kernel.
    """
    layers = list(model.layers)
    for layer in layers:
        moe = layer.feed_forward
        if type(moe) is not MuseSparkOpenTepGatherMoE:
            raise ValueError(
                "[musespark_open] moe_prologue extends the fused pool site (MuseSparkOpenTepGatherMoE); "
                f"layer {layer.layer_idx} has {type(moe).__name__}"
            )
    if any(getattr(layer.feed_forward, "_pool_inbox", None) is None for layer in layers):
        logger.warning(
            "[musespark_open] moe_prologue: the pool site is not fused; every layer keeps its path"
        )
        return 0
    mapping = layers[0].mapping
    if not agreed_build(mapping):
        raise ValueError(
            "[musespark_open] moe_prologue needs CuTe DSL and flashinfer's RMSNormFP4QuantKernel on every TP rank"
        )
    world = mapping.tp_size
    inbox = ProjectionInbox(mapping, row_words=_route.row_words(world))
    enable_pdl = LAUNCH_PDL
    for layer in layers:
        moe = layer.feed_forward
        moe.__class__ = MuseSparkOpenTepPrologueMoE
        moe._prologue_inbox = inbox
        moe._prologue_enable_pdl = bool(enable_pdl)
    logger.info(
        f"[musespark_open] moe_prologue: one-chain MoE prologue on the decode path of all {len(layers)} layers "
        f"(TP{world} router shard of {_shard.local_experts(world)} experts, routing and any "
        "replicated-expert placement "
        f"in the pool kernel; steps of at most {MAX_ROWS} rows; PDL {'on' if enable_pdl else 'off'})"
    )
    return len(layers)


def install_router_fork(model: MuseSparkOpenModel) -> int:
    """``router_fork``: give every prologue MoE layer the device's side stream for its router shard.

    Called after ``install`` re-classed the layers. Refuses, on every TP rank
    alike (``tp_agreement.agreed_refusal``, collective over the TP group), when
    a layer is not the prologue MoE, or when a rank could not create the stream.
    Returns the number of layers given the stream.

    Raises:
        ValueError: naming the refusal and, when not every rank gave it, the ranks.
    """
    layers = list(model.layers)
    reason = None
    stream = None
    wrong = [
        layer.layer_idx
        for layer in layers
        if type(layer.feed_forward) is not MuseSparkOpenTepPrologueMoE
    ]
    if wrong:
        reason = f"[musespark_open] router_fork forks the router shard; layers {wrong} do not run the prologue"
    else:
        try:
            stream = dedicated_stream()
        except RuntimeError as error:
            reason = f"[musespark_open] router_fork could not create its side stream: {error}"
    reason = _tp_agreement.agreed_refusal(layers[0].mapping if layers else None, reason)
    if reason is not None:
        raise ValueError(reason)
    for layer in layers:
        layer.feed_forward._router_stream = stream
    logger.info(
        f"[musespark_open] router_fork: side stream set on {len(layers)} prologue MoE layers; where a decode step of "
        "at "
        f"most {MAX_ROWS} rows takes the routed prologue (the layers whose expert input "
        "the pool kernel quantizes, i.e. the NVFP4-expert ones), the router shard runs on "
        "it as a plain launch while the pool GEMM follows the attention hop on the "
        "programmatic chain, joined before the routed pool kernel"
    )
    return len(layers)
