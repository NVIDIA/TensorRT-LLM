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
three-window one (``PLACE_BALANCE3``) and is recorded with its 96-expert window; on a
layer ``half_window_experts`` gave the half window (TP4) it is ``PLACE_HALF``, recorded
with its 96-expert window likewise.

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

# ---- edge_bf16_moe_front: the BF16-expert layers' decode front through the routed chain ----
#: Part of the ``edge_bf16_moe_front`` fast path (``install(bf16_front=True)``, which gives the
#: layers their ``_pf_unit_scale``). The two layers the checkpoint leaves with BF16 experts (0 and
#: 61) have no NVFP4 input quantization, so ``_pool_quant_args`` declines them and their decode
#: front is the built-in two-stream one: the replicated FP32 router over its whole 8 MiB weight on
#: every rank, the routing kernel and the ring-buddy placement on one stream, the pool GEMM, the
#: built-in column gather and the pre-expert RMSNorm on the other. With the fast path a decode step
#: of at most 8 rows of such a layer runs the NVFP4 layers' routed chain instead:
#:
#: * ``1``: the router shard (``ops/moe/router_shard``, on the router-fork side stream when
#:   installed), the pool GEMM, then ``ops/comm/pool_gather_route`` for the pool gather (its
#:   ``gathered`` row), the logit exchange, the routing and the placement; its NVFP4 outputs are
#:   not read. The pre-expert norm is then the built-in module on the gathered row.
#:   Value-identical by construction: every output the experts and the finalize read is a
#:   certified bitwise reproduction of the built-in chain (the router columns, the routing ids and
#:   FP32 weights, the placement's local ids and BF16 weights, the gathered row), and the norm is
#:   the built-in call on the same row.
#: * ``2``: as ``1``, with the pre-expert norm computed inside the routed kernel
#:   (``pool_gather_route(..., bf16_norm=weight)``): the gathered row's sum of squares is the one
#:   the kernel already forms for the quantization, which is flashinfer's ``row_reduce`` over the
#:   same 64-thread row geometry as flashinfer's ``RMSNormKernel`` at this width, and the output is
#:   ``bf16((x * rstd) * (w + 1))`` as that kernel computes it (``ops/comm/pool_gather_route``).
#:
#: ``0`` keeps the built-in front. Every rank takes the same branch (shapes, dtypes and
#: construction-time state only). The layer's ``_pf_unit_scale``, set only by
#: ``install(model, bf16_front=True)`` (the ``edge_bf16_moe_front`` fast path), is what turns the
#: front on: without it ``_pf_bf16_site`` is False and both modes keep the built-in front.
PF_BF16_FRONT = "2"


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
    if getattr(state, "half", False):
        return _route.PLACE_HALF, local, state
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
    #: ``wide_prologue``: the TP4 wide inbox (``WIDE_MAX_ROWS`` rows) of the 16..32-row
    #: prologue, or None (those steps keep the built-in prepare).
    _wide_inbox: Optional[ProjectionInbox] = None
    #: ``wide_prologue``: whether the 16..32-row router shard gives each row its
    #: one-request bits (``router_shard_rows``) instead of the built-in tensor-core
    #: router's (``router_shard_wide``); the ``wide_router_per_row`` lever.
    _wide_router_rows: bool = False
    #: ``wide_prologue``: whether ``pool_into_route`` and ``unpool_into_boundary`` also
    #: build their 16..32-row direct transports; the ``wide_direct_transports`` lever.
    _wide_direct: bool = False
    #: ``edge_bf16_moe_front``: the routed kernel's unit global scale on a BF16-expert layer (set
    #: by ``install(bf16_front=True)``), else None.
    _pf_unit_scale: Optional[torch.Tensor] = None

    def prepare_moe_inputs(self, hidden_states: torch.Tensor):
        if self._wide_inbox is not None:
            wide = self._wide_prepare(hidden_states)
            if wide is not None:
                return wide
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
            if (
                self._pool_direct_armed is not None
                and PF_BF16_FRONT == "2"
                and self._pf_bf16_site()
            ):
                # PF_BF16_FRONT 2 on the direct transport (``pool_into_route``, the armed site).
                return self._pool_direct_armed.prepare_bf16(self, hidden_states, warps)
            if PF_BF16_FRONT in ("1", "2") and self._pf_bf16_front_applies(hidden_states):
                return self._pf_bf16_front(hidden_states, warps, fused_norm=PF_BF16_FRONT == "2")
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

    def _wide_router_shard(
        self, hidden_states: torch.Tensor, rank: int, warps: int, out: torch.Tensor, world: int
    ) -> None:
        """This rank's router logit columns of a 16..32-row call into ``out`` (``[rows, L]`` FP32).

        By default ``router_shard_wide``: the built-in 16..32-row router's bits (its TF32x3
        tensor-core split-K and reduce). With ``_wide_router_rows``, ``router_shard_rows`` at
        ``warps``: each row's logits are the built-in small-batch router's for that row in a
        one-request call, one launch and no reduce, but other bits than the built-in
        16..32-row launch's.
        """
        if self._wide_router_rows:
            _shard.router_shard_rows(
                hidden_states,
                self.router.weight,
                rank,
                self.metap_multiplier,
                warps,
                out,
                ranks=world,
            )
        else:
            _shard.router_shard_wide(
                hidden_states, self.router.weight, rank, self.metap_multiplier, out, ranks=world
            )

    def _wide_prepare(self, hidden_states: torch.Tensor):
        """``wide_prologue``: the prologue of a 16..32-row decode step, or None (nothing launched).

        Where the built-in decode would run its tensor-core router and the fused routing on
        a pool-deferring call of 16..32 rows: this rank's router columns
        (``_wide_router_shard``) on the router fork's side stream, the pool slice GEMM on the
        current one, and ``pool_gather_route`` on the
        wide inbox -- the pool gather + quantization, the logit exchange and the routing
        (no placement: above 8 rows the experts are the built-in ones and read only the
        routing memo). The pool slice is this rank's columns of the built-in replicated
        pool projection, by a GEMM of its own shape.
        """
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else 0
        if not _shard.WIDE_MIN_ROWS <= rows <= _shard.WIDE_MAX_ROWS:
            return None
        if self._prologue_router_conditions(hidden_states, _shard.WIDE_MAX_ROWS) is None:
            return None
        inbox = self._wide_inbox
        world = inbox.world
        columns = _route.local_columns(world)
        probe = hidden_states.view(-1)[: rows * columns].view(rows, columns)
        args = self._pool_quant_args(probe, max_rows=inbox.max_rows)
        if args is None:
            return None
        gain, global_scale = args
        # One request's warp count (the built-in launch's at 8 rows), for router_shard_rows.
        warps = _shard.builtin_warps(
            _shard.MAX_ROWS, getattr(self, "_prepare_stream", None) is not None
        )
        if self._pool_direct_armed is not None:
            # pool_into_route's wide site: the same chain with the producers publishing.
            return self._pool_direct_armed.prepare(self, hidden_states, warps, gain, global_scale)
        routing = self.experts.routing_method
        side = self._router_stream
        local = _shard.local_experts(world)
        with self.nvtx("router"):
            main = torch.cuda.current_stream()
            logits_local = torch.empty(
                (rows, local), device=hidden_states.device, dtype=torch.float32
            )
            if side is None:
                self._wide_router_shard(hidden_states, inbox.rank, warps, logits_local, world)
            else:
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    self._wide_router_shard(hidden_states, inbox.rank, warps, logits_local, world)
        with self.nvtx("pool"):
            projected = self._replicated_projection(
                self.pooler, hidden_states, self._pool_gather_workspace
            )
            if side is not None:
                main.wait_stream(side)
            bias = routing.e_score_correction_bias
            _require_aligned(pooled=projected, gain=gain, bias=bias, logits=logits_local)
            out = _route.pool_gather_route(
                projected,
                logits_local,
                inbox,
                gain,
                global_scale,
                self.pre_expert_norm.variance_epsilon,
                bias,
                routing.renorm_eps,
                _route.PLACE_NONE,
                enable_pdl=self._prologue_enable_pdl,
                return_gathered=False,
            )
        self._prologue_record_routing(out, _route.PLACE_NONE, None, world)
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
            elif place == _route.PLACE_HALF:
                state.placement += (_route.HALF_SPAN,)
            state.last_stats = out.stats

    # ---- edge_bf16_moe_front (PF_BF16_FRONT) ----
    def _pf_bf16_front_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this call of a BF16-expert layer takes the routed chain (``edge_bf16_moe_front``).

        Called after ``_prologue_router_warps`` accepted the call (a pool-deferring decode call
        of at most 8 contiguous BF16 rows, the built-in router and routing) and
        ``_pool_quant_args`` declined it. Requires the BF16 leaf under the replicated experts'
        two-window stacked path (whose experts read the placement the routed kernel records),
        no direct pool transport armed on this layer, the 4096-wide pool and the gemma pre-expert
        norm. Reads construction-time state, shapes and dtypes only, so every rank answers alike.
        """
        return self._pool_direct_armed is None and self._pf_bf16_site()

    def _pf_bf16_site(self) -> bool:
        """``_pf_bf16_front_applies`` without its transport condition (construction-time state).

        Also what makes a BF16-expert layer a direct-transport site (``pool_into_route``'s
        ``_routes_bf16``, ``OB_BF16_DIRECT``).
        """
        state = getattr(self, "_replicated_experts", None)
        norm = self.pre_expert_norm
        weight = getattr(norm, "weight", None)
        return (
            self._prologue_inbox is not None
            and getattr(self, "_pf_unit_scale", None) is not None
            and state is not None
            and state.active
            and state.kind == "bf16"
            and getattr(state, "windows", 2) == 2
            and self.expert_hidden_size == _route.HIDDEN
            and getattr(self, "_pool_gather_workspace", None) is not None
            and tuple(self.pooler.weight.shape) == (_route.HIDDEN, _shard.HIDDEN)
            and getattr(norm, "use_gemma", False)
            and isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16
            and tuple(weight.shape) == (_route.HIDDEN,)
            and weight.is_contiguous()
        )

    def _pf_bf16_front(self, hidden_states: torch.Tensor, warps: int, fused_norm: bool):
        """``edge_bf16_moe_front``: the routed chain for a BF16-expert layer; ``(router_logits, normed pool)``.

        The NVFP4 prologue's order and streams (router shard, pool GEMM, join, routed pool
        kernel), with the routed kernel's ``gathered`` row as the pool gather. ``fused_norm``
        (``PF_BF16_FRONT`` 2): the kernel also writes the pre-expert norm of that
        row; otherwise the built-in ``prepare_expert_input`` normalizes it. The NVFP4 payload the
        kernel forms from ``gain`` and a unit global scale is not read.
        """
        routing = self.experts.routing_method
        inbox = self._prologue_inbox
        world = inbox.world
        side = self._router_stream
        rows = hidden_states.shape[0]
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
            # The NVFP4 operands only shape a payload this front does not read (mode 1) or that the
            # kernel does not form (mode 2): the norm's own weight and the install-time unit scale.
            gain = self.pre_expert_norm.weight
            _require_aligned(pooled=projected, gain=gain, bias=bias, logits=logits_local)
            norm_weight = gain if fused_norm else None
            out = _route.pool_gather_route(
                projected,
                logits_local,
                inbox,
                gain,
                self._pf_unit_scale,
                self.pre_expert_norm.variance_epsilon,
                bias,
                routing.renorm_eps,
                place,
                static_from,
                enable_pdl=self._prologue_enable_pdl,
                return_gathered=not fused_norm,
                bf16_norm=norm_weight,
            )
            normed = out.normed if fused_norm else self.prepare_expert_input(out.gathered)
        self._prologue_record_routing(out, place, state, world)
        return out.logits, normed

    # ---- edge_bf16_moe_front: end ----

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
        if self._prologue_router_conditions(hidden_states, MAX_ROWS) is None:
            return None
        overlaps = getattr(self, "_prepare_stream", None) is not None
        return _shard.builtin_warps(hidden_states.shape[0], overlaps)

    def _prologue_router_conditions(self, hidden_states: torch.Tensor, max_rows: int):
        """True when a pool-deferring call of 1..``max_rows`` BF16 rows meets every built-in
        condition of the router launch and the fused routing kernel; else None.

        Reads nothing that differs between ranks.
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
            or not 1 <= hidden_states.shape[0] <= max_rows
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
        return True


def agreed_build(mapping) -> bool:
    """Whether every TP rank can build the routed kernel (collective over the TP group).

    Agreeing first makes every rank raise together when one cannot.
    """
    everyone = Distributed.get(mapping).tp_allgather(bool(_route.available()), small_payload=True)
    return all(everyone)


def install(model: MuseSparkOpenModel, bf16_front: bool = False) -> int:
    """Re-class every layer's projection-gather MoE as the prologue MoE; build its inbox.

    The inbox rows carry the pool slice and the logit block: 512 words at TP8,
    1024 at TP4 (``pool_gather_route.row_words``).

    ``bf16_front``: the ``edge_bf16_moe_front`` fast path (``PF_BF16_FRONT``), whose routed
    kernel reads a unit global scale on the BF16-expert layers.

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
    # edge_bf16_moe_front: the routed kernel's global-scale operand on a BF16-expert layer (its
    # NVFP4 payload is not read there); made here, before any graph capture.
    unit_scale = (
        torch.ones(1, dtype=torch.float32, device=torch.cuda.current_device())
        if bf16_front
        else None
    )
    for layer in layers:
        moe = layer.feed_forward
        moe.__class__ = MuseSparkOpenTepPrologueMoE
        moe._prologue_inbox = inbox
        moe._prologue_enable_pdl = bool(enable_pdl)
        moe._pf_unit_scale = unit_scale
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
        if not isinstance(layer.feed_forward, MuseSparkOpenTepPrologueMoE)
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
