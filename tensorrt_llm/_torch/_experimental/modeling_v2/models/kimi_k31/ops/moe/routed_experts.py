# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts of one MoE layer: the MXFP4 expert weights in the TRTLLM-Gen runner layout and the call.

`K31RoutedExperts` holds this rank's expert window (whole experts under MoE EP, the intermediate shard of each expert
under MoE TP) in the runner layout the expert loader packs (``mxfp4_expert_pack``), and runs a routed call the way
the K3.1 targets serve it:

1. under attention DP, a step where some rank has no rows pads every rank's rows to the step's largest count (zero
   rows), so no rank enters the token exchange without rows;
2. the call runs in ``ceil(rows / moe_config.max_num_tokens)`` chunks (``rows``: the exchange's ``ranks x largest
   count`` under attention DP, else this rank's rows), each rank's rows split as evenly as possible. Under attention DP
   a rank with an empty chunk runs its first chunk again in its place (every rank must enter every exchange) and drops
   that output. Without the one-sided all-to-all, chunks alternate between the chunk-overlap stream (even chunks) and
   the caller's stream;
3. per chunk: the routing and MXFP8 input (the target's fused route + quantize where it serves, else the target's
   separated routing and the input quantization; under attention TP the general target routes inside the runner
   instead), the route-capture and expert-statistic hooks, the token exchange under attention DP (the expert output
   written straight into the one-sided all-to-all's combine buffer), this rank's experts, the combine;
4. under attention DP the output keeps this rank's rows.

Under per-layer dynamic EPLB (``moe_config.load_balancer`` ``mode: per_layer_dynamic``, the engine's
``per_layer_dynamic_eplb`` package; a target passes the model's manager, see :class:`K31RoutedExperts`) the routed
experts keep their ``H = experts / EP`` resident experts and gain ``S = (num_slots - experts) / EP`` helper slots per
rank: after loading, the four runner-layout weight tensors move onto the layer's ``H + S`` live-weight arena. Every
chunk's routing goes to the layer's HALO-Q planner (one kernel on the auxiliary SMs of the copy stream: it exchanges the
routing counts of every EP rank, picks the hot experts, rewrites each routed pick to a physical slot ``rank * (H + S) +
slot`` and copies the hot experts' weights into the helper slots of their ranks over NVLink multicast); the exchange
dispatches the physical slots (``num_slots`` routes to ranks by ``slot // (H + S)``) after the routes are written, and
the experts read their ``H + S`` slots once every rank published its helper copies.

The experts are :func:`k31_routed_experts`: one rank's pre-routed MXFP4 x MXFP8 call of up to :data:`MAX_ROWS` rows
on the CuTe DSL grouped GEMMs (:data:`.w4a8_moe_cute.k31_w4a8_moe_cute`: its own sort, FC1 + SiTU + MXFP8 requant +
FC2 and the finalize, bitwise to the TRTLLM-Gen runner, with ``pick_tiling``'s kernel); larger calls (prefill chunks),
calls under CUDA-graph capture before the first eager call allocated the workspace, and padded intermediate shards
take the runner (:func:`.w4a8_moe_runner.k31_runner_moe`: the pinned tactic of the table, else the autotuned engine
op). A target may replace the routing and expert entries (:meth:`K31RoutedExperts.route`, ``_experts``). The MoE-TP
all-reduce handle (attention TP over more than one rank) is built here; the caller runs the reduction.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import dispatch_issued
from tensorrt_llm._torch.route_capture import get_active_route_capture
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.logger import logger

from ..comm.allreduce_module import tp_allreduce
from .ep_dispatch_combine import K31EpAllToAll, create_ep_all_to_all
from .moe_prologue import quantize_input, route_quant, separated_routing
from .mxfp4_expert_pack import EXPERT_WEIGHT_DTYPE, expert_weight_shapes, intermediate_size_lean
from .routing_statistic import record_expert_statistic
from .w4a8_moe_cute import (
    HIDDEN,
    k31_w4a8_moe_cute,
    max_tiles_for,
    moe_workspace_bytes,
    pick_n_tile,
    pick_tiling,
)
from .w4a8_moe_runner import NUM_EXPERTS, k31_runner_moe

#: Experts per token (K3.1).
TOP_K = 16

Routed = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]

#: The runner-layout tensors indexed by local expert that move onto the per-layer dynamic EPLB live-weight arena (the
#: SiTU caps are the same for every expert and are sized to the slots instead).
LIVE_WEIGHT_PLANES = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")


#: Rows of a pre-routed call the CuTe DSL routed experts serve: attention-DP decode steps (up to 256 rows per
#: rank over 16 ranks); prefill chunks keep the runner, whose FC2 is faster at their size.
MAX_ROWS = 4096

#: Largest row count of the 1-CTA kernel (``pick_tiling`` with TP-split experts); its zeroed workspace is sized for
#: it (the CTA-pair kernel takes none).
_ONE_CTA_MAX_ROWS = 511

_WORKSPACES: Dict[Tuple[Optional[int], int, int, int], torch.Tensor] = {}


def _workspace(device: torch.device, inter: int, local: int, slot: int) -> Optional[torch.Tensor]:
    """The zeroed workspace of ``slot`` per device and expert shape, shared by every layer (the op re-arms it on each
    call, so two calls that may run at the same time take different slots); the first eager call allocates it, and a
    call under CUDA-graph capture before that gets None."""
    key = (device.index, inter, local, slot)
    ws = _WORKSPACES.get(key)
    if ws is None and not torch.cuda.is_current_stream_capturing():
        size = max(
            moe_workspace_bytes(max_tiles_for(m, local, pick_n_tile(m)), inter, pick_n_tile(m))
            for m in (47, _ONE_CTA_MAX_ROWS)
        )
        ws = _WORKSPACES[key] = torch.zeros(size, dtype=torch.uint8, device=device)
    return ws


def _cute_serves(
    x: torch.Tensor, w3_w1_weight: torch.Tensor, inter: int, valid_inter: int, local: int
) -> bool:
    return (
        x.dtype == torch.float8_e4m3fn
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.shape[1] == HIDDEN
        and valid_inter == inter
        and w3_w1_weight.shape[0] == local
        and w3_w1_weight.shape[1] == 2 * inter
        and inter % 128 == 0
    )


def k31_routed_experts(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    intermediate_size: int,
    valid_intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    gate_cap: float,
    linear_cap: float,
    tune_max_num_tokens: int,
    use_dp: bool,
    output: Optional[torch.Tensor] = None,
    workspace_slot: int = 0,
    num_experts: int = NUM_EXPERTS,
) -> torch.Tensor:
    """One rank's routed experts of a pre-routed call: ``[rows, 3584]`` BF16 ``sum_k topk_weights[r, k] *
    expert_k(x_r)`` over this rank's experts (``local_expert_offset`` .. ``+ local_num_experts`` of the id space
    ``num_experts``: the routed experts, or under per-layer dynamic EPLB the physical expert slots).

    ``x`` E4M3 ``[rows, 3584]``, ``x_sf`` UE8M0 ``[rows, 112]`` linear, ``topk_ids`` int32 / ``topk_weights`` BF16
    ``[rows, 16]`` with global expert ids; the weights are the runner layout (``w3_w1_weight`` ``[local, 2 *
    intermediate_size, 1792]`` and the rest); ``intermediate_size`` the padded per-rank width and
    ``valid_intermediate_size`` its unpadded part; ``gemm1_alpha`` / ``gemm1_beta`` the per-local-expert FP32 SiTU caps
    the runner reads (``gate_cap`` / ``linear_cap`` as floats for the CuTe DSL kernels); ``output`` a ``[rows, 3584]``
    BF16 buffer the experts write and return; ``workspace_slot`` the CuTe DSL workspace of the call (calls that can
    overlap on two streams take different slots)."""
    ws = None
    if _cute_serves(x, w3_w1_weight, intermediate_size, valid_intermediate_size, local_num_experts):
        ws = _workspace(x.device, intermediate_size, local_num_experts, workspace_slot)
    if ws is None:
        return k31_runner_moe(
            x,
            x_sf,
            topk_ids,
            topk_weights,
            w3_w1_weight,
            w3_w1_weight_scale,
            gemm1_alpha,
            gemm1_beta,
            w2_weight,
            w2_weight_scale,
            intermediate_size,
            valid_intermediate_size,
            local_expert_offset,
            local_num_experts,
            tune_max_num_tokens,
            use_dp,
            output,
            num_experts=num_experts,
        )
    n_tile, pair = pick_tiling(x.shape[0], intermediate_size)
    logger.info_once(
        f"Kimi K3.1 routed experts on the CuTe DSL W4A8 GEMMs ({local_num_experts} local experts)",
        key="k31_routed_experts_cute",
    )
    return k31_w4a8_moe_cute(
        x,
        x_sf,
        topk_ids,
        topk_weights,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        local_expert_offset,
        local_num_experts,
        n_tile,
        ws,
        gate_cap=float(gate_cap),
        linear_cap=float(linear_cap),
        output=output,
        pair=pair,
        num_experts=num_experts,
    )


def dense_dwdp_defers_routed_experts() -> bool:
    """Whether the routed experts load after the dense weights are sharded: a standalone dense DWDP manager is
    registered (``create_py_executor`` registers it before the model is built and sets dense DWDP up right after the
    load, before the deferred weights load). With expert DWDP (the attached mode) its owner places the experts."""
    from tensorrt_llm._torch.modules.dwdp.dense.manager import get_global_dense_dwdp_manager
    from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode

    manager = get_global_dense_dwdp_manager()
    return manager is not None and manager.mode is DenseMode.STANDALONE


def split_chunk(rows: int, chunks: int) -> List[int]:
    """``rows`` split into ``chunks`` chunks as evenly as possible (the larger chunks first)."""
    size, extra = divmod(rows, chunks)
    return [size + 1] * extra + [size] * (chunks - extra)


def num_chunks(all_rank_num_tokens: List[int], moe_max_num_tokens: int, exchanges: bool) -> int:
    """Chunks of a call under the MoE token budget ``moe_max_num_tokens``: of the exchange's ``ranks x largest count``
    rows where tokens are exchanged (attention DP), else of this rank's rows."""
    if exchanges:
        rows = len(all_rank_num_tokens) * max(all_rank_num_tokens)
    else:
        assert len(all_rank_num_tokens) == 1, all_rank_num_tokens
        rows = all_rank_num_tokens[0]
    return (rows + moe_max_num_tokens - 1) // moe_max_num_tokens


class ChunkPlan(NamedTuple):
    #: This rank's rows per chunk.
    sizes: List[int]
    #: Per chunk, every rank's row count the exchange is given (None without an exchange).
    exchange_rows: List[Optional[List[int]]]
    #: Per chunk, whether its output is this rank's (False: an empty chunk that ran the rank's first chunk again).
    used: List[bool]


def chunk_plan(
    rows: int,
    chunks: int,
    rows_per_rank: Optional[List[int]] = None,
    rank: int = 0,
    one_sided: bool = False,
) -> ChunkPlan:
    """How a call of ``rows`` rows runs in ``chunks`` chunks. Under attention DP (``rows_per_rank``: every rank's
    rows, ``rank``: this one) every rank's rows split alike; the one-sided all-to-all (``one_sided``) cannot take a
    rank without rows, so a 0 count is given as 1 first. A rank's empty chunk then runs its first chunk again, and
    every rank's exchange counts say so (they must agree on every rank)."""
    if rows_per_rank is None:
        return ChunkPlan(split_chunk(rows, chunks), [None] * chunks, [True] * chunks)
    per_rank = [split_chunk(r, chunks) for r in rows_per_rank]
    exchange_rows = [[sizes[c] for sizes in per_rank] for c in range(chunks)]
    if one_sided:
        exchange_rows = [[1 if r == 0 else r for r in counts] for counts in exchange_rows]
    sizes = per_rank[rank]
    assert sizes[0] != 0, "chunk 0 shouldn't be empty"
    for c in range(chunks):
        for r, own in enumerate(per_rank):
            if own[c] == 0:
                exchange_rows[c][r] = own[0]
    return ChunkPlan(sizes, exchange_rows, [n != 0 for n in sizes])


class K31RoutedExperts(nn.Module):
    """One MoE layer's routed experts on this rank; see the module docstring.

    A target may override :meth:`route` (the fused route + quantize, or None where it declines) and ``_experts``
    (this rank's experts on routed MXFP8 rows, written into ``output`` where given), and set ``_routes_in_kernel``
    where the runner routes from the logits without a token exchange.

    ``load_balancer``: the model's per-layer dynamic EPLB manager (``PerLayerDynamicMoeLoadBalancer``, the engine's
    ``get_moe_load_balancer()`` while the model is built), or None. With one, the layer registers its handle
    (``layer_load_balancer``) at construction; the experts must be exchanged over a pure-EP attention-DP group. A
    target that overrides ``_experts`` takes its ``replica_plan`` argument (:meth:`_expert_window`).
    """

    #: Without a token exchange the experts route inside the runner from the logits (no separated routing).
    _routes_in_kernel = False

    def __init__(
        self,
        model_config,
        mapping,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        gate_cap: float,
        linear_cap: float,
        layer_idx: int,
        e_score_correction_bias: Callable[[], torch.Tensor],
        routed_scaling_factor: float,
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        load_balancer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mapping = mapping
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size, self.tp_rank = mapping.moe_tp_size, mapping.moe_tp_rank
        local = num_experts // mapping.moe_ep_size
        self.expert_size_per_partition = local
        self.slot_start = mapping.moe_ep_rank * local
        self.initial_local_expert_ids = list(range(self.slot_start, self.slot_start + local))
        self.use_dp = mapping.enable_attention_dp
        # Per-layer dynamic EPLB: this layer's handle of the model's manager, the routed id space (the physical expert
        # slots of every rank) and the slots this rank computes (resident experts, then helper slots).
        self.layer_load_balancer = None
        self.num_slots = num_experts
        self.compute_slot_count = local
        # Read by the routed calls under EPLB: a function the next replica plan runs beside its HALO-Q kernel
        # (:meth:`replica_plan_overlap`) and the model's sampled load record (``ops/moe/replica_balance``).
        self._replica_overlap_fn: Optional[Callable[[], None]] = None
        self.balance_log = None
        if load_balancer is not None:
            self._register_load_balancer(load_balancer)
        self.tune_max_num_tokens = model_config.max_num_tokens
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        self.caps = (float(gate_cap), float(linear_cap))
        self._bias = e_score_correction_bias
        self.routed_scaling_factor = float(routed_scaling_factor)
        # The runner layout of this rank's experts (packed per slot by the expert loader).
        shapes = expert_weight_shapes(local, intermediate_size, hidden_size, self.tp_size)
        for name, shape in shapes._asdict().items():
            self.register_parameter(
                name,
                nn.Parameter(torch.empty(shape, dtype=EXPERT_WEIGHT_DTYPE), requires_grad=False),
            )
        # Slots the expert loader filled (it checks them all after the load).
        self._packed_mxfp4_loaded_slots: set = set()
        self.intermediate_size_padded = self.w3_w1_weight.shape[1] // 2
        self.intermediate_size_lean = intermediate_size_lean(
            intermediate_size, self.tp_size, self.tp_rank
        )
        # Per-local-expert SiTU caps the runner reads (FP32), filled once the weights are on their device.
        self.act_alpha = self.act_beta = None
        # The direct MoE-TP latent reduction the caller runs (attention TP over more than one rank).
        self.all_reduce = (
            tp_allreduce(mapping, model_config.allreduce_strategy, torch.bfloat16)
            if not self.use_dp and mapping.tp_size > 1
            else None
        )
        # The token exchange over the EP group under attention DP.
        # The most rows one rank routes in one chunk: the HALO-Q planner's route capacity under EPLB.
        self.max_tokens_per_rank_capacity = model_config.max_num_tokens
        # Under EPLB the exchange routes physical slots: slot ``s`` lives on rank ``s // compute_slot_count``.
        self.comm = (
            create_ep_all_to_all(
                mapping,
                num_slots=self.num_slots,
                top_k=TOP_K,
                max_num_tokens_per_rank=model_config.max_num_tokens,
                hidden_size=hidden_size,
                dtype=model_config.torch_dtype,
                use_low_precision_combine=model_config.use_low_precision_moe_combine,
            )
            if self.use_dp and mapping.dp_size > 1
            else None
        )
        # The chunk-overlap stream of a call split into chunks (a MoE token budget below max_num_tokens x dp).
        self._chunk_stream = None
        if self.moe_max_num_tokens < model_config.max_num_tokens * mapping.dp_size:
            self._chunk_stream = (
                aux_stream_dict[AuxStreamType.MoeChunkingOverlap]
                if aux_stream_dict is not None
                else torch.cuda.Stream()
            )
            self._chunk_events = (torch.cuda.Event(), torch.cuda.Event())

    # -- per-layer dynamic EPLB -------------------------------------------------------------------------------------

    def _register_load_balancer(self, manager: Any) -> None:
        """Register this layer with the model's per-layer dynamic EPLB ``manager`` (collective-free; the arena and the
        planner are built after loading, :meth:`_bind_live_weight_planes`)."""
        mapping = self.mapping
        if not (self.use_dp and mapping.dp_size > 1 and mapping.moe_tp_size == 1):
            raise ValueError(
                "Kimi K3.1 per-layer dynamic EPLB plans the routed rows the experts exchange: it needs attention "
                f"DP over a pure-EP group (dp_size {mapping.dp_size}, moe_tp {mapping.moe_tp_size})"
            )
        layer = manager.add_layer(self.num_experts, TOP_K, manager.num_local_slots)
        if layer.resident_experts_per_rank != self.expert_size_per_partition:
            raise ValueError(
                f"Kimi K3.1 per-layer dynamic EPLB: {layer.resident_experts_per_rank} resident experts per rank, the "
                f"layer holds {self.expert_size_per_partition}"
            )
        self.layer_load_balancer = layer
        self.compute_slot_count = int(layer.compute_slots_per_rank)
        self.num_slots = self.compute_slot_count * mapping.moe_ep_size

    @property
    def ep_process_group(self) -> None:
        """No torch EP process group: the EPLB runtime forms an MPI communicator of the EP ranks from ``mapping``."""
        return None

    def _bind_live_weight_planes(self) -> None:
        """Move the runner-layout expert tensors onto the layer's ``H + S`` live-weight arena and build the layer's
        HALO-Q / copy producer (``per_layer_dynamic_eplb.live_planes``). Collective over the EP group: every rank binds
        the same layers in the same order (the loader's post-load walk). Once bound, a later call only checks that
        the tensors still are the arena's views."""
        from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import (
            bind_live_weight_planes,
        )

        layer = self.layer_load_balancer
        arena = layer.live_weight_arena
        if arena is not None:
            bound = all(
                getattr(self, name).data_ptr() == view.data_ptr()
                for name, view in zip(LIVE_WEIGHT_PLANES, arena.tekit_alias_views)
            )
            if not bound:
                raise RuntimeError(
                    f"Kimi K3.1 layer {self.layer_idx}: the per-layer dynamic EPLB arena holds the weights loaded "
                    "first; reloaded expert weights cannot be rebound to it"
                )
            return
        bind_live_weight_planes(
            self,
            LIVE_WEIGHT_PLANES,
            hidden=self.hidden_size,
            intermediate=self.intermediate_size_padded,
            ep_process_group=self.ep_process_group,
        )
        # The replaced tensors are free: return them to the driver before the next layer's arena allocation.
        torch.cuda.empty_cache()
        layer.bind_backend(self)

    @contextlib.contextmanager
    def replica_plan_overlap(self, fn: Callable[[], None]) -> Iterator[None]:
        """Run ``fn`` inside the next replica plan's route, right after its HALO-Q + helper-weight copy kernel is
        submitted (independent work on the current stream, e.g. the shared expert, then overlaps the planning and
        the copy). Only the first route consumes it; if none does, the caller still owns that work. Every EP rank
        installs the same function for the same calls (the route is collective)."""
        if not callable(fn):
            raise TypeError("the replica-plan overlap must be callable")
        self._replica_overlap_fn = fn
        try:
            yield
        finally:
            self._replica_overlap_fn = None

    def _plan_replicas(self, ids: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Submit one chunk's logical routes ``ids`` int32 ``[rows, 16]`` to the layer's planner: returns the physical
        slots (a view of the planner's output, written on the copy stream: read it after the plan's
        ``wait_for_routes``) and the replica plan, which the caller finishes after the experts' kernels."""
        overlap_fn, self._replica_overlap_fn = self._replica_overlap_fn, None
        layer = self.layer_load_balancer
        slots = layer.route(ids.contiguous(), overlap_fn=overlap_fn)
        return slots, layer.require_replica_plan()

    def _expert_window(self, replica_plan: Optional[Any]) -> Tuple[int, int]:
        """``(local_expert_offset, local_num_experts)`` of this rank's experts in the routed id space: the resident
        experts, or under a replica plan the rank's ``H + S`` physical slots."""
        if replica_plan is None:
            return self.slot_start, self.expert_size_per_partition
        slots = self.compute_slot_count
        if (
            int(replica_plan.resident_experts_per_rank) != self.expert_size_per_partition
            or int(replica_plan.compute_slots_per_rank) != slots
            or int(self.w3_w1_weight.shape[0]) != slots
        ):
            raise RuntimeError(
                f"Kimi K3.1 layer {self.layer_idx}: the replica plan's resident / compute slots "
                f"{replica_plan.resident_experts_per_rank} / {replica_plan.compute_slots_per_rank} do not match the "
                f"experts' {self.expert_size_per_partition} / {slots} (weight rows {self.w3_w1_weight.shape[0]})"
            )
        return self.mapping.moe_ep_rank * slots, slots

    # -- weights ---------------------------------------------------------------------------------------------------

    def defer_weight_materialization(self) -> None:
        """Keep this rank's expert weights on the meta device through the model load
        (``model_loader.DEFER_MATERIALIZATION_ATTR``): the target allocates and loads them after the standalone dense
        DWDP setup (``load_deferred_weights`` of its ``weights`` module), so that a rank never holds the replicated
        dense weights and its experts at once. Called at construction, while the weights are still on meta."""
        from tensorrt_llm._torch.pyexecutor.model_loader import DEFER_MATERIALIZATION_ATTR

        for param in self.parameters(recurse=False):
            setattr(param, DEFER_MATERIALIZATION_ATTR, True)

    @property
    def weights_deferred(self) -> bool:
        """Whether the expert weights are still on the meta device, waiting for the deferred load."""
        return self.w3_w1_weight.device.type == "meta"

    def cache_derived_state(self) -> None:
        """Fill the SiTU caps on the weights' device (they are configuration, not checkpoint tensors), one per slot the
        experts compute; under per-layer dynamic EPLB then bind the live-weight arena."""
        device = self.w3_w1_weight.device
        slots = self.compute_slot_count
        self.act_alpha = torch.full((slots,), self.caps[0], dtype=torch.float32, device=device)
        self.act_beta = torch.full((slots,), self.caps[1], dtype=torch.float32, device=device)
        if self.layer_load_balancer is not None:
            self._bind_live_weight_planes()

    def post_load_weights(self) -> None:
        # Deferred weights get their derived state once the deferred load has filled them.
        if not self.weights_deferred:
            self.cache_derived_state()

    # -- target entries --------------------------------------------------------------------------------------------

    def route(self, x: torch.Tensor, router_logits: torch.Tensor) -> Optional[Routed]:
        """``(ids, weights, x_fp8, x_sf)`` of a call from the MoE prologue (the runner's routing weights under
        attention TP), or None where it declines."""
        return route_quant(
            router_logits, self._bias(), self.routed_scaling_factor, x, not self.use_dp
        )

    def _experts(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        ids: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        workspace_slot: int = 0,
        replica_plan: Optional[Any] = None,
    ) -> torch.Tensor:
        """This rank's experts on pre-routed MXFP8 rows (``router_logits`` is for targets that route in the
        runner; ``workspace_slot`` the CuTe DSL workspace, one per chunk stream; ``replica_plan``: the call's
        per-layer dynamic EPLB plan, ``ids`` are then physical slots)."""
        local_offset, local_num = self._expert_window(replica_plan)
        return k31_routed_experts(
            x_fp8,
            x_sf,
            ids,
            weights,
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w2_weight,
            self.w2_weight_scale,
            self.act_alpha,
            self.act_beta,
            self.intermediate_size_padded,
            self.intermediate_size_lean,
            local_offset,
            local_num,
            *self.caps,
            self.tune_max_num_tokens,
            self.use_dp,
            output,
            workspace_slot,
            num_experts=self.num_slots,
        )

    # -- the call --------------------------------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
        routed: Optional[Routed] = None,
    ) -> torch.Tensor:
        """The routed experts of ``x`` (the routed latent ``[rows, 3584]`` BF16) routed by ``router_logits`` (FP32
        ``[rows, 896]``): this rank's partial ``[rows, 3584]`` BF16 under attention TP, this rank's rows of the combined
        output under attention DP. ``all_rank_num_tokens``: every rank's row count this step (attention DP).
        ``routed``: the ``(ids, weights, x_fp8, x_sf)`` a caller already computed for this one-chunk call (the MoE
        front; then neither ``x`` nor the logits are read)."""
        if all_rank_num_tokens is None:
            all_rank_num_tokens = [x.shape[0]]
        rows_per_rank = all_rank_num_tokens
        chunks = num_chunks(rows_per_rank, self.moe_max_num_tokens, self.comm is not None)
        use_dp_padding = None
        max_rows = max(all_rank_num_tokens)
        if self.comm is not None and max_rows > 0 and any(t == 0 for t in rows_per_rank):
            # A rank without rows would stall the exchange: every rank pads its rows to the largest count.
            use_dp_padding = True
            rows_per_rank = [max_rows] * len(all_rank_num_tokens)
            if x.shape[0] < max_rows:
                pad = max_rows - x.shape[0]
                x = torch.cat([x, x.new_zeros((pad, x.shape[1]))], dim=0)
                router_logits = torch.cat(
                    [router_logits, router_logits.new_zeros((pad, router_logits.shape[1]))], dim=0
                )
        if chunks == 1:
            out = self._chunk(x, router_logits, rows_per_rank, use_dp_padding, 0, routed)
        else:
            assert routed is None, "a pre-routed call is one chunk"
            out = self._chunks(x, router_logits, chunks, rows_per_rank, use_dp_padding)
        if self.use_dp and self.mapping.tp_size > 1:
            out = out[: all_rank_num_tokens[self.mapping.tp_rank]]
        return out

    def _chunk(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        rows_per_rank: List[int],
        use_dp_padding: Optional[bool],
        row_offset: Optional[int],
        routed: Optional[Routed] = None,
        workspace_slot: int = 0,
    ) -> torch.Tensor:
        """One chunk: routing, hooks, quantization, exchange, experts, combine. ``row_offset``: the chunk's first row
        in the call (route capture), None for a chunk that repeats another one's rows; ``workspace_slot``: the
        experts' workspace (one per chunk stream).

        Under per-layer dynamic EPLB the chunk's logical routes go to the layer's planner right after the routing
        hooks (which see the logical experts); the input quantization (where the routing did not produce it) runs
        beside the planner, the exchange of the physical slots waits for the routes, and the experts wait for every
        rank's helper copies, after the dispatch (the work queued behind it, ``dispatch_issued``, does not wait).
        The plan is released once the experts' kernels are queued."""
        if self._routes_in_kernel and self.comm is None:
            x_fp8, x_sf = quantize_input(x)
            return self._experts(
                x_fp8, x_sf, None, None, router_logits=router_logits, workspace_slot=workspace_slot
            )
        if routed is None:
            routed = self.route(x, router_logits)
        if routed is None:
            ids, weights = separated_routing(
                router_logits, self._bias(), self.routed_scaling_factor, self.use_dp
            )
            x_fp8 = x_sf = None
        else:
            ids, weights, x_fp8, x_sf = routed
        ids = ids.to(torch.int32)
        if row_offset is not None:
            route_capture = get_active_route_capture()
            if route_capture is not None:
                route_capture.capture(self.layer_idx, ids, row_offset)
        weights = weights.to(torch.bfloat16)
        record_expert_statistic(self.layer_idx, self.num_experts, ids)
        if self.layer_load_balancer is not None:
            return self._replica_chunk(x, ids, weights, x_fp8, x_sf, rows_per_rank, use_dp_padding)
        if x_fp8 is None:
            x_fp8, x_sf = quantize_input(x)
        if self.comm is None:
            return self._experts(x_fp8, x_sf, ids, weights, workspace_slot=workspace_slot)
        x_fp8, x_sf, ids, weights = self.comm.dispatch(
            x_fp8, x_sf, ids, weights, rows_per_rank, use_dp_padding
        )
        output = None
        if isinstance(self.comm, K31EpAllToAll):
            # The experts write straight into the combine buffer of the one-sided all-to-all.
            output = self.comm.combine_payload(self.hidden_size, torch.bfloat16)
        y = self._experts(x_fp8, x_sf, ids, weights, output=output, workspace_slot=workspace_slot)
        return self.comm.combine(y, max(rows_per_rank), payload_in_workspace=output is not None)

    def _replica_chunk(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        x_fp8: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor],
        rows_per_rank: List[int],
        use_dp_padding: Optional[bool],
    ) -> torch.Tensor:
        """:meth:`_chunk` past the routing hooks under per-layer dynamic EPLB (see there): plan, exchange the physical
        slots, run the rank's ``H + S`` slots, release the plan, combine. Chunks stay on the stream the planner
        bound (the caller's)."""
        slots, plan = self._plan_replicas(ids)
        if x_fp8 is None:
            x_fp8, x_sf = quantize_input(x)
        # The dispatch is the first reader of the physical slots.
        plan.wait_for_routes()
        x_fp8, x_sf, recv_slots, weights = self.comm.dispatch(
            x_fp8, x_sf, slots, weights, rows_per_rank, use_dp_padding
        )
        dispatch_issued()
        if self.balance_log is not None:
            # Beside the helper-weight copy, off the dispatch's path.
            self.balance_log.record(self.layer_load_balancer.layer_idx, ids, slots)
        output = None
        if isinstance(self.comm, K31EpAllToAll):
            output = self.comm.combine_payload(self.hidden_size, torch.bfloat16)
        # The helper slots this rank computes hold their experts once every rank published its copies.
        plan.wait_for_helper_weights()
        y = self._experts(x_fp8, x_sf, recv_slots, weights, output=output, replica_plan=plan)
        self.layer_load_balancer.finish_replica_plan(plan)
        return self.comm.combine(y, max(rows_per_rank), payload_in_workspace=output is not None)

    def _chunks(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        chunks: int,
        rows_per_rank: List[int],
        use_dp_padding: Optional[bool],
    ) -> torch.Tensor:
        """A call in ``chunks`` chunks (module docstring, step 2)."""
        one_sided = isinstance(self.comm, K31EpAllToAll)
        plan = chunk_plan(
            x.shape[0],
            chunks,
            rows_per_rank if self.comm is not None else None,
            self.mapping.rank,
            one_sided,
        )
        x_chunks = x.split(plan.sizes)
        logit_chunks = router_logits.split(plan.sizes)
        # Every chunk of a per-layer dynamic EPLB layer submits its plan from the stream the planner bound.
        overlap = (
            not one_sided and self._chunk_stream is not None and self.layer_load_balancer is None
        )
        if overlap:
            main_event, chunk_event = self._chunk_events
            main_event.record()
            with torch.cuda.stream(self._chunk_stream):
                main_event.wait()
        outputs = []
        row_offset = 0
        for c in range(chunks):
            # an empty chunk runs the first chunk's rows again, uncaptured, and its output is dropped
            own = c if plan.used[c] else 0
            args = (
                x_chunks[own],
                logit_chunks[own],
                plan.exchange_rows[c],
                use_dp_padding,
                row_offset if plan.used[c] else None,
            )
            if overlap and c % 2 == 0:
                with torch.cuda.stream(self._chunk_stream):
                    y = self._chunk(*args, workspace_slot=0)
            else:
                # the two streams' chunks can run at the same time: each stream has its own workspace
                y = self._chunk(*args, workspace_slot=1 if overlap else 0)
            row_offset += plan.sizes[c]
            if plan.used[c]:
                outputs.append(y)
        if overlap:
            with torch.cuda.stream(self._chunk_stream):
                chunk_event.record()
            chunk_event.wait()
        return torch.cat(outputs)
