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

The experts are :func:`k31_routed_experts`: one rank's pre-routed MXFP4 x MXFP8 call of up to :data:`MAX_ROWS` rows
on the CuTe DSL grouped GEMMs (:data:`.w4a8_moe_cute.k31_w4a8_moe_cute`: its own sort, FC1 + SiTU + MXFP8 requant +
FC2 and the finalize, bitwise to the TRTLLM-Gen runner, with ``pick_tiling``'s kernel); larger calls (prefill chunks),
calls under CUDA-graph capture before the first eager call allocated the workspace, and padded intermediate shards
take the runner (:func:`.w4a8_moe_runner.k31_runner_moe`: the pinned tactic of the table, else the autotuned engine
op). A target may replace the routing and expert entries (:meth:`K31RoutedExperts.route`, ``_experts``). The MoE-TP
all-reduce handle (attention TP over more than one rank) is built here; the caller runs the reduction.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

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
from .w4a8_moe_runner import k31_runner_moe

#: Experts per token (K3.1).
TOP_K = 16

Routed = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]


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
) -> torch.Tensor:
    """One rank's routed experts of a pre-routed call: ``[rows, 3584]`` BF16 ``sum_k topk_weights[r, k] *
    expert_k(x_r)`` over this rank's experts (``local_expert_offset`` .. ``+ local_num_experts``).

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
    )


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
    where the runner routes from the logits without a token exchange. A target with another expert format overrides
    ``_declare_expert_weights``, ``cache_derived_state``, :meth:`route` and ``_experts`` together.
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
        self.tune_max_num_tokens = model_config.max_num_tokens
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        self.caps = (float(gate_cap), float(linear_cap))
        self._bias = e_score_correction_bias
        self.routed_scaling_factor = float(routed_scaling_factor)
        self._declare_expert_weights()
        # The direct MoE-TP latent reduction the caller runs (attention TP over more than one rank).
        self.all_reduce = (
            tp_allreduce(mapping, model_config.allreduce_strategy, torch.bfloat16)
            if not self.use_dp and mapping.tp_size > 1
            else None
        )
        # The token exchange over the EP group under attention DP.
        self.comm = (
            create_ep_all_to_all(
                mapping,
                num_slots=num_experts,
                top_k=TOP_K,
                max_num_tokens_per_rank=model_config.max_num_tokens,
                hidden_size=hidden_size,
                dtype=model_config.torch_dtype,
                use_low_precision_combine=model_config.use_low_precision_moe_combine,
                workspace_cache=model_config.extra_attrs.setdefault("kimi_k31_ep_workspaces", {}),
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

    def _declare_expert_weights(self) -> None:
        """This rank's expert weights: the MXFP4 runner layout, packed per slot by the expert loader. A target with
        another expert format declares its own."""
        local = self.expert_size_per_partition
        shapes = expert_weight_shapes(local, self.intermediate_size, self.hidden_size, self.tp_size)
        for name, shape in shapes._asdict().items():
            self.register_parameter(
                name,
                nn.Parameter(torch.empty(shape, dtype=EXPERT_WEIGHT_DTYPE), requires_grad=False),
            )
        # Slots the expert loader filled (it checks them all after the load).
        self._packed_mxfp4_loaded_slots: set = set()
        self.intermediate_size_padded = self.w3_w1_weight.shape[1] // 2
        self.intermediate_size_lean = intermediate_size_lean(
            self.intermediate_size, self.tp_size, self.tp_rank
        )
        # Per-local-expert SiTU caps the runner reads (FP32), filled once the weights are on their device.
        self.act_alpha = self.act_beta = None

    def cache_derived_state(self) -> None:
        """Fill the SiTU caps on the weights' device (they are configuration, not checkpoint tensors)."""
        device = self.w3_w1_weight.device
        local = self.expert_size_per_partition
        self.act_alpha = torch.full((local,), self.caps[0], dtype=torch.float32, device=device)
        self.act_beta = torch.full((local,), self.caps[1], dtype=torch.float32, device=device)

    def post_load_weights(self) -> None:
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
    ) -> torch.Tensor:
        """This rank's experts on pre-routed MXFP8 rows (``router_logits`` is for targets that route in the
        runner; ``workspace_slot`` the CuTe DSL workspace, one per chunk stream)."""
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
            self.slot_start,
            self.expert_size_per_partition,
            *self.caps,
            self.tune_max_num_tokens,
            self.use_dp,
            output,
            workspace_slot,
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
        experts' workspace (one per chunk stream)."""
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
        overlap = not one_sided and self._chunk_stream is not None
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
