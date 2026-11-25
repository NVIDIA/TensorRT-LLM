# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized routed experts on the TRTLLM-Gen runner without its routing kernel, as stock's TRTLLM-Gen NVFP4 MoE
does (`moe/fused_moe/trtllm_gen/trtllm_nvfp4.py::try_fused_route_quant`): the routing + NVFP4 quantization launch also
writes the runner's routing tables (`with_routing_tables`, this rank's expert window), and the runner takes them
instead of routing (`routing_tables` / `routing_tiles` of `torch.ops.trtllm.fp4_block_scale_moe_runner`).

Three entries: `dsv4_route_nvfp4_quant_tables` (the launch, quantizing the rows), `dsv4_route_tables` (the launch for
rows the mHC boundary already quantized: stock's `try_fused_route`, a programmatic dependent of the router GEMM that
lets
the expert GEMMs launch while it routes) and `fp4_moe_runner_tables` (the runner on the tables: FC1-only TRTLLM-Gen run
plus the CuTe DSL FC2 when unfinalized, the GEMM grids bounded by the tables, and dynamic expert parallelism).

Dynamic expert parallelism (stock MR !11040, `ConfigurableMoE._dynamic_expert_parallelism_applies`): every rank holds
its MoE-TP shard of all experts and the route launch splits each call's used experts over the two EP ranks home-first
(each keeps up to ceil(used / 2) of its own half's used experts, lowest first, and hands the rest to the other), so the
busier rank streams ceil(used / 2) experts instead of up to all of them; both ranks derive the same split from the same
logits.

Both launches write the tables into a buffer whose rows are 16-byte aligned (`_tables`): the next boundary reads an
unfinalized run's permutation (a table row) as a 16-byte aligned tensor."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route_nvfp4_quant,
)

ROUTING_TABLES_MAX_TOKENS = 8  # the engine's `dsv4_route_quant.ROUTING_TABLES_MAX_TOKENS`
ROUTING_TABLE_TILES = (8, 16, 32)  # the engine's `dsv4_route_quant.ROUTING_TABLE_TILES`
DYNAMIC_EP_MAX_ROWS = (
    ROUTING_TABLES_MAX_TOKENS  # the engine's `TrtllmTrtllmGenNvfp4Impl.dynamic_ep_max_rows`
)
# the engine's `_DYNAMIC_EP_MIN_MOE_TP`: twice the static routed weights per rank, at most a quarter
DYNAMIC_EP_MIN_MOE_TP = 4


def dynamic_ep_applies(
    max_batch_size, tokens_per_step: int, attention_dp: bool, ep: int, moe_tp: int
) -> bool:
    """Whether the target splits each call's experts over its two EP ranks (stock's rule, shapes and config only): a
    batch-1 worker (batch invariance: a request's sums depend only on its own routing) whose decode steps fit the
    route launch's tables, attention TP, MoE EP 2 with MoE TP >= 4 (memory). A setup-free predicate."""
    return (
        max_batch_size == 1
        and 0 < tokens_per_step <= DYNAMIC_EP_MAX_ROWS
        and not attention_dp
        and ep == 2
        and moe_tp >= DYNAMIC_EP_MIN_MOE_TP
    )


def routing_table_max_ctas(
    num_tokens: int, top_k: int, num_experts: int, dynamic_rank: Optional[int]
) -> int:
    """The most CTAs a rank's tables of the two-rank dynamic split assign (the expert GEMMs launch that many: the
    engine's `routing_table_max_ctas`); 0 without the split (the kernels' worst case). A setup-free helper."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import (
        routing_table_max_ctas as max_ctas,
    )

    return max_ctas(
        num_tokens, top_k, num_experts, None if dynamic_rank is None else (dynamic_rank, 2)
    )


_TABLE_ROW_ALIGN = 4  # int32 entries: every table row starts 16 bytes aligned


def _tables(
    num_tokens: int, top_k: int, local_count: int, total_experts: int, tiles: Sequence[int], device
) -> torch.Tensor:
    """The launch's `[len(tiles), 5, width]` tables buffer, `width` the engine's (`_routing_tables`: the widest route
    map,
    read one past its padded rows) rounded up to 4 entries: an unfinalized run's permutation is row 0 of its tile's
    table
    (`table_permutation`), and the boundary kernel that finalizes it reads it as a 16-byte aligned tensor. The engine's
    width alone puts tile i's rows at `i * 5 * width * 4` bytes, which a width not a multiple of 4 leaves unaligned for
    every tile but the first (the launch and the runner take any width: compact `[tiles, 5, width]`)."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import _max_ctas

    width = max(
        [num_tokens * top_k] + [_max_ctas(num_tokens, top_k, local_count, t) * t + 1 for t in tiles]
    )
    width = -(-width // _TABLE_ROW_ALIGN) * _TABLE_ROW_ALIGN
    return torch.empty((len(tiles), 5, width), dtype=torch.int32, device=device)


def _route_launch(
    logits,
    selection_bias,
    x,
    global_scale,
    top_k,
    routed_scaling_factor,
    local_first,
    local_count,
    tiles,
    hidden=None,
    dynamic_rank=None,
):
    """The engine's routing launch with tables (`dsv4_route_nvfp4_quant` with `x`, else `dsv4_route`: each first tries
    the one-cluster `dsv4_route_one`, then the CuTe DSL route kernel), bit for bit the same arguments to the same
    kernels,
    on the 16-byte aligned tables buffer of `_tables`. `dynamic_rank`: the two-rank dynamic split over every expert."""
    from tensorrt_llm._torch.moe.kernels.blackwell import dsv4_route_one as route_one
    from tensorrt_llm._torch.moe.kernels.blackwell import dsv4_route_quant as rq

    n, experts = logits.shape
    dynamic_ep = None if dynamic_rank is None else (dynamic_rank, 2)
    local = None if dynamic_ep is not None else (local_first, local_count)
    local_start, num_local = rq._local_range(local, experts)
    rq._check_dynamic_ep(dynamic_ep, local, True)
    if not routing_tables_fit(n, num_local, experts) or not set(tiles) <= set(ROUTING_TABLE_TILES):
        raise ValueError(
            f"no routing tables for {n} tokens over {num_local} experts, tiles {tiles}"
        )
    dev = logits.device
    tiles = tuple(tiles)
    ids = torch.empty((n, top_k), dtype=torch.int32, device=dev)
    weights = torch.empty((n, top_k), dtype=torch.bfloat16, device=dev)
    tables = _tables(n, top_k, num_local, experts, tiles, dev)
    one = route_one.applies(n, experts, top_k, tiles)
    if x is None:
        if one and route_one.route_one(
            logits,
            selection_bias,
            routed_scaling_factor,
            ids,
            weights,
            tables,
            tiles,
            local_start,
            num_local,
            pdl=rq.TRTLLM_ENABLE_PDL,
            dynamic_ep=dynamic_ep,
        ):
            return ids, weights, tables
        if n > 0:
            rq._compile(
                experts, top_k, routed_scaling_factor, hidden, torch.bfloat16, tiles, quant=False, num_local=num_local,
                pdl=rq.TRTLLM_ENABLE_PDL, cta_scoring=rq._cta_scoring(n, experts), dynamic_ep=dynamic_ep or (0, 0),
            )(logits, selection_bias, None, None, ids, weights, None, None, tables, local_start,
              None, None, None, None, None, None, 0, None, None)  # fmt: skip
        return ids, weights, tables
    hidden = x.shape[1]
    x_fp4 = torch.empty((n, hidden // 8), dtype=torch.int32, device=dev)
    x_sf = torch.empty((n, hidden // 16), dtype=torch.uint8, device=dev)
    if one and route_one.route_one(
        logits,
        selection_bias,
        routed_scaling_factor,
        ids,
        weights,
        tables,
        tiles,
        local_start,
        num_local,
        pdl=False,
        dynamic_ep=dynamic_ep,
        x=x,
        global_scale=global_scale.view(1),
        x_fp4=x_fp4,
        x_sf=x_sf,
    ):
        return ids, weights, x_fp4.view(torch.uint8), x_sf, tables
    if n > 0:
        rq._compile(
            experts, top_k, routed_scaling_factor, hidden, torch.bfloat16, tiles, None, False, False, False, False,
            False,
            num_local=num_local, cta_scoring=rq._cta_scoring(n, experts), splits=1, dynamic_ep=dynamic_ep or (0, 0),
        )(logits, selection_bias, x, global_scale.view(1), ids, weights, x_fp4, x_sf, tables, local_start,
          None, None, None, None, None, None, 0, None, None)  # fmt: skip
    return ids, weights, x_fp4.view(torch.uint8), x_sf, tables


def routing_tables_fit(num_tokens: int, local_experts: int, total_experts: int) -> bool:
    """Whether the launch can write tables for this shape (a setup-free predicate; shapes only)."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import can_build_routing_tables

    return can_build_routing_tables(num_tokens, local_experts, total_experts)


# The rows up to which an un-finalized call runs the CuTe DSL FC2 behind the FC1-only TRTLLM-Gen run: stock
# DeepseekV41ForCausalLM's `MoEDecodeOptions.cute_fc2_rows` (`_MOE_CUTE_FC2_ROWS`), which the engine's op takes as
# `cute_fc2_rows` (it decides per call; the bring-up's op took a flag).
_CUTE_FC2_ROWS = 16


class Dsv4RouteNvfp4QuantTables(OpWrapper):
    """`dsv4_route_nvfp4_quant` (bit for bit: ids `[n, k]` int32 global expert ids, bf16 weights, NVFP4 rows and
    row-major scales) plus `tables [len(tiles), 5, width]` int32: the runner's routing tables for tiles `tiles` over
    the experts `[local_first, local_first + local_count)`, numbered from 0 (-1: another rank's expert); with
    `dynamic_rank` (0 or 1) over this EP rank's home-first share of all the experts in use instead (every rank holds
    every expert; `local_first` / `local_count` are then ignored). Up to 8 rows of 384 experts at top-6 this is the
    engine's one-cluster route kernel (`dsv4_route_one`, quantizing in blocks 1-7), a plain launch."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows, this EP rank's 192 of 384 experts, top-6",
            spec=dict(n=6, experts=384, first=192, count=192),
        ),
        Cell(why="one row, the first EP window", spec=dict(n=1, experts=384, first=0, count=192)),
        Cell(
            why="eight rows: the launch's table limit",
            spec=dict(n=8, experts=384, first=192, count=192),
        ),
        Cell(
            why="tep8 decode, dynamic EP rank 1",
            spec=dict(n=6, experts=384, first=0, count=384, dynamic_rank=1),
        ),
    )
    note = "The tables' format is the runner's; the runner entry's test proves them by running the experts on them."

    def __call__(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _route_launch(
            logits,
            selection_bias,
            x,
            global_scale,
            top_k,
            routed_scaling_factor,
            local_first,
            local_count,
            tiles,
            dynamic_rank=dynamic_rank,
        )

    def reference(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> tuple:
        """The family entry's routing and quantization; the tables have no plain-torch form (None)."""
        return (
            *dsv4_route_nvfp4_quant.reference(
                logits, selection_bias, x, global_scale, top_k, routed_scaling_factor
            ),
            None,
        )

    def is_valid(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> None:
        dsv4_route_nvfp4_quant.is_valid(
            logits, selection_bias, x, global_scale, top_k, routed_scaling_factor
        )
        _check_scope(logits, local_first, local_count, tiles, dynamic_rank)


def _check_scope(
    logits: torch.Tensor, local_first: int, local_count: int, tiles, dynamic_rank
) -> None:
    n, experts = logits.shape
    count = experts if dynamic_rank is not None else local_count
    assert routing_tables_fit(n, count, experts), (
        "the launch writes tables for at most 8 rows, 256..1024 experts"
    )
    assert 0 <= local_first and local_first + local_count <= experts
    assert tiles and set(tiles) <= set(ROUTING_TABLE_TILES)
    assert dynamic_rank in (None, 0, 1), "the dynamic split is a two-rank rule"


class Dsv4RouteTables(OpWrapper):
    """`(ids [n, k] int32, weights [n, k] bf16, tables [len(tiles), 5, width] int32)`: `dsv4_route_nvfp4_quant_tables`
    without the quantization, for rows whose NVFP4 copy the mHC boundary already wrote (stock's
    `TrtllmTrtllmGenNvfp4Impl.try_fused_route` -> `dsv4_route`). The routing and the tables are bit for bit that
    entry's; up to 8 rows of 384 experts at top-6 it is the one-cluster kernel launched as a programmatic dependent
    of the router GEMM, whose CTAs release their dependents right after their own wait: the TRTLLM-Gen FC1, which
    reads nothing before its dependency wait, launches while the routing runs."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows, static EP window of rank 1",
            spec=dict(n=6, experts=384, first=192, count=192),
        ),
        Cell(
            why="tep8 decode: six rows, dynamic EP rank 0",
            spec=dict(n=6, experts=384, first=0, count=384, dynamic_rank=0),
        ),
        Cell(
            why="tep8 decode: six rows, dynamic EP rank 1",
            spec=dict(n=6, experts=384, first=0, count=384, dynamic_rank=1),
        ),
        Cell(
            why="one row, dynamic EP rank 1",
            spec=dict(n=1, experts=384, first=0, count=384, dynamic_rank=1),
        ),
        Cell(
            why="eight rows: the launch's table limit, dynamic EP rank 0",
            spec=dict(n=8, experts=384, first=0, count=384, dynamic_rank=0),
        ),
    )
    note = "The runner entry's test proves the tables (static and dynamic) by running the experts on them."

    def __call__(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden_size: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _route_launch(
            logits,
            selection_bias,
            None,
            None,
            top_k,
            routed_scaling_factor,
            local_first,
            local_count,
            tiles,
            hidden_size,
            dynamic_rank=dynamic_rank,
        )

    def reference(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden_size: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> tuple:
        """The family entry's routing (ids, weights); the tables have no plain-torch form (None)."""
        x = torch.zeros(logits.shape[0], hidden_size, dtype=torch.bfloat16, device=logits.device)
        g = torch.ones(1, dtype=torch.float32, device=logits.device)
        ids, w, _, _ = dsv4_route_nvfp4_quant.reference(
            logits, selection_bias, x, g, top_k, routed_scaling_factor
        )
        return ids, w, None

    def is_valid(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden_size: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
        dynamic_rank: Optional[int] = None,
    ) -> None:
        assert logits.dtype == torch.float32 and logits.is_contiguous() and logits.dim() == 2
        assert selection_bias.dtype == torch.float32 and selection_bias.shape == (logits.shape[1],)
        assert hidden_size % 16 == 0
        _check_scope(logits, local_first, local_count, tiles, dynamic_rank)


class Fp4MoeRunnerTables(OpWrapper):
    """The TRTLLM-Gen NVFP4 MoE runner (the catalog's `fp4_block_scale_moe_runner`, same arguments) on routing tables:
    a tactic whose tile is in `routing_tiles` skips the runner's routing kernel and reads `routing_tables`; any other
    tactic routes itself from `topk_ids` (global ids). Unfinalized, a run on the tables returns an empty `[0, tile]`
    permutation: the permutation is row 0 of that tile's table (`table_permutation`).

    `routing_method_type` / `n_group` / `topk_group` / `routed_scaling_factor`: what the runner is told about the
    routing
    (the ids and weights are the caller's). The runner's own routing kernel (tactics off the tables) and the autotuner's
    dummy routing follow them, so a target passes what the engine's MoE passes for its model.

    The engine's options of the same op (stock MR !11040): `cute_fc2` runs an unfinalized decode-sized call as the
    FC1-only TRTLLM-Gen run plus the CuTe DSL FC2 (`moe_lowm_fc2`, bit for bit the 8-token-tile FC2 cubins it replaces;
    other tactics keep the cubins); `routing_max_ctas` > 0 launches the expert GEMMs with that many CTAs in the batch
    dimension (`run_moe_bounded*`: the most the tables assign, `routing_table_max_ctas`) instead of the worst case;
    `dynamic_experts` > 0 (dynamic expert parallelism): the per-expert operands hold all that many experts and the
    tables assign this call's share of them, while the tuner and runs without the tables see the static window
    `[local_expert_offset, local_expert_offset + local_num_experts)` of them."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows over this rank's 128-expert window of 256, unfinalized",
            spec=dict(n=6, finalize=False),
        ),
        Cell(why="six rows, finalized in the runner", spec=dict(n=6, finalize=True)),
        Cell(why="one row", spec=dict(n=1, finalize=False)),
        Cell(
            why="tep8 decode: six rows, CuTe FC2, bounded grid, dynamic EP rank 0 of 384 experts",
            spec=dict(n=6, finalize=False, cute_fc2=True, dynamic_rank=0),
        ),
        Cell(
            why="tep8 decode: six rows, CuTe FC2, bounded grid, dynamic EP rank 1 of 384 experts",
            spec=dict(n=6, finalize=False, cute_fc2=True, dynamic_rank=1),
        ),
        Cell(
            why="one row, CuTe FC2 on a static window of 384 experts",
            spec=dict(n=1, finalize=False, cute_fc2=True),
        ),
    )
    note = """
    Same kernels and tactic as the runner's own routing: the test gates the outputs bit for bit against it (a dynamic
    run's rows against the runner over every expert, each (token, k) row wherever it is computed).
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
        cute_fc2: bool = False,
        routing_max_ctas: int = 0,
        dynamic_experts: int = 0,
    ) -> List[torch.Tensor]:
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            None, None, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, None, None, None,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, None, output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
            local_num_experts, routed_scaling_factor, routing_method_type, do_finalize, act_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids, routing_tables=routing_tables, routing_tiles=list(routing_tiles),
            cute_fc2_rows=_CUTE_FC2_ROWS if cute_fc2 else 0, dynamic_experts=dynamic_experts,
            routing_max_ctas=routing_max_ctas,
        )  # fmt: skip

    def reference(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
        cute_fc2: bool = False,
        routing_max_ctas: int = 0,
        dynamic_experts: int = 0,
    ) -> List[torch.Tensor]:
        """The catalog runner on the same operands, routing itself (the computation the tables must reproduce): over the
        static window, or with `dynamic_experts` over every expert (each pick's FC2 row, wherever the split puts it)."""
        if dynamic_experts:
            local_expert_offset, local_num_experts = 0, dynamic_experts
        return fp4_block_scale_moe_runner(
            None, None, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, None, None, None,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, None, output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
            local_num_experts, routed_scaling_factor, routing_method_type, do_finalize, act_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )  # fmt: skip

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
        cute_fc2: bool = False,
        routing_max_ctas: int = 0,
        dynamic_experts: int = 0,
    ) -> None:
        n = hidden_states.shape[0]
        assert topk_ids.shape == topk_weights.shape == (n, top_k) and topk_ids.dtype == torch.int32
        assert (
            routing_tables.dtype == torch.int32
            and routing_tables.dim() == 3
            and routing_tables.shape[:2] == (len(routing_tiles), 5)
        ), "one [5, width] table per tile, from `dsv4_route_nvfp4_quant_tables` of this window"
        assert routing_tiles and set(routing_tiles) <= set(ROUTING_TABLE_TILES)
        assert not (cute_fc2 and do_finalize), (
            "the CuTe FC2 serves unfinalized calls only (the runner ignores it otherwise)"
        )
        experts = dynamic_experts or local_num_experts
        assert gemm1_weights.shape[0] == gemm2_weights.shape[0] == experts, (
            "per-expert operands of every held expert"
        )
        if dynamic_experts:
            assert dynamic_experts == num_experts and routing_tables.shape[2] >= n * top_k, (
                "dynamic EP: every rank holds every expert; the tables are the dynamic split's"
            )
        assert routing_max_ctas >= 0


def table_permutation(
    outputs: List[torch.Tensor], tables: torch.Tensor, tiles: Sequence[int], n: int, top_k: int
):
    """-> (the unfinalized run's `[n, top_k]` expanded -> permuted row index, the tile it ran on or None when the runner
    routed itself). A setup-free helper over the runner's outputs (no launch: a view of the table's head)."""
    perm = outputs[2]
    if perm.numel() == 0:
        tile = perm.shape[1]
        return tables[list(tiles).index(tile), 0][: n * top_k].view(n, top_k), tile
    return perm, None


dsv4_route_nvfp4_quant_tables = Dsv4RouteNvfp4QuantTables()
dsv4_route_tables = Dsv4RouteTables()
fp4_moe_runner_tables = Fp4MoeRunnerTables()
