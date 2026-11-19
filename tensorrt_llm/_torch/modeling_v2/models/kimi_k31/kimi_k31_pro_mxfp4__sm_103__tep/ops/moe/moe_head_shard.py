# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE head above the MoE front's rows, sharded over the attention-TP group: router logits and routed
latent of every row from 1 / TP of the head weight per rank, bit for bit the replicated head's; or, with the MoE
prologue folded into the all-gather, the routing and the quantized latent."""

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN = 7168
ROUTER_ROWS = 896
LATENT_ROWS = 3584
TOP_K = 16
#: Rows the sharded head serves: above the MoE front's rows, up to the one-launch head's last (``moe_head_gemm``).
MIN_ROWS = 33
MAX_ROWS = 176
#: The replicated head's K split per row range (K tiles of 128; 56 per row): the router GEMM (``dsv3_router_gemm_op``,
#: cuBLAS split-K with FP32 output) sums three K chunks of 19 / 19 / 18 tiles in order up to 64 rows and two halves
#: above; the latent down projection (``k31_fused_a_gemm``, cuBLAS) two halves at 33-48 and 65-96 rows and the whole K
#: at 49-64 and 97-176 (GB300, cuBLAS kernels per row count; every bound checked bit for bit against both GEMMs and,
#: at 65-176 rows, against the one-launch head ``moe_head_gemm``, which sums the same chunks).
#: ``(max rows, (latent K-tile bounds, router K-tile bounds))``.
SPLITS: Tuple[Tuple[int, Tuple[Tuple[int, ...], Tuple[int, ...]]], ...] = (
    (48, ((28,), (19, 38))),
    (64, ((), (19, 38))),
    (96, ((28,), (28,))),
    (176, ((), (28,))),
)
#: Token tile and A/B stages of the sharded GEMM per row range (GB300, cold weights, one rank's slice, us per layer:
#: 65-96 rows 5.0 at TP16 and 5.2 at TP8 against 14.3 for the replicated one-launch head; 97-176 rows 7.4-7.7 and
#: 7.8-7.9 against 14.3-15.3).
#: ``(max rows, (cta_n, stages))``.
TILES: Tuple[Tuple[int, Tuple[int, int]], ...] = (
    (64, (16, 12)),
    (176, (32, 11)),
)


def shard_config(rows: int) -> Optional[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]]:
    """``(cta_n, stages, latent bounds, router bounds)`` of the sharded GEMM at ``rows``, or None outside the rows."""
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return None
    latent_bounds, router_bounds = next(s for bound, s in SPLITS if rows <= bound)
    tile = next(t for bound, t in TILES if rows <= bound)
    return tile + (latent_bounds, router_bounds)


def shard_serves(rows: int, world: int) -> bool:
    """Whether the sharded head serves ``rows`` rows on an attention-TP group of ``world`` ranks."""
    return (
        shard_config(rows) is not None
        and world > 1
        and ROUTER_ROWS % (4 * world) == 0
        and LATENT_ROWS % (8 * world) == 0
    )


def shard_views(
    head_weight: torch.Tensor, rank: int, world: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(latent rows, router rows)`` of ``rank``: views of the stacked head weight ``[896 + 3584, 7168]``."""
    rw, lw = ROUTER_ROWS // world, LATENT_ROWS // world
    return (
        head_weight[ROUTER_ROWS + rank * lw : ROUTER_ROWS + (rank + 1) * lw],
        head_weight[rank * rw : (rank + 1) * rw],
    )


def gather_bytes(rows: int, world: int) -> int:
    """Bytes of one Lamport buffer the all-gather of ``rows`` rows over ``world`` ranks needs: the group's workspace is
    grown to ``gather_bytes(max rows, W)`` before CUDA-graph capture."""
    from ...kernels.moe.moe_head_shard import gather_bytes as _gather_bytes

    return _gather_bytes(rows, world)


def compile_shard(
    world: int,
    hidden_states: torch.Tensor,
    w_latent: torch.Tensor,
    w_router: torch.Tensor,
    config: Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
    pdl: Optional[bool] = None,
    *,
    routed: bool,
) -> None:
    """JIT-compile the kernels of the sharded head's calls of ``hidden_states``' rows (``config = shard_config(rows)``)
    before CUDA-graph capture: the slice GEMM runs once on ``hidden_states`` (it compiles on its first launch), the
    all-gather of :data:`k31_moe_head_shard` compiles, and with ``routed`` the fused all-gather + MoE prologue of
    :data:`k31_moe_head_shard_routed` and, where the config takes it, its push GEMM and poll kernel.
    ``w_latent`` / ``w_router``: this rank's ``shard_views``. ``pdl``: compile for programmatic dependent launch (None
    reads the engine's PDL flag, as the calls do). Launches nothing besides the GEMM and reads no workspace."""
    from ...kernels.moe import moe_head_push as _push
    from ...kernels.moe import moe_head_shard as _shard

    if pdl is None:
        pdl = get_env_enable_pdl()

    rows = hidden_states.shape[0]
    out_latent, out_router = _shard.head_planes(
        rows, w_latent, w_router, config[2], config[3], hidden_states.device
    )
    _shard.run_head_shard_gemm(
        hidden_states, w_latent, w_router, out_latent, out_router, config, pdl
    )
    _shard.compile_head_gather(world, out_router, out_latent, pdl)
    if not routed:
        return
    _shard.compile_head_route_quant(world, out_router, out_latent, pdl)
    if _push.push_serves(config):
        _push.compile_head_push(
            world, hidden_states, w_latent, w_router, out_latent, out_router, config, pdl
        )


class K31MoeHeadShard(OpWrapper):
    """``(logits, latent)`` of ``hidden_states [M, 7168]`` BF16 against the stacked head weight ``[896 + 3584, 7168]``
    BF16 (router rows first), identical on every rank of the attention-TP group: ``logits = hidden_states @
    head_weight[:896].T`` FP32 ``[M, 896]`` and ``latent = hidden_states @ head_weight[896:].T`` BF16 ``[M, 3584]``.

    Rank ``r`` of ``W`` multiplies ``hidden_states`` (the same on every rank) by its router rows ``[r * 896 / W, ...)``
    and latent rows ``[r * 3584 / W, ...)`` (``kimi_k31_pro_mxfp4__sm_103__tep/kernels/moe/moe_head_shard``, K
    summed in the replicated head's chunks, ``SPLITS``) and the ranks all-gather the slices over the group's MNNVL
    Lamport workspace. At ``MIN_ROWS``-``MAX_ROWS`` rows these are the bits of the router (``dsv3_router_gemm_op``,
    cuBLAS with FP32 output) and of the latent down projection (``k31_fused_a_gemm``, cuBLAS) that the head runs
    replicated, except that a ``-0.0`` logit, or a latent pair word ``0x80000000``, arrives as ``+0.0`` (the Lamport
    sentinel; the values compare equal).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: Tuple[Cell, ...] = tuple(
        Cell(
            why=f"TEP16-like group of one tray (W = 4), {rows} verify rows ({why})",
            spec=dict(world=4, rows=rows),
        )
        for rows, why in (
            (33, "the first row served; latent in two K halves"),
            (35, "B7 x 5"),
            (48, "the last row with the latent in two K halves"),
            (49, "the first row with the latent's whole K"),
            (60, "B12 x 5"),
            (64, "the last row of the router's three K chunks"),
            (65, "the router's two K halves; latent in two K halves"),
            (96, "the last row with the latent in two K halves"),
            (97, "the latent's whole K above 96 rows"),
            (120, "B24 x 5"),
            (176, "the last row served"),
        )
    )
    note = """
    Serves ``shard_serves(rows, W)`` only. ``workspace`` is the group's engine MNNVL Lamport workspace
    (``oproj_allreduce_push.lamport_workspace``, one-shot, grown before CUDA-graph capture to
    ``moe_head_shard.gather_bytes(MAX_ROWS, W)``), shared with every MNNVL all-reduce of the group: calls must be
    in the same order on every rank and ordered by stream. The GEMM JIT-compiles per config on its first call, which
    must come before CUDA-graph capture. ``pdl`` launches both kernels with programmatic dependent launch (each
    waits for the previous grid before reading its inputs or the rotation state); None reads the engine's PDL flag
    (``TRTLLM_ENABLE_PDL``).
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...kernels.moe.moe_head_shard import head_planes, run_head_gather, run_head_shard_gemm

        if pdl is None:
            pdl = get_env_enable_pdl()
        rows = hidden_states.shape[0]
        config = shard_config(rows)
        if config is None or not shard_serves(rows, world):
            raise ValueError(f"the sharded MoE head serves {MIN_ROWS}-{MAX_ROWS} rows, got {rows}")
        w_latent, w_router = shard_views(head_weight, rank, world)
        dev = hidden_states.device
        out_latent, out_router = head_planes(rows, w_latent, w_router, config[2], config[3], dev)
        run_head_shard_gemm(hidden_states, w_latent, w_router, out_latent, out_router, config, pdl)
        logits = torch.empty(rows, ROUTER_ROWS, dtype=torch.float32, device=dev)
        latent = torch.empty(rows, LATENT_ROWS, dtype=torch.bfloat16, device=dev)
        run_head_gather(workspace, out_router, out_latent, logits, latent, pdl)
        return logits, latent

    def reference(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The replicated head: router logits through ``dsv3_router_gemm_op``, latent through ``k31_fused_a_gemm``, with
        ``-0.0`` logits and latent pair words ``0x80000000`` as ``+0.0`` (see the class docstring)."""
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
            k31_fused_a_gemm,
        )
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
            dsv3_router_gemm_op,
        )

        logits = dsv3_router_gemm_op(hidden_states, head_weight[:ROUTER_ROWS].t())
        latent = k31_fused_a_gemm(hidden_states, head_weight[ROUTER_ROWS:].t())
        logits = torch.where(
            (logits == 0) & torch.signbit(logits), torch.zeros_like(logits), logits
        )
        words = latent.view(torch.int32)
        latent = torch.where(words == -(2**31), torch.zeros_like(words), words).view(torch.bfloat16)
        return logits, latent

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> None:
        assert shard_serves(hidden_states.shape[0], world), (
            "rows or group outside the sharded head's range"
        )
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.shape[1] == HIDDEN
        assert hidden_states.is_contiguous() and hidden_states.data_ptr() % 32 == 0
        assert head_weight.dtype == torch.bfloat16 and tuple(head_weight.shape) == (
            ROUTER_ROWS + LATENT_ROWS,
            HIDDEN,
        )
        assert head_weight.is_contiguous() and head_weight.data_ptr() % 32 == 0
        assert 0 <= rank < world


k31_moe_head_shard = K31MoeHeadShard()


class K31MoeHeadShardRouted(OpWrapper):
    """``(topk_ids, topk_weights, x_fp8, x_sf)`` of the sharded head's rows: :data:`k31_moe_head_shard`'s ``(logits,
    latent)`` through the MoE prologue with the TRTLLM-Gen runner's routing weights (``k31_moe_prologue(logits, bias,
    1.0, latent, runner_weights=True)``: top-16 of ``sigmoid(logit) + bias``, unit scale; MXFP8 of the latent), with
    the all-gather and the prologue in one launch that reads the slices straight from the Lamport buffer
    (``kimi_k31_pro_mxfp4__sm_103__tep/kernels/moe/moe_head_shard.MoeHeadRouteQuantKernel``; the prologue's own
    routing and quantization code).

    Returns ``int32 [M, 16]``, ``bf16 [M, 16]``, ``float8_e4m3fn [M, 3584]``, ``uint8 [M, 112]``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: Tuple[Cell, ...] = K31MoeHeadShard.CELLS
    note = K31MoeHeadShard.note

    def __call__(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from ...kernels.moe import moe_head_push as _push
        from ...kernels.moe.moe_head_shard import (
            head_planes,
            run_head_route_quant,
            run_head_shard_gemm,
        )

        if pdl is None:
            pdl = get_env_enable_pdl()
        rows = hidden_states.shape[0]
        config = shard_config(rows)
        if config is None or not shard_serves(rows, world):
            raise ValueError(f"the sharded MoE head serves {MIN_ROWS}-{MAX_ROWS} rows, got {rows}")
        w_latent, w_router = shard_views(head_weight, rank, world)
        dev = hidden_states.device
        out_latent, out_router = head_planes(rows, w_latent, w_router, config[2], config[3], dev)
        # The GEMM multicasts its slices from its epilogue where one CTA holds every final value (the
        # multi-accumulator rows, ``kimi_k31_pro_mxfp4__sm_103__tep/kernels/moe/moe_head_push``), instead of the
        # gather reading them back and pushing them.
        push = _push.push_serves(config)
        if push:
            _push.run_head_push_gemm(
                workspace, hidden_states, w_latent, w_router, out_latent, out_router, config, pdl
            )
        else:
            run_head_shard_gemm(
                hidden_states, w_latent, w_router, out_latent, out_router, config, pdl
            )
        topk_ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=dev)
        topk_weights = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=dev)
        x_fp8 = torch.empty(rows, LATENT_ROWS, dtype=torch.float8_e4m3fn, device=dev)
        x_sf = torch.empty(rows, LATENT_ROWS // 32, dtype=torch.uint8, device=dev)
        if push:
            _push.run_head_poll_route_quant(
                workspace,
                out_router,
                out_latent,
                e_score_correction_bias,
                topk_ids,
                topk_weights,
                x_fp8,
                x_sf,
                pdl,
            )
            return topk_ids, topk_weights, x_fp8, x_sf
        run_head_route_quant(
            workspace,
            out_router,
            out_latent,
            e_score_correction_bias,
            topk_ids,
            topk_weights,
            x_fp8,
            x_sf,
            pdl,
        )
        return topk_ids, topk_weights, x_fp8, x_sf

    def reference(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The replicated head (:meth:`K31MoeHeadShard.reference`) through the MoE prologue with the runner's
        weights."""
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_prologue import (
            k31_moe_prologue,
        )

        logits, latent = k31_moe_head_shard.reference(
            hidden_states, head_weight, rank, world, workspace, pdl
        )
        return k31_moe_prologue(logits, e_score_correction_bias, 1.0, latent, True)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        rank: int,
        world: int,
        workspace,
        pdl: Optional[bool] = None,
    ) -> None:
        k31_moe_head_shard.is_valid(hidden_states, head_weight, rank, world, workspace, pdl)
        assert (LATENT_ROWS // world) % 32 == 0, "a rank's latent slice must be whole MXFP8 blocks"
        assert e_score_correction_bias.dtype == torch.float32
        assert tuple(e_score_correction_bias.shape) == (ROUTER_ROWS,)
        assert e_score_correction_bias.is_contiguous()
        assert e_score_correction_bias.data_ptr() % 16 == 0


k31_moe_head_shard_routed = K31MoeHeadShardRouted()
