# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sharded MoE front on a BF16 latent_down (SM107, collective over the TP group, up to
32 rows): this rank's latent tiles pushed to every rank, then ``rubin_ull_front``'s route kernel.

Each rank multiplies the step's rows by its 128-row tiles of latent_down (1 / W of the weight)
and multicasts the BF16 result into its slot of every rank's front workspace; the router logits
(computed by the caller, as on the replicated path) go into this rank's logits rows; the route
kernel polls both and writes the top-16, this rank's moe_sort tables and the NVFP4 latent. This
replaces the replicated latent_down, the routing tail and the input quantization.
:func:`rubin_ull_front_workspace`, :func:`rubin_ull_front_logits_views` and
:func:`rubin_ull_front_share` are the runtime builders.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....ops.moe._rubin_moe_common import noaux_topk, quantize_nvfp4, token_ordered_sort
from ._rubin_ull_common import (
    HIDDEN,
    LATENT,
    MAX_M,
    NUM_EXPERTS,
    block_slice,
    gemm_bf16,
    push_tactic,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_ull_front import ull_front_workspace as _front_workspace
    from ...kernels.moe.rubin_ull_push_gemm import front_logits_views as _logits_views
    from ...kernels.moe.rubin_ull_push_gemm import run_ull_front_bf16 as _launch
    from ...kernels.moe.rubin_ull_push_gemm import tactic_of


class RubinUllFrontBf16(OpWrapper):
    """The MoE front of one rank for ``x`` BF16 ``[m <= 32, 7168]`` (collective: every rank of the
    TP group calls it at the same point).

    ``logits``: the FP32 router logits ``[m, 896]`` of ``x``; ``latent_rows``: this rank's BF16
    rows of latent_down (the contiguous 128-row tiles :func:`rubin_ull_front_share` names);
    ``logits_views``: the FP32 logits rows of the workspace's two halves
    (``front_logits_views``); ``ws``: the front workspace
    (``rubin_ull_front.ull_front_workspace``); ``tactic``: the push GEMM's
    ``(mma_m, mma_n, split_k, ab_stages)``. Returns ``(routing, codes, scales)``:
    ``rubin_route_tail``'s eight outputs for the expert window
    ``[local_offset, local_offset + local_num)``, and the NVFP4 latent (uint8 ``[m, 1792]`` codes,
    uint8 ``[m, 224]`` linear-layout scales) of ``bf16(x @ latent_down^T)`` under ``global_sf``.
    ``zero``: an int32 buffer the route kernel zeroes (FC12's output and counters).
    """

    ARCHS: frozenset[Arch] = frozenset()

    CELLS: Tuple[Cell, ...] = (
        Cell(
            why="8 rows (a TEP8 b1 k7 verify step) over TP4, 7 latent tiles per rank",
            spec=dict(rows=8, world=4),
        ),
        Cell(why="one row over TP4", spec=dict(rows=1, world=4)),
        Cell(why="17 rows: a second router tile", spec=dict(rows=17, world=4)),
        Cell(why="32 rows: the largest call", spec=dict(rows=32, world=4)),
        Cell(why="8 rows over TP2", spec=dict(rows=8, world=2)),
    )

    note = """
    Three launches on the current stream: the push GEMM (split-K, the latent rounded once to
    BF16, a reordered FP32 sum of the reference's), a copy of the logits into the workspace
    (adding +0 sends -0 as +0 for the poll), and the front's route kernel (the routing bitwise
    rubin_route_tail's on the same logits; the NVFP4 quantization fp4_quantize's arithmetic).
    The workspace halves alternate call by call; a captured graph replays that alternation. The
    first call of a shape compiles, so warm every row count before capture. Cells are one VR200
    tray's worlds; TP8 spans two trays (4 tiles on ranks 0-3, 3 on 4-7).
    """

    def __call__(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        latent_rows: torch.Tensor,
        logits_views: Tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
        routed_scaling_factor: float,
        global_sf: torch.Tensor,
        local_offset: int,
        local_num: int,
        ws,
        tactic: Tuple[int, int, int, int],
        *,
        zero: torch.Tensor,
    ):
        return _launch(
            x,
            logits,
            latent_rows,
            logits_views,
            bias,
            routed_scaling_factor,
            global_sf,
            local_offset,
            local_num,
            ws,
            tactic=tactic_of(tactic),
            zero=zero,
        )

    def reference(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        latent_down: torch.Tensor,
        bias: torch.Tensor,
        routed_scaling_factor: float,
        global_sf: float,
        local_offset: int,
        local_num: int,
    ):
        """``(routing, latent)``: the routing of ``rubin_route_tail.reference`` on ``logits`` and
        the NVFP4 latent dequantized (FP32 ``[m, 3584]``) of ``bf16(x @ latent_down^T)``.

        A collective, so the documented exception to mirroring: it takes the whole BF16
        ``latent_down [3584, 7168]`` every rank's rows were cut from plus this rank's window, not
        one rank's rows and workspace."""
        ids, weights = noaux_topk(logits, bias, routed_scaling_factor)
        routing = (ids, weights, *token_ordered_sort(ids, local_offset, local_num))
        latent = gemm_bf16(x, latent_down)
        return routing, quantize_nvfp4(latent.float(), float(global_sf))

    def is_valid(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        latent_rows: torch.Tensor,
        logits_views: Tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor,
        routed_scaling_factor: float,
        global_sf: torch.Tensor,
        local_offset: int,
        local_num: int,
        ws,
        tactic: Tuple[int, int, int, int],
        *,
        zero: torch.Tensor,
    ) -> None:
        m = x.shape[0]
        assert x.dtype == latent_rows.dtype == torch.bfloat16, "BF16 rows and weight"
        assert x.is_contiguous() and tuple(x.shape[1:]) == (HIDDEN,), "x [m, 7168], contiguous"
        assert 1 <= m <= MAX_M, f"1 to {MAX_M} rows"
        assert logits.dtype == torch.float32 and tuple(logits.shape) == (m, NUM_EXPERTS)
        # Another rank's tiles land in this rank's slot without an error.
        _, n = block_slice(LATENT, ws.world, ws.rank)
        assert tuple(latent_rows.shape) == (n, HIDDEN), "the rank's own 128-row tiles"
        assert latent_rows.is_contiguous(), "latent rows read as dense K-major tiles"
        assert all(v.shape == (MAX_M, NUM_EXPERTS) for v in logits_views), "both halves' rows"
        assert 1 <= local_num and local_offset + local_num <= NUM_EXPERTS, "an expert window"
        assert zero.dtype == torch.int32 and zero.is_contiguous(), "zero must be dense int32"


rubin_ull_front_bf16 = RubinUllFrontBf16()


def rubin_ull_front_logits_views(ws) -> Tuple[torch.Tensor, torch.Tensor]:
    """The FP32 ``[32, 896]`` logits rows of the front workspace's two halves on this rank, the
    ``logits_views`` operand (a runtime builder, not an entry; build once, outside capture)."""
    return _logits_views(ws)


def rubin_ull_front_workspace(mapping):
    """The front's MNNVL workspace of ``mapping``'s TP group, the ``ws`` operand (a runtime builder,
    not an entry; collective: every rank calls it in ``post_load_weights``, outside capture)."""
    return _front_workspace(mapping)


def rubin_ull_front_share(world: int, rank: int, max_tokens: int = MAX_M) -> tuple:
    """``(r0, n, tactics)`` of ``rank`` (a runtime builder, not an entry): its 128-row tiles
    ``[r0, r0 + n)`` of latent_down and the push GEMM's tactic for each row count,
    ``tactics[m]`` for ``m`` in ``[1, max_tokens]`` (``tactics[0]`` is None)."""
    if not 1 <= max_tokens <= MAX_M:
        raise ValueError(f"{max_tokens} rows: the front serves 1 to {MAX_M}")
    r0, n = block_slice(LATENT, world, rank)
    return r0, n, (None, *(push_tactic(m, n, HIDDEN) for m in range(1, max_tokens + 1)))
