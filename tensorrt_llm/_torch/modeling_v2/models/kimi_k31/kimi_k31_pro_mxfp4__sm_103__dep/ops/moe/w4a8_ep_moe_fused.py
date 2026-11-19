# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 expert-parallel routed experts (attention DP) with the combine fused into the FC2 epilogue.

One rank's call, the whole routed MoE of its tokens across the EP group (W ranks, pure EP over the attention-DP TP
group, one symmetric workspace ``kimi_k31_pro_mxfp4__sm_103__dep/kernels/comm/moe_a2a.MoeA2AWorkspace``):

1. the family push dispatch (:mod:`...kernels.comm.moe_a2a`): each token's MXFP8 row, scales, expert ids and
   routing weights to every rank owning one of its experts (rows ``s * T + t``; ``-1`` ids elsewhere);
2. the runner sort of the received rows over this rank's experts, with expanded indices in ``p2t``;
3. FC1 + SiTU + MXFP8 requant on CTA pairs (:mod:`....kernels.moe.w4a8_moe_gemm2`);
4. FC2 on CTA pairs pushing each expert output row into its token's slot row on the token's rank, then flagging
   (:class:`...kernels.moe.ep_moe_fused.EpFc2PushKernel`);
5. on the token's rank, the sum of the slot rows (:class:`...kernels.moe.ep_moe_fused.EpCombineReduceKernel`).

Bitwise equal to the separate path it replaces (dispatch, the CuTe W4A8 op with its finalize, the one-sided
all-to-all combine): the same expert output rows, the finalize's FP32 slot-order sum per expert rank rounded to
BF16, the combine's FP32 tree over the ranks. It drops the finalize launch and the separate combine (its pushes
run under the FC2 weight stream).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.runner_sort import (
    CLUSTER_MAX_ROWS,
    max_tiles_for,
    run_runner_sort,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.w4a8_moe_gemm2 import (
    run_w4a8_moe_2cta,
)
from tensorrt_llm.logger import logger

from ...kernels.comm import moe_a2a as _a2a
from ...kernels.moe.ep_moe_fused import run_ep_combine_reduce, run_ep_dispatch, run_ep_fc2_push

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
_SF_VEC = 32
#: Token tile of the CTA-pair GEMMs (128 rows per pair MMA, the MMA N sized to each tile's rows).
N_TILE = 128


def slot_bytes(world: int) -> int:
    """Per-row ``out_bytes`` of the workspace whose combine region holds the slot rows ``[T, 16, HIDDEN]`` BF16
    (the region is ``W * T * out_bytes``)."""
    return -(-(TOP_K * HIDDEN * 2) // (world * 16)) * 16


def get_ep_moe_workspace(mapping, max_tokens_per_rank: int) -> _a2a.MoeA2AWorkspace:
    """The EP group's workspace for K3.1 rows (allocated on first use; collective)."""
    return _a2a.get_workspace(
        mapping,
        max_tokens_per_rank,
        hidden_bytes=HIDDEN,
        sf_bytes=HIDDEN // _SF_VEC,
        top_k=TOP_K,
        weight_bytes=2 * TOP_K,
        out_bytes=slot_bytes(mapping.moe_ep_size),
    )


class K31W4A8EpMoeFused(OpWrapper):
    """This rank's routed MoE output ``[ntok, 3584]`` BF16 for its ``ntok`` tokens; see the module docstring.

    ``xq`` E4M3 ``[ntok, 3584]``, ``xsf`` UE8M0 ``[ntok, 112]`` linear, ``topk_ids`` int32 / ``topk_weights`` BF16
    ``[ntok, 16]``; ``tmax`` the group's largest per-rank token count of the step (the same on every rank). Weights:
    the runner-layout ``w3_w1`` / ``w2`` tensors and scales of this rank's ``local_num`` experts from
    ``local_offset``. Every rank of the group calls it in the same order with the same ``tmax``. ``real_rows`` (int32
    device word, optional): this rank's tokens at or past it are padding (the attention-DP CUDA-graph padding
    requests, the tail of the step), dispatched nowhere, their output rows zero; the other rows are unchanged bit for
    bit. ``place`` (int32 ``[896]`` device table, optional): each routed expert's physical slot ``rank * local_num +
    slot`` (the group's expert placement; ``local_offset`` / ``local_num`` and the weights are then this rank's
    physical slots). The output is the same bits as without placement: an expert's output rows do not depend on where
    it runs, and the reduce groups each token's slots by their routed experts' owner ranks of the identity placement.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode rows of the attention-DP target: 48 tokens per rank, one 128-row tile per expert",
            spec=dict(world=4, num_tokens=(48, 48, 48, 48)),
        ),
        Cell(
            why="200 tokens per rank: 800 dispatched rows, experts with more than one 128-row tile",
            spec=dict(world=4, num_tokens=(200, 200, 200, 200)),
        ),
        Cell(
            why="uneven ranks, one without tokens: padded dispatch rows and a zero-token reduce",
            spec=dict(world=4, num_tokens=(37, 0, 64, 5)),
        ),
        Cell(
            why="real-row words (CUDA-graph padding requests at the tail of the step): padding rows skipped",
            spec=dict(world=4, num_tokens=(52, 52, 52, 52), real=(33, 52, 0, 40)),
        ),
        Cell(
            why="a static expert placement (a permutation of the 896 experts over the slots)",
            spec=dict(world=4, num_tokens=(64, 48, 64, 7), placed=True),
        ),
    )
    note = """
    Collective over the EP group (the attention-DP TP group, pure EP): every rank calls it in the same order with the
    same ``tmax``, on the group's ``get_ep_moe_workspace`` (allocated outside CUDA-graph capture). The Kimi K3.1
    attention-DP target calls it from its routed MoE for single-chunk steps of up to ``MAX_ROWS`` dispatched rows,
    with the runner-layout weights, BF16 routing weights and the step's real-row word.

    The cells run on one GB300 tray (4 ranks, 224 experts each); the served DEP16 group (16 ranks, 56 experts each)
    runs the same kernels with a wider dispatch and is not driven here. The test also checks the op bit for bit
    against the separate path (dispatch, the CuTe W4A8 op with its finalize, the one-sided combine).

    ARCHS lists only what was driven: sm_103, the only architecture this target serves (on sm_107 the general target
    serves these deployments).
    """

    def __call__(
        self,
        ws: _a2a.MoeA2AWorkspace,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        local_offset: int,
        local_num: int,
        tmax: int,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        out: Optional[torch.Tensor] = None,
        reduce_threads: int = 128,
        real_rows: Optional[torch.Tensor] = None,
        place: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ntok = topk_ids.shape[0]
        dev = xq.device
        world = ws.world
        if out is None:
            out = torch.empty(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)
        rows = world * tmax
        assert rows <= CLUSTER_MAX_ROWS and tmax <= ws.max_tokens
        run_ep_dispatch(xq, xsf.view(ntok, HIDDEN // _SF_VEC), topk_ids, topk_weights, ws.peers, ws.base, ws.offs,
                            tmax, ws.rank, world, NUM_EXPERTS, real_rows=real_rows, place=place)  # fmt: skip
        x = ws.recv(0, tmax, xq.dtype, HIDDEN)
        x_sf = ws.recv(1, tmax, torch.uint8, HIDDEN // _SF_VEC)
        ids_r = ws.recv(2, tmax, torch.int32, TOP_K)
        inter = w1.shape[1] // 2
        max_tiles = max_tiles_for(rows, local_num, N_TILE)
        i32 = dict(dtype=torch.int32, device=dev)
        e2p = torch.empty(rows * TOP_K, **i32)
        p2x = torch.empty(max_tiles * N_TILE, **i32)
        total = torch.empty(1, **i32)
        t2e = torch.empty(max_tiles, **i32)
        t2l = torch.empty(max_tiles, **i32)
        nt = torch.empty(1, **i32)
        run_runner_sort(
            ids_r, local_offset, local_num, N_TILE, e2p, p2x, total, t2e, t2l, nt, expanded=True
        )
        h = torch.empty(max_tiles * N_TILE, inter, dtype=torch.float8_e4m3fn, device=dev)
        h_sf = torch.empty(max_tiles * N_TILE, inter // _SF_VEC, dtype=torch.uint8, device=dev)
        run_w4a8_moe_2cta(w1, w1_sf, None, None, x, x_sf, h, h_sf, None, t2e, t2l, p2x, nt, n_tile=N_TILE,
                              gate_cap=gate_cap, linear_cap=linear_cap, p2t_div=TOP_K, max_acc_stages=3)  # fmt: skip
        slot_off = ws.offsets[4]
        run_ep_fc2_push(w2, w2_sf, h, h_sf, t2e, t2l, p2x, nt, ws.peers, ws.base, slot_off, tmax, ws.rank, world,
                            n_tile=N_TILE, top_k=TOP_K)  # fmt: skip
        run_ep_combine_reduce(topk_ids, topk_weights, out, ws.base, slot_off, world, NUM_EXPERTS,
                                  threads=reduce_threads, real_rows=real_rows)  # fmt: skip
        return out

    def reference(
        self,
        ws: _a2a.MoeA2AWorkspace,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: Sequence[torch.Tensor],
        w1_sf: Sequence[torch.Tensor],
        w2: Sequence[torch.Tensor],
        w2_sf: Sequence[torch.Tensor],
        local_offset: int,
        local_num: int,
        tmax: int,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        out: Optional[torch.Tensor] = None,
        reduce_threads: int = 128,
        real_rows: Optional[torch.Tensor] = None,
        place: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This rank's output in plain torch: for each rank of the group in rank order, the routed rows of the experts
        that rank owns under the identity placement (:meth:`K31W4A8MoeCute.reference`, finalized in FP32 and rounded
        to BF16, as the fused FC2 slot sum is), summed in FP32 over the ranks and rounded once to BF16; rows at or past
        ``real_rows`` are zero.

        The documented collective exception: ``w1`` / ``w1_sf`` / ``w2`` / ``w2_sf`` are every rank's tensors in rank
        order (each rank's physical slots; with ``place``, logical expert ``e`` is slot ``place[e]``). Ignores ``ws``,
        ``local_offset``, ``tmax``, ``out`` and ``reduce_threads``.
        """
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_cute import (
            k31_w4a8_moe_cute,
        )

        world = len(w1)
        slots = place.tolist() if place is not None else None
        acc = torch.zeros(topk_ids.shape[0], HIDDEN, dtype=torch.float32, device=xq.device)
        for g in range(world):
            if slots is None:
                weights = (w1[g], w1_sf[g], w2[g], w2_sf[g])
            else:
                owned = [slots[e] for e in range(g * local_num, (g + 1) * local_num)]
                weights = tuple(
                    torch.stack([t[s // local_num][s % local_num] for s in owned])
                    for t in (w1, w1_sf, w2, w2_sf)
                )
            part = k31_w4a8_moe_cute.reference(
                xq, xsf, topk_ids, topk_weights, *weights, g * local_num, local_num, N_TILE, None,
                gate_cap=gate_cap, linear_cap=linear_cap,
            )  # fmt: skip
            acc += part.float()
        result = acc.to(torch.bfloat16)
        if real_rows is not None:
            result[int(real_rows[0]) :] = 0
        return result

    def is_valid(
        self,
        ws: _a2a.MoeA2AWorkspace,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        local_offset: int,
        local_num: int,
        tmax: int,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        out: Optional[torch.Tensor] = None,
        reduce_threads: int = 128,
        real_rows: Optional[torch.Tensor] = None,
        place: Optional[torch.Tensor] = None,
    ) -> None:
        ntok = topk_ids.shape[0]
        assert local_num * ws.world == NUM_EXPERTS and local_offset == ws.rank * local_num, (
            f"rank {ws.rank} of {ws.world} must own experts [{ws.rank * local_num}, +{local_num}) of {NUM_EXPERTS}"
        )
        assert w1.shape[0] == local_num and w2.shape[0] == local_num, (
            "the weights are this rank's local experts"
        )
        assert ntok <= tmax, f"tmax {tmax} below this rank's {ntok} tokens"
        assert topk_ids.dtype == torch.int32 and tuple(topk_ids.shape) == (ntok, TOP_K)
        assert topk_weights.dtype == torch.bfloat16 and tuple(topk_weights.shape) == (ntok, TOP_K)
        assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
        assert xq.dtype == torch.float8_e4m3fn and tuple(xq.shape) == (ntok, HIDDEN)
        assert xsf.dtype == torch.uint8 and xsf.numel() == ntok * (HIDDEN // _SF_VEC)
        if real_rows is not None:
            assert real_rows.dtype == torch.int32 and real_rows.is_cuda
        if place is not None:
            assert place.dtype == torch.int32 and tuple(place.shape) == (NUM_EXPERTS,)


k31_w4a8_ep_moe_fused = K31W4A8EpMoeFused()


#: Largest dispatched row count (ranks x the step's largest per-rank token count) of the fused path: the CuTe W4A8
#: op's decode limit (larger calls take the routed experts' separate path).
MAX_ROWS = 4096

_FUSED_WS: dict = {}
_REAL_ROWS: dict = {}
_PADDED_STEPS_LOGGED = 0


def real_rows_buffer(device: torch.device) -> torch.Tensor:
    """The device's int32 word holding this step's real (non-padding) row count of the attention-DP rank (the fused
    path's dispatch leaves the tail rows past it out); ``2**31 - 1`` (no padding known) until a step publishes one.
    Create it outside CUDA-graph capture."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _REAL_ROWS.get(index)
    if buf is None:
        buf = _REAL_ROWS[index] = torch.full(
            (4,), 2**31 - 1, dtype=torch.int32, device=f"cuda:{index}"
        )
    return buf


def publish_real_rows(num_tokens: int, num_generations: int, num_ctx_tokens: int, request_ids, dummy_id_floor: int,
                      device: torch.device) -> None:  # fmt: skip
    """Write this step's real row count before its forward (eager, capture or graph replay): the CUDA-graph padding
    requests (ids at or above ``dummy_id_floor``) are appended at the tail of the generation requests, each with the
    step's rows per generation request; the rows before them are real."""
    if torch.cuda.is_current_stream_capturing():
        return
    pad = 0
    for request_id in reversed(request_ids or ()):
        if request_id < dummy_id_floor:
            break
        pad += 1
    rows = num_tokens
    if pad and num_generations:
        rows = num_tokens - pad * ((num_tokens - num_ctx_tokens) // num_generations)
    real_rows_buffer(device)[0].fill_(max(0, rows))
    global _PADDED_STEPS_LOGGED
    if pad and _PADDED_STEPS_LOGGED < 16:
        _PADDED_STEPS_LOGGED += 1
        logger.info(
            f"Kimi K3.1 routed MoE fused path: step of {num_tokens} rows, {max(0, rows)} real "
            f"({pad} CUDA-graph padding requests skipped)"
        )


def fused_moe_decline(
    mapping, w3_w1_weight: torch.Tensor, one_sided_exchange: bool
) -> Optional[str]:
    """Why the fused path does not serve this layer's routed experts (``None`` when it does: attention DP over a
    pure-EP group of 2..32 ranks on SM100 / SM103 whose exchange is the one-sided all-to-all, 896 experts top-16,
    full-width experts for the CTA-pair GEMMs).
    ``w3_w1_weight``: the layer's runner-layout FC1 weight ``[local, 2 * inter, 1792]``. ``one_sided_exchange``: the
    layer built the one-sided all-to-all, which only an NVLink fabric spanning the whole EP group gives (its build is
    collective over the group); the fused path's multicast workspace needs the same fabric."""
    checks = (
        (mapping.enable_attention_dp and mapping.tp_size > 1, "no all-to-all"),
        (one_sided_exchange, "no one-sided all-to-all (no NVLink fabric across the EP group)"),
        (_a2a.push_all_to_all_serves(mapping, NUM_EXPERTS, TOP_K), "mapping"),
        (isinstance(w3_w1_weight, torch.Tensor) and w3_w1_weight.dim() == 3, "weights"),
    )
    for ok, why in checks:
        if not ok:
            return why
    w1 = w3_w1_weight
    if w1.shape[2] * w1.element_size() * 2 != HIDDEN or w1.shape[1] // 2 < 2048:
        return f"weight shape {tuple(w1.shape)} {w1.dtype}"
    return None


def fused_moe_max_tokens(mapping, max_num_tokens: int) -> int:
    """Largest per-rank token count the fused path takes (the workspace's capacity)."""
    return max(0, min(int(max_num_tokens), MAX_ROWS // mapping.moe_ep_size))


def fused_moe_workspace(
    mapping, all_rank_num_tokens, max_tokens: int
) -> Optional[_a2a.MoeA2AWorkspace]:
    """The EP group's fused-path workspace for this call, or ``None`` where the call keeps the separate path: a step
    row count above ``max_tokens`` (or none), or a CUDA-graph capture before the group's workspace exists. Every rank
    decides alike: all inputs to the decision are group-wide. The first call outside a CUDA-graph capture allocates
    the group's workspace, whatever its size: every rank makes it in the same layer of the same step (the warmup /
    first step runs eagerly before any capture)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank, max_tokens)
    ws = _FUSED_WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = _FUSED_WS[key] = get_ep_moe_workspace(mapping, max_tokens)
        logger.info(
            f"Kimi K3.1 routed MoE: fused dispatch + experts + combine over {mapping.moe_ep_size} ranks "
            f"(up to {max_tokens} tokens per rank)"
        )
    tmax = max(all_rank_num_tokens)
    if tmax < 1 or tmax > max_tokens:
        return None
    return ws


def real_rows_of(device: torch.device) -> Optional[torch.Tensor]:
    """The device's published real-row word (:func:`publish_real_rows`), or None before any step published one."""
    return _REAL_ROWS.get(device.index if device.index is not None else torch.cuda.current_device())


__all__ = [
    "MAX_ROWS",
    "N_TILE",
    "get_ep_moe_workspace",
    "k31_w4a8_ep_moe_fused",
    "fused_moe_max_tokens",
    "fused_moe_decline",
    "fused_moe_workspace",
    "publish_real_rows",
    "real_rows_buffer",
    "real_rows_of",
    "slot_bytes",
]

# the flag words of the fused path sit clear of the all-to-all kernels' words
assert (
    _a2a.DFLAGS + _a2a.MAX_WORLD <= 288 and _a2a.CFLAGS + _a2a.MAX_WORLD * _a2a.MAX_CHUNKS <= 40960
)
assert 40960 + 32 * _a2a.MAX_WORLD <= _a2a.FLAGS_BYTES // 4
