# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 row-parallel BF16 o_proj and its TP all-reduce as one push chain over the MNNVL workspace."""

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    allreduce_module,
    mnnvl_group,
)

from ...kernels.comm import bf16_ar_push_gemm as _push_gemm

#: Kimi K3.1's hidden size (o_proj output features).
_HIDDEN = 7168
#: o_proj input features per rank at attention TP 4 / 16 / 32 (96 heads x 128 = 12288 split).
_K_TP4, _K_TP16, _K_TP32 = 3072, 768, 384
_NONE = int(AllReduceFusionOp.NONE)

#: One-shot transport (multicast every row into every rank) up to this many bytes of the full
#: message (``rows * hidden * world * 2``); two-shot above. The engine's MNNVL threshold by default,
#: so the transport is the engine kernel's own; either transport gives the same bits.
DEFAULT_ONE_SHOT_MAX_BYTES = _dist_ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES

#: Lamport views of the engine MNNVL workspaces, keyed by the workspace's local buffer address,
#: its size and the rank (a grown workspace is a new buffer, so a new view).
_VIEWS: dict = {}


def lamport_workspace(
    all_reduce: AllReduce, rows: int, hidden: int, one_shot: bool
) -> "_lam.LamportWorkspace":
    """The Lamport view of `all_reduce`'s MNNVL workspace, grown (outside graph capture) to cover a
    call of `rows` x `hidden` BF16 on the `one_shot` transport. Never more than the engine all-reduce
    of the same call needs, so a workspace the engine pre-scaled for its largest call is never grown
    here."""
    group_mapping, dtype = mnnvl_group(all_reduce)
    world = group_mapping.tp_size
    need = _lam.required_bytes(rows, hidden, world, one_shot)
    workspace = _dist_ops.get_or_scale_allreduce_mnnvl_workspace(
        group_mapping, dtype, buffer_size_bytes=need
    )
    key = (
        workspace["uc_buffer"].data_ptr(),
        int(workspace["buffer_size_bytes"]),
        group_mapping.tp_rank,
    )
    view = _VIEWS.get(key)
    if view is None:
        view = _lam.LamportWorkspace.from_mnnvl(workspace, world, group_mapping.tp_rank)
        _VIEWS[key] = view
    return view


def push_serves(input: torch.Tensor, weight: torch.Tensor, all_reduce: Optional[AllReduce]) -> bool:
    """Whether the push chain serves this call (else the op runs cuBLAS + the engine all-reduce)."""
    group = mnnvl_group(all_reduce)
    return group is not None and group[0].tp_size > 1 and _push_gemm.can_push(input, weight)


class OprojAllreducePush(OpWrapper):
    """`all_reduce(input @ weight^T)`: the row-parallel o_proj and the TP sum of its partials.

    `input` [M, K] and `weight` [N, K] are contiguous bf16 (K = this rank's slice of the heads),
    `all_reduce` the engine `AllReduce` module the target built for the attention TP group. Where the
    group runs MNNVL and the push GEMM has a tactic for (M, N, K), two kernels run: a tcgen05 low-M
    GEMM whose epilogue stores its bf16 rows straight into the Lamport slots of the MNNVL workspace
    (one-shot: multicast to every rank; two-shot: to the row's owner), then the Lamport finish
    (poll, rank-order fp32 sum, round, broadcast for two-shot). `DEFAULT_ONE_SHOT_MAX_BYTES` picks
    the transport by full message size (`M * N * world * 2`). Anything else runs `F.linear` and the
    engine all-reduce (`allreduce_module`, NONE), which is the current target path.

    Returns a new bf16 [M, N] tensor, identical on every rank of the group.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 o_proj + AR, K={k} ({layout}), M={rows}: {why}",
                spec=dict(world=4, rows=rows, k=k, hidden=_HIDDEN),
            )
            for k, layout, rows_list in (
                (
                    _K_TP32,
                    "TEP32 shard",
                    (
                        (1, "decode row"),
                        (8, "ll B1 verify block"),
                        (32, "one 32-row token tile"),
                        (48, "one 48-row token tile"),
                        (64, "B12-B13 verify rows"),
                        (128, "one 128-row token tile"),
                        (192, "last row count with a tactic"),
                        (256, "past the table: fallback"),
                    ),
                ),
                (
                    _K_TP16,
                    "TEP16 shard",
                    (
                        (1, "decode row"),
                        (6, "mid B1 verify rows"),
                        (24, "mid verify rows"),
                        (72, "M2 B12 verify rows"),
                        (80, "last one-tile row count"),
                        (128, "two 64-row token tiles"),
                        (192, "last row count with a tactic"),
                        (256, "past the table: fallback"),
                    ),
                ),
                (
                    448,
                    "DFlash2 drafter MLP down, TEP32 shard: a K tail read as zeros",
                    (
                        (8, "ll B1 drafter block"),
                        (24, "three requests' blocks"),
                        (40, "past the table: fallback"),
                    ),
                ),
                (_K_TP4, "TEP4 shard", ((8, "no tactic for this K: fallback"),)),
            )
            for rows, why in rows_list
        ),
    )
    note = """
    Numerics. The all-reduce half is bitwise by construction: the finish sums
    the pushed partials in the engine kernel's order (rank order; chunks of 8
    ranks above 8 for the one-shot kernel, up to 16 for two-shot) with the
    engine's -0 / +0 handling, so push GEMM + finish equals the engine MNNVL
    all-reduce of the push GEMM's partials (FlashInfer's split-K GEMM at the
    same tactic gives the same partials). The GEMM half is not bitwise by
    construction: the push GEMM accumulates K in its own tile order (FP32, one
    rounding per rank partial). Measured against the current path (cuBLAS
    o_proj + engine all-reduce) it gave the same bits in every draw: every
    cell here, and random draws at 4, 16 and 32 ranks over the tactic table
    (all split_k = 1 for K = 256 / 384 / 512 / 768 / 896, the K order cuBLAS
    takes on these shapes); a cuBLAS heuristic change could break that, so a
    target switching to this op keeps an in-model identity check (greedy
    probes). Deterministic, and the same bits on every rank.

    Workspace: the engine MNNVL Lamport workspace of the group (shared with
    every other MNNVL all-reduce of the group; calls alternate on its rotation
    state). The call grows it eagerly when needed; grow it before CUDA-graph
    capture (a warm-up call at the largest row count, or
    `MNNVLAllReduce.prescale_workspace`). Calls that share the workspace must be
    in the same order on every rank, ordered by stream, and every PDL kernel
    between two calls must execute griddepcontrol.wait.

    Tactics (push GEMM) exist for (N, K) = (7168, 1536) up to 72 rows,
    (7168, 768) and (7168, 384) up to 192 rows; other shapes and row counts
    fall back. At K = 384 (attention TP32) above 48 rows the push chain is
    bitwise to the current path by construction of its tiles: cuBLAS runs
    these shapes as one whole-K kernel (no K split) and every single-pass
    tile accumulates K in the same order, and the finish sums the 32 ranks in
    the engine two-shot kernel's two 16-rank chunks. The cells run the bar shapes' K (the attention TP16 /
    TP32 shards) over a four-rank group on one tray, MNNVL forced: the GEMM
    half is the served one, the all-reduce half the same kernels with W = 4.
    The served groups (W = 16 / 32, multi-tray MNNVL) were driven by the
    microbenchmark, not by these cells.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
    ) -> torch.Tensor:
        if not push_serves(input, weight, all_reduce):
            return allreduce_module(
                torch.nn.functional.linear(input, weight), all_reduce, _NONE, None, None
            )
        rows, hidden = input.shape[0], weight.shape[0]
        world = mnnvl_group(all_reduce)[0].tp_size
        one_shot = _lam.one_shot_for(rows, hidden, world, DEFAULT_ONE_SHOT_MAX_BYTES)
        ws = lamport_workspace(all_reduce, rows, hidden, one_shot)
        cpp_one_shot = _lam.one_shot_for(
            rows, hidden, ws.world, _dist_ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES
        )
        out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=input.device)
        use_pdl = get_env_enable_pdl()
        _push_gemm.run_bf16_gemm_ar_push(input, weight, ws, one_shot, out, use_pdl=use_pdl)
        return _lam.run_lamport_finish(
            ws, out, one_shot, use_pdl=use_pdl, cpp_one_shot=cpp_one_shot
        )

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        all_reduce: AllReduce,
    ) -> torch.Tensor:
        """What every rank should get back.

        The collective exception to mirroring `__call__`: takes every rank's `input` and `weight`,
        in rank order (`all_reduce` is ignored). The sum over ranks of
        `input_r @ weight_r^T` in fp32, rounded once to bf16.
        """
        acc = None
        with true_fp32_matmul():
            for x, w in zip(inputs_in_rank_order, weights_in_rank_order):
                part = x.float() @ w.float().t()
                acc = part if acc is None else acc + part
        return acc.to(torch.bfloat16)

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
    ) -> None:
        assert input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
            f"bf16 input and weight only; got {input.dtype} / {weight.dtype}"
        )
        assert input.dim() == 2 and weight.dim() == 2 and input.shape[1] == weight.shape[1], (
            f"input [M, K] and weight [N, K]; got {tuple(input.shape)} / {tuple(weight.shape)}"
        )


oproj_allreduce_push = OprojAllreducePush()
