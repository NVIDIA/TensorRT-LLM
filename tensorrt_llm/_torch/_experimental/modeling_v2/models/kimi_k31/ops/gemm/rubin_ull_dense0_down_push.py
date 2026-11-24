# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L0's dense MLP, second half: the down low-M split-K GEMM whose owner CTAs push the BF16 rows into
the Lamport slots of an all-reduce (``kernels/gemm/rubin_ull_k3_dense0.py``'s ``FitPushGemmKernel``)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import rubin_ull_k3_dense0 as _k

#: The TP16 slice of the dense MLP's 33792 columns, K-padded to whole 128-wide tiles.
INTER = 33792 // 16
K_PADDED = _k.pad_k(INTER)


def _cell(why: str, rows: int, world: int, one_shot: bool) -> Cell:
    _, down = _k.default_tactics(rows)
    tactic = (down.mma_m, down.mma_n, down.split_k, down.ab_stages)
    return Cell(why=why, spec=dict(rows=rows, tactic=tactic, world=world, one_shot=one_shot))


class Dense0DownPush(OpWrapper):
    """Push ``bf16(x @ weight^T)`` (``x`` [M, K] the SiTU rows, ``weight`` [7168, K], K a multiple of
    128 with zero padding columns in both) into ``target``'s all-reduce: ``target.workspace`` (a
    ``k3_lamport_ar.LamportWorkspace``: ``world``, ``rank``, ``mc_base``, ``peer_ptrs``, ``flags``),
    ``target.one_shot`` (multicast store of row ``t`` into slot ``t * world + rank`` of the current
    buffer; else the two-shot store into owner ``t % world``'s slot ``(t // world) * world + rank``),
    ``target.output`` the [M, 7168] tensor that sets the launch (not written). ``-0`` is stored as
    ``+0``. Sets ``target.pushed``; returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "b1 verify, L1 finish: 8 rows one-shot into the AttnRes mailbox (world 1)", 8, 1, True
        ),
        _cell("b1 verify, L0 finish: 8 rows two-shot over TP16", 8, 16, False),
        _cell("b2 verify: 16 rows two-shot over TP16", 16, 16, False),
        _cell("b4 verify: 32 rows two-shot over TP16 (two slot rows per owner)", 32, 16, False),
        _cell("q1 decode, 4 requests: 4 rows one-shot over TP16 (multicast)", 4, 16, True),
    )

    note = """
    The producer half of the served L0 dense all-reduce; the consumer (``run_lamport_finish``, or
    L1's AttnRes reading the mailbox) completes it. The current Lamport buffer and its size are read
    from ``flags`` on the device, so the launch is graph-capturable. Compiled per (tactic, PDL,
    world, one_shot) on first use (``FlatDense0`` precompiles before capture). PDL: the weight
    streams before the grid dependency wait and the dependents are released after the activation
    warp's wait.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, target, tactic, pdl: bool) -> None:
        _k.run_down_push(x, weight, target, tactic, pdl=pdl)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, target, tactic, pdl: bool):
        """The [M, 7168] BF16 rows the kernel pushes: the FP64 product rounded once. ``target``,
        ``tactic`` and ``pdl`` are not read (where the rows land is the slot rule above)."""
        return (x.double() @ weight.double().t()).to(torch.bfloat16)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, target, tactic, pdl: bool) -> None:
        """The launcher checks the tactic against the shapes; it reads ``x`` and ``weight`` as dense
        rows from their base pointers and ``target.output`` only for its shape. (``workspace.rank``
        is a slot offset, not checked against ``world``: the L1 mailbox passes ``world`` 1 and
        ``rank`` = rank x mailbox rows.)"""
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("dense0_down_push: x and weight must be contiguous")
        if x.shape[1] != weight.shape[1] or x.shape[1] % 128:
            raise ValueError("dense0_down_push: K must match and be whole 128-wide tiles")
        if tuple(target.output.shape) != (x.shape[0], weight.shape[0]):
            raise ValueError("dense0_down_push: target.output must be [M, N]")


dense0_down_push = Dense0DownPush()
