# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finish an MNNVL Lamport all-reduce whose input every rank pushed (``k3_lamport_push`` or a
GEMM's push epilogue): poll the slots, sum them in the C++ kernel's order, re-arm the previous
call's buffer and advance the workspace (CuTe DSL, ``kernels/k3_lamport_ar.py``)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm.k3_lamport_ar import LamportWorkspace, reduce_chunk, run_lamport_finish


def sanitized(x: torch.Tensor) -> torch.Tensor:
    """BF16 ``x`` with ``-0`` replaced by ``+0``: what a push stores into a Lamport slot (the
    sentinel word is a ``-0`` pair)."""
    return torch.where(x.view(torch.int16) == -32768, torch.zeros_like(x), x)


def lamport_sum_reference(inputs_in_rank_order: Sequence[torch.Tensor], chunk: int) -> torch.Tensor:
    """BF16 sum of the ranks' pushed rows: FP32 partial sums of ``chunk`` consecutive ranks, each
    from zero, added in rank order and rounded once (the C++ MNNVL kernels' order)."""
    total: Optional[torch.Tensor] = None
    for base in range(0, len(inputs_in_rank_order), chunk):
        acc = torch.zeros_like(inputs_in_rank_order[0], dtype=torch.float32)
        for part in inputs_in_rank_order[base : base + chunk]:
            acc = acc + sanitized(part).float()
        total = acc if total is None else total + acc
    return total.to(torch.bfloat16)


class K3LamportFinish(OpWrapper):
    """Write the all-reduce of the call every rank pushed on ``ws`` into ``out`` (contiguous BF16
    ``[M, hidden]``) and return ``out``. ``one_shot``: the transport the producers used;
    ``cpp_one_shot``: the C++ kernel whose bits to reproduce (default: the same transport);
    ``ag``: the call carries all-gather rows, ``out = bf16(y + bf16(sum))``."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The group is the test harness's (2 and 4 GPUs of one node); K3 runs 8 (helix8) and 16 (helix16 ULL).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=why,
            spec=dict(rows=rows, hidden=7168, one_shot=one_shot, cpp_one_shot=cpp_one_shot),
        )
        for why, rows, one_shot, cpp_one_shot in (
            ("one token, one-shot", 1, True, True),
            ("helix16 ULL 1-request step: 8 rows, one-shot", 8, True, True),
            ("helix16 ULL 4-request step: 32 rows, two-shot", 32, False, False),
            ("two-shot transport reproducing the one-shot C++ sums: 24 rows", 24, False, True),
            ("one-shot transport reproducing the two-shot C++ result: 16 rows", 16, True, False),
            ("helix8 b16 mtp5 verify step: 96 rows, two-shot", 96, False, False),
        )
    )

    note = """
    The second half of a pushed all-reduce. ws (kernels/k3_lamport_ar
    .LamportWorkspace) holds the workspace's device addresses: this rank's
    unicast buffer, the multicast base, every rank's unicast mapping and the
    uint32[9] rotation state the C++ MNNVL kernels share, so pushed and C++
    calls alternate on one workspace. Writes out, re-arms the previous call's
    buffer with the sentinel and advances the rotation state. Every rank makes
    the same sequence of calls on a workspace, ordered by streams.
    One-shot: every rank sums the W slots of each token; two-shot: the owner of
    token t (t % W) reduces it and multicasts the row, the others copy it.
    Worker CTAs read the state and poll without griddepcontrol.wait (the
    producer releases this grid after its own wait); the updater CTA waits for
    the producer before it advances the state. Sums in FP32 in the C++ order
    (chunks of eight ranks above eight ranks for the one-shot result, up to 16
    in one chunk for the two-shot one); bitwise the C++ all-reduce of the same
    inputs. A process-wide cache (`_compiled_finish`) holds one compiled kernel
    per (W, hidden, transport, PDL, wait_first, cpp_one_shot, ag); compile each
    key eagerly before graph capture. The all-gather variant is reached only
    through the column-parallel MoE tail and is not in this entry's cells.
    """

    def __call__(
        self,
        ws: LamportWorkspace,
        out: torch.Tensor,
        one_shot: bool,
        use_pdl: bool = True,
        wait_first: bool = False,
        cpp_one_shot: Optional[bool] = None,
        ag: bool = False,
    ) -> torch.Tensor:
        return run_lamport_finish(
            ws,
            out,
            one_shot,
            use_pdl=use_pdl,
            wait_first=wait_first,
            cpp_one_shot=cpp_one_shot,
            ag=ag,
        )

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        out: torch.Tensor,
        one_shot: bool,
        use_pdl: bool = True,
        wait_first: bool = False,
        cpp_one_shot: Optional[bool] = None,
        ag: bool = False,
        ag_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The collective exception: every rank's pushed ``[M, hidden]`` rows in rank order instead
        of ``ws``; ``ag_rows`` the all-gathered ``y`` with ``ag``. ``out`` is not read;
        ``use_pdl`` and ``wait_first`` only change scheduling. Returns the expected ``out``."""
        if cpp_one_shot is None:
            cpp_one_shot = one_shot
        chunk = reduce_chunk(len(inputs_in_rank_order), cpp_one_shot)
        total = lamport_sum_reference(inputs_in_rank_order, chunk)
        if ag:
            total = (ag_rows.float() + total.float()).to(torch.bfloat16)
        if not cpp_one_shot:
            total = sanitized(total)
        return total

    def is_valid(
        self,
        ws: LamportWorkspace,
        out: torch.Tensor,
        one_shot: bool,
        use_pdl: bool = True,
        wait_first: bool = False,
        cpp_one_shot: Optional[bool] = None,
        ag: bool = False,
    ) -> None:
        # The kernel stores dense rows of `hidden` BF16 values at out's address.
        assert out.dtype == torch.bfloat16 and out.dim() == 2 and out.is_contiguous(), (
            "out must be a contiguous BF16 [M, hidden] tensor"
        )
        assert out.device == ws.flags.device, "out must be on the workspace's device"


k3_lamport_finish = K3LamportFinish()
