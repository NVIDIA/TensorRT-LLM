# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the K3.1 Rubin Lamport all-reduce entries share: the all-reduce reference, the row guard
and the note. Not an entry: the comm entries beside it import from it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

if TYPE_CHECKING:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace


def reduce_chunk(world: int, cpp_one_shot: bool) -> int:
    """Ranks per partial sum of the C++ kernel a finish reproduces: chunks of eight ranks for the
    one-shot kernel above eight ranks, up to sixteen ranks in one chunk for the two-shot one."""
    return (world if world <= 8 else 8) if cpp_one_shot else min(world, 16)


def sanitize_bf16(x: torch.Tensor) -> torch.Tensor:
    """``x`` with every bf16 ``-0`` replaced by ``+0`` (what enters a Lamport slot)."""
    return torch.where(x == 0, torch.zeros_like(x), x)


def lamport_allreduce_reference(
    inputs_in_rank_order: Sequence[torch.Tensor],
    chunk: int,
    ag_rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The bf16 all-reduce a finish produces, from every rank's partial in rank order.

    Each partial with ``-0`` sent as ``+0``; fp32 sums, round to nearest, ``chunk`` ranks per
    partial sum (each from 0), the partial sums added in rank order; rounded once to bf16, then
    ``-0`` as ``+0``. ``ag_rows`` (the call's all-gather rows): ``bf16(y + bf16(sum))``.
    """
    first = inputs_in_rank_order[0]
    acc = torch.zeros(first.shape, dtype=torch.float32, device=first.device)
    world = len(inputs_in_rank_order)
    for c0 in range(0, world, chunk):
        part = torch.zeros_like(acc)
        for r in range(c0, min(c0 + chunk, world)):
            part = part + sanitize_bf16(inputs_in_rank_order[r]).float()
        acc = acc + part
    out = sanitize_bf16(acc.to(torch.bfloat16))
    if ag_rows is not None:
        out = (ag_rows.float() + out.float()).to(torch.bfloat16)
    return out


def check_rows(name: str, ws: LamportWorkspace, x: torch.Tensor) -> None:
    assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous(), (
        f"{name} must be a contiguous bf16 [m, hidden] tensor"
    )
    assert x.is_cuda and x.device == ws.flags.device, f"{name} must be on the workspace's device"


LAMPORT_NOTE = """
    Calls that share a workspace must be issued in the same order by every rank and ordered by
    streams; every PDL kernel between two calls must execute griddepcontrol.wait. The producer
    reads the rotation state after its own wait; the finish workers poll as soon as they are
    resident. Results are bitwise the C++ MNNVL all-reduce of the same partials; ``use_pdl``
    changes launch scheduling, not results. The launchers raise when the buffer is too small or
    hidden % 8.
    """
