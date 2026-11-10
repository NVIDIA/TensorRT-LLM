# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode block's TP sum and the hyper-connection site's injection + grouped norm in one launch.

`native_ops.hc_allreduce_inject_norm` (`csrc/hcAllreduceInjectNorm.cu`) does the IPC one-shot
Lamport all-reduce of `catalog/comm/allreduce` over the same workspace and protocol (so it
interleaves with the plain all-reduce's calls), then `_hc_inject_norm_launch`'s injection and
norm from the summed row, its sum of squares in the Triton program's reduction tree at
`program_warps` warps. Both outputs are bitwise the all-reduce followed by the Triton kernel.

Only the norm weights are read before the grid-dependency wait; the residual, injection weights
and shared-expert operands are read after it.
"""

from __future__ import annotations

import torch

from ... import native_ops

HIDDEN = 2560
HC_COUNT = 4


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and t.stride(0) % 8 == 0


def fits(hc, partial, inject, shared=None, gate=None) -> bool:
    """Whether the native kernel takes these operands (else the caller sums and injects apart)."""
    rows = hc.shape[0]
    if partial.dim() != 2 or partial.shape != (rows, HIDDEN) or not partial.is_contiguous():
        return False
    if hc.shape != (rows, HC_COUNT * HIDDEN) or not _aligned(hc):
        return False
    if inject is None or inject.shape[0] != rows or inject.shape[1] < HC_COUNT:
        return False
    if inject.stride(-1) != 1:
        return False
    if (shared is None) != (gate is None):
        return False
    if shared is not None and (shared.shape != (rows, HIDDEN) or not _aligned(shared)):
        return False
    dtypes = {t.dtype for t in (hc, partial, inject, shared, gate) if t is not None}
    return dtypes == {torch.bfloat16}


def hc_allreduce_inject_norm(
    partial: torch.Tensor,
    workspace: torch.Tensor,
    rank: int,
    nranks: int,
    hc: torch.Tensor,
    inject: torch.Tensor,
    norm: torch.Tensor,
    eps: float,
    shared: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    program_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(hc_out, normed)` of the site after a block whose per-rank output is `partial`: bitwise
    `_hc_inject_norm_launch(hc, tp_sum(partial), inject, norm, eps, 4, shared, gate,
    program_warps)`."""
    hc_out, normed = native_ops.hc_allreduce_inject_norm(
        partial,
        workspace,
        int(rank),
        int(nranks),
        hc,
        inject,
        shared,
        gate,
        norm,
        float(eps),
        int(program_warps),
        # trigger_completion_at_end=False: release the next launch right after this kernel's wait.
        False,
    )
    return hc_out, normed
