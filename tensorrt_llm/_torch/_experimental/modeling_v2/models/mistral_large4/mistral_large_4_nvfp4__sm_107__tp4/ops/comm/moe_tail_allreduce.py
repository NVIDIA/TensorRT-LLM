# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MoE layer's two closing all-reduces on a generation step, as the one-shot kernels of the
Lamport workspace that `cute_dsl_allreduce_add_rmsnorm` does not launch.

`allreduce_add_rmsnorm_nvfp4` is the post-attention all-reduce + residual add + RMSNorm that
also writes the norm output as the routed experts' NVFP4 input (the workspace's
`fused_quant`): e2m1 codes and linear-layout e4m3 scales, one per 16 columns, under the
layer's static global scale, with fp4_quantize's arithmetic on the same bf16 norm rows.
`allreduce_add2_rmsnorm` is the post-MoE all-reduce + residual add + RMSNorm over the sum of
the routed and shared partials (the workspace's `fused_add2`), added in fp32 and rounded once
to bf16 as `torch.add` of the two is. Both are otherwise the entry's one-shot kernel: the same
transport, slots and signal state, the same sum in rank order, residual update and norm, and
the same PDL release at entry.

One-shot only: a call takes at most `ONE_SHOT_ROWS` rows, every rank the same row count.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import (
    TWOSHOT_ABOVE,
    LamportWorkspace,
    check_rows,
    tail,
)

#: The most rows the one-shot kernels take; the entry runs two-shot above it.
ONE_SHOT_ROWS = TWOSHOT_ABOVE

#: NVFP4: two e2m1 codes per byte, one e4m3 scale per 16 columns.
_E2M1_PER_BYTE = 2
_SF_VEC = 16


def _check(partial: torch.Tensor, workspace: LamportWorkspace) -> None:
    rows = int(partial.shape[0])
    check_rows(rows, workspace)
    assert rows <= ONE_SHOT_ROWS, f"{rows} rows exceed the one-shot kernels' {ONE_SHOT_ROWS}"


def allreduce_add_rmsnorm_nvfp4(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: LamportWorkspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`residual += sum over ranks of partial`; returns `(norm, codes, scales)`.

    `norm` is `rmsnorm(residual) * weight`, bf16 `[rows, hidden]`, the entry's output. `codes`
    is uint8 `[rows, hidden / 2]` and `scales` the flat float8_e4m3fn `[rows * hidden / 16]`
    linear layout, the operands the NVFP4 MoE runner takes. `global_scale` is the layer's fp32
    `[1]` quantizer scale."""
    _check(partial, workspace)
    rows, hidden = partial.shape
    norm = torch.empty_like(partial)
    codes = torch.empty((rows, hidden // _E2M1_PER_BYTE), dtype=torch.int8, device=partial.device)
    scales = torch.empty((rows, hidden // _SF_VEC), dtype=torch.int8, device=partial.device)
    workspace.fused_quant(
        partial, residual, weight, global_scale, norm, codes, scales, residual,
        *tail(workspace, fused=True),
    )  # fmt: skip
    return norm, codes.view(torch.uint8), scales.view(torch.float8_e4m3fn).view(-1)


def allreduce_add2_rmsnorm(
    partial: torch.Tensor,
    partial2: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    workspace: LamportWorkspace,
) -> torch.Tensor:
    """`residual += sum over ranks of bf16(partial + partial2)`; `out = rmsnorm(residual) *
    weight`, written and returned. Needs a workspace built with `add_second_input`."""
    _check(partial, workspace)
    assert workspace.fused_add2 is not None, "the workspace was built without add_second_input"
    workspace.fused_add2(
        partial, partial2, residual, weight, out, residual, *tail(workspace, fused=True)
    )
    return out
