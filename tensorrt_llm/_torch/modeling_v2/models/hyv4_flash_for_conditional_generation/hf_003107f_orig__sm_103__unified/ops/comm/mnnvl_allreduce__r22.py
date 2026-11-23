# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks over multi-node NVLink, then add the residual and
RMS-normalize, at this checkpoint's hidden width and row counts.

Local variant of ``catalog/comm/mnnvl_allreduce.py`` (``__r22``) owned by the
Hy4-Flash unified target: same op call, same reference and same guards, with
``CELLS`` replaced by the shapes this target calls it with. The shared entry and
its cells are unchanged; this file is what the target imports.

The call site is the target's cross-node TP all-reduce at tep8
(``Hyv4FlashCore.forward``, both phases), where the TP group spans two GB300
trays. Read the cell list for the certified range.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.comm.allreduce import allreduce as _allreduce
from tensorrt_llm._torch.modeling_v2.catalog.comm.mnnvl_allreduce import (
    RESIDUAL_RMS_NORM,
    mnnvl_workspace_args,
    workspace_bytes,
)

__all__ = ["MnnvlAllreduce", "mnnvl_allreduce", "mnnvl_workspace_args", "workspace_bytes"]


class MnnvlAllreduce(OpWrapper):
    """Sum `input` over the TP group through a multicast NVLink workspace,
    add `residual`, and RMS-normalize the sum with `norm_weight`.

    Returns fresh `(norm_out, residual_out)` and leaves `input` and `residual`
    untouched, the same contract as `comm/allreduce` with `RESIDUAL_RMS_NORM`,
    so the target can call either one in the same slot.

    The workspace is this op's fixture, built by `mnnvl_workspace_args` for a
    row budget. The kernel does not check the budget; `is_valid` does.
    """

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash geometry at tep8: hidden 3072, one call per layer over
    # every token the rank holds. The row counts are the phase ends and the
    # engine's token envelope; `eps` is the checkpoint's RMSNorm eps.
    _HY4 = dict(hidden=3072, dtype=torch.bfloat16, eps=1e-6, world=8)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, one token: the one-shot kernel at hidden 3072",
            spec=dict(rows=1, **_HY4),
        ),
        Cell(
            why="a 64-row decode batch: past the one-shot threshold, the two-shot kernel",
            spec=dict(rows=64, **_HY4),
        ),
        Cell(
            why="a 5120-token prefill chunk, the target's largest prefill call",
            spec=dict(rows=5120, **_HY4),
        ),
        Cell(
            why="the 8192-token max_num_tokens envelope: the workspace's largest row",
            spec=dict(rows=8192, **_HY4),
        ),
    )

    note = """
    Local variant (`__r22`) of `catalog/comm/mnnvl_allreduce.py`, owned by the
    Hy4-Flash unified target. The shared entry's own cells -- hidden 2880 and
    7168, rows 1/16/64/8192, world 8 -- stay with the shared entry: neither
    width is this checkpoint's.

    Only `RESIDUAL_RMS_NORM` is certified. The op also has a plain sum and
    FP8 / NVFP4 quantizing fusions; none is reachable through this entry.

    The workspace is shared per mapping with the built-in module, through
    `get_or_scale_allreduce_mnnvl_workspace`, and the target builds it for
    `max_num_tokens` 8192 rows: at hidden 3072 that is 100,663,296 bytes of
    Lamport buffer. Rows 1 runs the one-shot kernel; 64, 5120 and 8192 run the
    two-shot kernel. Building the workspace is a collective over the TP group:
    every rank must call `mnnvl_workspace_args`, in the same order.

    Under MPI its communicator is a split of the session communicator; with MPI
    disabled it is the mapping's TP process group, which needs a CPU (gloo)
    backend for the setup handshake.

    Measured on sm_103, world 8 over two GB300 trays, bf16, 5 seeds per cell,
    against `reference` (ulp of the row's largest element / relative RMS):
    rows 1 -> 1.13 / 0.874 on `norm_out`, 1.51 / 0.723 on `residual_out`;
    rows 64 -> 2.37 / 0.843 and 1.72 / 0.701; rows 5120 -> 2.52 / 0.839 and
    1.92 / 0.697; rows 8192 -> 2.52 / 0.838 and 1.97 / 0.697. The test
    certifies at 5 ulp element / 1.25 ulp RMS, and the error is flat in the row
    count: the kernel accumulates in fp32 at every size. CUDA-graph capture and
    replay were driven at batch sizes 1, 8, 13 and 64 rows and re-checked
    against fresh payloads.

    Not certified: hidden widths other than 3072, row counts above 8192, eps
    other than 1e-6, dtypes other than bf16, and worlds other than 8 over two
    GB300 trays.
    """

    def __call__(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        world: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`world` is not passed to the op, which reads it from the workspace;
        it is here so `is_valid` can size the call against the buffer."""
        norm_out, residual_out = torch.ops.trtllm.mnnvl_fusion_allreduce(
            input,
            norm_weight,
            residual,
            eps,
            comm_buffer,
            buffer_flags,
            True,
            None,
            RESIDUAL_RMS_NORM,
        )
        return norm_out, residual_out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The same arithmetic as `comm/allreduce`'s reference: fp32 sum, the
        residual added to it, the norm read from that fp32 value.

        The documented collective exception to mirroring `__call__`: it takes
        every rank's `input`, in rank order.
        """
        return _allreduce.reference(
            inputs_in_rank_order, residual, norm_weight, eps, RESIDUAL_RMS_NORM
        )

    def is_valid(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        world: int,
    ) -> None:
        assert residual.shape == input.shape, (residual.shape, input.shape)
        assert norm_weight.shape == (input.shape[-1],), (norm_weight.shape, input.shape)
        # The kernel indexes the Lamport buffer by the call's size and never
        # checks it against the buffer it was given.
        hidden = input.shape[-1]
        need = workspace_bytes(input.numel() // hidden, hidden, world, input.dtype)
        have = comm_buffer.shape[-1] * comm_buffer.element_size()
        assert need <= have, (
            f"{input.numel() // hidden} rows need {need} bytes of workspace, the buffer "
            f"holds {have}; build it with mnnvl_workspace_args and a larger max_rows"
        )


mnnvl_allreduce = MnnvlAllreduce()
