# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks through the all-reduce autotuner: the kernel the built-in ``AllReduce`` module's AUTO runs when
its TP group does not run over MNNVL. The Kimi K3 all-reduce module (``k3_allreduce.K3AllReduce``) calls it there."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    allreduce as _allreduce,
)
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

NONE = int(AllReduceFusionOp.NONE)
RMS_NORM = int(AllReduceFusionOp.RMS_NORM)
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The strategy the built-in module passes to the autotuner.
AUTO = int(AllReduceStrategy.AUTO)


def autotune_enabled() -> bool:
    """Whether the built-in module's AUTO runs the autotuner for its next call off MNNVL: MPI is on and
    ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE`` is not ``1``. The built-in module reads the variable on every call; so does
    this. Otherwise it runs the raw op (``catalog/comm/allreduce``) with strategy AUTO."""
    from tensorrt_llm._utils import mpi_disabled

    return os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0") != "1" and not mpi_disabled()


def autotuned_allreduce_args(mapping) -> dict:
    """The communicator side of the call for this rank's TP group, built as the built-in module builds it.

    A target calls this once in ``__init__``, while the model is built (the engine's model extra attributes are
    set), and passes ``**args`` at every call site. Like the built-in module's constructor it:

    * builds the mapping's IPC workspace (shared with ``catalog/comm/allreduce`` and every built-in module of the
      mapping);
    * reads ``allreduce_exclude_nccl_symmetric`` from the model extra attributes, or
      ``TRTLLM_AR_TUNE_NCCL_SYMMETRIC != 1``: NCCL_SYMMETRIC is then neither tuned nor run (plain NCCL instead);
    * hands the model's ``allreduce_max_num_tokens``, ``allreduce_hidden_size`` and ``allreduce_dtype`` to the
      autotuner, which sizes the NCCL window buffers it preallocates while it tunes NCCL_SYMMETRIC.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm._torch.utils import get_model_extra_attrs

    extra_attrs = get_model_extra_attrs()
    exclude = bool(extra_attrs and extra_attrs.get("allreduce_exclude_nccl_symmetric", False)) or (
        os.environ.get("TRTLLM_AR_TUNE_NCCL_SYMMETRIC", "1") != "1"
    )
    if extra_attrs:
        max_num_tokens = extra_attrs.get("allreduce_max_num_tokens")
        hidden_size = extra_attrs.get("allreduce_hidden_size")
        dtype = extra_attrs.get("allreduce_dtype")
        if max_num_tokens is not None:
            AllReduceRunner._prealloc_max_num_tokens = max_num_tokens
        if hidden_size is not None:
            AllReduceRunner._prealloc_hidden_size = hidden_size
        if dtype is not None:
            AllReduceRunner._prealloc_dtype = dtype
    return dict(
        workspace=get_allreduce_workspace(mapping),
        group=list(mapping.tp_group),
        exclude_nccl_symmetric=exclude,
    )


def _k3_cells() -> tuple[Cell, ...]:
    """The Kimi K3 targets' all-reduce module (``k3_allreduce.K3AllReduce``) off MNNVL, e.g. TP16 over
    two 8-GPU B200 nodes: the plain sum of a row-parallel output at the hidden size and at the routed experts' latent
    width, and the drafter's fused sums (DSpark's residual + RMSNorm, the context projection's RMSNorm). One node
    holds at most eight ranks."""
    shapes = (
        ("plain sum at hidden 7168", 7168, NONE, (1, 8, 64, 2048, 8192)),
        ("plain sum at the routed latent width 3584", 3584, NONE, (1, 8, 64, 2048, 8192)),
        ("residual + RMSNorm at hidden 7168", 7168, RESIDUAL_RMS_NORM, (1, 8, 64)),
        ("RMSNorm at hidden 7168", 7168, RMS_NORM, (1, 8, 64)),
    )
    return tuple(
        Cell(
            why=f"kimi k3, {name}, {rows} rows, tp{world}",
            spec=dict(world=world, rows=rows, hidden=hidden, dtype=torch.bfloat16, eps=1e-6, op=op),
        )
        for world in (2, 4, 8)
        for name, hidden, op, rows_list in shapes
        for rows in rows_list
    )


class AutotunedAllreduce(OpWrapper):
    """Sum ``input`` over every rank in ``group`` with the kernel the all-reduce autotuner picks for its shape;
    with ``op`` a fusion, also the fusion (``RMS_NORM``: normalize the sum; ``RESIDUAL_RMS_NORM``: add ``residual``,
    normalize, return ``(norm_out, residual_out)``).

    The call the built-in ``AllReduce`` module makes with strategy AUTO when MNNVL does not take it and
    ``autotune_enabled()`` holds: ``autotuned_allreduce`` (the native tactic lookup), or ``tunable_allreduce``
    (the Python autotuner) while the autotuner tunes or with ``exclude_nccl_symmetric``. A shape the autotuner has
    not tuned runs NCCL_SYMMETRIC (plain NCCL with ``exclude_nccl_symmetric``).

    NCCL_SYMMETRIC outputs live in NCCL window buffers. Each is leased until an NCCL window tensor scope ends
    (``tensorrt_llm._torch.nccl_window_tensor_scope``): the engine wraps every model forward in one, and a CUDA
    graph capture binds them to its pool (``nccl_window_graph_capture``). A caller outside the engine opens a
    scope around its calls; an output freed outside one is a lifecycle violation, which aborts the process.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = _k3_cells()

    note = """
    Certified untuned (each shape's tactic is the cache-miss fallback, NCCL_SYMMETRIC) and after an
    `autotune()` pass over the cells' shapes. The tactic a tuned shape runs depends on the tuning run, so a
    target certifies its numbers per tactic, not per entry: every tactic sums in fp32 or in bf16 per rank
    addition, inside `catalog/comm/allreduce`'s band.

    The quantizing fusions pass through to the op and are not certified. The built-in module's MNNVL,
    symmetric-memory and torch.distributed paths are not this entry's.

    Two calls in one process with the same arguments look up the same tactic: the autotuner's cache is per
    process, keyed by the group, dtype, fusion, whether the input lives in an NCCL window, and the
    NCCL_SYMMETRIC exclusion.
    """

    def __call__(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        exclude_nccl_symmetric: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """The built-in module's choice between the two autotuner symbols, read at call time as it reads it."""
        from tensorrt_llm._torch.distributed import ops as _ops

        if _ops._ALLREDUCE_AUTOTUNER_TUNING_MODE or exclude_nccl_symmetric:
            outs = torch.ops.trtllm.tunable_allreduce(
                input=input,
                residual=residual,
                norm_weight=norm_weight,
                scale=scale,
                bias=bias,
                workspace=workspace,
                group=group,
                strategy=AUTO,
                op=op,
                eps=eps,
                trigger_completion_at_end=trigger_completion_at_end,
                exclude_nccl_symmetric=exclude_nccl_symmetric,
            )
        else:
            outs = torch.ops.trtllm.autotuned_allreduce(
                input=input,
                residual=residual,
                norm_weight=norm_weight,
                scale=scale,
                bias=bias,
                workspace=workspace,
                group=group,
                strategy=AUTO,
                op=op,
                eps=eps,
                trigger_completion_at_end=trigger_completion_at_end,
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        op: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """``catalog/comm/allreduce``'s reference: every tactic computes the same sum and fusion. The documented
        collective exception to mirroring ``__call__``: it takes every rank's ``input``, in rank order."""
        return _allreduce.reference(inputs_in_rank_order, residual, norm_weight, eps, op)

    def is_valid(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        exclude_nccl_symmetric: bool,
    ) -> None:
        # The tactics include ONESHOT / TWOSHOT, which read the IPC workspace and every operand as packed rows.
        _allreduce.is_valid(
            input,
            residual,
            norm_weight,
            scale,
            bias,
            workspace,
            group,
            AUTO,
            op,
            eps,
            trigger_completion_at_end,
        )


autotuned_allreduce = AutotunedAllreduce()
