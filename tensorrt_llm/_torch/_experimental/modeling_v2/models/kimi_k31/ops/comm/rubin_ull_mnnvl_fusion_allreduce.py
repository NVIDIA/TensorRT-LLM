# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP16 MNNVL Lamport all-reduce on a workspace the target owns: the stock
``torch.ops.trtllm.mnnvl_fusion_allreduce`` with no fusion, a fused RMSNorm, or a fused residual
add + RMSNorm."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES
from tensorrt_llm.functional import AllReduceFusionOp

NONE = int(AllReduceFusionOp.NONE)
RMS_NORM = int(AllReduceFusionOp.RMS_NORM)
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)
FUSIONS = (NONE, RMS_NORM, RESIDUAL_RMS_NORM)

#: K3.1's hidden size and RMSNorm epsilon.
HIDDEN = 7168
EPS = 1e-5
#: The FP32 target-head pair exchange's width at TP16: the next power of two of
#: ``2 * 16 * ceil(10240 / 2048)`` (a 163840-token vocabulary over 16 ranks, 2048-column blocks).
PAIR_WIDTH = 256


def lamport_bytes(rows: int, hidden: int, world: int, element_size: int) -> int:
    """Bytes of one Lamport buffer the call needs: every rank's rows for the one-shot kernel, two
    stages of a world-aligned share for the two-shot one (the C++ op picks by the same threshold)."""
    one_shot = rows * hidden * world * element_size
    if one_shot <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
        return one_shot
    return 2 * -(-rows // world) * world * hidden * element_size


def _cell(why: str, rows: int, fusion: int = NONE, dtype=torch.bfloat16, hidden: int = HIDDEN):
    return Cell(why=why, spec=dict(rows=rows, hidden=hidden, dtype=dtype, fusion=fusion, eps=EPS))


class MnnvlFusionAllreduce(OpWrapper):
    """Sum ``input`` [rows, hidden] over the group of ``comm_buffer``'s multicast workspace, then,
    by ``fusion``:

    - ``NONE``: return the sum (a new tensor of ``input``'s dtype);
    - ``RMS_NORM``: return ``rmsnorm(sum) * norm_weight``;
    - ``RESIDUAL_RMS_NORM``: ``h = sum + residual``; return ``(rmsnorm(h) * norm_weight, h)``.

    ``comm_buffer`` is the [3, n] view (in ``input``'s dtype) of the workspace's three Lamport
    buffers, ``buffer_flags`` its nine uint32 rotation words; the call reads the current buffer,
    re-arms the previous call's and advances the rotation. ``world`` is the group size; the kernel
    reads it from the workspace, ``is_valid`` sizes the call with it. Inputs are not written.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("q1 decode, one request: the one-shot kernel at every world", 1),
        _cell("b1 q8 verify: the L0 KDA o-projection reduce (8 rows)", 8),
        _cell("b2 q8 verify (16 rows)", 16),
        _cell("b4 q8 verify (32 rows): the two-shot kernel at world 4 and 16", 32),
        _cell("b5 q8 verify (40 rows): the dense fallback reduce past the fused L0 MLP", 40),
        _cell("b8 q8 verify (64 rows): the largest served batch", 64),
        _cell("drafter hidden norm: all-reduce + RMSNorm, b1", 8, RMS_NORM),
        _cell("drafter hidden norm: all-reduce + RMSNorm, b4", 32, RMS_NORM),
        _cell("drafter layer: all-reduce + residual add + RMSNorm, b1", 8, RESIDUAL_RMS_NORM),
        _cell("drafter layer: all-reduce + residual add + RMSNorm, b8", 64, RESIDUAL_RMS_NORM),
        _cell("target head FP32 pair exchange, b1", 8, dtype=torch.float32, hidden=PAIR_WIDTH),
        _cell("target head FP32 pair exchange, b8", 64, dtype=torch.float32, hidden=PAIR_WIDTH),
    )

    note = """
    The K3.1 ULL target's ``comm.FlatAllReduce``: every one of its reductions (the L0 KDA
    o-projection, the dense L0 fallback, the drafter's and its context stream's, the FP32 target
    head pair exchange) is one call of this on its own workspace. A workspace is a private
    ``McastGPUBuffer`` of three Lamport buffers filled with FP32 -0.0 and flags
    ``[0, 2, bytes_per_buffer, 0, 0, 0, 0, 0, 0]``; one workspace serves one dtype and one stream,
    in the same order on every rank. The C++ op chooses the one-shot kernel up to
    ``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES`` of ``rows * hidden * world * element_size`` (1 MiB by
    default), the two-shot kernel above. Launched with PDL where ``TRTLLM_ENABLE_PDL`` is set.
    Cells are driven at the node's world (4 GB300 ranks); the served world is 16.
    """

    def __call__(
        self,
        input: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        fusion: int,
        world: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        result = torch.ops.trtllm.mnnvl_fusion_allreduce(
            input,
            norm_weight,
            residual,
            eps,
            comm_buffer,
            buffer_flags,
            norm_weight is not None,
            None,
            fusion,
        )
        return tuple(result) if len(result) > 1 else result[0]

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        eps: float,
        fusion: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """The collective exception to mirroring ``__call__``: every rank's ``input`` in rank order
        (and no workspace). FP32 sum rounded once; the norm reads the FP32 value (with the
        residual added in FP32 for ``RESIDUAL_RMS_NORM``)."""
        acc = torch.zeros(inputs_in_rank_order[0].shape, dtype=torch.float32, device="cuda")
        for x in inputs_in_rank_order:
            acc += x.float()
        dtype = inputs_in_rank_order[0].dtype
        if fusion == NONE:
            return acc.to(dtype)
        h = acc if fusion == RMS_NORM else acc + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        out = (normed * norm_weight.float()).to(dtype)
        return out if fusion == RMS_NORM else (out, h.to(dtype))

    def is_valid(
        self,
        input: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        fusion: int,
        world: int,
    ) -> None:
        """Where the kernel answers wrongly without a check of its own: a call larger than the
        Lamport buffer, a buffer viewed in another dtype (its empty value is the dtype's -0.0), a
        fusion outside the three this entry runs, a norm weight of another dtype or width, or a
        residual of another shape."""
        if fusion not in FUSIONS:
            raise ValueError(
                f"mnnvl_fusion_allreduce: fusion {fusion} is not NONE / RMS_NORM / RESIDUAL_RMS_NORM"
            )
        if comm_buffer.dtype != input.dtype or comm_buffer.dim() != 2 or comm_buffer.shape[0] != 3:
            raise ValueError(
                "mnnvl_fusion_allreduce: comm_buffer must be the [3, n] view in input's dtype"
            )
        hidden = input.shape[-1]
        rows = input.numel() // hidden
        need = lamport_bytes(rows, hidden, world, input.element_size())
        have = comm_buffer.shape[1] * comm_buffer.element_size()
        if need > have:
            raise ValueError(
                f"mnnvl_fusion_allreduce: {rows} rows need {need} bytes per Lamport buffer, the "
                f"workspace holds {have}"
            )
        if fusion != NONE and (
            norm_weight is None
            or norm_weight.dtype != input.dtype
            or norm_weight.shape != (hidden,)
        ):
            raise ValueError(
                "mnnvl_fusion_allreduce: the norm weight must be [hidden] in input's dtype"
            )
        if fusion == NONE and norm_weight is not None:
            raise ValueError(
                "mnnvl_fusion_allreduce: a norm weight turns NONE into RESIDUAL_RMS_NORM"
            )
        if fusion == RESIDUAL_RMS_NORM and (residual is None or residual.shape != input.shape):
            raise ValueError("mnnvl_fusion_allreduce: the residual must have input's shape")


mnnvl_fusion_allreduce = MnnvlFusionAllreduce()
