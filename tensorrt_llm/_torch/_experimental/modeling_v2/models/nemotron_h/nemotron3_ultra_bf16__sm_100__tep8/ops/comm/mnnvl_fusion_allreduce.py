# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum a tensor across the tensor-parallel group over MNNVL, add a residual and RMS-normalize
the result, in one op."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp

#: The one fusion this entry certifies: `norm_out = rmsnorm(sum + residual) * gamma` and
#: `residual_out = sum + residual`. The op also runs the plain sum, an RMSNorm without a residual
#: and four quantizing patterns; none of them is called by a target and none is certified here.
FUSION_RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: Nemotron-3 Ultra's hidden size: every all-reduce this target makes is `[T, 8192]` bf16.
HIDDEN = 8192

#: The op switches kernel at `rows * hidden * ranks * itemsize` bytes (1 MiB, the constant
#: `_MNNVL_ONE_SHOT_THRESHOLD_BYTES` in `_torch/distributed/ops.py`, mirrored in the C++):
#: at or below it the one-shot kernel runs, above it the two-shot one. At 8 ranks and hidden 8192
#: that is 8 rows; at 4 ranks 16.
ONE_SHOT_THRESHOLD_BYTES = 64 * 1024 * 8 * 2


def one_shot(rows: int, hidden: int, ranks: int, itemsize: int = 2) -> bool:
    """Whether the op takes its one-shot kernel for this payload."""
    return rows * hidden * ranks * itemsize <= ONE_SHOT_THRESHOLD_BYTES


def _cell(why: str, rows: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, hidden=HIDDEN, dtype=torch.bfloat16))


class MnnvlFusionAllreduce(OpWrapper):
    """Sum `input` over every rank of the MNNVL group, add `residual`, RMS-normalize.

    Returns `[norm_out, residual_out]`: two fresh tensors of `input`'s shape and dtype on every
    rank, where `residual_out = sum + residual` and `norm_out = residual_out`'s RMS normalization
    times `gamma`. `input` and `residual` are untouched. The kernel reads the group through a
    multicast NVLink workspace: `comm_buffer` is that workspace's unicast view as
    `get_or_scale_allreduce_mnnvl_workspace` hands it out (`workspace["uc_buffer"]` viewed as the
    activation dtype and as `[3, -1]`: three Lamport buffers), `buffer_flags` its protocol state.
    Both belong to one workspace and to the caller: nothing else may launch on it while a call is
    in flight, which is why a target allocates it once, on every rank at the same point, outside
    any CUDA graph capture.

    Which kernel runs is picked by size: one-shot (every rank reads every peer's rows, and the
    whole op is one kernel) up to 1 MiB of `rows * hidden * ranks * itemsize`, two-shot
    (reduce-scatter, then all-gather over the multicast mapping, followed by the normalization
    as a second kernel) above. A workspace must hold the bytes the kernel needs for the call
    (`MNNVLAllReduce.get_required_workspace_size`); a larger call needs a larger workspace, which
    the engine grows -- never inside a graph capture, where it raises.
    """

    ARCHS = frozenset({Arch.SM_100})

    # The row counts a target calls it with: one decode token, the decode batch sizes the CUDA
    # graphs capture up to the serving envelope's 256-row ceiling, and prefill chunks up to the
    # 8192-token step. The
    # one-shot / two-shot boundary moves with the group size (8 rows at 8 ranks, 16 at 4), so
    # the cells name the rows on both sides of both boundaries and the test derives the kernel
    # from the world size it runs at.
    CELLS: tuple[Cell, ...] = (
        _cell("decode: 1 row, one-shot at every group size", 1),
        _cell("the last one-shot row count at 8 ranks", 8),
        _cell("the first two-shot row count at 8 ranks (one-shot at 4)", 9),
        _cell("the last one-shot row count at 4 ranks", 16),
        _cell("the first two-shot row count at 4 ranks", 17),
        _cell("the decode CUDA-graph ceiling at max_batch_size 32: 32 rows", 32),
        _cell("the serving envelope's decode ceiling: 256 rows", 256),
        _cell("a mid prefill chunk: 300 rows", 300),
        _cell("a prefill chunk: 2048 rows", 2048),
        _cell("the engine's token ceiling: 8192 rows, the workspace's largest growth", 8192),
    )

    note = """
    Numerics. The rounding points are the kernels' own (see `reference`): the all-reduce result
    is rounded to the activation dtype before the residual is added, so a bf16 call carries one
    more rounding than an fp32 sum followed by a fused add and norm would. The two kernels round
    at the same places but associate the last fp32 product differently (`(x * rcp) * gamma` in
    the one-shot kernel, `(gamma * x) * rcp` in the two-shot one), so each can differ from
    `reference`, and from the other, by one bf16 flip of a rare element; the test's band is
    derived from that. Every rank must receive bit-identical `norm_out` and `residual_out`,
    which is what keeps the TP-replicated activations replicated.

    Protocol. The workspace is a Lamport (triple-buffered) protocol: each call spends one buffer,
    cleans the previous one and advances `buffer_flags`, so calls on one workspace must not
    overlap and the buffers must be initialized once, collectively, before the first call. The
    engine's `get_or_scale_allreduce_mnnvl_workspace` does both. Growing it after a CUDA graph
    captured on it leaves that graph launching on the old buffers, which the engine keeps alive.

    Scope. NVL72-class systems only (aarch64, fabric handles): `MNNVLAllReduce.is_mnnvl` is the
    engine's own eligibility check, and a target that fails it uses another all-reduce.
    """

    def __call__(
        self,
        input: torch.Tensor,
        gamma: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        eps: Optional[float],
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        rmsnorm_fusion: bool,
        scale: Optional[torch.Tensor],
        fusion_op: int,
    ) -> List[torch.Tensor]:
        return torch.ops.trtllm.mnnvl_fusion_allreduce(
            input, gamma, residual, eps, comm_buffer, buffer_flags, rmsnorm_fusion, scale, fusion_op
        )

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(norm_out, residual_out)` as the group should produce them, with the kernels' rounding points.

        The documented exception to mirroring `__call__`: a collective's output is not a function
        of what the calling rank holds, so this takes the group's inputs in ascending rank order.
        Both kernels (one-shot and two-shot) round at the same places, read off their source and
        held by the test: the sum over ranks is accumulated in fp32 and rounded to the activation
        dtype *before* the residual is added; `residual_out` is the activation-dtype sum of that and
        the residual; the squares the normalization averages are formed in the activation dtype
        (rounded), summed in fp32; the output is `gamma * residual_out * rsqrt(mean + eps)` in fp32,
        rounded once. What the kernels do not share with each other or with this function is the
        fp32 association of that last product and the order of the sum of squares, which is why the
        test gates `norm_out` in a band rather than bitwise (`residual_out` is bitwise).
        """
        dtype = residual.dtype
        total = torch.zeros(per_rank_inputs[0].shape, dtype=torch.float32, device=residual.device)
        for x in per_rank_inputs:
            total += x.float()
        reduced = total.to(dtype)
        residual_out = (reduced.float() + residual.float()).to(dtype)
        mean_square = (residual_out * residual_out).float().mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)
        norm_out = (gamma.float() * residual_out.float() * inv_rms).to(dtype)
        return norm_out, residual_out

    def is_valid(
        self,
        input: torch.Tensor,
        gamma: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        eps: Optional[float],
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        rmsnorm_fusion: bool,
        scale: Optional[torch.Tensor],
        fusion_op: int,
    ) -> None:
        # What the op's own checks leave open: it requires contiguous operands and a residual,
        # gamma and epsilon for this pattern, but reads `gamma` and `residual` through bare
        # pointers without comparing their shapes or dtypes with `input`'s.
        assert fusion_op == FUSION_RESIDUAL_RMS_NORM, (
            f"fusion_op {fusion_op}: only RESIDUAL_RMS_NORM ({FUSION_RESIDUAL_RMS_NORM}) is "
            "certified here"
        )
        assert rmsnorm_fusion, "rmsnorm_fusion must be True for RESIDUAL_RMS_NORM"
        assert residual is not None and gamma is not None and eps is not None, (
            "residual, gamma and eps are all required for RESIDUAL_RMS_NORM"
        )
        assert input.dtype in (torch.bfloat16, torch.float16), (
            f"{input.dtype}: the fused pattern is certified for 16-bit activations"
        )
        assert residual.shape == input.shape and residual.dtype == input.dtype, (
            "residual must have input's shape and dtype: the op does not compare them"
        )
        assert gamma.shape == (input.shape[-1],) and gamma.dtype == input.dtype, (
            "gamma must be a [hidden] tensor in the activation dtype: the op does not compare them"
        )
        assert comm_buffer.dim() == 2 and comm_buffer.shape[0] == 3, (
            "comm_buffer must be the workspace's unicast buffer viewed as [3, -1]"
        )
        assert comm_buffer.dtype == input.dtype, (
            "comm_buffer must be viewed in the activation dtype"
        )


mnnvl_fusion_allreduce = MnnvlFusionAllreduce()
