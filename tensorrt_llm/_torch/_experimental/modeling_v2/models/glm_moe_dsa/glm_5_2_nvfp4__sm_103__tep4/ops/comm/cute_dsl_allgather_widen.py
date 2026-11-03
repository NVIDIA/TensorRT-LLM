# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lm_head's vocabulary all-gather, widened to fp32: one Lamport launch over TP4."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._lamport import LamportWorkspace, check_rows_by_numel, tail

#: The vocabulary shard each rank's `lm_head` produces at TP4: 154880 / 4.
SHARD_N = 38720


def takes(local: torch.Tensor, workspace: LamportWorkspace) -> bool:
    """Whether the op takes this rank's shard: the workspace built the all-gather,
    `local` is contiguous bf16 `[M >= 1, 38720]`, and the shard fits one peer region of
    the workspace. Every rank sees the same `M`, so every rank answers the same."""
    return (
        workspace.gather is not None
        and local.dtype is torch.bfloat16
        and local.ndim == 2
        and local.shape[0] >= 1
        and local.shape[1] == SHARD_N
        and local.is_contiguous()
        and local.shape[0] * SHARD_N <= workspace.max_rows * workspace.hidden
    )


class CuteDslAllgatherWiden(OpWrapper):
    """`out[:, r*N:(r+1)*N] = float(shard_r)` for every rank `r`: the full-vocabulary fp32
    logits on every rank, from each rank's bf16 `[M, 38720]` shard. Returns `out`.

    What upstream does in three steps (the `lm_head`'s NCCL all-gather, the
    concatenation, the logits processor's `.float()`) in one launch on the
    all-reduces' Lamport workspace: same peer buffers, same in-band readiness
    sentinel (bf16 -0.0), same device slot counter, so it interleaves with them
    inside one CUDA graph. Because -0.0 is the sentinel, every -0.0 of a shard
    arrives as +0.0, on every rank alike; it is the only difference from an exact
    gather and moves no argmax, softmax or sampled token.

    Collective: every rank of the group calls it at the same point with the same
    `M`. `out` is a contiguous fp32 `[M, 4 * 38720]` buffer.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The logits rows of a step: a decode step's (32 requests x MTP 2 at most), a
    # context step's (the last row of each prompt plus the generation rows), and
    # the largest shard the 8192 x 6144 workspace region holds.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one logits row", spec=dict(rows=1)),
        Cell(why="the target accept of 32 requests x MTP 2", spec=dict(rows=64)),
        Cell(why="an odd row count", spec=dict(rows=37)),
        Cell(
            why="the largest shard one workspace region holds",
            spec=dict(rows=8192 * 6144 // SHARD_N),
        ),
    )

    note = """
    The target builds the all-gather into the all-reduces' workspace at load
    (`_lamport.build_workspace`; the build is agreed across the group, and a
    group that could not build it keeps `workspace.gather` None) and takes this
    op for its logits wherever `takes` holds; every other call keeps upstream's
    `LogitsProcessor`. The certification is the 4-rank matrix shared with the
    all-reduce entries, interleaved with them on one workspace.
    """

    def __call__(
        self, local: torch.Tensor, out: torch.Tensor, workspace: LamportWorkspace
    ) -> torch.Tensor:
        workspace.gather(local, out, *tail(workspace, fused=False))
        return out

    def reference(
        self, locals_: list, out: torch.Tensor, workspace: LamportWorkspace
    ) -> torch.Tensor:
        """The group's shards in rank order, concatenated and widened, with -0.0 as +0.0.

        A collective: what a rank receives is not a function of its own shard, so
        the reference takes every rank's (see `OpWrapper.reference`). `out` and
        `workspace` are ignored.
        """
        cat = torch.cat([s.float() for s in locals_], dim=-1)
        return torch.where(cat == 0, torch.zeros_like(cat), cat)

    def is_valid(self, local: torch.Tensor, out: torch.Tensor, workspace: LamportWorkspace) -> None:
        # The kernel takes the row count from its input and writes `out` through a
        # raw pointer at that count: a shard past one peer region writes past every
        # peer's buffer, and a short or strided `out` is written past its end.
        check_rows_by_numel(int(local.shape[0]) * SHARD_N, workspace)
        assert out.dtype is torch.float32 and out.is_contiguous(), (
            "out must be a contiguous fp32 buffer"
        )
        assert tuple(out.shape) == (local.shape[0], workspace.world * SHARD_N), (
            f"out must be [{local.shape[0]}, {workspace.world * SHARD_N}], got {tuple(out.shape)}"
        )


cute_dsl_allgather_widen = CuteDslAllgatherWiden()
