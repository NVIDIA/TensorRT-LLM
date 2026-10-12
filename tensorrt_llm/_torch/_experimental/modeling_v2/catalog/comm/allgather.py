# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 at dep4 always passes `sizes=None`: it pads every rank to a
    # common height first, so the ragged form the op also supports is not
    # something a shipped target reaches. Row counts span what an attention-DP
    # rank feeds its expert-parallel MoE call, from one decode token to a full
    # prefill chunk. deepseek-v4.1 pads the same way; its cells follow.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, hidden 7168",
            spec=dict(rows=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch, the common case between decode and prefill",
            spec=dict(rows=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        # deepseek-v4.1 under attention DP: per-rank rows n at the ends of the
        # swept range 1..16384 (16384 is the largest max_num_tokens served) and
        # at 1536, the largest n a decode graph captures (batch 256 x next_n 6).
        # The matrix sweeps every n in between for each form.
        *(
            Cell(
                why=f"deepseek-v4.1 {form} n={n}: {role}",
                spec=dict(rows=n, trailing=trailing, dtype=dtype, sizes=None),
            )
            for form, trailing, dtype in (
                ("moe bf16 [n, 5120]", (5120,), torch.bfloat16),
                ("engram int32 [n, 48]", (48,), torch.int32),
                ("engram int64 [n, 24]", (24,), torch.int64),
            )
            for n, role in (
                (1, "decode end of the swept range"),
                (1536, "top of the decode capture set"),
                (16384, "max_num_tokens, prefill end of the swept range"),
            )
        ),
        *(
            Cell(
                why=f"deepseek-v4.1 image mask uint8 [n] n={n}: {role}",
                spec=dict(rows=n, trailing=(), dtype=torch.uint8, sizes=None),
            )
            for n, role in (
                (1, "eager only, decode end of the swept range"),
                (16384, "eager only, prefill end of the swept range"),
            )
        ),
    )

    note = """
    The ragged form (`sizes` as an explicit per-rank vector) and dtypes other
    than bf16 (beyond the DeepSeek-V4.1 forms below) work and are exercised by
    the op matrix, but no shipped target reaches them, so they are not cells.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis either: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks.

    DeepSeek-V4.1: under attention DP every rank is padded to
    max(all_rank_num_tokens) and gathers with `sizes=None`. The MoE input,
    bf16 [n, 5120], is gathered on every layer; on the engram layers (only with
    the device-resident engram table) the int64 [n, 24] engram ids are gathered
    as their int32 [n, 48] view, and int64 [n, 24] is certified beside it. The
    multimodal target, in every eager step and never under capture, views its
    bool [T] image-span mask as uint8, zero-pads it at the end to n, gathers it
    and reads the result back with `.view(torch.bool)`; bool itself is not
    certified. No metadata. The matrix sweeps every per-rank n in 1..16384 for
    the three core forms and for the mask (with the int32 id gather that follows
    it), ascending and then shuffled, bitwise; it captures one decode graph per
    per-rank n of the capture set (batch G in 1..32, 40..128 step 8, 192, 256;
    n = G or 6G for next_n 1 or 6, plus 35 and 160: 84 values up to 1536), each
    holding the forward's 44 gathers, and replays them bitwise against the
    reference and against eager. A captured step must pad every rank to the same
    n, because `sizes` is frozen into the graph. The ragged form at V4.1 sizes
    is certified only at four dumped splits, eagerly. Not certified: n above
    16384; capture at an n outside that set; the mask under capture. The other
    group layouts the target runs -- world 2 (one pp2 x dep2 stage), world 4 cut
    into two groups of 2, world 8 across two trays, world 8 cut into two groups
    of 4 -- were measured on the MPI-session symbol under an mpirun launcher;
    this matrix runs one group of 4 on the ProcessGroup symbol and does not
    certify them.
    """

    def raw_call(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """`process_group` selects the communicator flavor, mirroring the
        product router in `_torch/distributed/ops.py`: None runs the
        MPI-session symbol -- the one a dep4 serving world calls -- and a boxed
        c10d ProcessGroup runs the `_pg` symbol the non-MPI backends use.
        """
        if process_group is None:
            return torch.ops.trtllm.allgather(input, sizes, group)
        return torch.ops.trtllm.allgather_pg(input, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation the collective is supposed to make.

        The documented exception to mirroring `raw_call`: a collective's output
        is not a function of what the calling rank holds, so a reference given
        only `input` could not state anything at all. It takes the group's
        inputs in ascending rank order instead. The matrix builds that list on
        one rank from the seed convention every rank shares, which is what keeps
        the reference arithmetic rather than a second collective.
        """
        return torch.cat(list(per_rank_inputs), dim=0)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """A gate of exactly zero: the op moves bytes and computes nothing.

        Tightened from the base default rather than loosened. Any difference at
        all is a wrong gather.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        if out.dtype is torch.float8_e4m3fn:
            # torch.testing cannot compare float8; for a pure data move the byte
            # pattern is the honest gate anyway.
            out, ref = out.view(torch.uint8), ref.view(torch.uint8)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption or
        # a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "AllgatherOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently gathered from the wrong elements"
        )
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
            "so a short list silently drops the trailing ranks and a long one "
            "appends uninitialized rows"
        )


allgather = Allgather()
