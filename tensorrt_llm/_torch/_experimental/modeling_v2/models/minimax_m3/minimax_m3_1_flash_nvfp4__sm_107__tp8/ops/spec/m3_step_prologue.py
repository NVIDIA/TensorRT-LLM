# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode graph's step-input prologue in one launch (this target's ``RuntimeHooks.pack_prologue_op``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3StepPrologue(OpWrapper):
    """``StepInputPack``'s early-copy prologue as one single-CTA C++ launch
    (trtllm::minimax_m3_sm107_tp8_step_prologue, minimaxM3Sm107Tp8/stepPrologue.cu).

    ``window`` is the pack's device window (``gdev``): an int64 header ``[host copies, device copies]``, then one
    ``(dst, src, bytes)`` descriptor per copy, host copies first (a host copy's ``src`` is a byte offset into the
    window, a device copy's a device address). The launch writes up to ``max_desc`` copies of each kind, every host
    copy before any device copy, then zeroes the header; with ``epoch`` it increments ``epoch[0]`` once. The default
    prologue does the same with two parallel-scatter launches (``step_inputs._graph_scatter_kernel``) and a header
    fill."""

    ARCHS = frozenset({Arch.SM_107})

    # The TP8 decode step's packs: a few dozen host copies of 4 to a few hundred bytes (positions, KV lengths, slots,
    # the spec sampler's uploads), a handful of device copies (the graph runner's static inputs), some reading what a
    # host copy of the same window wrote.
    CELLS: tuple[Cell, ...] = (
        Cell(why="an empty window (a replay with no new upload)", spec=dict(h2d=0, d2d=0)),
        Cell(why="host copies only, odd sizes and alignments", spec=dict(h2d=40, d2d=0)),
        Cell(
            why="host copies feeding device copies (a device copy reads a host copy's destination)",
            spec=dict(h2d=24, d2d=8, chained=True),
        ),
        Cell(
            why="every descriptor slot (128 + 128 of the pack's 256)", spec=dict(h2d=128, d2d=128)
        ),
    )

    note = """
    Byte copies: the reference (the default prologue's Triton launches and fill on the same window) is exact.
    Destinations are raw addresses; is_valid cannot check them.
    """

    def __call__(self, window: torch.Tensor, epoch: Optional[torch.Tensor], max_desc: int) -> None:
        torch.ops.trtllm.minimax_m3_sm107_tp8_step_prologue(window, epoch, max_desc)

    def reference(self, window: torch.Tensor, epoch: Optional[torch.Tensor], max_desc: int) -> None:
        """The default prologue on the same arguments: it writes the same destinations."""
        from tensorrt_llm._torch.pyexecutor.step_inputs import _graph_scatter_kernel

        stage_i64 = window.view(torch.int64)
        for device_src in (False, True):
            bump = epoch is not None and not device_src
            _graph_scatter_kernel[(max_desc,)](
                window,
                stage_i64,
                epoch if bump else stage_i64,
                DEVICE_SRC=device_src,
                BLOCK=256,
                BUMP_EPOCH=bump,
                num_warps=1,
            )
        window[:16].zero_()

    def is_valid(self, window: torch.Tensor, epoch: Optional[torch.Tensor], max_desc: int) -> None:
        assert window.dtype == torch.uint8 and window.is_contiguous(), "window: contiguous uint8"
        assert window.numel() >= 16 + 24 * max_desc, "window: header and max_desc descriptors"
        counts = window[:16].view(torch.int64).tolist()
        assert counts[0] + counts[1] <= max_desc, "at most max_desc copies in all"


m3_step_prologue = M3StepPrologue()
