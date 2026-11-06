# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final-logit tanh softcap in fp32: `tanh(logits.float() / cap) * cap`."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The head's output: every token's 151936 vocabulary logits, bf16, cap 30.
_HEAD = dict(vocab=151936, cap=30.0, dtype=torch.bfloat16)


class BeamLogitSoftcap(OpWrapper):
    """Softcap the head projection `[T, V]`: `(logits.float() / cap).tanh() * cap`, fp32.

    The head's activation-dtype output is widened to fp32, divided by `cap`, passed
    through tanh and multiplied by `cap`, each step in fp32; the fp32 result is
    returned, in `(-cap, cap)` for finite inputs and exactly `+-cap` once tanh
    saturates. Applied once, after the head projection and nothing else.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="head, 1 token", spec=dict(_HEAD, tokens=1)),
        Cell(why="head, 17 tokens", spec=dict(_HEAD, tokens=17)),
        Cell(why="head, 129 tokens: the longest context", spec=dict(_HEAD, tokens=129)),
        Cell(
            why="signed values from 0 to beyond saturation, -0 included",
            spec=dict(_HEAD, tokens=1, saturating=True),
        ),
    )

    note = """
    Not idempotent: a second application moves every nonzero logit toward zero
    (`30 * tanh(30 * tanh(x / 30) / 30)` < `30 * tanh(x / 30)` for x > 0), and
    omitting it changes every logit whose magnitude exceeds about 0.05. Both are
    therefore visible on ordinary logits by comparing the returned values with this
    expression of the captured head output, without any saturating input.
    """

    def __call__(self, logits: torch.Tensor, cap: float) -> torch.Tensor:
        return (logits.float() / cap).tanh() * cap

    def reference(self, logits: torch.Tensor, cap: float) -> torch.Tensor:
        """The same three fp32 steps, written out one by one."""
        widened = logits.to(torch.float32)
        scaled = torch.div(widened, cap)
        return torch.mul(torch.tanh(scaled), cap)

    def is_valid(self, logits: torch.Tensor, cap: float) -> None:
        # A tensor cap broadcasts and a non-positive cap flips or removes the cap; an
        # fp32 input that was rounded upstream is indistinguishable here. None raises.
        assert isinstance(cap, float) and cap > 0, f"cap must be a positive float, got {cap!r}"
        assert logits.dim() == 2, "logits must be [T, vocab]"
        assert logits.dtype == torch.bfloat16, "logits must be the bf16 head output"


beam_logit_softcap = BeamLogitSoftcap()
