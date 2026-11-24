# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Helix combine of the received partials: the stock ``torch.ops.trtllm.helix_post_process_native``
with the sources on dimension 1."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _rubin_ull_helix_reference as _ref


def _cell(why: str, tokens: int, cp_size: int, masked: bool = False) -> Cell:
    local = _ref.HEADS // 16  # the served heads per rank at CP16
    return Cell(why=why, spec=dict(tokens=tokens, cp_size=cp_size, heads=local, masked=masked))


class HelixPostProcessNative(OpWrapper):
    """``out[t, h] = sum_s w[t, s, h] * gathered_o[t, s, h] / sum_s w[t, s, h]`` with
    ``w = sum * exp(max - max_s max)`` from ``gathered_stats[t, s, h] = (max, sum)``, times
    ``scale``. ``gathered_o`` BF16 [T, P, L, V] (``cp_dim`` 1: the sources on dimension 1),
    ``gathered_stats`` FP32 [T, P, L, 2]; returns a new BF16 [T, L * V]. The op checks ranks,
    sizes, dtypes, contiguity and alignment itself."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 q8 at CP16: 16 sources, 6 heads of 128 values", 8, 16),
        _cell("b8 q8 at CP16 (64 rows)", 64, 16),
        _cell("b1 at CP16 with zero-KV sources (max -inf, sum 0)", 8, 16, True),
        _cell("one tray's group (P = 4), b4", 32, 4),
    )

    note = """
    The combine of ``comm.FlatHelix``'s FIFO transport, run after ``helix_alltoall_native`` where
    the fused Helix tail declines (it never does at the served shapes; the drafter's and the DSA
    layers' fallback). ``scale`` is 1.0 there. A token whose every source holds no key is NaN.
    """

    def __call__(
        self,
        gathered_o: torch.Tensor,
        gathered_stats: torch.Tensor,
        scale: float,
        cp_dim: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, scale, cp_dim)

    def reference(
        self,
        gathered_o: torch.Tensor,
        gathered_stats: torch.Tensor,
        scale: float,
        cp_dim: int,
    ) -> torch.Tensor:
        """FP64 combine rounded once; ``cp_dim`` 1 only (the target's layout)."""
        assert cp_dim == 1, "the target combines with the sources on dimension 1"
        values = list(gathered_o.transpose(0, 1))
        stats = list(gathered_stats.transpose(0, 1))
        out = _ref.combine(values, stats)
        return out if scale == 1.0 else (out * scale)


helix_post_process_native = HelixPostProcessNative()
