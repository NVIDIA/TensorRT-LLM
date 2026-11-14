# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3's MoE layer for 9..13 decode rows in one CuTe DSL launch (``m3_moe16``): the router, the routed NVFP4
experts and the shared MXFP8 expert, on caller-owned state (an ``M3Moe16State`` and one ``M3MoeLayer`` per MoE
layer)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..m3_moe.op import M3MoeLayer
from ..m3_moe.op16 import M3Moe16State
from .m3_moe_engine import moe_reference

__all__ = ["M3Moe16State", "m3_moe16_engine"]


class M3Moe16Engine(OpWrapper):
    """One MiniMax-M3 MoE layer for M (9..13 here; the state takes 9..16) decode rows:
    ``layer.state.run(layer, hidden, out, arp)``, bf16(shared + routed) [M, 6144].

    The same arguments and the same math as ``m3_moe_engine`` (the routed part operation for operation, the shared
    expert always folded in) on an ``M3Moe16State``: a different schedule for more rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One build per row count (m_max is a trace-time constant); the target routes 9..13 rows here (spec-decode verify
    # of 2 x 6 rows lands at 12).
    CELLS: tuple[Cell, ...] = (
        Cell(why="9 rows: the smallest build, a partial last slot group", spec=dict(tokens=9)),
        Cell(why="10 rows: its own build (m_max is a trace-time constant)", spec=dict(tokens=10)),
        Cell(why="11 rows: its own build, an odd row count", spec=dict(tokens=11)),
        Cell(why="12 rows: two requests verifying 5-token drafts", spec=dict(tokens=12)),
        Cell(why="13 rows: the largest the target routes here", spec=dict(tokens=13)),
    )

    note = """
    Bands as `m3_moe_engine` (same approximations, same quantization boundaries). Not covered: 14..16 rows, which the
    state takes and this target never routes here; the all-reduce push (`arp`). No `is_valid`, as `m3_moe_engine`.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.state.run(layer, hidden, out, arp=arp)

    def reference(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``m3_moe_engine``'s reference with the shared expert; `out` and `arp` are ignored."""
        return moe_reference(hidden, layer, shared=True)


m3_moe16_engine = M3Moe16Engine()
