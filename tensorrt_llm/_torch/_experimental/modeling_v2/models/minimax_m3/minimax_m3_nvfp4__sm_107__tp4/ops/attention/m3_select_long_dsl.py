# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-16 MSA block selection for decode rows of up to 9216 blocks, one row per 384-thread CTA (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_select_long import _M3, MAX_BLOCKS, M3SelectLong


class M3SelectLongDsl(OpWrapper):
    """``m3_select_long``'s contract, cells and reference, run by ``ops/m3_select_long_dsl``: the same selection
    bit for bit as the 1024-thread ``minimax_m3_sm107_tp4_select_long`` (also on rows where more than 2048 blocks tie
    at the 16th-largest score), from one 384-thread CTA per row. ``fast_tail`` (the 1024-thread kernel's warp-0 rank)
    does not apply and is ignored. Returns ``out``."""

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="32K-context engine (257 worst-case k-tiles), Eagle3 verify at one request: 4 query rows",
            spec=dict(num_blocks=257, total_q=4, valid="near_full", scores="random", **_M3),
        ),
        Cell(
            why="decode without speculation at batch 8 on a 2048-block engine: the 2-key-per-thread path at its edge",
            spec=dict(num_blocks=2048, total_q=8, valid="near_full", scores="random", **_M3),
        ),
        Cell(
            why="1M-context engine at batch 8 x qlen 4: rows on all three key widths, short rows and padding rows",
            spec=dict(num_blocks=8193, total_q=32, valid="mixed", scores="random", **_M3),
        ),
        Cell(
            why="duplicated scores: ties broken to the smaller block, and > 1024 keys at the bound (re-read path)",
            spec=dict(num_blocks=4500, total_q=4, valid="near_full", scores="ties", **_M3),
        ),
        Cell(
            why="non-finite scores: +inf, NaN ranked above +inf, and rows whose picks run into -inf (emitted -1)",
            spec=dict(num_blocks=600, total_q=8, valid="mixed", scores="nonfinite", **_M3),
        ),
    )

    def reference(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
        trigger_phase: int = 0,
        fast_tail: bool = True,
    ) -> torch.Tensor:
        """``m3_select_long``'s reference: the selection is the same bit for bit."""
        return M3SelectLong.reference(
            self,
            scores,
            n_valid_blocks,
            out,
            init_blocks,
            local_blocks,
            use_pdl,
            early_trigger,
            trigger_phase,
            fast_tail,
        )

    def is_valid(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
        trigger_phase: int = 0,
        fast_tail: bool = True,
    ) -> None:
        """``m3_select_long``'s domain."""
        M3SelectLong.is_valid(
            self,
            scores,
            n_valid_blocks,
            out,
            init_blocks,
            local_blocks,
            use_pdl,
            early_trigger,
            trigger_phase,
            fast_tail,
        )

    note = """
    Certified at one KV head per rank (TP4, Rubin) with M3's forcing
    (`init_blocks=0`, `local_blocks=1`), and bit for bit against the
    1024-thread `m3_select_long` on every cell. The kernel takes any head count
    and forcing; other forcing is not driven here.

    The kernel is compiled per (PDL launch, early trigger, trigger phase) with
    symbolic shapes and strides; the target compiles its keys before warmup
    (`ops/m3_select_long_dsl.prebuild`), and a call outside them compiles
    on first use.
    """

    def __call__(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
        trigger_phase: int = 0,
        fast_tail: bool = True,
    ) -> torch.Tensor:
        from ..m3_select_long_dsl import select_long_dsl

        if scores.shape[1] > MAX_BLOCKS:
            raise RuntimeError(
                f"m3_select_long_dsl supports at most {MAX_BLOCKS} blocks, got {scores.shape[1]}"
            )
        select_long_dsl(
            scores,
            n_valid_blocks,
            out,
            init_blocks,
            local_blocks,
            use_pdl,
            early_trigger,
            trigger_phase,
        )
        return out


m3_select_long_dsl = M3SelectLongDsl()
