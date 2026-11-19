# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's helix combine: exchange the ranks' window partials, merge by LSE.

Under helix every CP rank attends all heads of the drafter's block queries over its own pages of
the window (``rubin_dflash2_window_triton`` with ``stats``), projects the
partials through W_UV first (``[R, H, 512] -> [R, H, 128]``) and then this entry sends head
shard ``c`` to rank ``c`` and merges what it receives by log-sum-exp: the family's
``_helix_post_process`` (``ops/attention/helix.py``) with the target's mapping, so the exchange
reuses the target's transport (NCCL or the MNNVL FIFO) and its cached workspace.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ....ops.attention.helix import _helix_post_process


class RubinDFlash2HelixCombine(OpWrapper):
    """This rank's ``heads // cp`` heads of the CP-merged window attention.

    ``partial_v`` ``[R, heads, value_dim]`` is this rank's normalized partial (BF16) and
    ``stats`` ``[R, heads, 2]`` (float32) its natural-log ``(max, sum)`` per row and head; a row
    with no local key must already be a zero partial with ``(-inf, 0)`` (the window entries
    write it so), and then ``zero_kv_mask`` is None. A mask, when given, marks such rows for the
    exchange to sanitize instead. Returns ``[R, (heads // cp) * value_dim]`` for heads
    ``[cp_rank * heads // cp, (cp_rank + 1) * heads // cp)``, in ``partial_v``'s dtype.

    ``mapping`` is the target's mapping (its ``cp_config`` picks the transport). Unlike
    ``helix_target_combine``, the mask is an argument, not read off the target's attention
    metadata, which the drafter does not use.
    """

    ARCHS = frozenset()

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 generation, 16 requests at draft 5 (96 rows), 96 heads x 128 after W_UV",
            spec=dict(rows=96, heads=96, value_dim=128, cp=8, mask=False),
        ),
        Cell(
            why="CP4: empty rows on some ranks",
            spec=dict(rows=24, heads=96, value_dim=128, cp=4, mask=False),
        ),
        Cell(
            why="CP2 with an explicit zero_kv_mask instead of sanitized rows",
            spec=dict(rows=14, heads=96, value_dim=128, cp=2, mask=True),
        ),
    )

    note = """
    The exchange and the combine kernel are the family's Helix combine (`_helix_post_process` of
    `ops/attention/helix.py`: `alltoall_helix` / the FIFO all-to-all and
    `helix_post_process[_native]`), whose NCCL path runs the sanitize and reformats eagerly; this
    entry adds nothing else on the device. The reference
    is the LSE merge of `helix_target_combine.reference`, written with selects so an empty rank
    never multiplies into the sum. Multi-rank: certified by the 2/4/8-rank test, not on one GPU.
    """

    def __call__(
        self,
        partial_v: torch.Tensor,
        stats: torch.Tensor,
        mapping,
        heads: int,
        value_dim: int,
        zero_kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rows = partial_v.shape[0]
        partial_o = partial_v.reshape(rows, heads * value_dim)
        return _helix_post_process(
            partial_o,
            stats,
            mapping,
            heads // mapping.cp_size,
            value_dim,
            None,
            None,
            zero_kv_mask=zero_kv_mask,
        )

    def reference(
        self,
        partial_v: torch.Tensor,
        stats: torch.Tensor,
        mapping,
        heads: int,
        value_dim: int,
        zero_kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The group's inputs in rank order: ``partial_v`` ``[cp, R, heads, value_dim]``,
        ``stats`` ``[cp, R, heads, 2]``, ``zero_kv_mask`` ``[cp, R]`` or None; returns
        ``mapping.cp_rank``'s shard ``[R, (heads // cp) * value_dim]``."""
        cp = partial_v.shape[0]
        rows = partial_v.shape[1]
        partials = partial_v.float()
        maxima, sums = stats[..., 0].float(), stats[..., 1].float()
        if zero_kv_mask is not None:
            empty = zero_kv_mask[:, :rows, None]
            maxima = maxima.masked_fill(empty, float("-inf"))
            sums = sums.masked_fill(empty, 0.0)
            partials = partials.masked_fill(empty[..., None], 0.0)
        live = sums > 0
        peak = torch.where(live, maxima, float("-inf")).amax(dim=0, keepdim=True)
        peak = torch.where(torch.isfinite(peak), peak, 0.0)
        weights = torch.where(live, sums * torch.exp(torch.where(live, maxima, 0.0) - peak), 0.0)
        total = weights.sum(dim=0)
        merged = torch.where(live[..., None], weights[..., None] * partials, 0.0).sum(dim=0)
        merged = merged / total.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
        width = heads // cp
        rank = mapping.cp_rank
        shard = merged[:, rank * width : (rank + 1) * width]
        return shard.reshape(rows, width * value_dim).to(partial_v.dtype)

    def is_valid(
        self,
        partial_v: torch.Tensor,
        stats: torch.Tensor,
        mapping,
        heads: int,
        value_dim: int,
        zero_kv_mask: Optional[torch.Tensor] = None,
    ) -> None:
        rows = partial_v.shape[0]
        assert partial_v.shape == (rows, heads, value_dim)
        assert partial_v.dtype in (torch.bfloat16, torch.float16)
        assert stats.shape == (rows, heads, 2) and stats.dtype == torch.float32
        assert stats.is_contiguous(), "the exchange views the stats as [rows, cp, heads/cp * 2]"
        assert mapping.cp_size > 1 and heads % mapping.cp_size == 0
        assert zero_kv_mask is None or (
            zero_kv_mask.dtype == torch.bool and zero_kv_mask.shape[0] >= rows
        )
        if zero_kv_mask is None:
            # Rows with no local key must arrive sanitized: the combine weights by sum, and a
            # NaN partial times a zero weight is NaN (test-only read).
            empty = stats[..., 1] == 0
            assert bool((partial_v[empty] == 0).all()), "unsanitized empty rows"
            assert bool(torch.isneginf(stats[..., 0][empty]).all())


rubin_dflash2_helix_combine = RubinDFlash2HelixCombine()

__all__ = ["rubin_dflash2_helix_combine"]
