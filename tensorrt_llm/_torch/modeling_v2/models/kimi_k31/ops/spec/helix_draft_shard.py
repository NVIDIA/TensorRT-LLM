# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash drafter's paged context KV under Helix CP: page ownership and the combine of per-rank partials.

Global page ``g`` of a request belongs to CP rank ``g % cp_size``, which stores its pages back to back (local page
``g // cp_size``). The DFlash worker stores and reads its pages through :class:`HelixDraftShard`; the drafter's Helix
attention (``ops/attention/helix.py``) merges the ranks' partial outputs with :func:`combine_helix_partials`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HelixDraftShard:
    """One rank's share of a drafter's paged context KV under helix.

    Global page ``g`` of a request belongs to CP rank ``g % cp_size``, which stores its pages back to back: local
    page ``g // cp_size``. Every helper maps global token positions or lengths into that rank-local layout.
    """

    cp_size: int
    cp_rank: int
    page_size: int

    def local_slots(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-local slot of each global position, and whether this rank owns it.

        An unowned position maps to the slot this rank keeps for the token one or more pages away, never one of the
        ``page_size`` tokens around it.
        """
        page = torch.div(positions, self.page_size, rounding_mode="floor")
        slots = torch.div(page, self.cp_size, rounding_mode="floor") * self.page_size
        return slots + positions % self.page_size, page % self.cp_size == self.cp_rank

    def local_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Number of owned positions below each global length."""
        span = self.page_size * self.cp_size
        rem = lengths % span - self.cp_rank * self.page_size
        return torch.div(lengths, span, rounding_mode="floor") * self.page_size + rem.clamp(
            0, self.page_size
        )


def combine_helix_partials(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Merge per-rank partial attention over disjoint KV shards.

    Args:
        out: ``[cp, N, H, D]`` partial outputs, one per CP rank.
        lse: ``[cp, N, H]`` base-2 log-sum-exp of the matching scores, as TRTLLM-Gen returns it.
    Returns:
        ``[N, H, D]`` in ``out``'s dtype.
    """
    lse = lse.float()
    peak = lse.amax(dim=0, keepdim=True)
    weight = torch.exp2(lse - peak.clamp_min(torch.finfo(torch.float32).min))
    weight = weight.nan_to_num(nan=0.0)
    total = weight.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
    merged = (weight.unsqueeze(-1) * out.float().nan_to_num(0.0)).sum(dim=0)
    return (merged / total.unsqueeze(-1)).to(out.dtype)


__all__ = ["HelixDraftShard", "combine_helix_partials"]
