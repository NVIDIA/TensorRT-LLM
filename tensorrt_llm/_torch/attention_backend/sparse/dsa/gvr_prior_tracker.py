# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-identity tracking for the GVR top-k prior buffer."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


class GvrPriorTracker:
    """Keeps ``gvr_prior_indices`` rows aligned to request identity.

    The prior buffer is positional (decode batch row ``r`` reads buffer row
    ``r``) while the batch composition changes across steps: request
    completion shifts later rows left, context requests convert to
    generation, and disagg / full-prefix-reuse requests start decoding with
    no local prefill. Unrepaired, a shifted row reads its neighbour's
    previous top-k and a first-step-decode request reads zeros or a stale
    row. This tracker records which request id each row holds and, on
    composition change only, permutes the rows to follow their requests and
    seeds an in-range ``arange`` hint for requests never seen before. The
    steady state (unchanged composition) is a host-side list compare with no
    device work. Hints only affect GVR speed, never exactness, so every
    repair here is a hint-quality/contract fix.
    """

    def __init__(self) -> None:
        self._row_ids: List[int] = []
        self._arange: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Forget all row ownership (call whenever the buffer is zeroed)."""
        self._row_ids = []

    def realign(
        self,
        prior: torch.Tensor,
        gen_ids: Sequence[int],
        ctx_ids: Sequence[int],
    ) -> None:
        """Repair ``prior`` (``[num_layers, capacity, top_k]``) for this step.

        ``gen_ids``/``ctx_ids`` are this step's request ids in batch order.
        The buffer layout is generation-first: decode reads rows
        ``[0, len(gen_ids))`` and the prefill prior update writes rows
        ``[len(gen_ids), len(gen_ids) + len(ctx_ids))``.
        """
        gen_ids = list(gen_ids)
        num_gen = len(gen_ids)
        if num_gen > 0 and self._row_ids[:num_gen] != gen_ids:
            old_pos = {rid: i for i, rid in enumerate(self._row_ids)}
            src = [old_pos.get(rid, -1) for rid in gen_ids]
            moved = any(s >= 0 and s != i for i, s in enumerate(src))
            seeds = [i for i, s in enumerate(src) if s < 0]
            # Composition-change steps only. The previous step's prior
            # write-back may still be in flight on the indexer aux stream;
            # a full sync is the conservative ordering for this rare path.
            if prior.is_cuda:
                torch.cuda.synchronize()
            if moved:
                index = torch.tensor(
                    [max(s, 0) for s in src], dtype=torch.long, device=prior.device
                )
                prior[:, :num_gen].copy_(prior.index_select(1, index))
            if seeds:
                if self._arange is None or self._arange.numel() != prior.shape[-1]:
                    self._arange = torch.arange(
                        prior.shape[-1], dtype=prior.dtype, device=prior.device
                    )
                seed_index = torch.tensor(seeds, dtype=torch.long, device=prior.device)
                prior[:, seed_index] = self._arange
        self._row_ids = gen_ids + list(ctx_ids)
