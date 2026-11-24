# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch Helix combine shared by the target's Helix entries' references."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

#: K3.1's MLA geometry: 96 query heads, a 512-wide latent, 128-wide values.
HEADS = 96
LATENT = 512
VALUE = 128


def masked_stats(stats: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """``stats`` [T, H, 2] with the rows ``mask`` [T] marks as holding no key: max -inf, sum 0."""
    if mask is None:
        return stats.float()
    rows = mask[: stats.shape[0]].view(-1, 1)
    peak = stats[..., 0].float().masked_fill(rows, float("-inf"))
    total = stats[..., 1].float().masked_fill(rows, 0.0)
    return torch.stack((peak, total), dim=-1)


def combine(values: Sequence[torch.Tensor], stats: Sequence[torch.Tensor]) -> torch.Tensor:
    """The softmax-weighted combine of every source's ``values`` [T, L, V] by its ``stats``
    [T, L, 2] ``(max, sum)``, in FP64, rounded once to BF16: ``[T, L * V]``. A row no source holds
    a key for comes out NaN (0 / 0)."""
    v = torch.stack([x.double() for x in values])  # [P, T, L, V]
    s = torch.stack([x.double() for x in stats])  # [P, T, L, 2]
    peak = s[..., 0].amax(dim=0, keepdim=True)
    weight = s[..., 1] * torch.exp(s[..., 0] - peak)
    weight = weight.nan_to_num(nan=0.0)  # -inf - -inf: a source with no key
    out = (weight.unsqueeze(-1) * v).sum(0) / weight.sum(0).unsqueeze(-1)
    return out.reshape(out.shape[0], -1).to(torch.bfloat16)


def v_up(latent: torch.Tensor, w_uv: torch.Tensor, heads: slice) -> torch.Tensor:
    """BF16 ``[T, L, 128]`` V up-projection of the latent rows ``[T * H, 512]`` of ``heads`` in FP64,
    rounded once (what the tail's senders push)."""
    num_heads = w_uv.shape[0]
    lat = latent.view(-1, num_heads, LATENT)[:, heads].double()
    return torch.einsum("tjk,jnk->tjn", lat, w_uv[heads].double()).to(torch.bfloat16)


def tail(
    rows: Sequence[torch.Tensor],
    stats: Sequence[torch.Tensor],
    w_uv: torch.Tensor,
    masks: Sequence[Optional[torch.Tensor]],
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """What rank ``cp_rank`` of the Helix tail returns, from every rank's latent ``rows`` and
    ``stats`` (and zero-KV ``masks``), in rank order."""
    num_heads = w_uv.shape[0]
    local = num_heads // cp_size
    heads = slice(cp_rank * local, (cp_rank + 1) * local)
    values, weights = [], []
    for latent, st, mask in zip(rows, stats, masks):
        values.append(v_up(latent, w_uv, heads))
        weights.append(masked_stats(st.view(-1, num_heads, 2), mask)[:, heads])
    return combine(values, weights)
