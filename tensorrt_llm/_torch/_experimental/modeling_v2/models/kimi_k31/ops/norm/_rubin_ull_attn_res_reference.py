# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch ClusterAttnRes, shared by the cluster AttnRes entries' references."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

HIDDEN = 7168


def attn_res(
    key: Sequence,
    weights: Sequence[torch.Tensor],
    eps: Sequence[float],
    aux_mix: bool,
    delta: torch.Tensor,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
) -> dict:
    """What one ClusterAttnRes launch writes, from its inputs (nothing is mutated).

    ``key`` ``(nb, has_delta, write, final)``; ``weights`` the BF16 selection norm, projection and
    output norm; ``eps`` the selection and output epsilons; ``delta`` the FP32 [T, 7168] delta (the
    tensor, or the mailbox slots' sum); ``prefix`` BF16 [T, 7168]; ``blocks`` BF16 [T, >= nb, 7168].

    The new prefix row is ``bf16(prefix + delta)`` (``bf16(delta)`` without ``has_delta``). Each of
    the ``nb`` saved blocks and the new row gets the logit ``(v . norm * proj) * rsqrt(mean(v^2) +
    eps)``; their softmax mixes them in FP32, and the output is the RMSNorm of the mixture (rounded
    to BF16 first at the ``final`` boundary; of the new row alone without blocks) times the output
    norm. Returns ``out``, ``prefix`` (the new row: also ``blocks[:, write]`` where ``write >= 0``)
    and ``aux`` (the BF16 mixture with ``aux_mix``, else the new row).
    """
    nb, has_delta, write, final = key
    sel, proj, gamma = (w.float().reshape(-1) for w in weights)
    sel_eps, out_eps = eps
    new = (prefix.float() + delta if has_delta else delta).to(torch.bfloat16)
    if nb == 0:
        x = new.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + out_eps) * gamma
        return dict(out=out.to(torch.bfloat16), prefix=new, aux=new, write=write)
    values = torch.cat((blocks[:, :nb].float(), new.float().unsqueeze(1)), dim=1)  # [T, nb+1, H]
    rstd = torch.rsqrt(values.pow(2).mean(-1) + sel_eps)
    logits = (values * (sel * proj)).sum(-1) * rstd
    probs = torch.softmax(logits, dim=-1)
    mix = (probs.unsqueeze(-1) * values).sum(1)
    if final:
        mix = mix.to(torch.bfloat16).float()
    out = mix * torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + out_eps) * gamma
    aux = mix.to(torch.bfloat16) if aux_mix else new
    return dict(out=out.to(torch.bfloat16), prefix=new, aux=aux, write=write)


def mailbox_sum(mailbox: torch.Tensor, rows: int, sanitize: bool = False) -> torch.Tensor:
    """FP32 sum of the ``[slots, cap, 7168]`` mailbox's first ``rows`` rows in slot order, rounded
    to BF16 past one slot (the all-reduce boundary it replaces)."""
    total = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=mailbox.device)
    for slot in mailbox[:, :rows]:
        total += slot.float()
    if mailbox.shape[0] > 1:
        total = total.to(torch.bfloat16).float()
        if sanitize:
            total = total + 0.0
    return total


def as_delta(delta: Optional[torch.Tensor], rows: int) -> torch.Tensor:
    return delta[:rows].float()
