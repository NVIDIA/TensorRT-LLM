# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual selection + RMSNorm (+ residual add) on a persistent grid."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HIDDEN = 7168
_EPS = 1e-5


def _cells() -> tuple[Cell, ...]:
    cells = [
        Cell(
            why=f"DEP16 GEN b12 k7 verify step (96 rows, above its PDL kernel's 64), N={n}, "
            f"{'add' if add else 'no add'}",
            spec=dict(tokens=96, candidates=n, add=add, hidden=_HIDDEN, eps=_EPS),
        )
        for n in range(2, 10)
        for add in (False, True)
    ]
    cells += [
        Cell(
            why=f"T={tokens} ({site}), N={n}, {'add' if add else 'no add'}",
            spec=dict(tokens=tokens, candidates=n, add=add, hidden=_HIDDEN, eps=_EPS),
        )
        for tokens, site in (
            (33, "just above the per-token range"),
            (65, "just above DEP16's PDL kernel"),
            (129, "just above the TEP8 / helix8 PDL kernel"),
            (2048, "a large generation batch"),
        )
        for n in (2, 9)
        for add in (False, True)
    ]
    return tuple(cells)


class RubinAttnResAddRmsnormPersistent(OpWrapper):
    """attn_res_add_rmsnorm_fwd's contract on a persistent grid, rounded once (see ``note``):
    ``(updated, output)`` with ``updated = layer_residual + layer_residual_add`` (bf16) and
    ``output`` the RMSNorm of the mixture of the K snapshots of ``block_residual`` and
    ``updated``. ``layer_residual_add`` may be None: then the mixture's last candidate is
    ``layer_residual`` and ``updated`` IS ``layer_residual`` (an alias, not a copy). Shapes:
    [T, 1, 7168] and [K, T, 1, 7168], bf16.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = _cells()
    note = """
    The engine's trtllm::attn_res_add_rmsnorm_persistent_fwd. One CTA per SM loops over the
    tokens, so prefill-sized T is what it exists for; the per-token kernels
    (attn_res_add_rmsnorm_fwd, the PDL rubin_attn_res_* entries) serve small T. The op checks
    shapes, dtypes, devices and 2 <= N <= 9 itself, and needs an sm_100-family GPU.
    Rounding: the add rounds as a torch bf16 add. The mixture and its norm stay fp32 and the
    output rounds once, bf16(mix * rsqrt(mean(mix^2) + eps) * w); the kernel keeps the softmax
    weights unnormalized and folds their sum into the norm, the same value. The per-token kernels
    also round the mixture and the normalized value, so the two differ by one bf16 step in about
    a third of the elements. Fast-math exp2 / rsqrt and the summation order differ from torch in
    the last fp32 bits. Without an addend the first result aliases layer_residual: do not write
    it. The K3.1 Rubin targets call it in generation steps above their PDL kernel's row limit
    (DEP16 64 rows, TEP8 / helix8 128).
    """

    def __call__(
        self,
        layer_residual: torch.Tensor,
        layer_residual_add: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            layer_residual,
            layer_residual_add,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
        )

    def reference(
        self,
        layer_residual: torch.Tensor,
        layer_residual_add: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The bf16 add, then fp32 in the kernel's order, rounding once at the output."""
        updated = layer_residual
        if layer_residual_add is not None:
            updated = (layer_residual.float() + layer_residual_add.float()).to(torch.bfloat16)
        values = torch.cat((block_residual, updated.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * score_weight).sum(dim=-1, keepdim=True) * rsigma
        # Unnormalized softmax weights; their sum s folds into the norm, as in the kernel:
        # (acc / s) * rsqrt(mean((acc / s)^2) + eps) == acc * rsqrt(mean(acc^2) + eps * s^2).
        weights = torch.exp(logits - logits.amax(dim=0, keepdim=True))
        acc = (weights * values).sum(dim=0)
        s = weights.sum(dim=0)
        rsigma_out = torch.rsqrt(acc.square().mean(dim=-1, keepdim=True) + output_rms_eps * s * s)
        output = acc * rsigma_out * output_rms_weight.float().reshape(-1)
        return updated, output.to(torch.bfloat16)


rubin_attn_res_add_rmsnorm_persistent = RubinAttnResAddRmsnormPersistent()
