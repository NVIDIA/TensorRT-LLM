# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 residual add, attention-residual selection and the RMSNorm that consumes it, in one kernel."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

native_ops = load_family_ops(__name__)


class AttnResAddRmsnormFwd(OpWrapper):
    """Return `(updated, out)`: `updated = bf16(layer_residual + layer_residual_add)` and
    `out = attn_res_rmsnorm_fwd(updated, block_residual, ...)`, both new bf16 [T, 1, 7168] tensors.

    `updated` is the last candidate, after the K snapshots of `block_residual`. `res_weight`, `rms_weight` and
    `rms_eps` score the candidates as in `attn_res_fwd`; `output_rms_weight` and `output_rms_eps` belong to the
    trailing norm, which rounds the normalized value to bf16 before the weight multiply.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 T={tokens}, N={candidates}: snapshot depth and fused token count",
                spec=dict(
                    tokens=tokens, candidates=candidates, hidden=7168, eps=1e-6, output_eps=1e-6
                ),
            )
            for tokens in (1, 2, 8, 32)
            for candidates in range(2, 10)
        ),
        *(
            Cell(
                why=f"T={tokens}, N={candidates}: no snapshot or split-K beyond K3 depth",
                spec=dict(
                    tokens=tokens, candidates=candidates, hidden=7168, eps=1e-6, output_eps=1e-6
                ),
            )
            for tokens in (1, 8)
            for candidates in (1, 10, 12)
        ),
        *(
            Cell(
                why=f"T=1, N=4, eps=({eps},{output_eps}): distinct norm roots",
                spec=dict(tokens=1, candidates=4, hidden=7168, eps=eps, output_eps=output_eps),
            )
            for eps, output_eps in ((1e-5, 1e-5), (1e-2, 1e-6), (1e-6, 1e-2))
        ),
    )
    note = """
    Residual add, selection and trailing norm: the pre-norm mixture and scoring
    statistics are not returned. Inputs are read-only; both outputs are new.
    updated is the bf16-rounded sum, the value an unfused add would store; the
    mixture rounds to bf16, then normalization rounds to bf16 before the weight
    multiply rounds again, matching KimiK3RMSNorm. Fast-math rsqrt, softmax,
    reciprocal, denormal flushing and fixed reduction order can differ from
    torch. Identical inputs give identical bits, independent of token count.

    No metadata, workspace or caller-visible state. Same kernels and launch
    policy as attn_res_rmsnorm_fwd (N<=4: one 256-thread CTA per token; N>=5:
    one eight-CTA split-K cluster per token), with the add folded into the
    candidate load. With PDL enabled each CTA waits for its predecessor and
    releases dependents right after that wait, before writing either output.
    A PDL dependent must wait before reading the outputs. Shape-only launches
    capture after warm-up and replay eager bits. The Kimi K3 targets call this
    after attention and MLPs that do not append a snapshot, up to
    KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS (wide decode steps: 32), unless the
    persistent topology handles it. The fake op returns two empty_like tensors.
    """

    def __call__(
        self,
        layer_residual: torch.Tensor,
        layer_residual_add: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return native_ops.k3_attn_res_add_rmsnorm_fwd(
            layer_residual,
            layer_residual_add,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
            early_trigger=True,
        )

    def reference(
        self,
        layer_residual: torch.Tensor,
        layer_residual_add: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 evaluation of the contract's Semantics, rounding where the kernel rounds."""
        updated = layer_residual + layer_residual_add
        values = torch.cat((block_residual, updated.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * rsigma * score_weight).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        mixed = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16).float()
        normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps)
        return updated, output_rms_weight.reshape(-1) * normed.to(torch.bfloat16)


attn_res_add_rmsnorm_fwd = AttnResAddRmsnormFwd()
