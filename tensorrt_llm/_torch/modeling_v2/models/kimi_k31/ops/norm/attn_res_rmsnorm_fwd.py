# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual selection fused with the RMSNorm that consumes it."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HIDDEN = 7168
_EPS = 1e-5


class AttnResRmsnormFwd(OpWrapper):
    """Return rmsnorm(attn_res_fwd's bf16 mixture) * output_rms_weight as a new bf16 [T, 1, 7168] tensor.

    `res_weight`, `rms_weight` and `rms_eps` score the candidates as in `attn_res_fwd`;
    `output_rms_weight` and `output_rms_eps` belong to the trailing norm.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 one-token decode, N={candidates}: "
                f"{'single-CTA' if candidates <= 4 else 'eight-CTA split-K cluster'} kernel",
                spec=dict(
                    tokens=1, candidates=candidates, hidden=_HIDDEN, eps=_EPS, output_eps=_EPS
                ),
            )
            for candidates in range(2, 10)
        ),
        *(
            Cell(
                why=f"T=1, N={candidates}, eps=({eps},{output_eps}): distinct norm roots",
                spec=dict(
                    tokens=1, candidates=candidates, hidden=_HIDDEN, eps=eps, output_eps=output_eps
                ),
            )
            for candidates in (4, 9)
            for eps, output_eps in ((1e-2, 1e-5), (1e-5, 1e-2))
        ),
    )
    note = """
    Selection and trailing norm only: no residual add, pre-norm mixture or
    scoring statistics are returned. Inputs are read-only; the output is new.
    The mixture rounds to bf16, then the normalization rounds to bf16 before the
    weight multiply rounds again (the Kimi checkpoint RMSNorm's rounding, not a once-rounded
    RMSNorm). Fast-math rsqrt / softmax / reciprocal and a fixed reduction order
    can differ from torch in the last bits; identical inputs give identical bits.

    No metadata, workspace or caller-visible state. The op requires B == 1,
    H == 7168 and N in [1, 12]. N <= 4 runs one 256-thread CTA per token, N >= 5
    one eight-CTA split-K cluster per token. TRTLLM_ENABLE_PDL changes launch
    scheduling, not results. Shape-only launches capture after warm-up and replay
    bit-identically.

    The Kimi K3.1 target calls it only for a one-token rank batch (T == 1): before
    attention and before a snapshot-appending MLP, and at the model output, with
    N = 2..9 and both epsilons 1e-5. Larger token counts take attn_res_fwd and a
    separate RMSNorm.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.attn_res_rmsnorm_fwd(
            layer_residual,
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
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> torch.Tensor:
        """fp32 evaluation, rounding where the kernel rounds (the mixture, the normalized value, the product)."""
        values = torch.cat((block_residual, layer_residual.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * rsigma * score_weight).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        mixed = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16).float()
        normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps)
        # bf16 times bf16: the exact product, rounded to bf16 once more.
        return output_rms_weight.reshape(-1) * normed.to(torch.bfloat16)


attn_res_rmsnorm_fwd = AttnResRmsnormFwd()
