# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 residual add, attention-residual selection and the RMSNorm that consumes it, in one launch."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HIDDEN = 7168
_EPS = 1e-5


class AttnResAddRmsnormFwd(OpWrapper):
    """Return `(updated, output)`: `updated = bf16(layer_residual + layer_residual_add)` and
    `output = rmsnorm(mix(block_residual, updated)) * output_rms_weight`, both new bf16 [T, 1, 7168].

    The mixture is attn_res_fwd's (`res_weight`, `rms_weight`, `rms_eps`) over the K snapshots of
    `block_residual` and, as the last candidate, the updated residual; the trailing norm is
    attn_res_rmsnorm_fwd's (`output_rms_weight`, `output_rms_eps`).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 one-token decode after attention, N={candidates}: "
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
            for candidates in (3, 8)
            for eps, output_eps in ((1e-2, 1e-5), (1e-5, 1e-2))
        ),
    )
    note = """
    The residual add rounds to bf16 exactly as a torch bf16 add does
    (round-to-nearest of the fp32 sum), so `updated` is bit-identical to
    `layer_residual + layer_residual_add`; that rounded value is the last
    candidate of the mixture. The rest is attn_res_rmsnorm_fwd's contract:
    the mixture rounds to bf16, the normalization rounds to bf16 before the
    weight multiply rounds again, fast-math rsqrt / softmax / reciprocal and a
    fixed reduction order can differ from torch in the last bits; identical
    inputs give identical bits. Inputs are read-only; both outputs are new.

    No metadata, workspace or caller-visible state. The op requires B == 1,
    H == 7168 and N in [1, 12]; N <= 4 runs one CTA per token, N >= 5 one
    eight-CTA split-K cluster per token. TRTLLM_ENABLE_PDL changes launch
    scheduling, not results. Shape-only launches capture after warm-up and
    replay bit-identically.

    The Kimi K3.1 target calls it only for a one-token rank batch (T == 1), after
    the attention of a layer that appends no snapshot, with N = 2..9 and both
    epsilons 1e-5. Larger token counts take a torch add, attn_res_fwd and a
    separate RMSNorm.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
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
        return torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
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
        layer_residual_add: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The bf16 add, then the fp32 evaluation rounding where the kernel rounds."""
        updated = (layer_residual.float() + layer_residual_add.float()).to(torch.bfloat16)
        values = torch.cat((block_residual, updated.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * rsigma * score_weight).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        mixed = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16).float()
        normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps)
        return updated, output_rms_weight.reshape(-1) * normed.to(torch.bfloat16)


attn_res_add_rmsnorm_fwd = AttnResAddRmsnormFwd()
