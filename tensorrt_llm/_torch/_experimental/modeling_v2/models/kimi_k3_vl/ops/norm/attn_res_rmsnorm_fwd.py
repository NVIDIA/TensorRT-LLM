# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 attention-residual selection fused with the RMSNorm that consumes it."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

native_ops = load_family_ops(__name__)


class AttnResRmsnormFwd(OpWrapper):
    """Return rmsnorm(attn_res_fwd's bf16 mixture) * output_rms_weight as a new bf16 [T, 1, 7168] tensor.

    `res_weight`, `rms_weight` and `rms_eps` score the candidates as in `attn_res_fwd`;
    `output_rms_weight` and `output_rms_eps` belong to the trailing norm, which rounds the normalized
    value to bf16 before the weight multiply. With PDL enabled the kernel releases its dependents before
    it writes the output, so a dependent PDL kernel must wait on the grid dependency before reading it.
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
            for tokens in (*range(1, 9), 32, 300)
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
            for candidates in (1, 10, 11, 12)
        ),
        *(
            Cell(
                why=f"T={tokens}, N={candidates}, eps=({eps},{output_eps}): distinct norm roots",
                spec=dict(
                    tokens=tokens,
                    candidates=candidates,
                    hidden=7168,
                    eps=eps,
                    output_eps=output_eps,
                ),
            )
            for tokens, candidates in ((1, 4), (1, 9), (8, 2))
            for eps, output_eps in ((1e-5, 1e-5), (1e-2, 1e-6), (1e-6, 1e-2))
        ),
    )
    note = """
    Selection and trailing norm only: no residual add, pre-norm mixture or
    scoring statistics are returned. Inputs are read-only; output is new.
    The mixture rounds to bf16, then normalization rounds to bf16 before the
    weight multiply rounds again, matching KimiK3RMSNorm rather than a
    once-rounded RMSNorm. Fast-math rsqrt, softmax, reciprocal, denormal flushing
    and fixed reduction order can differ from torch. Identical inputs give
    identical bits, independent of token count or token position.

    No metadata, workspace or caller-visible state. Device architecture checks
    and dynamic-shared-memory limits are cached. TRTLLM_ENABLE_PDL is read once
    per process; unset or 1 enables PDL. N<=4 uses one 256-thread CTA per token;
    N>=5 uses one eight-CTA split-K cluster per token. The former retains bf16
    candidates in registers; the latter keeps fp32 candidates in shared memory
    and reduces through distributed shared memory, using 3652*N bytes per CTA.
    Both target 100f and use f32x2. There is no persistent path in this entry.
    The shared contract caps T at 16384 even though the grid accepts any T.

    With PDL enabled, each CTA waits for its predecessor and immediately releases
    dependents, before computing or writing output. A PDL dependent must wait
    before reading output or overwriting these inputs. Non-PDL kernels use
    normal stream ordering; disabling PDL changes scheduling, not arithmetic.
    Shape-only launches capture after warm-up and replay eager bits.
    Kimi K3 calls this before attention, snapshot-appending MLPs and model
    output up to KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS (default 1, topology mode 32),
    unless the persistent topology handles it. Drafter-captured mixtures use
    the unfused entry. The fake op returns empty_like(layer_residual).
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
        return native_ops.k3_attn_res_rmsnorm_fwd(
            layer_residual,
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
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> torch.Tensor:
        """fp32 evaluation of the contract's Semantics, rounding where the kernel rounds."""
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
