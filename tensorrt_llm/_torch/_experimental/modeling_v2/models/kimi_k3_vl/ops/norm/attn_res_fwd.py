# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 attention-residual selection: a per-token softmax mixture of residual snapshots."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class AttnResFwd(OpWrapper):
    """Mix each token's N = K + 1 candidates: the K snapshots of block_residual, then layer_residual.

    A candidate's logit is its fp32 RMSNorm (weight `rms_weight`, `rms_eps`) dotted with `res_weight`;
    the softmax of the logits weights the raw candidates. `layer_residual` is bf16 [T, 1, H] and
    `block_residual` bf16 [K, T, 1, H], both contiguous.

    Returns `(output, rsigma, probs, logits)`: the bf16 [T, 1, H] mixture and the fp32 [N, T, 1]
    per-candidate rsqrt(mean square + eps), softmax probabilities and logits, all newly allocated.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 T={tokens}, N={candidates}: snapshot depth and decode/prefill dispatch",
                spec=dict(tokens=tokens, candidates=candidates, hidden=7168, eps=1e-6),
            )
            for tokens in (*range(1, 9), 300, 2048)
            for candidates in range(2, 10)
        ),
        *(
            Cell(
                why=f"T={tokens}, N={candidates}: {why}",
                spec=dict(tokens=tokens, candidates=candidates, hidden=7168, eps=1e-6),
            )
            for tokens, candidates, why in (
                (1, 1, "single-CTA with no snapshot"),
                (1, 10, "online decode beyond K3 candidate range"),
                (1, 11, "online decode partial final candidate chunk"),
                (1, 12, "split-K maximum candidate count"),
                (8, 1, "online single candidate"),
                (8, 12, "online three full candidate chunks"),
                (1024, 12, "fixed-N=12 online variant"),
            )
        ),
        *(
            Cell(
                why=f"eps={eps} at T={tokens}, N={candidates}: scoring root and dispatch",
                spec=dict(tokens=tokens, candidates=candidates, hidden=7168, eps=eps),
            )
            for tokens, candidates in ((1, 2), (1, 8), (1, 9), (300, 9))
            for eps in (1e-5, 1e-2)
        ),
    )
    note = """
    Selection only: no residual add, trailing norm or snapshot write. The caller
    owns the snapshot bank; inputs are read-only and outputs do not alias them.
    Fast-math rsqrt, softmax and reciprocal, denormal flushing and reduction
    order differ from torch. Each kernel is deterministic, but decode and
    online kernels can differ in the last bits; the fixed-N=12 online variant
    also sums the denominator differently.

    No metadata, workspace or caller-visible state. Device SM counts,
    architecture checks and dynamic-shared-memory limits are cached, and
    TRTLLM_ENABLE_PDL is read once per process; these affect launches only.
    At H=7168, T=1 and N in {1,2,4} use one 256-thread CTA; N in {8,12} use
    an eight-CTA split-K cluster. Other calls use a persistent 288-thread
    online kernel per SM, except (T,N)=(1024,12), which uses SM count minus one.
    Online processing streams candidates, holds fp32 rows in Tensor Memory,
    and softmaxes chunks of four. Other hidden sizes/tilings are not certified.
    Kernels target 100f, using f32x2, tcgen05, bulk copies or distributed shared
    memory; online H=7168 uses 115040 bytes of dynamic shared memory per CTA.

    Unset or 1 enables PDL for decode kernels; they wait before reading inputs
    and release dependents after the final store. PDL consumers must wait for
    visibility. Online launches and non-PDL consumers use normal stream ordering.
    Shape-only launches can be captured after warm-up and replay eager bits.
    Kimi K3 uses this unfused path above KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS
    (default 1) unless persistent topology handles it, and wherever a drafter
    needs the pre-norm mixture. The fake op reports the same shapes and dtypes.
    """

    def __call__(
        self,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.attn_res_fwd(
            layer_residual, block_residual, res_weight, rms_weight, rms_eps
        )

    def reference(
        self,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 evaluation of the contract's Semantics; snapshots first, the layer residual last."""
        values = torch.cat((block_residual, layer_residual.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * rsigma.unsqueeze(-1) * score_weight).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        output = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16)
        return output, rsigma, probs, logits

    def is_valid(
        self,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
    ) -> None:
        tensors = (block_residual, res_weight, rms_weight)
        if layer_residual.is_cuda and all(tensor.is_cuda for tensor in tensors):
            assert all(tensor.device == layer_residual.device for tensor in tensors), (
                "all inputs must be on layer_residual's device; the kernel dereferences them there"
            )


attn_res_fwd = AttnResFwd()
