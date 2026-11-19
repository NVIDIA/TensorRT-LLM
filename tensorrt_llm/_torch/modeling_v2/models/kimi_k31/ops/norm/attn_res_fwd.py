# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention-residual selection: a per-token softmax mixture of residual snapshots."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Kimi K3.1's hidden size and RMSNorm epsilon (``rms_norm_eps`` of the checkpoint).
_HIDDEN = 7168
_EPS = 1e-5


class AttnResFwd(OpWrapper):
    """Mix each token's N = K + 1 candidates: the K snapshots of `block_residual`, then `layer_residual`.

    A candidate's logit is its fp32 RMSNorm (weight `rms_weight`, `rms_eps`) dotted with `res_weight`;
    the softmax of the logits over the candidates weights the raw candidates. `layer_residual` is bf16
    [T, 1, H] and `block_residual` bf16 [K, T, 1, H], both contiguous; `res_weight` and `rms_weight` are
    bf16 with H elements.

    Returns `(output, rsigma, probs, logits)`: the bf16 [T, 1, H] mixture and the fp32 [N, T, 1]
    per-candidate rsqrt(mean square + eps), softmax probabilities and logits, all newly allocated.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 T={tokens}, N={candidates}: snapshot depth x decode / verify / prefill rows",
                spec=dict(tokens=tokens, candidates=candidates, hidden=_HIDDEN, eps=_EPS),
            )
            for tokens in (1, 2, 5, 8, 40, 300, 2048)
            for candidates in range(2, 10)
        ),
        Cell(
            why="K3.1 T=16384, N=9: the largest slice the caller hands the op from a longer prefill chunk",
            spec=dict(tokens=16384, candidates=9, hidden=_HIDDEN, eps=_EPS),
        ),
        Cell(
            why="K3.1 T=1024, N=9: a prefill chunk at the deepest snapshot bank",
            spec=dict(tokens=1024, candidates=9, hidden=_HIDDEN, eps=_EPS),
        ),
        *(
            Cell(
                why=f"K3.1 sm_107 T=96, N={candidates}: Helix CP 8 b16 k5 / DEP16 b12 k7 verify step",
                spec=dict(tokens=96, candidates=candidates, hidden=_HIDDEN, eps=_EPS),
            )
            for candidates in range(2, 10)
        ),
        *(
            Cell(
                why=f"K3.1 sm_107 T=16384, N={candidates}: a 16k context chunk (TEP8 / DEP16)",
                spec=dict(tokens=16384, candidates=candidates, hidden=_HIDDEN, eps=_EPS),
            )
            for candidates in (2, 5)
        ),
    )
    note = """
    Selection only: no residual add, trailing norm or snapshot write. The caller
    owns the snapshot bank; inputs are read-only and outputs do not alias them.
    Fast-math rsqrt, softmax and reciprocal and the kernels' reduction order
    differ from torch in the last bits. Each kernel is deterministic.

    No metadata, workspace or caller-visible state. At H=7168 the op picks the
    kernel from (T, N): at T=1 a single CTA for N in {1, 2, 4} and an eight-CTA
    split-K cluster for N in {8, 12}; a persistent online kernel per SM
    everywhere else (with a fixed-N=12 variant at (T, N) == (1024, 12), which
    Kimi K3.1 never reaches). The op takes T <= 16384 tokens and B == 1; the
    Kimi K3.1 target slices longer prefill chunks into 16384-token calls, as the
    op is per token. TRTLLM_ENABLE_PDL changes launch scheduling, not results.
    Shape-only launches capture after warm-up and replay bit-identically.

    Kimi K3.1 (93 layers, snapshots every 12 layers) calls it with N = 2..9
    candidates, H = 7168 and eps = 1e-5: on every token count above the fused
    one-token epilogues (attn_res_rmsnorm_fwd / attn_res_add_rmsnorm_fwd), and
    on every token count where the DFlash2 drafter taps the pre-norm mixture.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
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
        """fp32 evaluation: snapshots first, the layer residual last; one bf16 rounding of the mixture."""
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
        # The op checks that every input is on a CUDA device but not that they share one; it
        # dereferences every pointer on layer_residual's device.
        tensors = (block_residual, res_weight, rms_weight)
        if layer_residual.is_cuda and all(tensor.is_cuda for tensor in tensors):
            assert all(tensor.device == layer_residual.device for tensor in tensors), (
                "all inputs must be on layer_residual's device; the kernel dereferences them there"
            )


attn_res_fwd = AttnResFwd()
