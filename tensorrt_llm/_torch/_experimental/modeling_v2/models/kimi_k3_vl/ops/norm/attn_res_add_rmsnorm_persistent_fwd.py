# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 optional residual add, attention-residual selection and trailing RMSNorm on a persistent grid."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

native_ops = load_family_ops(__name__)


class AttnResAddRmsnormPersistentFwd(OpWrapper):
    """Return `(updated, out)` like `attn_res_add_rmsnorm_fwd`, with `layer_residual_add` optional.

    With an addend, `updated = bf16(layer_residual + layer_residual_add)` is a new tensor and the last candidate.
    Without one, `layer_residual` is the last candidate and `updated` IS `layer_residual` (same storage, not a copy).
    `out` is a new bf16 [T, 1, 7168] tensor: rmsnorm(selection) * output_rms_weight, rounded as in
    `attn_res_rmsnorm_fwd`. `sm_reserve` SMs are left out of the grid; it must be in [0, SM count).
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3 T={tokens}, N={candidates}, add={add}: prefill and wide decode token counts",
                spec=dict(
                    tokens=tokens,
                    candidates=candidates,
                    hidden=7168,
                    eps=1e-6,
                    output_eps=1e-6,
                    add=add,
                    sm_reserve=0,
                ),
            )
            for tokens in (1, 33, 300, 2048)
            for candidates in (2, 5, 9)
            for add in (False, True)
        ),
        *(
            Cell(
                why=f"T=4096, N={candidates}, add=True: grid smaller than the token count",
                spec=dict(
                    tokens=4096,
                    candidates=candidates,
                    hidden=7168,
                    eps=1e-6,
                    output_eps=1e-6,
                    add=True,
                    sm_reserve=0,
                ),
            )
            for candidates in (3, 4, 6, 7, 8)
        ),
        *(
            Cell(
                why=f"sm_reserve={sm_reserve} at T=300, N=9: a smaller persistent grid",
                spec=dict(
                    tokens=300,
                    candidates=9,
                    hidden=7168,
                    eps=1e-6,
                    output_eps=1e-6,
                    add=True,
                    sm_reserve=sm_reserve,
                ),
            )
            for sm_reserve in (1, 16)
        ),
        *(
            Cell(
                why=f"T=64, N=4, eps=({eps},{output_eps}): distinct norm roots",
                spec=dict(
                    tokens=64,
                    candidates=4,
                    hidden=7168,
                    eps=eps,
                    output_eps=output_eps,
                    add=True,
                    sm_reserve=0,
                ),
            )
            for eps, output_eps in ((1e-5, 1e-5), (1e-2, 1e-6), (1e-6, 1e-2))
        ),
    )
    note = """
    Optional residual add, selection and trailing norm; no pre-norm mixture or
    scoring statistics. Inputs are read-only. Without an addend the first
    result aliases layer_residual, which the schema does not declare: callers
    must not consume it then (the fake op returns a new tensor). Rounding as in
    attn_res_add_rmsnorm_fwd; the reduction order differs from the per-token
    kernels, so the two can differ in the last bits. Each call is deterministic.

    No metadata, workspace or caller-visible state. One 288-thread CTA per SM
    (minus sm_reserve) loops over tokens: one producer warp issues bulk copies,
    eight consumer warps hold fp32 rows in Tensor Memory and keep the softmax
    state in registers; candidates are processed in chunks of three. Requires
    N in [2, 9]; H=7168 only; target 100f. With TRTLLM_ENABLE_PDL unset or 1
    the launch allows PDL; the kernel waits for its predecessor before reading
    and releases dependents after every warp's final store. Shape-only launches
    capture after warm-up and replay eager bits. The kernels are the family's
    copy of trtllm::attn_res_add_rmsnorm_persistent_fwd, compiled with the
    same flags; both give the same bits. The Kimi K3 targets call this with
    KIMI_K3_ATTN_RES_TOPOLOGY=persistent at every shape it takes, or =split
    (alias 1) above KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS; the default per_token
    never calls it.
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
        sm_reserve: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return native_ops.k3_attn_res_add_rmsnorm_persistent_fwd(
            layer_residual,
            layer_residual_add,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
            sm_reserve,
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
        sm_reserve: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 evaluation of the contract's Semantics, rounding where the kernel rounds."""
        updated = (
            layer_residual if layer_residual_add is None else layer_residual + layer_residual_add
        )
        values = torch.cat((block_residual, updated.unsqueeze(0)), dim=0).float()
        rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
        score_weight = rms_weight.float().reshape(-1) * res_weight.float().reshape(-1)
        logits = (values * rsigma * score_weight).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        mixed = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16).float()
        normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps)
        return updated, output_rms_weight.reshape(-1) * normed.to(torch.bfloat16)


attn_res_add_rmsnorm_persistent_fwd = AttnResAddRmsnormPersistentFwd()
