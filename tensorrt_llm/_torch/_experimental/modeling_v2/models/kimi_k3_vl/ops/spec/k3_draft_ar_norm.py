# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's all-reduce site: finish a pushed MNNVL all-reduce, add the residual and
apply the following RMSNorm in one CuTe DSL launch."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ...kernels.k3r_spec.dspark_draft_ar import ar_norm_config, run_draft_ar_norm


class K3DraftArNorm(OpWrapper):
    """``residual_out = bf16(bf16(sum over ranks of the pushed partials) + residual)`` and
    ``out = RMSNorm(residual_out) * gamma`` (BF16), written into the two contiguous ``[M,
    hidden]`` outputs. Returns None.

    Every rank pushed its BF16 partial into ``workspace`` (a ``k3_lamport_ar.LamportWorkspace``)
    with the ``one_shot`` or two-shot transport; ``cpp_one_shot`` selects which C++ fused
    all-reduce (``RESIDUAL_RMS_NORM``) the outputs reproduce bit for bit: its one-shot kernel
    (``bf16((x * r) * gamma)``) or two-shot + ``rmsNormLamport`` (``bf16((gamma * x) * r)``).
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    The ranks are summed in FP32 in rank order; the RMS reduction sums BF16 squares in a fixed
    order and uses ``rsqrt.approx``. Launched behind the producer with PDL: it releases its
    dependents first and polls the Lamport slots while the producer runs; only CTA (0, 0) waits
    for the producer before advancing the workspace's rotation state. Up to 64 rows with a
    single-load launch shape (``ar_norm_config``); raises ValueError otherwise. A process-wide
    compile cache holds one kernel per (world, hidden, threads, cluster, transport, C++ path,
    PDL). Every rank must make the same call.
    """

    def __call__(
        self,
        workspace,
        one_shot: bool,
        cpp_one_shot: bool,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        use_pdl: bool = True,
    ) -> None:
        run_draft_ar_norm(
            workspace,
            one_shot,
            cpp_one_shot,
            residual,
            gamma,
            eps,
            out,
            residual_out,
            use_pdl=use_pdl,
        )

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        one_shot: bool,
        cpp_one_shot: bool,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        use_pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(out, residual_out)``, BF16. A collective: ``partials`` (in place of ``workspace``)
        is the group's pushed BF16 partials in rank order. The sum is FP32 in rank order, rounded
        to BF16 and added to the residual with one rounding; the norm is FP64 from that BF16
        residual, rounded once. The transport flags and the output buffers are not read."""
        acc = torch.zeros(residual.shape, dtype=torch.float32, device=residual.device)
        for partial in partials:
            acc += partial.float()
        new_residual = (acc.to(torch.bfloat16).float() + residual.float()).to(torch.bfloat16)
        x = new_residual.double()
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        norm = (x * inv_rms * gamma.double()).to(torch.bfloat16)
        return norm, new_residual

    def is_valid(
        self,
        workspace,
        one_shot: bool,
        cpp_one_shot: bool,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        use_pdl: bool = True,
    ) -> None:
        # Raw addresses: shapes, dtypes and layouts are not checked by the launcher.
        for t in (residual, out, residual_out):
            assert t.dtype == torch.bfloat16 and t.is_contiguous() and t.shape == out.shape
        assert gamma.dtype == torch.bfloat16 and gamma.numel() == out.shape[1]
        assert ar_norm_config(out.shape[0], out.shape[1], out.device)[0], "no launch shape"


k3_draft_ar_norm = K3DraftArNorm()
