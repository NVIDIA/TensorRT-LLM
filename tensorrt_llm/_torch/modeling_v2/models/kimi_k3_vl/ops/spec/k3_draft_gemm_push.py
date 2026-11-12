# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's low-M BF16 split-K GEMM whose epilogue pushes its output into the Lamport
slots of the MNNVL all-reduce that follows (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ...kernels.k3r_gemm.low_m_bf16_splitk import SplitKTactic
from ...kernels.k3r_spec.dspark_draft_gemm_push import MAX_M, run_gemm_push, validate_push_tactic


class K3DraftGemmPush(OpWrapper):
    """Pushes ``x [M, K] @ weight[N, K]^T`` (BF16, FP32 accumulation over split-K slices) into
    ``target`` (a ``layers.LamportPushTarget`` of an all-reduce of ``[M, N]``) with the
    target's transport and sets ``target.pushed``; with ``store_out`` also writes it into ``out``
    (``[M, N]``, contiguous, whose shape sets the launch either way). Returns None.

    ``x`` and ``weight`` are contiguous BF16, 32-byte aligned; ``tactic`` a ``SplitKTactic`` that
    ``validate_push_tactic`` admits for ``(M, N, K)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    The owner CTA of each cluster reduces the split-K partials, stages the tile token-major and
    stores it as 16-byte vectors into every rank's slot (one-shot, through the multicast address)
    or the token owner's scatter slot (two-shot), ``-0`` as ``+0``; ``k3_draft_ar_norm`` or the
    Lamport finish then completes the all-reduce. The weight warp streams before
    ``griddepcontrol.wait``; dependents are released right after the activation warp's wait. Up
    to MAX_M (64) rows. A process-wide compile cache holds one kernel per (dtype, tactic, PDL,
    world, N, transport, store_out). Targets are finished in push order by every rank.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        target,
        tactic: SplitKTactic,
        pdl: bool = True,
        store_out: bool = False,
    ) -> None:
        run_gemm_push(x, weight, out, target, tactic, pdl=pdl, store_out=store_out)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        target,
        tactic: SplitKTactic,
        pdl: bool = True,
        store_out: bool = False,
    ) -> torch.Tensor:
        """The BF16 ``[M, N]`` product this rank pushes (and stores into ``out`` with
        ``store_out``): an FP32 product rounded once. Nothing is written."""
        return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        target,
        tactic: SplitKTactic,
        pdl: bool = True,
        store_out: bool = False,
    ) -> None:
        m, k = x.shape
        n = weight.shape[0]
        assert x.dtype == weight.dtype == out.dtype == torch.bfloat16
        assert x.is_contiguous() and weight.is_contiguous() and out.is_contiguous()
        assert tuple(out.shape) == (m, n) and m <= MAX_M
        validate_push_tactic(tactic, m, n, k)
        # The push lays the tile out for the target's [M, N] all-reduce.
        assert tuple(target.output.shape) == (m, n), "the target is for another shape"


k3_draft_gemm_push = K3DraftGemmPush()
