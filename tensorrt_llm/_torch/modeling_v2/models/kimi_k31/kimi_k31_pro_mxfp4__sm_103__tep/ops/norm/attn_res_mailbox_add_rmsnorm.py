# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 residual add of the fused MoE tail's output (read from its mailbox) + attention-residual
mix + RMSNorm through the family's PDL kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm import _attn_res_common as _c

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import (
        attn_res as _k3_attn_res,
    )


class K31AttnResMailboxAddRmsnorm(OpWrapper):
    """``(updated, out)``: ``updated = bf16(prefix + moe)`` with ``moe`` the first ``M`` rows of the
    fused latent-MoE tail's mailbox, then ``out`` as in ``k31_attn_res_rmsnorm``; re-arms the rows
    it read (every lane back to ``-0``).

    ``mailbox`` is what ``KimiK3LatentMoETail.forward_to_mailbox`` returned (``bf16 [1, capacity,
    7168]``, the local copy of the tail's multicast all-gather): the kernel polls each 16-byte
    vector after ``griddepcontrol.wait`` until none of its lanes is ``-0`` (the tail stores ``+0``
    for a ``-0`` result), so it replaces the tail's copy kernel, the residual add and the mix.
    ``updated`` has the bits of ``prefix + moe`` in torch (round-to-nearest add).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(c for c in _c.cells(add=True) if c.spec["rows"] <= 192)

    note = """
    The next layer's pre-attention mix after a fused-tail MoE layer at attention TP (rows up to the
    tail's capacity, 192). Every row of the mailbox the call reads must be filled by exactly one
    tail call and read by exactly one consumer before the next tail call on the same mailbox
    starts (the tail's producers wait for their predecessors before they trigger, so a consumer
    launched between two tail calls always completes first). A mailbox row that is never filled
    hangs the kernel. Layers whose mix cannot take it (the drafter's tap layers, the model's last
    layer) consume the mailbox with the tail's copy kernel instead.
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        mailbox: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k3_attn_res.k31_attn_res_mailbox_add_rmsnorm(
            prefix, mailbox, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )

    def reference(
        self,
        prefix: torch.Tensor,
        mailbox: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moe = mailbox.reshape(-1, mailbox.shape[-1])[: prefix.shape[0]]
        updated = prefix + moe
        out = _c.reference(updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)
        return updated, out

    def is_valid(
        self,
        prefix: torch.Tensor,
        mailbox: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        _c.check_operands(prefix, block, res_weight, rms_weight, out_weight)
        assert mailbox.dim() == 3 and mailbox.shape[0] == 1 and mailbox.shape[2] == _c.HIDDEN, (
            "mailbox must be [1, capacity, 7168]"
        )
        assert mailbox.shape[1] >= prefix.shape[0], "the mailbox holds fewer rows than M"
        assert mailbox.dtype == torch.bfloat16 and mailbox.is_contiguous(), (
            "contiguous BF16 mailbox"
        )


k31_attn_res_mailbox_add_rmsnorm = K31AttnResMailboxAddRmsnorm()
