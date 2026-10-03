# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's residual-update collectives on the catalog's Kimi K3 MNNVL entry.

A layer's post-attention step adds the attention output to the running prefix sum, selects the attention residual
and applies the post-attention RMSNorm. Under TP the attention output is a sum over the TP group, so the step is that
all-reduce followed by the residual update. On a step of at most `AR_ATTN_RES_MAX_TOKENS` tokens (wide decode steps
aside) the attention hands over its unreduced o_proj partial, and `K3DecodeComm.allreduce_attn_res` runs both as one
`comm/mnnvl_allreduce_attn_res` call: the one-shot MNNVL all-reduce with the residual update as its epilogue.

The state is one `MnnvlWorkspace` of the TP group (`K3DecodeComm.create`): collective over the group and eager, built
by the target in `post_load_weights` before any CUDA-graph capture. Every rank must make the same calls on it in the
same order. Whether a step takes the call is decided from its token count and kind alone, which every rank of the
group shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_attn_res import (
    mnnvl_allreduce_attn_res,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)

# The most tokens of a step whose post-attention all-reduce runs one-shot with the residual update in its epilogue.
AR_ATTN_RES_MAX_TOKENS = 16

# The workspace's buffer size: a one-shot call pushes T x 7168 bf16 from each of 16 ranks, 3.5 MiB at T = 16.
MNNVL_BUFFER_BYTES = 4 << 20


def _eps(norm: nn.Module) -> float:
    """The epsilon of a KimiK3RMSNorm (``eps``) or a stock RMSNorm (``variance_epsilon``)."""
    return float(norm.eps if hasattr(norm, "eps") else norm.variance_epsilon)


@dataclass(eq=False)
class K3DecodeComm:
    """The decode path's collective state for the TP group: the `MnnvlWorkspace` the post-attention all-reduces run
    on. Built by `create`; owned by the target and shared by every layer."""

    mnnvl: MnnvlWorkspace

    @classmethod
    def create(cls, mapping) -> "K3DecodeComm":
        """The state for ``mapping``'s TP group. Collective: every rank of the group calls it at the same point,
        eagerly, before any CUDA-graph capture; it fails on every rank or on none (``MnnvlWorkspace.create``)."""
        return cls(MnnvlWorkspace.create(mapping, MNNVL_BUFFER_BYTES))

    def takes_post_attention(self, hidden_states: torch.Tensor, step) -> bool:
        """Whether ``allreduce_attn_res`` runs the post-attention step of a layer whose attention input is
        ``hidden_states``: at most `AR_ATTN_RES_MAX_TOKENS` bf16 rows of a hidden size the op takes, on any step but
        a wide decode step (``step.wide``), whose post-attention update stays the all-reduce and the fused add +
        attn_res + RMSNorm."""
        if step is not None and step.wide:
            return False
        if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
            return False
        rows, hidden = hidden_states.shape
        return (
            hidden % 1024 == 0
            and hidden <= 8192
            and 0 < rows <= min(AR_ATTN_RES_MAX_TOKENS, self.mnnvl.max_one_shot_tokens(hidden))
        )

    def allreduce_attn_res(
        self,
        partial: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_proj: nn.Module,
        res_norm: nn.Module,
        out_norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` in one ``comm/mnnvl_allreduce_attn_res`` call: ``updated = prefix_sum +
        allreduce(partial)`` (the sum alone without ``prefix_sum``), ``normed = out_norm(attn_res(block_residual...,
        updated))``, the attention residual selected with ``res_proj`` and ``res_norm``. ``block_residual`` holds the
        valid snapshots, ``[S, T, H]``."""
        return mnnvl_allreduce_attn_res(
            partial.contiguous(),
            prefix_sum,
            block_residual,
            res_proj.weight.reshape(-1),
            res_norm.weight,
            out_norm.weight,
            _eps(res_norm),
            _eps(out_norm),
            self.mnnvl,
        )
