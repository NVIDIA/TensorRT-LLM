# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's residual-update collectives on the catalog's Kimi K3 MNNVL and sandwich entries.

A layer's post-attention step adds the attention output to the running prefix sum, selects the attention residual
and applies the post-attention RMSNorm. Under TP the attention output is a sum over the TP group, so the step is that
all-reduce followed by the residual update. On a step of at most `AR_ATTN_RES_MAX_TOKENS` tokens (wide decode steps
aside) one collective runs both:

* `K3DecodeComm.sandwich_oproj`, `comm/k3_sandwich_oproj`: o_proj, its all-reduce and the residual update in one
  kernel, at most `SANDWICH_MAX_TOKENS` tokens of an o_proj of the TP16 per-rank shape [7168, 768]. The attention
  hands over its gated o_proj input.
* `K3DecodeComm.allreduce_attn_res`, `comm/mnnvl_allreduce_attn_res`, everywhere else: the one-shot MNNVL all-reduce
  of the attention's unreduced o_proj partial, with the residual update as its epilogue.

The state is one `MnnvlWorkspace` and one `K3SandwichWorkspace` of the TP group (`K3DecodeComm.create`): collective
over the group and eager, built by the target in `post_load_weights` before any CUDA-graph capture. Every rank must
make the same calls on each in the same order. Which call a step takes is decided from its token count and kind and
from the load-time layout alone, which every rank of the group shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_oproj import (
    K3SandwichWorkspace,
    k3_sandwich_oproj,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_attn_res import (
    mnnvl_allreduce_attn_res,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)

# The kernel's support predicate: metadata reads only.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import op as _sandwich_op

# The most tokens of a step whose post-attention all-reduce runs one-shot with the residual update in its epilogue.
AR_ATTN_RES_MAX_TOKENS = 16

# The most tokens the sandwich kernel takes (one token tile).
SANDWICH_MAX_TOKENS = _sandwich_op.MAX_TOKENS

# The MNNVL workspace's buffer size: a one-shot call pushes T x 7168 bf16 from each of 16 ranks, 3.5 MiB at T = 16.
MNNVL_BUFFER_BYTES = 4 << 20


def _eps(norm: nn.Module) -> float:
    """The epsilon of a KimiK3RMSNorm (``eps``) or a stock RMSNorm (``variance_epsilon``)."""
    return float(norm.eps if hasattr(norm, "eps") else norm.variance_epsilon)


def _res_args(res_proj: nn.Module, res_norm: nn.Module, out_norm: nn.Module) -> tuple:
    """The residual update's weights and epsilons in the entries' order."""
    return (
        res_proj.weight.reshape(-1),
        res_norm.weight,
        out_norm.weight,
        _eps(res_norm),
        _eps(out_norm),
    )


@dataclass(eq=False)
class K3DecodeComm:
    """The decode path's collective state for the TP group: the `MnnvlWorkspace` and the `K3SandwichWorkspace` the
    post-attention steps run on. Built by `create`; owned by the target and shared by every layer."""

    mnnvl: MnnvlWorkspace
    sandwich: K3SandwichWorkspace

    @classmethod
    def create(cls, mapping, oproj_weight: Optional[torch.Tensor] = None) -> "K3DecodeComm":
        """The state for ``mapping``'s TP group. Collective: every rank of the group calls it at the same point,
        eagerly, before any CUDA-graph capture; each workspace fails on every rank or on none.

        ``oproj_weight``: an o_proj weight ``takes_oproj`` holds for. The sandwich kernel compiles here for its
        shape, with one call on a zero row of a zero weight, so no capture compiles it; the call advances the
        sandwich workspace on every rank alike."""
        state = cls(
            MnnvlWorkspace.create(mapping, MNNVL_BUFFER_BYTES),
            K3SandwichWorkspace.create(mapping),
        )
        if oproj_weight is not None:
            weight = torch.zeros_like(oproj_weight)
            hidden = weight.shape[0]
            ones = weight.new_ones(hidden)
            k3_sandwich_oproj(
                weight.new_zeros(1, weight.shape[1]),
                weight,
                None,
                weight.new_zeros(0, 1, hidden),
                weight.new_zeros(hidden),
                ones,
                ones,
                1e-6,
                1e-6,
                state.sandwich,
            )
            torch.cuda.synchronize(weight.device)
        return state

    def takes_post_attention(self, hidden_states: torch.Tensor, step) -> bool:
        """Whether one of this state's collectives runs the post-attention step of a layer whose attention input is
        ``hidden_states``: at most `AR_ATTN_RES_MAX_TOKENS` bf16 rows of a hidden size the MNNVL entry takes, on any
        step but a wide decode step (``step.wide``), whose post-attention update stays the all-reduce and the fused
        add + attn_res + RMSNorm."""
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

    @staticmethod
    def takes_oproj(o_proj: nn.Module, max_snapshots: int) -> bool:
        """Whether ``sandwich_oproj`` takes the layer of output projection ``o_proj`` on a step of at most
        `SANDWICH_MAX_TOKENS` tokens, decided once the weights are final: a bias-free, contiguous bf16 weight of the
        kernel's shape (the TP16 per-rank [7168, 768]) and a snapshot bank of at most ``max_snapshots`` rows the
        kernel's candidate count holds."""
        weight = getattr(o_proj, "weight", None)
        return (
            getattr(o_proj, "bias", None) is None
            and isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.is_cuda
            and _sandwich_op.supports(weight.new_empty((1, weight.shape[1])), weight, max_snapshots)
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
            *_res_args(res_proj, res_norm, out_norm),
            self.mnnvl,
        )

    def sandwich_oproj(
        self,
        core: torch.Tensor,
        o_weight: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_proj: nn.Module,
        res_norm: nn.Module,
        out_norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` as ``allreduce_attn_res`` of ``core @ o_weight.T``, in one ``comm/k3_sandwich_oproj``
        call: ``core`` is this rank's gated o_proj input ``[T, 768]``, ``o_weight`` its o_proj slice. Bit for bit
        o_proj followed by ``allreduce_attn_res``."""
        return k3_sandwich_oproj(
            core.contiguous(),
            o_weight,
            prefix_sum,
            block_residual,
            *_res_args(res_proj, res_norm, out_norm),
            self.sandwich,
        )
