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

The pre-attention step after a MoE layer whose row-parallel tail was handed on (a `PendingTail`, `decode_moe.py`) is
the same kind of collective: `K3DecodeComm.sandwich_tail`, `comm/k3_sandwich_tail`, runs the tail GEMV, its
all-reduce and the next layer's residual update (the final norm's, after the last layer) in one kernel.

The DSpark drafter (`K3DSparkDrafter`) runs its collectives with a plain residual add and RMSNorm on the same state:

* `K3DecodeComm.sandwich_plain`, `comm/k3_sandwich_plain`: a row-parallel projection of at most `SANDWICH_MAX_TOKENS`
  tokens (a layer's attention output projection, or its MLP's SiLU-and-mul and down projection), its all-reduce, the
  residual add and the RMSNorm in one kernel.
* `K3DecodeComm.allreduce_norm`, `comm/mnnvl_fusion_allreduce`: the all-reduce of an unreduced projection output of
  any token count the MNNVL workspace holds, with the residual add and the RMSNorm in its epilogue (the context
  projection's with a zero residual).

The state is one `MnnvlWorkspace` and one `K3SandwichWorkspace` of the TP group (`K3DecodeComm.create`): collective
over the group and eager, built by the target in `post_load_weights` before any CUDA-graph capture. Every rank must
make the same calls on each in the same order. Which call a step takes is decided from its token count and kind and
from the load-time layout alone, which every rank of the group shares.

The plain MNNVL all-reduces the decode path keeps (the stock modules') send one-shot up to
`DECODE_AR_ONE_SHOT_MAX_BYTES` (`use_decode_one_shot`), except a wide decode step's (`wide_all_reduce`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_oproj import (
    K3SandwichWorkspace,
    k3_sandwich_oproj,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_plain import (
    k3_sandwich_plain,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_sandwich_tail import (
    k3_sandwich_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce_attn_res import (
    mnnvl_allreduce_attn_res,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_fusion_allreduce import (
    mnnvl_fusion_allreduce,
    required_buffer_bytes,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)

# The kernel's support predicate: metadata reads only.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import op as _sandwich_op
from tensorrt_llm._torch.distributed import AllReduceParams

# The most tokens of a step whose post-attention all-reduce runs one-shot with the residual update in its epilogue.
AR_ATTN_RES_MAX_TOKENS = 16

# The most tokens the sandwich kernel takes (one token tile).
SANDWICH_MAX_TOKENS = _sandwich_op.MAX_TOKENS

# The MNNVL workspace's buffer size: a one-shot call pushes T x 7168 bf16 from each of 16 ranks, 3.5 MiB at T = 16.
MNNVL_BUFFER_BYTES = 4 << 20

# The one-shot ceiling of the stock MNNVL all-reduces on the decode path: 8 tokens x 7168 x 16 ranks x 2 B is
# 1.75 MiB, which the stock 1 MiB ceiling would send two-shot.
DECODE_AR_ONE_SHOT_MAX_BYTES = 4 << 20

# A wide decode step's ceiling: the stock 1 MiB. At 16 ranks two-shot is faster for every wide step's rows.
WIDE_AR_ONE_SHOT_MAX_BYTES = 1 << 20


def use_decode_one_shot(model: nn.Module) -> None:
    """Every stock MNNVL all-reduce of ``model`` sends one-shot up to `DECODE_AR_ONE_SHOT_MAX_BYTES`. Each grows its
    workspace on its first eager call of a larger size, before the capture of that size."""
    for module in model.modules():
        mnnvl = getattr(module, "mnnvl_allreduce", None)
        if mnnvl is not None:
            mnnvl.one_shot_max_bytes = DECODE_AR_ONE_SHOT_MAX_BYTES


def skip_all_reduce() -> AllReduceParams:
    """All-reduce parameters under which a row-parallel module returns this rank's unreduced output (its projection
    without its all-reduce): a ``Linear``'s ``all_reduce_params``, a ``GatedMLP``'s ``final_all_reduce_params``."""
    return AllReduceParams(enable_allreduce=False)


def wide_all_reduce(all_reduce: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``all_reduce(x)`` of a wide decode step (a stock ``AllReduce`` module, no fusion): its MNNVL all-reduce with
    the `WIDE_AR_ONE_SHOT_MAX_BYTES` ceiling, else the module itself."""
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if mnnvl is not None:
        out = mnnvl(
            x.contiguous(), AllReduceParams(), one_shot_max_bytes=WIDE_AR_ONE_SHOT_MAX_BYTES
        )
        if out is not None:
            return out
    return all_reduce(x)


class PendingTail(NamedTuple):
    """A MoE layer's row-parallel tail left to its consumer's fused pre-attention step (``K3DecodeComm.sandwich_tail``):
    the reduced latent ``[T, 3584]``, the shared experts' activation, the tail weight ``[latent up columns | padding |
    shared down]``, this rank's first latent column and the latent norm's epsilon."""

    latent: torch.Tensor
    act: torch.Tensor
    weight: torch.Tensor
    lo: int
    lat_eps: float


def _eps(norm: nn.Module) -> float:
    """The epsilon of a KimiK3RMSNorm or ``torch.nn.RMSNorm`` (``eps``; for torch's None, its default: the machine
    epsilon of the weight's dtype) or a stock RMSNorm (``variance_epsilon``)."""
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    return float(torch.finfo(norm.weight.dtype).eps if eps is None else eps)


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

    def compile_tail(self, latent_size: int, act_size: int, tail_weight: torch.Tensor) -> None:
        """Compile the sandwich tail kernel for a MoE tail of ``latent_size`` latent and ``act_size`` shared columns
        and ``tail_weight``'s shape, with one call on a zero row of a zero weight, before any capture. Collective: every
        rank of the group makes the call; it advances the sandwich workspace on every rank alike."""
        weight = torch.zeros_like(tail_weight)
        hidden = weight.shape[0]
        ones = weight.new_ones(hidden)
        k3_sandwich_tail(
            weight.new_zeros(1, latent_size),
            weight.new_zeros(1, act_size),
            weight,
            0,
            1e-6,
            None,
            weight.new_zeros(0, 1, hidden),
            weight.new_zeros(hidden),
            ones,
            ones,
            1e-6,
            1e-6,
            self.sandwich,
        )
        torch.cuda.synchronize(weight.device)

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

    @staticmethod
    def takes_tail(
        latent: torch.Tensor, act: torch.Tensor, tail_weight: torch.Tensor, max_snapshots: int
    ) -> bool:
        """Whether ``sandwich_tail`` takes a MoE tail of these tensors' shapes (rows of ``latent`` and ``act``, at most
        `SANDWICH_MAX_TOKENS`; ``tail_weight``) with a snapshot bank of at most ``max_snapshots`` rows: the TP16
        per-rank shapes."""
        return _sandwich_op.supports_tail(latent, act, tail_weight, max_snapshots)

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

    def sandwich_tail(
        self,
        pending: PendingTail,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_proj: nn.Module,
        res_norm: nn.Module,
        out_norm: nn.Module,
        updated_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` as ``allreduce_attn_res`` of a MoE layer's row-parallel tail ``pending``
        (``[RMSNorm(latent)[:, lo:lo + 224] | act] @ weight.T``), in one ``comm/k3_sandwich_tail`` call.
        ``updated_out``: a bf16 ``[T, H]`` tensor the call stores ``updated`` into (the consumer's snapshot bank row),
        returned as ``updated``."""
        return k3_sandwich_tail(
            pending.latent,
            pending.act,
            pending.weight,
            pending.lo,
            pending.lat_eps,
            prefix_sum,
            block_residual,
            *_res_args(res_proj, res_norm, out_norm),
            self.sandwich,
            updated_out=updated_out,
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

    def compile_plain(self, weight: torch.Tensor, swiglu: bool = False) -> bool:
        """Compile the plain sandwich (``sandwich_plain``) for a projection of ``weight``'s shape (with ``swiglu``, a
        down projection after the SiLU-and-mul) with one call on a zero row of a zero weight, before any capture;
        False, with nothing launched, where the kernel does not take that shape. Collective: every rank of the group
        makes the call; it advances the sandwich workspace on every rank alike."""
        zero = torch.zeros_like(weight)
        hidden, k_in = zero.shape
        x = zero.new_zeros(1, 2 * k_in if swiglu else k_in)
        residual = zero.new_zeros(1, hidden)
        ones = zero.new_ones(hidden)
        if not _sandwich_op.supports_plain(x, zero, residual, ones, swiglu):
            return False
        k3_sandwich_plain(x, zero, residual, ones, 1e-6, self.sandwich, swiglu=swiglu)
        torch.cuda.synchronize(zero.device)
        return True

    @staticmethod
    def takes_plain(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
        swiglu: bool = False,
    ) -> bool:
        """Whether ``sandwich_plain`` takes the call: at most `SANDWICH_MAX_TOKENS` contiguous bf16 rows ``x`` of a
        row-parallel slice ``weight`` [7168, K] (K a multiple of 128 up to 896; with ``swiglu`` K 896 and ``x`` the
        ``[gate | up]`` rows, 2 K wide), ``residual`` [rows, 7168], and ``norm``'s bf16 [7168] weight."""
        return _sandwich_op.supports_plain(x, weight, residual, norm.weight, swiglu)

    def sandwich_plain(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
        swiglu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` in one ``comm/k3_sandwich_plain`` call: ``updated = residual + allreduce(x @
        weight.T)`` (with ``swiglu``, of ``silu_and_mul(x) @ weight.T``), ``normed = norm(updated)`` for a plain
        RMSNorm ``norm``. Bit for bit, by the kernel's statement, the projection on ``k3_ctm_gemv`` at split 1 (with
        ``swiglu``, ``k3_ctm_gemv_swiglu`` at split 2) followed by ``allreduce_norm``'s one-shot call."""
        return k3_sandwich_plain(
            x.contiguous(),
            weight,
            residual.contiguous(),
            norm.weight,
            _eps(norm),
            self.sandwich,
            swiglu=swiglu,
        )

    def takes_allreduce_norm(self, rows: int, hidden: int) -> bool:
        """Whether ``allreduce_norm`` takes ``rows`` bf16 rows of ``hidden`` columns: the MNNVL workspace holds the
        call at the decode path's one-shot ceiling (`DECODE_AR_ONE_SHOT_MAX_BYTES`, two-shot above it)."""
        if rows <= 0 or hidden <= 0 or hidden % 8:
            return False
        world, buffer_bytes = self.mnnvl.world_size, self.mnnvl.buffer_bytes
        need = required_buffer_bytes(
            rows, hidden, world, torch.bfloat16, DECODE_AR_ONE_SHOT_MAX_BYTES
        )
        two_shot = rows * hidden * world * 2 > DECODE_AR_ONE_SHOT_MAX_BYTES
        return need <= buffer_bytes and not (two_shot and buffer_bytes % 32)

    def allreduce_norm(
        self, partial: torch.Tensor, residual: torch.Tensor, norm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(normed, updated)`` in one ``comm/mnnvl_fusion_allreduce`` call: ``updated = residual +
        allreduce(partial)``, ``normed = norm(updated)`` for a plain RMSNorm ``norm``, sent one-shot up to
        `DECODE_AR_ONE_SHOT_MAX_BYTES`. ``partial`` is this rank's unreduced ``[rows, hidden]`` bf16 output of a
        row-parallel projection, of a shape ``takes_allreduce_norm`` holds for."""
        return mnnvl_fusion_allreduce(
            partial.contiguous(),
            self.mnnvl,
            DECODE_AR_ONE_SHOT_MAX_BYTES,
            residual.contiguous(),
            norm.weight,
            _eps(norm),
        )
