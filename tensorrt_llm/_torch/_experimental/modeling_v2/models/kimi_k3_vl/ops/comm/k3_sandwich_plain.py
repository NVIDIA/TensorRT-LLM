# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A row-parallel projection, its TP all-reduce, the residual add and an RMSNorm in one kernel (Kimi K3's drafter
layers: the attention output projection, or with ``swiglu`` the MLP's SiLU-and-mul and down projection), over a
caller-owned :class:`K3SandwichWorkspace`."""

from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Importing the op module registers trtllm::k3_sandwich_*; the state type is the one the ops' buffers come from.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_sandwich.op import (
    K3SandwichWorkspace,
)

from .k3_sandwich_oproj import _WORKSPACE_NOTE, _reduce_reference

__all__ = ["K3SandwichWorkspace", "k3_sandwich_plain"]


class K3SandwichPlain(OpWrapper):
    """Return ``(normed, updated)``: ``updated = residual + allreduce(x @ weight^T)`` over ``workspace``'s TP group
    (with ``swiglu``, of ``silu_and_mul(x) @ weight^T``), ``normed = RMSNorm(updated) * norm_weight``. Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{tokens} decode tokens, swiglu={swiglu}, draw {draw}: drafter "
            f"{'down projection with gate-first SiLU-and-mul' if swiglu else 'attention projection'}",
            spec=dict(
                tokens=tokens,
                swiglu=swiglu,
                draw=draw,
                weights=(tokens + draw) % 3,
                hidden=7168,
                k=896 if swiglu else 384,
                dtype=torch.bfloat16,
                eps=1e-6,
                worlds=(4, 16),
            ),
        )
        for tokens in range(1, 9)
        for swiglu in (False, True)
        for draw in (0, 1)
    )

    note = (
        _WORKSPACE_NOTE
        + """
    reference takes x and weight sequences in TP-rank order and the
    replicated residual and norm_weight. SwiGLU has gates in the first
    half: fp32 silu(gate)*up rounded once to bf16, then projection in true
    fp32 rounded once per rank. The rank sum and residual addition each
    round to bf16. The reference RMSNorm accumulates in fp32 and rounds
    once. The kernel follows MNNVL RESIDUAL_RMS_NORM's bf16-rounded
    squares and reduction tree, with normed rounded once. SwiGLU follows
    k3_ctm_gemv_swiglu's reciprocal sigmoid and split-2 even/odd k-tile
    accumulators, (0+even)+odd; plain follows k3_ctm_gemv split 1.
    Kernel tests compare each projection plus MNNVL bit for bit.

    Inputs are read-only; both outputs are new contiguous bf16 [M,7168].
    Producing x, chaining residual and consuming normed remain outside.
    The drafter reuses the target's workspace, never creates its own.
    M=1..8 and both forms are covered with two draws each. Non-SwiGLU
    supports other K multiples of 128 through 896, but only TP16's
    K=384 is certified; SwiGLU uses K=896 and x width 1792. Unsupported
    shapes, dtypes, x/weight/residual layouts and M fail loudly, including
    SwiGLU on K=384 and mismatched x/weight widths, so is_valid inherits
    the base no-op. Distributed residual/form agreement is caller-owned.

    The process compile cache is result-neutral, keyed by
    (W, K, order, swiglu, PDL); each form needs an eager first call before
    graph capture, otherwise RuntimeError. TRTLLM_ENABLE_PDL is read
    every call, default on; it changes scheduling, not results.
    ipc_order=False is fixed: the alternative IPC one-shot uses fp32
    squares and another reduction tree, requires TP<=8 and is not certified.

    Exact power-of-two x/weights and gates 0,32,64 make SiLU-and-mul and
    partial products exact in fp32; updated is checked bit for bit.
    General-input SiLU rounding belongs to the kernel test, not this
    matrix. Normed is compared within 2e-2 of the reference's largest
    magnitude; all outputs are bitwise equal across ranks. Sequences
    alternate both forms across six drafter layers, twelve chained calls;
    the M=8 graph is replayed eight times between eager calls of other M.
    """
    )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        workspace: K3SandwichWorkspace,
        swiglu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed, updated = torch.ops.trtllm.k3_sandwich_plain(
            x,
            weight,
            residual,
            norm_weight,
            eps,
            workspace.uc,
            workspace.mc,
            workspace.flags,
            workspace.rank,
            swiglu=swiglu,
        )
        return normed, updated

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        swiglu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        partials = []
        with true_fp32_matmul():
            for x, weight in zip(inputs_in_rank_order, weights_in_rank_order):
                if swiglu:
                    gate, up = x.float().chunk(2, dim=-1)
                    x = (torch.nn.functional.silu(gate) * up).bfloat16()
                partials.append((x.float() @ weight.float().t()).bfloat16())
        updated = _reduce_reference(partials, residual)
        u = updated.float()
        normed = (
            u * (u.square().mean(dim=-1, keepdim=True) + eps).rsqrt() * norm_weight.float()
        ).bfloat16()
        return normed, updated


k3_sandwich_plain = K3SandwichPlain()
