# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 fused latent-MoE tail at attention TP: routed all-reduce + RMSNorm, shared
reduce-scatter, sharded up-projection with a multicast all-gather, through the family kernels."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

# The tail instance the entries take (``KimiK3LatentMoETail.get``, built outside CUDA-graph capture) and the row
# range of the two-shot-only instance, re-exported from here.
from ...kernels.moe.moe_tail_latent import (  # noqa: F401
    KimiK3LatentMoETail,
    fused_tail_twoshot_only_rows,
)
from ...kernels.moe.moe_tail_routed_output import UnfinalizedRoutedOutput

#: Kimi K3.1's hidden size, routed latent width and RMSNorm epsilon.
HIDDEN = 7168
LATENT = 3584
EPS = 1e-5


def _cell(world: int, rows: int, why: str, **extra) -> Cell:
    """``fold``: the routed partial is the routed experts' ``UnfinalizedRoutedOutput``; ``shared_pushed``:
    :data:`k31_latent_moe_tail_push_shared` stored the shared rows ahead."""
    return Cell(why=why, spec=dict(world=world, rows=rows, eps=EPS, **extra))


def _finalized(routed_partial: torch.Tensor | UnfinalizedRoutedOutput) -> torch.Tensor:
    """A rank's routed partial ``[M, 3584]``: an ``UnfinalizedRoutedOutput`` as its top-k weighted sum over the slots
    this rank owns (FP32 in slot order, rounded once to BF16: the finalize the tail folds), a tensor as it is."""
    if not isinstance(routed_partial, UnfinalizedRoutedOutput):
        return routed_partial
    rows, weights, slots = routed_partial
    out = torch.zeros(slots.shape[0], rows.shape[1], dtype=torch.float32, device=rows.device)
    for slot in range(slots.shape[1]):
        owned = (slots[:, slot] >= 0).unsqueeze(-1)
        term = (
            weights[:, slot].float().unsqueeze(-1)
            * rows[slots[:, slot].clamp_min(0).long()].float()
        )
        out += torch.where(owned, term, torch.zeros_like(term))
    return out.to(torch.bfloat16)


class K31LatentMoeTail(OpWrapper):
    """The MoE output ``[M, 7168]`` BF16 on every rank of the attention-TP group, from this rank's
    unreduced partials.

    ``routed_partial [M, 3584]``: this rank's routed-expert output (its experts' share of the
    finalized top-k sum, or its MoE-TP slice of it); ``shared_partial [M, 7168]``: this rank's
    shared-expert down projection before its row-parallel all-reduce; ``rms_weight [3584]``: the
    routed latent norm (``routed_expert_norm``); ``up_weight [7168, 3584]``: the replicated latent
    up projection (``routed_expert_up_proj.weight``); ``tail``: the process-wide
    ``KimiK3LatentMoETail`` of the group (``KimiK3LatentMoETail.get(mapping, hidden_size=7168,
    latent_size=3584, rms_eps=eps)``, as the target builds it, outside CUDA-graph capture).

    It computes ``bf16(up(rmsnorm(sum_r routed_r)) + sum_r shared_r)`` -- what the target's
    routed all-reduce with the RMSNorm fusion, the replicated up projection, the shared expert's
    all-reduce and the add compute -- in three PDL kernels: a one-shot (or, above its row limit,
    two-shot) multicast all-reduce of the routed partials with the RMSNorm and a reduce-scatter of
    the shared partials; this rank's ``7168 / tp`` rows of the up projection (``1 / tp`` of the
    weight bytes) with its shared shard added in the epilogue and stored through the multicast
    mapping of a mailbox (the all-gather); a copy out of the mailbox that re-arms it.

    ``routed_partial`` may instead be the routed experts' output before their top-k weighted sum
    (``UnfinalizedRoutedOutput``: FC2 rows in permuted order ``[rows, 3584]`` BF16, routing weights ``[M, 16]``
    BF16, the (token, slot) -> row map ``[M, 16]`` int32, negative for slots another rank owns); the collective
    forms the sum while it loads the row. ``shared_pushed``: :data:`k31_latent_moe_tail_push_shared`
    already stored this call's shared rows.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        _cell(4, 1, "one decode row (skinny SIMT up projection)"),
        _cell(4, 6, "TEP16 low-latency verify, one request x 6 rows"),
        _cell(4, 8, "TEP32 B1 verify (8 rows)"),
        _cell(4, 24, "TEP16 B4 verify (24 rows), past the TP16 one-shot limit"),
        _cell(4, 32, "TP32 capacity (one-shot only there)"),
        _cell(4, 72, "TEP16 B12 verify (72 rows), two-shot collective"),
        _cell(4, 128, "TP4 / TP8 dispatch limit"),
        # The attention-TP target hands over the routed experts' output before the top-k sum wherever the CuTe
        # DSL experts run (the front's runner sort, or the MoE prologue's routing above the front's rows).
        _cell(4, 1, "folded top-k sum: one decode row (skinny up projection)", fold=True),
        _cell(4, 8, "folded top-k sum: TEP32 B1 verify at the MoE front's rows", fold=True),
        _cell(4, 24, "folded top-k sum: TEP16 B4 verify (24 rows, one-shot at TP 4)", fold=True),
        _cell(
            4, 72, "folded top-k sum: TEP16 B12 verify (72 rows), two-shot collective", fold=True
        ),
        _cell(4, 128, "folded top-k sum: TP4 / TP8 dispatch limit (128 rows)", fold=True),
        # Under CUDA-graph capture the shared expert's stream pushes the shared rows of two-shot calls ahead.
        _cell(
            4,
            40,
            "shared rows pushed ahead: two-shot call of 40 rows (B8 x 5)",
            fold=True,
            shared_pushed=True,
        ),
        _cell(
            4,
            128,
            "shared rows pushed ahead: TP4 / TP8 dispatch limit (128 rows)",
            fold=True,
            shared_pushed=True,
        ),
    )
    note = """
    Built for hidden 7168 / latent 3584 and TP 4, 8, 16 or 32 (16 and 32 are the generation
    servers' groups; 4 is one GB300 tray, where the cells are driven). Rows per call: up to 192
    at TP 16 (one-shot collective up to 16 rows, two-shot above), 32 at TP 32 (one-shot only),
    128 at TP 4 / 8 (one-shot up to 32); at TP 32 a second instance (`twoshot_only=True`) takes
    33..192 rows with the two-shot collective, bitwise to the target's unfused path at those rows;
    `tail.can_run(rows)` says whether a row count is served.
    Attention TP only (every rank holds the same rows). One call at a time per group: the
    workspace is a set of Lamport buffers in one MNNVL multicast allocation, rotated per call, so
    every rank of the group must make the same sequence of calls (as the model's layers do).

    Numerics: the routed and shared sums are formed in rank order in FP32 and rounded once to BF16
    (as the MNNVL one-shot all-reduce); the RMSNorm squares the BF16 sum and rounds the weighted
    row once (the target's MNNVL RMSNorm fusion also rounds the normalized row before the weight
    product); the up projection accumulates in FP32 split over K and rounds to BF16 before the
    BF16 shared shard is added in FP32 and rounded. Not bitwise against the target's path (latent
    norm rounding, up-projection accumulation order): within one BF16 step per rounding. The same
    bits on every rank.
    """

    def __call__(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        tail,
        shared_pushed: bool = False,
    ) -> torch.Tensor:
        return tail(
            routed_partial, shared_partial, rms_weight, up_weight, shared_pushed=shared_pushed
        )

    def reference(
        self,
        routed_partials_in_rank_order: Sequence[torch.Tensor | UnfinalizedRoutedOutput],
        shared_partials_in_rank_order: Sequence[torch.Tensor],
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """What every rank should get back (collective: every rank's partials, in rank order).

        The routed sum rounded to BF16, its RMSNorm times the weight rounded once to BF16; the up
        projection in FP32 rounded to BF16; the shared sum rounded to BF16; their sum rounded to
        BF16. (The target's MNNVL RMSNorm fusion rounds the normalized row before the weight
        product as well, so its latent is one BF16 step from this one in about a quarter of the
        elements.) An ``UnfinalizedRoutedOutput`` partial counts as its top-k weighted sum, rounded
        once to BF16; ``shared_pushed`` changes nothing."""
        routed_partials = [_finalized(x) for x in routed_partials_in_rank_order]
        routed = torch.zeros_like(routed_partials[0], dtype=torch.float32)
        for x in routed_partials:
            routed += x.float()
        routed = routed.to(torch.bfloat16).float()
        rsigma = torch.rsqrt(routed.pow(2).mean(-1, keepdim=True) + eps)
        latent = (routed * rsigma * rms_weight.float()).to(torch.bfloat16)
        with true_fp32_matmul():
            up = (latent.float() @ up_weight.float().t()).to(torch.bfloat16)
        shared = torch.zeros_like(shared_partials_in_rank_order[0], dtype=torch.float32)
        for x in shared_partials_in_rank_order:
            shared += x.float()
        shared = shared.to(torch.bfloat16)
        return (up.float() + shared.float()).to(torch.bfloat16)

    def is_valid(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        tail,
        shared_pushed: bool = False,
    ) -> None:
        if isinstance(routed_partial, UnfinalizedRoutedOutput):
            rows, weights, slots = routed_partial
            m = slots.shape[0]
            assert rows.dim() == 2 and rows.shape[1] == LATENT, "gemm2_permuted [rows, 3584]"
            assert rows.is_contiguous(), "gemm2_permuted must be contiguous"
            # The collective reads the weights and the slot map as dense [M, top_k] rows.
            assert slots.dim() == 2 and slots.dtype == torch.int32 and slots.is_contiguous(), (
                "expanded_idx_to_permuted_idx must be contiguous int32 [M, top_k]"
            )
            assert tuple(weights.shape) == tuple(slots.shape) and weights.is_contiguous(), (
                "expert_weights must be contiguous [M, top_k]"
            )
            routed_tensors = (rows, weights)
        else:
            m = routed_partial.shape[0]
            assert routed_partial.dim() == 2 and routed_partial.shape[1] == LATENT, (
                "routed [M, 3584]"
            )
            routed_tensors = (routed_partial,)
        assert tuple(shared_partial.shape) == (m, HIDDEN), "shared [M, 7168]"
        assert tuple(up_weight.shape) == (HIDDEN, LATENT) and up_weight.is_contiguous(), (
            "up_weight must be the contiguous [7168, 3584] nn.Linear weight"
        )
        assert rms_weight.numel() == LATENT, "rms_weight has 3584 elements"
        for t in (*routed_tensors, shared_partial, rms_weight, up_weight):
            assert t.dtype == torch.bfloat16 and t.is_cuda, "BF16 CUDA tensors"
        assert tail.can_run(m) or 1 <= m <= tail.max_num_tokens, (
            f"rows must be in [1, {tail.max_num_tokens}]"
        )


k31_latent_moe_tail = K31LatentMoeTail()


class K31LatentMoeTailToMailbox(OpWrapper):
    """:data:`k31_latent_moe_tail` without its copy kernel: the tail's mailbox ``[1, capacity, 7168]`` BF16, whose
    first ``M`` rows receive the MoE output on every rank of the group.

    The same arguments and kernels as :data:`k31_latent_moe_tail` up to the up projection, whose multicast store fills
    the mailbox, so those rows are that entry's output bit for bit. They must be consumed exactly once before the
    group's next tail call: :data:`k31_latent_moe_tail_copy_out`, or a kernel that polls them and re-arms the mailbox
    as the copy kernel does (``k31_attn_res_mailbox_add_rmsnorm``).
    """

    ARCHS = frozenset({Arch.SM_103})
    # The attention-TP target hands a decoder layer's MoE output in the mailbox to the next layer's pre-attention
    # mix at the PDL mix's rows (1..191).
    CELLS: tuple[Cell, ...] = (
        _cell(4, 1, "mailbox for the next layer's mix: one decode row, folded", fold=True),
        _cell(4, 8, "mailbox: TEP32 B1 verify (8 rows), folded", fold=True),
        _cell(4, 24, "mailbox: TEP16 B4 verify (24 rows), finalized routed rows"),
        _cell(
            4,
            72,
            "mailbox: TEP16 B12 verify (72 rows), shared rows pushed ahead",
            fold=True,
            shared_pushed=True,
        ),
        _cell(4, 128, "mailbox: TP4 / TP8 dispatch limit (128 rows), folded", fold=True),
    )
    note = K31LatentMoeTail.note

    def __call__(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        tail,
        shared_pushed: bool = False,
    ) -> torch.Tensor:
        return tail.forward_to_mailbox(
            routed_partial, shared_partial, rms_weight, up_weight, shared_pushed=shared_pushed
        )

    def reference(
        self,
        routed_partials_in_rank_order: Sequence[torch.Tensor | UnfinalizedRoutedOutput],
        shared_partials_in_rank_order: Sequence[torch.Tensor],
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """The ``[M, 7168]`` rows the mailbox receives (collective: :meth:`K31LatentMoeTail.reference`)."""
        return k31_latent_moe_tail.reference(
            routed_partials_in_rank_order, shared_partials_in_rank_order, rms_weight, up_weight, eps
        )

    def is_valid(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        tail,
        shared_pushed: bool = False,
    ) -> None:
        k31_latent_moe_tail.is_valid(
            routed_partial, shared_partial, rms_weight, up_weight, tail, shared_pushed
        )


k31_latent_moe_tail_to_mailbox = K31LatentMoeTailToMailbox()


class K31LatentMoeTailCopyOut(OpWrapper):
    """The ``[num_tokens, 7168]`` BF16 MoE output a :data:`k31_latent_moe_tail_to_mailbox` call left in ``mailbox``,
    through the tail's copy kernel: it polls each row until every rank's shard has arrived, copies it out and
    re-arms it (the Lamport sentinel, ``-0.0``)."""

    ARCHS = frozenset({Arch.SM_103})
    # The calls that consume the mailbox entry's output.
    CELLS: tuple[Cell, ...] = K31LatentMoeTailToMailbox.CELLS
    note = """
    One call per mailbox call, with its row count, on the stream order of the group's tail calls; the target
    settles a pending MoE output this way where the next layer's mix does not take the mailbox.
    """

    def __call__(self, mailbox: torch.Tensor, num_tokens: int, tail) -> torch.Tensor:
        return tail.copy_out(mailbox, num_tokens)

    def reference(self, mailbox: torch.Tensor, num_tokens: int, tail) -> torch.Tensor:
        """The mailbox's first ``num_tokens`` rows once every rank's shard has arrived (``tail`` is ignored)."""
        return mailbox[0, :num_tokens].clone()

    def is_valid(self, mailbox: torch.Tensor, num_tokens: int, tail) -> None:
        assert mailbox.dim() == 3 and mailbox.shape[0] == 1 and mailbox.shape[2] == HIDDEN, (
            "mailbox [1, capacity, 7168]"
        )
        assert tail.can_run(num_tokens) or 1 <= num_tokens <= tail.max_num_tokens, (
            f"rows must be in [1, {tail.max_num_tokens}]"
        )


k31_latent_moe_tail_copy_out = K31LatentMoeTailCopyOut()


class K31LatentMoeTailPushShared(OpWrapper):
    """Store this rank's ``shared_partial [M, 7168]`` BF16 into the two-shot collective's slots of the group's next
    tail call, ahead of that call, on the stream that produced the rows (the shared expert's) while the routed
    experts run. The call must pass ``shared_pushed=True``. Returns None.

    The same stores and sums as the call's own push, so that call's output has the bits of a call that pushes its
    shared rows itself."""

    ARCHS = frozenset({Arch.SM_103})
    # The calls whose shared rows the target pushes ahead.
    CELLS: tuple[Cell, ...] = tuple(
        c
        for c in (*K31LatentMoeTail.CELLS, *K31LatentMoeTailToMailbox.CELLS)
        if c.spec.get("shared_pushed")
    )
    note = """
    Only for the calls of `tail.pushes_shared_early(M)` rows (the two-shot collective's); the target pushes under
    CUDA-graph capture only, where the push overlaps the routed experts.
    """

    def __call__(self, shared_partial: torch.Tensor, tail) -> None:
        tail.push_shared(shared_partial)

    def reference(self, shared_partial: torch.Tensor, tail) -> None:
        """Nothing returned: the stored rows show in the next call's output, which
        :meth:`K31LatentMoeTail.reference` states."""
        return None

    def is_valid(self, shared_partial: torch.Tensor, tail) -> None:
        assert shared_partial.dim() == 2 and shared_partial.shape[1] == HIDDEN, "shared [M, 7168]"
        assert shared_partial.dtype == torch.bfloat16 and shared_partial.is_cuda, "BF16 CUDA tensor"
        # A one-shot call pushes its shared rows itself and would never read these.
        assert tail.pushes_shared_early(shared_partial.shape[0]), (
            "only the two-shot collective's calls take their shared rows ahead"
        )


k31_latent_moe_tail_push_shared = K31LatentMoeTailPushShared()
