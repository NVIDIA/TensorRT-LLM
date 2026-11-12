# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode attention-residual mix + RMSNorm whose addend is a pushed MNNVL Lamport
all-reduce that the kernel finishes itself (CuTe DSL, ``kernels/k3r_attn_res/k3_attn_res.py``)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_attn_res import k3_attn_res as _kernel
from ..norm.k3_attn_res_norm import attn_res_rmsnorm_reference, mxfp8_rows_reference
from .k3_lamport_finish import lamport_sum_reference


class K3AttnResLamport(OpWrapper):
    """``(updated, output)`` with ``updated = bf16(prefix + addend)`` and ``output`` the RMSNorm of
    the attention-residual mix of ``block`` and ``updated``; with ``mxfp8`` also ``(xq, xs)``, the
    MXFP8 form of ``output``. ``addend`` is the all-reduce of the call every rank pushed on the
    Lamport workspace at ``uc_local`` (this rank's unicast buffer) / ``flags`` (the rotation
    state's address) / ``mc_base`` (multicast base): ``finish_world`` ranks, ``finish_chunk``
    ranks per FP32 partial sum, one-shot or ``finish_two_shot`` transport, this ``rank``.
    ``updated`` given: the kernel stores the updated prefix there (contiguous, ``prefix``'s shape
    and dtype) instead of a new tensor.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The group is the test harness's (2 and 4 GPUs of one node); K3 runs 8 (helix8) and 16 (helix16 ULL).
    # cpp_one_shot: the C++ all-reduce whose sums the call reproduces (finish_chunk follows from it).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=why,
            spec=dict(
                rows=rows,
                candidates=candidates,
                hidden=7168,
                two_shot=two_shot,
                cpp_one_shot=cpp_one_shot,
                add_rn=add_rn,
                early_poll=early_poll,
                entry_trigger=entry_trigger,
                mxfp8=mxfp8,
                into=into,
                early_trigger=False,
                rms_eps=1e-6,
                out_eps=1e-6,
            ),
        )
        for why, rows, candidates, two_shot, cpp_one_shot, add_rn, early_poll, entry_trigger, mxfp8, into in (
            (
                "helix8 A_mid finishing the o_proj all-reduce: 16 rows, N=5, one-shot",
                16,
                5,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ),
            (
                "helix8 b16 mtp5 verify A_mid: 96 rows, N=9, two-shot",
                96,
                9,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            (
                "two-shot transport with the one-shot C++ sums: 24 rows, N=7",
                24,
                7,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ),
            (
                "helix16 ULL o_proj sandwich (early poll, entry trigger): 8 rows, N=6",
                8,
                6,
                False,
                True,
                False,
                True,
                True,
                False,
                False,
            ),
            (
                "helix16 ULL row-parallel MoE tail: add_rn, MXFP8, early poll, 32 rows, N=8",
                32,
                8,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
            ),
            (
                "helix16 ULL tail into the snapshot slot: add_rn, MXFP8, 8 rows, N=4",
                8,
                4,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ),
        )
    ) + (
        Cell(
            why="helix8 A_mid, release after the grid wait: 16 rows, N=5, one-shot",
            spec=dict(
                rows=16,
                candidates=5,
                hidden=7168,
                two_shot=False,
                cpp_one_shot=True,
                add_rn=False,
                early_poll=False,
                entry_trigger=False,
                mxfp8=False,
                into=False,
                early_trigger=True,
                rms_eps=1e-6,
                out_eps=1e-6,
            ),
        ),
    )

    note = """
    Finishes the pushed call as k3_lamport_finish does (same sums, rotation
    advance and re-arm of the previous buffer), so no finish kernel runs before
    it; every rank must make this call at the same position of its sequence on
    the workspace. The producer must trigger only after its own
    griddepcontrol.wait; two-shot needs every rank's CTAs co-resident (grid of
    M * cluster CTAs). The addend has the bits of the finished all-reduce.
    add_rn: round-to-nearest FP32 add (the MoE residual update), else the C++
    kernels' fast-math add. early_poll: finish the call and compute the mix
    before griddepcontrol.wait, store after it (same bits). entry_trigger:
    release the dependent at kernel entry. early_trigger: release it right
    after griddepcontrol.wait. finish_ag (one-shot): the call
    carries all-gather rows, addend = bf16(y + bf16(slot sum)); reached only
    through the column-parallel MoE tail and not in this entry's cells.
    Process-wide compile cache as k3_attn_res_norm (keyed also by world, chunk,
    transport, add_rn, early_poll, finish_ag, early_trigger); compile each key
    before capture.
    Applies where kernels/k3_attn_res.can_use_k3_attn_res holds for the
    one-addend mix (H 7168, 2 <= N <= 12, contiguous BF16).
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        uc_local: int,
        flags: int,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        finish_world: int,
        finish_chunk: int,
        finish_two_shot: bool,
        mc_base: int,
        rank: int,
        add_rn: bool,
        early_poll: bool = False,
        entry_trigger: bool = False,
        finish_ag: bool = False,
        mxfp8: bool = False,
        updated: Optional[torch.Tensor] = None,
        early_trigger: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        if updated is not None:
            return _kernel.k3_attn_res_lamport_add_rmsnorm_into(
                updated,
                mxfp8,
                prefix,
                uc_local,
                flags,
                block,
                res_weight,
                rms_weight,
                out_weight,
                rms_eps,
                out_eps,
                finish_world,
                finish_chunk,
                finish_two_shot,
                mc_base,
                rank,
                add_rn,
                early_poll,
                entry_trigger,
                finish_ag,
                early_trigger=early_trigger,
            )
        op = (
            _kernel.k3_attn_res_lamport_add_rmsnorm_mx
            if mxfp8
            else _kernel.k3_attn_res_lamport_add_rmsnorm
        )
        return op(
            prefix,
            uc_local,
            flags,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            finish_world,
            finish_chunk,
            finish_two_shot,
            mc_base,
            rank,
            add_rn,
            early_poll,
            entry_trigger,
            finish_ag,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        prefix: torch.Tensor,
        inputs_in_rank_order: Sequence[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        finish_world: int,
        finish_chunk: int,
        finish_two_shot: bool,
        mc_base: int,
        rank: int,
        add_rn: bool,
        early_poll: bool = False,
        entry_trigger: bool = False,
        finish_ag: bool = False,
        mxfp8: bool = False,
        updated: Optional[torch.Tensor] = None,
        ag_rows: Optional[torch.Tensor] = None,
        early_trigger: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """The collective exception: every rank's pushed rows in rank order replace the workspace
        addresses ``uc_local`` / ``flags``; ``ag_rows`` is the all-gathered ``y`` with
        ``finish_ag``. The addend is the rank-order sum of ``k3_lamport_finish``'s reference, the
        add is FP32 rounded to BF16 (``add_rn`` and the fast-math add agree away from
        denormals), then ``k3_attn_res_norm``'s mix and norm. ``updated`` is not read; the
        scheduling flags, ``finish_two_shot``, ``mc_base`` and ``rank`` do not change values."""
        addend = lamport_sum_reference(inputs_in_rank_order, finish_chunk)
        if finish_ag:
            addend = (ag_rows.float() + addend.float()).to(torch.bfloat16)
        row = (prefix.float() + addend.float()).to(torch.bfloat16)
        output = attn_res_rmsnorm_reference(
            row, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        if mxfp8:
            return (row, output, *mxfp8_rows_reference(output))
        return row, output

    def is_valid(
        self,
        prefix: torch.Tensor,
        uc_local: int,
        flags: int,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        finish_world: int,
        finish_chunk: int,
        finish_two_shot: bool,
        mc_base: int,
        rank: int,
        add_rn: bool,
        early_poll: bool = False,
        entry_trigger: bool = False,
        finish_ag: bool = False,
        mxfp8: bool = False,
        updated: Optional[torch.Tensor] = None,
        early_trigger: bool = False,
    ) -> None:
        # The launcher does not re-validate: operands outside the kernel's shape are read as
        # dense rows of 7168 and give a plausible wrong mix.
        assert _kernel.can_use_k3_attn_res(
            prefix, (prefix,), block, res_weight, rms_weight, out_weight
        ), "outside can_use_k3_attn_res (H 7168, 2 <= N <= 12, M <= 1024, contiguous BF16)"
        # A rank or chunk that does not match the workspace polls slots nobody writes, or sums in
        # another order than the all-reduce the result must reproduce.
        assert 0 <= rank < finish_world, f"rank {rank} outside a group of {finish_world}"
        assert finish_world % finish_chunk == 0, "finish_chunk must divide finish_world"


k3_attn_res_lamport = K3AttnResLamport()
