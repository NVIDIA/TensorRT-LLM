# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3.1-owned fused KDA head of two requests' q8 verify rows (16 rows) at FP32 state:
``PreparedKdaHeadMultiFp32.run_verify`` (``kernels/ssm/rubin_ull_kda_head_multi_fp32.py``, MR8's
``KdaHeadMulti`` whose two F_A waits trap past ``kernels/common/rubin_ull_spin_trap.py``'s bound
instead of hanging)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference

REQUESTS = 2
TOKENS = 8
HEADS = 6
HIDDEN = 7168


def _cell(why: str, accepted: tuple[int, int], padded: tuple[int, ...] = ()) -> Cell:
    return Cell(
        why=why,
        spec=dict(accepted=accepted, padded=padded, requests=REQUESTS, tokens=TOKENS, heads=HEADS),
    )


class KdaHeadMultiFp32(OpWrapper):
    """The KDA core of two requests' q verify rows straight from the layer input, in place, on
    the K3.1-owned multi-request head (FP32 state, bounded F_A waits).

    ``head`` is ``PreparedKdaHeadMultiFp32`` over the layer's packed BF16 [pitch, hidden]
    weight, its ``PreparedKda`` core (FP32 state) and (2 q, q) as its only ``verify`` shape;
    ``x`` BF16 [2 q, hidden] contiguous, request-major; ``out`` BF16 [2 q, H * 128];
    ``caches``, ``slots`` [2, q] and ``accepted`` [>= 2] as ``kda_verify``; ``rearm`` must be
    None (no polled output). Forks the head's side stream (the caller joins it once per step).
    Writes ``out``, the state rows and the conv pages; returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("2 x q8 head, every draft rejected / every draft accepted", (1, 8)),
        _cell("2 x q8 head, three and six drafts accepted", (4, 7)),
        _cell("2 x q8 head, every draft accepted in both", (8, 8)),
        _cell("2 x q8 head, the second request padded", (5, 1), (1,)),
    )

    note = """
    The 16-row verify step of the K3.1 ULL target's KDA layers when
    it is prepared by ``FlatKda._prepare_multi``, derived from
    MR8 df90f706c5 ``cutedsl/kda/head_multi.py`` (``kernels/SOURCE.json``). The change is
    the time bound of the two F_A waits (``KdaHeadVerifyFp32``'s ``poll_fa`` site 211 and
    ``rearm`` site 212). Against the 16-row split pair the projected inputs are summed in
    another FP32 order (Q / K / V / gate: four split-K partials instead of
    ``SplitKTmemGemm64``'s six; F_A / beta: ``SignalGemm``'s split 8), each rounded once to
    BF16; given the same projected inputs the recurrence, outputs and checkpoints are
    ``KdaSplit``'s per request. The F_A buffer and the [ready, done] counter are the head's own
    and re-armed by their last reader, so a head runs at most once per step. Compiled before
    graph capture (``PreparedKdaHeadMultiFp32.prepare_caches``).
    """

    def __call__(
        self,
        head,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        rearm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return head.run_verify(x, out, caches=caches, slots=slots, accepted=accepted, rearm=rearm)

    def reference(
        self,
        head,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        rearm: torch.Tensor | None = None,
    ):
        """``(out, states, pages)`` in FP64: the packed rows ``bf16(x @ W^T)`` (FP32 then BF16,
        as the kernels round) through ``_rubin_ull_kda_reference.kda_core_reference``. ``out``
        is not read."""
        projected = (x.double() @ head.weight.double().T).float().bfloat16()
        return kda_core_reference(head.core, projected, caches, slots, accepted, slots.shape[1])

    def is_valid(
        self,
        head,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
        rearm: torch.Tensor | None = None,
    ) -> None:
        """Raise where the kernel would misplace a read or write without an error: a state other
        than FP32, rows other than two requests' q8, a polled buffer, an accepted count outside
        1..q, an active request without a conv page or reusing a slot, non-contiguous input rows
        or a mis-shaped output."""
        if caches is not None and caches[1].dtype != torch.float32:
            raise ValueError("kda_head_multi_fp32: the KDA state must be FP32")
        if rearm is not None:
            raise ValueError("kda_head_multi_fp32: the 16-row head has no polled output")
        rows, dim = REQUESTS * TOKENS, head.core.dim
        if slots.dim() != 2 or slots.shape[0] < REQUESTS or slots.shape[1] != TOKENS:
            raise ValueError("kda_head_multi_fp32: slots must be [2, q8]")
        if x.shape != (rows, head.weight.shape[1]) or x.dtype != torch.bfloat16:
            raise ValueError("kda_head_multi_fp32: x must be two requests' BF16 [2 q, hidden] rows")
        if not x.is_contiguous():
            raise ValueError("kda_head_multi_fp32: x must be contiguous")
        if out.dtype != torch.bfloat16 or out.shape != (rows, dim) or out.stride(1) != 1:
            raise ValueError("kda_head_multi_fp32: out must be BF16 [2 q, dim]")
        if accepted.numel() < REQUESTS:
            raise ValueError("kda_head_multi_fp32: accepted needs one count per request")
        for request in range(REQUESTS):
            count, row = int(accepted[request]), slots[request].tolist()
            if not 1 <= count <= TOKENS:
                raise ValueError(f"kda_head_multi_fp32: accepted {count} outside 1..q")
            if row[count - 1] > 0:
                if row[0] <= 0:
                    raise ValueError("kda_head_multi_fp32: an active request needs its conv page")
                active = [s for s in row if s > 0]
                if len(set(active)) != len(active):
                    raise ValueError("kda_head_multi_fp32: a request's positions share a slot")


kda_head_multi_fp32 = KdaHeadMultiFp32()
