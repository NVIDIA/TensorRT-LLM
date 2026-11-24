# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused KDA head of one request's verify rows: MR8's ``PreparedKdaHead.run_verify``
(``KdaHeadVerify``: the Q / K / V / gate in-projection and the q8 recurrence in one kernel, with
F_A / beta projected by ``SignalGemm`` on the layer's side stream)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference

TOKENS = 8
HEADS = 6
HIDDEN = 7168


def _cell(why: str, accepted: int, polled: bool = False) -> Cell:
    return Cell(why=why, spec=dict(accepted=accepted, polled=polled, tokens=TOKENS, heads=HEADS))


class KdaHeadVerify(OpWrapper):
    """The KDA core of one request's q verify rows straight from the layer input, in place.

    ``head`` is MR8's ``PreparedKdaHead`` over the layer's packed BF16 [pitch, hidden] weight,
    its ``PreparedKda`` core and (q, q) in its ``verify`` shapes; ``x`` BF16 [q, hidden]
    contiguous; ``out`` BF16 [q, H * 128] (with ``rearm``: the layer's polled buffer, written
    with canonical BF16 stores); ``caches``, ``slots`` [1, q] and ``accepted`` [>= 1] as
    ``kda_verify``; ``rearm``: the previous fused-head layer's polled buffer, set back to its
    sentinel (``POLLED_OUT``), or None. Forks the head's side stream (the caller joins it
    once per step). Writes ``out``, the state rows, the conv page and (with ``rearm``) the
    re-armed buffer; returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 x q8 head, every draft rejected", 1),
        _cell("b1 x q8 head, three drafts accepted", 4),
        _cell("b1 x q8 head, every draft accepted", 8),
        _cell("b1 x q8 head into a polled buffer, re-arming the previous layer's", 6, True),
    )

    note = """
    The 8-row verify step of the K3.1 ULL target's KDA layers on Rubin (BF16 in-projection
    weights; the W8A16 codes keep the split path). The head's BF16 Q / K / V / gate tiles are
    the split-K projection's bit for bit; F_A / beta come from a split-K 8 projection (another
    FP32 summation order than ``ops.gemm.prepared_projection``). The F_A buffer and the [ready, done] counter
    are the head's own and re-armed by their last reader, so a head runs at most once per
    step. ``POLLED_OUT`` is a Rubin (10.7) policy; GB300 drives the polled kernel directly.
    Compiled before graph capture (``PreparedKdaHead.prepare_caches``).
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
        as the kernels round) through ``_kda_reference.kda_core_reference``. ``out`` and
        ``rearm`` are not read (the re-armed buffer is all ``OUT_SENT``)."""
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
        """Raise where the kernel would misplace a read or write without an error: rows other
        than one request's q, an accepted count outside 1..q, an active request without a conv
        page or reusing a slot, non-contiguous input rows or a mis-shaped output."""
        tokens, dim = slots.shape[1], head.core.dim
        if x.shape != (tokens, head.weight.shape[1]) or x.dtype != torch.bfloat16:
            raise ValueError("kda_head_verify: x must be one request's BF16 [q, hidden] rows")
        if not x.is_contiguous():
            raise ValueError("kda_head_verify: x must be contiguous")
        if out.dtype != torch.bfloat16 or out.shape != (tokens, dim) or out.stride(1) != 1:
            raise ValueError("kda_head_verify: out must be BF16 [q, dim]")
        if rearm is not None and rearm.shape != out.shape:
            raise ValueError("kda_head_verify: the re-armed buffer must be [q, dim]")
        count, row = int(accepted[0]), slots[0].tolist()
        if not 1 <= count <= tokens:
            raise ValueError(f"kda_head_verify: accepted {count} outside 1..q")
        if row[count - 1] > 0:
            if row[0] <= 0:
                raise ValueError("kda_head_verify: an active request needs its conv page")
            active = [s for s in row if s > 0]
            if len(set(active)) != len(active):
                raise ValueError("kda_head_verify: the request's positions share a slot")


kda_head_verify = KdaHeadVerify()
