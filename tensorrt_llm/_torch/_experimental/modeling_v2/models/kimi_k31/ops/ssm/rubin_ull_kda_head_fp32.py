# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3.1-owned fused KDA head of one request's verify rows at FP32 state:
``PreparedKdaHeadFp32.run_verify`` (``kernels/ssm/rubin_ull_kda_head_fp32.py``, MR8's
``KdaHeadVerify`` whose two F_A waits trap past ``kernels/common/rubin_ull_spin_trap.py``'s bound
instead of hanging)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference
from .rubin_ull_kda_head_verify import kda_head_verify as _mr8_head

TOKENS = 8
HEADS = 6
HIDDEN = 7168


def _cell(why: str, accepted: int, polled: bool = False) -> Cell:
    return Cell(why=why, spec=dict(accepted=accepted, polled=polled, tokens=TOKENS, heads=HEADS))


class KdaHeadVerifyFp32(OpWrapper):
    """The KDA core of one request's q verify rows straight from the layer input, in place, on
    the K3.1-owned head (FP32 state, bounded F_A waits).

    ``head`` is ``PreparedKdaHeadFp32`` over the layer's packed BF16 [pitch, hidden] weight, its
    ``PreparedKda`` core (FP32 state) and (q, q) as its only ``verify`` shape; ``x`` BF16
    [q, hidden] contiguous; ``out`` BF16 [q, H * 128] (with ``rearm``: the layer's polled buffer,
    written with canonical BF16 stores); ``caches``, ``slots`` [1, q] and ``accepted`` [>= 1] as
    ``kda_verify``; ``rearm``: the previous fused-head layer's polled buffer, set back to its
    sentinel (``POLLED_OUT``), or None. Forks the head's side stream (the caller joins it once
    per step). Writes ``out``, the state rows, the conv page and (with ``rearm``) the re-armed
    buffer; returns ``out``. Outputs and checkpoints are MR8's ``kda_head_verify``'s bit for bit.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 x q8 head, every draft rejected", 1),
        _cell("b1 x q8 head, three drafts accepted", 4),
        _cell("b1 x q8 head, every draft accepted", 8),
        _cell("b1 x q8 head into a polled buffer, re-arming the previous layer's", 6, True),
    )

    note = """
    The K3.1 ULL target's fused KDA head (``FlatKda._prepare_head``), derived from MR8 df90f706c5
    ``cutedsl/kda/head.py`` / ``head_verify.py`` (``kernels/SOURCE.json``). The change is the
    time bound of the two F_A waits (``poll_fa`` site 211, ``rearm`` site 212): past
    ``spin_trap.BOUND_NS`` (about 69 s) the waiting thread writes the trap record into mapped
    pinned host memory and traps, so a producer that never publishes ends the process with a
    CUDA error and one ``K31_POLL_TIMEOUT`` line instead of hanging it. Every other instruction
    is MR8's. The launches differ: ``SignalGemm`` (F_A / beta, side stream) records a programmatic
    event that fires once all of its CTAs have started, and the main stream waits on it before the
    head, so the head is a programmatic dependent of both the AttnRes before it and
    ``SignalGemm``. A head cluster spinning on F_A can then never hold the SMs a not-yet-placed
    ``SignalGemm`` CTA needs (both take a whole SM's shared memory). One request x q verify rows
    only (no q = 1 batches), FP32 state only.
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
        and ``rearm`` are not read (the re-armed buffer is all ``OUT_SENT``)."""
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
        """``kda_head_verify``'s checks, and an FP32 state (the only one this head compiles)."""
        if caches is not None and caches[1].dtype != torch.float32:
            raise ValueError("kda_head_verify_fp32: the KDA state must be FP32")
        _mr8_head.is_valid(head, x, out, caches, slots, accepted, rearm)


kda_head_verify_fp32 = KdaHeadVerifyFp32()
