# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA verify core: MR8's ``PreparedKda.run`` over a step's ``requests x q`` packed rows
(``KdaSplit`` at q > 1 up to 16 requests; the schedule's wide variants start above that)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference

TOKENS = 8  # DFlash2 DL7: the target token and seven drafts per request
HEADS = 6  # 96 KDA heads over TP16


def _cell(why: str, accepted: tuple[int, ...], padded: tuple[int, ...] = ()) -> Cell:
    """``accepted[b]``: request ``b``'s accepted count (target token included, 1..8);
    ``padded``: requests whose slots are all 0 (a CUDA-graph padding request)."""
    return Cell(why=why, spec=dict(accepted=accepted, padded=padded, tokens=TOKENS, heads=HEADS))


class KdaVerify(OpWrapper):
    """The KDA core of ``requests x q`` verify rows, in place.

    ``core`` is MR8's ``PreparedKda`` prepared for ``(requests, q)`` over ``caches``;
    ``projected`` the BF16 [requests * q, pitch] packed in-projection rows
    ``[q | k | v | g | f_a | beta | pad]``; ``out`` BF16 [requests * q, H * 128];
    ``caches`` (conv, state): the BF16 [rows, 3 H * 128, q + 2] conv pages and the FP32
    [rows, H, 128, 128] recurrent checkpoints; ``slots`` int32 [requests, q] (row
    ``slots[b, t]`` receives the state after token ``t``; ``slots[b, 0]`` is the request's
    conv page); ``accepted`` int32 [>= requests] (the state is read from
    ``slots[b, accepted[b] - 1]``). Writes ``out``, the state rows of ``slots`` and the conv
    pages; returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 x q8 (8 rows), every draft rejected", (1,)),
        _cell("b1 x q8, every draft accepted: the state from the last checkpoint", (8,)),
        _cell("b2 x q8 (16 rows)", (3, 6)),
        _cell("b3 x q8 (24 rows)", (2, 5, 8)),
        _cell("b4 x q8 (32 rows)", (1, 4, 7, 8)),
        _cell(
            "b4 x q8 with a padding request (slots 0): zero rows, nothing written",
            (2, 1, 5, 3),
            (1,),
        ),
    )

    note = """
    The split path of the K3.1 ULL target's KDA layers: the verify rows the fused head does
    not take (16, 24, 32; 8 without the head). vLLM MR8 spec-cache semantics (checkpoint per
    position, conv page rewritten at ``slots[b, 0]``); a request whose read slot is <= 0 is
    padding: zero output rows, no state or conv write. Active requests must own distinct
    slots. The kernel is compiled for ``core``'s policies before graph capture
    (``PreparedKda.prepare_caches``); it reads the FP32 conv weights and ``f_b`` as given.
    Launched with PDL on the current stream.
    """

    def __call__(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> torch.Tensor:
        return core.run(projected, out, caches=caches, slots=slots, accepted=accepted)

    def reference(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ):
        """``(out, states, pages)`` in FP64 (``_kda_reference.kda_core_reference``); ``out``
        is not read."""
        tokens = slots.shape[1]
        return kda_core_reference(core, projected, caches, slots, accepted, tokens)

    def is_valid(
        self,
        core,
        projected: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        """Raise where the kernel reads or writes out of place without an error: an accepted
        count outside 1..q, an active request without a conv page, two active requests on one
        slot, a conv page narrower than q + 2, a state view without contiguous key rows, or
        operand rows narrower than the packed layout."""
        tokens, requests = slots.shape[1], projected.shape[0] // slots.shape[1]
        dim, heads = core.dim, core.heads
        conv, state = caches[:2]
        if projected.shape[0] != requests * tokens or projected.shape[1] < 4 * dim + 128 + heads:
            raise ValueError("kda_verify: projected must be [requests * q, >= 4 dim + 128 + H]")
        if out.dtype != torch.bfloat16 or out.shape != (projected.shape[0], dim):
            raise ValueError("kda_verify: out must be BF16 [requests * q, dim]")
        if state.stride()[-2:] != (128, 1) or conv.stride(-1) != 1:
            raise ValueError("kda_verify: state rows and conv windows must be contiguous")
        if conv.shape[-1] != tokens + 2:
            raise ValueError("kda_verify: the conv pages must hold q + 2 columns")
        counts = accepted[:requests].tolist()
        table = slots[:requests].tolist()
        used: set[int] = set()
        for b in range(requests):
            if not 1 <= counts[b] <= tokens:
                raise ValueError(f"kda_verify: request {b} accepted {counts[b]} outside 1..q")
            if table[b][counts[b] - 1] <= 0:
                continue
            if table[b][0] <= 0:
                raise ValueError(f"kda_verify: active request {b} has no conv page")
            mine = {s for s in table[b] if s > 0}
            if mine & used:
                raise ValueError("kda_verify: active requests share a slot")
            used |= mine


kda_verify = KdaVerify()
