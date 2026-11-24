# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3.1-owned folded KDA verify pair of three or four requests' q8 rows (24 / 32 rows) at FP32
state: ``PreparedKdaFold.run_fold`` (``kernels/ssm/rubin_ull_kda_fold_fp32.py``: MR8's
``KdaInProj64`` with ``KdaSplit``'s prologue in its epilogue, then ``KdaVerifyCols`` on the
prepared operands, its F_A wait bounded)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_kda_reference import kda_core_reference

TOKENS = 8
HEADS = 6
REQUESTS = (3, 4)


def _cell(why: str, accepted: tuple[int, ...], padded: tuple[int, ...] = ()) -> Cell:
    return Cell(why=why, spec=dict(accepted=accepted, padded=padded, tokens=TOKENS, heads=HEADS))


class KdaFoldFp32(OpWrapper):
    """The KDA core of three or four requests' q verify rows straight from the layer input, in
    place, on the K3.1-owned folded pair (FP32 state, bounded F_A wait).

    ``fold`` is ``PreparedKdaFold`` over the layer's packed BF16 [pitch, hidden] weight, its
    ``PreparedKda`` core (FP32 state), its own armed projection buffer and a prepared scratch,
    compiled for the row count; ``x`` BF16 [r, hidden] contiguous, request-major (r = 24 or 32);
    ``out`` BF16 [r, H * 128]; ``caches``, ``slots`` [r / q, q] and ``accepted`` [>= r / q] as
    ``kda_verify``. Writes ``out``, the state rows and the conv pages; returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("3 x q8 (24 rows)", (2, 5, 8)),
        _cell("3 x q8, the last request padded", (8, 1, 3), (2,)),
        _cell("4 x q8 (32 rows)", (1, 4, 7, 8)),
        _cell("4 x q8 with a padding request (slots 0)", (2, 1, 5, 3), (1,)),
    )

    note = """
    The 24- and 32-row verify steps of the K3.1 ULL target's KDA layers when
    ``FlatKda.fold_supported`` selects the folded pair, derived from MR8
    df90f706c5 ``KDA_FOLD64`` (``cutedsl/kda/inproj64.py``, ``cutedsl/kda/verify.py``;
    ``kernels/SOURCE.json``). The in-projection's entry loads read only the step's requests (the
    64-column tile's other columns are padding) and the verify's F_A wait traps past
    ``spin_trap.BOUND_NS`` (sites 213 / 214). Bitwise the plain pair at the same in-projection
    tactic (the 64-row ``SplitKTmemGemm64`` tactic: Rubin's at these rows). The projection buffer
    is the layer's own, its F_A columns armed once and re-armed by each verify's last reader.
    Compiled before graph capture (``PreparedKdaFold.prepare_caches``).
    """

    def __call__(
        self,
        fold,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> torch.Tensor:
        return fold.run_fold(x, out, caches=caches, slots=slots, accepted=accepted)

    def reference(
        self,
        fold,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ):
        """``(out, states, pages)`` in FP64: the packed rows ``bf16(x @ W^T)`` through
        ``_rubin_ull_kda_reference.kda_core_reference``. ``out`` is not read."""
        projected = (x.double() @ fold.weight.double().T).float().bfloat16()
        return kda_core_reference(fold.core, projected, caches, slots, accepted, slots.shape[1])

    def is_valid(
        self,
        fold,
        x: torch.Tensor,
        out: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        """Raise where the kernels would misplace a read or write without an error: a state other
        than FP32, rows other than three or four requests' q8, an accepted count outside 1..q, an
        active request without a conv page or reusing a slot, non-contiguous input rows or a
        mis-shaped output."""
        if caches is not None and caches[1].dtype != torch.float32:
            raise ValueError("kda_fold_fp32: the KDA state must be FP32")
        rows, dim = x.shape[0], fold.core.dim
        requests = rows // TOKENS
        if rows % TOKENS or requests not in REQUESTS or rows not in fold.codes:
            raise ValueError("kda_fold_fp32: x must be 3 or 4 requests' q8 rows, compiled")
        if slots.dim() != 2 or slots.shape[0] < requests or slots.shape[1] != TOKENS:
            raise ValueError("kda_fold_fp32: slots must be [requests, q8]")
        if x.shape[1] != fold.weight.shape[1] or x.dtype != torch.bfloat16:
            raise ValueError("kda_fold_fp32: x must be BF16 [r, hidden] rows")
        if not x.is_contiguous():
            raise ValueError("kda_fold_fp32: x must be contiguous")
        if out.dtype != torch.bfloat16 or out.shape != (rows, dim) or out.stride(1) != 1:
            raise ValueError("kda_fold_fp32: out must be BF16 [r, dim]")
        if accepted.numel() < requests:
            raise ValueError("kda_fold_fp32: accepted needs one count per request")
        for request in range(requests):
            count, row = int(accepted[request]), slots[request].tolist()
            if not 1 <= count <= TOKENS:
                raise ValueError(f"kda_fold_fp32: accepted {count} outside 1..q")
            if row[count - 1] > 0:
                if row[0] <= 0:
                    raise ValueError("kda_fold_fp32: an active request needs its conv page")
                active = [s for s in row if s > 0]
                if len(set(active)) != len(active):
                    raise ValueError("kda_fold_fp32: a request's positions share a slot")


kda_fold_fp32 = KdaFoldFp32()
