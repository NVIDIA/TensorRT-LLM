# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M FP8-weight GEMM on a prequantized MXFP8 activation whose epilogue pushes the output into
the next MNNVL Lamport call: all-reduce slots, a K slice of wider operands, or all-gather columns."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm import k3_lamport_ar
from ...kernels.k3r_gemm import k3_fp8w_lowm_mxfp8 as _mx
from . import _fp8_reference as ref


class K3Fp8wLowmMxfp8Push(OpWrapper):
    """Push this rank's BF16 ``dequant(xq, xs) @ dequant(w, w_scale)^T`` into the next Lamport call
    on ``ws``; returns None. ``xq`` E4M3 ``[M, K]``, ``xs`` UE8M0 ``[M, K / 32]`` row-major,
    ``w_scale`` DeepGEMM-packed.

    * Default: into the all-reduce slots (``one_shot`` or two-shot transport).
    * ``k1``: on the K columns ``[k0, k0 + k1)`` of wider ``xq`` / ``xs`` / ``w``, read in place
      (``w_scale`` the packed scale of the ``[N, k1]`` slice); with ``addend`` (contiguous BF16
      ``[M, N]``) the pushed values are ``bf16(result + addend)``.
    * ``hidden``: into columns ``[col0, col0 + N)`` of the call's all-gather rows of width
      ``hidden`` (one-shot transport; ``one_shot`` is ignored).

    ``tactic``: the low-M MXFP8 tactic (default ``default_mx_tactic`` of the GEMM's own shape;
    the all-gather caller passes the full projection's so every 128-row tile is computed as there).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: The helix decode pushes, at 8 rows per request: ``variant`` ``ar`` (all-reduce), ``slice``
    #: (each rank's K slice ``[rank * k, (rank + 1) * k)`` of a ``k_total``-wide operand, plus an
    #: addend) or ``ag`` (each rank's ``n`` output columns of a ``hidden``-wide projection, with
    #: the full projection's tactic).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 o_proj on the KDA verify's MXFP8 rows, one-shot all-reduce, b1",
            spec={"variant": "ar", "n": 7168, "k": 768, "m": 8, "one_shot": True},
        ),
        Cell(
            why="helix8 o_proj (K = 1536), four requests, two-shot all-reduce",
            spec={"variant": "ar", "n": 7168, "k": 1536, "m": 32, "one_shot": False},
        ),
        Cell(
            why="MoE row-parallel tail: latent_up on this rank's K slice of the MXFP8 latent, "
            "plus the shared-expert addend",
            spec={"variant": "slice", "n": 7168, "k": 896, "k_total": 3584, "m": 8},
        ),
        Cell(
            why="MoE column-parallel tail: this rank's latent_up output columns into the "
            "all-gather rows, with the full projection's tactic",
            spec={"variant": "ag", "hidden": 7168, "k": 3584, "m": 8},
        ),
    )

    note = """
    The GEMM of :class:`K3Fp8wLowmMxfp8Gemm` (same tactic, K-tile order and rounding) with the
    Lamport push epilogue; ``k3_lamport_ar.run_lamport_finish`` (``ag=True`` for all-gather rows)
    or a consumer's fused finish completes the call. ``-0`` is sent as ``+0``. A PDL dependent:
    the weight streams before ``griddepcontrol.wait``.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool = True,
        k0: int = 0,
        k1: Optional[int] = None,
        addend: Optional[torch.Tensor] = None,
        col0: int = 0,
        hidden: Optional[int] = None,
        tactic: Optional["_mx.K3LowMMxTactic"] = None,
    ) -> None:
        if hidden is not None:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_ag_push(
                xq, xs, w, w_scale, ws, col0, hidden, tactic=tactic
            )
        elif k1 is not None:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push(
                xq, xs, w, w_scale, k0, k1, ws, one_shot, addend=addend, tactic=tactic
            )
        else:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_ar_push(xq, xs, w, w_scale, ws, one_shot, tactic=tactic)

    def reference(
        self,
        xq: Sequence[torch.Tensor],
        xs: Sequence[torch.Tensor],
        w: Sequence[torch.Tensor],
        w_scale: Sequence[torch.Tensor],
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool = True,
        k0: Optional[Sequence[int]] = None,
        k1: Optional[int] = None,
        addend: Optional[Sequence[torch.Tensor]] = None,
        col0: Optional[Sequence[int]] = None,
        hidden: Optional[int] = None,
        tactic: Optional["_mx.K3LowMMxTactic"] = None,
    ) -> torch.Tensor:
        """A collective: ``xq``, ``xs``, ``w``, ``w_scale`` (``k0``, ``addend``, ``col0``) are the
        group's inputs in rank order. All-reduce: the BF16 ``[M, N]`` the finish yields, the all-reduce of
        the ranks' BF16 GEMM outputs. All-gather (``hidden``): the BF16 ``[M, hidden]`` all-gather
        rows, rank ``r``'s output in columns ``[col0[r], col0[r] + N_r)`` (zero elsewhere; the
        finish adds them to the call's reduced slots). ``ws``, ``one_shot`` and ``tactic`` do not
        change the result."""
        parts = []
        for r in range(len(xq)):
            a, b = xq[r], w[r]
            e = xs[r]
            if k1 is not None:
                lo = 0 if k0 is None else k0[r]
                a, b = a[:, lo : lo + k1], b[:, lo : lo + k1]
                e = e[:, lo // 32 : (lo + k1) // 32]
            y = ref.matmul_bf16(ref.dequant(a, e, 32), ref.fp8_block_weight(b, w_scale[r]))
            if addend is not None:
                y = (y.float() + addend[r].float()).to(torch.bfloat16)
            parts.append(y)
        if hidden is None:
            return ref.all_reduce_bf16(parts)
        rows = torch.zeros(
            (parts[0].shape[0], hidden), dtype=torch.bfloat16, device=parts[0].device
        )
        for r, y in enumerate(parts):
            rows[:, col0[r] : col0[r] + y.shape[1]] = y
        return rows

    def is_valid(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool = True,
        k0: int = 0,
        k1: Optional[int] = None,
        addend: Optional[torch.Tensor] = None,
        col0: int = 0,
        hidden: Optional[int] = None,
        tactic: Optional["_mx.K3LowMMxTactic"] = None,
    ) -> None:
        # Each launcher ignores the other variants' arguments without an error.
        assert hidden is None or (k1 is None and addend is None), (
            "the all-gather push takes no K slice and no addend"
        )
        assert k1 is not None or (k0 == 0 and addend is None), (
            "k0 and addend need the K-slice push (k1)"
        )
        assert xq.dtype == torch.float8_e4m3fn and xs.dtype == torch.uint8
        assert w_scale.dtype == torch.int32, "w_scale must be the DeepGEMM-packed UE8M0 scale"


k3_fp8w_lowm_mxfp8_push = K3Fp8wLowmMxfp8Push()
