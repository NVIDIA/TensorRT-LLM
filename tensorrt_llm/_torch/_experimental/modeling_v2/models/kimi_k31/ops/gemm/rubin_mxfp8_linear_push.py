# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rubin_mxfp8_linear_prequantized`` whose epilogue pushes its BF16 output into an MNNVL
all-reduce.

``act [M, K] x weight [N, K]^T`` as ``rubin_mxfp8_linear_prequantized`` computes it on the shared
low-M / mid-M GEMM (``fp8_block_scale_low_m_gemm``), but instead of storing the BF16 product the
GEMM's epilogue stores it into this rank's Lamport slots of the next all-reduce call on a workspace
(``kernels/comm/rubin_lamport_ar.py``'s row store ``LamportRowPush``): the token owner's scatter
slot (two-shot) or every rank's slot through the multicast address (one-shot), ``-0`` sent as
``+0``. The consumer that finishes the call (``rubin_lamport_finish``, or the Lamport mix
``rubin_attn_res_lamport_add_rmsnorm``) then reads it, as after ``rubin_lamport_push`` of the plain
product. Nothing here is registered as a torch op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..comm._rubin_comm_common import HIDDEN, WORLD, sanitize_bf16
from .rubin_mxfp8_linear import _low_m, low_m_gemm_max_m, rubin_mxfp8_linear_prequantized

if TYPE_CHECKING:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace

#: The KDA o_proj input width of one rank of an 8-way head split (12 heads x 128).
_OPROJ_K = 1536


def _cells() -> tuple[Cell, ...]:
    cells = []
    for rows, site in (
        (6, "helix8 b1 k5, low-M, split-K 4"),
        (12, "helix8 b2 k5, low-M, split-K 4"),
        (24, "helix8 b4 k5, low-M, split-K 2"),
        (54, "helix8 b9 k5, low-M, split-K 2"),
        (66, "helix8 b11 k5, mid-M"),
        (96, "helix8 b16 k5, mid-M"),
        (126, "helix8 b21 k5, mid-M"),
    ):
        for one_shot in (False, True):
            cells.append(
                Cell(
                    why=f"{rows} rows x [{HIDDEN}, {_OPROJ_K}] o_proj ({site}), {WORLD} ranks, "
                    f"{'one' if one_shot else 'two'}-shot",
                    spec=dict(world=WORLD, rows=rows, n=HIDDEN, k=_OPROJ_K, one_shot=one_shot),
                )
            )
    return tuple(cells)


def pushes_from_epilogue(m: int, n: int, k: int, weight_scale: torch.Tensor) -> bool:
    """Whether an ``[m, k] x [n, k]^T`` call of :data:`rubin_mxfp8_linear_prequantized_push` runs
    the GEMM's push epilogue (else the plain GEMM, then ``rubin_lamport_push``)."""
    low_m = _low_m() if m <= low_m_gemm_max_m(n, k) and n % 16 == 0 else None
    return low_m is not None and low_m.row_store_supported(m, n, k, weight_scale)


class RubinMxfp8LinearPrequantizedPush(OpWrapper):
    """Push this rank's ``act [M, K] x weight [N, K]^T`` (operands as
    ``rubin_mxfp8_linear_prequantized``'s, BF16 result rows of ``N`` values) into the Lamport
    slots of the next call on ``ws`` (a ``LamportWorkspace`` of the kernel module; ``desc`` its
    ``row_push_desc``), on the ``one_shot`` or two-shot transport. Returns whether the GEMM's
    epilogue pushed (False: the plain GEMM ran, then ``rubin_lamport_push``). ``use_pdl``: that
    push kernel's launch."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Where rubin_mxfp8_linear_prequantized runs the shared low-M / mid-M GEMM and its row store fits
    (pushes_from_epilogue), the same kernel and tactic run with the Lamport row store: the
    epilogue rounds each value to BF16 as the output store does, stages its rows token-major in
    shared memory and stores 16-byte vectors into the slots, after its own griddepcontrol.wait and
    the read of the workspace's rotation state (the Lamport producer rule). The GEMM's activation
    warp releases the next grid right after its wait, so a finish that polls before its own wait
    reads this call's state. Any other call runs the plain entry and rubin_lamport_push: the same
    slot payload. Calls that share a workspace must be issued in the same order by every rank;
    the payload of this rank's slots is the product with -0 sent as +0. The cells drive 4 ranks
    (one VR200 tray); helix8's 8-rank groups are driven by the target.
    """

    def __call__(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        ws: LamportWorkspace,
        desc: torch.Tensor,
        one_shot: bool,
        use_pdl: bool,
    ) -> bool:
        from ...kernels.comm.rubin_lamport_ar import (
            LamportRowPush,
            required_bytes,
            run_lamport_push,
        )

        m, k = act.shape
        n = weight.shape[0]
        if required_bytes(m, n, ws.world, one_shot) > ws.bytes_per_buffer:
            raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {n}")
        if pushes_from_epilogue(m, n, k, weight_scale):
            _low_m().run_prequantized(
                act,
                act_scale,
                weight,
                weight_scale,
                None,
                act_r128c4=True,
                row_store=LamportRowPush(ws.world, n, bool(one_shot)),
                store_desc=desc,
            )
            return True
        out = rubin_mxfp8_linear_prequantized(act, act_scale, weight, weight_scale)
        run_lamport_push(ws, out, one_shot, use_pdl)
        return False

    def reference(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        ws: LamportWorkspace,
        desc: torch.Tensor,
        one_shot: bool,
        use_pdl: bool,
    ) -> torch.Tensor:
        """The payload of this rank's slots: ``rubin_mxfp8_linear_prequantized``'s reference
        product with ``-0`` sent as ``+0``. A finish of the call is ``lamport_allreduce_reference``
        of every rank's payload; the transport and ``use_pdl`` do not enter it."""
        return sanitize_bf16(
            rubin_mxfp8_linear_prequantized.reference(act, act_scale, weight, weight_scale)
        )

    def is_valid(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        ws: LamportWorkspace,
        desc: torch.Tensor,
        one_shot: bool,
        use_pdl: bool,
    ) -> None:
        rubin_mxfp8_linear_prequantized.is_valid(act, act_scale, weight, weight_scale)
        assert isinstance(one_shot, bool) and isinstance(use_pdl, bool), "flags"
        assert act.device == ws.flags.device, "operands on the workspace's device"
        assert desc.dtype == torch.int64 and tuple(desc.shape) == (4,), "row_push_desc(ws)"
        assert desc.device == ws.flags.device, "the descriptor on the workspace's device"


rubin_mxfp8_linear_prequantized_push = RubinMxfp8LinearPrequantizedPush()
