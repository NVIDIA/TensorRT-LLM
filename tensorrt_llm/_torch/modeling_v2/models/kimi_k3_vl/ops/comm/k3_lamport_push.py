# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Push a BF16 tensor into the Lamport slots of the next MNNVL all-reduce on a workspace (the
C++ kernels' first phase as a kernel of its own; CuTe DSL, ``kernels/k3_lamport_ar.py``)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm.k3_lamport_ar import LamportWorkspace, run_lamport_push
from .k3_lamport_finish import sanitized


class K3LamportPush(OpWrapper):
    """Store this rank's contiguous BF16 ``x [M, hidden]`` into its slots of the next call on
    ``ws``: slot ``(t, rank)`` of every rank's buffer (``one_shot``, multicast) or slot
    ``(t // W, rank)`` of the scatter stage of token ``t``'s owner ``t % W`` (two-shot). Returns
    None; ``k3_lamport_finish`` or a consumer that finishes the call (``k3_attn_res_lamport``)
    completes it."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The group is the test harness's (2 and 4 GPUs of one node); K3 runs 8 (helix8) and 16 (helix16 ULL).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=rows, hidden=7168, one_shot=one_shot))
        for why, rows, one_shot in (
            ("one token, one-shot", 1, True),
            ("helix16 ULL 1-request step (o_proj / shared-expert push): 8 rows, one-shot", 8, True),
            ("helix16 ULL 4-request step: 32 rows, one-shot", 32, True),
            ("32 rows, two-shot (the transport above the one-shot size)", 32, False),
            ("helix8 b16 mtp5 verify step: 96 rows, two-shot", 96, False),
            ("helix8 b16 mtp5 verify step: 96 rows, one-shot", 96, True),
        )
    )

    note = """
    For a producer without a push epilogue. ws (kernels/k3_lamport_ar
    .LamportWorkspace) holds the device addresses of the workspace; the kernel
    reads the rotation state after griddepcontrol.wait, releases its dependents
    right after, and writes only the current buffer's slots (values sanitized:
    -0 stored as +0, the sentinel being a -0 pair). x is read-only. Grid
    (M, ceil(hidden / 1024)) CTAs of 128 threads, one 16-byte vector each.
    A process-wide cache (`_compiled_push`) holds one compiled kernel per
    (W, hidden, transport, PDL); compile each key before graph capture.
    """

    def __call__(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool = True
    ) -> None:
        run_lamport_push(ws, x, one_shot, use_pdl=use_pdl)

    def reference(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool = True
    ) -> torch.Tensor:
        """The rows this rank's slots hold once the push has landed: ``x`` with ``-0`` stored as
        ``+0``. ``ws``, ``one_shot`` and ``use_pdl`` decide where they land and when, not what."""
        return sanitized(x)

    def is_valid(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool = True
    ) -> None:
        # The kernel loads dense rows of `hidden` BF16 values from x's address.
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous(), (
            "x must be a contiguous BF16 [M, hidden] tensor"
        )


k3_lamport_push = K3LamportPush()
