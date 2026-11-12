# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M FP8-weight GEMM of BF16 rows whose epilogue pushes the output into the next MNNVL
all-reduce on a Lamport workspace (instead of storing it)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm import k3_lamport_ar
from ...kernels.k3r_gemm import k3_fp8w_lowm_gemm as _lowm
from . import _fp8_reference as ref


class K3Fp8wLowmGemmPush(OpWrapper):
    """Push this rank's BF16 ``x @ dequant(w, w_scale)^T`` (``[M, N]``) into the Lamport slots of
    the next all-reduce on ``ws`` (``one_shot`` or two-shot transport); returns None.
    ``k3_lamport_ar.run_lamport_finish`` with the same ``one_shot`` (or a consumer that finishes
    the call) completes it.

    ``gate``: the output-gated GEMM (``x * gate``, PyTorch's BF16 multiply). ``strided``: ``x`` and
    ``w`` are column slices of wider tensors (unit column stride, 16-byte aligned rows) read in
    place, ``w_scale`` the packed scale of the ``[N, K]`` slice, and with ``addend`` (contiguous
    BF16 ``[M, N]``) the pushed values are ``bf16(result + addend)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: Row-parallel decode projections of the helix deployments, at 8 rows per request.
    #: ``k_total`` / ``strided``: each rank's ``[rank * k, (rank + 1) * k)`` column slice of a
    #: ``k_total``-wide operand. ``sm_107``: the push of a split-K tactic, which only SM107's
    #: ``ar_push_tactic`` admits (GB300 keeps its ``split_k == 1`` rule there).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 KDA / MLA o_proj all-reduce of one request, one-shot transport",
            spec={"n": 7168, "k": 768, "m": 8, "one_shot": True},
        ),
        Cell(
            why="helix16 o_proj at eight requests, two-shot transport",
            spec={"n": 7168, "k": 768, "m": 64, "one_shot": False},
        ),
        Cell(
            why="helix16 o_proj with the output gate multiplied in (gated kernel)",
            spec={"n": 7168, "k": 768, "m": 8, "one_shot": True, "gate": True},
        ),
        Cell(
            why="MoE row-parallel tail: latent_up on this rank's K slice of the latent, read in "
            "place, plus the shared-expert addend (split-K owners push)",
            spec={
                "n": 7168,
                "k": 896,
                "k_total": 3584,
                "m": 8,
                "one_shot": True,
                "strided": True,
                "addend": True,
                "sm_107": True,
            },
        ),
    )

    note = """
    The low-M kernel's tactic, so with the finish the call is bitwise the GEMM followed by the
    MNNVL all-reduce of its output; above the low-M range (ungated, unsliced) the mid-M GEMM's push
    epilogue after its quantization kernel. ``-0`` is sent as ``+0``. PDL: the dependent (the
    finish) is released right after the GEMM's own ``griddepcontrol.wait``; the weight streams
    before it. The workspace's rotation state is read, not advanced (the finish advances it).
    """

    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        gate: Optional[torch.Tensor] = None,
        addend: Optional[torch.Tensor] = None,
        strided: bool = False,
    ) -> None:
        if strided:
            _lowm.run_k3_fp8w_lowm_gemm_split_ar_push(x, w, w_scale, ws, one_shot, addend=addend)
        else:
            _lowm.run_k3_fp8w_lowm_gemm_ar_push(x, w, w_scale, ws, one_shot, gate=gate)

    def reference(
        self,
        x: Sequence[torch.Tensor],
        w: Sequence[torch.Tensor],
        w_scale: Sequence[torch.Tensor],
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        gate: Optional[Sequence[torch.Tensor]] = None,
        addend: Optional[Sequence[torch.Tensor]] = None,
        strided: bool = False,
    ) -> torch.Tensor:
        """A collective: ``x``, ``w``, ``w_scale`` (and ``gate`` / ``addend``) are the group's
        inputs in rank order. Returns the BF16 ``[M, N]`` the finish of the call yields on every
        rank: the all-reduce of each rank's BF16 GEMM output. ``ws``, ``one_shot`` and
        ``strided`` do not change the result."""
        parts = []
        for r in range(len(x)):
            xr = x[r] if gate is None else x[r] * gate[r]
            y = ref.lowm_gemm(xr.contiguous(), w[r].contiguous(), w_scale[r])
            if addend is not None:
                y = (y.float() + addend[r].float()).to(torch.bfloat16)
            parts.append(y)
        return ref.all_reduce_bf16(parts)

    def is_valid(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool,
        gate: Optional[torch.Tensor] = None,
        addend: Optional[torch.Tensor] = None,
        strided: bool = False,
    ) -> None:
        # Each launcher ignores the other variant's argument without an error.
        assert not (strided and gate is not None), "the strided push has no gated variant"
        assert strided or addend is None, "only the strided push adds an addend"


k3_fp8w_lowm_gemm_push = K3Fp8wLowmGemmPush()
