# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routed latent all-reduce + RMSNorm (MNNVL one-shot) that releases its programmatic
dependents right after its own ``griddepcontrol.wait`` (CuTe DSL,
``kernels/k3r_comm/k3_latent_ar_norm.py``)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_comm.k3_latent_ar_norm import run_k3_latent_ar_norm, run_k3_latent_ar_norm_row
from ..norm.k3_attn_res_norm import mxfp8_rows_reference
from .k3_lamport_finish import lamport_sum_reference

#: Ranks per FP32 partial sum of the C++ one-shot kernel the result reproduces.
_RANK_CHUNK = 8


class K3LatentArRmsnorm(OpWrapper):
    """``out = bf16((s * rsqrt(mean(s^2) + eps)) * gamma)`` with ``s`` the BF16 all-reduce of every
    rank's ``x [M, 3584]`` (``norm=False``: ``s`` alone), over the MNNVL Lamport workspace at
    ``uc_addr`` (this rank's unicast buffer) / ``mc_addr`` (multicast base) / ``flags`` (rotation
    state), as this ``rank`` of ``world_size``. Returns ``out``, or ``(out, xq, xs)`` with
    ``mxfp8_out = (xq, xs)``: also the MXFP8 form of ``out``, scales row-major ``[M, 112]`` or with
    ``sf_blocks`` in the mid-M GEMM's atoms. ``row``: the one-CTA-per-row kernel (up to 128 rows,
    2..16 ranks), else the cluster kernel (up to 38 rows); same bits."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The group is the test harness's (2 and 4 GPUs of one node); K3 runs 8 (helix8) and 16 (helix16 ULL).
    # mxfp8: "rows" (row-major scales, the low-M GEMM's operand), "atoms" (the mid-M GEMM's, one
    # 128-row block) or None.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=why,
            spec=dict(
                rows=rows,
                hidden=3584,
                row=row,
                mxfp8=mxfp8,
                early_state=early_state,
                eps=1e-6,
            ),
        )
        for why, rows, row, mxfp8, early_state in (
            ("one token, row kernel", 1, True, None, False),
            ("helix8 own-workspace latent all-reduce: 16 rows, row kernel", 16, True, None, False),
            (
                "helix8 b16 mtp5 verify: 96 rows, MXFP8 in the mid-M GEMM's atoms",
                96,
                True,
                "atoms",
                False,
            ),
            ("cluster kernel at its 38-row maximum", 38, False, None, False),
            ("helix16 ULL tail sandwich: 8 rows, MXFP8 for the low-M GEMM", 8, True, "rows", False),
            (
                "helix16 ULL column-parallel tail: 32 rows, MXFP8, early state",
                32,
                True,
                "rows",
                True,
            ),
        )
    )

    note = """
    Bit-identical to the C++ one-shot MNNVL all-reduce with the RMSNorm
    epilogue (oneshotAllreduceFusionKernel), so C++ and these calls alternate on
    one workspace (same uint32[9] rotation state); every rank makes the same
    sequence of calls. Each Lamport buffer must hold M x W x 3584 x 2 bytes.
    Pushes x (sanitized) through the multicast address, re-arms the previous
    call's buffer, polls the W slots and sums them in FP32 (chunks of eight
    ranks), rounds to BF16; the norm squares in BF16, sums in the C++ reduction
    tree, rsqrt.approx. Triggers dependents right after griddepcontrol.wait.
    Writes out (and mxfp8_out); x and gamma are read-only.
    prepushed: the producer already stored x into this rank's slots (the kernel
    skips its push). early_state: the state read, arrival count and re-arm run
    before griddepcontrol.wait (the previous call on the workspace completed
    before x's producer started). sf_blocks > 0: scales in the mid-M GEMM's
    tcgen05 atoms of sf_blocks 128-row blocks, padding rows neutral 127.
    Process-wide compile caches (`_compiled`, `_compiled_row`) keyed by
    (W, PDL, norm, mxfp8[, sf_blocks, prepushed, early_state]); compile each
    key before graph capture.
    """

    def __call__(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        *,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
        use_pdl: bool = True,
        norm: bool = True,
        mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sf_blocks: int = 0,
        prepushed: bool = False,
        early_state: bool = False,
        row: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if row:
            run_k3_latent_ar_norm_row(
                x,
                gamma,
                eps,
                out,
                uc_addr=uc_addr,
                mc_addr=mc_addr,
                flags=flags,
                rank=rank,
                world_size=world_size,
                use_pdl=use_pdl,
                norm=norm,
                mxfp8_out=mxfp8_out,
                sf_blocks=sf_blocks,
                prepushed=prepushed,
                early_state=early_state,
            )
        else:
            run_k3_latent_ar_norm(
                x,
                gamma,
                eps,
                out,
                uc_addr=uc_addr,
                mc_addr=mc_addr,
                flags=flags,
                rank=rank,
                world_size=world_size,
                use_pdl=use_pdl,
                norm=norm,
                mxfp8_out=mxfp8_out,
            )
        if mxfp8_out is None:
            return out
        return (out, *mxfp8_out)

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        *,
        uc_addr: int = 0,
        mc_addr: int = 0,
        flags: Optional[torch.Tensor] = None,
        rank: int = 0,
        world_size: int = 0,
        use_pdl: bool = True,
        norm: bool = True,
        mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sf_blocks: int = 0,
        prepushed: bool = False,
        early_state: bool = False,
        row: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """The collective exception: every rank's ``x`` in rank order replaces ``x``. The sum in
        the C++ order (chunks of eight ranks), the norm in FP32 rounded once; with ``mxfp8_out``
        the MXFP8 form of the expected ``out`` with row-major scales (``sf_blocks`` only moves
        them). ``out`` and the workspace arguments are not read."""
        total = lamport_sum_reference(inputs_in_rank_order, _RANK_CHUNK)
        if not norm:
            return total
        s = total.float()
        expected = (s * torch.rsqrt(s.square().mean(dim=-1, keepdim=True) + eps)) * gamma.float()
        expected = expected.to(torch.bfloat16)
        if mxfp8_out is None:
            return expected
        return (expected, *mxfp8_rows_reference(expected))

    def is_valid(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        *,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
        use_pdl: bool = True,
        norm: bool = True,
        mxfp8_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sf_blocks: int = 0,
        prepushed: bool = False,
        early_state: bool = False,
        row: bool = True,
    ) -> None:
        # The launchers do not re-validate: the kernels read dense 3584-wide rows from x and gamma
        # and store dense rows into out.
        from ...kernels.k3r_comm.k3_latent_ar_norm import can_run, can_run_row

        served = can_run_row(x, gamma, world_size) if row else can_run(x, gamma, world_size)
        assert served, "outside the kernel's shape (3584-wide contiguous BF16 rows, M, W)"
        assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
        assert 0 <= rank < world_size, f"rank {rank} outside a group of {world_size}"
        # The cluster kernel takes none of these and would drop them without an error.
        assert row or not (sf_blocks or prepushed or early_state), (
            "sf_blocks / prepushed / early_state are row-kernel options"
        )


k3_latent_ar_rmsnorm = K3LatentArRmsnorm()
