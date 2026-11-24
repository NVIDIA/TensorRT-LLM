# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin routed-latent all-reduce + RMSNorm (MNNVL one-shot) with an early PDL trigger.

Bit-identical to the C++ one-shot MNNVL all-reduce with the RMSNorm fusion, but it releases its
programmatic dependents right after its own ``griddepcontrol.wait``, so latent_up streams its
weight while the all-reduce waits for the other ranks: one 512-thread CTA per row. The kernel is
``kernels/comm/rubin_latent_ar_norm.py`` (``run_k3_latent_ar_norm_row``); ``__call__`` calls its
launcher directly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ..norm._rubin_norm_common import mxfp8_reference
from ._rubin_comm_common import lamport_allreduce_reference

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.comm.rubin_latent_ar_norm import run_k3_latent_ar_norm_row as _launch_row

#: The kernel's row width (the routed latent) and the ranks summed per partial sum. The kernel
#: rejects other group sizes itself.
LATENT = 3584
RANK_CHUNK = 8
#: Kimi K3.1's routed_expert_norm epsilon, and the group the op tests drive (one VR200 tray).
EPS = 1e-5
WORLD = 4


def latent_ar_norm_reference(
    inputs_in_rank_order: Sequence[torch.Tensor],
    gamma: torch.Tensor,
    eps: float,
    norm: bool,
) -> torch.Tensor:
    """The bf16 sum of every rank's partial (``-0`` sent as ``+0``, fp32, eight ranks per partial
    sum, rounded once), then with ``norm`` ``bf16(s * rsqrt(mean(s^2) + eps) * gamma)`` in fp32,
    rounded once (the kernel squares in bf16; the band of its test covers that)."""
    s = lamport_allreduce_reference(inputs_in_rank_order, RANK_CHUNK)
    if not norm:
        return s
    x = s.float()
    y = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma.float()
    return y.to(torch.bfloat16)


def _check(
    x: torch.Tensor,
    gamma: torch.Tensor,
    out: torch.Tensor,
    flags: torch.Tensor,
    rank: int,
    world_size: int,
    mxfp8_out: Optional[tuple[torch.Tensor, torch.Tensor]],
    sf_blocks: int = 0,
) -> None:
    """What the launchers take without checking: they read and write raw pointers."""
    for name, t in (("x", x), ("out", out)):
        assert t.dim() == 2 and t.shape[1] == LATENT, f"{name} must be [M, {LATENT}]"
        assert t.dtype == torch.bfloat16 and t.is_contiguous(), f"{name}: bf16 contiguous"
        assert t.data_ptr() % 16 == 0, f"{name}: 16-byte aligned"
    assert out.shape == x.shape, "out must have x's shape"
    assert gamma.dtype == torch.bfloat16 and tuple(gamma.shape) == (LATENT,), "gamma: bf16 [3584]"
    assert gamma.is_contiguous() and gamma.data_ptr() % 16 == 0, "gamma: contiguous, aligned"
    for t in (gamma, out, flags):
        assert t.device == x.device, "every tensor on x's device"
    assert 0 <= rank < world_size, f"rank {rank} outside a group of {world_size}"
    if mxfp8_out is not None:
        xq, xs = mxfp8_out
        assert xq.dtype == torch.float8_e4m3fn and xq.shape == x.shape, "xq: e4m3 like x"
        groups = (LATENT // 128) * sf_blocks * 512 if sf_blocks else x.shape[0] * (LATENT // 32)
        assert xs.dtype == torch.uint8 and xs.numel() == groups, f"xs: {groups} uint8 bytes"


class RubinLatentArNormRow(OpWrapper):
    """``out`` [M, 3584] bf16 = RMSNorm(``gamma``, ``eps``) of the sum of every rank's ``x``, one
    512-thread CTA per row (no row limit in the kernel). ``sf_blocks`` > 0 stores the MXFP8 scales
    in the mid-M FP8 GEMM's atoms
    (M <= 128 * sf_blocks); ``prepushed``: the producer already stored ``x`` into this rank's
    slots; ``early_state``: the state read, the arrival count and the re-arm run before the
    grid wait. Returns ``out``."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{rows} rows ({site}), {WORLD} ranks",
                spec=dict(world=WORLD, rows=rows, early_trigger=True, norm=True, mxfp8=False),
            )
            for rows, site in (
                (1, "one decode row"),
                (8, "TEP8 b1 k7"),
                (96, "helix8 b16 k5"),
                (128, "the largest row count the targets send here"),
            )
        ),
        Cell(
            why="96 rows, sum only (norm=False)",
            spec=dict(world=WORLD, rows=96, early_trigger=True, norm=False, mxfp8=False),
        ),
        Cell(
            why="96 rows, late trigger",
            spec=dict(world=WORLD, rows=96, early_trigger=False, norm=True, mxfp8=False),
        ),
        Cell(
            why="8 rows, early state (state read and re-arm before the grid wait)",
            spec=dict(
                world=WORLD, rows=8, early_trigger=True, norm=True, mxfp8=False, early_state=True
            ),
        ),
        Cell(
            why="96 rows, MXFP8 side output, row-major scales",
            spec=dict(world=WORLD, rows=96, early_trigger=True, norm=True, mxfp8=True),
        ),
        Cell(
            why="96 rows, MXFP8 side output, scales in one 128-row atom block",
            spec=dict(world=WORLD, rows=96, early_trigger=True, norm=True, mxfp8=True, sf_blocks=1),
        ),
    )
    note = """
    Every rank of the group launches it with its ``rank`` on the same MNNVL Lamport workspace
    (``uc_addr`` local unicast base, ``mc_addr`` multicast base, ``flags`` the shared 9-word
    rotation state); the C++ one- and two-shot kernels can use the workspace before or after it.
    One Lamport buffer must hold M x world x 3584 x 2 bytes (the launchers do not check). The
    norm sums bf16 squares in element order and takes rsqrt.approx, as the C++ kernel does;
    ``norm=False`` returns the bf16 sum. ``early_trigger`` releases the dependents right after
    the wait instead of after the reduction; ``use_pdl`` changes scheduling. Neither changes
    results. ``mxfp8_out`` also stores the low-M FP8 GEMM's quantization of ``out``. The cells
    drive 4 ranks (one VR200 tray); the targets' 8-rank groups are driven by target-level runs.
    ``prepushed`` has no cell: every target passes False.
    """

    def __call__(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
        use_pdl: bool,
        early_trigger: bool,
        norm: bool,
        mxfp8_out: Optional[tuple[torch.Tensor, torch.Tensor]],
        sf_blocks: int,
        prepushed: bool,
        early_state: bool,
    ) -> torch.Tensor:
        _launch_row(
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
            early_trigger=early_trigger,
            norm=norm,
            mxfp8_out=mxfp8_out,
            sf_blocks=sf_blocks,
            prepushed=prepushed,
            early_state=early_state,
        )
        return out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        gamma: torch.Tensor,
        eps: float,
        norm: bool,
        mxfp8: bool,
        sf_blocks: int = 0,
    ):
        """What every rank gets back (the collective exception: every rank's ``x`` in rank
        order); with ``mxfp8``, ``(out, xq, xs)``, ``sf_blocks`` > 0: the scales in atoms."""
        out = latent_ar_norm_reference(inputs_in_rank_order, gamma, eps, norm)
        if not mxfp8:
            return out
        return (out, *mxfp8_reference(out, sf_blocks > 0, max(sf_blocks, 1)))

    def is_valid(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
        use_pdl: bool,
        early_trigger: bool,
        norm: bool,
        mxfp8_out: Optional[tuple[torch.Tensor, torch.Tensor]],
        sf_blocks: int,
        prepushed: bool,
        early_state: bool,
    ) -> None:
        _check(x, gamma, out, flags, rank, world_size, mxfp8_out, sf_blocks)


rubin_latent_ar_norm_row = RubinLatentArNormRow()
