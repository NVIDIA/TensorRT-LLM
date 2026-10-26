# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multicast all-reduce + residual + RMSNorm (+ MXFP8 copy) of the decode layer boundary, 2 ranks on SM107."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0


def mxfp8_rows(normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of BF16 rows as ``trtllm::mxfp8_quantize(normed, True)``: E4M3 [M, H] and 128x4-swizzled UE8M0 bytes.

    One scale per 32 columns, the smallest power of two at or above ``amax / 448`` (biased exponent, round up),
    data = row / scale rounded to nearest E4M3. The scale buffer is padded to 128 rows, padding bytes left as the
    op leaves them (the comparison masks them out).
    """
    m, h = normed.shape
    blocks = normed.float().view(m, h // 32, 32)
    amax = blocks.abs().amax(-1)
    bits = (amax / _E4M3_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    scale = torch.exp2(byte.float() - 127.0).unsqueeze(-1)
    data = torch.where(amax.unsqueeze(-1) != 0, blocks / scale, torch.zeros_like(blocks))
    data = data.view(m, h).to(torch.float8_e4m3fn)
    return data, swizzle_sf(byte.to(torch.uint8))


def swizzle_sf(byte: torch.Tensor) -> torch.Tensor:
    """Scatter [rows, H / 32] scale bytes into the 128x4-swizzled buffer the op writes (zero padding)."""
    rows, cols = byte.shape
    padded_rows = (rows + 127) // 128 * 128
    r = torch.arange(rows, device=byte.device).view(-1, 1)
    c = torch.arange(cols, device=byte.device).view(1, -1)
    index = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * cols)
    )
    out = torch.zeros(padded_rows * cols, dtype=torch.uint8, device=byte.device)
    out[index.reshape(-1)] = byte.reshape(-1)
    return out


def sf_written(rows: int, cols: int, device) -> torch.Tensor:
    """Mask of the swizzled scale bytes the op writes for ``rows`` rows (the padding rows are left unwritten)."""
    return swizzle_sf(torch.ones(rows, cols, dtype=torch.uint8, device=device)).bool()


def boundary(
    partials: Sequence[torch.Tensor],
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    quant: bool,
    drafter: bool = False,
) -> tuple[torch.Tensor, ...]:
    """The boundary arithmetic once each rank's BF16 partial is known (shared with the finalize entry).

    ``drafter``: the Eagle3 drafter boundary (opts bit 16), whose stock is a plain BF16 all-reduce, then FlashInfer
    fused_add_rmsnorm: the RMS statistic and the normalization take the unrounded FP32 sum bf16(rank sum) + residual;
    only the stored residual is rounded to BF16.

    The stock rounding points of this target (a BF16 all-reduce, then the fused residual add + RMSNorm): ranks are
    summed in FP32 in rank order and rounded to BF16, the residual added and rounded to BF16 again; the RMS statistic
    and the normalization are done in float64 and rounded to BF16 once.
    """
    total = torch.zeros(partials[0].shape, dtype=torch.float32, device=residual.device)
    for p in partials:
        total = total + p.float()
    pre = total.to(torch.bfloat16).float() + residual.float()
    updated = pre.to(torch.bfloat16)
    u = pre.double() if drafter else updated.double()
    inv = torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps)
    normed = (u * inv * norm_w.double()).to(torch.bfloat16)
    if not quant:
        return normed, updated
    return (normed, updated, *mxfp8_rows(normed))


class M3McAllreduceNorm(OpWrapper):
    """One TP group's boundary: ``updated = bf16(residual + bf16(sum_r bf16(x_r + x2_r)))``, ``normed =
    RMSNorm(updated)``: bitwise the stock all-reduce + fused residual RMSNorm (+ mxfp8_quantize) of SM107 TP2.

    Every rank pushes its partial through the multicast mapping ``ws_mc_ptr`` into every rank's exchange, then polls
    its own view ``ws_uc`` (re-arming the words it read) and rotates the per-CTA call parities in ``flags``; the
    workspace is the ``McAllReduceWorkspace`` bucket of ``maxt`` tokens. All ranks must make the same calls in the
    same stream order. ``prepushed`` 2 means a producer already pushed ``x`` (and launched this grid after its
    pushes); 3 that two producers pushed ``x`` into exchange set A and ``x2`` into set B. Returns
    ``[normed, updated]`` or, with ``quant``, also the MXFP8 copy of ``normed`` (E4M3 rows, swizzled UE8M0).
    """

    ARCHS = frozenset({Arch.SM_107})

    # The tp2 decode boundaries (hidden 6144, 2 ranks, eps 1e-6): every decode step fits the 8-token bucket, epilogue
    # option 7 at 1, 2, 7, 8 tokens and 15 at 3..6 (ops/m3_mc_allreduce.AR_OPTS_BY_M; every option is
    # bit-identical).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode token at the attention boundary, x only, option 7",
            spec=dict(m=1, maxt=8, opts=7, x2=False, quant=False, prepushed=0),
        ),
        Cell(
            why="MoE boundary at 4 rows (one request verifying 1 + 3): shared + routed summed in-kernel, MXFP8 copy",
            spec=dict(m=4, maxt=8, opts=15, x2=True, quant=True, prepushed=0),
        ),
        Cell(
            why="MoE boundary at 8 rows (two requests), MXFP8 copy for the next QKV",
            spec=dict(m=8, maxt=8, opts=7, x2=True, quant=True, prepushed=0),
        ),
        Cell(
            why="attention boundary whose 8 rows the o_proj GEMV already pushed (prepushed 2)",
            spec=dict(m=8, maxt=8, opts=7, x2=False, quant=False, prepushed=2),
        ),
        # Every other decode row count (modeling's MC_ALLREDUCE), the second addend and the MXFP8 copy alternating.
        *(
            Cell(
                why=f"{m} rows: a boundary row count between the ones above",
                spec=dict(
                    m=m,
                    maxt=8,
                    opts=(7, 7, 15, 15, 15, 15, 7, 7)[m - 1],
                    x2=m % 2 == 0,
                    quant=m % 3 == 0,
                    prepushed=0,
                ),
            )
            for m in range(1, 9)
            if m not in (1, 4, 8)
        ),
        # B3, the batch-4 decode steps (modeling's MC_ALLREDUCE past 8 rows): every row count 9..16 on the 16-token
        # bucket (option 15), the second addend and the MXFP8 copy alternating.
        *(
            Cell(
                why=f"{m} rows on the 16-token bucket (batch-4 decode step)",
                spec=dict(m=m, maxt=16, opts=15, x2=m % 2 == 0, quant=m % 3 == 0, prepushed=0),
            )
            for m in range(9, 17)
        ),
        # The batch-8 decode steps: every row count 17..32 on the 48-token bucket (option 15), the second addend and
        # the MXFP8 copy alternating.
        *(
            Cell(
                why=f"{m} rows on the 48-token bucket (batch-8 decode step)",
                spec=dict(m=m, maxt=48, opts=15, x2=m % 2 == 0, quant=m % 3 == 0, prepushed=0),
            )
            for m in range(17, 33)
        ),
        # The Eagle3 drafter boundaries (modeling's DRAFT_MC_AR, SpecTail): option 16 (the drafter's stock rounding) on
        # 23 / 31 at every draft-pass row count 1..8 (pass 0: 4 x requests, later passes: requests), x only.
        *(
            Cell(
                why=f"Eagle3 drafter boundary at {m} rows (opts {23 if m in (1, 2, 7, 8) else 31})",
                spec=dict(
                    m=m,
                    maxt=8,
                    opts=23 if m in (1, 2, 7, 8) else 31,
                    x2=False,
                    quant=False,
                    prepushed=0,
                    drafter=True,
                ),
            )
            for m in range(1, 9)
        ),
        # Every other row count the o_proj pushes (modeling's ATTN_PUSH).
        *(
            Cell(
                why=f"{m} rows prepushed by the o_proj GEMV",
                spec=dict(
                    m=m,
                    maxt=8,
                    opts=(7, 7, 15, 15, 15, 15, 7, 7)[m - 1],
                    x2=False,
                    quant=False,
                    prepushed=2,
                ),
            )
            for m in range(1, 8)
        ),
        # B6, the batch-4 decode steps (modeling's ATTN_PUSH past 8 rows): every row count 9..16 the o_proj pushes into
        # the 16-token bucket (option 15).
        *(
            Cell(
                why=f"{m} rows prepushed by the o_proj GEMV on the 16-token bucket (batch-4 decode step)",
                spec=dict(m=m, maxt=16, opts=15, x2=False, quant=False, prepushed=2),
            )
            for m in range(9, 17)
        ),
    )

    note = """
    The epilogue option bits (1 early residual / weight load, 2 st.async cluster exchange, 4 early parity rotation,
    8 max carveout) are bit-identical by construction; only the four (bucket, option) pairs of the sm_103 kernel are
    built. SM107 TP2 against the stock AUTO all-reduce + fused residual RMSNorm + mxfp8_quantize, rows 1..24, x alone
    and x + x2: normed, updated and the MXFP8 copy (data and the scale bytes of valid rows) bitwise equal. `prepushed`
    1 (pushed, grid wait kept) is accepted by the op but no caller passes it. `pdl` changes only launch overlap and is
    driven off in the cells.

    `normed` is gated to one BF16 rounding step: the kernel's RMS statistic is a warp / cluster tree of FP32 sums
    and its `rsqrtf` is approximate, so a row whose scaled value lands next to a BF16 midpoint can round either
    way. `updated` and the MXFP8 copy of the op's own `normed` are bit-exact.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        prepushed: int,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_sm107_tp2_mc_allreduce_norm(
            x,
            x2,
            residual,
            norm_w,
            eps,
            ws_uc,
            ws_mc_ptr,
            flags,
            rank,
            world,
            cluster,
            pdl,
            None,
            None,
            maxt,
            quant,
            prepushed,
            opts,
            cap_out,
            cap_w,
            cap_eps,
        )

    def reference(
        self,
        per_rank_x: Sequence[torch.Tensor],
        per_rank_x2: Optional[Sequence[torch.Tensor]],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        quant: bool,
        drafter: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """The boundary the group is supposed to compute (``drafter``: the opts-16 drafter boundary).

        The documented collective exception: it takes every rank's ``x`` (and ``x2``) in rank order, not the calling
        rank's. Each rank's partial is ``bf16(x + x2)`` (``x`` alone without ``x2``), whatever pushed it.
        """
        partials = []
        for r, x in enumerate(per_rank_x):
            if per_rank_x2 is None:
                partials.append(x)
            else:
                partials.append((x.float() + per_rank_x2[r].float()).to(torch.bfloat16))
        return boundary(partials, residual, norm_w, eps, quant, drafter)

    def is_valid(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        prepushed: int,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
    ) -> None:
        h = residual.shape[-1]
        # The op checks norm_w's element count, not its layout: a strided view is read as packed memory.
        assert norm_w.is_contiguous(), (
            "norm_w must be contiguous; the kernel reads it as packed memory"
        )
        # The workspace is addressed as [set][parity][maxt][world][H / 2] words and never bounds-checked: a smaller
        # buffer (another bucket's, or one built for fewer ranks) is polled and re-armed out of range.
        need = 4 * maxt * world * (h // 2)
        assert ws_uc.is_contiguous() and ws_uc.numel() >= need, (
            f"ws_uc holds {ws_uc.numel()} words, the {maxt}-token bucket of {world} ranks needs {need}"
        )
        # Flags are per (token, cluster CTA) of the bucket; the op checks the count, not that they belong to the
        # workspace. Flags of another bucket hold another call history, and the parity they select is not the one the
        # peers push into: the poll then waits on the other half forever or reads last call's rows.
        assert flags.numel() == maxt * cluster, (
            f"flags has {flags.numel()} counters; the {maxt}-token bucket's are exactly {maxt * cluster}"
        )


m3_mc_allreduce_norm = M3McAllreduceNorm()
