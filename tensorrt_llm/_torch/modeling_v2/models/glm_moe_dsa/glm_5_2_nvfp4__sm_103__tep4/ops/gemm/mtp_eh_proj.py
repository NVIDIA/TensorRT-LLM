# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP module's front end, `eh_proj(concat(enorm(e), hnorm(h)))`, as two ops:
`mtp_eh_norm` (the two input norms into the projection's operand) and `mtp_eh_proj`.

At the K-split instances' row counts each rank norms one 3072-wide quarter of the concat --
ranks 0-1 halves of `e`, ranks 2-3 halves of `h` (`cute_dsl_mtp_norm_shard`) -- and the
projection's K-split partials (`cute_dsl_mtp_eh_proj`) are summed by the collective, which
`mtp_eh_proj` issues as the K-split path's last step. MTP5's 192 draft-step-0 rows take it on
a `Phase.wide` step only, and so do that step's other row counts in 65..191 (a batch of fewer
than 32 sequences), on the next instance up (128 / 192) at their exact rows; with
`wide_bucket` off they keep bf16, never the MXF8 context path. An eager draft step's or an
MTP3 / MTP5 decode step's count up to 64 without an instance takes the next instance up
(`cell_up`). Context rows above 64 take the MXF8 path where it built: the two norms write the
MXF8 GEMM's operand halves (`cute_dsl_mtp_eh_norm_cat_mxf8`). What none takes runs the two
bf16 norms, the concat (two copies into one buffer) and the replicated projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from ..comm.cute_dsl_allreduce import CommHandle, allreduce
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from ..norm import cute_dsl_mtp_eh_norm_cat_mxf8 as _eh8_op
from ..norm.cute_dsl_hidden_rmsnorm import HiddenRmsnormHandle, hidden_rmsnorm
from ..norm.cute_dsl_mtp_eh_norm_cat_mxf8 import cute_dsl_mtp_eh_norm_cat_mxf8
from ..norm.cute_dsl_mtp_norm_shard import cute_dsl_mtp_norm_shard
from . import cute_dsl_mtp_eh_proj as _mtp_eh
from .cute_dsl_mtp_eh_proj import cute_dsl_mtp_eh_proj
from .cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8
from .projection import DECODE_MAX_ROWS, ProjectionHandle, cell_up, projection, wide_cell


@dataclass(frozen=True)
class MtpEhHandle:
    """What the front end launches. `ehproj`: the K-split runner (every instance), on this
    rank's weight slice `eh_local`; `own`: the `WIDE_ROWS` counts on it; `wide`: a
    `Phase.wide` step's other counts in 65..191 re-pinned to the next instance up (empty
    with `wide_bucket` off); `up`: the counts up to 64 without an instance, re-pinned to the
    next one up (None where none launches). `norm_shard`: the quarter norm compiled for this
    rank's offset `offset`. `eh8`: the MXF8 path's norms and weight (None: not built), on
    `proj.mxf8`. `proj`: the replicated bf16 projection's handle over `eh` (column-major).
    `norm` / `copy`: the bf16 fallback's row ops; `comm` the K-split partials' collective."""

    ehproj: Any
    eh_local: torch.Tensor
    own: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    up: Mapping[int, Any]
    norm_shard: Callable
    offset: int
    use_e: bool
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: torch.Tensor
    eps: float
    eh8: Any
    proj: ProjectionHandle
    norm: HiddenRmsnormHandle
    copy: RowCopyHandle
    comm: CommHandle
    count: Count


def build(
    *,
    ehproj,
    eh_local: torch.Tensor,
    norm_shard: Callable,
    tp_rank: int,
    enorm: torch.Tensor,
    hnorm: torch.Tensor,
    eh: torch.Tensor,
    eps: float,
    eh8,
    proj: ProjectionHandle,
    norm: HiddenRmsnormHandle,
    copy: RowCopyHandle,
    comm: CommHandle,
    wide_bucket: bool,
    count: Count,
) -> MtpEhHandle:
    """The front end's handle over the kernels built at load; re-pins the K-split runner to
    every count a step can present without an instance (`cute_dsl_mtp_eh_proj.at_rows`,
    which compiles nothing)."""
    at_rows = _mtp_eh.at_rows
    own = dict.fromkeys(_mtp_eh.WIDE_ROWS, ehproj)
    wide = (
        {r: at_rows(ehproj, r) for r in range(DECODE_MAX_ROWS + 1, max(_mtp_eh.WIDE_ROWS))}
        if wide_bucket
        else {}
    )
    up = {
        r: at_rows(ehproj, r) if r < DECODE_MAX_ROWS else None
        for r in range(1, DECODE_MAX_ROWS + 1)
    }
    return MtpEhHandle(
        ehproj=ehproj,
        eh_local=eh_local,
        own=own,
        wide=wide,
        wide_bucket=wide_bucket,
        up=up,
        norm_shard=norm_shard,
        offset=(tp_rank % 2) * _mtp_eh.LOCAL_INPUT_SIZE,
        use_e=tp_rank < 2,
        enorm=enorm,
        hnorm=hnorm,
        eh=eh,
        eps=eps,
        eh8=eh8,
        proj=proj,
        norm=norm,
        copy=copy,
        comm=comm,
        count=count,
    )


@dataclass(frozen=True)
class MtpEhOperand:
    """`mtp_eh_norm`'s output, which `mtp_eh_proj` takes: `kind` the path (`"ksplit"`,
    `"mxf8"`, `"bf16"`), `x` the operand (the local quarter, the MXF8 bytes, or the bf16
    concat), `sf` the MXF8 scales, `runner` the K-split runner at the rows."""

    kind: str
    x: torch.Tensor
    sf: torch.Tensor | None = None
    runner: Any = None


def _ksplit_runner(ph, rows: int, h: MtpEhHandle):
    """The K-split runner serving `rows`, or None."""
    if rows in _mtp_eh.SUPPORTED_M_VALUES or (ph.wide and rows in _mtp_eh.MTP_ROWS):
        return h.ehproj
    if ph.wide and rows > DECODE_MAX_ROWS:
        return wide_cell(h.own, h.wide, h.wide_bucket, rows, h.count)
    if cell_up(ph, rows):
        # An eager draft step's or an MTP3 / MTP5 decode step's count: the next supported
        # instance up at its exact rows.
        runner = h.up.get(rows)
        h.count("draft_cell" if ph.up == "draft" else "dec_cell_up", runner is not None)
        return runner
    return None


class MtpEhNorm(OpWrapper):
    """The front end's two input norms into the projection's operand (`MtpEhOperand`): this
    rank's quarter of the concat where a K-split instance serves the rows and both inputs are
    contiguous; else, on context rows above 64 where the MXF8 path built and takes them, the
    MXF8 operand halves (counted `mtp_eh_mxf8`); else the bf16 norms and the concat."""

    def __call__(self, ph, e: torch.Tensor, hs: torch.Tensor, h: MtpEhHandle) -> MtpEhOperand:
        rows = int(e.shape[0])
        runner = _ksplit_runner(ph, rows, h)
        if runner is not None and e.is_contiguous() and hs.is_contiguous():
            src, nw = (e, h.enorm) if h.use_e else (hs, h.hnorm)
            k = _mtp_eh.LOCAL_INPUT_SIZE
            local = torch.empty([rows, k], dtype=e.dtype, device=e.device)
            cute_dsl_mtp_norm_shard(src, nw, local, h.offset, h.eps, h.norm_shard)
            return MtpEhOperand("ksplit", local, runner=runner)
        eh8 = h.eh8
        if eh8 is not None and rows > DECODE_MAX_ROWS and not ph.wide and _eh8_op.supports(e, hs):
            # The two norms write the MXF8 GEMM's operand halves: no bf16 norms, no concat
            # copies, no activation quant.
            q8 = _eh8_op.operand(rows, e.device)
            cute_dsl_mtp_eh_norm_cat_mxf8(e, h.enorm, hs, h.hnorm, h.eps, *q8, eh8.norm)
            h.count("mtp_eh_mxf8", True)
            return MtpEhOperand("mxf8", q8[0], q8[1])
        # The concat written by two copies into one buffer: the copy kernel takes each half (a
        # column slice of the buffer), torch.cat would be one more launch.
        en = hidden_rmsnorm(e, h.enorm, h.eps, h.norm)
        hn = hidden_rmsnorm(hs, h.hnorm, h.eps, h.norm)
        cat = torch.empty([rows, 2 * en.shape[1]], dtype=en.dtype, device=en.device)
        row_copy(cat[:, : en.shape[1]], en, h.copy)
        row_copy(cat[:, en.shape[1] :], hn, h.copy)
        return MtpEhOperand("bf16", cat)

    def reference(self, ph, e: torch.Tensor, hs: torch.Tensor, h: MtpEhHandle) -> MtpEhOperand:
        """The fp32 RMSNorms of `e` and `hs`, concatenated, rounded once to bf16."""

        def rms(x, w):
            x = x.float()
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + h.eps) * w.float()

        cat = torch.cat([rms(e, h.enorm), rms(hs, h.hnorm)], dim=-1).to(torch.bfloat16)
        return MtpEhOperand("bf16", cat)


class MtpEhProj(OpWrapper):
    """`eh_proj` of `mtp_eh_norm`'s operand, the front end's output `[rows, 6144]` bf16: the
    K-split GEMM of this rank's quarter, then the collective summing the ranks' partials;
    the MXF8 GEMM; or the replicated projection (the prefill GEMM above the decode row
    counts, else cuBLAS)."""

    def __call__(self, ph, operand: MtpEhOperand, h: MtpEhHandle) -> torch.Tensor:
        if operand.kind == "ksplit":
            partial = cute_dsl_mtp_eh_proj(operand.x, h.eh_local, operand.runner)
            return allreduce(partial, h.comm)
        if operand.kind == "mxf8":
            rows = int(operand.x.shape[0])
            runner = h.proj.mxf8
            out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=operand.x.device)
            pre = (operand.x, operand.sf)
            return cute_dsl_projection_mxf8(operand.x, h.eh8.w8, out, runner, pre=pre)
        return projection(ph, operand.x, h.eh, h.proj)

    def reference(self, ph, operand: MtpEhOperand, h: MtpEhHandle) -> torch.Tensor:
        """`concat @ eh` in fp32 of a bf16 operand, rounded once to bf16."""
        return (operand.x.float() @ h.eh.float()).to(torch.bfloat16)


mtp_eh_norm = MtpEhNorm()
mtp_eh_proj = MtpEhProj()
