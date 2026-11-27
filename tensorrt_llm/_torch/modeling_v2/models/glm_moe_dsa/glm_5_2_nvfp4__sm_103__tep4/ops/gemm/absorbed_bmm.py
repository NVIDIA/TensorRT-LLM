# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed-MLA batched GEMMs as the target calls them, each with its fallbacks inside:
`absorbed_bmm` (the bf16 K-B or V-B), `k_b_q_base` (the generation rows' K-B, on the decode
cell that writes the e4m3 q-base where the rope kernel beside it skips it) and `v_b` (the
generation rows' V-B, on the cell that also writes o_proj's decode MXF8 operand).

Each kind's handle (`build`) holds the compiled objects by the row counts they serve: the
decode cells (2..64, one object per row-count group), the MTP decode cells (128 / 192,
`Phase.wide` only) and the row counts between them on the next one up, and the prefill
kernel for every other count up to `PREFILL_MAX_ROWS`; `bmm_out` takes what none does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper, true_fp32_matmul

from .._build import Count
from ..attention.mla_rope import MlaQuery, RopeWidths
from . import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .bmm_out import bmm_out
from .cute_dsl_absorbed_bmm import cute_dsl_absorbed_bmm
from .projection import DECODE_MAX_ROWS, ProjectionHandle, cell_up, takes_decode_mxf8, wide_cells
from .projection import wide_cell as _wide_cell


@dataclass(frozen=True)
class AbsorbedBmmHandle:
    """One kind's ("k_b" / "v_b") bf16 objects: `cells` the decode cells by row count,
    `mtp_cells` the MTP decode cells and `wide` the row counts between them (empty with
    `wide_bucket` off), `prefill` the prefill kernel (None: `bmm_out` above the decode
    cells)."""

    kind: str
    cells: Mapping[int, Any]
    mtp_cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    prefill: Any
    count: Count


@dataclass(frozen=True)
class KbQBaseHandle:
    """The generation rows' K-B: `cells` / `mtp_cells` / `wide` the decode cells writing the
    e4m3 q-base (empty: the bf16 K-B on every row), `bf16` the plain K-B's handle, `widths`
    the rows per sequence the specialised rope kernel takes."""

    cells: Mapping[int, Any]
    mtp_cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    bf16: AbsorbedBmmHandle
    widths: RopeWidths
    count: Count


@dataclass(frozen=True)
class VbHandle:
    """The generation rows' V-B: `o8_cells` the decode cells that also write o_proj's decode
    MXF8 operand, `o8_mtp` / `o8_wide` the MTP cell writing the wide decode o_proj's (empty:
    o_proj quantizes itself), `o_proj` the projection whose GEMM reads that operand, `bf16`
    the plain V-B's handle."""

    o8_cells: Mapping[int, Any]
    o8_mtp: Mapping[int, Any]
    o8_wide: Mapping[int, Any]
    wide_bucket: bool
    o_proj: ProjectionHandle
    bf16: AbsorbedBmmHandle
    count: Count


def build(
    kind: str,
    *,
    cells: Mapping[int, Any],
    mtp_cells: Mapping[int, Any],
    prefill: Any,
    wide_bucket: bool,
    count: Count,
) -> AbsorbedBmmHandle:
    """`kind`'s handle over the objects built at load (`cute_dsl_absorbed_bmm.build` /
    `build_prefill`). The MTP object launches at any row count up to its own, so every
    count between its cells maps onto the next one up; compiles nothing."""
    return AbsorbedBmmHandle(
        kind=kind,
        cells=dict(cells),
        mtp_cells=dict(mtp_cells),
        wide=wide_cells(mtp_cells) if wide_bucket else {},
        wide_bucket=wide_bucket,
        prefill=prefill,
        count=count,
    )


def build_k_b(
    *,
    cells: Mapping[int, Any],
    mtp_cells: Mapping[int, Any],
    bf16: AbsorbedBmmHandle,
    widths: RopeWidths,
    count: Count,
) -> KbQBaseHandle:
    """The generation rows' K-B handle over the fp8 q-base cells built at load
    (`cute_dsl_absorbed_bmm.build_decode_fp8_q_base`)."""
    return KbQBaseHandle(
        cells=dict(cells),
        mtp_cells=dict(mtp_cells),
        wide=wide_cells(mtp_cells) if bf16.wide_bucket else {},
        wide_bucket=bf16.wide_bucket,
        bf16=bf16,
        widths=widths,
        count=count,
    )


def build_v_b(
    *,
    o8_cells: Mapping[int, Any],
    o8_mtp: Mapping[int, Any],
    o_proj: ProjectionHandle,
    bf16: AbsorbedBmmHandle,
    count: Count,
) -> VbHandle:
    """The generation rows' V-B handle over the side-output cells built at load
    (`cute_dsl_absorbed_bmm.build_v_b_mxf8_side`)."""
    return VbHandle(
        o8_cells=dict(o8_cells),
        o8_mtp=dict(o8_mtp),
        o8_wide=wide_cells(o8_mtp) if bf16.wide_bucket else {},
        wide_bucket=bf16.wide_bucket,
        o_proj=o_proj,
        bf16=bf16,
        count=count,
    )


def decode_cell(ph, cells: Mapping[int, Any], rows: int, count: Count):
    """`cells[rows]`, else, for a count in 9..64 without a cell where `cell_up` holds (MTP5 at
    2 / 3 requests: 12 / 18 rows; MTP3 at 3: 12), the M >= 16 group's object, which launches
    at any of them (`DECODE_RUNTIME_ROWS`), counted `dec_cell_up`. None otherwise."""
    cell = cells.get(rows)
    if cell is None and rows in _absorbed_bmm_op.DECODE_RUNTIME_ROWS and cell_up(ph, rows):
        cell = cells.get(DECODE_MAX_ROWS)
        count("dec_cell_up", cell is not None)
    return cell


class AbsorbedBmm(OpWrapper):
    """An absorbed-MLA BMM into `out_t` (`[H, M, N]` views of token-major storage): the decode
    kernel at its row counts (2..64; 9..64 where `cell_up` holds, `decode_cell`; 65..192 on
    a `Phase.wide` step, counted `dec_wide`), the prefill kernel at any other up to 8192,
    else `bmm_out`, which takes the `[H, K, N]` operand `w_kn` instead of `w_nk`."""

    def __call__(
        self,
        ph,
        x_t: torch.Tensor,
        w_kn: torch.Tensor,
        w_nk: torch.Tensor,
        out_t: torch.Tensor,
        handle: AbsorbedBmmHandle,
    ) -> None:
        rows = int(x_t.shape[1])
        runner = decode_cell(ph, handle.cells, rows, handle.count)
        if ph.wide and rows > DECODE_MAX_ROWS:
            runner = _wide_cell(
                handle.mtp_cells, handle.wide, handle.wide_bucket, rows, handle.count
            )
            handle.count("dec_wide", runner is not None)
        if runner is None and rows <= _absorbed_bmm_op.PREFILL_MAX_ROWS:
            runner = handle.prefill
        if runner is None:
            bmm_out(x_t, w_kn, out_t)
        else:
            cute_dsl_absorbed_bmm(x_t, w_nk, out_t, runner)

    def reference(
        self,
        ph,
        x_t: torch.Tensor,
        w_kn: torch.Tensor,
        w_nk: torch.Tensor,
        out_t: torch.Tensor,
        handle: AbsorbedBmmHandle,
    ) -> torch.Tensor:
        """`x_t @ w_kn`, fp32-accumulated (TF32 off), rounded once to `out_t`'s dtype."""
        with true_fp32_matmul():
            return (x_t.float() @ w_kn.float()).to(out_t.dtype)


class KbQBase(OpWrapper):
    """The generation rows' K-B of `q` (`[gen, heads * qk_dim]`) into a new `MlaQuery` for
    `mla_rope`. `rope` names the caller (`RopeWidths.takes`): where its `gen_p` rows per
    sequence take the specialised rope kernel, a decode cell's K-B writes the e4m3 q-base
    into the query itself (the bytes the rope kernel would convert) and releases the rope
    kernel at once (`from_kb`, counted `kb_fp8_qbase`): the rope skips the q-base and waits
    for the K-B only at its end, so the two run concurrently and the decode MLA's wait on the
    rope still covers the K-B. On a `Phase.wide` step's rows above 64 that is the MTP cell at
    its rows or the next one up. Else the bf16 K-B (`absorbed_bmm`) into the fused query's
    latent half, which the rope kernel reads, so it is issued first on the same stream."""

    def __call__(
        self,
        ph,
        q: torch.Tensor,
        k_b: torch.Tensor,
        k_b_nk: torch.Tensor,
        gen_p: int,
        rope: str,
        handle: KbQBaseHandle,
    ) -> MlaQuery:
        heads, nope, kv_lora = (int(s) for s in k_b.shape)
        gen = int(q.shape[0])
        qk_dim = int(q.shape[1]) // heads
        rope_dim = qk_dim - nope
        lat_dim = kv_lora + rope_dim
        dev, dt = q.device, q.dtype
        specialised = handle.widths.takes(rope, gen_p)
        q_nope, q_pe = torch.split(torch.reshape(q, [gen, heads, qk_dim]), [nope, rope_dim], -1)
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        kb8 = decode_cell(ph, handle.cells, gen, handle.count) if specialised else None
        if specialised and ph.wide and gen > DECODE_MAX_ROWS and handle.mtp_cells:
            kb8 = _wide_cell(handle.mtp_cells, handle.wide, handle.wide_bucket, gen, handle.count)
        if kb8 is not None:
            q_base = torch.split(quant_q.view(torch.uint8), [kv_lora, rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1), k_b_nk, torch.transpose(q_base, 0, 1), kb8
            )
            handle.count("kb_fp8_qbase", True)
        else:
            fq_nope = torch.split(fused_q, [kv_lora, rope_dim], -1)[0]
            absorbed_bmm(
                ph,
                torch.transpose(q_nope, 0, 1),
                k_b,
                k_b_nk,
                torch.transpose(fq_nope, 0, 1),
                handle.bf16,
            )
        return MlaQuery(fused_q, q_pe, quant_q, specialised, kb8 is not None)

    def reference(
        self,
        ph,
        q: torch.Tensor,
        k_b: torch.Tensor,
        k_b_nk: torch.Tensor,
        gen_p: int,
        rope: str,
        handle: KbQBaseHandle,
    ) -> torch.Tensor:
        """The absorbed query's latent half, `q_nope @ k_b` per head, fp32-accumulated and
        rounded once to bf16 `[gen, heads, kv_lora]` (the fp8 cell then converts those values
        to e4m3, nearest)."""
        heads, nope, _ = (int(s) for s in k_b.shape)
        gen = int(q.shape[0])
        q_nope = torch.reshape(q, [gen, heads, -1])[..., :nope]
        with true_fp32_matmul():
            prod = torch.einsum("rhn,hnk->rhk", q_nope.float(), k_b.float())
        return prod.to(q.dtype)


class Vb(OpWrapper):
    """The generation rows' V-B of `lat_out` (`[gen, heads * kv_lora]`) into `out`
    (`[gen, heads * v_dim]`). `o_pre` is the layer's load-time handoff (`_vb_pre`: a trunk
    layer whose o_proj has the decode MXF8 GEMM): where o_proj's decode or wide decode MXF8
    GEMM takes these rows (`projection.takes_decode_mxf8`) and a side-output cell serves the
    row count (up to 64: `decode_cell`; above: the MTP cell at its rows or the next one up,
    whose bf16 `out` is that cell's), the V-B also writes `out` as that GEMM's operand.
    Returns whether it did, which o_proj then takes as `dec_pre`; else the bf16 V-B
    (`absorbed_bmm`)."""

    def __call__(
        self,
        ph,
        lat_out: torch.Tensor,
        out: torch.Tensor,
        v_b_t: torch.Tensor,
        v_b_nk: torch.Tensor,
        o_pre: bool,
        handle: VbHandle,
    ) -> bool:
        heads, v_dim, kv_lora = (int(s) for s in v_b_nk.shape)
        gen = int(out.shape[0])
        side = o_pre and takes_decode_mxf8(ph, out, handle.o_proj)
        lat_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        out_t = torch.transpose(torch.reshape(out, [gen, heads, v_dim]), 0, 1)
        vb8 = None
        if side and gen <= DECODE_MAX_ROWS:
            vb8 = decode_cell(ph, handle.o8_cells, gen, handle.count)
        elif side and handle.o8_mtp:
            vb8 = _wide_cell(handle.o8_mtp, handle.o8_wide, handle.wide_bucket, gen, handle.count)
        if vb8 is not None:
            cute_dsl_absorbed_bmm(lat_t, v_b_nk, out_t, vb8)
            return True
        absorbed_bmm(ph, lat_t, v_b_t, v_b_nk, out_t, handle.bf16)
        return False

    def reference(
        self,
        ph,
        lat_out: torch.Tensor,
        out: torch.Tensor,
        v_b_t: torch.Tensor,
        v_b_nk: torch.Tensor,
        o_pre: bool,
        handle: VbHandle,
    ) -> torch.Tensor:
        """`out`'s values: `lat_out @ v_b_t` per head, fp32-accumulated, rounded once to
        bf16 `[gen, heads * v_dim]`."""
        heads, v_dim, kv_lora = (int(s) for s in v_b_nk.shape)
        gen = int(out.shape[0])
        lat = torch.reshape(lat_out, [gen, heads, kv_lora])
        with true_fp32_matmul():
            prod = torch.einsum("thk,hkv->thv", lat.float(), v_b_t.float())
        return prod.reshape(gen, heads * v_dim).to(out.dtype)


absorbed_bmm = AbsorbedBmm()
k_b_q_base = KbQBase()
v_b = Vb()
