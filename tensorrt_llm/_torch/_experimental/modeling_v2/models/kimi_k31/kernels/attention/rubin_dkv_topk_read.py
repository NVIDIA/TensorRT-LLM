# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DKV sparse MLA read: copy the latent rows a step's Top-K selects from their owners (CuTe DSL).

Row ``k`` of the selection (``k < count``; ``count`` is read on the device) is the history token
``pos[k]`` of the step, whose latent row is row ``src_rows[pos[k]] + row_offset`` of the source
(every rank's latent pool through the peer mapping, rows of ``row_bytes``): it lands in row
``dst_start + k`` of ``dst``. ``row_bytes / 16`` consecutive threads move one row, 16 bytes each
(non-coherent 16-byte loads; the rows were written by earlier kernels), a CTA moves
``rows_per_cta`` rows per pass, and a grid of at most a few waves of CTAs strides over the
selection: the host knows only a bound of ``count``.

The sparse MLA's indices over the buffer (``DkvTopkRowsKernel``): entry ``t`` of a Top-K row of
request ``q`` is the buffer row of history token ``hist_start[q] + t`` (``rank[...] + row_base``)
when ``0 <= t < hist_len[q]``, else its new-token page's row ``table[q, t >> shift] *
stride_factor + layer_rows + (t & mask)``, else 0 (a negative ``t``, a page past the table or a
negative table entry). The same launch zeroes the buffer's first ``zero_bytes`` (page 0, where the
padding points). A CTA of 128 threads covers ``512 * row_vecs`` entries of one Top-K row, each
thread loading ``row_vecs`` 16-byte vectors of it and then one 4-byte read per entry (its rank, its
table entry, or a dummy table read for padding), all issued before any is used.

Launches use programmatic dependent launch when ``TRTLLM_ENABLE_PDL`` is unset or ``1``: each
kernel waits for its producers (the selection, the Top-K, the ranks) before it reads or writes
global memory.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: Threads a CTA aims for; it takes the largest multiple of a row's lanes below it.
_THREADS = 256
#: CTAs per SM the grid stops at (one wave and some: the loop covers the rest).
_CTAS_PER_SM = 8
_OPTIONS = "--opt-level 2 --enable-tvm-ffi"
#: Threads of an index-kernel CTA.
_ROW_THREADS = 128
#: The index kernel's 16-byte vectors per thread by Top-K rows (``_row_vecs``): one below the
#: first count (more CTAs, less serial work per thread), two up to the second, then four without
#: history tokens (with them, two: their rank reads gain from the extra CTAs).
_ROW_VECS_ROWS = (1024, 8192)


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _l(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_l(addr, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_b64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_l(addr, loc=loc, ip=ip)],
            "ld.global.nc.b64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _copy_v4(src, dst, *, loc=None, ip=None):
    """Copy 16 bytes, both addresses 16-byte aligned."""
    _asm(
        None,
        [_l(src, loc=loc, ip=ip), _l(dst, loc=loc, ip=ip)],
        "{\n.reg .b32 a, b, c, d;\nld.global.nc.v4.b32 {a, b, c, d}, [$0];\n"
        "st.global.v4.b32 [$1], {a, b, c, d};\n}",
        "l,l",
        loc=loc,
        ip=ip,
    )


def _i(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_v4_b32(addr, *, loc=None, ip=None):
    """Four 32-bit words from a 16-byte aligned address."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_l(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Store four 32-bit words at a 16-byte aligned address."""
    _asm(
        None,
        [_l(addr, loc=loc, ip=ip)] + [_i(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v4_zero(addr, *, loc=None, ip=None):
    """Store 16 zero bytes at a 16-byte aligned address."""
    _asm(
        None,
        [_l(addr, loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {0, 0, 0, 0};",
        "l",
        loc=loc,
        ip=ip,
    )


class DkvGatherRowsKernel:
    """Rows ``k < count`` of a selection: ``dst[dst_start + k] = src row (src_rows[pos[k]] +
    row_offset)``, ``row_bytes / 16`` threads per row."""

    def __init__(self, row_bytes: int, use_pdl: bool):
        if row_bytes % 16 or not 16 <= row_bytes <= 16 * _THREADS:
            raise ValueError(f"rows of {row_bytes} bytes: a multiple of 16 up to {16 * _THREADS}")
        self.row_bytes = row_bytes
        self.lanes = row_bytes // 16
        self.rows_per_cta = _THREADS // self.lanes
        self.threads = self.lanes * self.rows_per_cta
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        src_rows: cute.Tensor,
        pos: cute.Tensor,
        count: cute.Tensor,
        dst: cute.Tensor,
        src_addr: Int64,
        dst_start: Int64,
        row_offset: Int64,
        num_ctas: Int32,
        stream,
    ):
        self.kernel(
            src_rows.iterator.toint(),
            pos.iterator.toint(),
            count.iterator.toint(),
            dst.iterator.toint(),
            src_addr,
            dst_start,
            row_offset,
            num_ctas,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        src_rows_addr: Int64,
        pos_addr: Int64,
        count_addr: Int64,
        dst_addr: Int64,
        src_addr: Int64,
        dst_start: Int64,
        row_offset: Int64,
        num_ctas: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % self.lanes
        first = bidx * self.rows_per_cta + tidx // self.lanes
        step = num_ctas * self.rows_per_cta
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        count = _ld_b32(count_addr)
        passes = Int32(0)
        if first < count:
            passes = (count - first + step - 1) // step
        for p in cutlass.range(0, passes, unroll=1):
            k = Int64(first) + Int64(p) * Int64(step)
            token = _ld_b64(pos_addr + k * 8)
            row = _ld_b64(src_rows_addr + token * 8) + row_offset
            _copy_v4(
                src_addr + row * self.row_bytes + Int64(lane) * 16,
                dst_addr + (dst_start + k) * self.row_bytes + Int64(lane) * 16,
            )


@functools.cache
def _compile(row_bytes: int, use_pdl: bool):
    src_rows_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (cute.sym_int(),), stride=(1,))
    pos_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (cute.sym_int(),), stride=(1,))
    count_fake = cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int(),), stride=(1,))
    dst_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(), row_bytes), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DkvGatherRowsKernel(row_bytes, use_pdl)
    return cute.compile(
        kernel,
        src_rows_fake,
        pos_fake,
        count_fake,
        dst_fake,
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(1),
        stream_fake,
        options=_OPTIONS,
    )


@functools.cache
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def gather_rows_into(
    dst: torch.Tensor,
    dst_start: int,
    src_addr: int,
    src_rows: torch.Tensor,
    pos: torch.Tensor,
    count: torch.Tensor,
    row_offset: int,
    *,
    use_pdl: bool | None = None,
) -> None:
    """Copy rows ``k < count[0]`` of the selection ``pos`` (int64, at least ``count[0]`` entries,
    each an index of ``src_rows``) into ``dst`` (uint8 ``[M, row_bytes]``, contiguous) from row
    ``dst_start``: row ``k`` is source row ``src_rows[pos[k]] + row_offset``, the source rows of
    ``row_bytes`` from address ``src_addr``. ``count`` is int32 on the device; ``pos.numel()``
    bounds it. Compiles once per row width on the first call."""
    bound = pos.numel()
    if bound == 0:
        return
    row_bytes = dst.shape[1]
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    kernel_rows = _THREADS // (row_bytes // 16)
    device = dst.device.index if dst.device.index is not None else torch.cuda.current_device()
    ctas = min(-(-bound // kernel_rows), _CTAS_PER_SM * _sm_count(device))
    _compile(row_bytes, use_pdl)(
        src_rows, pos, count, dst, src_addr, dst_start, row_offset, int(ctas)
    )


class DkvTopkRowsKernel:
    """``out[r, j]``: Top-K entry ``topk[r, j]`` of request ``req_idx[r]`` as a row of the read
    buffer (the module docstring's rule), and ``zero_bytes`` zero bytes from ``kv``'s first row.

    ``K / (512 * row_vecs)`` CTAs of 128 threads per row (CTA ``b``: row ``b // splits``, its
    part ``b % splits``); thread ``x`` owns the part's 16-byte vectors ``u * 128 + x`` (``u <
    row_vecs``), so each of its loads and stores is coalesced across the CTA. ``history`` False (a
    step without history tokens) never reads ``hist_*`` or ``rank``.
    """

    def __init__(
        self,
        topk: int,
        block_size: int,
        zero_bytes: int,
        history: bool,
        use_pdl: bool,
        row_vecs: int = 4,
    ):
        if row_vecs not in (1, 2, 4):
            raise ValueError(f"row_vecs must be 1, 2 or 4, got {row_vecs}")
        per_cta = 4 * row_vecs * _ROW_THREADS
        if topk <= 0 or topk % per_cta:
            raise ValueError(f"a Top-K of {topk}: a multiple of {per_cta}")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {block_size}")
        if zero_bytes < 0 or zero_bytes % 16:
            raise ValueError(f"{zero_bytes} zero bytes: a multiple of 16")
        self.topk = topk
        self.threads = _ROW_THREADS
        self.row_vecs = row_vecs
        self.splits = topk // per_cta
        self.shift = block_size.bit_length() - 1
        self.mask = block_size - 1
        self.zero_vecs = zero_bytes // 16
        self.history = history
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        topk: cute.Tensor,
        req_idx: cute.Tensor,
        hist_start: cute.Tensor,
        hist_len: cute.Tensor,
        rank: cute.Tensor,
        out: cute.Tensor,
        kv: cute.Tensor,
        table_addr: Int64,
        width: Int32,
        stride_factor: Int32,
        layer_rows: Int32,
        row_base: Int32,
        rows: Int32,
        stream,
    ):
        self.kernel(
            topk.iterator.toint(),
            req_idx.iterator.toint(),
            hist_start.iterator.toint(),
            hist_len.iterator.toint(),
            rank.iterator.toint(),
            out.iterator.toint(),
            kv.iterator.toint(),
            table_addr,
            width,
            stride_factor,
            layer_rows,
            row_base,
            rows,
        ).launch(
            grid=[rows * self.splits, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        topk_addr: Int64,
        req_addr: Int64,
        start_addr: Int64,
        len_addr: Int64,
        rank_addr: Int64,
        out_addr: Int64,
        kv_addr: Int64,
        table_addr: Int64,
        width: Int32,
        stride_factor: Int32,
        layer_rows: Int32,
        row_base: Int32,
        rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        nthr = self.threads
        vecs = self.row_vecs
        r = b // self.splits
        part = b - r * self.splits
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        # The buffer's page 0: 16 zero bytes per thread, striding over the grid.
        first = b * nthr + tidx
        step = rows * self.splits * nthr
        passes = Int32(0)
        if first < self.zero_vecs:
            passes = (self.zero_vecs - first + step - 1) // step
        for p in cutlass.range(0, passes, unroll=1):
            _st_v4_zero(kv_addr + Int64(first + p * step) * 16)

        q = _ld_b32(req_addr + Int64(r) * 4)
        row_addr = Int64(r) * (self.topk * 4) + Int64(part) * (vecs * nthr * 16)
        vals = []
        for u in cutlass.range_constexpr(vecs):
            vals.extend(_ld_v4_b32(topk_addr + row_addr + Int64(u * nthr + tidx) * 16))
        table_row = table_addr + Int64(q) * Int64(width) * 4
        hs = Int32(0)
        hl = Int32(0)
        if cutlass.const_expr(self.history):
            hs = _ld_b32(start_addr + Int64(q) * 4)
            hl = _ld_b32(len_addr + Int64(q) * 4)
        # One read per entry, every one issued before any is used: a history entry's rank, a new
        # token's table entry, else the row's first table entry (always there; never used).
        got = []
        for j in cutlass.range_constexpr(4 * vecs):
            t = vals[j]
            addr = table_row
            if t >= 0:
                if (t >> self.shift) < width:
                    addr = table_row + Int64(t >> self.shift) * 4
                if cutlass.const_expr(self.history):
                    if t < hl:
                        addr = rank_addr + Int64(hs + t) * 4
            got.append(_ld_b32(addr))
        outs = []
        for j in cutlass.range_constexpr(4 * vecs):
            t = vals[j]
            g = got[j]
            o = Int32(0)
            if t >= 0:
                if (t >> self.shift) < width:
                    if g >= 0:
                        o = g * stride_factor + layer_rows + (t & self.mask)
                if cutlass.const_expr(self.history):
                    if t < hl:
                        o = g + row_base
            outs.append(o)
        for u in cutlass.range_constexpr(vecs):
            _st_v4_b32(
                out_addr + row_addr + Int64(u * nthr + tidx) * 16,
                outs[4 * u],
                outs[4 * u + 1],
                outs[4 * u + 2],
                outs[4 * u + 3],
            )


@functools.cache
def _compile_rows(
    topk: int,
    block_size: int,
    zero_bytes: int,
    row_bytes: int,
    history: bool,
    use_pdl: bool,
    row_vecs: int,
):
    def ints():
        return cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int(),), stride=(1,))

    def rows_of(dtype, width):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (cute.sym_int(), width), stride_order=(1, 0)
        )

    kernel = DkvTopkRowsKernel(topk, block_size, zero_bytes, history, use_pdl, row_vecs)
    return cute.compile(
        kernel,
        rows_of(cutlass.Int32, topk),
        ints(),
        ints(),
        ints(),
        ints(),
        rows_of(cutlass.Int32, topk),
        rows_of(cutlass.Uint8, row_bytes),
        Int64(0),
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(0),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


def _row_vecs(rows: int, topk: int, history: bool) -> int:
    """The index kernel's vectors per thread for ``rows`` Top-K rows (``_ROW_VECS_ROWS``), the
    largest the Top-K width allows up to it."""
    small, large = _ROW_VECS_ROWS
    vecs = 1 if rows < small else 2 if history or rows < large else 4
    while vecs > 1 and topk % (4 * vecs * _ROW_THREADS):
        vecs //= 2
    return vecs


def topk_rows_into(
    out: torch.Tensor,
    kv: torch.Tensor,
    zero_rows: int,
    topk: torch.Tensor,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    stride_factor: int,
    layer_id: int,
    hist_start: torch.Tensor,
    hist_len: torch.Tensor,
    rank: torch.Tensor | None,
    row_base: int,
    *,
    use_pdl: bool | None = None,
    row_vecs: int | None = None,
) -> torch.Tensor:
    """Write the read buffer's rows of ``topk`` (int32 ``[rows, K]``, contiguous, ``rows >= 1``)
    into ``out`` (the same) and zero ``kv``'s first ``zero_rows`` rows (uint8 ``[M, row_bytes]``,
    contiguous) in one launch; returns ``out``.

    Entry ``t`` of row ``r`` (request ``q = req_idx[r]``): with ``rank`` (int32, the history
    tokens' running count of selected ones) and ``0 <= t < hist_len[q]``, ``rank[hist_start[q] +
    t] + row_base``; else ``block_table[q, t // block_size] * stride_factor + layer_id *
    block_size + t % block_size`` (``block_table`` int32 ``[requests, width]``, contiguous); 0 for
    a negative ``t``, a page past the table or a negative table entry. ``rank`` None: no history.
    ``row_vecs``: the 16-byte vectors a thread handles (default ``_row_vecs``: by the row count and
    whether there is history).
    Compiles once per (K, block size, page-0 bytes, row width, history, PDL, row_vecs) on the
    first call.
    """
    rows, k = topk.shape
    row_bytes = kv.shape[1]
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    history = rank is not None
    if row_vecs is None:
        row_vecs = _row_vecs(rows, k, history)
    compiled = _compile_rows(
        k, block_size, zero_rows * row_bytes, row_bytes, history, use_pdl, row_vecs
    )
    compiled(
        topk,
        req_idx,
        hist_start,
        hist_len,
        rank if history else hist_len,
        out,
        kv,
        block_table.data_ptr(),
        block_table.shape[1],
        stride_factor,
        layer_id * block_size,
        row_base,
        rows,
    )
    return out


__all__ = ["DkvGatherRowsKernel", "DkvTopkRowsKernel", "gather_rows_into", "topk_rows_into"]
