# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA top-k rows in ascending key order, remapped to MLA KV-pool rows, in one launch (CuTe DSL).

Row ``r`` of ``token_indices`` holds up to ``K`` distinct request-local key positions (``>= 0``)
and unused slots (``< 0``) in any order. The kernel writes ``out[r]``: the positions in ascending
order, then ``-1`` for every unused slot, each position ``tok`` mapped to its KV pool row
``block_table[req_id[r], tok >> shift] * stride_factor + layer_term + (tok & mask)``
(``-1`` when the page is past the table row or the table entry is negative). That is
``deterministic_topk_order`` followed by ``convert_req_index_to_global`` bit for bit: the order
of a row is a function of its set only, so the sparse MLA accumulates the keys in a fixed order
whatever order the top-k kernel wrote them in.

One CTA of ``K / 4`` threads per row, each thread loading 4 slots and starting the block-table
loads of their pool rows at once. Positions are ranked by counting: they set bits of a
shared-memory bitmap of ``CAP_WORDS`` words (``32 * CAP_WORDS`` positions per segment; rows with
larger positions take several segments), each thread counts the bits of a contiguous chunk of
words, one block-wide scan gives every word its exclusive prefix, and each position's rank is its
word's prefix plus the set bits below it. The (remapped) values are placed at their ranks in
shared memory and stored with 16-byte stores. A duplicate position in a row would land once
(top-k rows have none).
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Words of the shared-memory bitmap of one segment (32 KiB: 262144 positions).
CAP_WORDS = 8192
_CAP_SHIFT = 18
assert CAP_WORDS * 32 == 1 << _CAP_SHIFT


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


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ldg_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ldg_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4_zero(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "st.shared.v4.b32 [$0], {0, 0, 0, 0};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _red_shared_or(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_iv(value, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _warp_inclusive_sum(value, *, loc=None, ip=None):
    body = "".join(
        f"shfl.sync.up.b32 t|p, x, {off}, 0, 0xffffffff;\n@p add.s32 x, x, t;\n"
        for off in (1, 2, 4, 8, 16)
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\n.reg .pred p;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_max_s32(value, *, loc=None, ip=None):
    body = "".join(
        f"shfl.sync.bfly.b32 t, x, {off}, 0x1f, 0xffffffff;\nmax.s32 x, x, t;\n"
        for off in (16, 8, 4, 2, 1)
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


class DsaTopkOrderRemapKernel:
    """``out[r, :]`` = row ``r`` of the top-k, ascending, ``-1`` slots last, remapped. A PDL
    launch: the loads of the row wait for the previous grid, then the dependents are released."""

    def __init__(self, top_k: int):
        if top_k % 128 != 0 or not 128 <= top_k <= 4096:
            raise ValueError(f"top_k must be a multiple of 128 in [128, 4096], got {top_k}")
        self.top_k = top_k
        self.threads = top_k // 4
        self.num_warps = self.threads // 32
        # One bitmap clear: CAP_WORDS words in 16-byte stores.
        assert CAP_WORDS % (4 * self.threads) == 0
        self.clear_iters = CAP_WORDS // (4 * self.threads)

        @cute.struct
        class SharedStorage:
            bitmap: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, CAP_WORDS], 16]
            prefix: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, CAP_WORDS], 16]
            sorted: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, top_k], 16]
            warp_max: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]
            warp_cnt: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        rows: Int32,
        topk_addr: Int64,
        topk_stride: Int32,
        out_addr: Int64,
        req_addr: Int64,
        table_addr: Int64,
        table_stride: Int32,
        table_width: Int32,
        shift: Int32,
        page_mask: Int32,
        stride_factor: Int64,
        layer_term: Int64,
        rows_per_req: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            topk_addr,
            topk_stride,
            out_addr,
            req_addr,
            table_addr,
            table_stride,
            table_width,
            shift,
            page_mask,
            stride_factor,
            layer_term,
            rows_per_req,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        topk_addr: Int64,
        topk_stride: Int32,
        out_addr: Int64,
        req_addr: Int64,
        table_addr: Int64,
        table_stride: Int32,
        table_width: Int32,
        shift: Int32,
        page_mask: Int32,
        stride_factor: Int64,
        layer_term: Int64,
        rows_per_req: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bm = storage.bitmap.data_ptr().toint()
        pf = storage.prefix.data_ptr().toint()
        so = storage.sorted.data_ptr().toint()
        wmax = storage.warp_max.data_ptr().toint()
        wcnt = storage.warp_cnt.data_ptr().toint()
        nw_cta = self.num_warps
        nthr = self.threads

        for i in cutlass.range_constexpr(self.clear_iters):
            _sts_v4_zero(bm + (i * nthr + tidx) * 16)
        cute.arch.barrier()
        _griddepcontrol_wait()
        row_addr = topk_addr + (Int64(r) * Int64(topk_stride) + Int64(tidx * 4)) * Int64(4)
        a0, a1, a2, a3 = _ldg_v4_b32(row_addr)
        _griddepcontrol_launch_dependents()
        vals = (a0, a1, a2, a3)
        # Block-table entries of this thread's positions, loaded now and used only when the
        # ranks are known, so the loads overlap the sort.
        entries = [a0, a1, a2, a3]
        # Request of the row: req_id[r], or r // rows_per_req without a req_id tensor.
        req = r // rows_per_req
        if req_addr != Int64(0):
            req = _ldg_b32(req_addr + Int64(r) * Int64(4))
        for q in cutlass.range_constexpr(4):
            e = Int32(-1)
            if vals[q] >= 0:
                if (vals[q] >> shift) < table_width:
                    e = _ldg_b32(
                        table_addr
                        + (Int64(req) * Int64(table_stride) + Int64(vals[q] >> shift)) * Int64(4)
                    )
            entries[q] = e
        # Segment 0's bits go in before the first barrier.
        for q in cutlass.range_constexpr(4):
            d = vals[q]
            if d >= 0 and d < (1 << _CAP_SHIFT):
                _red_shared_or(bm + (d >> 5) * 4, Int32(1) << (d & 31))
        m = a0
        for q in cutlass.range_constexpr(1, 4):
            if vals[q] > m:
                m = vals[q]
        m = _warp_max_s32(m)
        if lane == 0:
            _sts_b32(wmax + warp * 4, m)
        cute.arch.barrier()
        top = _lds_b32(wmax)
        for w in cutlass.range_constexpr(1, nw_cta):
            x = _lds_b32(wmax + w * 4)
            if x > top:
                top = x
        nseg = Int32(0)
        if top >= 0:
            nseg = (top >> _CAP_SHIFT) + 1

        total = Int32(0)
        for seg in cutlass.range(nseg, unroll=1):
            lo = seg << _CAP_SHIFT
            if seg > 0:
                for i in cutlass.range_constexpr(self.clear_iters):
                    _sts_v4_zero(bm + (i * nthr + tidx) * 16)
                cute.arch.barrier()
                for q in cutlass.range_constexpr(4):
                    d = vals[q] - lo
                    if d >= 0 and d < (1 << _CAP_SHIFT):
                        _red_shared_or(bm + (d >> 5) * 4, Int32(1) << (d & 31))
                cute.arch.barrier()
            # Each thread owns a contiguous chunk of the words that can hold a set bit.
            nwords = ((top - lo) >> 5) + 1
            if nwords > CAP_WORDS:
                nwords = Int32(CAP_WORDS)
            chunk = (nwords + nthr - 1) // nthr
            w0 = tidx * chunk
            w1 = w0 + chunk
            if w1 > nwords:
                w1 = nwords
            cnt = Int32(0)
            for wi in cutlass.range(w0, w1, unroll=1):
                cnt = cnt + _popc(_lds_b32(bm + wi * 4))
            incl = _warp_inclusive_sum(cnt)
            if lane == 31:
                _sts_b32(wcnt + warp * 4, incl)
            cute.arch.barrier()
            run = total + incl - cnt
            seg_total = Int32(0)
            for w in cutlass.range_constexpr(nw_cta):
                c = _lds_b32(wcnt + w * 4)
                if w < warp:
                    run = run + c
                seg_total = seg_total + c
            # Exclusive prefix of every owned word (positions before it in the row).
            for wi in cutlass.range(w0, w1, unroll=1):
                _sts_b32(pf + wi * 4, run)
                run = run + _popc(_lds_b32(bm + wi * 4))
            cute.arch.barrier()
            # Each position's rank = its word's prefix + the set bits below it in the word.
            for q in cutlass.range_constexpr(4):
                d = vals[q] - lo
                if d >= 0 and d < (1 << _CAP_SHIFT):
                    word = _lds_b32(bm + (d >> 5) * 4)
                    below = word & ((Int32(1) << (d & 31)) - 1)
                    rank = _lds_b32(pf + (d >> 5) * 4) + _popc(below)
                    o = Int32(-1)
                    if entries[q] >= 0:
                        o = Int32(
                            Int64(entries[q]) * stride_factor
                            + layer_term
                            + Int64(vals[q] & page_mask)
                        )
                    _sts_b32(so + rank * 4, o)
            total = total + seg_total
            cute.arch.barrier()

        outs = []
        for i in cutlass.range_constexpr(4):
            res = Int32(-1)
            if tidx * 4 + i < total:
                res = _lds_b32(so + (tidx * 4 + i) * 4)
            outs.append(res)
        _stg_v4_b32(
            out_addr + (Int64(r) * Int64(self.top_k) + Int64(tidx * 4)) * Int64(4),
            outs[0],
            outs[1],
            outs[2],
            outs[3],
        )


@functools.cache
def _compiled(top_k: int):
    kernel = DsaTopkOrderRemapKernel(top_k)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(1),
        a,
        Int32(top_k),
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(6),
        Int32(63),
        Int64(1),
        Int64(0),
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def supports(token_indices: torch.Tensor) -> bool:
    """Whether the kernel serves these top-k rows: int32 CUDA ``[rows, K]`` with ``K`` a multiple
    of 128 in [128, 4096], unit inner stride and 16-byte aligned rows."""
    if token_indices.dim() != 2 or token_indices.dtype != torch.int32 or not token_indices.is_cuda:
        return False
    rows, k = token_indices.shape
    return (
        k % 128 == 0
        and 128 <= k <= 4096
        and (token_indices.stride(1) == 1 or k == 1)
        and (rows <= 1 or token_indices.stride(0) % 4 == 0)
        and token_indices.data_ptr() % 16 == 0
    )


def dsa_topk_order_remap_into(
    token_indices: torch.Tensor,
    out: torch.Tensor,
    req_id: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    stride_factor: int,
    layer_id: int,
    rows_per_req: int = 1,
) -> torch.Tensor:
    """Write the ordered and remapped rows of ``token_indices`` into ``out`` (int32 contiguous
    ``[rows, K]``) on the current stream and return ``out``.

    ``block_table`` int32 ``[requests, width]`` (unit inner stride), with row ``r``'s request
    ``req_id[r]`` (int32 ``[rows]``) or, when ``req_id`` is None, ``r // rows_per_req``;
    ``block_size`` must be a power of two. ``out`` may be ``token_indices`` itself (each CTA reads
    its row before it writes it). The first call per ``K`` compiles and must run outside
    CUDA-graph capture.
    """
    rows, k = token_indices.shape
    if rows == 0:
        return out
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    req_addr = req_id.data_ptr() if req_id is not None else 0
    table_addr = block_table.data_ptr()
    table_stride, table_width = block_table.stride(0), block_table.shape[1]
    shift = block_size.bit_length() - 1
    compiled = _compiled(k)
    compiled(
        Int32(rows),
        Int64(token_indices.data_ptr()),
        Int32(token_indices.stride(0) if rows > 1 else k),
        Int64(out.data_ptr()),
        Int64(req_addr),
        Int64(table_addr),
        Int32(table_stride),
        Int32(table_width),
        Int32(shift),
        Int32(block_size - 1),
        Int64(stride_factor),
        Int64(layer_id * block_size),
        Int32(max(1, rows_per_req)),
        cuda.CUstream(torch.cuda.current_stream(token_indices.device).cuda_stream),
    )
    return out


__all__ = ["CAP_WORDS", "DsaTopkOrderRemapKernel", "dsa_topk_order_remap_into", "supports"]
