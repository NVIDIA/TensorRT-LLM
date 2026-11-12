# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA's decode block top-k and token expansion in one CuTe DSL launch, a cluster of CTAs a row.

Each CTA holds a slice of the row's window in registers; radix passes (12, 11, 8 bits, top down)
narrow to the threshold class, every CTA adding its histogram into every CTA's copy through
distributed shared memory so the cluster narrows in step.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Consecutive scores a thread loads at once (one 16-byte load): a group.
KPT = 4
#: The radix passes: (first bit, bits). 31 magnitude bits, top down.
_PASSES = ((19, 12), (8, 11), (0, 8))
_HIST = 1 << 12
#: Most rows a launch serves (148 CTAs in clusters of 4); more take the family's one-CTA kernel.
MAX_ROWS = 37
#: Narrowest table this entry serves: narrower ones take the family's one-CTA kernel.
MIN_WIDTH = 4096 + KPT
#: Widest table (8192 + 128 pages): keeps a CTA's keys below 2^16, the width of its packed counts.
MAX_WIDTH = (8192 + 128) * 32 // 4
#: CTA size bounds (the scans give each thread at most `_HIST / _MIN_THREADS` bins).
_MAX_THREADS = 1024
_MIN_THREADS = 128


@dsl_user_op
def _red_add_cluster(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.cluster.shared::cluster.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _dsmem(ptr, rank):
    """The cluster-shared address of `ptr`'s location in CTA `rank` of the cluster."""
    return cutlass.Int32(cute.arch.map_dsmem_ptr(ptr, rank).toint())


def cluster_size(rows: int) -> int:
    """CTAs a row: 8 up to 12 rows (16 clusters of 8 overflow GB300's GPCs in one wave), else 4."""
    return 8 if rows <= 12 else 4


def geometry(width: int, cluster: int) -> tuple[int, int]:
    """`(threads, groups)` of a CTA: its `ceil(width / 4 / cluster)` groups over the fewest groups
    a thread 1024 threads allow (a group slot costs more than a wider CTA)."""
    q4 = -(-(-(-width // KPT)) // cluster)
    groups = max(1, -(-q4 // _MAX_THREADS))
    threads = max(_MIN_THREADS, -(-(-(-q4 // groups)) // 32) * 32)
    assert threads <= _MAX_THREADS, (width, cluster, threads)
    return threads, groups


class _TopkExpandClusterKernel:
    def __init__(self, *, width, key_stride, block_topk, compress, token_topk, pdl, cluster):
        self.width, self.key_stride, self.k, self.cr, self.token_topk, self.pdl = (
            width,
            key_stride,
            block_topk,
            compress,
            token_topk,
            pdl,
        )
        self.C = cluster
        self.threads, self.groups = geometry(width, cluster)
        self.warps = self.threads // 32
        self.final = token_topk + compress - 1

    def smem_bytes(self) -> int:
        words = _HIST + sum(1 << bits for _, bits in _PASSES) + 32 + 32 + 32 * self.groups + 16
        return words * 4 + 8 * 128

    @cute.jit
    def __call__(
        self,
        keys_t: cute.Tensor,
        end_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        len_ptr: cute.Pointer,
        topk_ptr: cute.Pointer,
        sel_ptr: cute.Pointer,
        rows: cutlass.Int32,
        stream,
    ):
        # Keys arrive as a tensor (the TVM-FFI entry takes its stream from one); the layout is
        # rebuilt with the static width and row stride the 16-byte loads need.
        mKeys = cute.make_tensor(
            keys_t.iterator, cute.make_layout((rows, self.width), stride=(self.key_stride, 1))
        )
        per_row = cute.make_layout((rows,), stride=(1,))
        mEnd = cute.make_tensor(end_ptr, per_row)
        mPos = cute.make_tensor(pos_ptr, per_row)
        mLen = cute.make_tensor(len_ptr, per_row)
        mTopk = cute.make_tensor(topk_ptr, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        mSel = cute.make_tensor(
            sel_ptr, cute.make_layout((rows, self.final), stride=(self.final, 1))
        )
        ld_keys = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), cutlass.Int32, num_bits_per_copy=32 * KPT
        )
        self.kernel(mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys).launch(
            grid=[rows * self.C, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.C, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _exclusive_scan(self, own, warp, lane, wsum):
        """`own`'s exclusive prefix over the CTA's threads in thread order (one barrier)."""
        incl = own
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        above = cutlass.Int32(0)
        if lane < warp:
            above = wsum[lane]
        return cute.arch.warp_reduction_sum(above) + incl - own

    @cute.jit
    def _narrow(
        self, hist, tidx, warp, lane, wsum, state, shift: cutlass.Constexpr, bits: cutlass.Constexpr
    ):
        """The cluster histogram of a pass read from the top bin down: the bin holding the
        `need`-th key of the class becomes the class. Every CTA reads the same counts."""
        nbins: cutlass.Constexpr = 1 << bits
        per: cutlass.Constexpr = -(-nbins // self.threads)
        prefix = state[0]
        mask = state[1]
        need = state[2]
        h = cute.make_rmem_tensor(cute.make_layout((per,), stride=(1,)), cutlass.Int32)
        own = cutlass.Int32(0)
        for j in cutlass.range_constexpr(per):
            b = nbins - 1 - (tidx * per + j)
            hv = cutlass.Int32(0)
            if b >= 0:
                hv = hist[b]
            h[j] = hv
            own = own + hv
        before = self._exclusive_scan(own, warp, lane, wsum)
        if before < need:
            if need <= before + own:
                run = before
                for j in cutlass.range_constexpr(per):
                    if run < need:
                        if need <= run + h[j]:
                            b = nbins - 1 - (tidx * per + j)
                            rem = need - run
                            state[0] = prefix | (b << shift)
                            state[1] = mask | ((nbins - 1) << shift)
                            state[2] = rem
                            state[3] = h[j]
                            state[4] = cutlass.Int32(
                                cutlass.select_((h[j] == rem) | (shift == 0), 1, 0)
                            )
                    run = run + h[j]
        cute.arch.sync_threads()

    @cute.jit
    def _pass(
        self,
        keys,
        tidx,
        warp,
        lane,
        local,
        merged,
        wsum,
        state,
        shift: cutlass.Constexpr,
        bits: cutlass.Constexpr,
        q4,
    ):
        """One radix pass unless narrowing is done: count this CTA's class keys, add the counts
        into every CTA's `merged`, cluster barrier, scan. Slot j holds keys only while
        `j * threads < q4`."""
        nbins: cutlass.Constexpr = 1 << bits
        if state[4] == 0:
            prefix = state[0]
            mask = state[1]
            for j in cutlass.range_constexpr(self.groups):
                if j * self.threads < q4:
                    for c in cutlass.range_constexpr(KPT):
                        kv = keys[c, j]
                        if kv >= 0:
                            if (kv & mask) == prefix:
                                cute.arch.atomic_add(
                                    local.iterator + ((kv >> shift) & (nbins - 1)),
                                    cutlass.Int32(1),
                                    scope="cta",
                                )
            cute.arch.sync_threads()
            for b in cutlass.range_constexpr(-(-nbins // self.threads)):
                i = tidx + b * self.threads
                if i < nbins:
                    v = local[i]
                    if v != 0:
                        local[i] = cutlass.Int32(0)
                        for r in cutlass.range_constexpr(self.C):
                            _red_add_cluster(_dsmem(merged.iterator + i, r), v)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            self._narrow(merged, tidx, warp, lane, wsum, state, shift, bits)

    @cute.kernel
    def kernel(self, mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys: cute.CopyAtom):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        rank = cute.arch.block_idx_in_cluster()
        C: cutlass.Constexpr = self.C
        nt: cutlass.Constexpr = self.threads
        nw: cutlass.Constexpr = self.warps
        G: cutlass.Constexpr = self.groups
        k: cutlass.Constexpr = self.k
        cr: cutlass.Constexpr = self.cr
        row = bidx // C
        smem = cutlass.utils.SmemAllocator()
        local = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((_HIST,), stride=(1,)), byte_alignment=128
        )
        merged0 = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1 << _PASSES[0][1],), stride=(1,)), byte_alignment=128
        )
        merged1 = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1 << _PASSES[1][1],), stride=(1,)), byte_alignment=128
        )
        merged2 = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1 << _PASSES[2][1],), stride=(1,)), byte_alignment=128
        )
        ex = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        wsum = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        wtot = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32 * G,), stride=(1,)), byte_alignment=128
        )
        state = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((16,), stride=(1,)), byte_alignment=128
        )

        # 0. The histograms zeroed and the row's host-filled metadata read before the wait;
        # the arrive releases the zeroed counts before any peer adds into them.
        for b in cutlass.range_constexpr(-(-_HIST // nt)):
            if tidx + b * nt < _HIST:
                local[tidx + b * nt] = cutlass.Int32(0)
                merged0[tidx + b * nt] = cutlass.Int32(0)
        for b in cutlass.range_constexpr(-(-(1 << _PASSES[1][1]) // nt)):
            if tidx + b * nt < (1 << _PASSES[1][1]):
                merged1[tidx + b * nt] = cutlass.Int32(0)
        for b in cutlass.range_constexpr(-(-(1 << _PASSES[2][1]) // nt)):
            if tidx + b * nt < (1 << _PASSES[2][1]):
                merged2[tidx + b * nt] = cutlass.Int32(0)
        n = mEnd[row]
        seqlen = mLen[row]
        visible = cutlass.Int32(mPos[row]) + 1
        if tidx == 0:
            state[0] = cutlass.Int32(0)  # the class's fixed bits
            state[1] = cutlass.Int32(0)  # their mask
            state[2] = cutlass.min(n, cutlass.Int32(k))  # still needed from the class
            state[3] = n  # keys in the class
            state[4] = cutlass.Int32(cutlass.select_(n <= k, 1, 0))  # narrowing done
        cute.arch.cluster_arrive()
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # 1. This CTA's slice of the window: groups [g_lo, g_hi) of `KPT` consecutive keys,
        # group `g_lo + j * nt + tidx` in slot j of thread tidx; -1 past the window.
        n4 = (n + (KPT - 1)) // KPT
        q4 = (n4 + (C - 1)) // C
        g_lo = rank * q4
        g_hi = cutlass.min(g_lo + q4, n4)
        gK = cute.logical_divide(mKeys, (None, KPT))
        keys = cute.make_rmem_tensor(cute.make_layout((KPT, G), stride=(1, KPT)), cutlass.Int32)
        keys.fill(-1)
        for j in cutlass.range_constexpr(G):
            gl = g_lo + j * nt + tidx
            if gl < g_hi:
                cute.copy(ld_keys, gK[row, (None, gl)], keys[None, j])
            for c in cutlass.range_constexpr(KPT):
                v = cutlass.max(keys[c, j], cutlass.Int32(0))
                if (gl >= g_hi) | (gl * KPT + c >= n):
                    v = cutlass.Int32(-1)
                keys[c, j] = v
        cute.arch.cluster_wait()

        self._pass(
            keys, tidx, warp, lane, local, merged0, wsum, state, _PASSES[0][0], _PASSES[0][1], q4
        )
        self._pass(
            keys, tidx, warp, lane, local, merged1, wsum, state, _PASSES[1][0], _PASSES[1][1], q4
        )
        self._pass(
            keys, tidx, warp, lane, local, merged2, wsum, state, _PASSES[2][0], _PASSES[2][1], q4
        )
        prefix = state[0]
        mask = state[1]
        need = state[2]

        # 3. Block order within the CTA (slot j of every thread, then slot j + 1): each
        # (thread, slot)'s counts above and in the class, packed (above + class << 16), and their
        # exclusive prefix. Slots past `q4` hold no keys in any thread (uniform skip).
        pre = cute.make_rmem_tensor(cute.make_layout((G,), stride=(1,)), cutlass.Int32)
        pre.fill(0)
        for j in cutlass.range_constexpr(G):
            if j * nt < q4:
                a = cutlass.Int32(0)
                for c in cutlass.range_constexpr(KPT):
                    kv = keys[c, j]
                    if kv >= 0:
                        mk = kv & mask
                        if mk > prefix:
                            a = a + 1
                        if mk == prefix:
                            a = a + (1 << 16)
                incl = a
                for off in (1, 2, 4, 8, 16):
                    up = cute.arch.shuffle_sync_up(incl, offset=off)
                    if lane >= off:
                        incl = incl + up
                if lane == 31:
                    wtot[j * 32 + warp] = incl
                pre[j] = incl - a
        cute.arch.sync_threads()
        run = cutlass.Int32(0)
        for j in cutlass.range_constexpr(G):
            if j * nt < q4:
                v = cutlass.Int32(0)
                if lane < nw:
                    v = wtot[j * 32 + lane]
                above = cutlass.Int32(0)
                if lane < warp:
                    above = v
                pre[j] = pre[j] + run + cute.arch.warp_reduction_sum(above)
                run = run + cute.arch.warp_reduction_sum(v)

        # 4. Every CTA's totals into every CTA (lane r of warp 0 writes CTA r's slot `rank`).
        if tidx < C:
            _st_cluster(_dsmem(ex.iterator + rank, tidx), run)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # The class's `need` are its highest blocks, so later CTAs fill first. `base`: selected
        # count of the CTAs before this one; `skip`: this CTA's lowest class keys left out.
        base = cutlass.Int32(0)
        skip = cutlass.Int32(0)
        later = cutlass.Int32(0)
        for rr in cutlass.range_constexpr(C):
            r = C - 1 - rr
            tot = ex[r]
            e_r = tot >> 16
            take = cutlass.max(cutlass.min(need - later, e_r), cutlass.Int32(0))
            later = later + e_r
            if r < rank:
                base = base + (tot & 0xFFFF) + take
            if r == rank:
                skip = e_r - take

        # 5. Each selected block at its position, with its tokens below the sequence length.
        for j in cutlass.range_constexpr(G):
            gl = g_lo + j * nt + tidx
            ab = pre[j] & 0xFFFF
            cb = pre[j] >> 16
            for c in cutlass.range_constexpr(KPT):
                kv = keys[c, j]
                if kv >= 0:
                    mk = kv & mask
                    sel = cutlass.Int32(0)
                    pos = cutlass.Int32(0)
                    if mk > prefix:
                        sel = cutlass.Int32(1)
                        pos = base + ab + cutlass.max(cb - skip, cutlass.Int32(0))
                        ab = ab + 1
                    if mk == prefix:
                        if cb >= skip:
                            sel = cutlass.Int32(1)
                            pos = base + ab + cb - skip
                        cb = cb + 1
                    if sel == 1:
                        col = gl * KPT + c
                        mTopk[row, pos] = col
                        for tt in cutlass.range_constexpr(cr):
                            tok = col * cr + tt
                            val = cutlass.Int32(-1)
                            if tok < seqlen:
                                val = tok
                            mSel[row, pos * cr + tt] = val

        # 6. Over the cluster's threads: the block list's -1 padding, then the causal tail and -1.
        total = cutlass.min(n, cutlass.Int32(k))
        for b in cutlass.range_constexpr(-(-k // (C * nt))):
            slot = (b * C + rank) * nt + tidx
            if slot < k:
                if slot >= total:
                    mTopk[row, slot] = cutlass.Int32(-1)
        tail_start = (visible // cr) * cr
        tail_count = visible - tail_start
        body = cutlass.min(total * cr, cutlass.Int32(self.token_topk))
        for b in cutlass.range_constexpr(-(-self.final // (C * nt))):
            col = (b * C + rank) * nt + tidx
            if col < self.final:
                if col >= body:
                    off = col - body
                    val = cutlass.Int32(-1)
                    if off < cr - 1:
                        if off < tail_count:
                            if tail_start + off < seqlen:
                                val = tail_start + off
                    mSel[row, col] = val


#: Compiled launches; the row count is a runtime argument.
_COMPILED: dict = {}
_INDEX_TYPES = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


def launch(
    scores,
    row_ends,
    query_positions,
    sequence_lengths,
    topk_indices,
    compress_ratio,
    token_topk,
    out=None,
):
    """Writes `topk_indices`; returns the int32 token selection (`out` when given)."""
    rows, width = scores.shape
    block_topk = topk_indices.shape[1]
    sel = out
    if sel is None:
        sel = torch.empty(
            (rows, token_topk + compress_ratio - 1), dtype=torch.int32, device=scores.device
        )
    if rows == 0:
        return sel
    pdl = bool(get_env_enable_pdl())
    keys = scores.view(torch.int32)
    cluster = cluster_size(rows)
    key = (
        width,
        scores.stride(0),
        block_topk,
        compress_ratio,
        token_topk,
        query_positions.dtype,
        sequence_lengths.dtype,
        pdl,
        cluster,
    )
    ptrs = (
        row_ends.data_ptr(),
        query_positions.data_ptr(),
        sequence_lengths.data_ptr(),
        topk_indices.data_ptr(),
        sel.data_ptr(),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        kernel = _TopkExpandClusterKernel(
            width=width,
            key_stride=scores.stride(0),
            block_topk=block_topk,
            compress=compress_ratio,
            token_topk=token_topk,
            pdl=pdl,
            cluster=cluster,
        )
        fn = _COMPILED[key] = cute.compile(
            kernel,
            from_dlpack(keys, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.Int32, ptrs[0], gm, assumed_align=4),
            make_ptr(
                _INDEX_TYPES[query_positions.dtype],
                ptrs[1],
                gm,
                assumed_align=query_positions.element_size(),
            ),
            make_ptr(
                _INDEX_TYPES[sequence_lengths.dtype],
                ptrs[2],
                gm,
                assumed_align=sequence_lengths.element_size(),
            ),
            make_ptr(cutlass.Int32, ptrs[3], gm, assumed_align=4),
            make_ptr(cutlass.Int32, ptrs[4], gm, assumed_align=4),
            cutlass.Int32(rows),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    fn(keys, *ptrs, rows)
    return sel


def supports(scores) -> bool:
    """Whether this entry serves a decode call over this score table (see `is_valid`)."""
    rows, width = scores.shape
    return 0 < rows <= MAX_ROWS and width % KPT == 0 and MIN_WIDTH <= width <= MAX_WIDTH


#: Score-table widths: 262144 tokens with / without MTP's extra pages, and 131072 tokens.
_TUNE_WIDTH = 65568
_TUNE_WIDTH_NO_MTP = 65536
_HALF_WIDTH = 32800
#: Decode windows of 0, 1, 2 and 9 blocks, k - 1, k, k + 1, the whole table, and between.
_EDGES_16 = (
    2,
    3,
    11,
    35,
    2047,
    2051,
    2055,
    16380,
    16388,
    99999,
    180224,
    262143,
    50000,
    150001,
    220000,
    7,
)


class QsaTopkExpandClusterDecode(OpWrapper):
    """`qsa_topk_select` (zero row starts), `qsa_canonical_selection` and
    `qsa_expand_block_indices` in one launch, for decode-sized calls over a wide score table.

    Per row: the `block_topk` largest of `scores[row, :row_ends[row]]` (ties to the higher
    block) into `topk_indices[row]` in block order, -1 padded, in place. Returns the token
    selection (int32 `[rows, token_topk + compress_ratio - 1]`, into `out` when given): selected
    groups' tokens below `sequence_lengths[row]`, then the causal tail, then -1.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="ll_tep4 draft step: 1 row at ~96k tokens, 65568 wide, relu-sum, graph",
            spec=dict(rows=1, pos=(96003,), relu_sum=True, width=_TUNE_WIDTH, graph=True),
        ),
        Cell(
            why="37 rows (MAX_ROWS, 148 CTAs in clusters of 4): 9 requests x 4 verify rows "
            "spread to ~200k tokens and a draft row",
            spec=dict(
                rows=37,
                requests=tuple((1000 + 24000 * r, 4) for r in range(9)) + ((131072, 1),),
                relu_sum=True,
                width=_TUNE_WIDTH,
            ),
        ),
        Cell(
            why="4 rows of bf16-resolution log-normal scores: exact ties at the threshold, all "
            "three passes, the tie rule across a cluster's CTAs",
            spec=dict(rows=4, pos=(100000, 30000, 4095, 2047), lognormal=True, width=_TUNE_WIDTH),
        ),
        Cell(
            why="2 relu-sparse rows (~1.4% positive, negative zeros among the zeros): the zero "
            "class straddles the budget across CTAs",
            spec=dict(rows=2, pos=(262143, 120000), relu_sparse=True, width=_TUNE_WIDTH_NO_MTP),
        ),
        Cell(
            why="3 rows of equal scores (windows 25000, 513 and 512 blocks): the tie class is "
            "the window, the highest blocks taken",
            spec=dict(rows=3, pos=(100000, 2051, 2047), equal=True, width=_TUNE_WIDTH),
        ),
        Cell(
            why="20 rows (clusters of 4) at the edges of k and the table, relu-sum, 65536 wide",
            spec=dict(
                rows=20,
                pos=_EDGES_16 + (262140, 4, 2048, 131071),
                relu_sum=True,
                width=_TUNE_WIDTH_NO_MTP,
            ),
        ),
        Cell(
            why="a 131072-token engine's 32800-wide table: 4 verify rows up to its last position",
            spec=dict(
                rows=4, requests=((131068, 4),), relu_sum=True, width=_HALF_WIDTH, graph=True
            ),
        ),
    )

    note = """
    Source: this target's CuTe DSL kernel, `cute.compile`d (`--enable-tvm-ffi`) on first call
    per table width and row stride, k, compress ratio, token budget, index dtypes, PDL and
    cluster size.

    Preconditions: zero row starts; `scores` fp32 `[rows, W]`, unit column stride, 16-byte
    aligned rows, `W % 4 == 0`; `topk_indices` int32 `[rows, K]` contiguous; `row_ends` int32
    (each at most `W`), positions and lengths int32 or int64, each a contiguous `[rows]`; the
    call within `supports` (1..37 rows, 4100 <= W <= 66560). Ties go to the higher block; a
    negative zero ranks as zero. Limits: rows x row stride below 2^31, positions read as Int32.

    Not certified: NaN scores inside the window (they rank above every finite score).
    """

    def __call__(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        return launch(
            scores,
            row_ends,
            query_positions,
            sequence_lengths,
            topk_indices,
            compress_ratio,
            token_topk,
            out=out,
        )

    def reference(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        """`(topk_indices_after, selection)` via a stable descending sort per row; ignores `out`."""
        rows, _ = scores.shape
        k = topk_indices.shape[1]
        device = scores.device
        blocks = torch.full((rows, k), -1, dtype=torch.int64, device=device)
        for r in range(rows):
            n = int(row_ends[r])
            s = scores[r, :n].float().clamp(min=0.0)
            # A stable sort of the reversed row puts the higher block first among equals.
            order = (n - 1) - torch.sort(torch.flip(s, [0]), descending=True, stable=True).indices[
                :k
            ]
            chosen = torch.sort(order).values
            blocks[r, : chosen.numel()] = chosen
        final = token_topk + compress_ratio - 1
        lengths = sequence_lengths.long()
        offsets = torch.arange(compress_ratio, device=device)
        expanded = (blocks[:, :, None] * compress_ratio + offsets).reshape(rows, token_topk)
        chosen = (blocks >= 0).repeat_interleave(compress_ratio, dim=1)
        keep = chosen & (expanded < lengths[:, None])
        sel = torch.full((rows, final), -1, dtype=torch.int64, device=device)
        sel[:, :token_topk] = torch.where(keep, expanded, -1)
        first_free = (blocks >= 0).sum(dim=1) * compress_ratio
        visible = query_positions.long() + 1
        tail_start = visible // compress_ratio * compress_ratio
        row = torch.arange(rows, device=device)
        for offset in range(compress_ratio - 1):
            column = first_free + offset
            token = tail_start + offset
            live = (offset < visible - tail_start) & (token < lengths) & (column < final)
            sel[row[live], column[live]] = token[live]
        return blocks.to(torch.int32), sel.to(torch.int32)

    def is_valid(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        assert scores.dtype == torch.float32 and scores.stride(1) == 1, (
            "fp32 scores, unit column stride"
        )
        assert scores.shape[1] % KPT == 0 and scores.shape[1] <= MAX_WIDTH, "W % 4 == 0, W <= 66560"
        assert scores.stride(0) % KPT == 0 and scores.data_ptr() % 16 == 0, "16-byte aligned rows"
        assert topk_indices.dtype == torch.int32 and topk_indices.is_contiguous(), (
            "int32 topk_indices"
        )
        rows = scores.shape[0]
        assert row_ends.dtype == torch.int32 and row_ends.shape == (rows,), "int32 row_ends [rows]"
        for t in (query_positions, sequence_lengths):
            assert t.dtype in _INDEX_TYPES and t.shape == (rows,), "int32/int64 [rows]"
        assert all(t.is_contiguous() for t in (row_ends, query_positions, sequence_lengths)), (
            "contiguous row_ends, positions and lengths"
        )
        assert token_topk == topk_indices.shape[1] * compress_ratio, (
            "token_topk = k * compress_ratio"
        )
        assert out is None or (
            out.dtype == torch.int32
            and out.is_contiguous()
            and out.shape == (scores.shape[0], token_topk + compress_ratio - 1)
        ), "out: contiguous int32 [rows, token_topk + compress_ratio - 1]"


qsa_topk_expand_cluster = QsaTopkExpandClusterDecode()
