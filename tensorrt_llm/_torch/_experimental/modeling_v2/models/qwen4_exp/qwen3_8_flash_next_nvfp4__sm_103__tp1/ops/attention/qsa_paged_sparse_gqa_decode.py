# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA for decode rows: a split-K pass that loads its tokens in one round trip,
then a merge spread over the SMs.

Tier 5 (CuTe DSL written for this target), both launches.

Split-K pass. One CTA (eight warps) covers `CAP` consecutive selected tokens of one KV
head of one row, so a decode row's 2051 selected tokens take 33 CTAs a KV head:

1. before its PDL wait the CTA stages the KV head's query heads in shared memory and from
   there in every warp's MMA A fragments, stages the request's block-table row (up to
   `BT_STAGE` pages) in shared memory, and releases its dependents;
2. after it, its tokens' pages come from shared memory and every token's K and V row
   (16 bytes a lane, a warp eight tokens) is requested at once with cp.async, so the CTA
   pays one DRAM round trip -- QSA's split-K kernel walks two or three 16-token tiles,
   each a chain of dependent loads (selection, page, K/V);
3. scores = q K^T on warp MMAs (bf16 in, fp32 out; a warp eight tokens), scaled into the
   log2 domain and masked; one softmax over the CTA's tokens (a warp two heads); P V on
   warp MMAs with P in bf16 (a warp 32 head dims); each head's partial times its
   reciprocal sum is staged in shared memory and stored in 16-byte pieces with its
   log2-sum-exp, as QSA's split-K pass stores them.

Merge. A CTA covers 128 elements of a head (two CTAs a head) with up to eight warps: warp
`w` reads its splits' partials of four adjacent elements a lane, one 16-byte load a split,
all in flight; every warp forms the split weights itself from two LSEs a lane; the warps'
shares are added in shared memory.

Both launches release their dependents at entry, before their own waits: the merge and,
through it, the output projection that follows launch while the top-k ahead of the
split-K pass still runs, and the projection streams its weight through that latency-bound
stretch. Before its wait the split-K pass reads q and the merge the output gate: the
caller's contract is that both are complete when the grid ahead of the split-K pass
releases it (the target's top-k does so only after its own wait; a launch without PDL
releases at its end).
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack, make_ptr

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Approved exception (as for `qsa_paged_sparse_gqa`): QSA's sparse path has no torch.ops
# surface; its row-count PDL policy is reached as Python.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import _qsa_pdl_enabled

from .qsa_paged_sparse_gqa import qsa_paged_sparse_gqa as _composition

#: Output elements a merge CTA covers (four adjacent ones a lane of each warp).
CHUNK = 128
#: Splits the merge reads: two LSEs a lane.
MAX_SPLITS = 64
#: Rows the target calls this entry with: the launches that run with PDL (QSA enables it up
#: to four rows); larger batches keep the composition.
MAX_ROWS = 4
#: Selected tokens a split-K CTA covers.
CAP = 64
#: MMA rows: the query heads of a KV head, padded.
M = 16
#: bf16 elements of padding after each shared K, V, q and P row (keeps the MMA fragment
#: reads free of bank conflicts).
PAD = 8
WARPS = 8
THREADS = WARPS * 32
#: Block-table entries staged in shared memory before the wait; a wider table is read
#: from global memory after it.
BT_STAGE = 4096
HEAD_DIM = 256
LOG2E = 1.4426950408889634


class _SplitKKernel:
    def __init__(
        self,
        *,
        group,
        width,
        splits,
        tpb,
        scale,
        bt_width,
        num_pages,
        k_strides,
        v_strides,
        num_requests,
        pdl,
    ):
        assert group <= M and CAP == 2 * 32 and CAP % WARPS == 0
        self.group, self.width, self.splits, self.tpb = group, width, splits, tpb
        self.scale = scale * LOG2E
        self.bt_width, self.num_pages, self.num_requests = bt_width, num_pages, num_requests
        self.stage_bt = bt_width <= BT_STAGE
        self.k_strides, self.v_strides = k_strides, v_strides
        self.ld = HEAD_DIM + PAD
        self.pdl = pdl

    def smem_bytes(self) -> int:
        def r(n):
            return -(-n // 128) * 128

        b = (
            r(M * self.ld * 2)
            + 2 * r(CAP * self.ld * 2)
            + r(M * (CAP + PAD) * 2)
            + r(M * (CAP + 4) * 4)
        )
        b += r(M * 2 * 4) + 2 * r(CAP * 8)
        if self.stage_bt:
            b += r(self.bt_width * 4)
        return b + 256

    @cute.jit
    def __call__(self, mQ, pK, pV, mBT, mSel, mReq, pPart, mLse, stream: cuda_drv.CUstream):
        rows = cute.size(mSel, mode=[0])
        kv_heads = cute.size(mQ, mode=[1]) // self.group
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            cute.make_layout((1, WARPS, 1)),
        )
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        st = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        self.kernel(mQ, pK, pV, mBT, mSel, mReq, pPart, mLse, mma, g2s, st).launch(
            grid=[rows * kv_heads * self.splits, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        pK,
        pV,
        mBT,
        mSel,
        mReq,
        pPart,
        mLse,
        mma: cute.TiledMma,
        g2s: cute.CopyAtom,
        st: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        group: cutlass.Constexpr = self.group
        splits: cutlass.Constexpr = self.splits
        width: cutlass.Constexpr = self.width
        tpb: cutlass.Constexpr = self.tpb
        bt_width: cutlass.Constexpr = self.bt_width
        ld: cutlass.Constexpr = self.ld
        dim: cutlass.Constexpr = HEAD_DIM
        heads = cute.size(mQ, mode=[1])
        kv_heads = heads // group
        rows = cute.size(mSel, mode=[0])
        row = bidx // (kv_heads * splits)
        kvh = (bidx // splits) % kv_heads
        split = bidx % splits
        c0 = split * CAP

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, dim), stride=(ld, 1)), byte_alignment=128
        )
        sK = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((CAP, dim), stride=(ld, 1)), byte_alignment=128
        )
        sV = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((CAP, dim), stride=(ld, 1)), byte_alignment=128
        )
        sP = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, CAP), stride=(CAP + PAD, 1)), byte_alignment=128
        )
        sS = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, CAP), stride=(CAP + 4, 1)), byte_alignment=128
        )
        sStat = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, 2), stride=(2, 1)), byte_alignment=128
        )
        sKo = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((CAP,), stride=(1,)), byte_alignment=128
        )
        sVo = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((CAP,), stride=(1,)), byte_alignment=128
        )

        # 1. Before the wait: the KV head's query heads (complete before the grid ahead of
        # this one released it) into shared memory and the A fragments, the block-table row
        # (host-filled), the release.
        cpr: cutlass.Constexpr = dim // 8  # 16-byte pieces a row: one a lane
        q_vec = cute.logical_divide(mQ, (None, None, 8))
        sq_vec = cute.logical_divide(sQ, (None, 8))
        for i in cutlass.range_constexpr(-(-(group * cpr) // THREADS)):
            c = tidx + i * THREADS
            cc = cutlass.min(c, group * cpr - 1)
            h = cc // cpr
            ch = cc % cpr
            if c < group * cpr:
                cute.copy(g2s, q_vec[row, kvh * group + h, (None, ch)], sq_vec[h, (None, ch)])
        cute.arch.cp_async_commit_group()
        for i in cutlass.range_constexpr((M - group) * dim // THREADS):
            e = tidx + i * THREADS
            sQ[group + e // dim, e % dim] = cutlass.BFloat16(0.0)
        req = mReq[row]
        req_ok = (req >= 0) & (req < self.num_requests)
        safe_req = cutlass.min(cutlass.max(req, 0), self.num_requests - 1)
        if const_expr(self.stage_bt):
            sBT = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout((bt_width,), stride=(1,)), byte_alignment=128
            )
            for i in cutlass.range_constexpr(-(-bt_width // THREADS)):
                e = tidx + i * THREADS
                ec = cutlass.min(e, bt_width - 1)
                v = mBT[safe_req, ec]
                if e < bt_width:
                    sBT[e] = v
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        thr = mma.get_slice(tidx)
        tCsQ = thr.partition_A(sQ)
        tCrQ = thr.make_fragment_A(tCsQ)
        cute.autovec_copy(tCsQ, tCrQ)
        if const_expr(self.pdl):
            # The merge waits before reading the partials and reads nothing before its wait
            # that the grid ahead of this one writes.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The CTA's tokens: each one's K and V row offsets in elements, or -1.
        ks0, ks1, ks2, _ = self.k_strides
        vs0, vs1, vs2, _ = self.v_strides
        if tidx < CAP:
            col = c0 + tidx
            colc = cutlass.min(col, width - 1)
            tok = mSel[row, colc]
            ok = req_ok & (col < width) & (tok >= 0)
            safe_tok = cutlass.max(tok, 0)
            lp = safe_tok // tpb
            ok = ok & (lp < bt_width)
            lpc = cutlass.min(lp, bt_width - 1)
            page = cutlass.Int32(0)
            if const_expr(self.stage_bt):
                page = sBT[lpc]
            else:
                page = mBT[safe_req, lpc]
            ok = ok & (page >= 0) & (page < self.num_pages)
            pg = cutlass.Int64(cutlass.max(page, 0))
            tip = cutlass.Int64(safe_tok % tpb)
            hk = cutlass.Int64(kvh)
            sKo[tidx] = cutlass.Int64(
                cutlass.select_(ok, pg * ks0 + hk * ks1 + tip * ks2, cutlass.Int64(-1))
            )
            sVo[tidx] = cutlass.Int64(
                cutlass.select_(ok, pg * vs0 + hk * vs1 + tip * vs2, cutlass.Int64(-1))
            )
        cute.arch.sync_threads()

        # Every token's K and V row: warp w the tokens 8w .. 8w + 7, a lane 16 bytes of each;
        # an invalid token's rows are zeros (its score is masked; V must stay finite).
        sk_vec = cute.logical_divide(sK, (None, 8))
        sv_vec = cute.logical_divide(sV, (None, 8))
        k_base = pK.toint() + cutlass.Int64(lane * 16)
        v_base = pV.toint() + cutlass.Int64(lane * 16)
        tpw: cutlass.Constexpr = CAP // WARPS
        for i in cutlass.range_constexpr(tpw):
            t = warp_i * tpw + i
            ko = sKo[t]
            vo = sVo[t]
            ksrc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16,
                    k_base + cutlass.max(ko, 0) * 2,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((8,), stride=(1,)),
            )
            vsrc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16,
                    v_base + cutlass.max(vo, 0) * 2,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((8,), stride=(1,)),
            )
            if ko >= 0:
                cute.copy(g2s, ksrc, sk_vec[t, (None, lane)])
                cute.copy(g2s, vsrc, sv_vec[t, (None, lane)])
            else:
                for e in cutlass.range_constexpr(8):
                    sK[t, lane * 8 + e] = cutlass.BFloat16(0.0)
                    sV[t, lane * 8 + e] = cutlass.BFloat16(0.0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 3. Scores: a warp eight tokens over the whole head dim, masked into shared memory.
        tCsK = thr.partition_B(sK)
        tCrK = thr.make_fragment_B(tCsK)
        cute.autovec_copy(tCsK, tCrK)
        scores = cute.make_rmem_tensor(thr.partition_shape_C((M, CAP)), cutlass.Float32)
        scores.fill(0.0)
        cute.gemm(mma, scores, tCrQ, tCrK, scores)
        cs = thr.partition_C(cute.make_identity_tensor((M, CAP)))
        neg_inf = cutlass.Float32(-math.inf)
        for j in cutlass.range_constexpr(cute.size(scores)):
            h, t = cs[j]
            v = neg_inf
            if sKo[t] >= 0:
                v = scores[j] * self.scale
            sS[h, t] = v
        cute.arch.sync_threads()
        # The softmax: warp w the heads w and w + WARPS, a lane two tokens; padding heads
        # get P = 0.
        for j in cutlass.range_constexpr(M // WARPS):
            h = warp_i + WARPS * j
            a = sS[h, lane]
            b = sS[h, lane + 32]
            mx = cute.arch.warp_reduction_max(cute.arch.fmax(a, b))
            safe_mx = cutlass.Float32(cutlass.select_(mx == neg_inf, cutlass.Float32(0.0), mx))
            keep = h < group
            pa = cutlass.Float32(
                cutlass.select_(keep, cute.math.exp2(a - safe_mx), cutlass.Float32(0.0))
            )
            pb = cutlass.Float32(
                cutlass.select_(keep, cute.math.exp2(b - safe_mx), cutlass.Float32(0.0))
            )
            den = cute.arch.warp_reduction_sum(pa + pb)
            sP[h, lane] = pa.to(cutlass.BFloat16)
            sP[h, lane + 32] = pb.to(cutlass.BFloat16)
            if lane == 0:
                sStat[h, 0] = mx
                sStat[h, 1] = den
        cute.arch.sync_threads()

        # P V: a warp 32 of the head dims (four 8-wide tiles).
        tCsP = thr.partition_A(sP)
        tCrP = thr.make_fragment_A(tCsP)
        cute.autovec_copy(tCsP, tCrP)
        sVt = cute.make_tensor(sV.iterator, cute.make_layout((dim, CAP), stride=(1, ld)))
        tCsV = thr.partition_B(sVt)
        tCrV = thr.make_fragment_B(tCsV)
        cute.autovec_copy(tCsV, tCrV)
        acc = cute.make_rmem_tensor(thr.partition_shape_C((M, dim)), cutlass.Float32)
        acc.fill(0.0)
        cute.gemm(mma, acc, tCrP, tCrV, acc)

        # Each head's partial times its reciprocal sum, staged over sK (free since the
        # softmax barrier) as [group, dim] fp32, then stored in 16-byte pieces: the CTA's
        # heads are contiguous in the partial tensor.
        sOut = cute.make_tensor(
            cute.recast_ptr(sK.iterator, dtype=cutlass.Float32),
            cute.make_layout((group, dim), stride=(dim, 1)),
        )
        co = thr.partition_C(cute.make_identity_tensor((M, dim)))
        h0 = lane // 4  # an m16n8 C fragment's rows: h0 and h0 + 8
        d0 = sStat[h0, 1]
        d1 = sStat[h0 + 8, 1]
        r0 = cutlass.Float32(0.0)
        r1 = cutlass.Float32(0.0)
        if d0 > 0.0:
            r0 = 1.0 / d0
        if d1 > 0.0:
            r1 = 1.0 / d1
        for j in cutlass.range_constexpr(cute.size(acc)):
            h, d = co[j]
            r = cutlass.Float32(cutlass.select_(h < 8, r0, r1))
            if h < group:
                sOut[h, d] = acc[j] * r
        cute.arch.sync_threads()
        base = ((cutlass.Int64(split) * rows + row) * heads + kvh * group) * dim
        out_ptr = pPart.toint() + base * 4
        so_vec = cute.make_tensor(
            sOut.iterator, cute.make_layout((group * dim // 4, 4), stride=(4, 1))
        )
        reg = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        for i in cutlass.range_constexpr(-(-(group * dim // 4) // THREADS)):
            c = tidx + i * THREADS
            if c < group * dim // 4:
                cute.autovec_copy(so_vec[c, None], reg)
                dst = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        out_ptr + cutlass.Int64(c * 16),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,), stride=(1,)),
                )
                cute.copy(st, reg, dst)
        if tidx < group:
            mx = sStat[tidx, 0]
            den = sStat[tidx, 1]
            lse = neg_inf
            if den > 0.0:
                lse = mx + cute.math.log2(den)
            mLse[split, row, kvh * group + tidx] = lse


class _MergeKernel:
    def __init__(self, *, heads, dim, splits, gated, pdl):
        assert dim % CHUNK == 0 and 1 < splits <= MAX_SPLITS
        self.heads, self.dim, self.splits, self.gated, self.pdl = heads, dim, splits, gated, pdl
        self.chunks = dim // CHUNK
        self.warps = min(8, splits)
        self.spw = -(-splits // self.warps)  # splits a warp, rounded up

    @cute.jit
    def __call__(self, mPart, mLse, mGate, mOut, stream: cuda_drv.CUstream):
        rows = cute.size(mOut, mode=[0])
        ld = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=128)
        self.kernel(mPart, mLse, mGate, mOut, ld).launch(
            grid=[rows * self.heads * self.chunks, 1, 1],
            block=[self.warps * 32, 1, 1],
            smem=self.warps * CHUNK * 4 + 64,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(self, mPart, mLse, mGate, mOut, ld: cute.CopyAtom):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        per_row: cutlass.Constexpr = self.heads * self.chunks
        spw: cutlass.Constexpr = self.spw
        row = bidx // per_row
        head = (bidx % per_row) // self.chunks
        c0 = (bidx % self.chunks) * CHUNK
        red = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.warps, CHUNK), stride=(CHUNK, 1)),
            byte_alignment=16,
        )
        # The gate, before the wait: see the module notes.
        g = cutlass.Float32(0.0)
        if const_expr(self.gated):
            if tidx < CHUNK:
                g = mGate[row, head * self.dim + c0 + tidx].to(cutlass.Float32)
        if const_expr(self.pdl):
            # The output projection after this reads only its weight before its own wait.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 1. The split weights, in every warp: exp2(lse - max), and their sum.
        neg_inf = cutlass.Float32(-math.inf)
        l0 = neg_inf
        l1 = neg_inf
        if lane < self.splits:
            l0 = mLse[lane, row, head]
        if lane + 32 < self.splits:
            l1 = mLse[lane + 32, row, head]
        # 2. Warp w's splits w * spw .. w * spw + spw - 1, four adjacent elements a lane:
        # one 16-byte load a split, all in flight. A slot past the split count reads the
        # last split again and weighs it 0.
        part_vec = cute.logical_divide(mPart, (None, None, None, 4))
        p = cute.make_rmem_tensor(cute.make_layout((spw, 4), stride=(4, 1)), cutlass.Float32)
        for j in cutlass.range_constexpr(spw):
            sidx = cutlass.min(warp_i * spw + j, self.splits - 1)
            cute.copy(ld, part_vec[sidx, row, head, (None, (c0 + 4 * lane) // 4)], p[j, None])
        mx = cute.arch.warp_reduction_max(cute.arch.fmax(l0, l1))
        w0 = cutlass.Float32(0.0)
        w1 = cutlass.Float32(0.0)
        if mx > neg_inf:
            w0 = cute.math.exp2(l0 - mx)
            w1 = cute.math.exp2(l1 - mx)
        den = cute.arch.warp_reduction_sum(w0 + w1)
        acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        acc.fill(0.0)
        for j in cutlass.range_constexpr(spw):
            sidx = warp_i * spw + j
            wa = cute.arch.shuffle_sync(w0, sidx % 32)
            wb = cute.arch.shuffle_sync(w1, sidx % 32)
            w = cutlass.Float32(cutlass.select_(sidx < 32, wa, wb))
            w = cutlass.Float32(cutlass.select_(sidx < self.splits, w, cutlass.Float32(0.0)))
            for e in cutlass.range_constexpr(4):
                acc[e] = w * p[j, e] + acc[e]
        for e in cutlass.range_constexpr(4):
            red[warp_i, 4 * lane + e] = acc[e]
        cute.arch.sync_threads()

        # 3. The warps' shares summed in warp order; the output gate on the rounded
        # attention value, as the composition applies it.
        if tidx < CHUNK:
            tot = cutlass.Float32(0.0)
            for w_ in cutlass.range_constexpr(self.warps):
                tot = tot + red[w_, tidx]
            merged = cutlass.Float32(0.0)
            if den > 0.0:
                merged = tot / den
            if const_expr(self.gated):
                att = merged.to(cutlass.BFloat16).to(cutlass.Float32)
                merged = att * (1.0 / (1.0 + cute.math.exp(-g)))
            mOut[row, head, c0 + tidx] = merged.to(mOut.element_type)


def _dl(t: torch.Tensor, align: int = 4):
    return from_dlpack(t, assumed_align=align)


def _ptr(t: torch.Tensor, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def splits(q, k_cache, selected_tokens) -> int:
    """The split-K CTAs a row and KV head: one per `CAP` selected tokens."""
    return -(-selected_tokens.shape[1] // CAP)


@functools.lru_cache(maxsize=None)
def _compiled_splitk(
    rows,
    q_shape,
    q_stride,
    group,
    width,
    tpb,
    scale,
    bt_shape,
    num_pages,
    k_strides,
    v_strides,
    pdl,
):
    heads = q_shape[1]
    n_splits = -(-width // CAP)
    kern = _SplitKKernel(
        group=group,
        width=width,
        splits=n_splits,
        tpb=tpb,
        scale=scale,
        bt_width=bt_shape[1],
        num_pages=num_pages,
        k_strides=k_strides,
        v_strides=v_strides,
        num_requests=bt_shape[0],
        pdl=pdl,
    )
    dev = "cuda"
    q = torch.empty_strided(q_shape, q_stride, dtype=torch.bfloat16, device=dev)
    kv = torch.empty(64, dtype=torch.bfloat16, device=dev)
    bt = torch.empty(bt_shape, dtype=torch.int32, device=dev)
    sel = torch.empty(rows, width, dtype=torch.int32, device=dev)
    req = torch.empty(rows, dtype=torch.int32, device=dev)
    part = torch.empty(64, dtype=torch.float32, device=dev)
    lse = torch.empty(n_splits, rows, heads, dtype=torch.float32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(q, 16),
        _ptr(kv, cutlass.BFloat16),
        _ptr(kv, cutlass.BFloat16),
        _dl(bt),
        _dl(sel),
        _dl(req),
        _ptr(part, cutlass.Float32),
        _dl(lse),
        stream,
    )


@functools.lru_cache(maxsize=None)
def _compiled_merge(rows, heads, dim, n_splits, gate_stride, dtype, pdl):
    kern = _MergeKernel(
        heads=heads, dim=dim, splits=n_splits, gated=gate_stride is not None, pdl=pdl
    )
    dev = "cuda"
    part = torch.empty(n_splits, rows, heads, dim, dtype=torch.float32, device=dev)
    lse = torch.empty(n_splits, rows, heads, dtype=torch.float32, device=dev)
    gate = torch.empty(rows, gate_stride or heads * dim, dtype=dtype, device=dev)[:, : heads * dim]
    out = torch.empty(rows, heads, dim, dtype=dtype, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kern, _dl(part, 16), _dl(lse), _dl(gate, 2), _dl(out, 2), stream)


def launch(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    tokens_per_block,
    softmax_scale,
    output_gate=None,
    pdl=True,
):
    """The launch behind `QsaPagedSparseGqaDecode.__call__`; returns `[rows, H, D]` in q's dtype."""
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    n_splits = -(-width // CAP)
    output = torch.empty((rows, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    if rows == 0:
        return output
    dev = q.device
    use_pdl = bool(pdl and _qsa_pdl_enabled(rows))
    partial_output = torch.empty(
        (n_splits, rows, num_q_heads, head_dim), dtype=torch.float32, device=dev
    )
    partial_lse = torch.empty((n_splits, rows, num_q_heads), dtype=torch.float32, device=dev)
    fn = _compiled_splitk(
        rows,
        tuple(q.shape),
        tuple(q.stride()),
        group_size,
        width,
        tokens_per_block,
        float(softmax_scale),
        tuple(block_table.shape),
        k_cache.shape[0],
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        use_pdl,
    )
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(q, 16),
        _ptr(k_cache, cutlass.BFloat16),
        _ptr(v_cache, cutlass.BFloat16),
        _dl(block_table),
        _dl(selected_tokens),
        _dl(request_indices),
        _ptr(partial_output, cutlass.Float32),
        _dl(partial_lse),
        stream,
    )
    gate2d = None if output_gate is None else output_gate.reshape(rows, num_q_heads * head_dim)
    mfn = _compiled_merge(
        rows,
        num_q_heads,
        head_dim,
        n_splits,
        None if gate2d is None else gate2d.stride(0),
        q.dtype,
        use_pdl,
    )
    mfn(
        _dl(partial_output, 16),
        _dl(partial_lse),
        _dl(output.view(rows, num_q_heads * head_dim) if gate2d is None else gate2d, 2),
        _dl(output, 2),
        stream,
    )
    return output


def supports(q, k_cache, selected_tokens) -> bool:
    """Whether these operands are ones this entry serves (see `is_valid`)."""
    return (
        q.dim() == 3
        and 0 < q.shape[0] <= MAX_ROWS
        and q.shape[2] == HEAD_DIM
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and q.shape[1] % k_cache.shape[1] == 0
        and q.shape[1] // k_cache.shape[1] <= M
        and 1 < splits(q, k_cache, selected_tokens) <= MAX_SPLITS
        and all(t.data_ptr() % 16 == 0 for t in (q, k_cache))
        and all(s % 8 == 0 for s in (*q.stride()[:2], *k_cache.stride()[:3]))
    )


class QsaPagedSparseGqaDecode(OpWrapper):
    """`qsa_paged_sparse_gqa` without a prefill pair, for decode rows: a split-K pass whose
    CTAs load their tokens' K/V in one round trip, then the merge -- and, with
    `output_gate`, the gate -- as one launch. Returns a new `[rows, H, D]` tensor in q's
    dtype.

    Fusion boundary: the split-K attention, the merge, the optional sigmoid output gate.
    The selection, the K/V store and the output projection are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, 1 row at position 8700, gated", spec=dict(pos=(8700,), gated=True)),
        Cell(
            why="decode, 4 rows, one a short sequence, gated",
            spec=dict(pos=(8700, 5000, 77, 3000), gated=True),
        ),
        Cell(
            why="decode, 1 row with half the selection -1",
            spec=dict(pos=(8700,), gated=True, holes=True),
        ),
    )

    note = """
    Metadata consumed: as `qsa_paged_sparse_gqa` (block table, request indices). Preconditions:
    as that entry's, plus: no prefill pair; at most `MAX_ROWS` rows; bf16 q and K/V with a head
    dim of 256 and at most 16 query heads a KV head; more than one and at most 64 split-K CTAs
    a row and KV head (`splits(...)`: 64 selected tokens a CTA); `output_gate` (when given)
    `[rows, H * D]` or `[rows, H, D]` with unit inner stride in q's dtype. `supports(...)` is
    the predicate a caller can check.

    PDL: the split-K pass reads q and the block table before its wait and the merge reads the
    output gate before its own; the selection and the K/V rows are read after the split-K
    pass's wait. So q and the gate must be complete when the grid ahead of this entry releases
    it: a launch without PDL releases at its end; a PDL producer of the selection (the
    target's top-k) must release only after its own wait.

    Numerics: one softmax over a CTA's 64 tokens (QSA's kernel runs an online softmax over
    16-token tiles) and the merge sums the splits in split order (QSA's merge reduces them as
    a tree), so the fp32 values can differ in the last bits and the bf16 output by an ulp on
    some elements.

    Not certified: a call without `output_gate` (the target always passes one).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return launch(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The composition entry's reference with the same operands."""
        return _composition.reference(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=None,
            compress_ratio=None,
            output_gate=output_gate,
        )

    def is_valid(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate: Optional[torch.Tensor] = None,
    ) -> None:
        assert q.ndim == 3 and q.shape[2] == HEAD_DIM, "q [rows, H, 256]"
        assert 0 < q.shape[0] <= MAX_ROWS, f"1 .. {MAX_ROWS} rows"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == v_cache.dtype == torch.bfloat16, (
            "bf16 q and K/V"
        )
        assert q.shape[1] % k_cache.shape[1] == 0 and q.shape[1] // k_cache.shape[1] <= M, (
            "<= 16 q heads a KV head"
        )
        assert 1 < splits(q, k_cache, selected_tokens) <= MAX_SPLITS, (
            "65 .. 4096 selected tokens a row"
        )
        assert q.stride(2) == 1 and k_cache.stride(3) == 1 and v_cache.stride(3) == 1, (
            "unit inner strides"
        )
        assert all(t.data_ptr() % 16 == 0 for t in (q, k_cache, v_cache)), (
            "16-byte aligned q and K/V"
        )
        assert all(
            s % 8 == 0 for s in (*q.stride()[:2], *k_cache.stride()[:3], *v_cache.stride()[:3])
        ), "q and K/V rows 16-byte aligned"
        if output_gate is not None:
            assert output_gate.dtype == q.dtype and output_gate.stride(-1) == 1, "gate in q's dtype"
            assert output_gate.numel() == q.numel(), "gate covers q"


qsa_paged_sparse_gqa_decode = QsaPagedSparseGqaDecode()
