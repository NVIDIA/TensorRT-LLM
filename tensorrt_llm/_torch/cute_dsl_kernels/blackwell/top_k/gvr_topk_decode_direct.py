# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct (short-row) GVR top-K tier — CuTe DSL, Blackwell SM100.

CuTe DSL translation of the original CUDA ``direct_topk_kernel``,
adapted for the production
``trtllm::cute_dsl_gvr_topk_decode`` contract (ragged N via a device
``seq_lens`` tensor + per-row degenerate identity emit; see the module
docstring of ``gvr_topk_decode_tp`` for the shared adaptation inventory).

Exact top-K indices for short padded rows (npad <= DKCMAX = 12288), one CTA
per row, TB = 1024 threads. No threshold solve: the whole row is collected
into SMEM as packed (f2u order key << 32 | index) u64 candidates with an
in-flight 2048-bin radix histogram over the top 11 key bits, then an
11/11/10-bit radix select with whole-bin early exit and boundary-bin
compaction. Tie-aware emit (strict-greater mandatory slots + tie tickets
filling to K) makes the output value multiset equal torch.topk exactly.

Ragged N: elements at index >= N_eff contribute key = f2u(-FLT_MAX) (the
minimum key) instead of their stale value; with the degenerate rows
(N_eff <= K) peeled off in-kernel, a masked element can never enter the
top-K, so out-of-range indices are never emitted.

Public entry: ``direct_topk(logits, seq_lens, out, K)`` with torch tensors
(logits [BS, npad] fp32 contiguous, seq_lens [BS] int32, out [BS, K] int32).
Compiled variants are cached per (K, TB); BS and npad are dynamic (sym_int)
so one variant serves every shape.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.utils.smem_allocator import SmemAllocator

FLT_MAX = 3.4028234663852886e38
DKCMAX = 12288  # direct-path candidate capacity (mirrors the CUDA development arm)


@cute.jit
def _f2u(v):
    """Order-preserving fp32 -> uint32 radix key (CUDA f2u).

    u ^ (sign ? 0xFFFFFFFF : 0x80000000): unsigned order of the result equals
    fp32 order (NaN-free inputs). All shifts on the returned Uint32 are
    logical, all compares unsigned.
    """
    u = cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, v.ir_value()))
    mask = (cutlass.Uint32(0) - (u >> cutlass.Uint32(31))) | cutlass.Uint32(0x80000000)
    return u ^ mask


class DirectTopKKernel:
    """One-CTA-per-row exact radix top-K for npad <= DKCMAX.

    Ctor knobs (compile-time): ``top_k`` in {512, 1024, 2048}, ``num_threads``
    (TB, default 1024), ``next_n`` / ``compress_ratio`` for the per-row
    N_eff arithmetic (constexpr; the direct tier reads no hints, so MTP
    support is the N_eff formula alone). SMEM layout
    mirrors the CUDA development arm's DSmem<TB> with the same packed (key << 32 | idx)
    u64 layout (measured: split key/idx arrays cost 2x smem
    transactions in collect + emits vs CUDA's single STS.64 / LDS.64 per
    candidate):

      cand[DKCMAX] u64 | side[SIDECAP] u64 (boundary-bin compaction) |
      hist[2048] i32 | iwred[NWARP] i32 | sel[2] i32 | cnt[3] i32
    """

    WARP_SIZE = 32

    def __init__(
        self, top_k: int, num_threads: int = 1024, next_n: int = 1, compress_ratio: int = 4
    ):
        if top_k not in (512, 1024, 2048):
            raise ValueError(f"unsupported top_k {top_k}")
        if num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32")
        self.top_k = top_k
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        # DCAP_RUNGS * TB / 2 in CUDA (side[] unions the dead ptcnt[]); the
        # direct path never allocates ptcnt so we keep only side[].
        self.sidecap = 16 * num_threads // 2

    # ------------------------------------------------------------------
    # Per-row valid length (ragged N) — mirrors the in-tree run_one_row
    # arithmetic exactly (see gvr_topk_decode_tp._row_n_eff).
    # ------------------------------------------------------------------
    @cute.jit
    def _row_n_eff(self, seq_lens: cute.Tensor, row):
        NN = cutlass.const_expr(self.next_n)
        seq_len = seq_lens[row // cutlass.Int32(NN)]
        actual_kv_len = seq_len - cutlass.Int32(NN) + (row % cutlass.Int32(NN)) + cutlass.Int32(1)
        if cutlass.const_expr(self.compress_ratio == 1):
            n_eff = actual_kv_len
        else:
            n_eff = actual_kv_len // cutlass.Int32(self.compress_ratio)
        return n_eff

    # ------------------------------------------------------------------
    # Warp suffix-inclusive scan (higher lane = higher bins), the
    # __shfl_down_sync ladder of the CUDA bin_select.
    # ------------------------------------------------------------------
    @cute.jit
    def _warp_suffix_scan(self, val, lane):
        x = val
        for d in cutlass.range_constexpr(5):
            off = cutlass.const_expr(1 << d)
            src = lane + cutlass.Int32(off)
            if src > cutlass.Int32(31):
                src = cutlass.Int32(31)
            up = cute.arch.shuffle_sync(x, src)
            if lane + cutlass.Int32(off) < cutlass.Int32(32):
                x = x + up
        return x

    # ------------------------------------------------------------------
    # bin_select<TB, NB>: find bin b such that above(b) < want <= above(b)
    # + hist[b] where above(b) = sum_{j>b} hist[j]. Writes smem_sel =
    # {b, above(b)}. Trailing barrier.
    # ------------------------------------------------------------------
    @cute.jit
    def _bin_select(
        self,
        nbins: cutlass.Constexpr,
        want,
        smem_hist,
        smem_iwred,
        smem_sel,
        tidx,
        lane,
        warp_id,
    ):
        per = cutlass.const_expr(nbins // self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        h = cute.make_rmem_tensor((per,), cutlass.Int32)
        s_sum = cutlass.Int32(0)
        for q in cutlass.range_constexpr(per):
            h[q] = smem_hist[tidx * cutlass.Int32(per) + cutlass.Int32(q)]
            s_sum = s_sum + h[q]
        # warp suffix-inclusive over lanes (higher lane = higher bins)
        x = self._warp_suffix_scan(s_sum, lane)
        if lane == cutlass.Int32(0):
            smem_iwred[warp_id] = x  # warp total
        cute.arch.barrier()
        if tidx < cutlass.Int32(32):
            wt = cutlass.Int32(0)
            if tidx < cutlass.Int32(num_warps):
                wt = smem_iwred[tidx]
            wx = self._warp_suffix_scan(wt, tidx)
            if tidx < cutlass.Int32(num_warps):
                smem_iwred[tidx] = wx - wt  # strictly-above-warp sum
        cute.arch.barrier()
        a_cnt = smem_iwred[warp_id] + (x - s_sum)  # bins strictly above chunk
        if a_cnt < want and want <= a_cnt + s_sum:
            run = a_cnt
            for qr in cutlass.range_constexpr(per):
                q = cutlass.const_expr(per - 1 - qr)
                if run < want and want <= run + h[q]:
                    smem_sel[0] = tidx * cutlass.Int32(per) + cutlass.Int32(q)
                    smem_sel[1] = run
                run = run + h[q]
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Warp-aggregated slot allocation: ballot + leader atomicAdd + popc.
    # Whole-warp uniform trip counts are guaranteed by Cpad % 32 == 0 and
    # TB % 32 == 0 (same argument as the CUDA original).
    # ------------------------------------------------------------------
    @cute.jit
    def _emit_prefix_ge(self, cand_cnt, shift, prefix, smem_cand, smem_cnt, out_row, tidx, lane):
        """Emit indices of all candidates with (key >> shift) >= prefix;
        the caller guarantees exactly K of them."""
        num_threads = cutlass.const_expr(self.num_threads)
        if tidx == cutlass.Int32(0):
            smem_cnt[0] = cutlass.Int32(0)
        cute.arch.barrier()
        cpad = (cand_cnt + cutlass.Int32(31)) & cutlass.Int32(-32)
        i = tidx
        while i < cpad:
            valid = i < cand_cnt
            kv = cutlass.Uint64(0)
            if valid:
                kv = smem_cand[i]
            key = cutlass.Uint32(kv >> cutlass.Uint64(32))
            e = valid and ((key >> shift) >= prefix)
            bal = cute.arch.vote_ballot_sync(e)
            if bal != cutlass.Uint32(0):
                tz = cutlass.Int32(
                    cute.arch.popc((bal & (cutlass.Uint32(0) - bal)) - cutlass.Uint32(1))
                )
                base = cutlass.Int32(0)
                if lane == tz:
                    base = cute.arch.atomic_add(
                        smem_cnt.iterator, cutlass.Int32(cute.arch.popc(bal)), scope="cta"
                    )
                base = cute.arch.shuffle_sync(base, tz)
                if e:
                    rk = cutlass.Int32(
                        cute.arch.popc(bal & cutlass.Uint32(cute.arch.lanemask_lt()))
                    )
                    out_row[base + rk] = cutlass.Int32(
                        cutlass.Uint32(kv & cutlass.Uint64(0xFFFFFFFF))
                    )
            i = i + cutlass.Int32(num_threads)

    @cute.jit
    def _emit_final(self, cand_cnt, kth, m, nt, smem_cand, smem_cnt, out_row, tidx, lane):
        """keys > kth are mandatory (slots [0, m)); keys == kth fill the
        remaining nt tie tickets (slots [m, m+nt))."""
        num_threads = cutlass.const_expr(self.num_threads)
        if tidx == cutlass.Int32(0):
            smem_cnt[0] = cutlass.Int32(0)
            smem_cnt[1] = cutlass.Int32(0)
        cute.arch.barrier()
        cpad = (cand_cnt + cutlass.Int32(31)) & cutlass.Int32(-32)
        i = tidx
        while i < cpad:
            valid = i < cand_cnt
            kv = cutlass.Uint64(0)
            if valid:
                kv = smem_cand[i]
            key = cutlass.Uint32(kv >> cutlass.Uint64(32))
            idx = cutlass.Int32(cutlass.Uint32(kv & cutlass.Uint64(0xFFFFFFFF)))
            man = valid and (key > kth)
            tie = valid and (key == kth)
            bm = cute.arch.vote_ballot_sync(man)
            bt = cute.arch.vote_ballot_sync(tie)
            if bm != cutlass.Uint32(0):
                tzm = cutlass.Int32(
                    cute.arch.popc((bm & (cutlass.Uint32(0) - bm)) - cutlass.Uint32(1))
                )
                base_m = cutlass.Int32(0)
                if lane == tzm:
                    base_m = cute.arch.atomic_add(
                        smem_cnt.iterator, cutlass.Int32(cute.arch.popc(bm)), scope="cta"
                    )
                base_m = cute.arch.shuffle_sync(base_m, tzm)
                if man:
                    rk = cutlass.Int32(cute.arch.popc(bm & cutlass.Uint32(cute.arch.lanemask_lt())))
                    out_row[base_m + rk] = idx
            if bt != cutlass.Uint32(0):
                tzt = cutlass.Int32(
                    cute.arch.popc((bt & (cutlass.Uint32(0) - bt)) - cutlass.Uint32(1))
                )
                base_t = cutlass.Int32(0)
                if lane == tzt:
                    base_t = cute.arch.atomic_add(
                        smem_cnt.iterator + 1, cutlass.Int32(cute.arch.popc(bt)), scope="cta"
                    )
                base_t = cute.arch.shuffle_sync(base_t, tzt)
                if tie:
                    p = base_t + cutlass.Int32(
                        cute.arch.popc(bt & cutlass.Uint32(cute.arch.lanemask_lt()))
                    )
                    if p < nt:
                        out_row[m + p] = idx
            i = i + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def direct_topk_kernel(
        self,
        logits: cute.Tensor,  # [BS, npad] fp32; tail beyond N_eff may be garbage
        seq_lens: cute.Tensor,  # [BS] int32 (uncompressed-token space)
        out: cute.Tensor,  # [BS, K] int32
    ):
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kK = cutlass.const_expr(self.top_k)
        sidecap = cutlass.const_expr(self.sidecap)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx & cutlass.Int32(31)
        warp_id = tidx // cutlass.Int32(32)

        npad = cutlass.Int32(logits.shape[1])
        logits_row = logits[bidx, None]
        out_row = out[bidx, None]

        # Ragged N: per-row valid length from seq_lens.
        n_eff = self._row_n_eff(seq_lens, bidx)

        # ---- SMEM (allocated unconditionally, before the dynamic branch,
        # so the DSL sizes the launch SMEM identically on every path) ----
        smem = SmemAllocator()
        s_cand = smem.allocate_tensor(
            cutlass.Uint64, cute.make_layout((DKCMAX,)), byte_alignment=128
        )
        s_side = smem.allocate_tensor(
            cutlass.Uint64, cute.make_layout((sidecap,)), byte_alignment=128
        )
        s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2048,)), byte_alignment=128)
        s_iwred = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((num_warps,)), byte_alignment=128
        )
        s_sel = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2,)), byte_alignment=16)
        s_cnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((3,)), byte_alignment=16)

        # ---- Degenerate rows (N_eff <= K): identity emit + -1 pad ----
        # (mirrors the in-tree kernel; CuTe DSL has no runtime return, so the
        # main body lives in the else branch).
        if n_eff <= cutlass.Int32(kK):
            jd = cutlass.Int32(tidx)
            while jd < n_eff:
                out_row[jd] = jd
                jd = jd + cutlass.Int32(num_threads)
            jp = n_eff + cutlass.Int32(tidx)
            if jp < cutlass.Int32(0):
                jp = cutlass.Int32(tidx)  # n_eff < 0 (defensive): pad all
            while jp < cutlass.Int32(kK):
                out_row[jp] = cutlass.Int32(-1)
                jp = jp + cutlass.Int32(num_threads)
        else:
            # ---- Collect: vectorized load + f2u keys + in-flight 2048-bin hist
            for z in cutlass.range_constexpr(2048 // num_threads):
                s_hist[tidx + cutlass.Int32(z * num_threads)] = cutlass.Int32(0)
            cute.arch.barrier()

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=128, invariant=True
            )
            row_addr = logits_row.iterator.toint()
            vcnt = npad >> cutlass.Int32(2)  # npad % 4 == 0 (host-asserted)
            # npad <= DKCMAX bounds the per-thread trip count
            # at U = DKCMAX/4/TB (= 3 for TB 1024), so the whole grid-stride
            # loop is flattened into one predicated register batch: issue ALL
            # (<=3) float4 loads back-to-back, then consume. A dynamic
            # `cutlass.range(iters, unroll=4)` loop never reaches its unrolled
            # body (trip <= 3) and leaves ONE load in flight per iteration —
            # 1.8x the long-scoreboard stall of the CUDA arm and a stable
            # ~1.03 cold-kernel gap at npad 8256. Keep the flat batch.
            n_batch = cutlass.const_expr((DKCMAX // 4 + num_threads - 1) // num_threads)
            frags = [
                cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(n_batch)
            ]  # Python-unrolled register batch
            for u in cutlass.range_constexpr(n_batch):
                i_vec = tidx + cutlass.Int32(u * num_threads)
                if i_vec < vcnt:
                    src_ptr = cute.make_ptr(
                        cutlass.Float32,
                        row_addr + cutlass.Int64(i_vec) * cutlass.Int64(16),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    src = cute.make_tensor(src_ptr, cute.make_layout((4,)))
                    cute.copy(copy_atom, src, frags[u])
            for u in cutlass.range_constexpr(n_batch):
                i_vec = tidx + cutlass.Int32(u * num_threads)
                if i_vec < vcnt:
                    gi = i_vec * cutlass.Int32(4)
                    for q in cutlass.range_constexpr(4):
                        # Ragged N: stale values beyond N_eff contribute the
                        # minimum key f2u(-FLT_MAX); with n_eff > K here they
                        # can never enter the top-K, so their (out-of-range)
                        # indices are never emitted.
                        v = frags[u][q]
                        if gi + cutlass.Int32(q) >= n_eff:
                            v = cutlass.Float32(-FLT_MAX)
                        key = _f2u(v)
                        s_cand[gi + cutlass.Int32(q)] = (
                            cutlass.Uint64(key) << cutlass.Uint64(32)
                        ) | cutlass.Uint64(cutlass.Uint32(gi + cutlass.Int32(q)))
                        cute.arch.atomic_add(
                            s_hist.iterator + cutlass.Int32(key >> cutlass.Uint32(21)),
                            cutlass.Int32(1),
                            scope="cta",
                        )
            cute.arch.barrier()

            # ---- radix_select_emit (11/11/10-bit, whole-bin early exit) ----
            if npad == cutlass.Int32(kK):
                # C == K: every candidate is admitted; identity emit. (With
                # seq_lens present this is unreachable — n_eff <= npad == K
                # lands in the degenerate branch — but kept for structural
                # parity with the CUDA source.)
                io = tidx
                while io < npad:
                    out_row[io] = cutlass.Int32(
                        cutlass.Uint32(s_cand[io] & cutlass.Uint64(0xFFFFFFFF))
                    )
                    io = io + cutlass.Int32(num_threads)
            else:
                want = cutlass.Int32(kK)
                m = cutlass.Int32(0)
                # level 0: top 11 bits (hist prebuilt during collect)
                self._bin_select(2048, want, s_hist, s_iwred, s_sel, tidx, lane, warp_id)
                b0 = s_sel[0]
                a0 = s_sel[1]
                h0 = s_hist[b0]
                if want == a0 + h0:
                    # k-th boundary == bin edge: whole bin admitted
                    self._emit_prefix_ge(
                        npad,
                        cutlass.Uint32(21),
                        cutlass.Uint32(b0),
                        s_cand,
                        s_cnt,
                        out_row,
                        tidx,
                        lane,
                    )
                else:
                    m = m + a0
                    want = want - a0
                    docompact = h0 <= cutlass.Int32(sidecap)
                    cute.arch.barrier()  # all reads of hist/sel done before reuse
                    if tidx == cutlass.Int32(0):
                        s_cnt[2] = cutlass.Int32(0)
                    for z1 in cutlass.range_constexpr(2048 // num_threads):
                        s_hist[tidx + cutlass.Int32(z1 * num_threads)] = cutlass.Int32(0)
                    cute.arch.barrier()
                    # level 1 sweep: mid 11 bits of boundary-bin members; compact
                    ub0 = cutlass.Uint32(b0)
                    i1 = tidx
                    while i1 < npad:
                        kv1 = s_cand[i1]
                        key1 = cutlass.Uint32(kv1 >> cutlass.Uint64(32))
                        if (key1 >> cutlass.Uint32(21)) == ub0:
                            cute.arch.atomic_add(
                                s_hist.iterator
                                + cutlass.Int32(
                                    (key1 >> cutlass.Uint32(10)) & cutlass.Uint32(0x7FF)
                                ),
                                cutlass.Int32(1),
                                scope="cta",
                            )
                            if docompact:
                                p1 = cute.arch.atomic_add(
                                    s_cnt.iterator + 2, cutlass.Int32(1), scope="cta"
                                )
                                s_side[p1] = kv1
                        i1 = i1 + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    self._bin_select(2048, want, s_hist, s_iwred, s_sel, tidx, lane, warp_id)
                    b1 = s_sel[0]
                    a1 = s_sel[1]
                    h1 = s_hist[b1]
                    p01 = (ub0 << cutlass.Uint32(11)) | cutlass.Uint32(b1)
                    if want == a1 + h1:
                        self._emit_prefix_ge(
                            npad,
                            cutlass.Uint32(10),
                            p01,
                            s_cand,
                            s_cnt,
                            out_row,
                            tidx,
                            lane,
                        )
                    else:
                        m = m + a1
                        want = want - a1
                        cute.arch.barrier()
                        for z2 in cutlass.range_constexpr(1024 // num_threads):
                            s_hist[tidx + cutlass.Int32(z2 * num_threads)] = cutlass.Int32(0)
                        cute.arch.barrier()
                        # level 2 sweep: low 10 bits
                        ub1 = cutlass.Uint32(b1)
                        if docompact:
                            i2 = tidx
                            while i2 < h0:
                                key2 = cutlass.Uint32(s_side[i2] >> cutlass.Uint64(32))
                                if ((key2 >> cutlass.Uint32(10)) & cutlass.Uint32(0x7FF)) == ub1:
                                    cute.arch.atomic_add(
                                        s_hist.iterator
                                        + cutlass.Int32(key2 & cutlass.Uint32(0x3FF)),
                                        cutlass.Int32(1),
                                        scope="cta",
                                    )
                                i2 = i2 + cutlass.Int32(num_threads)
                        else:
                            i3 = tidx
                            while i3 < npad:
                                key3 = cutlass.Uint32(s_cand[i3] >> cutlass.Uint64(32))
                                if (key3 >> cutlass.Uint32(10)) == p01:
                                    cute.arch.atomic_add(
                                        s_hist.iterator
                                        + cutlass.Int32(key3 & cutlass.Uint32(0x3FF)),
                                        cutlass.Int32(1),
                                        scope="cta",
                                    )
                                i3 = i3 + cutlass.Int32(num_threads)
                        cute.arch.barrier()
                        self._bin_select(1024, want, s_hist, s_iwred, s_sel, tidx, lane, warp_id)
                        b2 = s_sel[0]
                        a2 = s_sel[1]
                        m = m + a2
                        want = want - a2
                        kth = (p01 << cutlass.Uint32(10)) | cutlass.Uint32(b2)
                        self._emit_final(npad, kth, m, want, s_cand, s_cnt, out_row, tidx, lane)

    # ------------------------------------------------------------------
    # Host launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, logits: cute.Tensor, seq_lens: cute.Tensor, out: cute.Tensor, stream):
        num_rows = logits.shape[0]
        self.direct_topk_kernel(logits, seq_lens, out).launch(
            grid=(num_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,  # __launch_bounds__(TB, 1)
        )


# ---------------------------------------------------------------------------
# torch-facing entry with a compiled-variant cache keyed (K, TB).
# ---------------------------------------------------------------------------
_COMPILE_CACHE: dict = {}


def _get_compiled(top_k: int, num_threads: int = 1024, next_n: int = 1, cr: int = 4):
    key = (top_k, num_threads, next_n, cr)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        kernel = DirectTopKKernel(
            top_k=top_k, num_threads=num_threads, next_n=next_n, compress_ratio=cr
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (n_rows, n_cols), stride_order=(1, 0), assumed_align=16
        )
        seq_lens_fake = _crt.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kernel,
            logits_fake,
            seq_lens_fake,
            out_fake,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        _COMPILE_CACHE[key] = compiled
    return compiled


_CHECKED_SIGS: set = set()


def _check_contract(logits, seq_lens, out, K):
    assert logits.dtype == torch.float32 and logits.is_contiguous()
    assert out.dtype == torch.int32 and out.is_contiguous()
    assert seq_lens.dtype == torch.int32
    bs, npad = logits.shape
    assert npad <= DKCMAX, f"direct path requires npad <= {DKCMAX}, got {npad}"
    assert npad % 4 == 0, f"npad must be a multiple of 4, got {npad}"
    assert out.shape[0] == bs and out.shape[1] == K
    assert logits.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0


def direct_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    K: int,
    next_n: int = 1,
    cr: int = 4,
) -> None:
    """Exact top-K indices of ``logits`` rows into ``out`` (direct path).

    logits:   [BS, npad] fp32 contiguous; the tail beyond each row's N_eff
              may be stale garbage (masked in-kernel).
    seq_lens: [BS // next_n] int32, request-level, uncompressed-token space.
    out:      [BS, K] int32 contiguous.
    K:        512 / 1024 / 2048.

    Contract checks run once per (shape, K) signature to keep the hot
    launch path at bare tvm-ffi cost.
    """
    sig = (logits.shape, out.shape, K, next_n, cr)
    if sig not in _CHECKED_SIGS:
        _check_contract(logits, seq_lens, out, K)
        _CHECKED_SIGS.add(sig)
    compiled = _COMPILE_CACHE.get((K, 1024, next_n, cr))
    if compiled is None:
        compiled = _get_compiled(K, next_n=next_n, cr=cr)
    compiled(logits, seq_lens, out)


__all__ = ["DirectTopKKernel", "direct_topk", "DKCMAX"]
