/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MiniMax-M3 TP8 fused index score + top-16 block select in one launch.
//
// Scoring: grid (splits, batch), one CTA per SM. CTA (s, b) owns blocks s, s + splits, ... of request b and TMA-loads
// their index-K pages through a stage ring (one producer warp, 8 consumer warps; each consumer warp scores a 32-row
// slice of a page against all q columns with m16n8k16 FP16 MMAs, column = head * 4 + query row). Block scores go to a
// row-major scratch; the last CTA of the request (arrival counter) bulk-copies the request's rows back into its idle
// stage ring and selects the top 16 per (query row, KV head): a threshold bound from per-thread maxima, a candidate
// list, exact ranking by (key desc, block asc), and an exact radix select when ties overflow the candidate list.

#include "tensorrt_llm/kernels/minimaxM3Tp8/fusedScoreSelect.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kTopK = 16;
constexpr int kBlockK = 128;
constexpr int kHeadDim = 128;
constexpr int kStageBytes = kBlockK * kHeadDim; // fp8
constexpr int kConsumerWarps = 8;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kThreads = kConsumerThreads + 32; // + producer warp
constexpr int kMaxDql = 4;                      // B-operand column = head * kMaxDql + query row
constexpr int kCols = 8;                        // one n8 tile: heads * kMaxDql <= 8
constexpr int kMaxRows = 8;                     // kv_heads * dql
constexpr int kMaxLocal = 128;                  // blocks per CTA
constexpr int kCandCap = 256;
constexpr int kEpiBytes = kMaxLocal * 4 * kCols * 4;
constexpr float kInitScore = 1.0e30F;
constexpr float kLocalScore = 1.0e29F;
constexpr int kNStage = 12; // TMA stage ring depth (pages in flight per CTA)

using Params = MinimaxM3Tp8FusedScoreSelectParams;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ uint32_t twiddle(float x)
{
    uint32_t const bits = __float_as_uint(x);
    return bits ^ ((bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(uint32_t pair16)
{
    uint32_t out;
    unsigned short const in = static_cast<unsigned short>(pair16 & 0xFFFFu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(in));
    return out;
}

__device__ __forceinline__ void mma16816(
    float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity)
{
    uint32_t done;
    do
    {
        asm volatile(
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
    } while (!done);
}

__device__ __forceinline__ void tma_load_page(
    uint32_t dst, CUtensorMap const* tmap, int page, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(tmap)), "r"(0), "r"(0), "r"(page), "r"(bar), "l"(policy)
        : "memory");
}

// Exact radix select over n smem keys (fallback when the threshold bound leaves too many candidates).
__device__ void radix_emit(uint32_t const* kr, int n, uint32_t* hist, int32_t* dst, uint32_t neg_inf_key)
{
    int const lane = threadIdx.x & 31;
    uint32_t prefix = 0, mask = 0;
    int need = kTopK;
    bool exact_bin = false;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (int i = lane; i < 256; i += 32)
            hist[i] = 0;
        __syncwarp();
        for (int i = lane; i < n; i += 32)
        {
            uint32_t const k = kr[i];
            if ((k & mask) == prefix)
                atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncwarp();
        uint32_t c[8];
        uint32_t sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            c[j] = hist[255 - 8 * lane - j];
            sum += c[j];
        }
        uint32_t incl = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            uint32_t const v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
            if (lane >= o)
                incl += v;
        }
        uint32_t above = incl - sum;
        int found_bin = -1;
        uint32_t found_above = 0, found_cnt = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            if (found_bin < 0 && above < static_cast<uint32_t>(need) && above + c[j] >= static_cast<uint32_t>(need))
            {
                found_bin = 255 - 8 * lane - j;
                found_above = above;
                found_cnt = c[j];
            }
            above += c[j];
        }
        unsigned const bal = __ballot_sync(0xFFFFFFFFu, found_bin >= 0);
        int const src = __ffs(bal) - 1;
        int const bin = __shfl_sync(0xFFFFFFFFu, found_bin, src);
        uint32_t const fa = __shfl_sync(0xFFFFFFFFu, found_above, src);
        uint32_t const fc = __shfl_sync(0xFFFFFFFFu, found_cnt, src);
        __syncwarp();
        need -= static_cast<int>(fa);
        prefix |= static_cast<uint32_t>(bin) << shift;
        mask |= 255u << shift;
        if (fc == static_cast<uint32_t>(need))
        {
            exact_bin = true;
            break;
        }
    }
    int emitted = 0, ties_used = 0;
    uint32_t const lt = (1u << lane) - 1u;
    for (int c0 = 0; c0 < n; c0 += 32)
    {
        int const i = c0 + lane;
        bool const valid = i < n;
        uint32_t const k = valid ? kr[i] : 0u;
        uint32_t const km = k & mask;
        bool const neg = valid && k == neg_inf_key;
        bool sel = valid && (km > prefix || (exact_bin && km == prefix));
        bool const tie = valid && !exact_bin && km == prefix;
        unsigned const tie_bal = __ballot_sync(0xFFFFFFFFu, tie);
        if (tie)
            sel = ties_used + __popc(tie_bal & lt) < need;
        ties_used += __popc(tie_bal);
        bool const out_ok = sel && !neg;
        unsigned const ob = __ballot_sync(0xFFFFFFFFu, out_ok);
        if (out_ok)
            dst[emitted + __popc(ob & lt)] = i;
        emitted += __popc(ob);
    }
    if (lane >= emitted && lane < kTopK)
        dst[lane] = -1;
    __syncwarp();
}

// Descending bitonic sort of one value per lane (lane 0 ends with the largest).
__device__ __forceinline__ uint32_t warp_sort_desc(uint32_t v, int lane)
{
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
    {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            uint32_t const o = __shfl_xor_sync(0xFFFFFFFFu, v, j);
            bool const desc_block = (lane & k) == 0;
            bool const low = (lane & j) == 0;
            v = (low == desc_block) ? max(v, o) : min(v, o);
        }
    }
    return v;
}

__device__ __forceinline__ uint32_t forced_key(float sc, int i, int init_blocks, int local_start)
{
    if (i < init_blocks)
        sc = kInitScore;
    if (i >= local_start)
        sc = kLocalScore;
    return twiddle(sc);
}

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <int NSTAGE>
__global__ void __launch_bounds__(kThreads, 1)
    m3_fused_score_select_kernel(__grid_constant__ CUtensorMap const tmap, Params p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
    uint8_t* stages = smem;
    float* epi = reinterpret_cast<float*>(smem + NSTAGE * kStageBytes);         // [kMaxLocal][4 slices][kCols]
    uint64_t* bars
        = reinterpret_cast<uint64_t*>(smem + NSTAGE * kStageBytes + kEpiBytes); // full[NSTAGE], empty[NSTAGE]
    __shared__ uint32_t s_wtop[kMaxRows][kConsumerWarps][16];
    __shared__ int s_cnt[kMaxRows];
    __shared__ int32_t s_sel_blk[kMaxRows][kTopK];
    __shared__ int s_sel_neg[kMaxRows][kTopK];
    __shared__ int s_last;
    __shared__ int32_t s_sb[kMaxRows][32];
    __shared__ __align__(8) uint64_t s_selbar;

    int const s = blockIdx.x, b = blockIdx.y, splits = gridDim.x;
    int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int const g = lane >> 2, tq = lane & 3;
    uint32_t const full0 = smem_u32(bars), empty0 = smem_u32(bars + NSTAGE);

    if (tid == 0)
    {
        for (int i = 0; i < NSTAGE; ++i)
        {
            mbar_init(full0 + 8 * i, 1);
            mbar_init(empty0 + 8 * i, 4);
        }
        mbar_init(smem_u32(&s_selbar), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == kConsumerWarps && lane == 0)
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tmap)) : "memory");
    __syncthreads();

    // The producer lanes fetch their first page ids together with seq_lens (bounded by the table width), so a CTA
    // pays one dependent round trip before its loads go out. That happens, and the loads of the pages that hold only
    // tokens older than this step go out, before the grid dependency wait.
    int pid_first = 0;
    int seqlen = 0;
    uint64_t policy = 0;
    if (warp == kConsumerWarps)
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    int issued = 0; // producer lane: its first page is already in flight
    {
        if (warp == kConsumerWarps && lane < NSTAGE)
        {
            int const blk = s + lane * splits;
            if (blk < p.bt_width)
                pid_first = __ldg(p.bt + static_cast<int64_t>(b) * p.bt_stride + blk);
        }
        seqlen = p.seq_lens[b];
        if (warp == kConsumerWarps)
        {
            int const nb0 = seqlen > 0 ? min((seqlen + kBlockK - 1) / kBlockK, p.max_tiles) : 0;
            int const my0 = s < nb0 ? (nb0 - s + splits - 1) / splits : 0;
            int const stable = max(seqlen - p.dql, 0) / kBlockK; // blocks holding only older tokens
            int const blk = s + lane * splits;
            if (lane < min(my0, NSTAGE) && blk < stable)
            {
                mbar_expect_tx(full0 + 8 * lane, kStageBytes);
                tma_load_page(smem_u32(stages + lane * kStageBytes), &tmap, pid_first, full0 + 8 * lane, policy);
                issued = 1;
            }
        }
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");

    // The select's per-row valid-block count, loaded by every CTA alongside seq_lens (only the last one uses it).
    int const sel_tpr = p.dql * p.kv_heads <= 4 ? kConsumerThreads / 4 : kConsumerThreads / 8;
    int const nv_raw = tid < kConsumerThreads && tid / sel_tpr < p.dql * p.kv_heads
        ? p.n_valid[b * p.dql + (tid / sel_tpr) % p.dql]
        : 0;
    int const nb = seqlen > 0 ? min((seqlen + kBlockK - 1) / kBlockK, p.max_tiles) : 0;
    if (nb == 0)
    {
        // Nothing to score: every row of the request is all -1.
        if (s == 0 && tid < p.dql * p.kv_heads * kTopK)
        {
            int const r = tid / kTopK, j = tid % kTopK;
            int const t = b * p.dql + r % p.dql, kvh = r / p.dql;
            p.out[static_cast<int64_t>(kvh) * p.out_h + static_cast<int64_t>(t) * p.out_t + j] = -1;
        }
        return;
    }
    if (s >= nb)
        return;
    int const parts = min(splits, nb);
    int const my = (nb - s + splits - 1) / splits;
    int const q_start = seqlen - p.dql;
    int const group = p.heads / p.kv_heads;
    int const rows = p.dql * p.kv_heads;

    if (warp == kConsumerWarps)
    {
        if (lane < min(my, NSTAGE) && !issued)
        {
            mbar_expect_tx(full0 + 8 * lane, kStageBytes);
            tma_load_page(smem_u32(stages + lane * kStageBytes), &tmap, pid_first, full0 + 8 * lane, policy);
        }
        if (lane == 0)
        {
            for (int j = NSTAGE; j < my; ++j)
            {
                int const st = j % NSTAGE;
                int const page = __ldg(p.bt + static_cast<int64_t>(b) * p.bt_stride + s + j * splits);
                mbar_wait(empty0 + 8 * st, ((j / NSTAGE) - 1) & 1);
                mbar_expect_tx(full0 + 8 * st, kStageBytes);
                tma_load_page(smem_u32(stages + st * kStageBytes), &tmap, page, full0 + 8 * st, policy);
            }
        }
    }
    else
    {
        // Q fragments: column g is (head = g / kMaxDql, row = g % kMaxDql).
        uint32_t qlo[4][2], qhi[4][2];
        {
            int const head = g / kMaxDql, qpos = g % kMaxDql;
            bool const valid = head < p.heads && qpos < p.dql;
            uint8_t const* qrow = p.q + static_cast<int64_t>(b * p.dql + qpos) * p.q_st + head * p.q_sh;
            uint32_t w0[4], w1[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                w0[k] = valid ? *reinterpret_cast<uint32_t const*>(qrow + 32 * k + 4 * tq) : 0u;
                w1[k] = valid ? *reinterpret_cast<uint32_t const*>(qrow + 32 * k + 16 + 4 * tq) : 0u;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                qlo[k][0] = cvt_e4m3x2(w0[k]);
                qhi[k][0] = cvt_e4m3x2(w0[k] >> 16);
                qlo[k][1] = cvt_e4m3x2(w1[k]);
                qhi[k][1] = cvt_e4m3x2(w1[k] >> 16);
            }
        }
        int const slice = warp & 3;
#pragma unroll 1
        for (int lb = warp >> 2; lb < my; lb += kConsumerWarps / 4)
        {
            int const st = lb % NSTAGE;
            mbar_wait(full0 + 8 * st, (lb / NSTAGE) & 1);
            uint8_t const* stg = stages + st * kStageBytes;
            float acc[2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[m][e] = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
#pragma unroll
                for (int m = 0; m < 2; ++m)
                {
                    int const ra = 32 * slice + 16 * m + g; // rb = ra + 8 has the same (row & 7)
                    int const sw = ra & 7;
                    uint8_t const* pa = stg + ra * 128;
                    uint8_t const* pb = pa + 8 * 128;
                    uint32_t const d0 = *reinterpret_cast<uint32_t const*>(pa + (((2 * k) ^ sw) << 4) + 4 * tq);
                    uint32_t const d1 = *reinterpret_cast<uint32_t const*>(pb + (((2 * k) ^ sw) << 4) + 4 * tq);
                    uint32_t const d2 = *reinterpret_cast<uint32_t const*>(pa + (((2 * k + 1) ^ sw) << 4) + 4 * tq);
                    uint32_t const d3 = *reinterpret_cast<uint32_t const*>(pb + (((2 * k + 1) ^ sw) << 4) + 4 * tq);
                    uint32_t const l0 = cvt_e4m3x2(d0), l1 = cvt_e4m3x2(d1), l2 = cvt_e4m3x2(d2), l3 = cvt_e4m3x2(d3);
                    uint32_t const h0 = cvt_e4m3x2(d0 >> 16), h1 = cvt_e4m3x2(d1 >> 16), h2 = cvt_e4m3x2(d2 >> 16),
                                   h3 = cvt_e4m3x2(d3 >> 16);
                    mma16816(acc[m], l0, l1, l2, l3, qlo[k][0], qlo[k][1]);
                    mma16816(acc[m], h0, h1, h2, h3, qhi[k][0], qhi[k][1]);
                }
            }
            __syncwarp();
            if (lane == 0)
                mbar_arrive(empty0 + 8 * st);
            // Causal mask + max over this warp's 32 positions.
            int const blk = s + lb * splits;
            int const pos0 = blk * kBlockK + 32 * slice;
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                int const col = 2 * tq + j;
                int const qabs = q_start + col % kMaxDql;
                float v = -INFINITY;
#pragma unroll
                for (int m = 0; m < 2; ++m)
                {
                    float const a = qabs >= pos0 + 16 * m + g ? acc[m][j] : -INFINITY;
                    float const c = qabs >= pos0 + 16 * m + g + 8 ? acc[m][2 + j] : -INFINITY;
                    v = fmaxf(v, fmaxf(a, c));
                }
#pragma unroll
                for (int o = 4; o < 32; o <<= 1)
                    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
                if (g == 0)
                    epi[(lb * 4 + slice) * kCols + col] = v;
            }
        }
    }
    __syncthreads();

    // Per block: max over the 4 slices, amax over each KV head's index heads, row-major scratch stores.
    for (int lb = tid; lb < my; lb += kThreads)
    {
        int const blk = s + lb * splits;
        float v[kCols];
#pragma unroll
        for (int c = 0; c < kCols; ++c)
        {
            float x = epi[(lb * 4) * kCols + c];
#pragma unroll
            for (int sl = 1; sl < 4; ++sl)
                x = fmaxf(x, epi[(lb * 4 + sl) * kCols + c]);
            v[c] = x;
        }
        float* dst = p.scr + b * p.scr_b + blk;
#pragma unroll
        for (int r = 0; r < kMaxRows; ++r)
        {
            if (r < rows)
            {
                int const kvh = r / p.dql, qpos = r % p.dql;
                float x = -INFINITY;
#pragma unroll
                for (int c = 0; c < kCols; ++c)
                    if (c / kMaxDql >= kvh * group && c / kMaxDql < (kvh + 1) * group && c % kMaxDql == qpos)
                        x = fmaxf(x, v[c]);
                dst[r * p.scr_r] = x;
            }
        }
    }
    __syncthreads();

    __syncthreads();
    if (tid == 0)
    {
        int old;
        asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(old) : "l"(p.counters + b) : "memory");
        s_last = old == parts - 1;
    }
    __syncthreads();
    if (!s_last)
        return;
    asm volatile("griddepcontrol.launch_dependents;");

    // ---- last CTA of request b: top-16 of every (query row, KV head) ----
    // Row r is owned by tpr threads (2 warps for <= 4 rows, 1 warp for <= 8); thread rt of the row takes blocks
    // rt, rt + tpr, ... and stages their forced keys in the (now idle) stage ring.
    if (tid == 0)
        p.counters[b] = 0;
    if (tid < kMaxRows)
        s_cnt[tid] = 0;
    int const tpr = sel_tpr;
    int const r = tid / tpr, rt = tid % tpr;
    bool const active = tid < kConsumerThreads && r < rows;
    int const ldk = (p.max_tiles + 3) & ~3;
    uint32_t* keys = reinterpret_cast<uint32_t*>(stages) + (active ? r : 0) * ldk;
    // One bulk copy per row brings the request's block scores (blocks [0, nb), contiguous per row) into the stage
    // ring, so the whole row arrives in a single L2 round trip.
    uint32_t const selbar = smem_u32(&s_selbar);
    if (tid == 0)
    {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        uint32_t const bytes = static_cast<uint32_t>((nb + 3) & ~3) * 4u;
        mbar_expect_tx(selbar, bytes * rows);
        for (int rr = 0; rr < rows; ++rr)
        {
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                    smem_u32(stages) + static_cast<uint32_t>(rr * ldk) * 4u),
                "l"(p.scr + b * p.scr_b + rr * p.scr_r), "r"(bytes), "r"(selbar)
                : "memory");
        }
    }
    int n = 0, lst = 0;
    uint32_t mx = 0u;
    if (active)
    {
        n = max(0, min(nv_raw, p.max_tiles));
        lst = max(nv_raw - p.local_blocks, 0);
        mbar_wait(selbar, 0);
        float const* rowf = reinterpret_cast<float const*>(keys);
#pragma unroll 1
        for (int i0 = rt; i0 < n; i0 += 8 * tpr)
        {
            float x[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
                x[u] = i0 + u * tpr < n ? rowf[i0 + u * tpr] : 0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
                int const i = i0 + u * tpr;
                if (i < n)
                {
                    uint32_t const k = forced_key(x[u], i, p.init_blocks, lst);
                    keys[i] = k;
                    mx = max(mx, k);
                }
            }
        }
    }
    // T0 = 16th largest thread maximum of the row: 16 threads each hold a key >= T0, so every top-16 key (ties
    // included) is >= T0 (a row with fewer than 16 keys gets T0 = 0, i.e. all of them).
    uint32_t const sorted = warp_sort_desc(mx, lane);
    uint32_t t0 = __shfl_sync(0xFFFFFFFFu, sorted, kTopK - 1);
    if (tpr == 64 && active && lane < kTopK)
        s_wtop[r][rt >> 5][lane] = sorted;
    __syncthreads();
    uint32_t* cand_key = reinterpret_cast<uint32_t*>(epi); // [rows][kCandCap]
    int32_t* cand_blk = reinterpret_cast<int32_t*>(epi) + kMaxRows * kCandCap;
    if (active)
    {
        if (tpr == 64)
        {
            // 16th largest of two descending 16-lists A, B: max over i of min(A[i-1], B[15-i]), A[-1] = B[-1] = inf.
            uint32_t const* A = s_wtop[r][0];
            uint32_t const* B = s_wtop[r][1];
            uint32_t x = 0u;
            if (lane <= kTopK)
                x = min(lane > 0 ? A[lane - 1] : 0xFFFFFFFFu, lane < kTopK ? B[kTopK - 1 - lane] : 0xFFFFFFFFu);
            t0 = __reduce_max_sync(0xFFFFFFFFu, x);
        }
#pragma unroll 1
        for (int i0 = rt; i0 < n; i0 += 8 * tpr)
        {
            uint32_t kk[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
                kk[u] = i0 + u * tpr < n ? keys[i0 + u * tpr] : 0u;
            uint32_t hit = 0;
#pragma unroll
            for (int u = 0; u < 8; ++u)
                hit |= static_cast<uint32_t>(kk[u] >= t0 && kk[u] != 0u) << u;
            if (hit)
            {
#pragma unroll
                for (int u = 0; u < 8; ++u)
                {
                    if ((hit >> u) & 1u)
                    {
                        int const at = atomicAdd(&s_cnt[r], 1);
                        if (at < kCandCap)
                        {
                            cand_key[r * kCandCap + at] = kk[u];
                            cand_blk[r * kCandCap + at] = i0 + u * tpr;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    // One warp per row: exact rank by (key desc, block asc), drop -inf, emit ascending ids with a -1 tail.
    if (warp < rows)
    {
        int const rr = warp;
        int const t = b * p.dql + rr % p.dql, kvh = rr / p.dql;
        int32_t* dst = p.out + static_cast<int64_t>(kvh) * p.out_h + static_cast<int64_t>(t) * p.out_t;
        int const c = s_cnt[rr];
        uint32_t const neg_key = twiddle(-INFINITY);
        if (c <= 32)
        {
            // Lane j ranks candidate j against all of them (smem broadcasts), then places the selected ids by
            // counting the smaller selected ids.
            uint32_t const* ck = cand_key + rr * kCandCap;
            int32_t const* cb = cand_blk + rr * kCandCap;
            uint32_t const kj = lane < c ? ck[lane] : 0u;
            int const bj = lane < c ? cb[lane] : 0x7FFFFFFF;
            int rank = 0;
#pragma unroll 8
            for (int o = 0; o < 32; ++o)
            {
                uint32_t const ko = ck[o];
                int const bo = cb[o];
                rank += (o < c) & ((ko > kj) | ((ko == kj) & (bo < bj)));
            }
            bool const sel = lane < c && rank < kTopK && kj != neg_key;
            int const sb = sel ? bj : 0x7FFFFFFF;
            s_sb[rr][lane] = sb;
            __syncwarp();
            int pos = 0;
#pragma unroll 8
            for (int o = 0; o < 32; ++o)
                pos += s_sb[rr][o] < sb;
            if (sel)
                dst[pos] = bj;
            int const cnt = __popc(__ballot_sync(0xFFFFFFFFu, sel));
            if (lane >= cnt && lane < kTopK)
                dst[lane] = -1;
        }
        else if (c <= kCandCap)
        {
            for (int j = lane; j < c; j += 32)
            {
                uint32_t const kj = cand_key[rr * kCandCap + j];
                int const bj = cand_blk[rr * kCandCap + j];
                int rank = 0;
                for (int o = 0; o < c; ++o)
                {
                    uint32_t const ko = cand_key[rr * kCandCap + o];
                    rank += (ko > kj) || (ko == kj && cand_blk[rr * kCandCap + o] < bj);
                }
                if (rank < kTopK)
                {
                    s_sel_blk[rr][rank] = bj;
                    s_sel_neg[rr][rank] = kj == neg_key;
                }
            }
            __syncwarp();
            bool const has = lane < kTopK && !s_sel_neg[rr][lane & 15];
            int const sk = has ? s_sel_blk[rr][lane & 15] : 0x7FFFFFFF;
            int pos = 0;
#pragma unroll
            for (int l = 0; l < kTopK; ++l)
            {
                int const o = __shfl_sync(0xFFFFFFFFu, sk, l);
                pos += (o < sk) || (o == sk && l < lane);
            }
            if (lane < kTopK)
                dst[pos] = sk == 0x7FFFFFFF ? -1 : sk;
        }
        else
        {
            // Massive exact ties: exact radix select over the staged keys.
            int const nrr = max(0, min(__ldcg(p.n_valid + t), p.max_tiles));
            uint32_t const* kr = reinterpret_cast<uint32_t const*>(stages) + rr * ldk;
            uint32_t* hist = reinterpret_cast<uint32_t*>(stages) + rows * ldk + rr * 256;
            radix_emit(kr, nrr, hist, dst, neg_key);
        }
    }
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// TMA descriptor of the [pages, 128, 128] e4m3 index-K pool (one page per box, 128-byte swizzle), cached per
// (base, pages, strides).
bool make_page_map(CUtensorMap& map, void const* base, int64_t pages, int64_t stride_row, int64_t stride_page)
{
    using Key = std::tuple<uintptr_t, int64_t, int64_t, int64_t>;
    static std::mutex mu;
    static std::map<Key, CUtensorMap> cache;
    Key const key{reinterpret_cast<uintptr_t>(base), pages, stride_row, stride_page};
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end())
    {
        map = it->second;
        return true;
    }
    cuuint64_t dims[3]
        = {static_cast<cuuint64_t>(kHeadDim), static_cast<cuuint64_t>(kBlockK), static_cast<cuuint64_t>(pages)};
    cuuint64_t strides[2] = {static_cast<cuuint64_t>(stride_row), static_cast<cuuint64_t>(stride_page)};
    cuuint32_t box[3] = {kHeadDim, kBlockK, 1};
    cuuint32_t estr[3] = {1, 1, 1};
    CUresult const res = tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(&map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS)
    {
        return false;
    }
    if (cache.size() > 64)
    {
        cache.clear();
    }
    cache.emplace(key, map);
    return true;
}

constexpr size_t kSmemBytes = 1024 + kNStage * kStageBytes + kEpiBytes + 2 * kNStage * 8;

cudaError_t configure()
{
    return cudaFuncSetAttribute(m3_fused_score_select_kernel<kNStage>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
}

} // namespace

cudaError_t invokeMinimaxM3Tp8FusedScoreSelect(MinimaxM3Tp8FusedScoreSelectParams const& params, void const* kPages,
    int64_t numPages, int64_t kRowStride, int64_t kPageStride, int32_t batch, int32_t splits, bool usePdl,
    cudaStream_t stream)
{
    if (batch < 1 || batch > kMinimaxM3Tp8FusedScoreSelectMaxBatch || splits < 1
        || (params.max_tiles + splits - 1) / splits > kMaxLocal)
    {
        return cudaErrorInvalidValue;
    }
    CUtensorMap map;
    if (!make_page_map(map, kPages, numPages, kRowStride, kPageStride))
    {
        return cudaErrorInvalidValue;
    }
    // Kernel attributes are set once per process, on the first launch (outside CUDA-graph capture: warmup runs the
    // kernel eagerly first).
    static cudaError_t const configured = configure();
    if (configured != cudaSuccess)
    {
        return configured;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(splits, batch);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmemBytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = usePdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, m3_fused_score_select_kernel<kNStage>, map, params);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
