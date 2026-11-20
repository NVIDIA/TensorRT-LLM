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

// FSEL top-16 block select for MiniMax-M3 decode: page_topk_nt512r4.cu (low-latency reference,
// md5 c0fbfd7d, topk.page-top16-nt512r4, the fsel_kernel<512, 4, false, 4> his TP4 trace runs) ported as is. The body
// is his file verbatim except the lines marked PORT:
//   1. the score row is read at a block stride (our scorer writes [num_kv_heads, num_blocks, total_q], block-major);
//   2. the local window is placed from the unclamped valid count (the M3 select contract, as m3_select_long);
//   3. kMaxCap 8192 -> 9216 (m3_select_long's MAX_BLOCKS: 1M tokens + draft rows at 128 tokens per block);
//   4. a trigger phase (0 = his release right after the wait; 1 = after the row's keys are in registers; 2 = at exit);
//   5. no torch binding (csrc/m31/fselTopkOp.cpp), namespaces, and the invokeFselTopk entry;
//   6. the table stores follow the grid wait (his forced / short-row ids were stored before it). Our index scorer
//      releases this grid before its own wait, so a pre-wait store could land while older kernels still run; the
//      loads (n_valid: final before the step's first layer; the static block table) stay before the wait.
// PORT 7: the per-thread include masks are 32-bit (incm / lm / km: bit j of key j), so a body takes at most 32
//   keys per thread. Raising the cap to 9216 (3) gave the 256-thread instance's above-4096 body 36 keys (bits 32..35:
//   shifts past the mask width, undefined; rows went unwritten in the 2.7 pre-check UT, variant 5 only). That body is
//   capped back at his 32 keys (8192 pages at 256 threads, his original instance), the launcher runs a variant-5 call
//   over more than 8192 blocks on the 512-thread instance (18 keys, unchanged), and fsel_body asserts KPT <= 32.
// PORT(i8 MS4) 8: rows above 4096 blocks may run over a 4-CTA cluster (variant 7; his page_topk_nt512r4_ms.cu FSEL_MS
// 2,
//   reference r8.4, lines 24-36): each CTA of the row's cluster ranks one quarter of the row (the NT-512 body, at most
//   2304 blocks, 5 keys per thread), writes its top-`need` as (key << 32 | ~block, block id) slots into the leader's
//   shared memory over DSMEM, and after one cluster barrier the leader ranks the 4 x 16 candidates by (key desc, block
//   asc) and stores the table. Exact: the row's top-`need` lies in the union of the chunks' top-`need` sets, and the
//   merge uses the same packed order as the one-CTA rank. Ported with the lines above: the block-stride score reads
//   (1), the unclamped local window (2), the post-wait forced stores (6) and the trigger phases (4); rows of at most
//   4096 blocks run the one-CTA path in the cluster's first CTA (the others exit at entry), so their table and their
//   code are those of variant 6. His ticket mode (FSEL_MS 1, global slots) is not ported.
// His header follows.
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// page_topk_nt512.cu with the certified pool ranked by 4 threads per entry (16 compares each) instead of one warp's
// 32 / 64 serial compares, and the 512-thread body only above 4096 pages; topk.page-top16-nt512r4. Same table.
// Exact indexer top-16 page selector for sparse-attention decode, one CTA per (token, head) row: witness-bound
// pool, exact rank (key desc, page asc), forced pages always selected, radix fallback on pool overflow.
// Table: [H, nd, 16] main-KV page rows in ascending page order, -1 tail when nv <= 16.
#include "minimax_m3/csrc/m31/fselTopkSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace fsel
{
constexpr int kTopk = 16;
constexpr int kMaxCap = kFselTopkMaxBlocks; // PORT 3: 9216 (his 8192); every page below
                                            // min(n_valid, num_cols, kMaxCap) is selectable
constexpr int kPool = 64;
constexpr uint32_t kFull = 0xffffffffu;
#ifndef FSEL_MUT
#define FSEL_MUT 0
#endif

__device__ __forceinline__ void pdl_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ long long gtimer()
{
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// order-preserving key of the fp32 bits
__device__ __forceinline__ uint32_t okey(uint32_t raw)
{
    return (raw & 0x80000000u) ? ~raw : (raw | 0x80000000u);
}

struct Smem
{
    int spid[kMaxCap];          // main-KV page rows of the row's token, staged pre-wait
    unsigned long long pk[256]; // pool: key << 32 | ~page (larger = better); V2: 32 per warp
    uint32_t hist[256 + 4];     // radix scratch; [256..259] ctl
    uint32_t ckey[32];
    int cpg[32];
    uint32_t ws[32]; // per-warp witness / max
    int wsel[kTopk]; // winners' pages (any order)
    int wcnt[32];    // V2: per-warp pool segment counts
    int cnt;
};

// Winners' pages in s.wsel[0..need) (written, then made visible to warp 0) -> page-order store at orw[base + pos].
__device__ __forceinline__ void store_page_order(Smem& s, int need, int base, int* orw, int lane)
{
    int const pg = lane < need ? s.wsel[lane] : 0x7fffffff;
    int pos = 0;
#pragma unroll
    for (int j = 0; j < kTopk; ++j)
        pos += __shfl_sync(kFull, pg, j) < pg;
    if (lane < need)
        orw[base + pos] = s.spid[pg];
}

// PORT(i8 MS4) 8: the cluster split above 4096 blocks (variant 7).
constexpr int kMsK = 4;                              // CTAs per long row (one cluster)
constexpr int kMsMin = kFselTopkMsMinBlocks;         // rows above this many blocks split (the NT-512 tier)
constexpr int kMsSpan = (kMaxCap + kMsK - 1) / kMsK; // blocks per chunk at most (2304)
constexpr int kMsN = kMsK * kTopk;                   // merge candidates
static_assert(kMsMin == 16 * 256, "the split starts where the one-CTA path takes its NT-512 body");
static_assert(kMsN == kPool, "the merge ranks exactly one zero-free pool");
static_assert(kMsSpan <= kMsMin, "a chunk fits the staged block rows");

struct MsSmem
{
    unsigned long long mwin[kTopk]; // chunk winners' packed entries (pool path); 0 = reload the key (radix path)
    unsigned long long mcand[kMsN]; // leader: the cluster's candidates, written over DSMEM (chunk c at [16c, 16c+16))
    int mpid[kMsN];                 // leader: their block ids (page-table entries)
};

struct MsCtx
{
    float const* srw; // chunk score base (row base + lo blocks), for the radix-path winners' keys
    long long ps;     // score block stride (PORT 1)
    int* orw;
    MsSmem* ms;
    int lo, chunk, flo, need, need_c;
};

__device__ __forceinline__ uint32_t ms_dsmem(void const* p, uint32_t rank)
{
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p)), r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}

__device__ __forceinline__ void ms_cluster_arrive() // phase 1: this CTA has started (DSMEM may target it)
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void ms_cluster_wait()
{
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void ms_cluster_sync() // phase 2: slots written -> leader
{
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Chunk winners (local positions in s.wsel[0..need_c), packed entries in ms.mwin when the pool path found them) ->
// the leader's slots; the leader merges and stores the table. Called by every thread of every CTA of the cluster.
template <int NT, bool ST>
__device__ __forceinline__ void ms_tail(Smem& s, MsCtx const& m, int tid, long long* sr)
{
    static_assert(NT >= 4 * kPool, "4 threads per merge candidate");
    MsSmem& ms = *m.ms;
    __syncthreads();
    unsigned long long v = 0ull; // an empty slot (a real entry is never 0: its low word ~block has the top bit set)
    int pid = 0;
    if (tid < kTopk && tid < m.need_c)
    {
        int const p = s.wsel[tid];
        v = ms.mwin[tid];
        if (v == 0ull) // radix fallback: the key is reloaded (the pool path records it)
            v = static_cast<unsigned long long>(
                    okey(__ldcg(reinterpret_cast<uint32_t const*>(m.srw) + static_cast<long long>(p) * m.ps)))
                << 32;
        v = (v & 0xffffffff00000000ull) | static_cast<uint32_t>(~static_cast<uint32_t>(m.lo + p));
        pid = s.spid[p];
    }
    ms_cluster_wait(); // every CTA of the cluster has started (phase 1, arrived at entry)
    if (tid < kTopk)
    {
        uint32_t const a = ms_dsmem(&ms.mcand[m.chunk * kTopk + tid], 0);
        uint32_t const b = ms_dsmem(&ms.mpid[m.chunk * kTopk + tid], 0);
        asm volatile("st.shared::cluster.u64 [%0], %1;" ::"r"(a), "l"(v) : "memory");
        asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(b), "r"(pid) : "memory");
    }
    ms_cluster_sync();
    if constexpr (ST)
        if (tid == 0)
            sr[5] = gtimer();
    if (m.chunk != 0)
        return;
    if (tid < kTopk)
        s.wsel[tid] = 0x7fffffff;
    __syncthreads();
    if (tid < 4 * kPool)
    {
        int const c = tid >> 2, q = tid & 3;
        unsigned long long const ec = ms.mcand[c];
        int r = 0;
#pragma unroll
        for (int t = 0; t < kPool / 4; ++t)
            r += ms.mcand[q * (kPool / 4) + t] > ec;
        r += __shfl_xor_sync(kFull, r, 1);
        r += __shfl_xor_sync(kFull, r, 2);
        if (q == 0 && ec != 0ull && r < m.need)
            s.wsel[r] = c;
    }
    __syncthreads();
    if (tid < 32)
    {
        // winners' candidate slots -> block order (position = ~low word), block ids from the slots
        int const ci = tid < m.need ? s.wsel[tid] : 0;
        int const pg = tid < m.need ? static_cast<int>(~static_cast<uint32_t>(ms.mcand[ci])) : 0x7fffffff;
        int pos = 0;
#pragma unroll
        for (int j = 0; j < kTopk; ++j)
            pos += __shfl_sync(kFull, pg, j) < pg;
        if (tid < m.need)
            m.orw[m.flo + pos] = ms.mpid[ci];
        if constexpr (ST)
            if (tid == 0)
                sr[5] = gtimer();
    }
}

// Exact top-`need` (key desc, page asc) among the included keys (bit j of incm), block-wide MSD radix select with the
// block_select16 structure: 8-bit passes until <= 32 candidates, mass-tie path in page
// order. Winners' pages -> s.wsel[0..need). Called by every thread; ends with warp 0 holding the result (synced).
template <int NT, int KPT>
__device__ __forceinline__ void radix_select(const uint32_t (&k)[KPT], uint32_t incm, int need, Smem& s, int tid)
{
    int const warp = tid >> 5, lane = tid & 31;
    uint32_t* const hist = s.hist;
    uint32_t* const ctl = s.hist + 256; // [0] prefix, [1] remaining rank, [2] stop, [3] candidate count
    uint32_t prefix = 0u, krem = static_cast<uint32_t>(need);
    int sh = 24;
    __syncthreads();
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass, sh -= 8)
    {
        for (int i = tid; i < 256; i += NT)
            hist[i] = 0u;
        if (tid == 0)
            ctl[3] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < KPT; ++j)
            if (((incm >> j) & 1u) && (pass == 0 || (k[j] >> (sh + 8)) == prefix))
                atomicAdd(&hist[(k[j] >> sh) & 255u], 1u);
        __syncthreads();
        if (warp == 0)
        {
            uint32_t c[8], sum = 0u;
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
                uint32_t const t = __shfl_up_sync(kFull, incl, o);
                if (lane >= o)
                    incl += t;
            }
            uint32_t a = incl - sum;
            if (a < krem && incl >= krem)
            {
                int bin = -1;
                uint32_t above = 0u, inb = 0u;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    if (bin < 0 && a + c[j] >= krem)
                    {
                        bin = 255 - 8 * lane - j;
                        above = a;
                        inb = c[j];
                    }
                    a += c[j];
                }
                uint32_t const kr = krem - above;
                ctl[0] = (prefix << 8) | static_cast<uint32_t>(bin);
                ctl[1] = kr;
                ctl[2] = (static_cast<uint32_t>(need) - kr) + inb <= 32u ? 1u : 0u;
            }
        }
        __syncthreads();
        prefix = ctl[0];
        krem = ctl[1];
        if (ctl[2] != 0u)
            break;
    }
    auto take = [&](int j)
    {
        uint32_t const at = atomicAdd(&ctl[3], 1u);
        s.ckey[at] = k[j];
        s.cpg[at] = tid + NT * j;
    };
    if (sh >= 0)
    {
        uint32_t const thr = prefix << sh;
#pragma unroll
        for (int j = 0; j < KPT; ++j)
            if (((incm >> j) & 1u) && k[j] >= thr)
                take(j);
    }
    else
    {
        uint32_t const T = prefix;
        int const nt = static_cast<int>(krem);
#pragma unroll
        for (int j = 0; j < KPT; ++j)
            if (((incm >> j) & 1u) && k[j] > T)
                take(j);
        int run = 0;
#pragma unroll
        for (int j = 0; j < KPT; ++j)
        {
            if (run >= nt) // block-uniform
                break;
            bool const tie = ((incm >> j) & 1u) && k[j] == T;
            uint32_t const bm = __ballot_sync(kFull, tie);
            __syncthreads();
            if (lane == 0)
                hist[warp] = static_cast<uint32_t>(__popc(bm));
            __syncthreads();
            int woff = 0, tot = 0;
            for (int w = 0; w < NT / 32; ++w)
            {
                int const cw = static_cast<int>(hist[w]);
                woff += w < warp ? cw : 0;
                tot += cw;
            }
            if (tie && run + woff + __popc(bm & ((1u << lane) - 1u)) < nt)
                take(j);
            run += tot;
        }
    }
    __syncthreads();
    if (warp == 0)
    {
        int const nc = static_cast<int>(ctl[3]);
        uint32_t const ki = lane < nc ? s.ckey[lane] : 0u;
        int const pi = lane < nc ? s.cpg[lane] : 0x7fffffff;
        int r = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j)
        {
            uint32_t const kj = __shfl_sync(kFull, ki, j);
            int const pj = __shfl_sync(kFull, pi, j);
            r += kj > ki || (kj == ki && pj < pi);
        }
        bool const win = lane < nc && r < need;
        uint32_t const bw = __ballot_sync(kFull, win);
        if (win)
            s.wsel[__popc(bw & ((1u << lane) - 1u))] = pi;
        __syncwarp();
    }
}

// FSEL body for one row; KPT = keys per thread (cap <= NT * KPT). R = witness lanes per warp.
template <int NT, int R, int KPT, bool ST, int V, bool MS = false>
__device__ __forceinline__ void fsel_body(Smem& s, float const* srw, int* orw, int cap, int init, int fend, int flo,
    int need, int tid, int* diag, long long* sr, long long ps, int trig,
    MsCtx const* mc = nullptr) // PORT(i8 MS4) 8: MS = one chunk of a split row
{
    static_assert(KPT >= 1 && KPT <= 32, "PORT 7: the include masks are 32-bit (one bit per key)");
    static_assert(!MS || (V == 4 && NT >= 4 * kPool), "PORT(i8 MS4) 8: a chunk ranks its pool 4 threads per entry");
    constexpr int W = NT / 32;
    int const warp = tid >> 5, lane = tid & 31;
    uint32_t k[KPT];
    // one batch of independent loads (the predicate depends only on pre-wait data)
#pragma unroll
    for (int j = 0; j < KPT; ++j)
    {
        int const p = tid + NT * j;
        k[j] = p < cap ? __ldcg(reinterpret_cast<uint32_t const*>(srw) + p * ps) : 0u; // PORT 1: stride
    }
    uint32_t incm = 0u, lm = 0u;
#pragma unroll
    for (int j = 0; j < KPT; ++j)
    {
        int const p = tid + NT * j;
        k[j] = okey(k[j]);
        bool const v = p < cap && p >= init && p < fend; // non-forced, selectable
        incm |= v ? (1u << j) : 0u;
        lm = v ? max(lm, k[j]) : lm;
    }
    if (trig == 1) // PORT 4: release once the row's keys are in registers
        pdl_trigger();
    // per-warp top-R lane maxima (one lane removed per round): S = R-th largest
    uint32_t cur = lm, S = 0u;
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        uint32_t const m = __reduce_max_sync(kFull, cur);
        S = m;
        if (r + 1 < R)
        {
            uint32_t const b = __ballot_sync(kFull, cur == m);
            if (lane == __ffs(b) - 1)
                cur = 0u;
        }
    }
    if (lane == 0)
        s.ws[warp] = S;
    if constexpr (ST)
        if (tid == 0)
            sr[2] = gtimer();
    __syncthreads();
    // T = K-th largest S_w, K = ceil(need / R): K warps x R distinct threads hold keys >= T
    int const K = (need + R - 1) / R;
    uint32_t const sv = lane < W ? s.ws[lane] : 0u;
    int c = 0;
    if constexpr (V == 4)
    {
#pragma unroll
        for (int v = 0; v < W; ++v)
            c += s.ws[v] >= sv;
    }
    else
    {
#pragma unroll
        for (int v = 0; v < W; ++v)
            c += __shfl_sync(kFull, sv, v) >= sv;
    }
#if FSEL_MUT == 1 // negative control: too few witnesses (K - 2 warps) and no pool certificate
    int const Km = max(1, K - 2);
    int cm = 0;
#pragma unroll
    for (int v = 0; v < W; ++v)
        cm += __shfl_sync(kFull, sv, v) >= sv;
    uint32_t const T = __reduce_max_sync(kFull, (lane < W && cm >= Km) ? sv : 0u);
#else
    uint32_t const T = K > W ? 0u : __reduce_max_sync(kFull, (lane < W && c >= K) ? sv : 0u);
#endif
    uint32_t km = 0u;
#pragma unroll
    for (int j = 0; j < KPT; ++j)
        km |= (((incm >> j) & 1u) && k[j] >= T) ? (1u << j) : 0u;
    if constexpr (V == 2)
    {
        // V2 filter + compaction: per-warp segment of 32 pool slots, slot = ballot prefix (no atomics)
        uint32_t const lt = (1u << lane) - 1u;
        int base = 0;
#pragma unroll
        for (int j = 0; j < KPT; ++j)
        {
            bool const c = (km >> j) & 1u;
            uint32_t const b = __ballot_sync(kFull, c);
            if (c && base + __popc(b & lt) < 32)
                s.pk[32 * warp + base + __popc(b & lt)] = (static_cast<unsigned long long>(k[j]) << 32)
                    | static_cast<uint32_t>(~static_cast<uint32_t>(tid + NT * j));
            base += __popc(b);
        }
        if (lane == 0)
            s.wcnt[warp] = base;
        if constexpr (ST)
            if (tid == 0)
                sr[3] = gtimer();
        __syncthreads();
        int wc[W], n = 0, mx = 0;
#pragma unroll
        for (int v = 0; v < W; ++v)
        {
            wc[v] = s.wcnt[v];
            n += wc[v];
            mx = max(mx, wc[v]);
        }
        if (mx <= 32 && (n >= need || FSEL_MUT == 1)) // certificate (block-uniform)
        {
            if constexpr (ST)
                if (tid == 0)
                    sr[4] = gtimer();
            // every warp ranks its own segment against the whole pool (smem broadcast reads)
            int const myc = wc[warp];
            if (myc != 0)
            {
                unsigned long long const ei = lane < myc ? s.pk[32 * warp + lane] : ~0ull;
                int r = 0;
#pragma unroll
                for (int v = 0; v < W; ++v)
                {
#pragma unroll 4
                    for (int i = 0; i < wc[v]; ++i)
                    {
                        unsigned long long const x = s.pk[32 * v + i];
#if FSEL_MUT == 2 // negative control: ties broken toward the larger page
                        r += (x >> 32) > (ei >> 32) || ((x >> 32) == (ei >> 32) && x < ei);
#else
                        r += x > ei;
#endif
                    }
                }
                if (lane < myc && r < need)
                    s.wsel[r] = static_cast<int>(~static_cast<uint32_t>(ei));
            }
            if constexpr (ST)
                if (tid == 0)
                    sr[6] = gtimer();
            __syncthreads();
            if (warp == 0)
            {
                store_page_order(s, need, flo, orw, lane);
                if (diag != nullptr && lane == 0)
                    atomicAdd(diag + (mx <= 16 ? 0 : 1), 1);
                if constexpr (ST)
                    if (lane == 0)
                        sr[5] = gtimer();
            }
            return;
        }
    }
    else
    {
        if (km != 0u)
        {
            int at = atomicAdd(&s.cnt, __popc(km));
#pragma unroll
            for (int j = 0; j < KPT; ++j)
                if ((km >> j) & 1u)
                {
                    if (at < kPool)
                        s.pk[at] = (static_cast<unsigned long long>(k[j]) << 32)
                            | static_cast<uint32_t>(~static_cast<uint32_t>(tid + NT * j));
                    ++at;
                }
        }
        if constexpr (ST)
            if (tid == 0)
                sr[3] = gtimer();
        __syncthreads();
        int const n = s.cnt;
        if (n <= kPool && (n >= need || FSEL_MUT == 1)) // certificate (block-uniform)
        {
            if constexpr (V == 4 && NT >= 4 * kPool)
            {
                // 4 threads per pool entry, 16 compares each (the pool is zero-padded to kPool; 0 never beats a real
                // entry); winners land at wsel[rank], a permutation of the same set (the store below orders by page)
                if constexpr (ST)
                    if (tid == 0)
                        sr[4] = gtimer();
                if (tid < 4 * kPool)
                {
                    int const c = tid >> 2, q = tid & 3;
                    unsigned long long const ec = s.pk[c];
                    int r = 0;
#pragma unroll
                    for (int t = 0; t < kPool / 4; ++t)
                        r += s.pk[q * (kPool / 4) + t] > ec;
                    r += __shfl_xor_sync(kFull, r, 1);
                    r += __shfl_xor_sync(kFull, r, 2);
                    if (q == 0 && c < n && r < need)
                        s.wsel[r] = static_cast<int>(~static_cast<uint32_t>(ec));
                    if constexpr (MS) // PORT(i8 MS4) 8: the winner's packed entry (key, ~local block) for the merge
                        if (q == 0 && c < n && r < need)
                            mc->ms->mwin[r] = ec;
                }
                if constexpr (ST)
                    if (tid == 0)
                        sr[6] = gtimer();
                if constexpr (MS)
                {
                    ms_tail<NT, ST>(s, *mc, tid, sr);
                    return;
                }
                __syncthreads();
                if (warp != 0)
                    return;
                if constexpr (ST)
                    if (lane == 0)
                        sr[7] = gtimer();
                int const pg = lane < need ? s.wsel[lane] : 0x7fffffff;
                int pos = 0;
#pragma unroll
                for (int j = 0; j < kTopk; ++j)
                    pos += s.wsel[j] < pg;
                if (lane < need)
                    orw[flo + pos] = s.spid[pg];
                if (diag != nullptr && lane == 0)
                    atomicAdd(diag + (n <= 32 ? 0 : 1), 1);
                if constexpr (ST)
                    if (lane == 0)
                        sr[5] = gtimer();
                return;
            }
            if (warp != 0)
                return;
            if constexpr (ST)
                if (lane == 0)
                    sr[4] = gtimer();
            unsigned long long const e0 = lane < n ? s.pk[lane] : 0ull;
            unsigned long long const e1 = lane + 32 < n ? s.pk[lane + 32] : 0ull;
            int r0 = 0, r1 = 0;
            if constexpr (V == 4)
            {
                if (n <= 32)
                {
#pragma unroll
                    for (int t = 0; t < 32; ++t)
#if FSEL_MUT == 2
                        r0 += (s.pk[t] >> 32) > (e0 >> 32) || ((s.pk[t] >> 32) == (e0 >> 32) && s.pk[t] < e0);
#else
                        r0 += s.pk[t] > e0;
#endif
                }
                else
                {
#pragma unroll
                    for (int t = 0; t < kPool; ++t)
                    {
                        unsigned long long const x = s.pk[t];
                        r0 += x > e0;
                        r1 += x > e1;
                    }
                }
                if constexpr (ST)
                    if (lane == 0)
                        sr[6] = gtimer();
            }
            else if constexpr (V == 3)
            {
                // fixed trip counts, fully unrolled (no data-dependent loop); invalid entries are 0 and never count
                if (n <= 32)
                {
#pragma unroll
                    for (int t = 0; t < 32; ++t)
                        r0 += __shfl_sync(kFull, e0, t) > e0;
                }
                else
                {
#pragma unroll
                    for (int t = 0; t < 32; ++t)
                    {
                        unsigned long long const x = __shfl_sync(kFull, e0, t), y = __shfl_sync(kFull, e1, t);
                        r0 += (x > e0) + (y > e0);
                        r1 += (x > e1) + (y > e1);
                    }
                }
                if constexpr (ST)
                    if (lane == 0)
                        sr[6] = gtimer();
            }
            else
            {
                if (n <= 32)
                {
#pragma unroll 4
                    for (int t = 0; t < n; ++t)
#if FSEL_MUT == 2 // negative control: ties broken toward the larger page
                        r0 += (__shfl_sync(kFull, e0, t) >> 32) > (e0 >> 32)
                            || ((__shfl_sync(kFull, e0, t) >> 32) == (e0 >> 32) && __shfl_sync(kFull, e0, t) < e0);
#else
                        r0 += __shfl_sync(kFull, e0, t) > e0;
#endif
                }
                else
                {
#pragma unroll 4
                    for (int t = 0; t < 32; ++t)
                    {
                        unsigned long long const x = __shfl_sync(kFull, e0, t);
                        r0 += x > e0;
                        r1 += x > e1;
                    }
#pragma unroll 4
                    for (int t = 0; t < n - 32; ++t)
                    {
                        unsigned long long const x = __shfl_sync(kFull, e1, t);
                        r0 += x > e0;
                        r1 += x > e1;
                    }
                }
            }
            bool const w0 = lane < n && r0 < need, w1 = lane + 32 < n && r1 < need;
            uint32_t const b0 = __ballot_sync(kFull, w0), b1 = __ballot_sync(kFull, w1);
            uint32_t const lt = (1u << lane) - 1u;
            if (w0)
                s.wsel[__popc(b0 & lt)] = static_cast<int>(~static_cast<uint32_t>(e0));
            if (w1)
                s.wsel[__popc(b0) + __popc(b1 & lt)] = static_cast<int>(~static_cast<uint32_t>(e1));
            __syncwarp();
            if constexpr (ST && (V == 3 || V == 4))
                if (lane == 0)
                    sr[7] = gtimer();
            if constexpr (V == 4)
            {
                int const pg = lane < need ? s.wsel[lane] : 0x7fffffff;
                int pos = 0;
#pragma unroll
                for (int j = 0; j < kTopk; ++j)
                    pos += s.wsel[j] < pg;
                if (lane < need)
                    orw[flo + pos] = s.spid[pg];
            }
            else
                store_page_order(s, need, flo, orw, lane);
            if (diag != nullptr && lane == 0)
                atomicAdd(diag + (n <= 32 ? 0 : 1), 1);
            if constexpr (ST)
                if (lane == 0)
                    sr[5] = gtimer();
            return;
        }
    }
    radix_select<NT, KPT>(k, incm, need, s, tid); // exact fallback on the register keys (no reload)
    if constexpr (MS)
    {
        ms_tail<NT, ST>(s, *mc, tid, sr);
        return;
    }
    if (warp == 0)
    {
        store_page_order(s, need, flo, orw, lane);
        if (diag != nullptr && lane == 0)
            atomicAdd(diag + 2, 1);
        if constexpr (ST)
            if (lane == 0)
                sr[5] = gtimer();
    }
}

// PORT(i8 MS4) 8: one chunk of a long row (cap > kMsMin, need > 0): the chunk's block rows staged pre-wait, the forced
// blocks by chunk 0 after the wait (PORT 6), then the NT-512 body over [lo, hi) with need_c = min(need,
// selectable blocks in the chunk), then ms_tail.
template <int NT, int R, bool ST, int V>
__device__ __forceinline__ void ms_row(Smem& s, MsSmem& ms, float const* srw, long long ps, int const* prw, int* orw,
    int cap, int init, int fend, int flo, int fhi, int need, int chunk, int tid, long long* sr, int trig)
{
    int const span = (cap + kMsK - 1) / kMsK;
    int const lo = min(cap, chunk * span), hi = min(cap, lo + span);
    int const nsel = max(0, min(hi, fend) - max(lo, init));
    int const need_c = min(need, nsel);
    ms_cluster_arrive();
    for (int p = tid; p < hi - lo; p += NT)
        s.spid[p] = __ldcg(prw + lo + p);
    int const fpg = chunk != 0 ? 0
        : tid < flo            ? __ldcg(prw + tid)
                               : (tid >= kTopk - fhi && tid < kTopk ? __ldcg(prw + cap - (kTopk - tid)) : 0);
    if (tid == 0)
        s.cnt = 0;
    if (tid < kPool)
        s.pk[tid] = 0ull;
    if (tid < kTopk)
    {
        s.wsel[tid] = 0x7fffffff;
        ms.mwin[tid] = 0ull;
    }
    __syncthreads();
    pdl_wait();
    if (chunk == 0 && (tid < flo || (tid >= kTopk - fhi && tid < kTopk))) // PORT 6
        orw[tid] = fpg;
    if (trig == 0)                                                        // PORT 4
        pdl_trigger();
    if constexpr (ST)
        if (tid == 0)
            sr[1] = gtimer();
    float const* const crw = srw + static_cast<long long>(lo) * ps;
    MsCtx const m{crw, ps, orw, &ms, lo, chunk, flo, need, need_c};
    if (need_c > 0)
        fsel_body<NT, R, (kMsSpan + NT - 1) / NT, ST, V, true>(
            s, crw, orw, hi - lo, init - lo, fend - lo, flo, need_c, tid, nullptr, sr, ps, trig, &m);
    else
    {
        if (trig == 1) // PORT 4: no keys to load
            pdl_trigger();
        ms_tail<NT, ST>(s, m, tid, sr);
    }
}

// PORT(i8 MS4) 8: fsel_kernel's row body for the cluster kernel (a copy; fsel_kernel itself is unchanged, so variants
// 5 and 6 compile as before): the CTA is chunk `chunk` of the row's 4-CTA cluster; rows above kMsMin blocks with a
// selectable block split (ms_row), shorter rows (and all-forced rows) run the one-CTA path in chunk 0 and the other
// chunks exit at entry. Lines that differ from fsel_kernel are marked MSK.
template <int NT, int R, bool ST, int V>
__device__ __forceinline__ void fsel_ms_row(Smem& s, MsSmem* ms, int row, int chunk, float const* __restrict__ sc,
    long long sc_st, long long sc_sh, long long sc_ps, int const* n_valid, int const* pt, long long pt_s0, int num_cols,
    int* out, long long out_sh, long long out_st, int H, int init, int local, int trig, int* diag, long long* sr)
{
    int const hh = row % H, tt = row / H, tid = threadIdx.x;
    float const* const srw = sc + tt * sc_st + hh * sc_sh;
    int const* const prw = pt + tt * pt_s0;
    int* const orw = out + hh * out_sh + tt * out_st;
    if constexpr (ST)
        if (tid == 0)
            sr[0] = gtimer();
    // ---- pre-wait: n_valid and page rows are front outputs (K-BND E3, as landed topkpw) ----
    int const nraw = __ldcg(n_valid + tt); // PORT 2: the unclamped count places the local window
    int const nvv = min(nraw, num_cols);
    if (chunk != 0 && nvv <= kMsMin)       // MSK
        return;                            // short row: chunk 0 runs the one-CTA path alone
    if (nvv <= kTopk)
    {
        int const v = tid < kTopk && tid < nvv ? __ldcg(prw + tid) : -1;
        pdl_wait(); // PORT 6: the store follows the wait (his stored before it)
        if (tid < kTopk)
            orw[tid] = v;
        if (diag != nullptr && tid == 0)
            atomicAdd(diag + 3, 1);
        if (trig != 2) // PORT 4
            pdl_trigger();
        return;
    }
    int const cap = min(nvv, kMaxCap);
    int const fend = nraw - local; // PORT 2 (his: nvv - local)
    int const flo = min(init, cap), fhi = max(0, cap - max(fend, flo));
    int const need = kTopk - flo - fhi;
    if (cap > kMsMin) // MSK
    {
        if (need > 0)
        {
            ms_row<NT, R, ST, V>(s, *ms, srw, sc_ps, prw, orw, cap, init, fend, flo, fhi, need, chunk, tid, sr, trig);
            return;
        }
        if (chunk != 0)
            return; // all forced: chunk 0 copies them as the one-CTA path
    }
    for (int p = tid; p < cap; p += NT)
        s.spid[p] = __ldcg(prw + p);
    if (need <= 0)
    {
        // more forced pages than slots: the lowest 16 forced pages (all keys tie at the forced key)
        int const v = tid < kTopk ? __ldcg(prw + (tid < flo ? tid : cap - fhi + (tid - flo))) : 0;
        pdl_wait(); // PORT 6: the store follows the wait (his stored before it)
        if (tid < kTopk)
            orw[tid] = v;
        if (trig != 2) // PORT 4
            pdl_trigger();
        return;
    }
    // forced pages are the extreme pages: init at positions [0, flo), local at [16 - fhi, 16)
    // PORT 6: the forced ids are loaded here and stored after the wait (his stored them here)
    int const fpg
        = tid < flo ? __ldcg(prw + tid) : (tid >= kTopk - fhi && tid < kTopk ? __ldcg(prw + cap - (kTopk - tid)) : 0);
    if (tid == 0)
        s.cnt = 0;
    if (tid < kPool)
        s.pk[tid] = 0ull; // V4 ranks over a fixed, zero-padded pool (0 never beats a real entry)
    if (tid < kTopk)
        s.wsel[tid] = 0x7fffffff;
    __syncthreads();
    pdl_wait();
    if (tid < flo || (tid >= kTopk - fhi && tid < kTopk)) // PORT 6
        orw[tid] = fpg;
    if (trig == 0)                                        // PORT 4: his release point
        pdl_trigger();
    if constexpr (ST)
        if (tid == 0)
            sr[1] = gtimer();
    // at most 4096 pages: the 256-thread instantiation (threads 256.. of a 512-thread launch leave here; exited threads
    // count as arrived at every later barrier); above 4096 pages all NT threads (16 keys each at NT 512)
    if (cap <= 16 * 256)
    {
        if (tid >= 256)
            return;
        if (cap <= 6 * 256)
            fsel_body<256, R, 6, ST, V>(s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
        else
            fsel_body<256, R, 16, ST, V>(s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
    }
    else
        fsel_body<NT, R, ((kMaxCap + NT - 1) / NT < 32 ? (kMaxCap + NT - 1) / NT : 32), ST, V>( // PORT 7
            s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
}

template <int NT, int R, bool ST, int V = 2>
__global__ void __launch_bounds__(NT, 1) fsel_kernel(float const* __restrict__ sc, long long sc_st, long long sc_sh,
    long long sc_ps, int const* n_valid, int const* pt, long long pt_s0, int num_cols, int* out, long long out_sh,
    long long out_st, int H, int init, int local, int trig, int* diag, long long* st)
{
    __shared__ Smem s;
    int const row = blockIdx.x, hh = row % H, tt = row / H, tid = threadIdx.x;
    float const* const srw = sc + tt * sc_st + hh * sc_sh;
    int const* const prw = pt + tt * pt_s0;
    int* const orw = out + hh * out_sh + tt * out_st;
    long long* const sr = ST ? st + 8 * row : nullptr;
    if constexpr (ST)
        if (tid == 0)
            sr[0] = gtimer();
    // ---- pre-wait: n_valid and page rows are front outputs (K-BND E3, as landed topkpw) ----
    int const nraw = __ldcg(n_valid + tt); // PORT 2: the unclamped count places the local window
    int const nvv = min(nraw, num_cols);
    if (nvv <= kTopk)
    {
        int const v = tid < kTopk && tid < nvv ? __ldcg(prw + tid) : -1;
        pdl_wait(); // PORT 6: the store follows the wait (his stored before it)
        if (tid < kTopk)
            orw[tid] = v;
        if (diag != nullptr && tid == 0)
            atomicAdd(diag + 3, 1);
        if (trig != 2) // PORT 4
            pdl_trigger();
        return;
    }
    int const cap = min(nvv, kMaxCap);
    int const fend = nraw - local; // PORT 2 (his: nvv - local)
    int const flo = min(init, cap), fhi = max(0, cap - max(fend, flo));
    int const need = kTopk - flo - fhi;
    for (int p = tid; p < cap; p += NT)
        s.spid[p] = __ldcg(prw + p);
    if (need <= 0)
    {
        // more forced pages than slots: the lowest 16 forced pages (all keys tie at the forced key)
        int const v = tid < kTopk ? __ldcg(prw + (tid < flo ? tid : cap - fhi + (tid - flo))) : 0;
        pdl_wait(); // PORT 6: the store follows the wait (his stored before it)
        if (tid < kTopk)
            orw[tid] = v;
        if (trig != 2) // PORT 4
            pdl_trigger();
        return;
    }
    // forced pages are the extreme pages: init at positions [0, flo), local at [16 - fhi, 16)
    // PORT 6: the forced ids are loaded here and stored after the wait (his stored them here)
    int const fpg
        = tid < flo ? __ldcg(prw + tid) : (tid >= kTopk - fhi && tid < kTopk ? __ldcg(prw + cap - (kTopk - tid)) : 0);
    if (tid == 0)
        s.cnt = 0;
    if (tid < kPool)
        s.pk[tid] = 0ull; // V4 ranks over a fixed, zero-padded pool (0 never beats a real entry)
    if (tid < kTopk)
        s.wsel[tid] = 0x7fffffff;
    __syncthreads();
    pdl_wait();
    if (tid < flo || (tid >= kTopk - fhi && tid < kTopk)) // PORT 6
        orw[tid] = fpg;
    if (trig == 0)                                        // PORT 4: his release point
        pdl_trigger();
    if constexpr (ST)
        if (tid == 0)
            sr[1] = gtimer();
    // at most 4096 pages: the 256-thread instantiation (threads 256.. of a 512-thread launch leave here; exited threads
    // count as arrived at every later barrier); above 4096 pages all NT threads (16 keys each at NT 512)
    if (cap <= 16 * 256)
    {
        if (tid >= 256)
            return;
        if (cap <= 6 * 256)
            fsel_body<256, R, 6, ST, V>(s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
        else
            fsel_body<256, R, 16, ST, V>(s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
    }
    else
        fsel_body<NT, R, ((kMaxCap + NT - 1) / NT < 32 ? (kMaxCap + NT - 1) / NT : 32), ST, V>( // PORT 7
            s, srw, orw, cap, init, fend, flo, need, tid, diag, sr, sc_ps, trig);
}

// PORT(i8 MS4) 8: grid = rows x kMsK, launched with a cluster of kMsK CTAs per row (cluster rank = chunk); stamps
// (st) hold kMsK x rows rows of 8.
template <int NT, int R, bool ST, int V = 4>
__global__ void __launch_bounds__(NT, 1) fsel_ms_kernel(float const* __restrict__ sc, long long sc_st, long long sc_sh,
    long long sc_ps, int const* n_valid, int const* pt, long long pt_s0, int num_cols, int* out, long long out_sh,
    long long out_st, int H, int init, int local, int trig, int* diag, long long* st)
{
    __shared__ Smem s;
    __shared__ MsSmem ms;
    int const chunk = blockIdx.x % kMsK, row = blockIdx.x / kMsK;
    fsel_ms_row<NT, R, ST, V>(s, &ms, row, chunk, sc, sc_st, sc_sh, sc_ps, n_valid, pt, pt_s0, num_cols, out, out_sh,
        out_st, H, init, local, trig, diag, ST ? st + 8 * blockIdx.x : nullptr);
}

} // namespace fsel

// Plain launcher (no torch): variant 5 (NT 256), 6 (NT 512) or 7 (6 + the 4-CTA split above 4096 blocks). diag:
// [fast32, fast64, fallback, copy] or nullptr; st: [rows][8] %globaltimer stamps (entry, post-wait, pre-BAR1, pre-BAR2,
// pool ready, end) or nullptr.
static inline cudaError_t fsel_launch(float const* sc, long long sc_st, long long sc_sh, long long sc_ps, int const* nv,
    int const* pt, long long pt_s0, int num_cols, int* out, long long out_sh, long long out_st, int rows, int H,
    int init, int local, int variant, bool pdl, int trig, int* diag, long long* st, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    // PORT(i8 MS4) 8: variant 7 = variant 6 with rows above kMsMin blocks split over a 4-CTA cluster; when no row can
    // be that long (num_cols <= kMsMin) it is variant 6 exactly.
    bool const ms = variant == 7 && num_cols > fsel::kMsMin;
    if (variant == 7 && !ms)
        variant = 6;
    // variant 5: NT 256; variant 6: NT 512 (above 4096 pages 16 keys per thread instead of 32: the NT-256 KPT-32
    // instance costs ~8.4 us post-wait at 4145 pages vs 1.6 us at 4067 on sm_107a, same selection) PORT 7: the
    // 256-thread body covers at most 256 x 32 = 8192 pages; wider calls run the 512-thread instance
    int const nt = (variant == 6 || num_cols > 256 * 32) ? 512 : 256;
    cfg.gridDim = dim3(rows, 1, 1);
    cfg.blockDim = dim3(nt, 1, 1);
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    bool const ts = st != nullptr;
    if (variant != 5 && variant != 6 && !ms)
        return cudaErrorInvalidValue;
    if (trig < 0 || trig > 2 || num_cols > fsel::kMaxCap || rows <= 0) // PORT
        return cudaErrorInvalidValue;
    if (ms)                                                            // PORT(i8 MS4) 8
    {
        cfg.gridDim = dim3(rows * fsel::kMsK, 1, 1);
        cfg.blockDim = dim3(512, 1, 1);
        attr[1].id = cudaLaunchAttributeClusterDimension;
        attr[1].val.clusterDim.x = fsel::kMsK;
        attr[1].val.clusterDim.y = 1;
        attr[1].val.clusterDim.z = 1;
        cfg.numAttrs = 2;
        return ts ? cudaLaunchKernelEx(&cfg, fsel::fsel_ms_kernel<512, 4, true, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt,
                   pt_s0, num_cols, out, out_sh, out_st, H, init, local, trig, diag, st)
                  : cudaLaunchKernelEx(&cfg, fsel::fsel_ms_kernel<512, 4, false, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt,
                      pt_s0, num_cols, out, out_sh, out_st, H, init, local, trig, diag, st);
    }
    if (nt == 512)
        return ts ? cudaLaunchKernelEx(&cfg, fsel::fsel_kernel<512, 4, true, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt, pt_s0,
                   num_cols, out, out_sh, out_st, H, init, local, trig, diag, st)
                  : cudaLaunchKernelEx(&cfg, fsel::fsel_kernel<512, 4, false, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt,
                      pt_s0, num_cols, out, out_sh, out_st, H, init, local, trig, diag, st);
    return ts ? cudaLaunchKernelEx(&cfg, fsel::fsel_kernel<256, 4, true, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt, pt_s0,
               num_cols, out, out_sh, out_st, H, init, local, trig, diag, st)
              : cudaLaunchKernelEx(&cfg, fsel::fsel_kernel<256, 4, false, 4>, sc, sc_st, sc_sh, sc_ps, nv, pt, pt_s0,
                  num_cols, out, out_sh, out_st, H, init, local, trig, diag, st);
}

// PORT 5: his torch binding (fsel_topk) is replaced by csrc/m31/fselTopkOp.cpp.

cudaError_t invokeFselTopk(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t const* pageTable, int64_t pageTableRowStride, int32_t numBlocks, int32_t* out,
    int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads, int32_t totalQ, int32_t initBlocks,
    int32_t localBlocks, int32_t variant, bool usePdl, int32_t triggerPhase, int32_t* diag, int64_t* stamps,
    cudaStream_t stream)
{
    if (numKvHeads <= 0 || totalQ <= 0)
        return cudaErrorInvalidValue;
    return fsel_launch(scores, queryStride, headStride, blockStride, nValidBlocks, pageTable, pageTableRowStride,
        numBlocks, out, outHeadStride, outQueryStride, totalQ * numKvHeads, numKvHeads, initBlocks, localBlocks,
        variant, usePdl, triggerPhase, diag, reinterpret_cast<long long*>(stamps), stream);
}

} // namespace minimaxM3
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
