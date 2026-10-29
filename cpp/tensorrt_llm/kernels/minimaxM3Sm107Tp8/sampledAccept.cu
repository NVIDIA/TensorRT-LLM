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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/sampledAccept.h"

#include "tensorrt_llm/common/assert.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused sampled verification for the MiniMax-M3 Eagle3 target head (TP-sharded vocabulary, decode rows <= 48).
//
// sm_107 (Rubin) TP8-owned copy of kernels/minimaxM3Sm107Tp2/sampledAccept.cu (bit for bit the same kernel: the
// seam for the TP8 SA-1R round-0 decision), itself the TP2 copy of kernels/minimaxM3/sampledAccept.cu: identical
// algorithm and arithmetic; the vocab shard limit is raised to the TP2 shard (200064 / 2 = 100032 -> kMaxVs 100352), so
// a CTA stages up to kMaxVs / kSplits BF16 logits (25 KB at 8 splits, 50 KB at 4) and the launch opts in to that
// dynamic shared memory.
//
// Replaces, per decode step, the full-vocabulary all-gather of the target logits, the fused sampling kernel and the
// strict-acceptance ops for rows sampled with temperature and top-p (the AgentX contract). Rows with top-k or min-p
// active must stay on the generic path; the caller decides.
//
// Sampling is top-p rejection sampling, distributed over the TP ranks. With w = exp(logit / T - max) over the FULL
// vocabulary and Z = sum(w), a token is in the nucleus iff the mass strictly above its weight is at most top_p * Z
// (the fused sampling kernel's kept set, ties included). A candidate is drawn proportionally to w and accepted iff it
// is in the nucleus; a rejected candidate lies below the nucleus, so the next candidate is drawn over the tokens
// heavier than it. The accepted token is distributed as w restricted to the nucleus. The first candidate lands in the
// nucleus with probability >= top_p, so most rows decide on their second exchange.
//
// Per verify row, 8 CTAs (rows <= 32) or 4 CTAs (more rows) on every rank each own a slice of the rank's logits shard,
// staged in shared memory once. A candidate is drawn in two stages without per-token noise: each CTA draws a local
// candidate by inverse CDF over its slice, and the CTA whose log-mass plus Gumbel noise is largest supplies the
// candidate (CTA j wins with probability T_j / sum T, its token with w / T_j). Uniforms come from a counter-based
// Philox keyed by the request's seed and offset, the row, the attempt and the CTA. Each round every CTA multicasts one
// 4-word record (key, candidate logit, and max and mass in round 0 or the mass above the current candidate later) and
// reads the records of all CTAs of the row on all ranks. Every CTA takes the same decision from the same records in the
// same order, so all ranks agree without any intra-GPU handshake. Words carry the call's tag: a slot needs no clearing
// and two call halves keep a rank one call ahead from overwriting a slot still being read.
//
// SA-1R (Args.one_round_k 9 / 82; MoE prototype moe/sa1r/6c753632ec, exact on 39,968 rows, eager + graph): each CTA
// also multicasts its largest bf16 logits with the round-0 record (3 per extra word). After round 0 every CTA knows M,
// the candidate logit cv and wc = __expf(cv - M); round 1 would accept iff S <= target, S = the fp32 sum (its tree) of
// w = __expf(__fmul_rn(x, tInv) - M) over every token with w > wc. x -> w is monotone (__fmul_rn by tInv > 0, fsub,
// __expf: checked exhaustively over all non-positive fp32 inputs), so if no CTA's unlisted value can beat wc (82: the
// bound tau; 9: the 9th listed), the listed w > wc are exactly round 1's summands, bitwise. Two fp32 sums of the same
// m non-negative terms differ by < m * 2^-23 * S, so accepting iff S_1r + m * 2^-21 * S_1r (round up) <= target
// implies round 1 accepts; m <= 1 is exact. Otherwise round 1 runs as before. An accepted row outputs the round-0
// candidate, as the two-round kernel does, and consumes no Philox draw it would not.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{
namespace sampledAccept
{
namespace
{
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRec = kMaxWorld * kMaxSplits;
constexpr int kMaxAttempts = 16;      // exchange slots per row, reused cyclically by the rounds (no round cap)
constexpr int kTagBits = 64 - 32 - kIdBits;
constexpr float kTempEpsilon = 1e-6f; // the fused sampling kernel's 1/T guard
// Record polls trap after this many spins (the multicast all-reduce's bound): a missing writer becomes a CUDA error,
// never a hang.
constexpr long long kMaxSpins = 1LL << 31;

// SA-1R variant kTopK: 0 none; K exact top-K list; 10 * K + P pooled (each thread keeps its top P plus its (P + 1)-th
// as a bound; the CTA lists the pool's top K plus tau = max(K-th listed, max thread bound)).
__host__ __device__ constexpr int mode_k(int m)
{
    return m >= 10 ? m / 10 : m;
}

__host__ __device__ constexpr int mode_pool(int m)
{
    return m >= 10 ? m % 10 : 0;
}

__host__ __device__ constexpr int mode_vals(int m)
{
    return mode_k(m) + (mode_pool(m) ? 1 : 0);
}

// Words per record: key, logit of key, f2, f3, then the round-0 list (3 bf16 + a 16-bit tag per word).
template <int kTopK>
struct Rec
{
    static constexpr int kXw = (mode_vals(kTopK) + 2) / 3;
    static constexpr int kEx = 4 + kXw;
};

// Multicast exchange, per call half: [kMaxAttempts][kMaxRows][world * kMaxSplits][kEx] u64.
template <int kTopK>
__host__ __device__ inline long long half_words(int world)
{
    return static_cast<long long>(kMaxAttempts) * kMaxRows * world * kMaxSplits * Rec<kTopK>::kEx;
}

__device__ __forceinline__ uint32_t ordered(float v)
{
    uint32_t const b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// bf16 bits <-> an order-preserving 16-bit key.
__device__ __forceinline__ uint32_t ord16(uint32_t b)
{
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ uint32_t unord16(uint32_t k)
{
    return (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu);
}

// Philox4x32-10.
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k)
{
#pragma unroll
    for (int i = 0; i < 10; ++i)
    {
        uint32_t const hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        uint32_t const hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}

// u in (0, 1): (k + 1/2) / 2^23, exact in fp32.
__device__ __forceinline__ float uniform(uint32_t bits)
{
    return (static_cast<float>(bits >> 9) + 0.5f) * (1.0f / 8388608.0f);
}

// Key on the wire: perturbed log-mass, then the candidate token id; 0 = no candidate. Tag bits are added on the wire.
__device__ __forceinline__ unsigned long long gkey(float lg, int id)
{
    return (static_cast<unsigned long long>(ordered(lg)) << kIdBits)
        | static_cast<unsigned long long>((1u << kIdBits) - 1u - static_cast<uint32_t>(id));
}

__device__ __forceinline__ int key_token(unsigned long long k)
{
    return static_cast<int>((1u << kIdBits) - 1u - static_cast<uint32_t>(k & ((1u << kIdBits) - 1u)));
}

__device__ __forceinline__ void st_sys(unsigned long long* p, unsigned long long v)
{
    asm volatile("st.global.relaxed.sys.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long ld_sys(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.global.relaxed.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

template <int kSplits, int kTopK>
struct Shared
{
    __nv_bfloat16 row[kMaxVs / kSplits];
    float redf[kWarps];
    unsigned long long redk[kWarps];
    unsigned long long rk[kMaxRec];
    float rv[kMaxRec], r2[kMaxRec], r3[kMaxRec];
    float wsum[kWarps];
    int owner;
    int pick;
    float pickx;
    bool last;
    float dS[kWarps];                                     // SA-1R: per-warp partial mass above the candidate
    int dM[kWarps];                                       // SA-1R: per-warp count | incomplete << 16
    uint32_t wtop[kWarps][kTopK > 0 ? mode_k(kTopK) : 1]; // SA-1R: per-warp top-K keys
    uint32_t wbound[kWarps];                              // SA-1R pooled: per-warp max thread bound
    uint32_t ctop[kTopK > 0 ? mode_vals(kTopK) : 1];      // SA-1R: the CTA's top-K keys (descending), by rank
    unsigned long long rx[kMaxRec][Rec<kTopK>::kXw > 0 ? Rec<kTopK>::kXw : 1];
};

template <typename S>
__device__ __forceinline__ float block_max(float v, S& s)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0)
        s.redf[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = s.redf[0];
    for (int i = 1; i < kWarps; ++i)
        r = fmaxf(r, s.redf[i]);
    __syncthreads();
    return r;
}

// Fixed-order block sum: identical result for identical inputs.
template <typename S>
__device__ __forceinline__ float block_sum(float v, S& s)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0)
        s.redf[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < kWarps; ++i)
        r += s.redf[i];
    __syncthreads();
    return r;
}

template <typename S>
__device__ __forceinline__ unsigned long long block_max_u64(unsigned long long v, S& s)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        unsigned long long const w = __shfl_xor_sync(0xffffffffu, v, o);
        v = w > v ? w : v;
    }
    if ((threadIdx.x & 31) == 0)
        s.redk[threadIdx.x >> 5] = v;
    __syncthreads();
    unsigned long long r = s.redk[0];
    for (int i = 1; i < kWarps; ++i)
        r = s.redk[i] > r ? s.redk[i] : r;
    __syncthreads();
    return r;
}

// Exclusive prefix of v over the threads in thread order, and the total.
template <typename S>
__device__ __forceinline__ float block_excl_scan(float v, float& total, S& s)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        float const t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o)
            inc += t;
    }
    if (lane == 31)
        s.wsum[warp] = inc;
    __syncthreads();
    float base = 0.f;
    total = 0.f;
    for (int i = 0; i < kWarps; ++i)
    {
        base += i < warp ? s.wsum[i] : 0.f;
        total += s.wsum[i];
    }
    __syncthreads();
    return base + inc - v;
}

// Warp-wide integer max / sum / or (the redux instructions from sm_80; a butterfly otherwise, same results).
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __reduce_max_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        uint32_t const w = __shfl_xor_sync(0xffffffffu, v, o);
        v = w > v ? w : v;
    }
    return v;
#endif
}

__device__ __forceinline__ uint32_t warp_add_u32(uint32_t v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __reduce_add_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}

__device__ __forceinline__ uint32_t warp_or_u32(uint32_t v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __reduce_or_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v |= __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}

// Insert key into a descending register list of K (fully unrolled: no local memory).
template <int K>
__device__ __forceinline__ void topk_insert(uint32_t (&a)[K], uint32_t key)
{
#pragma unroll
    for (int j = 0; j < K; ++j)
    {
        uint32_t const hi = key > a[j] ? key : a[j];
        key = key > a[j] ? a[j] : key;
        a[j] = hi;
    }
}

template <int K>
__device__ __forceinline__ void topk_pop(uint32_t (&a)[K])
{
#pragma unroll
    for (int j = 0; j + 1 < K; ++j)
        a[j] = a[j + 1];
    a[K - 1] = 0u;
}

// Per-thread insertion with an early reject: after the first few elements almost every key fails the K-th test.
template <int K>
__device__ __forceinline__ void thread_insert(uint32_t (&a)[K], uint32_t key)
{
    if (key > a[K - 1])
        topk_insert<K>(a, key);
}

// The warp's K largest keys (descending) into s.wtop[warp]; published by the caller's next block barrier.
template <int K, typename S>
__device__ __forceinline__ void warp_topk(uint32_t (&a)[K], S& s)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < K; ++j)
    {
        uint32_t const m = warp_max_u32(a[0]);
        if (a[0] == m && m != 0u)
            topk_pop<K>(a);
        if (lane == 0)
            s.wtop[warp][j] = m;
    }
}

// Pooled: K rounds over each lane's P pool keys (descending) and the warp's max thread bound.
template <int K, int P, typename S>
__device__ __forceinline__ void warp_pool(uint32_t (&a)[P + 1], S& s)
{
    int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t const bound = warp_max_u32(a[P]);
    uint32_t pl[P];
#pragma unroll
    for (int j = 0; j < P; ++j)
        pl[j] = a[j];
#pragma unroll
    for (int j = 0; j < K; ++j)
    {
        uint32_t const m = warp_max_u32(pl[0]);
        if (pl[0] == m && m != 0u)
            topk_pop<P>(pl);
        if (lane == 0)
            s.wtop[warp][j] = m;
    }
    if (lane == 0)
        s.wbound[warp] = bound;
}

// A row's kSplits CTAs wait for each other's records, so they launch as one thread-block cluster: the hardware
// schedules a cluster's CTAs together, whatever other streams hold. Rows never wait on other rows.
template <int kSplits, int kTopK, bool kIlpWalk>
__global__ void __launch_bounds__(kThreads, 2) sampled_accept_kernel(Args const a)
{
    using R = Rec<kTopK>;
    extern __shared__ __align__(16) unsigned char smem[];
    Shared<kSplits, kTopK>& s = *reinterpret_cast<Shared<kSplits, kTopK>*>(smem);
    float constexpr kNegInf = -__builtin_huge_valf();
    int const tid = threadIdx.x;
    int const row = blockIdx.x / kSplits;
    int const split = blockIdx.x % kSplits;
    int const world = a.world;
    int const nrec = world * kSplits;
    // The row's sampling parameters are step inputs (the spec metadata, written before the target forward), not the
    // predecessor head GEMM's output, so with prewait_params they load before the dependency wait and overlap the
    // head's tail; the same values either way.
    float temperature = 0.f, topP = 0.f;
    long long seed_in = 0, offset_in = 0;
    auto load_params = [&]()
    {
        temperature = a.temperatures[row];
        topP = a.top_ps[row];
        seed_in = a.seeds[row];
        offset_in = a.offsets[row];
    };
    if (a.prewait_params)
        load_params();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (!a.prewait_params)
        load_params();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // early_trigger: the dependent grid (the first draft pass's input) may launch now and park at its own
    // griddepcontrol.wait, which still returns only once this grid has completed and flushed; it skips the launch
    // latency after this kernel ends. Otherwise dependents are released at the acceptance epilogue (the CTAs of
    // splits 1.. exit without releasing).
    if (a.early_trigger)
        asm volatile("griddepcontrol.launch_dependents;");
#endif
    float const tempInv = 1.0f / (temperature + kTempEpsilon);
    unsigned long long const seed = static_cast<unsigned long long>(seed_in);
    unsigned long long const offset = static_cast<unsigned long long>(offset_in);
    unsigned int const call = *reinterpret_cast<unsigned int const volatile*>(a.counters + 1);
    // Wire tags: never 0 (fresh memory), different between calls of the same half and between consecutive laps of
    // the slot ring (round r uses slot r % kMaxAttempts and lap parity (r / kMaxAttempts) & 1). A rank decides round
    // r only after reading every record of round r, so no rank is more than one round ahead and a slot is rewritten
    // only after every reader has consumed it; the lap bit tells the new record from the previous lap's.
    auto ktag_of = [&](int r) -> unsigned long long
    {
        unsigned int const lap = static_cast<unsigned int>(r / kMaxAttempts) & 1u;
        return (((call >> 1) << 1) | lap) % ((1u << kTagBits) - 1u) + 1u;
    };
    auto ftag_of = [&](int r) -> unsigned long long
    {
        unsigned int const lap = static_cast<unsigned int>(r / kMaxAttempts) & 1u;
        return static_cast<unsigned long long>(((((call >> 1) << 1) | lap) & 0x7FFFFFFFu) | 0x80000000u) << 32;
    };
    // SA-1R round-0 list words: a 16-bit tag, never 0.
    unsigned long long const xtag = (((call >> 1) << 1) & 0x7FFFu) | 0x8000u;
    long long const rec_stride = static_cast<long long>(world) * kMaxSplits * R::kEx;
    unsigned long long* const xu = a.ws_uc + static_cast<long long>(call & 1u) * half_words<kTopK>(world)
        + static_cast<long long>(row) * rec_stride;
    unsigned long long* const xm = a.ws_mc + static_cast<long long>(call & 1u) * half_words<kTopK>(world)
        + static_cast<long long>(row) * rec_stride + (a.rank * kMaxSplits + split) * R::kEx;
    long long const round_stride = static_cast<long long>(kMaxRows) * rec_stride;

    // ---- stage this CTA's slice and its max -------------------------------------------------------------------------
    __nv_bfloat16 const* src = a.logits + row * a.ld;
    int const sl_len = ((a.Vs + kSplits - 1) / kSplits + 7) / 8 * 8;
    int const s0 = min(split * sl_len, a.Vs);
    int const n = min(s0 + sl_len, a.Vs) - s0;
    int const id0 = a.vocab_offset + s0;
    float tmax = kNegInf;
    // SA-1R: each thread's largest (ord16(bf16) << 16 | ~index) keys while staging (only rows with a nucleus cut).
    bool const want_x = kTopK > 0 && topP < 1.0f;
    constexpr int kK = mode_k(kTopK), kPool = mode_pool(kTopK), kVals = mode_vals(kTopK);
    constexpr int kTk = kPool ? kPool + 1 : (kK > 0 ? kK : 1);
    uint32_t tk[kTk];
#pragma unroll
    for (int j = 0; j < kTk; ++j)
        tk[j] = 0u;
    bool const vec = (a.ld % 8) == 0 && (reinterpret_cast<uintptr_t>(a.logits) & 15) == 0;
    int const nvec = vec ? n / 8 : 0;
    for (int v = tid; v < nvec; v += kThreads)
    {
        uint4 const w = *reinterpret_cast<uint4 const*>(src + s0 + 8 * v);
        *reinterpret_cast<uint4*>(&s.row[8 * v]) = w;
        __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&w);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            tmax = fmaxf(tmax, __fmul_rn(__bfloat162float(e[j]), tempInv));
        if constexpr (kTopK > 0)
        {
            if (want_x)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    uint32_t const key
                        = (ord16(__bfloat16_as_ushort(e[j])) << 16) | (0xFFFFu - static_cast<uint32_t>(8 * v + j));
                    if constexpr (kPool > 0)
                        topk_insert<kTk>(tk, key); // branchless: P + 1 compare-swaps
                    else
                        thread_insert<kTk>(tk, key);
                }
            }
        }
    }
    for (int i = nvec * 8 + tid; i < n; i += kThreads)
    {
        s.row[i] = src[s0 + i];
        tmax = fmaxf(tmax, __fmul_rn(__bfloat162float(src[s0 + i]), tempInv));
        if constexpr (kTopK > 0)
        {
            if (want_x)
                topk_insert<kTk>(
                    tk, (ord16(__bfloat16_as_ushort(src[s0 + i])) << 16) | (0xFFFFu - static_cast<uint32_t>(i)));
        }
    }
    if constexpr (kTopK > 0)
    {
        if (tid < kVals)
            s.ctop[tid] = 0u; // published with s.wtop by block_max's barrier
        if (want_x)
        {
            if constexpr (kPool > 0)
                warp_pool<kK, kPool>(tk, s); // s.wtop / s.wbound published by block_max's barrier
            else
                warp_topk<kK>(tk, s);        // s.wtop published by block_max's barrier
        }
    }
    float const smax = block_max(tmax, s); // the staged slice is visible to all threads after this barrier
    if constexpr (kTopK > 0)
    {
        // SA-1R: the CTA's top K of the warp lists by rank (keys are unique): a lane quad per key, no serial merge;
        // s.ctop is read after the round-0 draw's first barrier
        constexpr int kN = kWarps * kK;
        constexpr int kSp = kN * 4 <= kThreads ? 4 : 1; // threads per key
        if (want_x && tid < kN * kSp)
        {
            uint32_t const* const wl = &s.wtop[0][0];
            int const part = tid % kSp;
            uint32_t const key = wl[tid / kSp];
            int rank = 0;
#pragma unroll
            for (int j = part; j < kN; j += kSp)
                rank += wl[j] > key ? 1 : 0;
            if constexpr (kSp == 4)
            {
                rank += __shfl_xor_sync(0xffffffffu, rank, 1);
                rank += __shfl_xor_sync(0xffffffffu, rank, 2);
            }
            if (part == 0 && key != 0u && rank < kK)
                s.ctop[rank] = key;
        }
    }
    float M = 0.f, wc = 0.f, target = 0.f, cv = 0.f; // global max, candidate weight and logit, top_p * Z
    unsigned long long ckey = 0ULL;
    int const gid = a.rank * kMaxSplits + split;
    for (int r = 0;; ++r)
    {
        // ---- local candidate by inverse CDF over the slice's tokens heavier than the candidate (w > 0 in round 0)
        float const ref = r == 0 ? smax : M;
        float const wmin = r == 0 ? 0.f : wc;
        float ts = 0.f;
        for (int i = tid; i < n; i += kThreads)
        {
            float const w = __expf(__fmul_rn(__bfloat162float(s.row[i]), tempInv) - ref);
            ts += w > wmin ? w : 0.f;
        }
        float total;
        float const excl = block_excl_scan(ts, total, s);
        if constexpr (kTopK > 0)
        {
            // SA-1R round 0: the list words depend only on the staged slice, so they go out now (s.ctop / s.wbound are
            // published by the barrier above), one lane per word, ahead of the main record: off the readers' path
            if (r == 0 && want_x && tid < R::kXw)
            {
                uint32_t tau = 0u;
                if constexpr (kPool > 0)
                {
                    tau = s.ctop[kK - 1]; // the largest unlisted pool key is <= the K-th listed
#pragma unroll
                    for (int wq = 0; wq < kWarps; ++wq)
                        tau = s.wbound[wq] > tau ? s.wbound[wq] : tau;
                }
                unsigned long long w = xtag;
#pragma unroll
                for (int t = 0; t < 3; ++t)
                {
                    int const e = 3 * tid + t;
                    uint32_t const kv = e < kK ? s.ctop[e] : (kPool > 0 && e == kK ? tau : 0u);
                    // missing entries: -inf (bf16 0xFF80), w = 0
                    uint32_t const b = kv != 0u ? unord16(kv >> 16) : 0xFF80u;
                    w |= static_cast<unsigned long long>(b) << (16 * (3 - t));
                }
                st_sys(xm + 4 + tid, w); // round 0 = slot 0
            }
        }
        uint4 const rnd = philox(make_uint4(static_cast<uint32_t>(gid), static_cast<uint32_t>(row + 64 * r),
                                     static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32)),
            make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)));
        float const tgt = uniform(rnd.x) * total;
        if (tid == 0)
            s.owner = -1;
        __syncthreads();
        if (ts > 0.f && excl <= tgt)
            atomicMax(&s.owner, tid);
        __syncthreads();
        if (tid == s.owner)
        {
            float acc = excl;
            int pick = -1;
            float px = 0.f;
            if constexpr (kIlpWalk)
            {
                // SA-PW: the owner's strided tokens 8 at a time. The 8 x and w (the same expressions) are
                // independent, so their MUFU work overlaps; the serial sum then adds them in the same order with
                // the same early exit: the same acc sequence, pick and px as the one-token walk, bit for bit.
                constexpr int kW = 8;
                bool done = false;
                for (int i0 = tid; i0 < n && !done; i0 += kW * kThreads)
                {
                    float xs[kW], wv[kW];
#pragma unroll
                    for (int k = 0; k < kW; ++k)
                    {
                        int const i = i0 + k * kThreads;
                        xs[k] = i < n ? __fmul_rn(__bfloat162float(s.row[i]), tempInv) : 0.f;
                        wv[k] = __expf(xs[k] - ref);
                    }
#pragma unroll
                    for (int k = 0; k < kW; ++k)
                    {
                        int const i = i0 + k * kThreads;
                        if (!done && i < n && wv[k] > wmin)
                        {
                            acc += wv[k];
                            pick = i;
                            px = xs[k];
                            if (acc > tgt)
                                done = true;
                        }
                    }
                }
            }
            else
            {
                for (int i = tid; i < n; i += kThreads)
                {
                    float const x = __fmul_rn(__bfloat162float(s.row[i]), tempInv);
                    float const w = __expf(x - ref);
                    if (w > wmin)
                    {
                        acc += w;
                        pick = i;
                        px = x;
                        if (acc > tgt)
                            break;
                    }
                }
            }
            s.pick = pick;
            s.pickx = px;
        }
        __syncthreads();
        unsigned long long sb = 0ULL;
        if (total > 0.f && s.owner >= 0)
            sb = gkey(__logf(total) + ref - logf(-logf(uniform(rnd.y))), id0 + s.pick);
        float const smab = total;
        unsigned long long const ktag = ktag_of(r);
        unsigned long long const ftag = ftag_of(r);
        unsigned long long* const xr = xu + (r % kMaxAttempts) * round_stride;
        bool const with_x = want_x && r == 0;
        if (tid == 0)
        {
            unsigned long long* const xw = xm + (r % kMaxAttempts) * round_stride;
            st_sys(xw + 0, (sb << kTagBits) | ktag);
            st_sys(xw + 1, ftag | __float_as_uint(sb != 0ULL ? s.pickx : 0.f));
            st_sys(xw + 2, ftag | __float_as_uint(r == 0 ? smax : smab));
            st_sys(xw + 3, ftag | __float_as_uint(r == 0 ? smab : 0.f));
        }
        // ---- every record of the row, every rank ---------------------------------------------------------------
        int const nw = with_x ? R::kEx : 4;
        for (int t = tid; t < nrec * nw; t += kThreads)
        {
            int const q = t / nw, f = t % nw;
            unsigned long long const* const p = xr + (q / kSplits * kMaxSplits + q % kSplits) * R::kEx + f;
            unsigned long long v;
            long long spins = 0;
            if (f == 0)
            {
                do
                {
                    v = ld_sys(p);
                    if (++spins > kMaxSpins)
                        __trap();
                } while ((v & ((1ULL << kTagBits) - 1ULL)) != ktag);
                s.rk[q] = v >> kTagBits;
            }
            else if (f < 4)
            {
                do
                {
                    v = ld_sys(p);
                    if (++spins > kMaxSpins)
                        __trap();
                } while ((v & 0xFFFFFFFF00000000ULL) != ftag);
                float const fv = __uint_as_float(static_cast<uint32_t>(v));
                (f == 1 ? s.rv : f == 2 ? s.r2 : s.r3)[q] = fv;
            }
            else
            {
                do
                {
                    v = ld_sys(p);
                    if (++spins > kMaxSpins)
                        __trap();
                } while ((v & 0xFFFFULL) != xtag);
                s.rx[q][f - 4] = v;
            }
        }
        __syncthreads();
        // ---- the decision, identical in every CTA of the row on every rank: fixed record order -----------------
        unsigned long long gk = 0ULL;
        float gv = 0.f;
        for (int q = 0; q < nrec; ++q)
        {
            if (s.rk[q] > gk)
            {
                gk = s.rk[q];
                gv = s.rv[q];
            }
        }
        bool accept;
        if (r == 0)
        {
            M = kNegInf;
            for (int q = 0; q < nrec; ++q)
                M = fmaxf(M, s.r2[q]);
            float Z = 0.f;
            for (int q = 0; q < nrec; ++q)
                Z += s.r2[q] == kNegInf ? 0.f : s.r3[q] * __expf(s.r2[q] - M);
            target = topP * Z;
            accept = !(topP < 1.0f); // no nucleus cut: the first draw stands
            ckey = gk;
            cv = gv;
            if constexpr (kTopK > 0)
            {
                // SA-1R: every thread takes (record, entry) items; the per-warp partials are published by the round's
                // barrier below and every thread combines them in warp order after it (no new barrier)
                if (with_x)
                {
                    float const wcand = __expf(cv - M); // = round 1's wmin
                    float part = 0.f;
                    int cnt = 0, inc = 0;
                    for (int t = tid; t < nrec * kVals; t += kThreads)
                    {
                        int const q = t / kVals, e = t % kVals;
                        uint32_t const b = static_cast<uint32_t>(s.rx[q][e / 3] >> (16 * (3 - e % 3))) & 0xFFFFu;
                        float const w = __expf(
                            __fmul_rn(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(b))), tempInv)
                            - M); // w bitwise as round 1 computes it
                        if (w > wcand)
                        {
                            if (kPool > 0 && e == kK)
                                inc = 1; // tau (the largest unlisted value) is heavier than the candidate
                            else
                            {
                                part += w;
                                ++cnt;
                                if (kPool == 0 && e == kK - 1)
                                    inc = 1; // the K-th is heavier: unlisted heavier tokens may exist
                            }
                        }
                    }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        part += __shfl_xor_sync(0xffffffffu, part, o); // any order: the margin covers it
                    cnt = static_cast<int>(warp_add_u32(static_cast<uint32_t>(cnt)));
                    inc = static_cast<int>(warp_or_u32(static_cast<uint32_t>(inc)));
                    if ((tid & 31) == 0)
                    {
                        s.dS[tid >> 5] = part;
                        s.dM[tid >> 5] = cnt | (inc << 16);
                    }
                }
            }
        }
        else
        {
            float S = 0.f; // mass strictly above the current candidate
            for (int q = 0; q < nrec; ++q)
                S += s.r2[q];
            accept = S <= target || gk == 0ULL;
            if (!accept)
            {
                ckey = gk;
                cv = gv;
            }
        }
        wc = __expf(cv - M);
        __syncthreads(); // records consumed before the next round's writes to s
        bool decided = false;
        if constexpr (kTopK > 0)
        {
            if (r == 0 && with_x)
            {
                float S1 = 0.f;
                int m = 0, inc = 0;
                for (int wq = 0; wq < kWarps; ++wq)
                {
                    S1 += s.dS[wq];
                    m += s.dM[wq] & 0xFFFF;
                    inc |= s.dM[wq] >> 16;
                }
                if (!inc)
                {
                    float const margin = m <= 1 ? 0.f : S1 * static_cast<float>(m) * (1.0f / 2097152.0f);
                    decided = __fadd_ru(S1, margin) <= target;
                    accept = accept || decided;
                }
            }
        }
        if (accept)
        {
            if (split != 0)
                return;
            if (tid == 0)
            {
                a.tokens[row] = key_token(ckey);
                if (a.extra_attempts != nullptr && r > 1)
                    atomicAdd(a.extra_attempts, r - 1);
                if (a.decided != nullptr && decided)
                    atomicAdd(a.decided, 1);
            }
            break;
        }
    }

    // ---- split 0: the last row to finish runs the strict acceptance ----------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (!a.early_trigger)
        asm volatile("griddepcontrol.launch_dependents;");
#endif
    __threadfence();
    __syncthreads();
    if (tid == 0)
        s.last = atomicAdd(a.counters, 1u) == static_cast<unsigned>(a.rows) - 1;
    __syncthreads();
    if (!s.last)
        return;
    __threadfence();
    if (tid == 0)
    {
        a.counters[0] = 0u;
        a.counters[1] = call + 1u;
    }
    if (tid < a.nreq)
    {
        int const* tok = a.tokens;
        int const b = tid;
        int const k1 = a.draft_len + 1;
        int nacc = 1;
        bool run = true;
        for (int k = 0; k < a.draft_len; ++k)
        {
            int const tt = *reinterpret_cast<int const volatile*>(tok + b * k1 + k);
            a.acc_tokens[b * k1 + k] = tt;
            run = run && a.draft[b * a.draft_len + k] == tt;
            nacc += run ? 1 : 0;
        }
        a.acc_tokens[b * k1 + a.draft_len] = *reinterpret_cast<int const volatile*>(tok + b * k1 + a.draft_len);
        if (a.force_vec != nullptr)
            nacc = a.force_vec[b];
        else if (a.force_total > 0)
            nacc = a.force_total;
        a.num_acc[b] = nacc;
        a.gather_ids[b] = static_cast<long long>(a.batch_idx[b]) * k1 + nacc - 1;
        if (a.next_new != nullptr)
            a.next_new[b * a.next_stride]
                = *reinterpret_cast<int const volatile*>(tok + a.batch_idx[b] * k1 + nacc - 1);
    }
}

template <int kSplits, int kTopK, bool kIlpWalk>
cudaError_t launch(Args const& a, bool use_pdl, cudaStream_t stream)
{
    static int max_clusters = 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(a.rows * kSplits);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = sizeof(Shared<kSplits, kTopK>);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSplits;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    if (max_clusters == 0)
    {
        cudaError_t err = cudaFuncSetAttribute(sampled_accept_kernel<kSplits, kTopK, kIlpWalk>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sizeof(Shared<kSplits, kTopK>)));
        if (err == cudaSuccess)
        {
            err = cudaOccupancyMaxActiveClusters(&max_clusters, sampled_accept_kernel<kSplits, kTopK, kIlpWalk>, &cfg);
        }
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    TLLM_CHECK_WITH_INFO(max_clusters >= 1, "m3_sampled_accept: a %d-CTA cluster does not fit on one GPC", kSplits);
    return cudaLaunchKernelEx(&cfg, sampled_accept_kernel<kSplits, kTopK, kIlpWalk>, a);
}

template <int kTopK>
cudaError_t launchRows(Args const& a, bool use_pdl, cudaStream_t stream)
{
    // TP4 caps 2 / 4 / 8 (8 / 16 / 32 rows) and TP8 cap 1 (4 rows): 8 CTAs per row, at most 256 CTAs (2 per SM);
    // more rows: 4 CTAs per row.
    if (a.ilp_walk)
    {
        return a.rows <= 32 ? launch<8, kTopK, true>(a, use_pdl, stream) : launch<4, kTopK, true>(a, use_pdl, stream);
    }
    return a.rows <= 32 ? launch<8, kTopK, false>(a, use_pdl, stream) : launch<4, kTopK, false>(a, use_pdl, stream);
}

} // namespace

long long sampledAcceptWsWords(int world, int oneRoundK)
{
    TLLM_CHECK_WITH_INFO(
        oneRoundK == 0 || oneRoundK == 9 || oneRoundK == 82, "SA-1R variant %d is not built", oneRoundK);
    return 2 * (oneRoundK == 9 ? half_words<9>(world) : oneRoundK == 82 ? half_words<82>(world) : half_words<0>(world));
}

cudaError_t invokeSampledAccept(Args const& a, bool usePdl, cudaStream_t stream)
{
    switch (a.one_round_k)
    {
    case 0: return launchRows<0>(a, usePdl, stream);
    case 9: return launchRows<9>(a, usePdl, stream);
    case 82: return launchRows<82>(a, usePdl, stream);
    default: TLLM_CHECK_WITH_INFO(false, "SA-1R variant %d is not built", a.one_round_k);
    }
    return cudaErrorInvalidValue;
}

} // namespace sampledAccept
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
