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

#include "tensorrt_llm/kernels/minimaxM3/sampledAccept.h"

#include "tensorrt_llm/common/assert.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused sampled verification for the MiniMax-M3 Eagle3 target head (TP-sharded vocabulary, decode rows <= 48).
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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace sampledAccept
{
namespace
{
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRec = kMaxWorld * kMaxSplits;
constexpr int kMaxAttempts = 16;      // exchange slots per row, reused cyclically by the rounds (no round cap)
constexpr int kEx = 4;                // words per record: key, logit of key, f2, f3
constexpr int kTagBits = 64 - 32 - kIdBits;
constexpr float kTempEpsilon = 1e-6f; // the fused sampling kernel's 1/T guard

// Multicast exchange, per call half: [kMaxAttempts][kMaxRows][world * kMaxSplits][kEx] u64.
__host__ __device__ inline long long half_words(int world)
{
    return static_cast<long long>(kMaxAttempts) * kMaxRows * world * kMaxSplits * kEx;
}

__device__ __forceinline__ uint32_t ordered(float v)
{
    uint32_t const b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
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

template <int kSplits>
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

// Every CTA of the grid must be co-resident: a row's CTAs wait for each other's records.
template <int kSplits>
__global__ void __launch_bounds__(kThreads, 2) sampled_accept_kernel(Args const a)
{
    extern __shared__ __align__(16) unsigned char smem[];
    Shared<kSplits>& s = *reinterpret_cast<Shared<kSplits>*>(smem);
    float constexpr kNegInf = -__builtin_huge_valf();
    int const tid = threadIdx.x;
    int const row = blockIdx.x / kSplits;
    int const split = blockIdx.x % kSplits;
    int const world = a.world;
    int const nrec = world * kSplits;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
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
    long long const rec_stride = static_cast<long long>(world) * kMaxSplits * kEx;
    unsigned long long* const xu
        = a.ws_uc + static_cast<long long>(call & 1u) * half_words(world) + static_cast<long long>(row) * rec_stride;
    unsigned long long* const xm = a.ws_mc + static_cast<long long>(call & 1u) * half_words(world)
        + static_cast<long long>(row) * rec_stride + (a.rank * kMaxSplits + split) * kEx;
    long long const round_stride = static_cast<long long>(kMaxRows) * rec_stride;

    float const tempInv = 1.0f / (a.temperatures[row] + kTempEpsilon);
    float const topP = a.top_ps[row];
    unsigned long long const seed = static_cast<unsigned long long>(a.seeds[row]);
    unsigned long long const offset = static_cast<unsigned long long>(a.offsets[row]);

    // ---- stage this CTA's slice and its max -------------------------------------------------------------------------
    __nv_bfloat16 const* src = a.logits + row * a.ld;
    int const sl_len = ((a.Vs + kSplits - 1) / kSplits + 7) / 8 * 8;
    int const s0 = min(split * sl_len, a.Vs);
    int const n = min(s0 + sl_len, a.Vs) - s0;
    int const id0 = a.vocab_offset + s0;
    float tmax = kNegInf;
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
    }
    for (int i = nvec * 8 + tid; i < n; i += kThreads)
    {
        s.row[i] = src[s0 + i];
        tmax = fmaxf(tmax, __fmul_rn(__bfloat162float(src[s0 + i]), tempInv));
    }
    float const smax = block_max(tmax, s);           // the staged slice is visible to all threads after this barrier
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
        if (tid == 0)
        {
            unsigned long long* const xw = xm + (r % kMaxAttempts) * round_stride;
            st_sys(xw + 0, (sb << kTagBits) | ktag);
            st_sys(xw + 1, ftag | __float_as_uint(sb != 0ULL ? s.pickx : 0.f));
            st_sys(xw + 2, ftag | __float_as_uint(r == 0 ? smax : smab));
            st_sys(xw + 3, ftag | __float_as_uint(r == 0 ? smab : 0.f));
        }
        // ---- every record of the row, every rank ---------------------------------------------------------------
        if (tid < nrec * kEx)
        {
            int const q = tid / kEx, f = tid % kEx;
            unsigned long long v;
            if (f == 0)
            {
                do
                    v = ld_sys(xr + (q / kSplits * kMaxSplits + q % kSplits) * kEx);
                while ((v & ((1ULL << kTagBits) - 1ULL)) != ktag);
                s.rk[q] = v >> kTagBits;
            }
            else
            {
                do
                    v = ld_sys(xr + (q / kSplits * kMaxSplits + q % kSplits) * kEx + f);
                while ((v & 0xFFFFFFFF00000000ULL) != ftag);
                float const fv = __uint_as_float(static_cast<uint32_t>(v));
                (f == 1 ? s.rv : f == 2 ? s.r2 : s.r3)[q] = fv;
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
        if (accept)
        {
            if (split != 0)
                return;
            if (tid == 0)
            {
                a.tokens[row] = key_token(ckey);
                if (a.extra_attempts != nullptr && r > 1)
                    atomicAdd(a.extra_attempts, r - 1);
            }
            break;
        }
    }

    // ---- split 0: the last row to finish runs the strict acceptance ----------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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

template <int kSplits>
cudaError_t launch(Args const& a, bool use_pdl, cudaStream_t stream)
{
    static int max_ctas = 0;
    if (max_ctas == 0)
    {
        int dev = 0, sms = 0, per_sm = 0;
        cudaError_t err = cudaGetDevice(&dev);
        if (err == cudaSuccess)
        {
            err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        }
        if (err == cudaSuccess)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, sampled_accept_kernel<kSplits>, kThreads, sizeof(Shared<kSplits>));
        }
        if (err != cudaSuccess)
        {
            return err;
        }
        max_ctas = per_sm * sms;
    }
    TLLM_CHECK_WITH_INFO(a.rows * kSplits <= max_ctas, "m3_sampled_accept: %d CTAs cannot all be co-resident (max %d)",
        a.rows * kSplits, max_ctas);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(a.rows * kSplits);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = sizeof(Shared<kSplits>);
    cfg.stream = stream;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, sampled_accept_kernel<kSplits>, a);
}

} // namespace

long long sampledAcceptWsWords(int world)
{
    return 2 * half_words(world);
}

cudaError_t invokeSampledAccept(Args const& a, bool usePdl, cudaStream_t stream)
{
    // TP4 caps 2 / 4 / 8 (8 / 16 / 32 rows) and TP8 cap 1 (4 rows): 8 CTAs per row, at most 256 CTAs (2 per SM);
    // more rows: 4 CTAs per row.
    if (a.rows <= 32)
    {
        return launch<8>(a, usePdl, stream);
    }
    return launch<4>(a, usePdl, stream);
}

} // namespace sampledAccept
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
