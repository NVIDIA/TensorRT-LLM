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

// The draft prob1 kernel headers (namespaces dkmkpw / dkmkp1w / dkr inside the family namespace) are included only by
// this translation unit.
#include "minimax_m3/csrc/m31/draftProb1Tp8/dspark_markov_prob1_w.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "minimax_m3/csrc/m31/draftProb1Tp8.h"

#include <math.h>

namespace D8 = TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp8::draftProb1;
static_assert(TRTLLM_MV2_NS::dkmkpw::Lim<D8::kWorld>::kMaxC == D8::kMaxRows, "peer kMaxC");
static_assert(TRTLLM_MV2_NS::dkmkpw::Lim<D8::kWorld>::kMaxW == D8::kMaxWorld, "peer kMaxW");
static_assert(TRTLLM_MV2_NS::dkmkp1w::kMaxK == D8::kMaxK, "peer kMaxK");
static_assert(TRTLLM_MV2_NS::dkmkpw::kR == D8::kRank, "peer kR");
static_assert(TRTLLM_MV2_NS::dkmkpw::kMaxGrid == D8::kMaxGrid, "peer kMaxGrid");

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace minimaxM3Sm107Tp8
{

namespace draftProb1
{

int maxGrid(int rows)
{
    if (rows < 1 || rows > kMaxRows)
    {
        return 0;
    }
    int dev = 0;
    int sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess
        || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    {
        return 0;
    }
    int const g = sms * dkmkp1w::ctas_per_sm<kWorld>(rows);
    return g < kMaxGrid ? g : kMaxGrid;
}

cudaError_t launchProb1(Prob1Params const& p, int grid, bool pdl, cudaStream_t stream)
{
    // the peer extension's steps(), field for field (dspark_markov_prob1_w8_ext.cu)
    dkmkp1w::Args<kWorld> a{};
    a.base = static_cast<__nv_bfloat16 const*>(p.base);
    a.ld = p.ld;
    a.w1 = static_cast<__nv_bfloat16 const*>(p.w1);
    a.w2s = static_cast<__nv_bfloat16 const*>(p.w2s);
    a.Vs = p.vs;
    a.voff = p.voff;
    a.C = p.rows;
    a.k = p.k;
    a.prev_in = p.prevIn;
    a.prev32 = nullptr;
    a.prev32_stride = 0;
    a.prev_buf[0] = p.prev0;
    a.prev_buf[1] = p.prev1;
    a.tok_out = p.tokOut;
    a.tok_ld = p.tokLd;
    a.idx_map = p.idxMap;
    a.idx_map64 = nullptr;
    a.sample_pos = p.samplePos;
    a.temperature = p.temperature;
    a.seeds = p.seeds;
    for (int c = 0; c < kMaxWorld; ++c)
    {
        a.caches[c] = static_cast<__nv_bfloat16*>(p.caches[c]);
    }
    a.ncache = p.ncache;
    a.cs0 = p.cs0;
    a.cs1 = p.cs1;
    a.dstat = nullptr;
    a.part = p.part;
    a.part_ms = p.partMs;
    a.counter = p.counter;
    a.epoch = p.epoch;
    a.mine = p.mine;
    for (int s = 0; s < kMaxWorld; ++s)
    {
        a.peers[s] = p.peers[s];
    }
    a.err = p.err;
    a.rank = p.rank;
    a.world = p.world;
    return dkmkp1w::launch<kWorld>(a, grid, pdl, stream);
}

namespace
{

__device__ __forceinline__ void stReleaseSys(unsigned long long* p, unsigned long long v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long ldAcquireSys(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned long long globalTimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void prepKernel(int rows, int k, int64_t const* slotIds, int64_t dummySlotRow, int64_t const* offsets,
    int64_t rngBase, int64_t span, int32_t* idxMap, int64_t* samplePos)
{
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x)
    {
        int const r = i / k;
        int const s = i % k;
        bool const pad = dummySlotRow > 0 && slotIds[r] == dummySlotRow;
        idxMap[i] = pad ? -1 : r;
        samplePos[i] = offsets[r] + rngBase + static_cast<int64_t>(s) * span + 1;
    }
}

struct FaultPeers
{
    unsigned long long* p[kMaxWorld];
};

// One warp. Lane s < world: publish this rank's word into rank s's fault word [rank], then wait for rank s's word in
// this rank's fault word [s].
__global__ void agreeKernel(int rank, int world, int* err, unsigned int* counter, unsigned long long* epoch,
    unsigned long long* mine, unsigned long long* faultMine, FaultPeers peers, unsigned long long* seqp, int64_t capNs,
    int* fault, int64_t* faultCount)
{
    int const lane = threadIdx.x;
    unsigned long long const seq = *seqp + 1ull; // every rank runs this kernel once per draft: same seq
    int const myErr = *reinterpret_cast<int volatile*>(err);
    unsigned long long const word = (seq << 8) | (myErr != 0 ? 1ull : 0ull);
    bool bad = false;
    if (lane < world)
    {
        stReleaseSys(peers.p[lane] + rank, word);
        unsigned long long const t0 = globalTimer();
        unsigned long long w = 0;
        while (((w = ldAcquireSys(faultMine + lane)) >> 8) != seq)
        {
            if (static_cast<int64_t>(globalTimer() - t0) > capNs)
            {
                break;
            }
        }
        bad = ((w >> 8) != seq) || ((w & 0xFFull) != 0ull);
    }
    bool const anyBad = __any_sync(0xffffffffu, bad);
    if (lane == 0)
    {
        *seqp = seq;
        *fault = anyBad ? 1 : 0;
        if (anyBad)
        {
            // every rank takes this branch for the same seq: the peer kernel's state back to its launch-0 values
            *err = 0;
            *counter = 0u;
            *epoch = 0ull;
            *faultCount += 1;
        }
    }
    if (anyBad)
    {
        for (int i = lane; i < kSlotWords; i += 32)
        {
            mine[i] = 0ull;
        }
    }
    __threadfence_system();
}

// The draft row this block serves: (r, s) = (blockIdx.x / k, blockIdx.x % k); chunk blockIdx.y of kStatSplit.
__device__ __forceinline__ bool rowLive(ProbsParams const& p, int r, float& t)
{
    t = p.temperature[r];
    bool const pad = p.dummySlotRow > 0 && p.slotIds[r] == p.dummySlotRow;
    return !pad && t != 0.f && *p.fault == 0;
}

__global__ void __launch_bounds__(256) statsKernel(ProbsParams p)
{
    int const row = blockIdx.x;
    int const r = row / p.k;
    int const s = row % p.k;
    float t;
    if (!rowLive(p, r, t))
    {
        return;
    }
    __nv_bfloat16 const* lg = static_cast<__nv_bfloat16 const*>(p.cache) + static_cast<int64_t>(r) * p.cs0
        + static_cast<int64_t>(s) * p.cs1;
    int const chunk = (p.vocab + kStatSplit - 1) / kStatSplit;
    int const v0 = blockIdx.y * chunk;
    int const v1 = min(p.vocab, v0 + chunk);
    dkmkpw::MS m{-INFINITY, 0.f};
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x)
    {
        m = dkmkpw::ms_merge(m, dkmkpw::MS{dkr::div_full(__bfloat162float(lg[v]), t), 1.f});
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        dkmkpw::MS const q{__shfl_xor_sync(0xffffffffu, m.m, o), __shfl_xor_sync(0xffffffffu, m.s, o)};
        m = dkmkpw::ms_merge(m, q);
    }
    __shared__ dkmkpw::MS sm[8];
    if ((threadIdx.x & 31) == 0)
    {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        dkmkpw::MS a{-INFINITY, 0.f};
        for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w)
        {
            a = dkmkpw::ms_merge(a, sm[w]);
        }
        float* st = p.stats + (static_cast<int64_t>(row) * kStatSplit + blockIdx.y) * 2;
        st[0] = a.m;
        st[1] = a.s;
    }
}

__global__ void __launch_bounds__(256) writeKernel(ProbsParams p)
{
    int const row = blockIdx.x;
    int const r = row / p.k;
    int const s = row % p.k;
    bool const flt = *p.fault != 0;
    bool const pad = p.dummySlotRow > 0 && p.slotIds[r] == p.dummySlotRow;
    float const t = p.temperature[r];
    int64_t const tok = flt ? 0 : p.tok[static_cast<int64_t>(r) * p.tokLd + s];
    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
        p.tokens32[row] = static_cast<int32_t>(tok);
    }
    __shared__ float s_m, s_inv;
    bool const live = !flt && !pad && t != 0.f;
    if (live)
    {
        if (threadIdx.x == 0)
        {
            dkmkpw::MS a{-INFINITY, 0.f};
            float const* st = p.stats + static_cast<int64_t>(row) * kStatSplit * 2;
            for (int c = 0; c < kStatSplit; ++c)
            {
                a = dkmkpw::ms_merge(a, dkmkpw::MS{st[2 * c], st[2 * c + 1]});
            }
            s_m = a.m;
            s_inv = 1.f / a.s;
        }
        __syncthreads();
    }
    int64_t const slot = p.slotIds[r];
    float* out = p.draftProbs + slot * p.ds0 + static_cast<int64_t>(s) * p.ds1;
    __nv_bfloat16 const* lg = static_cast<__nv_bfloat16 const*>(p.cache) + static_cast<int64_t>(r) * p.cs0
        + static_cast<int64_t>(s) * p.cs1;
    int const chunk = (p.vocab + kStatSplit - 1) / kStatSplit;
    int const v0 = blockIdx.y * chunk;
    int const v1 = min(p.vocab, v0 + chunk);
    float const uni = 1.f / static_cast<float>(p.vocab);
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x)
    {
        float q;
        if (live)
        {
            q = dkr::tl_exp(dkr::div_full(__bfloat162float(lg[v]), t) - s_m) * s_inv;
        }
        else if (pad && !flt)
        {
            q = uni;
        }
        else
        {
            q = (v == tok) ? 1.f : 0.f; // T == 0 (the draw is the argmax) or an agreed fault (token 0)
        }
        out[v] = q;
    }
}

} // namespace

cudaError_t launchPrep(int rows, int k, int64_t const* slotIds, int64_t dummySlotRow, int64_t const* offsets,
    int64_t rngBase, int64_t span, int32_t* idxMap, int64_t* samplePos, cudaStream_t stream)
{
    prepKernel<<<1, 128, 0, stream>>>(rows, k, slotIds, dummySlotRow, offsets, rngBase, span, idxMap, samplePos);
    return cudaGetLastError();
}

cudaError_t launchAgree(int rank, int world, int* err, unsigned int* counter, unsigned long long* epoch,
    unsigned long long* mine, unsigned long long* faultMine, unsigned long long* const* faultPeers,
    unsigned long long* seq, int64_t capNs, int* fault, int64_t* faultCount, cudaStream_t stream)
{
    FaultPeers fp{};
    for (int s = 0; s < kMaxWorld; ++s)
    {
        fp.p[s] = s < world ? faultPeers[s] : nullptr;
    }
    agreeKernel<<<1, 32, 0, stream>>>(
        rank, world, err, counter, epoch, mine, faultMine, fp, seq, capNs, fault, faultCount);
    return cudaGetLastError();
}

cudaError_t launchProbs(ProbsParams const& p, cudaStream_t stream)
{
    dim3 const grid(p.rows * p.k, kStatSplit);
    statsKernel<<<grid, 256, 0, stream>>>(p);
    cudaError_t const e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        return e;
    }
    writeKernel<<<grid, 256, 0, stream>>>(p);
    return cudaGetLastError();
}

} // namespace draftProb1

} // namespace minimaxM3Sm107Tp8

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
