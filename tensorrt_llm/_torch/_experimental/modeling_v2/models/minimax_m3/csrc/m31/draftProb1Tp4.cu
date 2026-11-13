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

// The draft prob1 kernel headers (namespaces dkmkp / dkmkp1 / dkr inside the family namespace) are included only by
// this translation unit.
#include "minimax_m3/csrc/m31/draftProb1Tp4/dspark_markov_prob1.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "minimax_m3/csrc/m31/draftProb1Tp4.h"

#include "minimax_m3/csrc/m31/prob1Agreement.h"

#include <math.h>

static_assert(
    TRTLLM_MV2_NS::dkmkp::kMaxC == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::draftProb1::kMaxRows, "peer kMaxC");
static_assert(
    TRTLLM_MV2_NS::dkmkp::kMaxW == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::draftProb1::kMaxWorld, "peer kMaxW");
static_assert(
    TRTLLM_MV2_NS::dkmkp1::kMaxK == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::draftProb1::kMaxK, "peer kMaxK");
static_assert(TRTLLM_MV2_NS::dkmkp::kR == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::draftProb1::kRank, "peer kR");
static_assert(
    TRTLLM_MV2_NS::dkmkp::kMaxGrid == TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp4::draftProb1::kMaxGrid, "peer kMaxGrid");

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace minimaxM3Sm107Tp4
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
    int const g = sms * dkmkp1::ctas_per_sm(rows);
    return g < kMaxGrid ? g : kMaxGrid;
}

cudaError_t launchProb1(Prob1Params const& p, int grid, bool pdl, cudaStream_t stream)
{
    // The peer steps() mapping, plus full-generation readiness and the immutable local reader-safe launch gate
    // (dspark_markov_prob1_ext.cu)
    dkmkp1::Args a{};
    a.base = static_cast<__nv_bfloat16 const*>(p.base);
    a.ld = p.ld;
    a.w1 = static_cast<__nv_bfloat16 const*>(p.w1);
    a.w1_rows = p.w1Rows;
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
    a.launchGate = p.launchGate;
    a.rank = p.rank;
    a.world = p.world;
    return dkmkp1::launch(a, grid, pdl, stream);
}

namespace
{

struct FaultPeers
{
    unsigned long long* p[kMaxWorld];
};

__global__ void prepKernel(int rows, int k, int64_t const* slotIds, int64_t dummySlotRow, int64_t slotCount, int* err,
    int64_t const* offsets, int64_t rngBase, int64_t span, int32_t* idxMap, int64_t* samplePos, FaultPeers peers,
    int world, unsigned long long const* seq, int64_t capNs, unsigned long long* gate)
{
    if (threadIdx.x == 0)
    {
        bool invalid = false;
        for (int r = 0; r < rows; ++r)
            invalid = invalid || slotIds[r] < 0 || slotIds[r] >= slotCount;
        unsigned long long const ready = prob1Agreement::prepare(world, peers.p, seq, capNs);
        if (invalid)
            atomicExch(err, 3);
        *gate = invalid ? 0 : ready;
    }
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x)
    {
        int const r = i / k;
        int const s = i % k;
        bool const pad = dummySlotRow > 0 && slotIds[r] == dummySlotRow;
        idxMap[i] = pad ? -1 : r;
        samplePos[i] = offsets[r] + rngBase + static_cast<int64_t>(s) * span + 1;
    }
}

// One warp. Lane s < world: publish this rank's word into rank s's fault word [rank], then wait for rank s's word in
// this rank's fault word [s].
__global__ void agreeKernel(int rank, int world, int* err, unsigned int* counter, unsigned long long* epoch,
    unsigned long long* mine, unsigned long long* faultMine, FaultPeers peers, unsigned long long* seqp, int64_t capNs,
    int* fault, int64_t* faultCount)
{
    (void) counter;
    (void) epoch;
    (void) mine;
    (void) faultMine;
    prob1Agreement::agree(rank, world, err, peers.p, seqp, capNs, fault, faultCount);
}

// The draft row this block serves: (r, s) = (blockIdx.x / k, blockIdx.x % k); chunk blockIdx.y of kStatSplit.
__device__ __forceinline__ bool rowLive(ProbsParams const& p, int r, float& t)
{
    t = p.temperature[r];
    bool const pad = p.dummySlotRow > 0 && p.slotIds[r] == p.dummySlotRow;
    return p.slotIds[r] >= 0 && p.slotIds[r] < p.slotCount && !pad && t != 0.f && *p.fault == 0;
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
    dkmkp::MS m{-INFINITY, 0.f};
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x)
    {
        m = dkmkp::ms_merge(m, dkmkp::MS{dkr::div_full(__bfloat162float(lg[v]), t), 1.f});
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        dkmkp::MS const q{__shfl_xor_sync(0xffffffffu, m.m, o), __shfl_xor_sync(0xffffffffu, m.s, o)};
        m = dkmkp::ms_merge(m, q);
    }
    __shared__ dkmkp::MS sm[8];
    if ((threadIdx.x & 31) == 0)
    {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        dkmkp::MS a{-INFINITY, 0.f};
        for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w)
        {
            a = dkmkp::ms_merge(a, sm[w]);
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
    int64_t const slot = p.slotIds[r];
    bool const validSlot = slot >= 0 && slot < p.slotCount;
    bool const flt = *p.fault != 0 || !validSlot;
    bool const pad = p.dummySlotRow > 0 && p.slotIds[r] == p.dummySlotRow;
    float const t = p.temperature[r];
    int64_t const tok = flt ? 0 : p.tok[static_cast<int64_t>(r) * p.tokLd + s];
    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
        p.tokens32[row] = static_cast<int32_t>(tok);
    }
    __shared__ float s_m, s_inv;
    if (!validSlot)
        return;
    bool const live = !flt && !pad && t != 0.f;
    if (live)
    {
        if (threadIdx.x == 0)
        {
            dkmkp::MS a{-INFINITY, 0.f};
            float const* st = p.stats + static_cast<int64_t>(row) * kStatSplit * 2;
            for (int c = 0; c < kStatSplit; ++c)
            {
                a = dkmkp::ms_merge(a, dkmkp::MS{st[2 * c], st[2 * c + 1]});
            }
            s_m = a.m;
            s_inv = 1.f / a.s;
        }
        __syncthreads();
    }
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

cudaError_t launchPrep(int rows, int k, int64_t const* slotIds, int64_t dummySlotRow, int64_t slotCount, int* err,
    int64_t const* offsets, int64_t rngBase, int64_t span, int32_t* idxMap, int64_t* samplePos,
    unsigned long long* const* faultPeers, int world, unsigned long long const* seq, int64_t capNs,
    unsigned long long* gate, cudaStream_t stream)
{
    FaultPeers fp{};
    for (int r = 0; r < world; ++r)
        fp.p[r] = faultPeers[r];
    prepKernel<<<1, 128, 0, stream>>>(rows, k, slotIds, dummySlotRow, slotCount, err, offsets, rngBase, span, idxMap,
        samplePos, fp, world, seq, capNs, gate);
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
    cudaError_t const writeError = cudaGetLastError();
    if (writeError != cudaSuccess)
        return writeError;
    prob1Agreement::finishReaders<4><<<1, 1, 0, stream>>>(p.readerAck, p.seq);
    return cudaGetLastError();
}

} // namespace draftProb1

} // namespace minimaxM3Sm107Tp4

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
