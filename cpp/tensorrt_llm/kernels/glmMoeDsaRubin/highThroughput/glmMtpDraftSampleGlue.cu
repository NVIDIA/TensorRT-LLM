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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/highThroughput/glmMtpDraftSampleGlue.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"

#include <climits>
#include <cmath>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_rubin_mtp
{
namespace
{

using tensorrt_llm::common::loadPackedVolatile;
using tensorrt_llm::common::VolatilePackedLoad;

constexpr int kBlockSize = 512;
constexpr int kNumWarps = kBlockSize / 32;
// FlashInfer RMSNormKernel geometry for H = 6144: thread t of a 64-thread group owns the 8-element vectors at columns
// 8 t + 512 b, b = 0..11.
constexpr int kNormThreads = 64;
constexpr int kNormVecElts = 8;
constexpr int kNormVecs = kDraftSampleGlueHidden / (kNormThreads * kNormVecElts);
constexpr int kNumNormGroups = kBlockSize / kNormThreads;
static_assert(kNormVecs * kNormThreads * kNormVecElts == kDraftSampleGlueHidden, "hidden size must be 512-aligned");
// An unwritten word of the Lamport workspace (the MNNVL all-gather's empty word).
constexpr uint32_t kEmptyWord = 0xFFFFFFFFU;

struct Candidate
{
    float value;
    int index;
};

// Total order of at::native GreaterOrNan (the order torch.max and torch.argmax use): a NaN beats every number and the
// lowest-index NaN wins; otherwise the larger value, and the lower index among equal values.
__device__ __forceinline__ bool isBetter(float v1, int i1, float v2, int i2)
{
    bool const nan1 = v1 != v1;
    bool const nan2 = v2 != v2;
    if (nan1 || nan2)
    {
        return nan1 && (!nan2 || i1 < i2);
    }
    return v1 > v2 || (v1 == v2 && i1 < i2);
}

__device__ __forceinline__ void offer(Candidate& best, float value, int index)
{
    if (isBetter(value, index, best.value, best.index))
    {
        best.value = value;
        best.index = index;
    }
}

__device__ __forceinline__ Candidate warpArgmax(Candidate c)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float const value = __shfl_xor_sync(0xFFFFFFFFU, c.value, offset);
        int const index = __shfl_xor_sync(0xFFFFFFFFU, c.index, offset);
        offer(c, value, index);
    }
    return c;
}

// BF16 element 2k (low half) and 2k + 1 (high half) of a 32-bit word, widened exactly.
__device__ __forceinline__ float bf16Low(uint32_t word)
{
    return __uint_as_float(word << 16);
}

__device__ __forceinline__ float bf16High(uint32_t word)
{
    return __uint_as_float(word & 0xFFFF0000U);
}

__device__ __forceinline__ bool isEmptyPacket(VolatilePackedLoad<float4> const& packet)
{
    return packet.words[0] == kEmptyWord || packet.words[1] == kEmptyWord || packet.words[2] == kEmptyWord
        || packet.words[3] == kEmptyWord;
}

__device__ __forceinline__ void namedBarrier(int id, int numThreads)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(numThreads) : "memory");
}

__device__ __forceinline__ float rsqrtApprox(float x)
{
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t packBf16(float lo, float hi)
{
    __nv_bfloat162 const v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&v);
}

// (x * rstd) * (w + 0) of 8 elements, rounded to BF16.
__device__ __forceinline__ uint4 normalizeVector(uint4 const& x, uint4 const& w, float rstd)
{
    auto const scaled = [rstd](float xv, float wv) { return __fmul_rn(__fmul_rn(xv, rstd), __fadd_rn(wv, 0.F)); };
    uint4 y;
    y.x = packBf16(scaled(bf16Low(x.x), bf16Low(w.x)), scaled(bf16High(x.x), bf16High(w.x)));
    y.y = packBf16(scaled(bf16Low(x.y), bf16Low(w.y)), scaled(bf16High(x.y), bf16High(w.y)));
    y.z = packBf16(scaled(bf16Low(x.z), bf16Low(w.z)), scaled(bf16High(x.z), bf16High(w.z)));
    y.w = packBf16(scaled(bf16Low(x.w), bf16Low(w.w)), scaled(bf16High(x.w), bf16High(w.w)));
    return y;
}

__device__ __forceinline__ float sumOfSquares(uint4 const& v, float acc)
{
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        float const lo = bf16Low(words[k]);
        float const hi = bf16High(words[k]);
        acc = __fadd_rn(acc, __fmul_rn(lo, lo));
        acc = __fadd_rn(acc, __fmul_rn(hi, hi));
    }
    return acc;
}

// RMSNorm of one row (src, or a zero row when null) by one 64-thread group, with the arithmetic of FlashInfer's
// RMSNormKernel: each thread sums the squares of its 96 elements in column order, the warp adds the partial sums in a
// butterfly (offsets 1, 2, ..., 16), the row sum is the first warp's total plus the second's; then
// rsqrt.approx(sum / H + eps) and (x * rstd) * (w + 0) in FP32, rounded to BF16. Columns [lo, hi) of the row (half
// coordinates) are stored at out[col - lo].
__device__ void normRowWindow(uint4 const* src, __nv_bfloat16 const* weights, float eps, int lo, int hi,
    __nv_bfloat16* out, int group, float (*groupSums)[2])
{
    int const t = static_cast<int>(threadIdx.x) % kNormThreads;
    int const warpInGroup = t / 32;
    int const lane = t % 32;
    uint4 x[kNormVecs];
#pragma unroll
    for (int b = 0; b < kNormVecs; ++b)
    {
        x[b] = src != nullptr ? src[t + kNormThreads * b] : make_uint4(0U, 0U, 0U, 0U);
    }
    float acc = 0.F;
#pragma unroll
    for (int b = 0; b < kNormVecs; ++b)
    {
        acc = sumOfSquares(x[b], acc);
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFU, acc, offset));
    }
    if (lane == 0)
    {
        groupSums[group][warpInGroup] = acc;
    }
    namedBarrier(1 + group, kNormThreads);
    float const sum = __fadd_rn(groupSums[group][0], groupSums[group][1]);
    float const rstd = rsqrtApprox(__fadd_rn(__fdiv_rn(sum, static_cast<float>(kDraftSampleGlueHidden)), eps));
#pragma unroll
    for (int b = 0; b < kNormVecs; ++b)
    {
        int const col = kNormVecElts * t + kNormThreads * kNormVecElts * b;
        if (col >= lo && col < hi)
        {
            uint4 const w = *reinterpret_cast<uint4 const*>(weights + col);
            *reinterpret_cast<uint4*>(out + (col - lo)) = normalizeVector(x[b], w, rstd);
        }
    }
    // The group's sums are rewritten by its next row.
    namedBarrier(1 + group, kNormThreads);
}

// Inclusive warp prefix sum.

// Position ids and sequence/KV lengths of the next step.
// (lane r handles request r). One warp.
__device__ void advanceStep(DraftSampleGlueParams const& p)
{
    int const lane = static_cast<int>(threadIdx.x) % 32;
    bool const active = lane < p.numRows;
    int32_t seq = 0;
    int32_t kv = 0;
    if (active)
    {
        if (p.positionsOut != nullptr)
        {
            int64_t const src = p.gatherIds != nullptr ? p.gatherIds[lane] : lane;
            if (p.positions64)
            {
                static_cast<int64_t*>(p.positionsOut)[lane] = static_cast<int64_t const*>(p.positionsIn)[src] + 1;
            }
            else
            {
                static_cast<int32_t*>(p.positionsOut)[lane] = static_cast<int32_t const*>(p.positionsIn)[src] + 1;
            }
        }
        if (p.seqLens != nullptr)
        {
            seq = p.fillSeqLens ? 1 : p.seqLens[lane];
            if (p.fillSeqLens)
            {
                p.seqLens[lane] = 1;
            }
        }
        if (p.kvLens != nullptr)
        {
            kv = p.kvLens[lane];
            if (p.kvUpdate == DraftSampleGlueKvUpdate::kIncrement)
            {
                kv += 1;
                p.kvLens[lane] = kv;
            }
            else if (p.kvUpdate == DraftSampleGlueKvUpdate::kRewind)
            {
                kv -= p.draftLen - p.kvRewindAccepted[lane];
                p.kvLens[lane] = kv;
            }
        }
    }
}

// The draft step under attention DP with a tensor-parallel LM head (draftSampleGlueAdpKernel): one CTA of kBlockSize
// threads per stacked row of the logits. After the grid dependency wait every CTA scans its row; warp 0 publishes the
// row's pair (slot [row][rank] of the current buffer, through the multicast address) and polls the row's pairs of
// every rank, which every rank publishes; in this rank's rows it pulls each candidate's embedding row into L2 as its
// pair lands and then picks the token. Beside the exchange, warp 1 empties the CTA's share of the previous call's
// buffer and, in the CTA of this rank's first row, writes the next step's state; norm group kAdpHnormGroup normalizes
// the row's hidden half of the eh_proj input. Norm group 0 then normalizes the token's embedding row, and warp
// kAdpRowsWarp writes the next-new-tokens row. A thread of CTA 0 rotates the flags once every CTA has read them. Same
// outputs for this rank's rows.
constexpr int kAdpHnormGroup = 1;
// Logit vectors each thread loads per round: one round covers a 40,960-column shard (GLM-5.2's is 38,720).
constexpr int kAdpScanBatch = 10;
// 128-byte lines of an embedding row.
constexpr int kEmbedRowLines = kDraftSampleGlueHidden * static_cast<int>(sizeof(__nv_bfloat16)) / 128;
constexpr int kAdpRowsWarp = 2 * kAdpHnormGroup;
constexpr int kAdpFlagsThread = kBlockSize - 32;
// Index of the arrival counter among the nine words of the Lamport flags.
constexpr int kFlagArrivals = 8;
static_assert(kDraftSampleGlueAdpMaxRanks <= 32 && kDraftSampleGlueAdpMaxSegmentRows <= 32,
    "one lane per rank polls a row's pairs; one lane per row writes the next step's state");
static_assert(kAdpFlagsThread / 32 > kAdpRowsWarp, "the flags thread works beside the other roles");

__device__ __forceinline__ void arriveRelease(uint32_t* counter)
{
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(1U) : "memory");
}

__device__ __forceinline__ uint32_t loadAcquire(uint32_t const* ptr)
{
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

// The largest of the 8 BF16 of v, widened exactly; NaN when one of them is NaN.
__device__ __forceinline__ float vectorMaxOrNan(uint4 const& v)
{
    __nv_bfloat162 const* const h = reinterpret_cast<__nv_bfloat162 const*>(&v);
    __nv_bfloat162 const m = __hmax2_nan(__hmax2_nan(h[0], h[1]), __hmax2_nan(h[2], h[3]));
    uint32_t const word = *reinterpret_cast<uint32_t const*>(&m);
    float const lo = bf16Low(word);
    float const hi = bf16High(word);
    return lo != lo ? lo : (hi != hi ? hi : fmaxf(lo, hi));
}

// This thread's best element of a logits row (vocabShard / 8 vectors of 8 BF16), the element scanRow returns: the
// first vector (in column order) whose maximum is the thread's largest (any NaN beats every number), then the first
// element of that vector equal to its maximum (the first NaN). kAdpScanBatch vectors are loaded per round.
__device__ Candidate scanRowByVectorMax(uint4 const* row, int numVecs)
{
    float bestMax = -INFINITY;
    int bestVec = -1;
    uint4 bestData = make_uint4(0U, 0U, 0U, 0U);
    for (int base = static_cast<int>(threadIdx.x); base < numVecs; base += kAdpScanBatch * kBlockSize)
    {
        uint4 vecs[kAdpScanBatch];
#pragma unroll
        for (int k = 0; k < kAdpScanBatch; ++k)
        {
            int const v = base + k * kBlockSize;
            if (v < numVecs)
            {
                vecs[k] = row[v];
            }
        }
#pragma unroll
        for (int k = 0; k < kAdpScanBatch; ++k)
        {
            int const v = base + k * kBlockSize;
            if (v < numVecs)
            {
                float const m = vectorMaxOrNan(vecs[k]);
                bool const bestIsNan = bestMax != bestMax;
                bool const better = bestVec < 0 || (m != m ? !bestIsNan : (!bestIsNan && m > bestMax));
                if (better)
                {
                    bestMax = m;
                    bestVec = v;
                    bestData = vecs[k];
                }
            }
        }
    }
    Candidate best{-INFINITY, INT_MAX};
    if (bestVec >= 0)
    {
        bool const nan = bestMax != bestMax;
        uint32_t const words[4] = {bestData.x, bestData.y, bestData.z, bestData.w};
#pragma unroll
        for (int e = 7; e >= 0; --e)
        {
            float const x = e % 2 == 0 ? bf16Low(words[e / 2]) : bf16High(words[e / 2]);
            if (nan ? x != x : x == bestMax)
            {
                best = Candidate{x, bestVec * kNormVecElts + e};
            }
        }
    }
    return best;
}

__global__ void __launch_bounds__(kBlockSize) draftSampleGlueAdpKernel(DraftSampleGlueAdpParams const ap)
{
    __shared__ __align__(16) __nv_bfloat16 sWeights[2][kDraftSampleGlueHidden];
    __shared__ Candidate sWarpBest[kNumWarps];
    __shared__ int sToken;
    __shared__ float sGroupSums[kNumNormGroups][2];
    __shared__ uint4 sFlags[2];

    DraftSampleGlueParams const& p = ap.step;
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid / 32;
    int const lane = tid % 32;
    int const row = static_cast<int>(blockIdx.x);
    int const segment = row / ap.segmentRows;
    int const local = row - segment * ap.segmentRows;
    bool const own = segment == p.rank && local < p.numRows;
    bool const glue = own && p.glueOut != nullptr;
    // The halves of the concatenated row that the eh_proj input window overlaps.
    bool const embedHalf = glue && p.colLo < kDraftSampleGlueHidden;
    bool const hiddenHalf = glue && p.colHi > kDraftSampleGlueHidden;

    // The norm weights are constants: copy the window columns before the grid dependency wait.
    if (glue)
    {
        for (int h = 0; h < 2; ++h)
        {
            if ((h == 0 && !embedHalf) || (h == 1 && !hiddenHalf))
            {
                continue;
            }
            int const lo = max(p.colLo - h * kDraftSampleGlueHidden, 0);
            int const hi = min(p.colHi - h * kDraftSampleGlueHidden, kDraftSampleGlueHidden);
            auto const* src = reinterpret_cast<uint4 const*>(h == 0 ? p.enormWeight : p.hnormWeight);
            auto* dst = reinterpret_cast<uint4*>(sWeights[h]);
            for (int v = lo / kNormVecElts + tid; v < hi / kNormVecElts; v += kBlockSize)
            {
                dst[v] = src[v];
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The logits come from the preceding kernel (the draft LM head). Dependents read the outputs only after their
    // own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    if (tid == 0)
    {
        sFlags[0] = reinterpret_cast<uint4 const*>(p.exchangeFlags)[0];
        sFlags[1] = reinterpret_cast<uint4 const*>(p.exchangeFlags)[1];
        // This CTA has read the flags (the release orders the reads first): CTA 0 rotates them once every CTA has.
        arriveRelease(p.exchangeFlags + kFlagArrivals);
    }

    // Local argmax of the row: all threads scan, each warp reduces, warp 0 combines below.
    Candidate const scanned = warpArgmax(scanRowByVectorMax(
        reinterpret_cast<uint4 const*>(p.logits + row * p.logitsStride), p.vocabShard / kNormVecElts));
    if (lane == 0)
    {
        sWarpBest[warp] = scanned;
    }
    __syncthreads();

    uint4 const flags = sFlags[0];
    uint32_t const curIndex = flags.x;
    uint32_t const dirtyIndex = flags.y;
    uint32_t const bytesPerBuffer = flags.z;
    uint32_t const dirtyStages = flags.w;
    int const tp = p.tpSize;
    auto* exchangeLocal = static_cast<char*>(p.exchangeLocal);

    if (warp == 0)
    {
        Candidate best = lane < kNumWarps ? sWarpBest[lane] : Candidate{-INFINITY, INT_MAX};
        best = warpArgmax(best);
        if (lane == 0)
        {
            int const globalIndex = best.index + p.rank * p.vocabShard;
            auto* curMcast = reinterpret_cast<uint4*>(static_cast<char*>(p.exchangeMcast) + curIndex * bytesPerBuffer);
            curMcast[row * tp + p.rank]
                = make_uint4(static_cast<uint32_t>(globalIndex), __float_as_uint(best.value), 0U, 0U);
        }
        // Lane r polls rank r's pair of this row until it has landed.
        Candidate pair{-INFINITY, INT_MAX};
        if (lane < tp)
        {
            auto const* slot
                = exchangeLocal + curIndex * bytesPerBuffer + (row * tp + lane) * kDraftSampleGluePacketBytes;
            VolatilePackedLoad<float4> packet;
            do
            {
                packet = loadPackedVolatile<float4>(slot);
            } while (isEmptyPacket(packet));
            pair = Candidate{__uint_as_float(packet.words[1]), static_cast<int>(packet.words[0])};
            // The winner's embedding row is one of the candidates': start loading this one into L2.
            if (embedHalf && pair.index >= 0 && pair.index < p.vocabSize)
            {
                char const* line = reinterpret_cast<char const*>(
                    p.embedTable + static_cast<int64_t>(pair.index) * kDraftSampleGlueHidden);
                for (int l = 0; l < kEmbedRowLines; ++l)
                {
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(line + l * 128) : "memory");
                }
            }
        }
        // The global argmax: the order is total over distinct indices, so any combination order picks the same pair.
        Candidate const token = warpArgmax(pair);
        if (own && lane == 0)
        {
            sToken = token.index;
            p.draftTokens[local * p.numDraftTokens + p.draftStep] = token.index;
        }
    }
    else if (warp == 1)
    {
        // Empty this CTA's share of the previous call's buffer; the peers write it again two calls later. Every
        // packet of that buffer was read by the previous call.
        if (dirtyStages > 0)
        {
            auto* dirty = reinterpret_cast<uint4*>(exchangeLocal + dirtyIndex * bytesPerBuffer);
            int const dirtyPackets = static_cast<int>(sFlags[1].x / kDraftSampleGluePacketBytes);
            for (int i = row * 32 + lane; i < dirtyPackets; i += static_cast<int>(gridDim.x) * 32)
            {
                dirty[i] = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
            }
        }
        if (row == p.rank * ap.segmentRows)
        {
            advanceStep(p);
        }
    }
    else if (hiddenHalf && warp / 2 == kAdpHnormGroup)
    {
        // hnorm of the hidden row does not need the token: it runs during the poll.
        int const lo = max(p.colLo - kDraftSampleGlueHidden, 0);
        int const hi = min(p.colHi - kDraftSampleGlueHidden, kDraftSampleGlueHidden);
        int const outShift = kDraftSampleGlueHidden + lo - p.colLo;
        normRowWindow(reinterpret_cast<uint4 const*>(p.hidden + local * p.hiddenStride), sWeights[1], p.eps, lo, hi,
            p.glueOut + local * p.glueOutStride + outShift, kAdpHnormGroup, sGroupSums);
    }
    __syncthreads();

    if (embedHalf && warp < 2)
    {
        // enorm of the token embedding: an id outside [0, vocabSize) gives a zero row, as the Embedding module does.
        int64_t const token = sToken;
        bool const valid = token >= 0 && token < p.vocabSize;
        normRowWindow(valid ? reinterpret_cast<uint4 const*>(p.embedTable + token * kDraftSampleGlueHidden) : nullptr,
            sWeights[0], p.eps, p.colLo, min(p.colHi, kDraftSampleGlueHidden), p.glueOut + local * p.glueOutStride, 0,
            sGroupSums);
    }
    else if (own && warp == kAdpRowsWarp && lane == 0 && p.nextNewTokens != nullptr)
    {
        // The row's output after the last draft step: the last accepted token, then the draft tokens of every step.
        int32_t* out = p.nextNewTokens + local * (p.numDraftTokens + 1);
        out[0] = p.acceptedTokens[p.batchIndices[local] * p.acceptedTokensStride + p.numAccepted[local] - 1];
        for (int j = 0; j < p.numDraftTokens; ++j)
        {
            out[1 + j] = j == p.draftStep ? sToken : p.draftTokens[local * p.numDraftTokens + j];
        }
    }
    else if (row == 0 && tid == kAdpFlagsThread)
    {
        // Rotate the Lamport flags as LamportFlags::update does once every CTA has read them: the current buffer
        // becomes the dirty one.
        while (loadAcquire(p.exchangeFlags + kFlagArrivals) < gridDim.x)
        {
        }
        auto* flagPtr = reinterpret_cast<uint4*>(p.exchangeFlags);
        flagPtr[0] = make_uint4((curIndex + 1) % 3, curIndex, bytesPerBuffer, 1U);
        flagPtr[1] = make_uint4(gridDim.x * static_cast<uint32_t>(tp * kDraftSampleGluePacketBytes), 0U, 0U, 0U);
        p.exchangeFlags[kFlagArrivals] = 0U;
    }
}

} // namespace

void invokeDraftSampleGlueAdp(DraftSampleGlueAdpParams const& params, cudaStream_t stream)
{
    DraftSampleGlueParams const& s = params.step;
    TLLM_CHECK_WITH_INFO(params.segmentRows >= 1 && params.segmentRows <= kDraftSampleGlueAdpMaxSegmentRows,
        "draft sample glue (ADP) serves 1 to %d rows per rank, got %d", kDraftSampleGlueAdpMaxSegmentRows,
        params.segmentRows);
    TLLM_CHECK_WITH_INFO(s.numRows >= 1 && s.numRows <= params.segmentRows,
        "draft sample glue (ADP): %d rows of this rank exceed the %d rows per rank", s.numRows, params.segmentRows);
    TLLM_CHECK_WITH_INFO(s.tpSize >= 2 && s.tpSize <= kDraftSampleGlueAdpMaxRanks && s.rank >= 0 && s.rank < s.tpSize,
        "draft sample glue (ADP) serves 2 to %d ranks, got rank %d of %d", kDraftSampleGlueAdpMaxRanks, s.rank,
        s.tpSize);
    TLLM_CHECK_WITH_INFO(s.vocabShard > 0 && s.vocabShard % 8 == 0 && s.logitsStride % 8 == 0,
        "draft sample glue (ADP) needs 16-byte aligned rows of a multiple of 8 logits, got %d", s.vocabShard);
    TLLM_CHECK_WITH_INFO(s.glueOut == nullptr
            || (s.colLo >= 0 && s.colLo < s.colHi && s.colHi <= 2 * kDraftSampleGlueHidden && s.colLo % 8 == 0
                && s.colHi % 8 == 0),
        "draft sample glue (ADP): invalid eh_proj input window [%d, %d)", s.colLo, s.colHi);
    TLLM_CHECK_WITH_INFO(s.glueOut == nullptr || s.colLo >= kDraftSampleGlueHidden || s.embedTable != nullptr,
        "draft sample glue (ADP): the window needs the embedding table");
    TLLM_CHECK_WITH_INFO(s.glueOut == nullptr || s.colHi <= kDraftSampleGlueHidden || s.hidden != nullptr,
        "draft sample glue (ADP): the window needs the hidden rows");
    int const stackedRows = s.tpSize * params.segmentRows;
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "draftSampleGlueAdp", draftSampleGlueAdpKernel, dim3(stackedRows), dim3(kBlockSize), 0, stream, params);
}

} // namespace kernels::glm_rubin_mtp

TRTLLM_NAMESPACE_END
