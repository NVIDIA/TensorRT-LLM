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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmDraftSampleGlue.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"

#include <cooperative_groups.h>

#include <climits>
#include <cmath>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

using tensorrt_llm::common::loadPackedVolatile;
using tensorrt_llm::common::VolatilePackedLoad;

constexpr int kBlockSize = 512;
constexpr int kNumWarps = kBlockSize / 32;
// Logit vectors (8 BF16) each thread loads before it compares them.
constexpr int kScanBatch = 8;
// FlashInfer RMSNormKernel geometry for H = 6144: thread t of a 64-thread group owns the 8-element vectors at columns
// 8 t + 512 b, b = 0..11.
constexpr int kNormThreads = 64;
constexpr int kNormVecElts = 8;
constexpr int kNormVecs = kDraftSampleGlueHidden / (kNormThreads * kNormVecElts);
constexpr int kNumNormGroups = kBlockSize / kNormThreads;
static_assert(kNormVecs * kNormThreads * kNormVecElts == kDraftSampleGlueHidden, "hidden size must be 512-aligned");
// An unwritten word of the Lamport workspace (the MNNVL all-gather's empty word).
constexpr uint32_t kEmptyWord = 0xFFFFFFFFU;
constexpr int kMaxSlots = kDraftSampleGlueMaxRows * 32;

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

__device__ __forceinline__ void offerVector(Candidate& best, uint4 const& v, int col)
{
    offer(best, bf16Low(v.x), col);
    offer(best, bf16High(v.x), col + 1);
    offer(best, bf16Low(v.y), col + 2);
    offer(best, bf16High(v.y), col + 3);
    offer(best, bf16Low(v.z), col + 4);
    offer(best, bf16High(v.z), col + 5);
    offer(best, bf16Low(v.w), col + 6);
    offer(best, bf16High(v.w), col + 7);
}

// This thread's best element of a logits row (vocabShard / 8 vectors of 8 BF16).
__device__ Candidate scanRow(uint4 const* row, int numVecs)
{
    Candidate best{-INFINITY, INT_MAX};
    for (int base = static_cast<int>(threadIdx.x); base < numVecs; base += kScanBatch * kBlockSize)
    {
        uint4 vecs[kScanBatch];
#pragma unroll
        for (int k = 0; k < kScanBatch; ++k)
        {
            int const v = base + k * kBlockSize;
            if (v < numVecs)
            {
                vecs[k] = row[v];
            }
        }
#pragma unroll
        for (int k = 0; k < kScanBatch; ++k)
        {
            int const v = base + k * kBlockSize;
            if (v < numVecs)
            {
                offerVector(best, vecs[k], v * kNormVecElts);
            }
        }
    }
    return best;
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
__device__ __forceinline__ int64_t warpInclusiveSum(int64_t value)
{
    int const lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        int64_t const other = __shfl_up_sync(0xFFFFFFFFU, value, offset);
        if (lane >= offset)
        {
            value += other;
        }
    }
    return value;
}

// Position ids, sequence and KV lengths of the next step, then the DSA decode metadata of fused_dsa_decode_metadata
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
    if (p.reqIdxPerToken == nullptr)
    {
        return;
    }
    int64_t const seq64 = active ? seq : 0;
    int64_t const kv64 = active ? kv : 0;
    int64_t const seqIncl = warpInclusiveSum(seq64);
    int64_t const kvIncl = warpInclusiveSum(kv64);
    int64_t const cachedIncl = warpInclusiveSum(kv64 - seq64);
    if (lane == 0)
    {
        p.genKvIndptr[0] = 0;
        p.genCachedTokenIndptr[0] = 0;
    }
    if (!active)
    {
        return;
    }
    p.genKvIndptr[1 + lane] = kvIncl;
    p.genCachedTokenIndptr[1 + lane] = cachedIncl;
    int64_t const flatStart = seqIncl - seq64;
    int64_t const startPos = kv64 - seq64;
    int64_t const tokens = seq64 < p.maxQueryLanes ? seq64 : p.maxQueryLanes;
    for (int64_t j = 0; j < tokens; ++j)
    {
        int64_t const g = flatStart + j;
        if (g >= p.numTokens)
        {
            break;
        }
        int64_t const globalPos = startPos + j;
        // Truncating division and a non-negative remainder, then the clamp, as the Triton kernel computes them.
        int64_t blk = globalPos / p.tokensPerBlock;
        int64_t pos = globalPos % p.tokensPerBlock;
        pos = pos < 0 ? pos + p.tokensPerBlock : pos;
        blk = blk < 0 ? 0 : (blk > p.maxBlocks - 1 ? p.maxBlocks - 1 : blk);
        int64_t const blockId = p.blockOffsets[lane * p.blockOffsetsStride0 + blk * p.blockOffsetsStride1];
        p.reqIdxPerToken[g] = lane;
        p.slotMappingFp8[g] = blockId * p.blockStride + pos * p.dataBytesPerToken;
        p.slotMappingScale[g] = blockId * p.blockStride + p.scaleBaseOffset + pos * p.scaleSize;
    }
}

__global__ void __launch_bounds__(kBlockSize) draftSampleGlueKernel(DraftSampleGlueParams const p)
{
    __shared__ __align__(16) __nv_bfloat16 sWeights[2][kDraftSampleGlueHidden];
    __shared__ Candidate sCandidates[kDraftSampleGlueMaxRows][kNumWarps];
    __shared__ Candidate sSlots[kMaxSlots];
    __shared__ int sTokens[kDraftSampleGlueMaxRows];
    __shared__ float sGroupSums[kNumNormGroups][2];
    __shared__ uint4 sFlags[2];

    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid / 32;
    int const lane = tid % 32;
    bool const glue = p.glueOut != nullptr;
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
    }

    // Local argmax of every row: all threads scan, each warp reduces, warp 0 combines below.
    int const numVecs = p.vocabShard / kNormVecElts;
    for (int row = 0; row < p.numRows; ++row)
    {
        Candidate const c
            = warpArgmax(scanRow(reinterpret_cast<uint4 const*>(p.logits + row * p.logitsStride), numVecs));
        if (lane == 0)
        {
            sCandidates[row][warp] = c;
        }
    }
    __syncthreads();

    uint4 const flags = sFlags[0];
    uint32_t const curIndex = flags.x;
    uint32_t const dirtyIndex = flags.y;
    uint32_t const bytesPerBuffer = flags.z;
    uint32_t const dirtyStages = flags.w;
    int const numSlots = p.numRows * p.tpSize;
    auto* exchangeLocal = static_cast<char*>(p.exchangeLocal);

    if (warp == 0)
    {
        // Publish this rank's pair of every row through the multicast address: slot [row][rank] of the current buffer.
        auto* curMcast = reinterpret_cast<uint4*>(static_cast<char*>(p.exchangeMcast) + curIndex * bytesPerBuffer);
        for (int row = 0; row < p.numRows; ++row)
        {
            Candidate c = lane < kNumWarps ? sCandidates[row][lane] : Candidate{-INFINITY, INT_MAX};
            c = warpArgmax(c);
            if (lane == 0)
            {
                int const globalIndex = c.index + p.rank * p.vocabShard;
                curMcast[row * p.tpSize + p.rank]
                    = make_uint4(static_cast<uint32_t>(globalIndex), __float_as_uint(c.value), 0U, 0U);
            }
        }
        // Poll the local copy until every rank's pair has landed.
        auto const* curLocal = exchangeLocal + curIndex * bytesPerBuffer;
        for (int s = lane; s < numSlots; s += 32)
        {
            VolatilePackedLoad<float4> packet;
            do
            {
                packet = loadPackedVolatile<float4>(curLocal + s * kDraftSampleGluePacketBytes);
            } while (isEmptyPacket(packet));
            sSlots[s] = Candidate{__uint_as_float(packet.words[1]), static_cast<int>(packet.words[0])};
        }
        __syncwarp();
        // Global argmax per row: the ranks' pairs in rank order (the lower rank holds the lower indices).
        if (lane < p.numRows)
        {
            Candidate best = sSlots[lane * p.tpSize];
            for (int r = 1; r < p.tpSize; ++r)
            {
                offer(best, sSlots[lane * p.tpSize + r].value, sSlots[lane * p.tpSize + r].index);
            }
            sTokens[lane] = best.index;
            p.draftTokens[lane * p.numDraftTokens + p.draftStep] = best.index;
        }
    }
    else if (warp == 1)
    {
        // Empty the buffer of the previous call; the peers write it again two calls later.
        if (dirtyStages > 0)
        {
            auto* dirty = reinterpret_cast<uint4*>(exchangeLocal + dirtyIndex * bytesPerBuffer);
            int const dirtyPackets = static_cast<int>(sFlags[1].x / kDraftSampleGluePacketBytes);
            for (int i = lane; i < dirtyPackets; i += 32)
            {
                dirty[i] = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
            }
        }
        advanceStep(p);
    }
    else if (hiddenHalf)
    {
        // hnorm of the hidden rows does not need the tokens: groups 1..7 (warps 2..15) run it during the poll.
        int const group = warp / 2;
        int const lo = max(p.colLo - kDraftSampleGlueHidden, 0);
        int const hi = min(p.colHi - kDraftSampleGlueHidden, kDraftSampleGlueHidden);
        int const outShift = kDraftSampleGlueHidden + lo - p.colLo;
        for (int row = group - 1; row < p.numRows; row += kNumNormGroups - 1)
        {
            normRowWindow(reinterpret_cast<uint4 const*>(p.hidden + row * p.hiddenStride), sWeights[1], p.eps, lo, hi,
                p.glueOut + row * p.glueOutStride + outShift, group, sGroupSums);
        }
    }
    __syncthreads();

    if (embedHalf)
    {
        // enorm of the token embeddings: an id outside [0, vocabSize) gives a zero row, as the Embedding module does.
        int const group = warp / 2;
        int const lo = p.colLo;
        int const hi = min(p.colHi, kDraftSampleGlueHidden);
        for (int row = group; row < p.numRows; row += kNumNormGroups)
        {
            int64_t const token = sTokens[row];
            bool const valid = token >= 0 && token < p.vocabSize;
            normRowWindow(
                valid ? reinterpret_cast<uint4 const*>(p.embedTable + token * kDraftSampleGlueHidden) : nullptr,
                sWeights[0], p.eps, lo, hi, p.glueOut + row * p.glueOutStride, group, sGroupSums);
        }
    }

    if (warp == 1 && p.nextNewTokens != nullptr && lane < p.numRows)
    {
        // Output rows after the last draft step: the last accepted token, then the draft tokens of every step.
        int32_t* out = p.nextNewTokens + lane * (p.numDraftTokens + 1);
        out[0] = p.acceptedTokens[p.batchIndices[lane] * p.acceptedTokensStride + p.numAccepted[lane] - 1];
        for (int j = 0; j < p.numDraftTokens; ++j)
        {
            out[1 + j] = j == p.draftStep ? sTokens[lane] : p.draftTokens[lane * p.numDraftTokens + j];
        }
    }

    if (tid == 0)
    {
        // Rotate the Lamport flags as LamportFlags::update does: the current buffer becomes the dirty one.
        auto* flagPtr = reinterpret_cast<uint4*>(p.exchangeFlags);
        flagPtr[0] = make_uint4((curIndex + 1) % 3, curIndex, bytesPerBuffer, 1U);
        flagPtr[1] = make_uint4(static_cast<uint32_t>(numSlots * kDraftSampleGluePacketBytes), 0U, 0U, 0U);
        p.exchangeFlags[8] = 0U;
    }
}

// The draft step with the next step's eh_proj, all-reduce and input_layernorm (draftEhprojTableKernel). The cluster
// covers the hidden row: thread t < kTableSliceVecs of CTA c owns columns [c * kTableSliceElts + 8 t, + 8).
constexpr int kTableCtas = kDraftEhprojTableClusterSize;
constexpr int kTableSliceElts = kDraftSampleGlueHidden / kTableCtas;
constexpr int kTableSliceVecs = kTableSliceElts / kNormVecElts;
constexpr int kTableSliceBytes = kTableSliceElts * static_cast<int>(sizeof(__nv_bfloat16));
constexpr int kTableVecWarps = kTableSliceVecs / 32;
// After the vector warps: a warp that polls the (index, value) packets and scans, then a warp that updates the next
// step's state (CTA 0) or empties the previous call's buffer (the other CTAs) beside the exchange.
constexpr int kTableExchangeWarp = kTableVecWarps;
constexpr int kTableStateWarp = kTableVecWarps + 1;
constexpr int kTableMainThreads = kTableStateWarp * 32;
constexpr int kTableThreads = kTableMainThreads + 32;
constexpr int kTableMainWarps = kTableMainThreads / 32;
// Rank slots of the kernel (its template argument): eight up to TP8, kDraftEhprojTableMaxRanks above.
constexpr int kTableTp8RankSlots = kTableCtas;
// Logit vectors each scanning thread loads before it compares them.
constexpr int kTableScanBatch = 4;
// Slot layout: the packet, the BF16 token-table row, then the rank's FP32 hidden-half outputs.
constexpr int kTableRowOffset = kDraftSampleGluePacketBytes;
constexpr int kTableHiddenOffset = kTableRowOffset + 2 * kDraftSampleGlueHidden;
// Named barrier of the main warps (the state warp runs on its own).
constexpr int kTableMainBarrier = 1;
static_assert(kTableSliceVecs % 32 == 0, "the slice must be whole warps of vectors");
static_assert(kTableSliceElts * kTableCtas == kDraftSampleGlueHidden, "the cluster must cover the row");

__device__ __forceinline__ uint4 sanitizeWords(uint4 v)
{
    // A word with every bit set is the empty word; such a payload word (two NaN halves) is sent as another NaN.
    v.x = v.x == kEmptyWord ? kEmptyWord - 1U : v.x;
    v.y = v.y == kEmptyWord ? kEmptyWord - 1U : v.y;
    v.z = v.z == kEmptyWord ? kEmptyWord - 1U : v.z;
    v.w = v.w == kEmptyWord ? kEmptyWord - 1U : v.w;
    return v;
}

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFU, v, offset));
    }
    return v;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterArrive()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

#endif

template <int kMaxRanks>
__global__ void __launch_bounds__(kTableThreads) draftEhprojTableKernel(DraftEhprojTableParams const p)
{
    // sCtaBest is read by the other CTAs of the cluster after the first cluster barrier; sClusterSums is written by
    // every CTA (slot = the writer's rank in the cluster) before the second one and read locally after it. The rest
    // is shared by the main warps through the named barrier.
    __shared__ Candidate sWarpBest[kTableMainWarps];
    __shared__ float sWarpSums[kTableMainWarps];
    __shared__ Candidate sCtaBest[kDraftSampleGlueMaxRows];
    __shared__ Candidate sRankBest[kDraftSampleGlueMaxRows];
    __shared__ float sClusterSums[kDraftSampleGlueMaxRows][kTableCtas];
    __shared__ float sRstd[kDraftSampleGlueMaxRows];
    __shared__ Candidate sPairs[kDraftSampleGlueMaxRows][kMaxRanks];
    __shared__ __align__(16) __nv_bfloat16 sX[kDraftSampleGlueMaxRows][kTableSliceElts];
    __shared__ uint4 sFlags;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    DraftSampleGlueParams const& s = p.step;
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid / 32;
    int const lane = tid % 32;
    int const cta = static_cast<int>(cluster.block_rank());
    int const tp = s.tpSize;
    int const colBase = cta * kTableSliceElts;
    bool const vecThread = tid < kTableSliceVecs;
    // The rank whose share of the hidden-half outputs (H / tp columns) holds this vector thread's eight columns, and
    // their offset in that share: a CTA's columns lie in one share up to TP8 and span two shares at TP16.
    int const shareCols = kDraftSampleGlueHidden / tp;
    int const vecCol = colBase + kNormVecElts * tid;
    int const hiddenOwner = vecCol / shareCols;
    int const shareOffset = vecCol - hiddenOwner * shareCols;
    int const slotBytes = draftEhprojTableSlotBytes(tp);

    // The norm weight is a constant: load it before the grid dependency wait.
    uint4 normWeight = make_uint4(0U, 0U, 0U, 0U);
    if (vecThread)
    {
        normWeight = reinterpret_cast<uint4 const*>(p.normWeight + colBase)[tid];
    }

    // The logits come from the preceding kernel (the draft LM head), the hidden-half outputs from a kernel this launch
    // depends on through an event. Dependents read the outputs only after their own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    if (warp == kTableStateWarp)
    {
        // Arrive for the first cluster barrier before the state work, which nothing in the cluster waits for.
        clusterArriveRelaxed();
        if (cta == 0)
        {
            advanceStep(s);
        }
        else
        {
            // Empty this CTA's share of the previous call's buffer; the peers write it again two calls later.
            uint4 const flags = reinterpret_cast<uint4 const*>(s.exchangeFlags)[0];
            uint4 const sizes = reinterpret_cast<uint4 const*>(s.exchangeFlags)[1];
            if (flags.w > 0)
            {
                auto* dirty = reinterpret_cast<uint4*>(
                    static_cast<char*>(s.exchangeLocal) + static_cast<size_t>(flags.y) * flags.z);
                int const vectors = static_cast<int>(sizes.x / sizeof(uint4));
                for (int i = (cta - 1) * 32 + lane; i < vectors; i += (kTableCtas - 1) * 32)
                {
                    dirty[i] = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
                }
            }
        }
        clusterWait();
        // The second (last) cluster barrier.
        clusterArrive();
        clusterWait();
        return;
    }

    if (tid == 0)
    {
        sFlags = reinterpret_cast<uint4 const*>(s.exchangeFlags)[0];
    }

    // Per row: this CTA's best logit (the CTAs scan consecutive chunks of the shard).
    int const numVecs = s.vocabShard / kNormVecElts;
    int const chunk = (numVecs + kTableCtas - 1) / kTableCtas;
    int const vecBegin = min(cta * chunk, numVecs);
    int const vecEnd = min(vecBegin + chunk, numVecs);
    for (int row = 0; row < s.numRows; ++row)
    {
        auto const* logitsRow = reinterpret_cast<uint4 const*>(s.logits + row * s.logitsStride);
        Candidate best{-INFINITY, INT_MAX};
        for (int base = vecBegin + tid; base < vecEnd; base += kTableScanBatch * kTableMainThreads)
        {
            uint4 vecs[kTableScanBatch];
#pragma unroll
            for (int k = 0; k < kTableScanBatch; ++k)
            {
                int const v = base + k * kTableMainThreads;
                if (v < vecEnd)
                {
                    vecs[k] = logitsRow[v];
                }
            }
#pragma unroll
            for (int k = 0; k < kTableScanBatch; ++k)
            {
                int const v = base + k * kTableMainThreads;
                if (v < vecEnd)
                {
                    offerVector(best, vecs[k], v * kNormVecElts);
                }
            }
        }
        best = warpArgmax(best);
        if (lane == 0)
        {
            sWarpBest[warp] = best;
        }
        namedBarrier(kTableMainBarrier, kTableMainThreads);
        Candidate ctaBest = sWarpBest[0];
        for (int w = 1; w < kTableMainWarps; ++w)
        {
            offer(ctaBest, sWarpBest[w].value, sWarpBest[w].index);
        }
        if (tid == 0)
        {
            sCtaBest[row] = ctaBest;
        }
        // The winner's token-table row is one of the cluster's candidates: pull this CTA's candidate row into L2 while
        // the cluster combines them.
        if (vecThread && ctaBest.index >= 0 && ctaBest.index < s.vocabShard)
        {
            auto const* row = p.tokenTable + static_cast<int64_t>(ctaBest.index) * kDraftSampleGlueHidden;
            char const* line = reinterpret_cast<char const*>(row) + tid * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line) : "memory");
        }
        // sWarpBest is rewritten for the next row only after this barrier.
        namedBarrier(kTableMainBarrier, kTableMainThreads);
    }

    uint4 const flags = sFlags;
    uint32_t const curIndex = flags.x;
    uint32_t const bytesPerBuffer = flags.z;
    char* const curMcast = static_cast<char*>(s.exchangeMcast) + static_cast<size_t>(curIndex) * bytesPerBuffer;
    char const* const curLocal
        = static_cast<char const*>(s.exchangeLocal) + static_cast<size_t>(curIndex) * bytesPerBuffer;

    // This rank's hidden-half outputs do not depend on the exchange: publish them first.
    if (vecThread && hiddenOwner == s.rank)
    {
        for (int row = 0; row < s.numRows; ++row)
        {
            auto const* h = reinterpret_cast<uint4 const*>(p.hiddenHalf + row * p.hiddenHalfStride + shareOffset);
            char* const slot = curMcast + static_cast<size_t>(row * tp + s.rank) * slotBytes;
            auto* dst = reinterpret_cast<uint4*>(slot + kTableHiddenOffset + shareOffset * sizeof(float));
            uint4 const h0 = h[0];
            uint4 const h1 = h[1];
            dst[0] = sanitizeWords(h0);
            dst[1] = sanitizeWords(h1);
        }
    }

    clusterArrive();
    clusterWait();

    // This rank's candidate of every row: the CTAs' candidates in CTA order. These are the last reads of another CTA's
    // shared memory; every CTA makes them before it arrives at the second cluster barrier.
    if (tid < s.numRows)
    {
        Candidate rankBest = *cluster.map_shared_rank(&sCtaBest[tid], 0);
        for (int c = 1; c < kTableCtas; ++c)
        {
            Candidate const other = *cluster.map_shared_rank(&sCtaBest[tid], c);
            offer(rankBest, other.value, other.index);
        }
        sRankBest[tid] = rankBest;
    }
    namedBarrier(kTableMainBarrier, kTableMainThreads);

    for (int row = 0; row < s.numRows; ++row)
    {
        Candidate const rankBest = sRankBest[row];
        char* const slot = curMcast + static_cast<size_t>(row * tp + s.rank) * slotBytes;
        if (vecThread)
        {
            // This CTA's columns of the token-table row of the rank's candidate.
            uint4 const q = reinterpret_cast<uint4 const*>(
                p.tokenTable + static_cast<int64_t>(rankBest.index) * kDraftSampleGlueHidden + colBase)[tid];
            reinterpret_cast<uint4*>(slot + kTableRowOffset + cta * kTableSliceBytes)[tid] = sanitizeWords(q);
        }
        else if (cta == 0 && tid == kTableExchangeWarp * 32)
        {
            int const globalIndex = rankBest.index + s.rank * s.vocabShard;
            *reinterpret_cast<uint4*>(slot)
                = make_uint4(static_cast<uint32_t>(globalIndex), __float_as_uint(rankBest.value), 0U, 0U);
        }

        // Poll the local copy: the ranks' rows of this CTA's columns and the two vectors of hidden-half outputs that
        // cover them (vector threads), the ranks' packets (exchange warp). Every round has all its loads in flight.
        char const* const rowLocal = curLocal + static_cast<size_t>(row * tp) * slotBytes;
        VolatilePackedLoad<float4> rows[kMaxRanks];
        VolatilePackedLoad<float4> hidden0;
        VolatilePackedLoad<float4> hidden1;
        if (vecThread)
        {
            char const* const rowSlice = rowLocal + kTableRowOffset + cta * kTableSliceBytes + tid * sizeof(uint4);
            char const* const hiddenPtr = rowLocal + static_cast<size_t>(hiddenOwner) * slotBytes + kTableHiddenOffset
                + shareOffset * sizeof(float);
            bool empty = true;
            while (empty)
            {
#pragma unroll
                for (int r = 0; r < kMaxRanks; ++r)
                {
                    if (r < tp)
                    {
                        rows[r] = loadPackedVolatile<float4>(rowSlice + static_cast<size_t>(r) * slotBytes);
                    }
                }
                hidden0 = loadPackedVolatile<float4>(hiddenPtr);
                hidden1 = loadPackedVolatile<float4>(hiddenPtr + sizeof(uint4));
                empty = isEmptyPacket(hidden0) || isEmptyPacket(hidden1);
#pragma unroll
                for (int r = 0; r < kMaxRanks; ++r)
                {
                    if (r < tp)
                    {
                        empty = empty || isEmptyPacket(rows[r]);
                    }
                }
            }
        }
        else if (warp == kTableExchangeWarp && lane < tp)
        {
            VolatilePackedLoad<float4> packet;
            do
            {
                packet = loadPackedVolatile<float4>(rowLocal + static_cast<size_t>(lane) * slotBytes);
            } while (isEmptyPacket(packet));
            sPairs[row][lane] = Candidate{__uint_as_float(packet.words[1]), static_cast<int>(packet.words[0])};
        }
        namedBarrier(kTableMainBarrier, kTableMainThreads);

        // The global argmax: the ranks' packets in rank order (the lower rank holds the lower indices).
        int winner = 0;
        Candidate best = sPairs[row][0];
        for (int r = 1; r < tp; ++r)
        {
            if (isBetter(sPairs[row][r].value, sPairs[row][r].index, best.value, best.index))
            {
                best = sPairs[row][r];
                winner = r;
            }
        }
        if (cta == 0 && tid == 0)
        {
            s.draftTokens[row * s.numDraftTokens + s.draftStep] = best.index;
        }

        float sumSq = 0.F;
        if (vecThread)
        {
            uint32_t qw[4] = {rows[0].words[0], rows[0].words[1], rows[0].words[2], rows[0].words[3]};
#pragma unroll
            for (int r = 1; r < kMaxRanks; ++r)
            {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    qw[k] = r == winner ? rows[r].words[k] : qw[k];
                }
            }
            // x = token-table row + hidden-half output: one FP32 add and one BF16 rounding per element.
            uint32_t const hw[8] = {hidden0.words[0], hidden0.words[1], hidden0.words[2], hidden0.words[3],
                hidden1.words[0], hidden1.words[1], hidden1.words[2], hidden1.words[3]};
            uint32_t xw[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                xw[k] = packBf16(__fadd_rn(bf16Low(qw[k]), __uint_as_float(hw[2 * k])),
                    __fadd_rn(bf16High(qw[k]), __uint_as_float(hw[2 * k + 1])));
            }
            uint4 const x = make_uint4(xw[0], xw[1], xw[2], xw[3]);
            reinterpret_cast<uint4*>(sX[row])[tid] = x;
            reinterpret_cast<uint4*>(p.residualOut + row * p.outStride + colBase)[tid] = x;
            sumSq = sumOfSquares(x, 0.F);
        }
        sumSq = warpSum(sumSq);
        if (lane == 0)
        {
            sWarpSums[warp] = sumSq;
        }
        namedBarrier(kTableMainBarrier, kTableMainThreads);
        if (tid < kTableCtas)
        {
            float sum = sWarpSums[0];
            for (int w = 1; w < kTableVecWarps; ++w)
            {
                sum = __fadd_rn(sum, sWarpSums[w]);
            }
            *cluster.map_shared_rank(&sClusterSums[row][cta], tid) = sum;
        }
        // sPairs and sWarpSums are rewritten for the next row only after this barrier.
        namedBarrier(kTableMainBarrier, kTableMainThreads);
    }

    // Every CTA has sent its sums of squares and read the others' candidates: after this barrier no CTA touches
    // another's shared memory.
    clusterArrive();
    clusterWait();

    // input_layernorm: the row's sum of squares over the cluster in CTA order.
    if (tid < s.numRows)
    {
        float sum = sClusterSums[tid][0];
        for (int c = 1; c < kTableCtas; ++c)
        {
            sum = __fadd_rn(sum, sClusterSums[tid][c]);
        }
        sRstd[tid] = rsqrtApprox(__fadd_rn(__fdiv_rn(sum, static_cast<float>(kDraftSampleGlueHidden)), p.normEps));
    }
    namedBarrier(kTableMainBarrier, kTableMainThreads);
    if (vecThread)
    {
        for (int row = 0; row < s.numRows; ++row)
        {
            uint4 const x = reinterpret_cast<uint4 const*>(sX[row])[tid];
            reinterpret_cast<uint4*>(p.normedOut + row * p.outStride + colBase)[tid]
                = normalizeVector(x, normWeight, sRstd[row]);
        }
    }

    if (cta == 0 && tid == 0)
    {
        // Every CTA read the flags before the first cluster barrier, every state warp before the second: rotate them
        // as LamportFlags::update does.
        auto* flagPtr = reinterpret_cast<uint4*>(s.exchangeFlags);
        flagPtr[0] = make_uint4((curIndex + 1) % 3, curIndex, bytesPerBuffer, 1U);
        flagPtr[1] = make_uint4(static_cast<uint32_t>(s.numRows * tp * slotBytes), 0U, 0U, 0U);
        s.exchangeFlags[8] = 0U;
    }
#endif
}

// The draft step of draftSampleGlueKernel on a cluster of kWideCtas CTAs (draftSampleGlueWideKernel). Right after
// the grid dependency wait, in every CTA:
// - warps 0..kWideScanWarps-1 scan the CTA's chunk (the c-th of kWideCtas consecutive chunks) of every row of the
//   logits shard and push the CTA's best (index, value) of each row into CTA 0's shared memory; CTA 0 combines them
//   in cluster-rank order and publishes the rank's pairs; every CTA that needs the tokens polls the pairs itself;
//   norm group 0 (warps 0 and 1) of CTA c then normalizes the embedding row of row c's token; CTA 0 writes the tokens
//   and the next-new-tokens rows and rotates the flags;
// - the remaining warps work beside the scan and the exchange: CTA 0's kWideClearWarp empties the previous call's
//   buffer, its kWideStateWarp writes the next step's state, and norm group kWideHnormGroup of CTA c normalizes
//   hidden row c (from a copy that cp.async brings into shared memory).
// Same outputs as draftSampleGlueKernel.
constexpr int kWideCtas = 8;
constexpr int kWideThreads = 512;
constexpr int kWideWarps = kWideThreads / 32;
constexpr int kWideScanWarps = 12;
constexpr int kWideScanThreads = kWideScanWarps * 32;
// Logit vectors of each row a scanning thread has in flight per round, and the rows of a scan pass.
constexpr int kWideScanVecs = 2;
constexpr int kWideScanRows = 4;
constexpr int kWideClearWarp = kWideScanWarps;
constexpr int kWideStateWarp = kWideScanWarps + 1;
// The last two warps.
constexpr int kWideHnormGroup = kNumNormGroups - 1;
// Named barriers: norm group g uses 1 + g (normRowWindow); group 0's warps also meet on its id for the tokens; the
// scanning warps meet on their own.
constexpr int kWideGroup0Barrier = 1;
constexpr int kWideScanBarrier = 1 + kNumNormGroups;
static_assert(kWideThreads == kBlockSize, "the norm groups and slots are those of draftSampleGlueKernel");
static_assert(kDraftSampleGlueMaxRows <= kWideCtas, "CTA c normalizes row c");
static_assert(kDraftSampleGlueMaxRows % kWideScanRows == 0, "whole scan passes");
static_assert(2 * kWideHnormGroup + 2 == kWideWarps && kWideStateWarp < 2 * kWideHnormGroup && kWideScanWarps >= 2,
    "the warp roles must not overlap");
static_assert(kDraftSampleGlueMaxRows <= kWideScanThreads, "one thread per row combines the warps' bests");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
// 16 bytes from global into shared memory without a register (cp.async, cached in L2 only).
__device__ __forceinline__ void cpAsync16(void* smem, void const* gmem)
{
    uint32_t const dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem) : "memory");
}

// Every cp.async of this thread has landed.
__device__ __forceinline__ void cpAsyncWaitAll()
{
    asm volatile("cp.async.wait_all;" ::: "memory");
}

#endif

__global__ void __launch_bounds__(kWideThreads) draftSampleGlueWideKernel(DraftSampleGlueParams const p)
{
    __shared__ __align__(16) __nv_bfloat16 sWeights[2][kDraftSampleGlueHidden];
    // Hidden row c: thread t of the hnorm group copies, and later reads, the vectors normRowWindow assigns to it.
    __shared__ uint4 sHidden[kNormVecs * kNormThreads];
    __shared__ Candidate sWarpBest[kDraftSampleGlueMaxRows][kWideScanWarps];
    // CTA 0 only: the CTAs' bests, written by CTA c into row c before the cluster barrier, read after it.
    __shared__ Candidate sClusterBest[kWideCtas][kDraftSampleGlueMaxRows];
    __shared__ Candidate sSlots[kMaxSlots];
    __shared__ int sTokens[kDraftSampleGlueMaxRows];
    __shared__ float sGroupSums[kNumNormGroups][2];
    __shared__ uint4 sFlags;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const cta = static_cast<int>(cluster.block_rank());
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid / 32;
    int const lane = tid % 32;
    bool const glue = p.glueOut != nullptr;
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
            for (int v = lo / kNormVecElts + tid; v < hi / kNormVecElts; v += kWideThreads)
            {
                dst[v] = src[v];
            }
        }
    }
    __syncthreads();

    // The logits come from the preceding kernel (the draft LM head). Dependents read the outputs only after their
    // own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    bool waited = false;
    if (warp < kWideScanWarps)
    {
        if (tid == 0)
        {
            sFlags = reinterpret_cast<uint4 const*>(p.exchangeFlags)[0];
        }

        // This CTA's chunk of every row, kWideScanRows rows per pass: the loads of a pass's round in flight together.
        int const numVecs = p.vocabShard / kNormVecElts;
        int const chunk = (numVecs + kWideCtas - 1) / kWideCtas;
        int const vecBegin = min(cta * chunk, numVecs);
        int const vecEnd = min(vecBegin + chunk, numVecs);
        Candidate best[kDraftSampleGlueMaxRows];
#pragma unroll
        for (int r = 0; r < kDraftSampleGlueMaxRows; ++r)
        {
            best[r] = Candidate{-INFINITY, INT_MAX};
        }
#pragma unroll
        for (int r0 = 0; r0 < kDraftSampleGlueMaxRows; r0 += kWideScanRows)
        {
            if (r0 >= p.numRows)
            {
                continue;
            }
            for (int base = vecBegin + tid; base < vecEnd; base += kWideScanVecs * kWideScanThreads)
            {
                uint4 vecs[kWideScanRows][kWideScanVecs];
#pragma unroll
                for (int r = 0; r < kWideScanRows; ++r)
                {
#pragma unroll
                    for (int k = 0; k < kWideScanVecs; ++k)
                    {
                        int const v = base + k * kWideScanThreads;
                        if (r0 + r < p.numRows && v < vecEnd)
                        {
                            vecs[r][k] = reinterpret_cast<uint4 const*>(p.logits + (r0 + r) * p.logitsStride)[v];
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < kWideScanRows; ++r)
                {
#pragma unroll
                    for (int k = 0; k < kWideScanVecs; ++k)
                    {
                        int const v = base + k * kWideScanThreads;
                        if (r0 + r < p.numRows && v < vecEnd)
                        {
                            offerVector(best[r0 + r], vecs[r][k], v * kNormVecElts);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kDraftSampleGlueMaxRows; ++r)
        {
            if (r < p.numRows)
            {
                Candidate const c = warpArgmax(best[r]);
                if (lane == 0)
                {
                    sWarpBest[r][warp] = c;
                }
            }
        }
        namedBarrier(kWideScanBarrier, kWideScanThreads);
        if (tid < p.numRows)
        {
            Candidate ctaBest = sWarpBest[tid][0];
            for (int w = 1; w < kWideScanWarps; ++w)
            {
                offer(ctaBest, sWarpBest[tid][w].value, sWarpBest[tid][w].index);
            }
            *cluster.map_shared_rank(&sClusterBest[cta][tid], 0) = ctaBest;
        }
        // The pushes above become visible to CTA 0's waiter.
        clusterArrive();

        uint4 const flags = sFlags;
        uint32_t const curIndex = flags.x;
        uint32_t const bytesPerBuffer = flags.z;
        int const numSlots = p.numRows * p.tpSize;
        // CTA 0 writes the tokens and the next-new-tokens rows; CTA c normalizes the embedding row c.
        bool const needTokens = cta == 0 || (embedHalf && cta < p.numRows);

        if (warp == 0 && needTokens)
        {
            if (cta == 0)
            {
                clusterWait();
                waited = true;
                // The rank's pair of every row (the CTAs' bests in cluster-rank order, the order of their chunks),
                // published through the multicast address: slot [row][rank] of the current buffer.
                if (lane < p.numRows)
                {
                    Candidate rankBest = sClusterBest[0][lane];
                    for (int c = 1; c < kWideCtas; ++c)
                    {
                        offer(rankBest, sClusterBest[c][lane].value, sClusterBest[c][lane].index);
                    }
                    int const globalIndex = rankBest.index + p.rank * p.vocabShard;
                    auto* curMcast
                        = reinterpret_cast<uint4*>(static_cast<char*>(p.exchangeMcast) + curIndex * bytesPerBuffer);
                    curMcast[lane * p.tpSize + p.rank]
                        = make_uint4(static_cast<uint32_t>(globalIndex), __float_as_uint(rankBest.value), 0U, 0U);
                }
            }
            // Poll the local copy until every rank's pair has landed.
            auto const* curLocal = static_cast<char const*>(p.exchangeLocal) + curIndex * bytesPerBuffer;
            for (int s = lane; s < numSlots; s += 32)
            {
                VolatilePackedLoad<float4> packet;
                do
                {
                    packet = loadPackedVolatile<float4>(curLocal + s * kDraftSampleGluePacketBytes);
                } while (isEmptyPacket(packet));
                sSlots[s] = Candidate{__uint_as_float(packet.words[1]), static_cast<int>(packet.words[0])};
            }
            __syncwarp();
            // Global argmax per row: the ranks' pairs in rank order (the lower rank holds the lower indices).
            if (lane < p.numRows)
            {
                Candidate rowBest = sSlots[lane * p.tpSize];
                for (int r = 1; r < p.tpSize; ++r)
                {
                    offer(rowBest, sSlots[lane * p.tpSize + r].value, sSlots[lane * p.tpSize + r].index);
                }
                sTokens[lane] = rowBest.index;
                if (cta == 0)
                {
                    p.draftTokens[lane * p.numDraftTokens + p.draftStep] = rowBest.index;
                }
            }
            if (cta == 0 && lane == 0)
            {
                // Every CTA read the flags before it arrived at the cluster barrier: rotate them as
                // LamportFlags::update does, the current buffer becoming the dirty one.
                auto* flagPtr = reinterpret_cast<uint4*>(p.exchangeFlags);
                flagPtr[0] = make_uint4((curIndex + 1) % 3, curIndex, bytesPerBuffer, 1U);
                flagPtr[1] = make_uint4(static_cast<uint32_t>(numSlots * kDraftSampleGluePacketBytes), 0U, 0U, 0U);
                p.exchangeFlags[8] = 0U;
            }
        }

        if (warp < 2 && needTokens)
        {
            // Group 0: the tokens of warp 0, then the embedding row of this CTA.
            namedBarrier(kWideGroup0Barrier, kNormThreads);
            if (cta == 0 && warp == 1 && p.nextNewTokens != nullptr && lane < p.numRows)
            {
                // Output rows after the last draft step: the last accepted token, then the draft tokens of every
                // step.
                int32_t* out = p.nextNewTokens + lane * (p.numDraftTokens + 1);
                out[0] = p.acceptedTokens[p.batchIndices[lane] * p.acceptedTokensStride + p.numAccepted[lane] - 1];
                for (int j = 0; j < p.numDraftTokens; ++j)
                {
                    out[1 + j] = j == p.draftStep ? sTokens[lane] : p.draftTokens[lane * p.numDraftTokens + j];
                }
            }
            if (embedHalf && cta < p.numRows)
            {
                // enorm of the token embedding: an id outside [0, vocabSize) gives a zero row, as the Embedding
                // module does.
                int64_t const token = sTokens[cta];
                bool const valid = token >= 0 && token < p.vocabSize;
                normRowWindow(
                    valid ? reinterpret_cast<uint4 const*>(p.embedTable + token * kDraftSampleGlueHidden) : nullptr,
                    sWeights[0], p.eps, p.colLo, min(p.colHi, kDraftSampleGlueHidden),
                    p.glueOut + cta * p.glueOutStride, 0, sGroupSums);
            }
        }
    }
    else if (warp == kWideClearWarp)
    {
        if (cta == 0)
        {
            // Empty the buffer of the previous call; the peers write it again two calls later. The release arrive
            // orders the reads of the flags before CTA 0 rotates them.
            uint4 const flags = reinterpret_cast<uint4 const*>(p.exchangeFlags)[0];
            uint4 const sizes = reinterpret_cast<uint4 const*>(p.exchangeFlags)[1];
            clusterArrive();
            if (flags.w > 0)
            {
                auto* dirty = reinterpret_cast<uint4*>(
                    static_cast<char*>(p.exchangeLocal) + static_cast<size_t>(flags.y) * flags.z);
                int const dirtyPackets = static_cast<int>(sizes.x / kDraftSampleGluePacketBytes);
                for (int i = lane; i < dirtyPackets; i += 32)
                {
                    dirty[i] = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
                }
            }
        }
        else
        {
            clusterArriveRelaxed();
        }
    }
    else if (warp == kWideStateWarp)
    {
        // The next step's state does not need the tokens; nothing in the cluster waits for it.
        clusterArriveRelaxed();
        if (cta == 0)
        {
            advanceStep(p);
        }
    }
    else
    {
        // The hnorm group: hnorm of hidden row cta does not need the tokens; nothing in the cluster waits for it.
        clusterArriveRelaxed();
        if (hiddenHalf && cta < p.numRows)
        {
            int const t = tid % kNormThreads;
            auto const* src = reinterpret_cast<uint4 const*>(p.hidden + cta * p.hiddenStride);
#pragma unroll
            for (int b = 0; b < kNormVecs; ++b)
            {
                cpAsync16(&sHidden[t + kNormThreads * b], &src[t + kNormThreads * b]);
            }
            // Each thread reads back only the vectors it copied.
            cpAsyncWaitAll();
            int const lo = max(p.colLo - kDraftSampleGlueHidden, 0);
            int const hi = min(p.colHi - kDraftSampleGlueHidden, kDraftSampleGlueHidden);
            int const outShift = kDraftSampleGlueHidden + lo - p.colLo;
            normRowWindow(sHidden, sWeights[1], p.eps, lo, hi, p.glueOut + cta * p.glueOutStride + outShift,
                kWideHnormGroup, sGroupSums);
        }
    }

    // CTA 0 outlives the pushes into its shared memory.
    if (!waited)
    {
        clusterWait();
    }
#endif
}

} // namespace

void invokeDraftSampleGlue(DraftSampleGlueParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.numRows > 0 && params.numRows <= kDraftSampleGlueMaxRows,
        "draft sample glue serves 1 to %d rows, got %d", kDraftSampleGlueMaxRows, params.numRows);
    TLLM_CHECK_WITH_INFO(params.tpSize >= 1 && params.numRows * params.tpSize <= kDraftSampleGlueMaxRows * 32,
        "draft sample glue: %d rows x %d ranks exceed the exchange slots", params.numRows, params.tpSize);
    TLLM_CHECK_WITH_INFO(params.vocabShard > 0 && params.vocabShard % 8 == 0,
        "draft sample glue needs a vocabulary shard of a multiple of 8 columns, got %d", params.vocabShard);
    TLLM_CHECK_WITH_INFO(params.glueOut == nullptr
            || (params.colLo >= 0 && params.colLo < params.colHi && params.colHi <= 2 * kDraftSampleGlueHidden
                && params.colLo % 8 == 0 && params.colHi % 8 == 0),
        "draft sample glue: invalid eh_proj input window [%d, %d)", params.colLo, params.colHi);
    TLLM_CHECK_WITH_INFO(
        params.glueOut == nullptr || params.colLo >= kDraftSampleGlueHidden || params.embedTable != nullptr,
        "draft sample glue: the window needs the embedding table");
    TLLM_CHECK_WITH_INFO(
        params.glueOut == nullptr || params.colHi <= kDraftSampleGlueHidden || params.hidden != nullptr,
        "draft sample glue: the window needs the hidden rows");
    TLLM_CHECK_WITH_INFO(params.reqIdxPerToken == nullptr || (params.seqLens != nullptr && params.kvLens != nullptr),
        "draft sample glue: the DSA metadata needs the sequence and KV lengths");
    bool const wide = params.tpSize == 4;
    if (wide)
    {
        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = kWideCtas;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(kWideCtas);
        config.blockDim = dim3(kWideThreads);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 2;
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, draftSampleGlueWideKernel, params));
        return;
    }
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "draftSampleGlue", draftSampleGlueKernel, dim3(1), dim3(kBlockSize), 0, stream, params);
}

void invokeDraftEhprojTable(DraftEhprojTableParams const& params, cudaStream_t stream)
{
    DraftSampleGlueParams const& s = params.step;
    TLLM_CHECK_WITH_INFO(s.numRows > 0 && s.numRows <= kDraftSampleGlueMaxRows,
        "draft eh_proj table serves 1 to %d rows, got %d", kDraftSampleGlueMaxRows, s.numRows);
    TLLM_CHECK_WITH_INFO(draftEhprojTableSupportsTpSize(s.tpSize),
        "draft eh_proj table serves 2 to %d ranks with whole 8-column shares of the hidden size, got %d",
        kDraftEhprojTableMaxRanks, s.tpSize);
    TLLM_CHECK_WITH_INFO(s.vocabShard > 0 && s.vocabShard % 8 == 0 && s.logitsStride % 8 == 0,
        "draft eh_proj table needs 16-byte aligned rows of a multiple of 8 logits, got %d", s.vocabShard);
    TLLM_CHECK_WITH_INFO(params.hiddenHalf != nullptr && params.hiddenHalfStride % 8 == 0
            && reinterpret_cast<uintptr_t>(params.hiddenHalf) % 32 == 0,
        "draft eh_proj table needs 32-byte aligned rows of FP32 hidden-half outputs");
    TLLM_CHECK_WITH_INFO(s.glueOut == nullptr && s.nextNewTokens == nullptr,
        "draft eh_proj table: no eh_proj input window and no next-new-tokens rows");
    TLLM_CHECK_WITH_INFO(params.tokenTable != nullptr && params.normWeight != nullptr && params.normedOut != nullptr
            && params.residualOut != nullptr && params.outStride % 8 == 0,
        "draft eh_proj table: missing token table, norm weight or 16-byte aligned output rows");
    TLLM_CHECK_WITH_INFO(s.reqIdxPerToken == nullptr || (s.seqLens != nullptr && s.kvLens != nullptr),
        "draft eh_proj table: the DSA metadata needs the sequence and KV lengths");
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kDraftEhprojTableClusterSize;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kDraftEhprojTableClusterSize);
    config.blockDim = dim3(kTableThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    // Up to TP8 the kernel keeps eight rank slots (registers of the polled rows); TP16 takes sixteen.
    auto* const kernel = s.tpSize <= kTableTp8RankSlots ? &draftEhprojTableKernel<kTableTp8RankSlots>
                                                        : &draftEhprojTableKernel<kDraftEhprojTableMaxRanks>;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
